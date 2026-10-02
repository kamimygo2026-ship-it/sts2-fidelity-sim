"""事件**引擎能力**测试：进入条件、转化、奖励、诚实性闸门。

与 ``tests/test_events.py`` 的分工：那边验"事件图抽得对不对"（结构 + 状态机），
这里验"这一轮补上的引擎能力"本身，每条断言都要能追到源码。

覆盖的能力与执行点：

* ``IsAllowed`` 的**结构化解析**（守卫串 / ``||`` / ``!`` / 牌组谓词 / 动态变量阈值）
  —— ``EventModel.IsAllowed``；
* **随机转化**（``CardCmd.TransformToRandom`` → ``CardFactory.CreateRandomCardForTransform``）
  与 ``select_card(purpose="transform")`` 的挂起/结算；
* **奖励**（``RewardsCmd.OfferCustom`` → 一条奖励一个待选项、
  ``RelicReward.Populate`` 走遗物抓包、``PotionFactory`` 的药水池与稀有度掷点）；
* **诚实性闸门**：算不出、点不动、走不到结束的事件一律不可用。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from sts2_sim import content, events
from sts2_sim.rng import RngSet

CONTENT = Path(__file__).resolve().parent.parent / "data" / "content" / "repo"


def setUpModule() -> None:               # noqa: N802 - unittest 约定
    if not CONTENT.exists():
        raise unittest.SkipTest(f"缺内容目录 {CONTENT}")
    from sts2_sim.featurize import configure_content

    configure_content(str(CONTENT))


def tearDownModule() -> None:            # noqa: N802 - unittest 约定
    from sts2_sim.content import load_builtin
    from sts2_sim.featurize import rebuild_vocab

    load_builtin()
    rebuild_vocab()


# ==========================================================================
# 进入条件（``EventModel.IsAllowed``）
# ==========================================================================
class TestGateParsing(unittest.TestCase):
    """``IsAllowed`` 的结构化解析：守卫串 / ``||`` / ``!`` / 谓词。"""

    def _gate(self, body: str, **kwargs):
        definition = events.EventDef(eid="x", name="x", pages={}, gate_body=body)
        kind, args = events.parse_gate(body, kwargs.pop("vars", None))
        return events.EventDef(eid="x", name="x", pages={}, gate_body=body,
                               gate_kind=kind, gate_args=tuple(args))

    def test_guard_chain_with_returned_expression(self):
        """``CrystalSphere.cs:33-40``：``if (A) { return B; } return false;`` → A && B。"""
        body = ("if (runState.Players.All((Player p) => p.Gold >= 100)) "
                "{ return runState.CurrentActIndex > 0; } return false;")
        definition = self._gate(body)
        self.assertEqual(definition.gate_kind, "and")
        self.assertIs(events.gate_ok(definition, act_index=1, hp=80, max_hp=80,
                                     gold=100), True)
        self.assertIs(events.gate_ok(definition, act_index=0, hp=80, max_hp=80,
                                     gold=100), False)
        self.assertIs(events.gate_ok(definition, act_index=1, hp=80, max_hp=80,
                                     gold=99), False)

    def test_guard_clauses_all_false_returns_expression(self):
        """``TeaMaster.cs:43-50``：一串 ``if (…) return false;`` 后才是真条件。"""
        body = ("if (runState.CurrentActIndex < 2) "
                "{ return runState.Players.All((Player p) => p.Gold >= 150); } "
                "return false;")
        definition = self._gate(body)
        self.assertIs(events.gate_ok(definition, act_index=1, hp=80, max_hp=80,
                                     gold=150), True)
        self.assertIs(events.gate_ok(definition, act_index=1, hp=80, max_hp=80,
                                     gold=149), False)
        self.assertIs(events.gate_ok(definition, act_index=2, hp=80, max_hp=80,
                                     gold=999), False)

    def test_single_player_early_true_covers_multiplayer_only_tail(self):
        """``DenseVegetation.cs:46-60``：单人时整体恒真（后面那段是多人专用）。"""
        body = ("if (runState.Players.Count == 1) { return true; } "
                "foreach (Player player in runState.Players) { "
                "if ((decimal)player.Creature.CurrentHp <= 1m) { return false; } } "
                "return true;")
        definition = self._gate(body)
        self.assertEqual(definition.gate_kind, "always")
        self.assertIs(events.gate_ok(definition, act_index=0, hp=1, max_hp=80,
                                     gold=0), True)

    def test_or_and_negation(self):
        """``FakeMerchant.cs:101-112``：``act>=1`` 且（金币够 **或** 有 FoulPotion）。"""
        body = ("if (runState.CurrentActIndex < 1) { return false; } "
                "if (runState.Players.Count > 1) { return false; } "
                "return runState.Players.All((Player player) => player.Gold >= 100 "
                "|| player.Potions.Any((PotionModel potion) => potion is FoulPotion));")
        definition = self._gate(body)
        self.assertIs(events.gate_ok(definition, act_index=1, hp=80, max_hp=80,
                                     gold=100, potions=0), True)
        self.assertIs(events.gate_ok(definition, act_index=1, hp=80, max_hp=80,
                                     gold=0, potions=1,
                                     potion_ids=("foul_potion",)), True)
        self.assertIs(events.gate_ok(definition, act_index=1, hp=80, max_hp=80,
                                     gold=0, potions=1,
                                     potion_ids=("fire_potion",)), False)
        self.assertIs(events.gate_ok(definition, act_index=0, hp=80, max_hp=80,
                                     gold=999), False)

    def test_dynamic_var_threshold_becomes_a_number(self):
        """``ZenWeaver.cs:32-35``：阈值写成 ``DynamicVars["X"].BaseValue``（=125）。"""
        body = ('return runState.Players.All((Player p) => (decimal)p.Gold >= '
                'base.DynamicVars["EmotionalAwarenessCost"].BaseValue);')
        definition = self._gate(body, vars={"EmotionalAwarenessCost": 125})
        self.assertEqual(definition.gate_kind, "gold_ge")
        self.assertEqual(definition.gate_args, (125,))
        # 变量表里没有这个变量 → 算不出（**不猜**）
        unknown = self._gate(body)
        self.assertEqual(unknown.gate_kind, "unknown")

    def test_deck_predicate_requires_runtime_deck(self):
        """``WoodCarvings.cs``：牌组谓词。**没有牌组数据时必须返回 None**。"""
        body = ("return runState.Players.All((Player p) => "
                "CardPile.Get(PileType.Deck, p).Cards.Any((CardModel c) => "
                "c != null && c.Rarity == CardRarity.Basic && c.IsRemovable));")
        definition = self._gate(body)
        self.assertEqual(definition.gate_kind, "deck_any")
        self.assertIsNone(events.gate_ok(definition, act_index=0, hp=80,
                                         max_hp=80, gold=0),
                          "缺牌组数据时必须算不出，而不是猜 true")
        from sts2_sim.core import CardInstance

        deck = [CardInstance("strike_ironclad")]
        self.assertIs(events.gate_ok(definition, act_index=0, hp=80, max_hp=80,
                                     gold=0, deck=deck), True)
        eternal = [CardInstance("ascenders_bane")]
        self.assertIs(events.gate_ok(definition, act_index=0, hp=80, max_hp=80,
                                     gold=0, deck=eternal), False)

    def test_event_pet_gate(self):
        """``ByrdonisNest.cs:36`` + ``Player.cs:248-255``：AddsPet 遗物 或 蛋。"""
        definition = self._gate("return runState.Players.All((Player p) => "
                                "!p.HasEventPet());")
        self.assertEqual(definition.gate_kind, "not")
        self.assertIsNone(events.gate_ok(definition, act_index=0, hp=80,
                                         max_hp=80, gold=0))
        from sts2_sim.core import CardInstance

        plain = [CardInstance("strike_ironclad")]
        self.assertIs(events.gate_ok(definition, act_index=0, hp=80, max_hp=80,
                                     gold=0, deck=plain, relics=()), True)
        egg = plain + [CardInstance("byrdonis_egg")]
        self.assertIs(events.gate_ok(definition, act_index=0, hp=80, max_hp=80,
                                     gold=0, deck=egg, relics=()), False)
        self.assertIs(events.gate_ok(definition, act_index=0, hp=80, max_hp=80,
                                     gold=0, deck=plain, relics=("byrdpip",)),
                      False)

    def test_unlock_state_gate_stays_unknown(self):
        """``ColorfulPhilosophers.cs``：解锁状态引擎没建模 → 如实 unknown。"""
        kind, _args = events.parse_gate(
            "return runState.Players.All((Player p) => "
            "p.UnlockState.CharacterCardPools.Count() > 1);")
        self.assertEqual(kind, "unknown")


# ==========================================================================
# 随机转化（``CardCmd.TransformToRandom``）
# ==========================================================================
class TestTransform(unittest.TestCase):
    """转化：候选池规则 + 用哪条 RNG + 挂起选牌。"""

    def test_default_options_restrict_rarity_by_source(self):
        """``CardFactory.cs:189-212``：原牌不是 Status/Curse 时只留 普通/罕见/稀有。"""
        from sts2_sim.runeffects import transform_options

        options = transform_options("strike_ironclad")      # Basic，Ironclad 池
        self.assertTrue(options)
        for cid in options:
            self.assertIn(content.CARD_DB[cid].rarity, ("common", "uncommon", "rare"))
            self.assertNotEqual(cid, "strike_ironclad")

    def test_status_and_curse_cards_keep_every_rarity(self):
        """``(uint)(rarity - 8) <= 1``（Status/Curse）时不加稀有度限制。"""
        from sts2_sim.runeffects import transform_options

        options = transform_options("ascenders_bane")       # Curse，Ironclad 池
        self.assertTrue(options)
        rarities = {content.CARD_DB[cid].rarity for cid in options}
        self.assertTrue(rarities - {"common", "uncommon", "rare"},
                        f"Curse 牌不该被限制成三档稀有度，实际 {rarities}")

    def test_replacement_uses_the_passed_rng(self):
        """``CardFactory.cs:177-181``：``rng.NextItem(options)`` —— 传哪条就用哪条。"""
        from sts2_sim.runeffects import transform_options, transform_replacement

        class FixedRng:
            def __init__(self, index):
                self.index = index

            def next_index(self, length):
                return self.index

        options = transform_options("strike_ironclad")
        self.assertEqual(transform_replacement("strike_ironclad", FixedRng(0)),
                         options[0])
        self.assertEqual(transform_replacement("strike_ironclad", FixedRng(-1)),
                         options[-1])

    def test_transform_selection_replaces_in_place(self):
        """挂起 → 选一张 → **原地**换成另一张（``CardCmd.cs:398-420``）。"""
        from sts2_sim import run as runmod
        from sts2_sim.core import CardInstance

        env = runmod.RunEnv(seed=5, attempt_budget=1)
        env.reset()
        state = env.raw_state
        definition = content.EVENT_DB["aroma_of_chaos"]
        run = events.initial_run(definition,
                                 events.event_rng(1, "aroma_of_chaos"))
        index = next(i for i, o in enumerate(definition.pages["INITIAL"].options)
                     if o.name == "LET_GO")
        log: list[str] = []
        events.apply_option(definition, run, index, state, log)
        selection = state.pending_selection
        self.assertIsNotNone(selection, "转化必须先挂起，由玩家选牌")
        self.assertEqual(selection.purpose, "transform")
        self.assertIsNotNone(selection.rng, "转化必须带上调用方指定的 RNG")

        target = selection.candidates[0]
        before = state.player.deck[target].cid
        size = len(state.player.deck)
        from sts2_sim.runeffects import resolve_run_selection

        resolve_run_selection(state, target, log, obtain_relic=env._obtain_relic)
        self.assertEqual(len(state.player.deck), size, "转化是原地替换，不是删了再加")
        self.assertNotEqual(state.player.deck[target].cid, before)

    def test_transform_candidates_exclude_quest_cards(self):
        """``CardSelectCmd.cs:591``：``c.Type != Quest && c.IsTransformable``。"""
        from sts2_sim import run as runmod
        from sts2_sim.core import CardInstance

        env = runmod.RunEnv(seed=5, attempt_budget=1)
        env.reset()
        state = env.raw_state
        state.player.deck.append(CardInstance("byrdonis_egg", uid=999))
        definition = content.EVENT_DB["aroma_of_chaos"]
        run = events.initial_run(definition,
                                 events.event_rng(1, "aroma_of_chaos"))
        index = next(i for i, o in enumerate(definition.pages["INITIAL"].options)
                     if o.name == "LET_GO")
        events.apply_option(definition, run, index, state, [])
        cids = [state.player.deck[i].cid for i in state.pending_selection.candidates]
        self.assertNotIn("byrdonis_egg", cids, "任务牌不能被转化")


# ==========================================================================
# 奖励（``RewardsCmd.OfferCustom``）
# ==========================================================================
class TestOfferRewards(unittest.TestCase):
    """奖励：一条奖励一个待选项；药水/遗物候选按源码的池与稀有度。"""

    def _state(self, seed=7):
        from sts2_sim import run as runmod

        env = runmod.RunEnv(seed=seed, attempt_budget=1)
        env.reset()
        return env, env.raw_state

    def test_each_reward_item_becomes_its_own_pending_group(self):
        """``PotionCourier.cs:36-45``：3 个 ``PotionReward`` = **3 次**拿/跳过。

        ⚠️ 早先实现把份数当成"一个组里的候选数"，于是"给 3 瓶"变成"三选一拿 1 瓶"。
        """
        from sts2_sim.runeffects import apply_run_effects
        from sts2_sim.content import Effect

        env, state = self._state()
        effect = Effect(op="offer_rewards", amount=3, target="self",
                        card_filter=(("kind", "potion"),
                                     ("potion", "glowwater_potion")))
        log: list[str] = []
        self.assertTrue(apply_run_effects(state, (effect,), log))
        self.assertEqual(len(state.pending_rewards), 3)
        for group in state.pending_rewards:
            self.assertEqual(group.kind, "potion")
            self.assertEqual(group.options, ("glowwater_potion",))

    def test_pool_mode_uses_character_and_shared_potions(self):
        """``PotionFactory.GetPotionOptions``（``PotionFactory.cs:88-91``）：角色池 ∪ 共享池。"""
        from sts2_sim.runeffects import _potion_pool

        env, state = self._state()
        pool = _potion_pool(state)
        self.assertTrue(pool)
        for pid in pool:
            self.assertIn(content.POTIONS[pid].pool, ("shared", "ironclad"))
        rare = _potion_pool(state, "uncommon")
        self.assertTrue(rare)
        self.assertTrue(all(content.POTIONS[pid].rarity == "uncommon"
                            for pid in rare))

    def test_factory_mode_rolls_rarity_by_potion_factory_odds(self):
        """``PotionFactory.cs:74-77``：``NextFloat()`` ≤0.1 稀有 / ≤0.35 罕见 / 其余普通。"""
        from sts2_sim.runeffects import _roll_potion_rarity

        class FixedRng:
            def __init__(self, value):
                self.value = value

            def next_float(self, _stream="misc"):
                return self.value

        self.assertEqual(_roll_potion_rarity(FixedRng(0.05)), "rare")
        self.assertEqual(_roll_potion_rarity(FixedRng(0.3)), "uncommon")
        self.assertEqual(_roll_potion_rarity(FixedRng(0.9)), "common")

    def test_relic_reward_pulls_from_the_grab_bag(self):
        """``RelicReward.Populate``（``RelicReward.cs:75-94``）：抓包前端取走不再出现。"""
        from sts2_sim.runeffects import apply_run_effects
        from sts2_sim.content import Effect

        env, state = self._state()
        effect = Effect(op="offer_rewards", amount=1, target="self",
                        card_filter=(("kind", "relic"), ("rarity", "common")))
        log: list[str] = []
        self.assertTrue(apply_run_effects(state, (effect,), log))
        pulled = state.pending_rewards[0].options[0]
        self.assertIn(pulled, content.RELICS)
        self.assertNotIn(pulled, state.relic_bag._deques["common"],
                         "取走的遗物本局不再出现")

    def test_unknown_potion_id_is_not_offered(self):
        """候选来源不存在时必须"给不出来"，而不是给个空 id。"""
        from sts2_sim.runeffects import apply_run_effects
        from sts2_sim.content import Effect

        env, state = self._state()
        effect = Effect(op="offer_rewards", amount=1, target="self",
                        card_filter=(("kind", "potion"), ("potion", "nope_potion")))
        log: list[str] = []
        self.assertFalse(apply_run_effects(state, (effect,), log))
        self.assertFalse(state.pending_rewards)


# ==========================================================================
# 抽取修正（委派摊平 / 具体诅咒）
# ==========================================================================
class TestExtraHopExtraction(unittest.TestCase):
    """选项体只写一句私有方法调用时，效果必须在**那个方法里**。"""

    def test_wellspring_bathe_adds_the_curse(self):
        """``Wellspring.cs:44-55``：移除一张 **且** 加 1 张 Guilty。

        ⚠️ 少掉后面那半就是**白拿一次移除**，而且日志看不出来。
        """
        definition = content.EVENT_DB["wellspring"]
        bathe = next(o for o in definition.pages["INITIAL"].options
                     if o.name == "BATHE")
        ops = [(e["op"], e.get("card"), e.get("purpose"))
               for e in bathe.effects_raw]
        self.assertIn(("select_card", None, "removal"), ops)
        self.assertIn(("add_card", "guilty", None), ops)

    def test_unrest_site_rest_adds_the_specific_curse(self):
        """``UnrestSite.cs:44-50``：回血 + ``AddCursesToDeck([PoorSleep])``。"""
        definition = content.EVENT_DB["unrest_site"]
        rest = next(o for o in definition.pages["INITIAL"].options
                    if o.name == "REST")
        self.assertIn(("add_card", "poor_sleep"), [(e["op"], e.get("card"))
                                                   for e in rest.effects_raw])

    def test_bugslayer_generic_helper_resolves_the_card(self):
        """``Bugslayer.cs:39-44``：``AddAndPreview<T>`` 的 T 决定加哪张卡。"""
        definition = content.EVENT_DB["bugslayer"]
        self.assertTrue(definition.usable)
        cards = {o.name: [e.get("card") for e in o.effects_raw
                          if e["op"] == "add_card"]
                 for o in definition.pages["INITIAL"].options}
        self.assertEqual(cards["EXTERMINATION"], ["exterminate"])
        self.assertEqual(cards["SQUASH"], ["squash"])


# ==========================================================================
# 诚实性闸门
# ==========================================================================
class TestHonestyGates(unittest.TestCase):
    """算不出 / 点不动 / 走不到结束 —— 一律不可用。"""

    def _definition(self, initial_options, extra_pages=()):
        pages = {"INITIAL": events.EventPageDef(page_id="INITIAL",
                                                options=tuple(initial_options))}
        for page in extra_pages:
            pages[page.page_id] = page
        return events.EventDef(eid="x", name="x", pages=pages)

    def test_dead_end_page_is_flagged(self):
        """整页选项都"没有效果也没有去向" → 点下去会卡住。"""
        option = events.EventOptionDef(key="X.pages.INITIAL.options.A", name="A")
        gap = content._event_playability_gap(self._definition([option]))
        self.assertIn("没有效果也没有去向", gap)

    def test_page_without_live_options_is_flagged(self):
        """下一页全是锁住的选项 → 死页（Ancient 事件的选项池就是这个形态）。"""
        first = events.EventOptionDef(key="k1", name="A", outcome="goto",
                                      outcome_page="NEXT",
                                      effects_raw=({"op": "gain_gold"},))
        locked = events.EventOptionDef(key="k2", name="B", locked=True)
        gap = content._event_playability_gap(self._definition(
            [first], [events.EventPageDef(page_id="NEXT", options=(locked,))]))
        self.assertIn("没有可点的选项", gap)

    def test_event_without_a_finish_is_flagged(self):
        """只能无限翻页、走不到结束的事件也要挡住。"""
        first = events.EventOptionDef(key="k1", name="A", outcome="goto",
                                      outcome_page="NEXT",
                                      effects_raw=({"op": "gain_gold"},))
        second = events.EventOptionDef(key="k2", name="B", outcome="goto",
                                       outcome_page="INITIAL",
                                       effects_raw=({"op": "gain_gold"},))
        gap = content._event_playability_gap(self._definition(
            [first], [events.EventPageDef(page_id="NEXT", options=(second,))]))
        self.assertIn("结束", gap)

    def test_healthy_event_has_no_gap(self):
        first = events.EventOptionDef(key="k1", name="A", outcome="finished",
                                      effects_raw=({"op": "gain_gold"},))
        self.assertEqual(content._event_playability_gap(
            self._definition([first])), "")

    def test_pool_excludes_never_gated_events(self):
        """静态恒假的 ``IsAllowed`` 不进池（``WarHistorianRepy.cs:36-39``）。"""
        for eid in content.event_pool():
            self.assertNotEqual(content.EVENT_DB[eid].gate_kind, "never", eid)

    def test_unusable_events_always_carry_a_reason(self):
        for eid, definition in content.EVENT_DB.items():
            if not definition.usable:
                self.assertTrue(definition.reasons, f"{eid} 不可用却没写理由")


if __name__ == "__main__":
    unittest.main(verbosity=2)
