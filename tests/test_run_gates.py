"""Run 层"必须指哪张牌"的门禁 + 选牌张数的回归测试（``docs/12`` §2.18）。

三件事：

1. **Run 层的 `upgrade_card` / `downgrade_card` / `transform_card` 必须带明确目标卡**。
   真机这三个命令的目标往往来自 ``Rng.NextItem(候选)`` + ``Remove``（随机不重复）
   或 ``StableShuffle`` + ``Take``（随机 N 张）：:

       DoorsOfLightAndDark.Light:  Deck.Where(IsUpgradable).ToList()
                                   .StableShuffle(base.Rng).Take(Cards)
                                   → foreach CardCmd.Upgrade(item)
       Bellows:                    CardCmd.Upgrade(手牌**全部**)
       Reflections:                c = base.Rng.NextItem(upgradableCards); Remove(c); Upgrade(c)

   引擎的 Run 层只在"候选恰好 1 张"或"给了明确卡 id"时动手，其余只打一句
   ``⚠️ …需要玩家选择，跳过`` 然后**继续** —— 事件照样算"可跑"、遗物照样算
   "完全复刻"，而收益凭空消失。所以加载期就要拒绝。

2. **选牌张数来自变量时必须带上变量名**。旧实现只认 ``CardSelectorPrefs`` 里的
   字面量、其余**兜底成 1**：`YummyCookie`（真机升 4 张）在引擎里只升 1 张 ——
   静默变弱一档。

3. **"遍历选牌结果"的 foreach 变量也要去重**。``foreach (item in list)
   CardCmd.Upgrade(item)`` 里的升级已经由那条 ``select_card`` 的用途表达，
   不能再产出一条没有目标的 ``upgrade_card``。

⚠️ 这些测试证明的是"门禁判得对"，**不是**数值与真机一致（真机对拍为 0）。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"
MONSTERS = (ROOT / "data" / "decompiled" / "sts2"
            / "MegaCrit.Sts2.Core.Models.Monsters")


def restore_builtin() -> None:
    from sts2_sim.content import load_builtin
    from sts2_sim.featurize import rebuild_vocab
    load_builtin()
    rebuild_vocab()


def setup_content():
    if not CONTENT.exists():
        raise unittest.SkipTest(f"缺内容目录 {CONTENT}")
    from sts2_sim.featurize import configure_content
    configure_content(str(CONTENT))


class TestRunOpsNeedATarget(unittest.TestCase):
    """Run 层"必须指哪张牌"的算子：没有目标卡就整条拒绝。"""

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_random_target_events_are_now_adopted(self):
        """``doors_of_light_and_dark`` / ``endless_conveyor``：**已实现** → 可用。

        ⚠️ 这两者在 §2.18 被门禁拒掉（"跑得通但什么都不做"），
        §2.20 把"随机取 N 张不重复"做出来（含**事件自己的流** `base.Rng`）
        之后都转正：

        * `DoorsOfLightAndDark.Light` —— `StableShuffle(base.Rng).Take(Cards)`；
        * `EndlessConveyor.ObserveChef` —— `CardCmd.Upgrade(base.Rng.NextItem(…))`。
        """
        from sts2_sim import content
        for eid in ("doors_of_light_and_dark", "endless_conveyor"):
            definition = content.EVENT_DB.get(eid)
            self.assertIsNotNone(definition, f"{eid} 应当在事件表里")
            self.assertFalse(definition.reasons,
                             f"{eid} 现在应当可用：{definition.reasons}")

    def test_unknown_random_stream_is_still_refused(self):
        """门禁不能因为"实现了随机取牌"就整体放开：**流名不认识**仍要拒。

        把 `Rng.Niche` 写成 `Rng.SomewhereElse` 的钩子必须整条拒绝 ——
        运行期只会打一行 ⚠️ 跳过，而报告会说这个遗物"完全复刻"。
        """
        from sts2_sim.content import _run_effect_defect
        good = {"op": "random_deck_cards", "purpose": "upgrade", "rng": "niche",
                "pick": "shuffle"}
        self.assertIsNone(_run_effect_defect(good))
        self.assertIsNone(_run_effect_defect({**good, "rng": "event"}))
        self.assertIsNotNone(_run_effect_defect({**good, "rng": "somewhere"}))
        self.assertIsNotNone(_run_effect_defect({**good, "pick": "magic"}))
        self.assertIsNotNone(_run_effect_defect({**good, "purpose": "enchant"}))

    def test_bellows_upgrades_the_whole_hand(self):
        """``bellows``：真机 ``CardCmd.Upgrade(PileType.Hand.GetPile(owner).Cards)``
        —— **整手牌**升级（§2.24 实现了 `upgrade_hand`）。

        ⚠️ 这条测试原来叫 `test_relics_with_random_upgrade_are_refused`
        （§2.18 把"没有目标卡的 `upgrade_card`"整条拒了）。现在那个动作
        **被建出来了**（判据是"手牌那一堆**且没有 `.Where(...)`**"），
        所以换成正向断言 —— 门禁的牙齿由
        `test_unknown_random_stream_is_still_refused` 与
        `test_conditional_pile_is_a_gap` 继续守着。
        """
        from sts2_sim import content
        relic = content.RELICS.get("bellows")
        self.assertIsNotNone(relic, "bellows 应当在遗物表里")
        self.assertTrue(relic.has_behavior, "它现在应当有已复刻行为")
        ops = [e.op for hook in relic.hooks for e in hook.effects]
        self.assertIn("upgrade_hand", ops, f"应当是整手牌升级：{ops}")

    def test_upgrade_hand_upgrades_every_card_in_hand(self):
        """引擎行为：手牌**全部**升级，已升级的保持原样（真机是空操作）。"""
        from sts2_sim.content import Effect
        from sts2_sim.core import CardInstance, _apply_one, start_combat
        from sts2_sim.content import STARTING_DECK

        state = start_combat(STARTING_DECK, ("nibbit",), seed=1)
        state.hand = [CardInstance("strike_ironclad"),
                      CardInstance("defend_ironclad", upgraded=True),
                      CardInstance("bash")]
        _apply_one(state, Effect(op="upgrade_hand", amount=1, target="self"),
                   state.player, -1, [])
        self.assertTrue(all(card.upgraded for card in state.hand),
                        "整手牌都该是升级版")

    def test_deck_selection_relics_are_now_adopted(self):
        """``pomander`` / ``yummy_cookie``：真机是**玩家从牌组选牌升级** → 可用。

        它们此前带着一条多余的 ``upgrade_card``（没有目标卡）被门禁拒掉；
        现在那条被正确识别成"选牌用途的表达"（``select_card(purpose=upgrade)``），
        于是它们变成**正确可用**。
        """
        from sts2_sim import content
        for rid in ("pomander", "yummy_cookie"):
            relic = content.RELICS.get(rid)
            self.assertIsNotNone(relic, f"{rid} 应当在遗物表里")
            self.assertTrue(relic.has_behavior, f"{rid} 现在应当有行为")
            effects = [e for hook in relic.hooks for e in hook.effects]
            picks = [e for e in effects if e.op == "select_card"]
            self.assertTrue(picks, f"{rid} 应当是一条选牌效果：{effects}")
            self.assertEqual(picks[0].purpose, "upgrade")
            self.assertEqual(picks[0].select_from, "deck")


class TestSelectionCounts(unittest.TestCase):
    """选牌张数：来自变量的必须带上变量名，不许兜底成 1。"""

    @classmethod
    def setUpClass(cls):
        setup_config = None
        del setup_config
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_variable_count_carries_its_variable(self):
        """``YummyCookie``（真机 ``CardsVar(4)``）的张数必须跟着变量走。

        旧实现只认字面量、其余兜底成 1 → 真机升 4 张、引擎升 1 张（静默变弱）。
        """
        from sts2_sim import content
        relic = content.RELICS["yummy_cookie"]
        picks = [e for hook in relic.hooks for e in hook.effects
                 if e.op == "select_card"]
        self.assertTrue(picks)
        declared = None
        record = next(r for r in json.loads(
            (CONTENT / "relics_source.json").read_text(encoding="utf-8"))
            if r["rid"] == "yummy_cookie")
        effect_json = next(e for e in record["hooks"]["AfterObtained"]["effects"]
                           if e["op"] == "select_card")
        declared = effect_json.get("amount_var")
        self.assertEqual(declared, "Cards",
                         "张数应当由声明的变量表达，而不是写死")
        # 引擎解析后的实际张数 = 源码声明的 Cards 值（4），不是 1
        self.assertEqual(picks[0].amount, 4,
                         f"YummyCookie 应当升级 4 张：{picks[0]}")

    def test_variable_count_selection_is_not_rejected_as_variable(self):
        """⚠️ "变量张数"与"可变张数"是两回事：前者可解析，后者要拒绝。

        ``CardSelectorPrefs(prompt, Cards)`` 是"必须选 Cards 张"（可执行）；
        ``CardSelectorPrefs(prompt, 0, MAX)`` 才是"任意张"（引擎没有结束选择的动作）。
        """
        from sts2_sim.content import _source_reject_reason
        # 用**战斗侧**合法用途（`to_hand`）：`_source_reject_reason` 读的是
        # ENGINE_SELECTION_PURPOSES，Run 层的 `upgrade` 由 `_relic_hook` 校验。
        declared = {"op": "select_card", "from": "discard", "purpose": "to_hand",
                    "amount": None, "amount_var": "Cards", "target": "self"}
        self.assertIsNone(_source_reject_reason({"effects": [declared]},
                                                require_cost=False))
        variable = {"op": "select_card", "from": "discard", "purpose": "to_hand",
                    "amount": 0, "target": "self"}
        self.assertIsNotNone(_source_reject_reason({"effects": [variable]},
                                                   require_cost=False))


@unittest.skipUnless(MONSTERS.is_dir(), "缺反编译源码")
class TestMoveBranching(unittest.TestCase):
    def test_conditional_pile_is_a_gap(self):
        """``TheInsatiable.LiquifyMove``：落点按条件分叉 → 必须报缺口。

        真机 ``PileType newPileType = (i < 3) ? Draw : Discard;`` —— 6 张
        `FranticEscape` **前 3 张进抽牌堆、后 3 张进弃牌堆**。
        引擎的 `add_card` 只有一个 `pile`：取第一个会让 6 张**全进弃牌堆**，
        而这条招式在报告里是"干净"的。分叉写在**局部变量**里，所以判据
        也必须跟一层局部变量（第一版只看实参，整条漏掉）。
        """
        from tools import extract_moves
        record = extract_moves.extract_monster(
            next(MONSTERS.rglob("TheInsatiable.cs")))
        move = record["moves"]["LiquifyMove"]
        self.assertTrue(any("落点按条件分叉" in item for item in move["unsupported"]),
                        f"应当报落点分叉缺口：{move['unsupported']}")
        self.assertFalse([e for e in move["effects"] if e["op"] == "add_card"],
                         "分叉的加牌不该留在效果表里")


class TestRandomDeckCards(unittest.TestCase):
    """``random_deck_cards``：从牌组随机取 N 张不重复，再对它们做一件事。

    真机（`Whetstone` / `WarPaint`）::

        Deck.Cards.Where(c => c.Type == CardType.Attack && c.IsUpgradable).ToList()
            .StableShuffle(owner.RunState.Rng.Niche).Take(Cards);
        foreach (item in …) CardCmd.Upgrade(item);
    """

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def _state(self, seed: int = 7):
        from sts2_sim.core import CardInstance
        from sts2_sim.rng import RngSet

        class Hidden:
            rng = RngSet(seed)

        class Player:
            def __init__(self):
                self.hp = self.max_hp = 80
                self.gold = 0
                self.deck = ([CardInstance("strike_ironclad") for _ in range(4)]
                             + [CardInstance("defend_ironclad") for _ in range(2)])
                self.relics: list[str] = []

        class State:
            def __init__(self):
                self.player = Player()
                self.hidden = Hidden()
                self.pending_selection = None
                self.pending_rewards: list = []

        return State()

    def _effects(self, rid: str):
        from sts2_sim import content
        return [e for hook in content.RELICS[rid].hooks for e in hook.effects]

    def test_whetstone_upgrades_two_random_attacks(self):
        """只升级**攻击**牌，且恰好 2 张（`Cards=2`）—— 不是"任意两张"。"""
        from sts2_sim.runeffects import apply_run_effects
        state = self._state()
        apply_run_effects(state, self._effects("whetstone"), [])
        upgraded = [c for c in state.player.deck if c.upgraded]
        self.assertEqual(len(upgraded), 2, f"应当升级 2 张：{upgraded}")
        self.assertTrue(all(c.cid == "strike_ironclad" for c in upgraded),
                        "真机的谓词是 `c.Type == CardType.Attack`，防御牌不该被动")

    def test_war_paint_upgrades_skills(self):
        """`WarPaint` 的谓词是**技能**牌 —— 恰好与 Whetstone 相反。"""
        from sts2_sim.runeffects import apply_run_effects
        state = self._state()
        apply_run_effects(state, self._effects("war_paint"), [])
        upgraded = [c for c in state.player.deck if c.upgraded]
        self.assertTrue(upgraded, "应当有牌被升级")
        self.assertTrue(all(c.cid == "defend_ironclad" for c in upgraded),
                        f"只该升级技能牌：{upgraded}")

    def test_it_is_random_not_first_n(self):
        """**随机**取，不是"取前 N 张"：不同种子应当升级不同的牌。

        丢掉 `StableShuffle`/随机流就退化成"升级牌组里最前面两张" ——
        分布完全不同，而日志看不出差别。
        """
        from sts2_sim.runeffects import apply_run_effects
        picked = set()
        for seed in range(12):
            state = self._state(seed)
            apply_run_effects(state, self._effects("whetstone"), [])
            picked.add(tuple(c.uid for c in state.player.deck if c.upgraded))
        self.assertGreater(len(picked), 1,
                           "12 个种子里升级的始终是同一组牌 —— 随机流没接上")


class TestRandomRemoveLoop(unittest.TestCase):
    """``NextItem`` + ``Remove`` 的**循环**写法，以及 ``clone_deck``。

    真机（`Reflections.TouchAMirror`）::

        List<CardModel> upgradableCards = Deck.Cards.Where(c => c.IsUpgradable).ToList();
        for (int i = 0; i < 4; i++) {
            if (upgradableCards.Count <= 0) break;
            CardModel card = base.Rng.NextItem(upgradableCards);
            upgradableCards.Remove(card);              // ← 取一张删一张 = 不重复
            CardCmd.Upgrade(card, CardPreviewStyle.MessyLayout);
        }
    """

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_reflections_touch_a_mirror_is_adopted(self):
        """`Reflections`：降级 2 张（已升级的）+ 升级 4 张，都是**事件流**。"""
        from sts2_sim import content
        definition = content.EVENT_DB.get("reflections")
        self.assertIsNotNone(definition, "reflections 应当在事件表里")
        self.assertFalse(definition.reasons,
                         f"它现在应当可用：{definition.reasons}")
        option = next(o for page in definition.pages.values()
                      for o in page.options if o.name == "TOUCH_A_MIRROR")
        picks = [e for e in option.effects_raw if e.get("op") == "random_deck_cards"]
        self.assertEqual(len(picks), 2, f"该有降级 + 升级两条：{option.effects_raw}")
        kinds = {e["purpose"]: e for e in picks}
        self.assertEqual(set(kinds), {"downgrade", "upgrade"})
        self.assertEqual(kinds["downgrade"]["amount"], 2)
        self.assertEqual(kinds["upgrade"]["amount"], 4)
        for effect in picks:
            self.assertEqual(effect["rng"], "event", "事件里用的是 base.Rng")
            self.assertEqual(effect["pick"], "item", "NextItem + Remove = item 取法")

    def test_reflections_shatter_clones_the_deck(self):
        """`Shatter`：整副牌组复制一份 + 加一张 `bad_luck`。"""
        from sts2_sim import content
        definition = content.EVENT_DB["reflections"]
        option = next(o for page in definition.pages.values()
                      for o in page.options if o.name == "SHATTER")
        ops = [e.get("op") for e in option.effects_raw]
        self.assertIn("clone_deck", ops, f"应当有 clone_deck：{option.effects_raw}")
        self.assertIn("add_card", ops)

    def test_clone_deck_doubles_the_deck(self):
        """``clone_deck`` 的引擎行为：牌组翻倍，且保留升级状态。

        ⚠️ 实现必须先把原牌**快照**下来再追加 —— 直接迭代 `player.deck`
        会边加边遍历（死循环 / 无限增长）。
        """
        from sts2_sim.core import CardInstance
        from sts2_sim.runeffects import apply_run_effects
        from sts2_sim.rng import RngSet

        class Hidden:
            rng = RngSet(3)

        class Player:
            def __init__(self):
                self.hp = self.max_hp = 80
                self.gold = 0
                self.deck = [CardInstance("strike_ironclad"),
                             CardInstance("defend_ironclad", upgraded=True)]
                self.relics: list[str] = []

        class State:
            def __init__(self):
                self.player = Player()
                self.hidden = Hidden()
                self.pending_selection = None
                self.pending_rewards: list = []

        state = State()
        apply_run_effects(state, [_clone_deck_effect()], [])
        self.assertEqual(len(state.player.deck), 4, "牌组应当翻倍")
        self.assertEqual(sum(1 for c in state.player.deck if c.upgraded), 2,
                         "升级状态要跟着副本走")
        self.assertEqual(len({c.uid for c in state.player.deck}), 4,
                         "副本必须是**新实例**（uid 不同）")


def _clone_deck_effect():
    from sts2_sim.content import Effect
    return Effect(op="clone_deck", amount=0, target="self")


class TestAddRandomCards(unittest.TestCase):
    """``add_random_cards``：从**卡池**随机取 N 张不重复 → 加入牌组。

    真机（`DistinguishedCape.AfterObtained`）::

        pool = CardPool<CurseCardPool>().GetUnlockedCards(…)
                   .Where(c => c.CanBeGeneratedByModifiers).ToList();
        for (i < Curses) {
            c = Rng.Niche.NextItem(pool);  pool.Remove(c);
            Add(RunState.CreateCard(c, owner), PileType.Deck);
        }
        for (i < Cards) Add(CreateCard<Apparition>(owner), PileType.Deck);
    """

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_distinguished_cape_is_adopted(self):
        from sts2_sim import content
        relic = content.RELICS.get("distinguished_cape")
        self.assertIsNotNone(relic)
        self.assertTrue(relic.has_behavior, "它现在应当有已复刻行为")
        effects = [e for hook in relic.hooks for e in hook.effects]
        random_adds = [e for e in effects if e.op == "add_random_cards"]
        self.assertEqual(len(random_adds), 1, effects)
        self.assertEqual(dict(random_adds[0].card_filter).get("pool"),
                         "CurseCardPool")
        self.assertTrue(dict(random_adds[0].card_filter).get("generatable"),
                        "真机的候选过 `c.CanBeGeneratedByModifiers`")
        self.assertEqual(random_adds[0].rng, "niche")
        self.assertEqual(random_adds[0].pick, "item")
        # 第二段：`for (i < Cards) CreateCard<Apparition>()` —— **Cards 张**，
        # 不是 1 张（`CardPileCmd.Add` 分支绕过了循环次数套用，已补）。
        apparitions = [e for e in effects
                       if e.op == "add_card" and e.card == "apparition"]
        self.assertEqual(len(apparitions), 3, f"应当展开成 Cards(3) 张：{apparitions}")

    def test_not_generated_table_is_current(self):
        """``NOT_GENERATED_BY_MODIFIERS`` 必须与源码一致（扫描源码钉住）。

        上游新增"不能由修饰符生成"的卡时，这条测试会红 —— 而不是让真机
        发不出的牌静默出现在玩家牌组里。
        """
        import re
        from sts2_sim.content import NOT_GENERATED_BY_MODIFIERS
        from tools.extract_cards import snake
        cards_dir = (ROOT / "data" / "decompiled" / "sts2"
                     / "MegaCrit.Sts2.Core.Models.Cards")
        if not cards_dir.is_dir():
            self.skipTest("缺反编译源码")
        found = {snake(path.stem) for path in cards_dir.glob("*.cs")
                 if re.search(r"override bool CanBeGeneratedByModifiers",
                              path.read_text(encoding="utf-8", errors="replace"))}
        self.assertEqual(found, set(NOT_GENERATED_BY_MODIFIERS),
                         f"表与源码不一致：源码多 "
                         f"{sorted(found - set(NOT_GENERATED_BY_MODIFIERS))}，"
                         f"表多 {sorted(set(NOT_GENERATED_BY_MODIFIERS) - found)}")


class TestDuplicateSelection(unittest.TestCase):
    """`duplicate` 用途：`DollysMirror`（选一张 → 克隆 → 加回牌组）。

    真机::

        cardModel = await CardSelectCmd.FromDeckGeneric(prefs, 1, filter: Filter);
        if (cardModel != null)
            await CardPileCmd.Add(RunState.CloneCard(cardModel), PileType.Deck);

    用途判不出来（`FromDeckGeneric` 的 prompt 是通用的 `SelectionScreenPrompt`），
    只能从**后续动作** `CloneCard` 反推。
    """

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_dollys_mirror_is_adopted_with_duplicate_purpose(self):
        from sts2_sim import content
        relic = content.RELICS.get("dollys_mirror")
        self.assertIsNotNone(relic, "dollys_mirror 应当在遗物表里")
        self.assertTrue(relic.has_behavior, "它现在应当有已复刻行为")
        picks = [e for hook in relic.hooks for e in hook.effects
                 if e.op == "select_card"]
        self.assertEqual(len(picks), 1, f"应当恰好一条选牌：{picks}")
        self.assertEqual(picks[0].purpose, "duplicate")
        self.assertEqual(picks[0].select_from, "deck")

    def test_duplicate_copies_the_chosen_card(self):
        """引擎行为：选中的那张被**复制一份**（新实例、同升级状态）。"""
        from sts2_sim.core import CardInstance
        from sts2_sim.runeffects import RunSelection, resolve_run_selection
        from sts2_sim.rng import RngSet

        class Player:
            def __init__(self):
                self.hp = self.max_hp = 80
                self.gold = 0
                self.deck = [CardInstance("strike_ironclad"),
                             CardInstance("defend_ironclad", upgraded=True)]
                self.relics: list[str] = []

        class State:
            def __init__(self):
                self.player = Player()
                self.hidden = type("H", (), {"rng": RngSet(5)})()
                self.pending_selection = None
                self.pending_rewards: list = []

        state = State()
        original = state.player.deck[1]
        state.pending_selection = RunSelection(
            purpose="duplicate", count=1, source="deck", candidates=(1,))
        resolve_run_selection(state, 1, [])
        self.assertEqual(len(state.player.deck), 3, "牌组应当多一张")
        copy = state.player.deck[-1]
        self.assertEqual(copy.cid, original.cid)
        self.assertTrue(copy.upgraded, "升级状态要跟着副本走")
        self.assertNotEqual(copy.uid, original.uid, "必须是**新实例**")


class TestRandomDrawCards(unittest.TestCase):
    """``random_draw_cards``：战斗内从**抽牌堆**随机取 N 张升级（`StoneCracker`）。

    真机（`StoneCracker.AfterRoomEntered`）::

        PileType.Draw.GetPile(owner).Cards.Where(c => c.IsUpgradable)
            .ToList().StableShuffle(RunState.Rng.CombatCardSelection).Take(Cards);
        CardCmd.Upgrade(cards, CardPreviewStyle.HorizontalLayout);

    ⚠️ 与 `random_deck_cards`（Run 层的**牌组**）是两回事：这里动的是
    **战斗内的抽牌堆**，用的是战斗侧的 `combat_card_selection` 流。
    """

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_stone_cracker_is_adopted(self):
        from sts2_sim import content
        relic = content.RELICS.get("stone_cracker")
        self.assertIsNotNone(relic, "stone_cracker 应当在遗物表里")
        self.assertTrue(relic.has_behavior, "它现在应当有已复刻行为")
        effects = [e for hook in relic.hooks for e in hook.effects]
        draws = [e for e in effects if e.op == "random_draw_cards"]
        self.assertEqual(len(draws), 1, f"应当恰好一条：{effects}")
        self.assertEqual(draws[0].rng, "combat_card_selection",
                         "真机用的是战斗侧的这条流，不是 niche")
        self.assertTrue(dict(draws[0].card_filter).get("upgradable"))

    def test_it_upgrades_random_cards_in_the_draw_pile(self):
        """引擎行为：抽牌堆里被选中的那 N 张变成升级版。"""
        from sts2_sim.content import STARTING_DECK, Effect
        from sts2_sim.core import CardInstance, _apply_one, start_combat

        state = start_combat(STARTING_DECK, ("nibbit",), seed=4)
        state.draw_pile = [CardInstance("strike_ironclad") for _ in range(4)]
        state.hand = []
        _apply_one(state, Effect(op="random_draw_cards", amount=2, target="self",
                                 card_filter=(("upgradable", True),),
                                 rng="combat_card_selection", pick="shuffle"),
                   state.player, -1, [])
        self.assertEqual(sum(1 for c in state.draw_pile if c.upgraded), 2,
                         f"应当恰好升级 2 张：{state.draw_pile}")


if __name__ == "__main__":
    unittest.main()
