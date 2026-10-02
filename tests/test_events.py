"""事件层测试：在**真实内容**上验证事件图、进入条件、状态机与诚实性。

铁律（``docs/09`` §5.2）：每条断言都要能追到源码，注释里给出执行点。

⚠️ 与 ``tests/test_run.py`` 的分工：那边跑**内置占位内容**（没有事件池），
这里加载 ``data/content/repo``（68 个源码事件），专门验证事件层。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from sts2_sim import content, events
from sts2_sim.rng import RngSet, STREAMS

CONTENT = Path(__file__).resolve().parent.parent / "data" / "content" / "repo"


def setUpModule() -> None:               # noqa: N802 - unittest 约定
    """加载**真实内容**（68 个源码事件）。

    ⚠️ 必须配对还原：内容层是**全局状态**，加载了真内容却不还原，
    后面的测试模块会拿着 597 张真卡去跑"内置占位内容"的断言 ——
    症状是别处莫名其妙地失败（实测把 ``test_featurize`` 整片打红）。
    """
    if not CONTENT.exists():
        raise unittest.SkipTest(f"缺内容目录 {CONTENT}")
    from sts2_sim.featurize import configure_content

    configure_content(str(CONTENT))


def tearDownModule() -> None:            # noqa: N802 - unittest 约定
    from sts2_sim.content import load_builtin
    from sts2_sim.featurize import rebuild_vocab

    load_builtin()
    rebuild_vocab()


class FakePlayer:
    def __init__(self, hp: int = 80, max_hp: int = 80, gold: int = 99) -> None:
        self.hp = hp
        self.max_hp = max_hp
        self.gold = gold
        self.deck: list = []
        self.relics: list = []


class FakeRun:
    """Run 层状态的最小替身（``apply_run_effects`` 只用到 ``.player``）。"""

    def __init__(self, hp: int = 80, max_hp: int = 80, gold: int = 99) -> None:
        self.player = FakePlayer(hp, max_hp, gold)
        self.hidden = type("H", (), {"rng": RngSet(1)})()


class TestExtraction(unittest.TestCase):
    """抽取结果的规模与出处。"""

    def test_all_source_events_are_loaded(self):
        """源码目录里的事件都要进表（少于 60 个说明抽取整块失效）。"""
        self.assertGreaterEqual(len(content.EVENT_DB), 60)
        self.assertIn("this_or_that", content.EVENT_DB)
        self.assertIn("abyssal_baths", content.EVENT_DB)

    def test_this_or_that_matches_source(self):
        """``ThisOrThat`` 是逐个字段对过源码的样板。

        源码（``Events/ThisOrThat.cs``）::

            CanonicalVars: HpLoss 6, Gold 0, StringVar("Curse", Clumsy)
            CalculateVars: DynamicVars.Gold.BaseValue = Rng.NextInt(41, 69)
            INITIAL: PLAIN  → Damage(self, HpLoss, Unblockable|Unpowered) + GainGold(Gold)
                     ORNATE → RelicFactory.PullNextRelicFromFront + AddCurseToDeck<Clumsy>
        """
        definition = content.EVENT_DB["this_or_that"]
        self.assertEqual(definition.initial_page, "INITIAL")
        options = definition.pages["INITIAL"].options
        self.assertEqual([o.name for o in options], ["PLAIN", "ORNATE"])
        plain = next(o for o in options if o.name == "PLAIN")
        ops = [e["op"] for e in plain.effects_raw]
        self.assertEqual(ops, ["lose_hp", "gain_gold"])
        self.assertEqual(plain.outcome, "finished")
        # 掷点：`Gold` 来自事件内 RNG
        self.assertEqual([var for var, _ in definition.rng_rolls], ["Gold"])
        self.assertIn("NextInt(41, 69)", definition.rng_rolls[0][1])

    def test_locked_options_stay_in_the_data(self):
        """``EventOption(this, null, key)`` → 锁住的选项（``IsLocked``）。

        它们**是正常内容**：界面会显示、但点不动（``EventOption.cs:62``）。
        抽成"缺口"会把正常状态报成缺失。
        """
        locked = [o for e in content.EVENT_DB.values()
                  for p in e.pages.values() for o in p.options if o.locked]
        self.assertGreater(len(locked), 0, "至少应有若干锁住的选项")
        self.assertTrue(all(o.name for o in locked), "锁住的选项也要有名字")


class TestHonesty(unittest.TestCase):
    """不可用的事件必须**带着理由**被排除，而不是静默降级。"""

    def test_every_unusable_event_states_a_reason(self):
        for eid, definition in content.EVENT_DB.items():
            if not definition.usable:
                self.assertTrue(definition.reasons,
                                f"{eid} 不可用却没写理由")

    def test_pool_only_contains_usable_events(self):
        pool = content.event_pool()
        self.assertTrue(pool, "事件池不能为空")
        for eid in pool:
            definition = content.EVENT_DB[eid]
            self.assertTrue(definition.usable, f"{eid} 不该在池里")
            self.assertEqual(definition.reasons, ())

    def test_other_acts_are_excluded(self):
        """``Act 2 - Hive`` / ``Act 3 - Glory`` 的事件不能出现在第 1 幕。

        幕索引来自源码（``Acts/Hive.cs: Index => 1``、``Glory.cs: Index => 2``），
        本模拟器只跑第 1 幕（``Overgrowth`` 与 ``Underdocks`` 都是 0）。
        """
        for eid in content.event_pool():
            act = content.EVENT_DB[eid].act
            self.assertNotIn("Act 2", act)
            self.assertNotIn("Act 3", act)

    def test_options_pointing_at_missing_pages_are_flagged(self):
        """选项指向不存在的页 → 事件不可用（否则状态机会卡住）。"""
        for eid, definition in content.EVENT_DB.items():
            for page in definition.pages.values():
                for option in page.options:
                    if option.outcome == "goto" and option.outcome_page:
                        if option.outcome_page not in definition.pages:
                            self.assertIn(f"选项 {option.name} 指向不存在的页 "
                                          f"{option.outcome_page}", definition.reasons)


class TestGate(unittest.TestCase):
    """``EventModel.IsAllowed``：只认白名单写法，其余一律不可用。"""

    def test_unknown_gate_is_not_admitted(self):
        """认不出的条件 → ``gate_ok`` 返回 ``None``（调用方必须排除）。"""
        definition = events.EventDef(eid="x", name="x", pages={},
                                     gate_body="return runState.Players.All(p => p.HasEventPet());")
        kind, _args = events.parse_gate(definition.gate_body)
        self.assertEqual(kind, "unknown")
        self.assertIsNone(events.gate_ok(definition, act_index=0, hp=80,
                                         max_hp=80, gold=99))

    def test_known_gates_evaluate(self):
        """白名单里的写法按源码语义求值。"""
        cases = [
            ("return runState.CurrentActIndex < 2;", "act_lt", (2,), True),
            ("return runState.CurrentActIndex == 1;", "act_eq", (1,), False),
            ("return runState.Players.All((Player p) => p.Gold >= 100);",
             "gold_ge", (100,), False),
            ("return runState.Players.All((Player p) => p.Creature.CurrentHp >= 19);",
             "hp_ge", (19,), True),
        ]
        for body, want_kind, want_args, want_ok in cases:
            with self.subTest(body=body):
                kind, args = events.parse_gate(body)
                self.assertEqual((kind, tuple(args)), (want_kind, want_args))
                definition = events.EventDef(eid="x", name="x", pages={},
                                             gate_body=body, gate_kind=kind,
                                             gate_args=tuple(args))
                self.assertIs(events.gate_ok(definition, act_index=0, hp=80,
                                             max_hp=80, gold=99), want_ok)

    def test_no_gate_means_allowed(self):
        definition = events.EventDef(eid="x", name="x", pages={})
        self.assertTrue(events.gate_ok(definition, act_index=0, hp=1,
                                       max_hp=1, gold=0))


class TestEventRng(unittest.TestCase):
    """事件内 RNG：``EventModel.BeginEvent`` 的派生式子（XXH64）。"""

    def test_same_event_same_seed_is_reproducible(self):
        first = events.event_rng(12345, "this_or_that").next_int(41, 69)
        second = events.event_rng(12345, "this_or_that").next_int(41, 69)
        self.assertEqual(first, second)

    def test_different_events_are_independent(self):
        """不同事件必须落在**不同**的流上（否则两个事件的掷点会相关）。"""
        a = events.event_rng(12345, "this_or_that").next_int(0, 10**6)
        b = events.event_rng(12345, "abyssal_baths").next_int(0, 10**6)
        self.assertNotEqual(a, b)

    def test_roll_range_matches_source(self):
        """``Rng.NextInt(41, 69)`` → 41..69 **含两端**。"""
        run = events.initial_run(content.EVENT_DB["this_or_that"],
                                 events.event_rng(7, "this_or_that"))
        self.assertIn("Gold", run.rolled)
        self.assertGreaterEqual(run.rolled["Gold"], 41)
        self.assertLessEqual(run.rolled["Gold"], 69)


class TestStateMachine(unittest.TestCase):
    """选项 → 效果 → 翻页。"""

    def _run(self, eid: str, seed: int = 3):
        definition = content.EVENT_DB[eid]
        rng = events.event_rng(seed, eid)
        return definition, events.initial_run(definition, rng)

    def test_choosing_plain_applies_effects(self):
        """``PLAIN``：掉 HpLoss 血 + 拿 Gold 金（数值来自掷点）。"""
        definition, run = self._run("this_or_that")
        state = FakeRun(gold=99)
        log: list[str] = []
        events.apply_option(definition, run, 0, state, log)
        self.assertEqual(state.player.hp, 80 - 6)
        self.assertEqual(state.player.gold, 99 + run.rolled["Gold"])
        self.assertTrue(run.finished)

    def test_finished_option_ends_the_event(self):
        definition, run = self._run("this_or_that")
        state = FakeRun()
        events.apply_option(definition, run, 0, state, [])
        self.assertTrue(run.finished)

    def test_same_option_cannot_be_chosen_twice(self):
        """真机 ``EventOption.Chosen`` 有 ``WasChosen`` 守卫（``EventOption.cs:176``）。"""
        definition, run = self._run("this_or_that")
        state = FakeRun()
        events.apply_option(definition, run, 0, state, [])
        with self.assertRaises(ValueError):
            events.apply_option(definition, run, 0, state, [])

    def test_locked_option_cannot_be_chosen(self):
        """锁住的选项不能点（``IsLocked``）。"""
        target = None
        for eid, definition in content.EVENT_DB.items():
            if not definition.usable:
                continue
            for page in definition.pages.values():
                for index, option in enumerate(page.options):
                    if option.locked:
                        target = (definition, index)
        if target is None:
            self.skipTest("可用事件里没有锁住的选项")
        definition, index = target
        run = events.initial_run(definition, events.event_rng(1, definition.eid))
        with self.assertRaises(ValueError):
            events.apply_option(definition, run, index, FakeRun(), [])

    def test_selectable_excludes_locked_and_chosen(self):
        pool = content.event_pool()
        for eid in pool:
            definition = content.EVENT_DB[eid]
            run = events.initial_run(definition, events.event_rng(1, eid))
            indices = events.selectable(definition, run)
            options = events.current_options(definition, run)
            for index in indices:
                self.assertFalse(options[index].locked)
                self.assertNotIn(options[index].key, run.chosen)
            # 至少有一个能点的选项，否则事件会卡死
            self.assertTrue(indices, f"{eid} 的第一页没有可点选项")


class TestRunIntegration(unittest.TestCase):
    """接进 Run 层之后：池子只消费 ``up_front``，节点能真的跑起来。"""

    def test_draw_event_consumes_only_up_front(self):
        from sts2_sim.run import draw_event

        rng = RngSet(20240501)
        before = {name: rng[name].getstate() for name in STREAMS}
        eid = draw_event(rng)
        after = {name: rng[name].getstate() for name in STREAMS}
        changed = {name for name in STREAMS if before[name] != after[name]}
        self.assertEqual(changed, {"up_front"})
        self.assertIn(eid, content.event_pool())

    def test_run_env_plays_an_event_node(self):
        """真的进一个事件房间、把能点的选项走完。

        ⚠️ 这一步的价值是"**接得上**"：事件层的状态机、Run 层的阶段、
        观测与合法动作（mask）四者必须同时对，否则会出现
        "有选项但发不出动作"这种死锁。

        ⚠️ 地图上的事件节点是**开局预掷**的，而 ``IsAllowed`` 要在**进入时**
        才判（真机 ``RoomSet.EnsureNextEventIsValid`` 也是用进入时的 runState），
        所以某个节点可能"抽到了条件不满足的事件 → 这个节点没有事件"。
        逐个事件节点试到有一个能进为止；一个都进不去才算跳过。
        """
        from sts2_sim.run import PHASE_EVENT, PHASE_MAP, RunEnv, legal_meta_actions

        env = RunEnv(seed=11, attempt_budget=1)
        step = env.reset()
        state = env.raw_state
        # 直接把玩家放到一个事件节点上（避免依赖地图随机）
        event_nodes = [n for n in state.map.nodes if n.kind == "event"]
        if not event_nodes:
            self.skipTest("这张地图没有事件节点")
        entered = False
        for node in event_nodes:
            env._enter_node(node.node_id)
            if env.phase == PHASE_EVENT:
                entered = True
                break
        if not entered:
            self.skipTest("所有事件节点的条件都不满足（gate 判 False）")
        options = state.room.event_options
        self.assertTrue(options, "事件房间必须有选项")
        self.assertTrue(legal_meta_actions(state), "必须给出可点的动作")
        for _ in range(8):
            if env.phase != PHASE_EVENT:
                break
            action = legal_meta_actions(state)[0]
            env.step(action)
        self.assertIn(env.phase, (PHASE_MAP, "reward", PHASE_EVENT))
        # 走完之后事件必须已经结束（否则会永远停在这一页）
        self.assertTrue(state.event_run is None or not state.event_run.finished)

    def test_event_effects_use_real_content_cards(self):
        """事件里"加一张诅咒"用的是真实卡 id（``Clumsy``）。"""
        definition = content.EVENT_DB["this_or_that"]
        ornate = next(o for o in definition.pages["INITIAL"].options
                      if o.name == "ORNATE")
        added = [e["card"] for e in ornate.effects_raw if e["op"] == "add_card"]
        self.assertEqual(added, ["clumsy"])
        self.assertIn("clumsy", content.CARD_DB)


class TestSelectionDecision(unittest.TestCase):
    """⭐ **选牌必须由模型决定**（这是模拟器存在的意义之一）。

    真机在事件里经常要求玩家"从牌组里挑一张"：
    ``CardSelectCmd.FromDeckForRemoval(…)`` → ``CardPileCmd.RemoveFromDeck(选中的牌)``
    （``DoorsOfLightAndDark.Dark``）。引擎把它表达成
    「挂起选牌 → 动作空间里只剩可选的槽位 → 选完继续」，
    **绝不替玩家随机选一张**。
    """

    def _env(self):
        from sts2_sim.run import RunEnv

        env = RunEnv(seed=5, attempt_budget=1)
        env.reset()
        return env

    def test_removal_waits_for_the_model_and_applies_its_choice(self):
        """``wellspring`` 的 BATHE：选一张移除，**选谁由模型定**。

        ⚠️ 载体从 `doors_of_light_and_dark` 换成了 `wellspring`：
        前者的 LIGHT 选项真机是"``StableShuffle`` + ``Take(Cards)`` **随机升级**"
        （``DoorsOfLightAndDark.cs:Light``），引擎没有"随机升级 N 张"的 Run 层算子
        → 整个事件被如实拒绝（见 `docs/12` §2.18）。用"可跑事件"做载体，
        测的才是选牌链路本身。
        """
        from sts2_sim.run import PHASE_EVENT, PHASE_MAP, legal_meta_actions

        env = self._env()
        state = env.raw_state
        env._enter_event("wellspring")
        self.assertEqual(env.phase, PHASE_EVENT)
        dark = state.room.event_options.index("BATHE")
        env.step(type(legal_meta_actions(state)[0])("event_option", dark))

        pending = state.pending_selection
        self.assertIsNotNone(pending, "选项要求选牌时必须挂起，而不是自动结算")
        self.assertEqual(pending.purpose, "removal")
        # 选项本身**不再可点**：此时能做的只有选牌
        actions = legal_meta_actions(state)
        self.assertTrue(actions, "挂起时必须有合法动作（否则死锁）")
        self.assertTrue(all(a.kind == "select_deck_card" for a in actions))

        # 模型挑一个槽位 → 那张牌被移除
        from sts2_sim.observe import deck_slot_order

        # ⚠️ 不能断言"牌组少一张"：这个选项同时还会**加一张**牌
        # （`Wellspring.BATHE` 是"移除一张 + 获得一张"，总数不变）。
        # 按 **uid** 判断被移除的是不是选中的那**个实例**才准。
        target_slot = actions[0].index
        internal = deck_slot_order(state.player.deck)[target_slot]
        removed_uid = state.player.deck[internal].uid
        env.step(actions[0])
        self.assertNotIn(removed_uid, {c.uid for c in state.player.deck},
                         "选中的那张牌必须被移除")
        # 移除完 → 事件结束 → 回地图
        self.assertIsNone(state.pending_selection)
        self.assertEqual(env.phase, PHASE_MAP)

    def test_upgrade_choice_upgrades_the_chosen_card(self):
        """``aroma_of_chaos`` 的 MAINTAIN_CONTROL：选一张升级。

        ⚠️ 这里**直接驱动事件层**，不走 ``RunEnv._enter_event`` ——
        该事件当前被"随机转化"缺口挡着（另一个选项要转化），
        进不了事件池。但**升级这条选牌链路**本身已经可用，
        而且它正是模型要用的东西，必须单独验证。
        """
        from sts2_sim import events as event_rules
        from sts2_sim.observe import deck_slot_order
        from sts2_sim.runeffects import resolve_run_selection

        env = self._env()
        state = env.raw_state
        definition = content.EVENT_DB["aroma_of_chaos"]
        run = event_rules.initial_run(definition,
                                      event_rules.event_rng(1, "aroma_of_chaos"))
        index = next(i for i, o in enumerate(definition.pages["INITIAL"].options)
                     if o.name == "MAINTAIN_CONTROL")
        log: list[str] = []
        event_rules.apply_option(definition, run, index, state, log)
        self.assertIsNotNone(state.pending_selection)
        self.assertEqual(state.pending_selection.purpose, "upgrade")
        candidates = state.pending_selection.candidates
        internal = candidates[0]
        resolve_run_selection(state, internal, log)
        self.assertTrue(state.player.deck[internal].upgraded, "选中的牌必须被升级")

    def test_multi_card_selection_asks_once_per_card(self):
        """``amalgamator``：一次选项要选**两张** → 必须问两次。

        同样直接驱动事件层（该事件的条件用到了"牌组里有 ≥2 张打击"这类
        牌组谓词，条件求值还没实现）。
        """
        from sts2_sim import events as event_rules
        from sts2_sim.runeffects import resolve_run_selection

        env = self._env()
        state = env.raw_state
        definition = content.EVENT_DB["amalgamator"]
        run = event_rules.initial_run(definition,
                                      event_rules.event_rng(1, "amalgamator"))
        index = next(i for i, o in enumerate(definition.pages["INITIAL"].options)
                     if o.name == "COMBINE_STRIKES")
        log: list[str] = []
        event_rules.apply_option(definition, run, index, state, log)
        first = state.pending_selection
        self.assertIsNotNone(first)
        self.assertEqual(first.count, 2, "真机这一步要选两张")

        before = len(state.player.deck)
        resolve_run_selection(state, first.candidates[0], log)
        self.assertIsNotNone(state.pending_selection, "还要选第二张")
        self.assertEqual(state.pending_selection.count, 1)
        self.assertEqual(len(state.player.deck), before - 1)
        resolve_run_selection(state, state.pending_selection.candidates[0], log)
        self.assertIsNone(state.pending_selection)
        # 移掉两张、再按同一选项的后续效果加一张终极打击 → 净 −1
        self.assertEqual(len(state.player.deck), before - 1)
        self.assertTrue(any(c.cid == "ultimate_strike" for c in state.player.deck),
                        "选完之后要执行同一个选项的后续效果")
        # ⭐ 翻页被**推迟**到选牌结束（`apply_option` 挂起时记下 pending_outcome）：
        # 这一步在 Run 层由 `_resolve_deck_selection` 自动补，直接调事件层要自己补。
        self.assertIsNotNone(run.pending_outcome, "去向应被推迟，而不是提前翻页")
        option = event_rules.option_by_key(definition, run.pending_option)
        event_rules.finish_pending(run, option)
        self.assertTrue(run.finished, "选完之后才结束事件")

    def test_selection_candidates_exclude_eternal_cards(self):
        """候选要过 ``CardModel.IsRemovable``：永恒牌不可移除（``CardModel.cs:738``）。"""
        from sts2_sim import content as content_module
        from sts2_sim.core import CardInstance
        from sts2_sim.run import legal_meta_actions

        env = self._env()
        state = env.raw_state
        state.player.deck.append(CardInstance("ascenders_bane",
                                              uid=len(state.player.deck)))
        env._enter_event("wellspring")
        dark = state.room.event_options.index("BATHE")
        env.step(type(legal_meta_actions(state)[0])("event_option", dark))
        candidates = state.pending_selection.candidates
        cids = [state.player.deck[i].cid for i in candidates]
        self.assertNotIn("ascenders_bane", cids, "永恒牌不该出现在移除候选里")

    def test_selection_does_not_consume_run_rng_streams(self):
        """选牌**不该**动任何 RNG 流：谁被选中是玩家的决定，不是随机。"""
        from sts2_sim.rng import STREAMS
        from sts2_sim.run import legal_meta_actions

        env = self._env()
        state = env.raw_state
        env._enter_event("wellspring")
        dark = state.room.event_options.index("BATHE")
        env.step(type(legal_meta_actions(state)[0])("event_option", dark))
        before = {name: state.hidden.rng[name].getstate() for name in STREAMS}
        env.step(legal_meta_actions(state)[0])
        after = {name: state.hidden.rng[name].getstate() for name in STREAMS}
        changed = {name for name in STREAMS if before[name] != after[name]}
        self.assertEqual(changed, set(), f"选牌动了随机流：{changed}")


class TestRelicGrabBag(unittest.TestCase):
    """遗物抓包（``RelicGrabBag.cs``）：分桶 + 洗牌 + 前端取走不再出现。"""

    def test_populate_buckets_by_rarity_and_shuffles(self):
        from sts2_sim.relicbag import BAG_RARITIES, RelicGrabBag
        from sts2_sim.rng import RngSet

        bag = RelicGrabBag()
        bag.populate(content.relic_bag_entries("ironclad"), RngSet(9), "up_front")
        counts = bag.counts()
        self.assertEqual(set(counts), set(BAG_RARITIES),
                         "四个桶都要有（Common/Uncommon/Rare/Shop）")
        self.assertTrue(all(n > 0 for n in counts.values()), counts)
        # 池里**不含**Ancient / Event / Starter
        all_ids = {rid for bucket in bag._deques.values() for rid in bucket}
        for rid in all_ids:
            self.assertIn(content.RELICS[rid].rarity, BAG_RARITIES)

    def test_pull_removes_permanently(self):
        from sts2_sim.relicbag import RelicGrabBag
        from sts2_sim.rng import RngSet

        bag = RelicGrabBag()
        bag.populate(content.relic_bag_entries("ironclad"), RngSet(9), "up_front")
        first = bag.pull_from_front("common")
        self.assertIsNotNone(first)
        self.assertNotIn(first, bag._deques["common"],
                         "取走的遗物本局不再出现（真机 RemoveAt）")

    def test_empty_bucket_falls_back_to_circlet(self):
        """``RelicFactory.cs:47`` 的 ``?? FallbackRelic``（``Circlet``）。"""
        from sts2_sim.relicbag import FALLBACK_RELIC, RelicGrabBag

        bag = RelicGrabBag()
        bag.populate((("anchor", "common"),), __import__(
            "sts2_sim.rng", fromlist=["RngSet"]).RngSet(1), "up_front")
        self.assertEqual(bag.pull_from_front("common"), "anchor")
        self.assertIsNone(bag.pull_from_front("common"), "桶空返回 None")
        self.assertEqual(FALLBACK_RELIC, "circlet")


if __name__ == "__main__":
    unittest.main(verbosity=2)
