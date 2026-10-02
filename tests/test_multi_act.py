"""多幕（Act）相关测试：幕表、点类型配额、未知点掷骰、三幕流程。

覆盖的源码事实（``data/decompiled/sts2/``）：

| 事实 | 出处 | 用例 |
|---|---|---|
| 一局 3 个幕索引、共 4 张幕地图（密林/暗港二选一 → 蜂巢 → 荣耀） | ``ActModel.cs:528-596``、``ModelDb.cs:299-321`` | `test_acts_by_index_and_default_list` |
| 每张幕的房间数 / 弱遭遇数 | 各 ``Act.cs`` | `test_act_params_match_source` |
| 点类型配额公式（精英 5→8、商店 3、营火/未知按幕） | ``MapPointTypeCounts.cs:12-46`` | `test_elite_quota_follows_ascension`（在 test_run） |
| 未知点概率**跨房间累计**、精英基础概率为负 | ``UnknownMapPointOdds.cs:29-47,159-175`` | `test_unknown_odds_accumulate_and_reset` |
| 商店拉黑规则 | ``RunManager.cs:660-668`` | `test_shop_blacklist` |
| 全新存档的前两个未知点是事件、第三个是怪 | ``UnknownMapPointOdds.cs:125-136`` | `test_first_run_tutorial_rule` |
| Boss 倒下进下一幕、末幕才是胜利 | ``RunManager``（``docs/12`` F08） | `test_boss_defeat_advances_the_act` |
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from sts2_sim import Action, MetaAction, RunEnv
from sts2_sim.acts import (
    ACT_FALLBACK, ActDef, act_def, acts_by_index, default_act_list,
    random_act_list,
)
from sts2_sim.mapgen import BOSS, REST_SITE, SHOP, TREASURE, generate_map
from sts2_sim.pointodds import (
    BASE_ODDS, ELITE, EVENT, MONSTER, SHOP as ODDS_SHOP, TREASURE as ODDS_TREASURE,
    UnknownMapPointOdds, build_blacklist,
)
from sts2_sim.rng import Rng

REPO_SOURCE = Path("data/content/repo/acts_source.json")


# ==========================================================================
# 幕表
# ==========================================================================
class TestActTable(unittest.TestCase):
    def test_four_maps_three_indices(self):
        """4 张幕地图、3 个幕索引：索引 0 有两张（密林/暗港）。"""
        table = acts_by_index()
        self.assertEqual(sorted(table), [0, 1, 2])
        self.assertEqual({d.act_id for d in table[0]}, {"overgrowth", "underdocks"})
        self.assertEqual([d.act_id for d in table[1]], ["hive"])
        self.assertEqual([d.act_id for d in table[2]], ["glory"])

    def test_acts_by_index_and_default_list(self):
        """``GetDefaultList`` = 每个索引取 ``IsDefault`` 的那张。"""
        self.assertEqual([d.act_id for d in default_act_list()],
                         ["overgrowth", "hive", "glory"])

    def test_act_params_match_source(self):
        """房间数 / 弱遭遇数 / 索引 / 默认标记 / 解锁条件，逐个对源码。"""
        expected = {
            "overgrowth": (0, True, 15, 3, None),
            "underdocks": (0, False, 15, 3, "UnderdocksEpoch"),
            "hive": (1, True, 14, 2, None),
            "glory": (2, True, 13, 2, None),
        }
        for act_id, (index, is_default, rooms, weak, epoch) in expected.items():
            definition = act_def(act_id)
            self.assertEqual(definition.index, index, act_id)
            self.assertEqual(definition.is_default, is_default, act_id)
            self.assertEqual(definition.base_number_of_rooms, rooms, act_id)
            self.assertEqual(definition.number_of_weak_encounters, weak, act_id)
            self.assertEqual(definition.unlock_epoch, epoch, act_id)

    def test_number_of_floors_includes_boss_and_ancient(self):
        """``GetNumberOfFloors = GetNumberOfRooms + 2``（``ActModel.cs:276-279``）。"""
        for definition in ACT_FALLBACK.values():
            self.assertEqual(definition.number_of_floors(),
                             definition.number_of_rooms() + 2)
        # 联机少一个房间（``isMultiplayer`` 时 ``num--``）
        self.assertEqual(act_def("hive").number_of_rooms(is_multiplayer=True), 13)
        self.assertEqual(generate_map(1, "hive", is_multiplayer=True).rows, 15)

    def test_rest_spec_matches_source(self):
        """营火配额：密林/暗港/蜂巢是高斯，荣耀是 ``NextInt(5, 7)``。"""
        self.assertEqual(act_def("overgrowth").rest_spec, ("gaussian_int", 7, 1, 6, 7))
        self.assertEqual(act_def("underdocks").rest_spec, ("gaussian_int", 7, 1, 6, 7))
        self.assertEqual(act_def("hive").rest_spec, ("gaussian_int", 6, 1, 6, 7))
        self.assertEqual(act_def("glory").rest_spec, ("next_int", 5, 7))

    def test_unknown_offset_matches_source(self):
        """蜂巢/荣耀的未知点配额比标准值**少 1**（``GetMapPointTypes`` 里 ``- 1``）。"""
        self.assertEqual(act_def("overgrowth").unknown_offset, 0)
        self.assertEqual(act_def("underdocks").unknown_offset, 0)
        self.assertEqual(act_def("hive").unknown_offset, -1)
        self.assertEqual(act_def("glory").unknown_offset, -1)

    def test_fallback_matches_extracted_source(self):
        """兜底表与 ``acts_source.json`` 必须一致 —— 防两处数值各自漂移。"""
        if not REPO_SOURCE.exists():
            self.skipTest("acts_source.json 还没生成")
        records = {r["id"]: r for r in json.loads(REPO_SOURCE.read_text(encoding="utf-8"))}
        self.assertEqual(set(records), set(ACT_FALLBACK))
        for act_id, record in records.items():
            fallback = ACT_FALLBACK[act_id]
            self.assertEqual(fallback.index, record["index"], act_id)
            self.assertEqual(fallback.is_default, record["is_default"], act_id)
            self.assertEqual(fallback.base_number_of_rooms,
                             record["base_number_of_rooms"], act_id)
            self.assertEqual(fallback.number_of_weak_encounters,
                             record["number_of_weak_encounters"], act_id)
            unlocked = record["is_unlocked"]
            self.assertEqual(fallback.unlock_epoch,
                             None if unlocked is True else unlocked, act_id)
            rest = record["map_point_counts"]["rest"]
            if rest["kind"] == "next_int":
                self.assertEqual(fallback.rest_spec,
                                 ("next_int", rest["min"], rest["max"]), act_id)
            else:
                self.assertEqual(
                    fallback.rest_spec,
                    ("gaussian_int", rest["mean"], rest["stddev"],
                     rest["min"], rest["max"]), act_id)
            self.assertEqual(fallback.unknown_offset,
                             record["map_point_counts"]["unknown"]["offset"], act_id)

    def test_random_act_list_respects_unlock(self):
        """暗港没解锁时索引 0 只能是密林。"""
        acts = random_act_list(Rng(1, "act_selection"), unlocked_epochs=frozenset(),
                               discovered=frozenset({"underdocks"}))
        self.assertEqual(acts[0].act_id, "overgrowth")

    def test_random_act_list_forces_undiscovered_alt_act(self):
        """没见过的非默认幕**必定**被选中（真机 ``GetRandomList`` 的保底）。"""
        acts = random_act_list(Rng(1, "act_selection"),
                               unlocked_epochs=frozenset({"UnderdocksEpoch"}),
                               discovered=frozenset())
        self.assertEqual(acts[0].act_id, "underdocks")

    def test_random_act_list_varies_across_seeds(self):
        """已解锁且见过之后，索引 0 在密林/暗港之间随机。"""
        picked = set()
        for seed in range(40):
            acts = random_act_list(Rng(seed, "act_selection"),
                                   unlocked_epochs=frozenset({"UnderdocksEpoch"}),
                                   discovered=frozenset({"underdocks"}))
            picked.add(acts[0].act_id)
        self.assertEqual(picked, {"overgrowth", "underdocks"},
                         "第一幕应该在密林/暗港之间随机")


# ==========================================================================
# 未知点掷骰
# ==========================================================================
class TestUnknownMapPointOdds(unittest.TestCase):
    def _odds(self, seed: int = 0, number_of_runs: int = 1) -> UnknownMapPointOdds:
        return UnknownMapPointOdds(Rng(seed, "unknown_map_point"),
                                   number_of_runs=number_of_runs)

    def test_elite_is_never_rolled_at_base_odds(self):
        """精英基础概率是 ``-1`` = 负数 = **永不出现**（``UnknownMapPointOdds.cs:23``）。"""
        odds = self._odds(3)
        for _ in range(500):
            self.assertNotEqual(odds.roll(), ELITE)

    def test_rolls_only_known_room_types(self):
        odds = self._odds(5)
        for _ in range(300):
            self.assertIn(odds.roll(), {EVENT, MONSTER, ODDS_TREASURE, ODDS_SHOP})

    def test_odds_accumulate_then_reset_on_hit(self):
        """没掷中的类型**加上**自己的基础值；掷中的回落基础值。"""
        odds = self._odds(7)
        before = odds.snapshot()
        kind = odds.roll()
        after = odds.snapshot()
        for name, base in BASE_ODDS:
            if name == kind:
                self.assertEqual(after[name], base, f"{name} 掷中后应回落")
            else:
                self.assertAlmostEqual(after[name], before[name] + base, places=9,
                                       msg=f"{name} 没掷中应累计")

    def test_event_odds_grow_without_events(self):
        odds = self._odds(11)
        start = odds.event_odds
        for _ in range(6):
            if odds.roll() == EVENT:
                return                       # 掷出事件就回落了，换个断言路径
        self.assertGreater(odds.event_odds, start,
                           "一直不掷出事件，事件概率必须上涨")

    def test_reset_to_base(self):
        """``ResetToBase`` 在**幕与幕之间**调用（``UnknownMapPointOdds.cs:183``）。"""
        odds = self._odds(13)
        for _ in range(5):
            odds.roll()
        odds.reset_to_base()
        self.assertEqual(odds.snapshot(), dict(BASE_ODDS))

    def test_shop_blacklist(self):
        self.assertEqual(build_blacklist((ODDS_SHOP,), ()), frozenset({ODDS_SHOP}))
        self.assertEqual(build_blacklist((), (ODDS_SHOP, ODDS_SHOP)),
                         frozenset({ODDS_SHOP}))
        self.assertEqual(build_blacklist((MONSTER,), (ODDS_SHOP, MONSTER)),
                         frozenset())
        # 拉黑商店时不会掷出商店
        odds = self._odds(17)
        for _ in range(200):
            self.assertNotEqual(odds.roll(frozenset({ODDS_SHOP})), ODDS_SHOP)

    def test_first_run_tutorial_rule(self):
        """全新存档：第 1、2 个未知点是事件，第 3 个是怪（源码硬规则）。"""
        odds = self._odds(19, number_of_runs=0)
        self.assertEqual(odds.roll(unknown_visited=0), EVENT)
        self.assertEqual(odds.roll(unknown_visited=1), EVENT)
        self.assertEqual(odds.roll(unknown_visited=2), MONSTER)
        self.assertEqual(self._odds(19, number_of_runs=1).snapshot(),
                         dict(BASE_ODDS), "普通存档不该吃到教学规则")

    def test_snapshot_restore_roundtrip(self):
        odds = self._odds(23)
        for _ in range(4):
            odds.roll()
        saved = odds.snapshot()
        later = self._odds(23)
        later.restore(saved)
        self.assertEqual(later.snapshot(), saved)


# ==========================================================================
# 三幕流程
# ==========================================================================
class _WonResult:
    done = True
    won = True


class TestMultiActRun(unittest.TestCase):
    def test_run_starts_with_three_acts(self):
        env = RunEnv(seed=20260919)
        env.reset()
        state = env.raw_state
        self.assertEqual(len(state.acts), 3)
        self.assertEqual([a.act_id for a in state.acts][1:], ["hive", "glory"])
        self.assertEqual(state.act, 1)
        self.assertEqual(state.map.act_id,
                         state.acts[0].act_id)
        self.assertEqual(state.map.act_index, 0)

    def test_begin_act_swaps_map_and_resets_odds(self):
        env = RunEnv(seed=7)
        env.reset()
        state = env.raw_state
        # 先把未知点概率搅乱，验证换幕会重置（``ResetToBase``）
        state.hidden.point_odds.roll()
        dirty = state.hidden.point_odds.snapshot()
        env._begin_act(2)
        self.assertEqual(state.map.act_id, "hive")
        self.assertEqual(state.map.act_index, 1)
        self.assertEqual(state.map.rows, 16)          # 14 房 + Boss + 远古
        self.assertIsNone(state.position)
        self.assertEqual(state.visited, [])
        self.assertNotEqual(state.hidden.point_odds.snapshot(), dirty)
        env._begin_act(3)
        self.assertEqual(state.map.act_id, "glory")
        self.assertEqual(state.map.rows, 15)          # 13 房 + Boss + 远古

    def test_boss_defeat_advances_the_act(self):
        """Boss 倒下 → **先给奖励**（非最终幕）→ 领完进下一幕；末幕才是胜利。

        审计 F08：以前非最终幕 Boss 直接换幕，把 Boss 的三选一与金币吞掉了
        （源码 ``RewardsSet.cs:88`` 只对**最终幕** Boss 省略常规奖励）。
        """
        import sts2_sim.run as run_module

        env = RunEnv(seed=3)
        env.reset()
        state = env.raw_state
        original = run_module.combat_step
        run_module.combat_step = lambda combat, action: _WonResult()
        try:
            for expected_act, expected_map in ((2, "hive"), (3, "glory")):
                env._enter_node(state.map.boss)
                self.assertEqual(state.room.kind, "combat")
                step, _reward, done, _trunc, _info = env.step(Action("end_turn"))
                self.assertFalse(done, "还没到末幕，不该结束")
                self.assertEqual(state.room.kind, "card_reward",
                                 "非最终幕 Boss 必须给卡牌奖励")
                self.assertTrue(state.pending_act_advance)
                self.assertEqual(state.act, expected_act - 1, "奖励阶段还没换幕")
                step, _r, _d, _t, _i = env.step(MetaAction("skip_card"))
                self.assertEqual(state.act, expected_act)
                self.assertEqual(state.map.act_id, expected_map)
                self.assertFalse(state.pending_act_advance)
                self.assertEqual(state.room.kind, "map")
            env._enter_node(state.map.boss)
            step, reward, done, _trunc, _info = env.step(Action("end_turn"))
        finally:
            run_module.combat_step = original
        self.assertTrue(done)
        self.assertEqual(step.phase, "won")
        self.assertEqual(reward, 1.0)

    def test_boss_card_reward_is_rare(self):
        """Boss 奖励必定稀有（``bossRareOdds = 1.0``），且真的进了牌组。"""
        import sts2_sim.run as run_module
        from sts2_sim.content import CARD_DB, CONTENT_SOURCE

        if CONTENT_SOURCE == "builtin":
            self.skipTest("内置占位内容没有稀有度表")
        env = RunEnv(seed=3)
        env.reset()
        state = env.raw_state
        original = run_module.combat_step
        run_module.combat_step = lambda combat, action: _WonResult()
        try:
            env._enter_node(state.map.boss)
            env.step(Action("end_turn"))
            reward = state.room.card_reward
            picked = reward[0]
            env.step(MetaAction("pick_card", 0))
        finally:
            run_module.combat_step = original
        self.assertTrue(reward)
        self.assertEqual(CARD_DB[picked].rarity, "rare")
        self.assertIn(picked, [card.cid for card in state.player.deck])

    def test_double_boss_fights_the_second_boss_first(self):
        """A10：第一个 Boss 倒下后打**第二个** Boss，不换幕（``SecondBossMapPoint``）。"""
        import sts2_sim.run as run_module

        env = RunEnv(seed=3, ascension=10)
        env.reset()
        state = env.raw_state
        self.assertIsNotNone(state.map.second_boss)
        second = state.map.second_boss
        original = run_module.combat_step
        run_module.combat_step = lambda combat, action: _WonResult()
        try:
            env._enter_node(state.map.boss)
            env.step(Action("end_turn"))
            self.assertEqual(state.act, 1, "打完第一个 Boss 不该换幕")
            self.assertEqual(state.position, second)
            self.assertEqual(state.room.kind, "combat")
            env.step(Action("end_turn"))
            self.assertEqual(state.room.kind, "card_reward")
            self.assertTrue(state.pending_act_advance)
            env.step(MetaAction("skip_card"))
            self.assertEqual(state.act, 2)
        finally:
            run_module.combat_step = original

    def test_forced_first_act(self):
        env = RunEnv(seed=5, first_act="underdocks")
        env.reset()
        self.assertEqual(env.raw_state.acts[0].act_id, "underdocks")
        self.assertEqual(env.raw_state.map.act_id, "underdocks")

    def test_act_scoped_map_streams(self):
        """三幕各自的地图流必须不同 —— 否则三张图会长得一样。"""
        env = RunEnv(seed=11)
        env.reset()
        state = env.raw_state
        layouts = {state.act: tuple((n.row, n.col, n.kind) for n in state.map.nodes)}
        for act in (2, 3):
            env._begin_act(act)
            layouts[act] = tuple((n.row, n.col, n.kind) for n in state.map.nodes)
        self.assertEqual(len(set(layouts.values())), 3,
                         "三幕地图布局不应完全相同")


# ==========================================================================
# RNG 门面（``rng.Rng``）
# ==========================================================================
class TestRngFacade(unittest.TestCase):
    def test_next_int_upper_bound_is_exclusive(self):
        """``NextInt(minInclusive, maxExclusive)``：``NextInt(5, 7)`` 只出 5/6。"""
        rng = Rng(1, "test")
        values = {rng.next_int(5, 7) for _ in range(200)}
        self.assertTrue(values <= {5, 6}, values)

    def test_next_int_single_argument_is_max_exclusive(self):
        rng = Rng(2, "test")
        values = {rng.next_int(3) for _ in range(200)}
        self.assertTrue(values <= {0, 1, 2}, values)

    def test_gaussian_int_within_bounds(self):
        rng = Rng(3, "test")
        for _ in range(200):
            self.assertIn(rng.next_gaussian_int(7, 1, 6, 7), (6, 7))

    def test_stable_shuffle_is_order_independent(self):
        """``StableShuffle`` 先排序再洗：输入顺序不同，结果相同。"""
        a = Rng(4, "test").stable_shuffle([3, 1, 2, 5, 4])
        b = Rng(4, "test").stable_shuffle([5, 4, 3, 2, 1])
        self.assertEqual(a, b)

    def test_unstable_shuffle_depends_on_order(self):
        rng_a = Rng(4, "test")
        rng_b = Rng(4, "test")
        self.assertNotEqual(rng_a.shuffle([1, 2, 3, 4, 5]),
                            rng_b.shuffle([5, 4, 3, 2, 1]))

    def test_named_streams_are_deterministic_and_distinct(self):
        first = [Rng(9, "act_1_map").next_int(100) for _ in range(5)]
        self.assertEqual(first, [Rng(9, "act_1_map").next_int(100) for _ in range(5)])
        self.assertNotEqual(first, [Rng(9, "act_2_map").next_int(100) for _ in range(5)])


if __name__ == "__main__":
    unittest.main()
