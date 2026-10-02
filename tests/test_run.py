"""Run 层测试：地图、房间流转、RNG 分流纪律、Run 层反作弊、基线强度。"""

from __future__ import annotations

import unittest

from sts2_sim import (
    Action, MetaAction, RunEnv, RunObservation, STREAMS, RngSet,
    generate_map, observe_run,
)
from sts2_sim.content import ENCOUNTERS
from sts2_sim.mapgen import (
    ANCIENT, BOSS, ELITE, GRID_COLUMNS, MONSTER, REST_SITE, SHOP, TREASURE,
    UNKNOWN, clear_map_cache, generate_map,
)
from sts2_sim.run import (
    NodeContents, RunHidden, draw_card_reward, draw_encounter, draw_event,
    draw_gold, draw_relic, draw_shop_stock, legal_meta_actions,
    reachable_nodes,
)
from sts2_sim.runbot import RunBot


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------
def combat_fingerprint(combat) -> tuple:
    return (
        combat.turn, combat.energy, combat.player.hp, combat.player.block,
        tuple(sorted(combat.player.powers.items())),
        tuple(c.eid for c in combat.enemies),
        tuple((e.hp, e.block) for e in combat.enemies),
        tuple(c.cid for c in combat.hand),
        tuple(c.cid for c in combat.draw_pile),
        tuple(c.cid for c in combat.discard),
    )


def scripted_combat_action(step):
    for action in step.legal:
        if action.kind == "play_card" and action.hand_index == 0:
            return action
    return Action("end_turn")


def play_run(bot: RunBot, env: RunEnv, max_steps: int = 3000):
    """让 bot 打完一局，返回 (结局, 到达层数, 步数)。"""
    step = env.reset()
    steps = 0
    while step.phase not in ("won", "lost") and steps < max_steps:
        action = bot.act(step)
        if action is None:
            break
        step, _reward, done, _trunc, _info = env.step(action)
        steps += 1
        if done:
            break
    return step.phase, env.raw_state.floor, steps


# ==========================================================================
# 地图（忠实复刻 ``StandardActMap``：三幕索引 / 四张幕地图）
# ==========================================================================
ALL_ACTS = ("overgrowth", "underdocks", "hive", "glory")


class TestMapGeneration(unittest.TestCase):
    def test_single_boss_row_at_top(self):
        for act in ALL_ACTS:
            graph = generate_map(seed=1, act=act)
            boss = graph.by_id(graph.boss)
            self.assertEqual(boss.kind, BOSS)
            self.assertEqual(boss.row, graph.rows - 1)
            self.assertEqual(boss.edges, ())

    def test_ancient_is_the_only_start_point(self):
        """真机的起点是**一个** ``MapPointType.Ancient``（``(3, 0)``），不是一排怪。"""
        for act in ALL_ACTS:
            graph = generate_map(seed=4, act=act)
            ancient = graph.by_id(graph.ancient)
            self.assertEqual(ancient.kind, ANCIENT)
            self.assertEqual(ancient.row, 0)
            self.assertEqual(len(graph.row_nodes(0)), 1)
            # 远古点的子节点 = 第一行全部节点
            self.assertEqual(set(ancient.edges), set(graph.start))
            self.assertEqual(set(graph.row_nodes(1)), set(graph.start))

    def test_first_row_is_all_monster(self):
        for act in ALL_ACTS:
            graph = generate_map(seed=6, act=act)
            for node_id in graph.start:
                self.assertEqual(graph.by_id(node_id).kind, MONSTER)

    def test_fixed_treasure_and_rest_rows(self):
        """宝箱行与营火行是**固定行**（``AssignPointTypes``）：整数行必须先算对。"""
        for act, rooms in (("overgrowth", 15), ("underdocks", 15),
                           ("hive", 14), ("glory", 13)):
            graph = generate_map(seed=8, act=act)
            self.assertEqual(graph.rows, rooms + 2, "层数 = 房间数 + Boss + 远古")
            treasure_row = [n for n in graph.nodes if n.row == rooms - 6]
            rest_row = [n for n in graph.nodes if n.row == rooms]
            self.assertTrue(treasure_row, f"{act} 没有宝箱行")
            self.assertTrue(rest_row, f"{act} 没有营火行")
            for node in treasure_row:
                self.assertEqual(node.kind, TREASURE)
            for node in rest_row:
                self.assertEqual(node.kind, REST_SITE)

    def test_lower_half_has_no_elite_or_rest(self):
        """``_lowerMapPointRestrictions``：前 6 行不得出现精英与营火。"""
        for act in ALL_ACTS:
            for seed in range(6):
                graph = generate_map(seed=seed, act=act)
                for node in graph.nodes:
                    if node.row < 6:
                        self.assertNotIn(node.kind, (ELITE, REST_SITE),
                                         f"{act} seed={seed} 第 {node.row} 行出现了 {node.kind}")

    def test_shop_quota_is_three(self):
        """``MapPointTypeCounts.NumOfShops = 3``（源码常量）。"""
        for act in ALL_ACTS:
            for seed in range(6):
                graph = generate_map(seed=seed, act=act)
                shops = [n for n in graph.nodes if n.kind == SHOP]
                self.assertEqual(len(shops), 3, f"{act} seed={seed} 商店数不对")

    def test_elite_quota_follows_ascension(self):
        """``round(5 * (SwarmingElites(A1 起) ? 1.6 : 1))`` → A0 是 5、A1 起是 8。"""
        for act in ALL_ACTS:
            base = [n for n in generate_map(11, act).nodes if n.kind == ELITE]
            swarm = [n for n in generate_map(11, act, ascension=1).nodes
                     if n.kind == ELITE]
            self.assertEqual(len(base), 5, f"{act} A0 精英数")
            self.assertEqual(len(swarm), 8, f"{act} A1 精英数")

    def test_double_boss_adds_a_boss_layer(self):
        """A10 ``DoubleBoss``：第二个 Boss 层在最后（``SecondBossMapPoint``）。"""
        graph = generate_map(2, "hive", ascension=10)
        self.assertIsNotNone(graph.second_boss)
        second = graph.by_id(graph.second_boss)
        self.assertEqual(second.kind, BOSS)
        self.assertEqual(second.row, graph.rows)
        self.assertIn(graph.second_boss, graph.by_id(graph.boss).edges)
        self.assertIsNone(generate_map(2, "hive", ascension=9).second_boss)

    def test_every_node_is_reachable_from_the_ancient(self):
        """地图必须连通：剪枝**不能**剪出孤立点，否则 run 会卡死。"""
        for act in ALL_ACTS:
            for seed in range(8):
                graph = generate_map(seed=seed, act=act)
                seen = {graph.ancient}
                frontier = [graph.ancient]
                while frontier:
                    node = graph.by_id(frontier.pop())
                    for child in node.edges:
                        if child not in seen:
                            seen.add(child)
                            frontier.append(child)
                self.assertEqual(len(seen), len(graph),
                                 f"{act} seed={seed} 有不可达点")
                self.assertIn(graph.boss, seen)

    def test_edges_only_go_one_row_down(self):
        for act in ALL_ACTS:
            graph = generate_map(seed=13, act=act)
            for node in graph.nodes:
                for child in node.edges:
                    self.assertEqual(graph.by_id(child).row, node.row + 1)
                self.assertEqual(len(set(node.edges)), len(node.edges))

    def test_generation_is_deterministic(self):
        clear_map_cache()
        a = generate_map(seed=5, act="glory")
        clear_map_cache()
        b = generate_map(seed=5, act="glory")
        self.assertEqual([(n.row, n.col, n.kind, n.edges) for n in a.nodes],
                         [(n.row, n.col, n.kind, n.edges) for n in b.nodes])

    def test_map_cache_returns_the_same_graph(self):
        clear_map_cache()
        a = generate_map(seed=5, act="hive")
        b = generate_map(seed=5, act="hive")
        self.assertIs(a, b, "同 (seed, 幕, 进阶) 必须命中缓存（生成是纯函数）")
        clear_map_cache()

    def test_columns_within_bounds(self):
        graph = generate_map(seed=3)
        for node in graph.nodes:
            self.assertTrue(0 <= node.col < GRID_COLUMNS)

    def test_act_one_variants_share_geometry(self):
        """密林与暗港的**地图参数完全相同**（都是 ``Index 0``、15 房、同一套配额公式）。

        所以同 seed 下两张图结构一致 —— 差别在**内容**（遭遇/事件池），不在布局。
        这是从源码推出来的结论：``StandardActMap.CreateFor`` 只把 ``ActModel``
        传给 ``GetNumberOfRooms`` / ``GetMapPointTypes``，两者的实现逐字相同。
        """
        clear_map_cache()
        a = generate_map(seed=21, act="overgrowth")
        b = generate_map(seed=21, act="underdocks")
        self.assertEqual([(n.row, n.col, n.kind, n.edges) for n in a.nodes],
                         [(n.row, n.col, n.kind, n.edges) for n in b.nodes])

    def test_four_act_maps_have_distinct_layouts(self):
        clear_map_cache()
        shapes = {act: tuple((n.row, n.col, n.kind) for n in generate_map(9, act).nodes)
                  for act in ALL_ACTS}
        self.assertNotEqual(shapes["overgrowth"], shapes["hive"])
        self.assertNotEqual(shapes["hive"], shapes["glory"])
        self.assertNotEqual(shapes["overgrowth"], shapes["glory"])


# ==========================================================================
# RNG 分流纪律（docs/02 §2.4）
# ==========================================================================
class TestRngStreamIsolation(unittest.TestCase):
    """**每条抽取只允许消费它自己那条流。**

    分流错了会导致分布漂移：真机上"洗牌"与"掉落"是独立的，合成一条会让
    "抽到好牌"与"下一场遇到弱怪"产生真机不存在的相关性，模型会学到虚假规律。
    """

    def _assert_only(self, expected: str, fn) -> None:
        rng = RngSet(20240501)
        before = {name: rng[name].getstate() for name in STREAMS}
        fn(rng)
        after = {name: rng[name].getstate() for name in STREAMS}
        changed = {name for name in STREAMS if before[name] != after[name]}
        self.assertEqual(changed, {expected},
                         f"期望只消费 {expected!r} 流，实际动了 {changed}")

    def test_card_reward_uses_only_card_stream(self):
        """卡牌奖励只消费 ``rewards`` 流。

        ⚠️ 这条测试原先断言的是 ``up_front``，依据是"真机没有独立的卡牌奖励流"
        —— **那个结论是查漏了**。真机有 ``PlayerRngType { Rewards, Shops,
        Transformations }`` 三条**玩家级**流（``rng.STREAMS`` 的注释里更正了）。
        借错流会让奖励与下场遭遇产生真机不存在的相关性。
        """
        self._assert_only("rewards", lambda rng: draw_card_reward(rng))

    def test_shop_stock_uses_only_shops_stream(self):
        """商店货架只消费 ``shops`` 流（真机 ``PlayerRng.Shops``）。"""
        self._assert_only("shops", lambda rng: draw_shop_stock(rng))

    def test_encounter_uses_only_monster_stream(self):
        self._assert_only("up_front", lambda rng: draw_encounter(rng, MONSTER))

    def test_gold_uses_only_misc_stream(self):
        self._assert_only("niche", lambda rng: draw_gold(rng))

    def test_relic_uses_only_relic_stream(self):
        self._assert_only("up_front", lambda rng: draw_relic(rng))

    def test_event_uses_only_event_stream(self):
        """事件抽取只消费 ``up_front`` 流。

        ⚠️ 这里跑的是**内置占位内容**，它没有事件池，所以 ``draw_event`` 直接
        返回 ``""``（"这个节点没有事件"）并且**不消费任何流** —— 那种情况下
        "只消费 up_front"是空真。真正的事件抽取纪律（池子来自源码事件表、
        只动 ``up_front``）由 ``tests/test_events.py`` 在**真实内容**上验证。
        """
        from sts2_sim.content import event_pool

        if not event_pool():
            rng = RngSet(20240501)
            before = {name: rng[name].getstate() for name in STREAMS}
            self.assertEqual(draw_event(rng), "")
            after = {name: rng[name].getstate() for name in STREAMS}
            self.assertEqual({n for n in STREAMS if before[n] != after[n]}, set(),
                             "没有事件池时不该消费任何流")
            return
        self._assert_only("up_front", lambda rng: draw_event(rng))

    def test_map_generation_consumes_no_run_stream(self):
        """生成地图**不允许**动 ``STREAMS`` 里任何一条。

        真机的地图走 ``new Rng(seed, "act_N_map")`` 这条**另起**的派生流
        （``StandardActMap.cs:113``）。挂到 ``up_front`` 上会让"地图长什么样"
        与"开局掷出的内容"产生真机不存在的相关性（``docs/02`` §2.4）。
        """
        rng = RngSet(31)
        before = {name: rng[name].getstate() for name in STREAMS}
        with_copies = {name: rng[name].getstate() for name in STREAMS}
        for act in ALL_ACTS:
            generate_map(31, act)
        after = {name: rng[name].getstate() for name in STREAMS}
        self.assertEqual(before, with_copies)
        self.assertEqual(before, after, "生成地图消费了 run 流")

    def test_streams_are_statistically_independent(self):
        """不同流的输出不应高度重合（派生方式必须让它们独立）。"""
        a = [RngSet(7)[name].random() for name in STREAMS]
        b = [RngSet(8)[name].random() for name in STREAMS]
        self.assertEqual(len(set(a)), len(a), "同一主种子下各流输出不应重复")
        self.assertEqual(len(set(b)), len(b))


# ==========================================================================
# Run 层反作弊
# ==========================================================================
class TestRunLayerIsolation(unittest.TestCase):
    def test_unvisited_node_contents_do_not_affect_observation(self):
        """未进入的节点内容属于 L3：怎么变都不能影响观测。"""
        env_a = RunEnv(seed=9, attempt_budget=4)
        env_b = RunEnv(seed=9, attempt_budget=4)
        env_a.reset()
        env_b.reset()

        # 只改 b 的隐藏区：所有节点内容换成别的东西，RNG 也重掷
        state_b = env_b.raw_state
        state_b.hidden = RunHidden(RngSet(4242))
        for node in state_b.map.nodes:
            state_b.hidden.node_contents[node.node_id] = NodeContents(
                encounter=("cultist", "cultist"), card_reward=("bash", "bash", "bash"),
                relic="vajra", event_id="shrine", gold=9999)

        self.assertEqual(observe_run(env_a.raw_state).to_json(),
                         observe_run(state_b).to_json())

    def test_encounter_ids_absent_from_map_phase_observation(self):
        """地图阶段不该出现任何遭遇 / 事件 / 遗物的具体内容。"""
        env = RunEnv(seed=17)
        env.reset()
        payload = observe_run(env.raw_state).to_json()
        for table in ENCOUNTERS.values():
            for encounter in table:
                for eid in encounter:
                    self.assertNotIn(eid, payload, f"地图阶段泄漏了遭遇内容 {eid!r}")
        self.assertNotIn("bonfire", payload)
        self.assertNotIn("shrine", payload)
        self.assertNotIn("vajra", payload)

    def test_seed_not_in_run_observation(self):
        env = RunEnv(seed=987_654_321)
        env.reset()
        payload = observe_run(env.raw_state).to_json()
        self.assertNotIn("987654321", payload)
        for forbidden in ("seed", "rng", "hidden", "node_contents", "master"):
            self.assertNotIn(forbidden, payload.lower())

    def test_observation_is_immutable(self):
        import dataclasses
        env = RunEnv(seed=2)
        env.reset()
        obs = observe_run(env.raw_state)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            obs.gold = 999  # type: ignore[misc]


class TestRunRestartSemantics(unittest.TestCase):
    """SL 在 Run 层的对应物：重开节点必须遇到同一场遭遇、同一副牌序。"""

    def _enter_first_monster(self, env: RunEnv):
        step = env.reset()
        node_id = step.run_obs.reachable[0]
        step, _r, _d, _t, _i = env.step(MetaAction("choose_node", node_id))
        self.assertEqual(step.phase, "combat")
        return step

    def _scripted_combat(self, env: RunEnv, step, max_steps: int = 30):
        trace = []
        for _ in range(max_steps):
            if step.phase != "combat":
                break
            trace.append(combat_fingerprint(env.raw_state.combat))
            action = scripted_combat_action(step)
            step, _r, done, _t, _i = env.step(action)
            if done:
                break
        return trace

    def test_restart_node_replays_identical_combat(self):
        env = RunEnv(seed=31, attempt_budget=8)
        step = self._enter_first_monster(env)
        first = self._scripted_combat(env, step)
        self.assertTrue(env.restart_node(), "预算充足，节点重开应当成功")
        step = env._step_view()
        second = self._scripted_combat(env, step)
        self.assertEqual(first, second, "重开后战斗轨迹必须一致（同一副牌序）")

    def test_restart_budget_is_enforced(self):
        env = RunEnv(seed=31, attempt_budget=2)
        self._enter_first_monster(env)
        self.assertTrue(env.restart_node())
        self.assertTrue(env.restart_node())
        self.assertFalse(env.restart_node(), "预算耗尽后必须拒绝重开")
        self.assertEqual(env.nodes_restarted, 2)

    def test_on_entry_mode_also_replays_identically(self):
        """``content_mode="on_entry"``（进入节点才掷）同样必须满足重开一致。"""
        env = RunEnv(seed=31, attempt_budget=4, content_mode="on_entry")
        step = self._enter_first_monster(env)
        first = self._scripted_combat(env, step)
        env.restart_node()
        second = self._scripted_combat(env, env._step_view())
        self.assertEqual(first, second)


# ==========================================================================
# 完整 run 循环
# ==========================================================================
class TestRunLoop(unittest.TestCase):
    def test_full_run_terminates(self):
        bot = RunBot()
        for seed in range(12):
            env = RunEnv(seed=seed, attempt_budget=1)
            phase, floor, steps = play_run(bot, env)
            self.assertIn(phase, ("won", "lost"))
            self.assertLess(steps, 3000, f"seed={seed} 疑似死循环")

    def test_hp_never_exceeds_max_and_never_negative(self):
        bot = RunBot()
        for seed in range(8):
            env = RunEnv(seed=seed)
            step = env.reset()
            steps = 0
            while step.phase not in ("won", "lost") and steps < 3000:
                state = env.raw_state
                self.assertTrue(0 <= state.player.hp <= state.player.max_hp)
                self.assertGreaterEqual(state.player.gold, 0)
                action = bot.act(step)
                if action is None:
                    break
                step, _r, done, _t, _i = env.step(action)
                steps += 1
                if done:
                    break

    def test_legal_actions_are_consistent_with_phase(self):
        env = RunEnv(seed=4)
        step = env.reset()
        self.assertTrue(all(isinstance(a, MetaAction) for a in step.legal))
        node_id = step.run_obs.reachable[0]
        step, _r, _d, _t, _i = env.step(MetaAction("choose_node", node_id))
        self.assertTrue(all(isinstance(a, Action) for a in step.legal))

    def test_illegal_meta_action_is_rejected(self):
        env = RunEnv(seed=4)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(MetaAction("choose_node", 9999))
        with self.assertRaises(ValueError):
            env.step(MetaAction("pick_card", 0))

    def test_deck_grows_when_card_taken(self):
        env = RunEnv(seed=4)
        step = env.reset()
        before = len(env.raw_state.player.deck)
        step, _r, _d, _t, _i = env.step(
            MetaAction("choose_node", step.run_obs.reachable[0]))
        # 打完第一场战斗直到卡牌奖励
        steps = 0
        while step.phase == "combat" and steps < 400:
            step, _r, done, _t, _i = env.step(scripted_combat_action(step))
            steps += 1
            if done:
                break
        if step.phase == "card_reward":
            step, _r, _d, _t, _i = env.step(MetaAction("pick_card", 0))
            self.assertEqual(len(env.raw_state.player.deck), before + 1)

    def test_replay_reproduces_run(self):
        bot = RunBot()
        env = RunEnv(seed=6)
        step = env.reset()
        actions = []
        while step.phase not in ("won", "lost") and len(actions) < 3000:
            action = bot.act(step)
            if action is None:
                break
            actions.append(action)
            step, _r, done, _t, _i = env.step(action)
            if done:
                break
        a = env.replay(actions)
        b = env.replay(actions)
        self.assertEqual(
            (a.floor, a.player.hp, a.player.gold, len(a.player.deck)),
            (b.floor, b.player.hp, b.player.gold, len(b.player.deck)),
        )


class TestBaselineRunStrength(unittest.TestCase):
    """Run 层基线：``docs/04`` §4.1 要求的回归基线数字。"""

    def test_baseline_full_run_stats(self):
        bot = RunBot()
        trials = 80
        wins = 0
        floors = []
        for seed in range(trials):
            env = RunEnv(seed=1000 + seed, attempt_budget=1)
            phase, floor, _steps = play_run(bot, env)
            wins += phase == "won"
            floors.append(floor)
        average_floor = sum(floors) / len(floors)
        print(f"\n[baseline] 完整 run：通关率 {wins / trials:.1%}（{wins}/{trials}）"
              f" | 平均到达第 {average_floor:.1f} 层"
              f" | 最高层 {max(floors)} | 地图共 {generate_map(0).rows} 行")
        self.assertEqual(len(floors), trials)
        self.assertGreater(average_floor, 1.0, "基线 bot 太弱，连第一层都过不去")

    def test_content_modes_are_deterministic(self):
        """两种内容模式都必须**同种子完全可复现**。

        （不用"两种模式胜率相近"来断言：``predetermined`` 会在开局一次掷完所有节点
        的内容，``on_entry`` 走到哪掷到哪，两者消耗 RNG 的位置不同，因此面对的
        遭遇序列本就不同——那是预期行为，不是 bug。这个 flag 的用途是把
        ``docs/01`` §1.8 问题 3 变成一个可以两边都试的开关。）
        """
        bot = RunBot()
        for mode in ("predetermined", "on_entry"):
            runs = []
            for _ in range(2):
                outcomes = []
                for seed in range(12):
                    env = RunEnv(seed=seed, attempt_budget=1, content_mode=mode)
                    outcomes.append(play_run(bot, env))
                runs.append(outcomes)
            self.assertEqual(runs[0], runs[1], f"{mode} 模式不可复现")


class TestBaselineDiscrimination(unittest.TestCase):
    """基线要有**区分度**才叫基线。

    当前占位内容远比真机简单，规则 bot 几乎稳赢——所以"绝对胜率"这个数字此刻
    没有意义。有意义的是**相对**表现：同一个 harness 能否把好策略和坏策略分开。
    这一条才是回归基线的真正职责（``docs/04`` §4.1）。
    """

    def test_rulebot_substantially_beats_random_policy(self):
        import random

        class RandomBot:
            """对照组：完全随机选合法动作。"""

            def __init__(self, seed: int = 0) -> None:
                self.rng = random.Random(seed)

            def act(self, step):
                legal = list(step.legal)
                return self.rng.choice(legal) if legal else None

        def measure(bot) -> tuple[float, float]:
            trials = 60
            wins = 0
            floors = []
            for seed in range(trials):
                env = RunEnv(seed=5000 + seed, attempt_budget=1)
                phase, floor, _steps = play_run(bot, env, max_steps=1500)
                wins += phase == "won"
                floors.append(floor)
            return wins / trials, sum(floors) / len(floors)

        rule_rate, rule_floor = measure(RunBot())
        random_rate, random_floor = measure(RandomBot(seed=1))
        print(f"\n[discrimination] 规则 bot 通关 {rule_rate:.0%} / 平均第 {rule_floor:.1f} 层"
              f"  vs  随机策略 通关 {random_rate:.0%} / 平均第 {random_floor:.1f} 层")

        # 主判据用**平均到达楼层**：二值通关率在低胜率下方差极大，楼层是连续量、
        # 区分度好得多（docs/05 §5.1 也正是因此把它列为主指标之一）。
        self.assertGreater(rule_floor - random_floor, 2.0,
                           "harness 无法区分好策略与随机策略——基线失去意义")
        self.assertGreater(rule_rate, random_rate,
                           "规则 bot 的通关率不应低于随机策略")

    def test_random_policy_terminates(self):
        """随机策略可能乱走，但**必须**保证有限步内结束（不能死循环）。"""
        import random

        class RandomBot:
            def __init__(self, seed: int = 0) -> None:
                self.rng = random.Random(seed)

            def act(self, step):
                legal = list(step.legal)
                return self.rng.choice(legal) if legal else None

        for seed in range(10):
            env = RunEnv(seed=seed, attempt_budget=1)
            phase, _floor, steps = play_run(RandomBot(seed), env, max_steps=1500)
            self.assertIn(phase, ("won", "lost", "shop", "card_reward", "map", "rest",
                                  "event", "treasure", "combat"))
            self.assertLess(steps, 1500)


class TestRunThroughput(unittest.TestCase):
    def test_full_run_throughput(self):
        """完整 run 的吞吐——它直接决定 Run 层 RL 能做多少次实验。"""
        import time

        bot = RunBot()
        start = time.perf_counter()
        runs = 0
        decisions = 0
        while time.perf_counter() - start < 1.5:
            env = RunEnv(seed=10_000 + runs, attempt_budget=1)
            step = env.reset()
            decisions += 1
            while step.phase not in ("won", "lost") and decisions < 100_000:
                action = bot.act(step)
                if action is None:
                    break
                step, _r, done, _t, _i = env.step(action)
                decisions += 1
                if done:
                    break
            runs += 1
        elapsed = time.perf_counter() - start
        print(f"\n[bench] 完整 run：{runs / elapsed:,.1f} 局/秒 | "
              f"{decisions / elapsed:,.0f} 决策/秒（纯 Python，含 bot）")
        self.assertGreater(runs / elapsed, 1.0, "run 层吞吐过低")


if __name__ == "__main__":
    unittest.main(verbosity=2)
