"""引擎不变量、基线 bot 强度、吞吐、可复现性。"""

from __future__ import annotations

import time
import unittest

from sts2_sim import Action, RuleBot, SpireEnv, lint, start_combat, step, unverified_report
from sts2_sim.content import STARTING_DECK
from sts2_sim.core import CombatState

STEPS_PER_COMBAT_CAP = 400


def check_invariants(state: CombatState) -> list[str]:
    """返回违规列表（空 = 通过）。``docs/02`` §2.5 的"不变量"测试。"""
    problems: list[str] = []
    for combatant in [state.player, *state.enemies]:
        if combatant.hp < 0:
            problems.append(f"{combatant.name}: HP 为负 {combatant.hp}")
        if combatant.hp > combatant.max_hp:
            problems.append(f"{combatant.name}: HP 超过上限")
        if combatant.block < 0:
            problems.append(f"{combatant.name}: 格挡为负")
        for name, value in combatant.powers.items():
            if value <= 0:
                problems.append(f"{combatant.name}: 能力 {name} 层数非正 {value}")
    if state.energy < 0:
        problems.append(f"能量为负 {state.energy}")
    if state.energy > 20:
        problems.append(f"能量异常高 {state.energy}")
    expected = len(STARTING_DECK) + state.added_cards
    actual = len(state.all_piles())
    if actual != expected:
        problems.append(f"卡牌总数不守恒：{actual} != {expected}")
    return problems


def play_with_bot(env: SpireEnv, bot: RuleBot, max_steps: int = STEPS_PER_COMBAT_CAP):
    obs, info = env.reset()
    steps = 0
    problems: list[str] = []
    while steps < max_steps:
        problems.extend(check_invariants(env.raw_state))
        if info["phase"] in ("won", "lost"):
            break
        action = bot.act(obs, info["legal_actions"])
        obs, _reward, done, _trunc, info = env.step(action)
        steps += 1
        if done:
            break
    return env.raw_state.phase, steps, problems


class TestContent(unittest.TestCase):
    def test_lint_passes(self):
        self.assertEqual(lint(), [], "内容表静态检查未通过")

    def test_content_is_marked_unverified(self):
        """骨架内容全部未对拍——这是事实声明，不是缺陷。

        一旦某项通过了真机对拍，就把它改成 verified=True；
        训练时只使用 verified 的子集（docs/06 §6.5）。
        """
        report = unverified_report()
        self.assertTrue(report["cards"], "骨架阶段应当有未对拍内容")
        self.assertTrue(report["enemies"])


class TestEngineInvariants(unittest.TestCase):
    def test_selfplay_never_violates_invariants(self):
        bot = RuleBot()
        encounters = [
            ("jaw_worm",),
            ("cultist",),
            ("red_louse", "red_louse"),
            ("jaw_worm", "red_louse"),
            ("cultist", "red_louse", "red_louse"),
        ]
        for seed in range(12):
            for encounter in encounters:
                env = SpireEnv(seed=seed, encounter=encounter, attempt_budget=1)
                phase, steps, problems = play_with_bot(env, bot)
                self.assertEqual(problems, [], f"seed={seed} {encounter} 违反不变量")
                self.assertIn(phase, ("won", "lost"))
                self.assertLess(steps, STEPS_PER_COMBAT_CAP, "疑似死循环")

    def test_illegal_action_is_rejected(self):
        state = start_combat(STARTING_DECK, ("jaw_worm",), seed=1)
        state.energy = 0
        with self.assertRaises(ValueError):
            step(state, Action("play_card", 0, 0))

    def test_replay_is_deterministic(self):
        env = SpireEnv(seed=2024, encounter=("jaw_worm",))
        bot = RuleBot()
        obs, info = env.reset()
        actions = []
        while info["phase"] not in ("won", "lost") and len(actions) < 200:
            action = bot.act(obs, info["legal_actions"])
            actions.append(action)
            obs, _r, done, _t, info = env.step(action)
            if done:
                break
        a = env.replay(actions)
        b = env.replay(actions)
        self.assertEqual(render_fingerprint(a), render_fingerprint(b))

    def test_different_seeds_give_different_shuffles(self):
        seen = set()
        for seed in range(20):
            state = start_combat(STARTING_DECK, ("jaw_worm",), seed=seed)
            seen.add(tuple(c.cid for c in state.draw_pile))
        self.assertGreater(len(seen), 5, "洗牌似乎没有真正随机化")


def render_fingerprint(state: CombatState) -> tuple:
    return (
        state.player.hp, state.player.block, state.turn,
        tuple(sorted((e.eid, e.hp, e.block) for e in state.enemies)),
        tuple(c.cid for c in state.hand),
        tuple(c.cid for c in state.draw_pile),
        tuple(c.cid for c in state.discard),
    )


class TestBaselineBot(unittest.TestCase):
    """基线 bot 必须能赢下一部分战斗，否则它作为回归基线没有意义。"""

    def test_bot_beats_single_jaw_worm_often(self):
        bot = RuleBot()
        wins = 0
        trials = 40
        for seed in range(trials):
            env = SpireEnv(seed=seed, encounter=("jaw_worm",),
                           player_hp=80, attempt_budget=1)
            phase, _steps, _problems = play_with_bot(env, bot)
            wins += phase == "won"
        rate = wins / trials
        print(f"\n[baseline] 颚虫单怪胜率 {rate:.0%}（{wins}/{trials}）")
        self.assertGreater(rate, 0.5, "基线 bot 太弱，无法充当回归基线")

    def test_bot_handles_multi_enemy(self):
        bot = RuleBot()
        env = SpireEnv(seed=3, encounter=("red_louse", "red_louse"), attempt_budget=1)
        phase, _steps, problems = play_with_bot(env, bot)
        self.assertEqual(problems, [])
        self.assertIn(phase, ("won", "lost"))


class TestBaselineNoSL(unittest.TestCase):
    """K=1 基线：不允许重开时的表现（docs/01 §1.5）。"""

    def test_k1_baseline_runs(self):
        bot = RuleBot()
        wins = 0
        trials = 30
        for seed in range(trials):
            env = SpireEnv(seed=1000 + seed, encounter=("cultist",),
                           attempt_budget=1)
            phase, _s, _p = play_with_bot(env, bot)
            wins += phase == "won"
        print(f"[baseline] K=1 邪教徒胜率 {wins / trials:.0%}（{wins}/{trials}）")


class TestThroughput(unittest.TestCase):
    def test_throughput_measurement(self):
        """吞吐是项目的**第一生产力**——它决定你能做多少次实验（docs/02 §2.6）。"""
        bot = RuleBot()
        start = time.perf_counter()
        steps = 0
        combats = 0
        while time.perf_counter() - start < 1.0:
            env = SpireEnv(seed=combats, encounter=("jaw_worm", "red_louse"),
                           attempt_budget=1)
            obs, info = env.reset()
            while True:
                if info["phase"] in ("won", "lost"):
                    break
                action = bot.act(obs, info["legal_actions"])
                obs, _r, done, _t, info = env.step(action)
                steps += 1
                if done:
                    break
            combats += 1
        elapsed = time.perf_counter() - start
        print(f"\n[bench] {steps / elapsed:,.0f} 步/秒 | {combats / elapsed:,.1f} 局/秒 "
              f"（纯 Python，含 bot 决策）")
        self.assertGreater(steps / elapsed, 300,
                           "吞吐过低：先优化模拟器，不要加大网络")


if __name__ == "__main__":
    unittest.main(verbosity=2)
