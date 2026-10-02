"""SL（save-load）机制测试。

``docs/01`` §1.4 的 meta-episode：**重开是环境的一等决策**，不是外部脚本里的一个方法。
这一组测试锁住 SL 的完整语义——它是本项目相对 NoSL 训练的全部差异所在。
"""

from __future__ import annotations

import unittest

from sts2_sim import Action, RuleBot, SpireEnv, legal_actions, observe
from sts2_sim.env import (
    MAX_ATTEMPT_BUDGET_REAL, MAX_ATTEMPT_BUDGET_SIM, STAGE_COMBAT, STAGE_DONE,
    STAGE_META,
)
from sts2_sim.featurize import (
    CARD_VOCAB, MAX_REVEALED, REVEALED_OFFSET, TOKEN_REVEALED, encode_observation,
)

DECK = ("strike",) * 5 + ("defend",) * 4 + ("bash",)
ENCOUNTER = ("jaw_worm", "red_louse")


def scripted_action(obs, info):
    """固定脚本：总是打第一张合法牌，否则结束回合。与观测无关。"""
    for action in info["legal_actions"]:
        if action.kind == "play_card" and action.hand_index == 0:
            return action
    return Action("end_turn")


def play_to_meta(env: SpireEnv, max_steps: int = 120):
    """把当前试次打完，返回 (obs, info)。"""
    obs, info = env._observe(), env._info()
    steps = 0
    while info["stage"] == STAGE_COMBAT and steps < max_steps:
        action = scripted_action(obs, info)
        obs, _r, done, _t, info = env.step(action)
        steps += 1
        if done:
            break
    return obs, info


class TestSLStageMachine(unittest.TestCase):
    def test_nosl_has_no_restart_action(self):
        """预算 0（NoSL）时不应出现 restart 动作。"""
        env = SpireEnv(seed=1, encounter=ENCOUNTER, deck=DECK, attempt_budget=0)
        env.reset()
        kinds = [a.kind for a in env.legal_actions()]
        self.assertNotIn("restart", kinds)
        self.assertIn("end_turn", kinds)

    def test_zero_budget_finalizes_immediately_after_combat(self):
        env = SpireEnv(seed=2, encounter=ENCOUNTER, deck=DECK, attempt_budget=0)
        env.reset()
        _obs, info = play_to_meta(env)
        self.assertEqual(info["stage"], STAGE_DONE)
        self.assertTrue(info["phase"] in ("won", "lost"))

    def test_restart_action_available_when_budget_remains(self):
        env = SpireEnv(seed=1, encounter=ENCOUNTER, deck=DECK, attempt_budget=3)
        env.reset()
        kinds = [a.kind for a in env.legal_actions()]
        self.assertIn("restart", kinds)

    def test_meta_stage_appears_after_combat_when_budget_remains(self):
        env = SpireEnv(seed=2, encounter=ENCOUNTER, deck=DECK, attempt_budget=2)
        env.reset()
        _obs, info = play_to_meta(env)
        self.assertEqual(info["stage"], STAGE_META,
                         "预算还在时，战斗结束应进入 meta 阶段等待决策")
        kinds = [a.kind for a in info["legal_actions"]]
        self.assertIn("accept", kinds)
        self.assertIn("restart", kinds)

    def test_no_meta_stage_when_budget_exhausted(self):
        env = SpireEnv(seed=2, encounter=ENCOUNTER, deck=DECK, attempt_budget=0)
        env.reset()
        _obs, info = play_to_meta(env)
        self.assertEqual(info["stage"], STAGE_DONE)
        self.assertTrue(info["phase"] in ("won", "lost"))

    def test_accept_finalizes_with_best_attempt_outcome(self):
        """accept = "别试了，把目前最好的那次留下"（不是"留下最后一次"）。"""
        env = SpireEnv(seed=3, encounter=ENCOUNTER, deck=DECK, attempt_budget=2)
        env.reset()
        _obs, info = play_to_meta(env)
        best = info["best_score"]
        _obs, reward, done, _t, info2 = env.step(Action("accept"))
        self.assertTrue(done)
        self.assertEqual(info2["stage"], STAGE_DONE)
        self.assertAlmostEqual(reward, best)
        self.assertEqual(info2["kept_attempt"], 0)

    def test_budget_cap_is_enforced(self):
        """模拟器内 SL 次数上限 50（用户规格）。"""
        with self.assertRaises(ValueError):
            SpireEnv(seed=1, attempt_budget=MAX_ATTEMPT_BUDGET_SIM + 1)
        env = SpireEnv(seed=1, attempt_budget=MAX_ATTEMPT_BUDGET_SIM)
        self.assertEqual(env.attempt_budget, MAX_ATTEMPT_BUDGET_SIM)

    def test_real_machine_budget_constant_is_tighter(self):
        self.assertLess(MAX_ATTEMPT_BUDGET_REAL, MAX_ATTEMPT_BUDGET_SIM)
        self.assertEqual(MAX_ATTEMPT_BUDGET_REAL, 10)

    def test_restart_is_penalized(self):
        env = SpireEnv(seed=4, encounter=ENCOUNTER, deck=DECK, attempt_budget=3,
                       restart_penalty=-0.25)
        env.reset()
        _obs, reward, done, _t, info = env.step(Action("restart"))
        self.assertFalse(done)
        self.assertAlmostEqual(reward, -0.25)
        self.assertEqual(info["attempts_left"], 2)

    def test_restart_from_meta_starts_fresh_attempt_same_shuffle(self):
        env = SpireEnv(seed=5, encounter=ENCOUNTER, deck=DECK, attempt_budget=3)
        env.reset()
        first_hand = tuple(c.cid for c in env._observe().hand)

        # 打几手再重开
        obs, info = env._observe(), env._info()
        for _ in range(3):
            action = scripted_action(obs, info)
            obs, _r, done, _t, info = env.step(action)
            if done:
                break
        env.step(Action("restart"))
        second_hand = tuple(c.cid for c in env._observe().hand)
        self.assertEqual(first_hand, second_hand, "重开必须面对同一副牌序")

    def test_budget_is_enforced(self):
        env = SpireEnv(seed=6, encounter=ENCOUNTER, deck=DECK, attempt_budget=2)
        env.reset()
        env.step(Action("restart"))
        env.step(Action("restart"))
        self.assertEqual(env.attempts_left, 0)
        kinds = [a.kind for a in env.legal_actions()]
        self.assertNotIn("restart", kinds, "预算耗尽后 restart 必须从合法动作里消失")
        with self.assertRaises(ValueError):
            env.step(Action("restart"))

    def test_prior_attempts_accumulate_knowledge(self):
        env = SpireEnv(seed=7, encounter=ENCOUNTER, deck=DECK, attempt_budget=3)
        env.reset()
        self.assertEqual(len(env._observe().prior_attempts), 0)
        env.step(Action("restart"))
        obs = env._observe()
        self.assertEqual(len(obs.prior_attempts), 1)
        self.assertTrue(obs.prior_attempts[0].hand_by_turn,
                        "重开后必须携带上一遍观察到的牌序")
        env.step(Action("restart"))
        self.assertEqual(len(env._observe().prior_attempts), 2)

    def test_invalid_actions_are_rejected(self):
        env = SpireEnv(seed=8, encounter=ENCOUNTER, deck=DECK, attempt_budget=2)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(Action("accept"))          # combat 阶段不能 accept
        _obs, info = play_to_meta(env)
        with self.assertRaises(ValueError):
            env.step(Action("play_card", 0, 0))  # meta 阶段不能出牌


class TestSLKnowledgeReachesThePolicy(unittest.TestCase):
    """⭐ SL 的知识必须**真的进入策略输入**，否则重开学不到任何东西。"""

    def test_featurizer_encodes_revealed_cards(self):
        env = SpireEnv(seed=9, encounter=ENCOUNTER, deck=DECK, attempt_budget=3)
        obs, info = env.reset()
        before = encode_observation(obs, env.legal_actions())
        self.assertEqual(int((before["token_type"] == TOKEN_REVEALED).sum()), 0,
                         "还没重开过，不该有已揭示 token")
        self.assertEqual(float(before["token_mask"][REVEALED_OFFSET:].sum()), 0.0)
        self.assertAlmostEqual(float(before["token_num"][0, 18]), 0.0)

        env.step(Action("restart"))
        obs = env._observe()
        after = encode_observation(obs, env.legal_actions())

        self.assertEqual(int((after["token_type"] == TOKEN_REVEALED).sum()),
                         len(obs.prior_attempts[0].hand_by_turn[0]),
                         "已揭示 token 数应等于上一遍第 1 回合的手牌数")
        self.assertGreater(float(after["revealed_counts"].sum()), 0.0)
        self.assertGreater(float(after["token_num"][0, 18]), 0.0)   # has_knowledge
        self.assertGreater(float(after["token_num"][0, 16]), 0.0)   # attempts_left

    def test_revealed_cards_are_real_cards(self):
        env = SpireEnv(seed=10, encounter=ENCOUNTER, deck=DECK, attempt_budget=3)
        env.reset()
        env.step(Action("restart"))
        obs = env._observe()
        encoded = encode_observation(obs, env.legal_actions())
        revealed = encoded["token_type"] == TOKEN_REVEALED
        for entity in encoded["token_entity"][revealed]:
            self.assertIn(int(entity), range(len(CARD_VOCAB)))

    def test_knowledge_does_not_leak_unrevealed_cards(self):
        """只有**看到过**的牌才进知识区。第二遍开局时，未来回合的牌仍不可知。"""
        env = SpireEnv(seed=11, encounter=ENCOUNTER, deck=DECK, attempt_budget=3)
        env.reset()
        env.step(Action("restart"))
        obs = env._observe()
        revealed = sum(len(h) for a in obs.prior_attempts for h in a.hand_by_turn)
        self.assertLessEqual(revealed, MAX_REVEALED)
        # 第一遍只打了开局 5 张（重开发生在第 1 回合内）→ 知识只有 5 张
        self.assertLessEqual(revealed, 5)

    def test_restart_action_is_encodable(self):
        env = SpireEnv(seed=12, encounter=ENCOUNTER, deck=DECK, attempt_budget=3)
        env.reset()
        legal = env.legal_actions()
        encoded = encode_observation(env._observe(), legal)
        restart_index = next(i for i, a in enumerate(legal) if a.kind == "restart")
        self.assertAlmostEqual(float(encoded["act_mask"][restart_index]), 1.0)
        self.assertNotEqual(int(encoded["act_kind"][restart_index]), 0)


class TestBestOfSelection(unittest.TestCase):
    """"到次数之后选择最优解"。

    构造一个"第 1 遍打赢、第 2 遍故意打输"的局面，检查环境保留的是**最优**那次。
    """

    def _play_attempt(self, env: SpireEnv, use_bot: bool) -> dict:
        bot = RuleBot()
        obs, info = env._observe(), env._info()
        steps = 0
        while info["stage"] == STAGE_COMBAT and steps < 200:
            action = (bot.act(obs, info["legal_actions"]) if use_bot
                      else Action("end_turn"))
            obs, _r, done, _t, info = env.step(action)
            steps += 1
            if done:
                break
        return info

    def test_keeps_best_attempt_not_last(self):
        env = SpireEnv(seed=31, encounter=("jaw_worm",), deck=DECK,
                       attempt_budget=3, selection="best")
        env.reset()
        first = self._play_attempt(env, use_bot=True)          # 认真打 → 大概率赢
        self.assertGreater(first["best_score"], 0.0, "第 1 遍应当打赢，否则用例本身失效")

        env.step(Action("restart"))
        self._play_attempt(env, use_bot=False)                  # 故意摆烂 → 输
        _obs, reward, done, _t, info = env.step(Action("accept"))

        self.assertTrue(done)
        self.assertEqual(info["kept_attempt"], 0, "应当保留第 0 次（打赢的那次）")
        self.assertGreater(reward, 1.0, "保留最优解 → 回报应当是赢的分数")

    def test_last_mode_differs_from_best_mode(self):
        """"最后一次"模式下，摆烂那次的坏结果会被保留——两种模式必须可区分。"""

        def run(selection: str) -> tuple[float, int | None]:
            env = SpireEnv(seed=31, encounter=("jaw_worm",), deck=DECK,
                           attempt_budget=3, selection=selection)
            env.reset()
            self._play_attempt(env, use_bot=True)
            env.step(Action("restart"))
            self._play_attempt(env, use_bot=False)
            _obs, reward, _done, _t, info = env.step(Action("accept"))
            return reward, info["kept_attempt"]

        best_reward, best_kept = run("best")
        last_reward, last_kept = run("last")
        self.assertEqual(best_kept, 0)
        self.assertEqual(last_kept, 1)
        self.assertGreater(best_reward, last_reward,
                           "best 模式必须严格优于 last 模式（本例中）")

    def test_exhausting_budget_also_keeps_best(self):
        """预算耗尽时不需要再决策，直接保留最优解。"""
        env = SpireEnv(seed=41, encounter=("jaw_worm",), deck=DECK,
                       attempt_budget=2, selection="best")
        env.reset()
        first = self._play_attempt(env, use_bot=True)
        env.step(Action("restart"))
        env.step(Action("restart"))     # 第二次重开用光预算
        self.assertEqual(env.attempts_left, 0)

        # 预算耗尽后把这一遍打完 → 自动结束（不再有 meta 决策）
        obs, info = env._observe(), env._info()
        reward, done = 0.0, False
        steps = 0
        while not done and steps < 200:
            obs, reward, done, _t, info = env.step(Action("end_turn"))
            steps += 1
        self.assertTrue(done)
        self.assertEqual(info["stage"], STAGE_DONE)
        self.assertGreaterEqual(reward, first["best_score"] - 1e-6,
                                "耗尽预算后也必须保留最优解")


class TestSLCurriculumBaseline(unittest.TestCase):
    """K 越大胜率应当越高——否则说明 SL 没有可被利用的信息。"""

    def test_more_restarts_help_a_weak_policy(self):
        bot = RuleBot()

        def run(attempt_budget: int, trials: int = 60) -> float:
            wins = 0
            for seed in range(trials):
                env = SpireEnv(seed=20_000 + seed, encounter=("cultist",),
                               deck=DECK, attempt_budget=attempt_budget)
                obs, info = env.reset()
                steps = 0
                while info["stage"] != STAGE_DONE and steps < 400:
                    if info["stage"] == STAGE_META:
                        # 弱策略：输了就重开（预算允许时），否则接受
                        kinds = {a.kind for a in info["legal_actions"]}
                        action = (Action("restart") if info["phase"] == "lost"
                                  and "restart" in kinds else Action("accept"))
                    else:
                        action = bot.act(obs, info["legal_actions"])
                    obs, _r, done, _t, info = env.step(action)
                    steps += 1
                    if done:
                        break
                wins += info["phase"] == "won"
            return wins / trials

        k0 = run(0)          # NoSL
        k4 = run(4)
        print(f"\n[SL] 规则 bot：K=0(NoSL) 胜率 {k0:.0%} | K=4 胜率 {k4:.0%}")
        self.assertGreaterEqual(k4, k0, "更多重开机会不该让胜率下降")


if __name__ == "__main__":
    unittest.main(verbosity=2)
