"""反作弊测试套件（``docs/01`` §1.6）。

这些不是普通单元测试，而是**这套设计的科学诚信证明**：
一旦被绕过，所有实验结果都不可信，而且事后极难发现。
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib
import unittest

from sts2_sim import Action, RuleBot, SpireEnv, observe, reroll_hidden, start_combat
from sts2_sim.bot import FORBIDDEN_IDENTIFIERS
from sts2_sim.core import Hidden, legal_actions
from sts2_sim.rng import RngSet

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent.parent / "sts2_sim"

#: 描述"当前试次可见状态"的字段——重开对拍时只看这些
CURRENT_ATTEMPT_FIELDS = (
    "turn", "energy", "player_hp", "player_max_hp", "player_block",
    "player_powers", "hand", "draw_bag", "discard_bag", "exhaust_bag", "enemies",
    # ⭐ 审计 F04 补进来的公开字段也要纳入重开对拍：它们既然是公开信息，
    # 就必须在"重开面对同一局面"时逐位一致；反之，它们也不该携带
    # 任何未揭示信息（本文件同时有隐藏信息不变性测试盯着这一点）。
    "relics", "potions", "orbs", "orb_slots", "stars",
    "player_max_energy", "selection",
)


def visible_fingerprint(obs) -> tuple:
    """剔除环境公开的元信息（试次历史 / 预算），只留"这一遍打出来的东西"。"""
    return tuple(
        repr(getattr(obs, name))
        for name in CURRENT_ATTEMPT_FIELDS
    )


def scripted_actions(obs, info, turn_limit: int = 40) -> list[Action]:
    """固定动作脚本：总是打第一张合法牌，否则结束回合。

    用于重开对拍——脚本必须与观测无关，否则两次运行会因为知识不同而分叉。
    """
    legal = info["legal_actions"]
    for action in legal:
        if action.kind == "play_card" and action.hand_index == 0:
            return [action]
    return [Action("end_turn")]


class TestProjectionInvariance(unittest.TestCase):
    """测试 1：隐藏信息怎么变，观测都必须纹丝不动。"""

    def test_hidden_reroll_does_not_change_observation(self):
        state_a = start_combat(("strike",) * 5 + ("defend",) * 5, ("jaw_worm",), seed=7)
        state_b = start_combat(("strike",) * 5 + ("defend",) * 5, ("jaw_worm",), seed=7)
        # 可见状态此时完全一致；只把 b 的隐藏状态换掉
        reroll_hidden(state_b, 999_999)
        state_b.hidden.move_queue = ["chomp", "thrash", "bellow"]
        state_b.hidden.future_drops = ["card:rare", "relic:boss"]

        self.assertEqual(observe(state_a).to_json(), observe(state_b).to_json())

    def test_pile_order_is_not_in_observation(self):
        """抽牌堆的顺序属于 L3：观测里只能是多重集。"""
        state = start_combat(("strike",) * 5 + ("defend",) * 5, ("jaw_worm",), seed=11)
        before = observe(state).to_json()
        state.draw_pile.reverse()          # 只改顺序，不改内容
        self.assertEqual(before, observe(state).to_json())

    def test_observation_is_immutable(self):
        obs = observe(start_combat(("strike",), ("jaw_worm",), seed=1))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            obs.energy = 99  # type: ignore[misc]


class TestRestartSemantics(unittest.TestCase):
    """测试 2 ⭐：SL 语义。重开必须面对**同一副牌序**。

    如果 restore 重掷了 RNG，agent 会学到"重开可以刷到好牌"——这条策略在真机上
    根本不存在，训练结果全是假的。
    """

    def _run_script(self, env: SpireEnv, max_steps: int = 60) -> list[tuple]:
        """从**当前**节点状态跑一段固定脚本。

        注意：**不调用 reset()**——那会重建节点快照，使重开对拍失去意义。
        """
        obs = env._observe()
        info = env._info()
        # 必须从 combat 阶段开始，否则脚本会立刻退出（曾经的 SL 阶段机改动
        # 就让这条测试变成了恒真的空测试）
        assert info["stage"] == "combat", f"脚本应从 combat 开始，实际 {info['stage']}"
        trace = [visible_fingerprint(obs)]
        for _ in range(max_steps):
            if info["phase"] in ("won", "lost"):
                break
            action = scripted_actions(obs, info)[0]
            obs, _reward, done, _trunc, info = env.step(action)
            trace.append(visible_fingerprint(obs))
            if done:
                break
        return trace

    def test_restart_faces_identical_shuffle(self):
        env = SpireEnv(seed=1234, encounter=("jaw_worm", "red_louse"),
                       attempt_budget=3)
        env.reset()
        first = self._run_script(env)
        self.assertTrue(env.restart_attempt(), "预算充足，重开应当成功")
        second = self._run_script(env)
        self.assertEqual(first, second,
                         "重开后可见轨迹必须完全一致——否则 SL 语义被破坏")

    def test_reroll_changes_trajectory_proving_the_test_has_power(self):
        """反向验证：**故意**让 restore 重掷隐藏 RNG 时轨迹必须改变。

        没有这一条，上一个测试可能只是"恒真"而无鉴别力。
        注意要改的是**快照里**的隐藏状态——那正是"restore 实现错误"的形态。
        """
        env = SpireEnv(seed=1234, encounter=("jaw_worm", "red_louse"),
                       attempt_budget=3)
        env.reset()
        first = self._run_script(env)
        self.assertGreater(len(first), 3, "脚本太短，对拍没有鉴别力")
        env._node_snapshot.state.hidden = Hidden(RngSet(4242))
        self.assertTrue(env.restart_attempt())
        second = self._run_script(env)
        self.assertNotEqual(first, second, "重掷后轨迹应当不同；否则对拍测试无效")

    def test_attempt_budget_is_enforced_by_env(self):
        env = SpireEnv(seed=5, attempt_budget=3)
        env.reset()
        self.assertEqual(env.attempts_left, 3)
        for expected_left in (2, 1, 0):
            self.assertTrue(env.restart_attempt())
            self.assertEqual(env.attempts_left, expected_left)
        self.assertFalse(env.restart_attempt(), "预算耗尽后必须拒绝重开")
        self.assertEqual(env.attempts_used, 3)

    def test_prior_attempts_are_recorded_as_revealed_knowledge(self):
        env = SpireEnv(seed=77, attempt_budget=2)
        env.reset()
        obs = env._observe()
        self.assertEqual(len(obs.prior_attempts), 0)
        env.restart_attempt()
        obs = env._observe()
        self.assertEqual(len(obs.prior_attempts), 1, "重开后应携带前一次已揭示的观察")
        self.assertEqual(obs.prior_attempts[0].outcome, "aborted")
        self.assertTrue(obs.prior_attempts[0].hand_by_turn)


class TestSeedInvisibility(unittest.TestCase):
    """测试 5：种子及其派生物不得出现在观测里（否则模型会记忆种子）。"""

    def test_seed_never_appears_in_observation(self):
        seed = 987_654_321
        obs = observe(start_combat(("strike",) * 5 + ("defend",) * 5,
                                   ("jaw_worm",), seed=seed))
        payload = obs.to_json()
        self.assertNotIn(str(seed), payload)
        for forbidden in ("seed", "rng", "hidden", "master", "shuffle",
                          "future", "move_queue", "getstate"):
            self.assertNotIn(forbidden, payload.lower(),
                             f"观测中出现了疑似泄漏字段 {forbidden!r}")

    def test_observation_has_no_hidden_attributes(self):
        obs = observe(start_combat(("strike",), ("jaw_worm",), seed=3))
        for name in ("hidden", "rng", "state", "draw_pile", "seed"):
            self.assertFalse(hasattr(obs, name), f"Observation 不应有 {name!r}")


class TestAttribution(unittest.TestCase):
    """测试 4：扰动隐藏字段，策略输出必须逐位不变。"""

    def test_policy_output_invariant_under_hidden_mutation(self):
        state = start_combat(("strike",) * 5 + ("defend",) * 4 + ("bash",),
                             ("jaw_worm",), seed=42)
        bot = RuleBot()
        obs_before = observe(state)
        # 需要合法动作做决策；直接用引擎算
        from sts2_sim import legal_actions
        action_before = bot.act(obs_before, legal_actions(state))

        reroll_hidden(state, 31337)
        state.hidden.move_queue = ["bellow"]
        obs_after = observe(state)
        action_after = bot.act(obs_after, legal_actions(state))

        self.assertEqual(obs_before.to_json(), obs_after.to_json())
        self.assertEqual(action_before, action_after)


class TestPolicyModuleIsolation(unittest.TestCase):
    """测试 7：策略模块**在语法层面**无法触及隐藏信息（docs/01 §1.6 第二层）。"""

    ALLOWED_CORE_IMPORTS = {"Action"}

    def _iter_policy_sources(self):
        for name in ("bot.py",):
            yield PACKAGE_DIR / name

    def test_no_forbidden_identifiers_in_policy_modules(self):
        for path in self._iter_policy_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            used: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used.add(node.id)
                elif isinstance(node, ast.Attribute):
                    used.add(node.attr)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        used.add(alias.name)
            leaked = used & FORBIDDEN_IDENTIFIERS
            self.assertFalse(leaked, f"{path.name} 触及了隐藏信息标识符：{leaked}")

    def test_policy_modules_only_import_action_from_core(self):
        for path in self._iter_policy_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = (node.module or "").lstrip(".")
                    if module in ("core", "rng", "env"):
                        imported = {alias.name for alias in node.names}
                        illegal = imported - self.ALLOWED_CORE_IMPORTS
                        self.assertFalse(
                            illegal,
                            f"{path.name} 从 {module} 导入了非公开符号：{illegal}",
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
