"""PPO 侧的两条审计回归（``docs/12`` F12 / 工单 T12）。

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 改 gamma 后塑形不再是势函数塑形 | 塑形项写死 ``Φ(s')−Φ(s)``，与 GAE 的 γ 脱钩 | `TestShapingUsesTheSameGamma` |
| 内容换了、checkpoint 照旧能加载 | 只存权重与 config，没有内容/词表/协议指纹 | `TestCheckpointSelfDescribes` |
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from sts2_rl.ppo import (
    OBSERVATION_PROTOCOL, SL_PROTOCOL, CombatPool, PPOConfig,
    checkpoint_payload, load_checkpoint, shaped_reward, verify_checkpoint,
)
from sts2_rl.nets import PolicyValueNet


def tiny_pool(gamma: float = 1.0, n_envs: int = 4) -> CombatPool:
    torch.manual_seed(0)
    np.random.seed(0)
    # 本组测试跑在内置占位内容上 → 显式 allow_builtin（正式训练默认禁止）。
    return CombatPool(n_envs, base_seed=1, shaping=True, attempt_budget=0,
                      selection="best", allow_builtin=True, gamma=gamma)


class TestShapingUsesTheSameGamma(unittest.TestCase):
    """``docs/13`` §10.4：``r' = r + γ·Φ(s') − Φ(s)``，两处必须同一个 γ。"""

    def test_gamma_one_matches_the_old_formula(self):
        """γ=1 时与旧实现（``Φ(s')−Φ(s)``）逐位一致 —— 保证不改默认行为。"""
        self.assertAlmostEqual(shaped_reward(0.0, 0.3, 0.7, False, gamma=1.0),
                               0.7 - 0.3, places=9)

    def test_gamma_scales_the_next_potential(self):
        """反向验证：γ 不同时结果**必须不同**，否则说明 γ 根本没参与。"""
        one = shaped_reward(0.0, 0.3, 0.7, False, gamma=1.0)
        half = shaped_reward(0.0, 0.3, 0.7, False, gamma=0.5)
        self.assertNotAlmostEqual(one, half, places=9,
                                  msg="γ 没参与塑形 → 势函数保证失效")
        self.assertAlmostEqual(half, 0.5 * 0.7 - 0.3, places=9)

    def test_terminal_state_has_zero_potential(self):
        """终局的势取 0：``r' = r − Φ(s)``。"""
        self.assertAlmostEqual(shaped_reward(1.0, 0.4, 0.9, True, gamma=0.9),
                               1.0 - 0.4, places=9)

    def test_shaping_off_returns_the_raw_reward(self):
        self.assertAlmostEqual(shaped_reward(0.5, 0.3, 0.7, False, 1.0,
                                             shaping=False), 0.5, places=9)

    def test_pool_records_the_gamma_it_uses(self):
        self.assertEqual(tiny_pool(gamma=0.9).gamma, 0.9)


class TestCheckpointSelfDescribes(unittest.TestCase):
    """审计 F12：checkpoint 必须能自证"我跟当前内容是同一版"。"""

    def setUp(self):
        self.config = PPOConfig(n_envs=2, rollout_len=2, total_updates=1,
                                allow_builtin=True, d_model=32, n_layers=1, n_heads=2)
        self.net = PolicyValueNet(32, 1, 2)

    def test_payload_contains_all_required_fingerprints(self):
        payload = checkpoint_payload(self.net, self.config)
        for key in ("state_dict", "config", "content_fingerprint",
                    "vocab_fingerprint", "observation_protocol", "sl_protocol",
                    "content_source", "torch_rng", "numpy_rng", "python_rng"):
            self.assertIn(key, payload, f"checkpoint 缺少 {key}")
        self.assertEqual(payload["observation_protocol"], OBSERVATION_PROTOCOL)
        self.assertEqual(payload["sl_protocol"], SL_PROTOCOL)

    def test_matching_payload_verifies_clean(self):
        payload = checkpoint_payload(self.net, self.config)
        self.assertEqual(verify_checkpoint(payload), [])

    def test_stale_observation_protocol_is_rejected(self):
        payload = checkpoint_payload(self.net, self.config)
        payload["observation_protocol"] = "obs-v0"
        with self.assertRaises(ValueError):
            verify_checkpoint(payload)

    def test_stale_content_fingerprint_is_rejected(self):
        """内容变了而 checkpoint 没变 → **必须报错**，不能静默加载。"""
        payload = checkpoint_payload(self.net, self.config)
        payload["content_fingerprint"] = {"cards": 0, "content_hash": -1}
        with self.assertRaises(ValueError):
            verify_checkpoint(payload)

    def test_non_strict_mode_returns_the_problems(self):
        payload = checkpoint_payload(self.net, self.config)
        payload["vocab_fingerprint"] = {"vocab_hash": -1}
        problems = verify_checkpoint(payload, strict_content=False)
        self.assertTrue(any("词表指纹" in p for p in problems))

    def test_round_trip_through_disk(self):
        import tempfile
        from pathlib import Path
        payload = checkpoint_payload(self.net, self.config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            torch.save(payload, path)
            loaded = load_checkpoint(path, device="cpu")
        self.assertEqual(loaded["observation_protocol"], OBSERVATION_PROTOCOL)
        self.assertEqual(set(loaded["state_dict"]), set(self.net.state_dict()))


class TestEnvironmentFaultsAreCountedSeparately(unittest.TestCase):
    """``docs/13`` §12.3：环境异常 / 未知机制 / 操作失败必须与策略输局**分开计数**。

    否则一个环境 bug 会表现为"胜率下降"，而且没有任何信号。
    """

    def test_guard_swallows_the_fault_and_records_the_seed(self):
        from sts2_rl.ppo import _run_episode_guarded, make_env
        import random as _random
        env = make_env(1, _random.Random(1), attempt_budget=0, character="ironclad")
        faults: list[dict] = []

        def boom(_obs, _info):
            raise RuntimeError("模拟未知机制")

        result = _run_episode_guarded(env, boom, seed=4242, faults=faults)
        self.assertIsNone(result, "故障局面不应计入结果")
        self.assertEqual(len(faults), 1)
        self.assertEqual(faults[0]["seed"], 4242)
        self.assertIn("模拟未知机制", faults[0]["error"])

    def test_summary_excludes_faults_from_the_denominator(self):
        from sts2_rl.ppo import _summarize
        results = [{"won": True, "hp_left": 10, "turns": 3, "attempts_used": 0},
                   {"won": False, "hp_left": 0, "turns": 5, "attempts_used": 0}]
        summary = _summarize(results)
        # 两局里一胜一负 → 50%（没有故障局被算进去）
        self.assertAlmostEqual(summary["win_rate"], 0.5)
        self.assertEqual(summary["n_episodes"], 2)

    def test_faults_are_classified_by_cause(self):
        """``docs/13`` §7：未知机制 / 数据缺口 / 引擎 bug 必须分得开。

        混成一类时，"抽取器漏了一条数据"看起来会和"引擎写错了"一样，
        而两者的修法完全不同（补内容 vs 改代码）。
        """
        from sts2_rl.ppo import (
            FAULT_BUG, FAULT_DATA_GAP, FAULT_UNKNOWN_MECHANIC, classify_fault,
        )

        class UnsupportedMechanic(RuntimeError):
            pass

        class UnknownCondition(RuntimeError):
            pass

        self.assertEqual(classify_fault(UnsupportedMechanic("x")),
                         FAULT_UNKNOWN_MECHANIC)
        self.assertEqual(classify_fault(UnknownCondition("x")),
                         FAULT_UNKNOWN_MECHANIC)
        self.assertEqual(classify_fault(KeyError("missing_card")), FAULT_DATA_GAP)
        self.assertEqual(classify_fault(IndexError("oob")), FAULT_DATA_GAP)
        self.assertEqual(classify_fault(ValueError("协议错")), FAULT_BUG)

    def test_fault_counts_are_grouped(self):
        from sts2_rl.ppo import FAULT_BUG, FAULT_DATA_GAP, _summarize
        summary = _summarize(
            [{"won": True, "hp_left": 1, "turns": 1, "attempts_used": 0}],
            [{"seed": 1, "category": FAULT_DATA_GAP, "error": "KeyError"},
             {"seed": 2, "category": FAULT_BUG, "error": "ValueError"}])
        self.assertEqual(summary["n_faults"], 2)
        self.assertEqual(summary["faults_by_category"],
                         {FAULT_DATA_GAP: 1, FAULT_BUG: 1})
        self.assertEqual(summary["n_episodes"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
