"""训练侧回归测试。

这一组测试针对的是"**损失看起来正常，网络却坏掉了**"这一类问题——
它们无法从 loss 曲线发现，只能靠显式断言参数的数值健康度。
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from sts2_rl.nets import MASK_LOGIT, PolicyValueNet
from sts2_rl.ppo import CombatPool, to_torch
from sts2_sim.featurize import MAX_ACTIONS


def make_model_and_batch(device: str = "cpu", n_envs: int = 8):
    torch.manual_seed(0)
    np.random.seed(0)
    net = PolicyValueNet(d_model=64, n_layers=2, n_heads=2).to(device)
    # 本组测试跑在**内置占位内容**上（不加载数据目录），所以显式打开 allow_builtin。
    # 正式训练默认关闭这一项：审计 F03 要的就是"忘了传 --content 就别跑"。
    pool = CombatPool(n_envs, base_seed=1, shaping=True,
                      attempt_budget=0, selection="best", allow_builtin=True)
    batch = to_torch(pool.current_batch(), torch.device(device))
    return net, pool, batch


class TestNumericalHealth(unittest.TestCase):
    """⭐ 一次 PPO 更新之后，梯度与参数必须仍然有限。

    这条测试的价值在于它抓住了 ``-inf`` 掩码的经典坑：
    ``exp(logp) * logp`` 在屏蔽位上是 ``0 * (-inf) = NaN``，前向被 ``nan_to_num``
    掩盖（loss 看着正常），**反向却产生 NaN 梯度**，一步之后整个网络变成 NaN。
    """

    def test_masked_logits_are_finite(self):
        net, _pool, batch = make_model_and_batch()
        logits, _value = net(batch)
        self.assertTrue(torch.isfinite(logits).all().item(),
                        "屏蔽位应当是有限的大负数，不能是 -inf")
        self.assertLess(float(logits.min()), MASK_LOGIT / 2)

    def test_categorical_accepts_masked_logits(self):
        net, _pool, batch = make_model_and_batch()
        logits, _value = net(batch)
        # validate_args 会拒绝含非有限值的 logits
        dist = torch.distributions.Categorical(logits=logits)
        sample = dist.sample()
        self.assertTrue(torch.isfinite(dist.log_prob(sample)).all().item())

    def test_one_ppo_update_keeps_parameters_finite(self):
        net, pool, batch = make_model_and_batch()
        optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)

        with torch.no_grad():
            action, logp, value = net.act(batch)
        rewards, dones, _finals = pool.step(action.numpy())
        adv = torch.as_tensor(rewards - value[:, 0].numpy())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        logits, new_value = net(batch)
        log_probs = torch.log_softmax(logits, dim=-1)
        new_logp = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
        ratio = (new_logp - logp).exp()
        loss = (-torch.min(ratio * adv,
                           torch.clamp(ratio, 0.8, 1.2) * adv).mean()
                - 0.01 * entropy
                + 0.5 * torch.nn.functional.mse_loss(new_value[:, 0], adv))

        self.assertTrue(torch.isfinite(loss).item(), "loss 应当是有限值")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        for name, param in net.named_parameters():
            if param.grad is None:
                continue
            self.assertTrue(torch.isfinite(param.grad).all().item(),
                            f"{name} 的梯度出现非有限值（这就是 -inf 掩码的坑）")

        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        self.assertTrue(torch.isfinite(grad_norm).item(), "grad_norm 必须是有限值")
        optimizer.step()

        for name, param in net.named_parameters():
            self.assertTrue(torch.isfinite(param).all().item(),
                            f"更新一步后 {name} 变成非有限值")

    def test_entropy_is_finite_at_masked_positions(self):
        net, _pool, batch = make_model_and_batch()
        logits, _value = net(batch)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy_terms = log_probs.exp() * log_probs
        masked = batch["act_mask"] < 0.5
        self.assertTrue(torch.isfinite(entropy_terms[masked]).all().item(),
                        "屏蔽位的熵项必须是有限值")


class TestMaskSemantics(unittest.TestCase):
    def test_masked_actions_get_near_zero_probability(self):
        net, _pool, batch = make_model_and_batch()
        logits, _value = net(batch)
        probs = torch.softmax(logits, dim=-1)
        illegal = batch["act_mask"] < 0.5
        self.assertLess(float(probs[illegal].max()), 1e-6)
        legal_sums = probs.masked_fill(illegal, 0.0).sum(-1)
        self.assertTrue(torch.allclose(legal_sums, torch.ones_like(legal_sums),
                                       atol=1e-5))

    def test_sampled_actions_are_always_legal(self):
        net, _pool, batch = make_model_and_batch(n_envs=16)
        for _ in range(20):
            action, _logp, _value = net.act(batch)
            legal = batch["act_mask"].gather(-1, action.unsqueeze(-1)).squeeze(-1)
            self.assertTrue(bool((legal > 0.5).all()), "采样到了非法动作")

    def test_all_masked_row_is_rejected_loudly(self):
        """全掩码行会让 softmax 变成 NaN 并**静默污染整个 batch**，必须报错。"""
        net, _pool, batch = make_model_and_batch()
        broken = dict(batch)
        broken["act_mask"] = torch.zeros_like(batch["act_mask"])
        with self.assertRaises(ValueError):
            net(broken)

    def test_action_count_within_capacity(self):
        net, pool, _batch = make_model_and_batch(n_envs=16)
        for _ in range(30):
            batch = to_torch(pool.current_batch(), torch.device("cpu"))
            self.assertLessEqual(batch["act_mask"].sum(-1).max().item(), MAX_ACTIONS)
            action, _logp, _value = net.act(batch)
            pool.step(action.numpy())


class TestPolicyIsAFunctionOfObservationOnly(unittest.TestCase):
    """策略输出必须只依赖观测——训练侧的等价于 docs/01 §1.6 的判据。"""

    def test_deterministic_forward_gives_identical_logits(self):
        net, _pool, batch = make_model_and_batch()
        net.eval()
        with torch.no_grad():
            a, _va = net(batch)
            b, _vb = net(batch)
        self.assertTrue(torch.equal(a, b), "同一输入两次前向应当逐位相同")


if __name__ == "__main__":
    unittest.main(verbosity=2)
