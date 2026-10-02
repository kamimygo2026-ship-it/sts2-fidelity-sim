"""训练侧：向量化战斗环境池 + PPO + 评估。

对应 ``docs/02`` §2.7 目录结构里的 ``sts2-rl/``。特征化（``sts2_sim.featurize``）
留在模拟器包里，因为它是"观测契约"的一部分，要和模拟器的测试一起跑。
"""

from .nets import EntityEncoder, PolicyValueNet
from .ppo import (
    CombatPool, OBSERVATION_PROTOCOL, PPOConfig, SL_PROTOCOL,
    checkpoint_payload, draft_pool, encounter_pool, evaluate, evaluate_rulebot,
    load_checkpoint, random_deck, shaped_reward, train, verify_checkpoint,
)

__all__ = [
    "EntityEncoder", "PolicyValueNet", "CombatPool", "PPOConfig",
    "evaluate", "evaluate_rulebot", "random_deck", "train",
    "draft_pool", "encounter_pool", "shaped_reward",
    "checkpoint_payload", "load_checkpoint", "verify_checkpoint",
    "OBSERVATION_PROTOCOL", "SL_PROTOCOL",
]
