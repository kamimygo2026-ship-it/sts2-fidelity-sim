"""PPO 训练器（战斗层）。

对应 ``docs/04`` §4.3。当前只训战斗层——它的奖励密集，是 ``docs/07`` §7.1 那份
失败实验里唯一训得动的部分。

奖励设计（``docs/04`` §4.3）：

    终局：胜 +1 / 负 -1
    塑形：r += γ·Φ(s') − Φ(s)，  Φ = 我方 HP 比例 − 敌方平均 HP 比例

势函数塑形有"不改变最优策略"的理论保证，所以可以放心用。
**明确不做**：奖励造成的总伤害（会教出不打格挡送死）、奖励剩余能量为 0（会乱出牌）。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from sts2_sim.bot import RuleBot
from sts2_sim import eligibility
from sts2_sim.content import CARD_DB, CHARACTERS, ENCOUNTERS, STARTING_DECK
from sts2_sim.core import Action
from sts2_sim.env import (
    STAGE_COMBAT, STAGE_META, SpireEnv,
)
from sts2_sim.featurize import batch_encode, encode_observation, vocab_fingerprint
from sts2_sim.observe import Observation

from .nets import PolicyValueNet

#: 训练默认角色。``docs/13`` §5：先做铁甲战士，单角色可信后再扩展。
DEFAULT_TRAIN_CHARACTER = "ironclad"


#: 训练时随机牌组可用的卡池。
#:
#: ⚠️ **必须是函数而不是模块级常量**：内容表在运行期才加载，常量会被固化在内置
#: 占位内容上。而且真机基础牌按角色命名（``strike_ironclad``），硬编码 ``"strike"``
#: 会直接 KeyError——这个坑踩过不止一次。
#:
#: ⚠️ 审计 F03：旧实现只筛"稀有度 + 有没有效果"，418 张里混进 223 张
#: ``effects_incomplete``，还会跨角色混牌。现在**统一走 `eligibility`**：
#: 内容门禁只有一处定义，采样器与报告不可能再对不上。
def draft_pool(character: str = DEFAULT_TRAIN_CHARACTER) -> tuple[str, ...]:
    return eligibility.card_pool(character)


def encounter_pool(character: str = DEFAULT_TRAIN_CHARACTER
                   ) -> tuple[tuple[str, ...], ...]:
    return eligibility.training_encounters().get("monster") or ()


# ==========================================================================
# 环境池
# ==========================================================================
def random_deck(rng: random.Random,
                character: str = DEFAULT_TRAIN_CHARACTER) -> list[str]:
    """随机牌组——**泛化的关键**。固定牌组会过拟合（``docs/04`` §4.3）。

    以**指定角色**的真实起始牌组为底，再随机加入若干该角色可抽的合格牌，
    模拟不同发育阶段的牌组。

    ⚠️ 审计 F03：旧实现从**所有角色**的起始牌组里随机挑一副，再跨角色加牌 ——
    铁甲战士的牌组里会出现静默的 ``neutralize``。那是"另一个游戏"的课程分布，
    不是自然爬塔分布。现在角色是显式参数，跨角色混牌必须**显式**要求。
    """
    deck = list(eligibility.random_deck_cards(character))
    pool = draft_pool(character)
    if pool:
        for _ in range(rng.randint(2, 6)):
            deck.append(rng.choice(pool))
    return deck


class CombatPool:
    """B 个战斗环境，同步推进。每个 episode 随机牌组 / 遭遇 / 起始 HP。"""

    def __init__(self, n_envs: int, base_seed: int = 0, shaping: bool = True,
                 attempt_budget: int = 0, selection: str = "best",
                 character: str = DEFAULT_TRAIN_CHARACTER,
                 allow_builtin: bool = False, gamma: float = 1.0) -> None:
        # ⭐ 内容门禁：池子或遭遇为空时**立刻失败**，不静默退化成内置占位内容。
        # 审计 F03：默认启动训练跑的是 12 张卡 / 3 只怪，训练"跑起来了"但训的是
        # 另一个游戏，而且没有任何提示。
        eligibility.assert_trainable(character, allow_builtin=allow_builtin)
        self.character = character
        self.n_envs = n_envs
        self.shaping = shaping
        #: 塑形用的折扣必须与 GAE 用的**同一个 γ**（审计 F12）。
        self.gamma = float(gamma)
        self.attempt_budget = attempt_budget
        self.selection = selection
        self.rng = random.Random(base_seed)
        self.episode_seed = base_seed * 1_000_003
        self.envs: list[SpireEnv] = []
        self.obs: list[Observation] = []
        self.legal: list[list] = []
        self.phi = np.zeros(n_envs, dtype=np.float32)
        self.episode_id = np.zeros(n_envs, dtype=np.int64)
        self.episode_count = 0
        self.total_attempts = 0
        self.total_restarts = 0
        self._reset_all()

    # ---- 生命周期 -----------------------------------------------------
    def _reset_all(self) -> None:
        self.envs = []
        self.obs = []
        self.legal = []
        for i in range(self.n_envs):
            self._spawn(i)

    def _spawn(self, i: int) -> None:
        self.episode_seed += 1
        env = SpireEnv(seed=self.episode_seed,
                       encounter=self.rng.choice(encounter_pool(self.character)),
                       deck=random_deck(self.rng, self.character),
                       player_hp=self.rng.randint(60, 80),
                       attempt_budget=self.attempt_budget,
                       selection=self.selection,
                       # ⭐ **角色必须传下去**（外部复核 R2）：不传的话
                       # `SpireEnv` 会退回默认角色，于是 `CombatPool(character="defect")`
                       # 里跑的是铁甲战士的战斗，"生成一张能力牌"按错的卡池取牌，
                       # 而报告里一切正常。`player_hp` 是**声明过的课程变量**
                       # （起始血量随机 60–80），与角色不是一回事。
                       character=self.character)
        obs, info = env.reset()
        if i < len(self.envs):
            self.envs[i] = env
            self.obs[i] = obs
            self.legal[i] = info["legal_actions"]
        else:
            self.envs.append(env)
            self.obs.append(obs)
            self.legal.append(info["legal_actions"])
        self.phi[i] = _potential(obs)
        self.episode_id[i] = self.episode_count
        self.episode_count += 1

    # ---- 交互 ---------------------------------------------------------
    def step(self, actions: Sequence[int]):
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.float32)
        finals: list[tuple[int, float, float]] = []       # (episode_id, 终局奖励, 结束HP比例)

        for i, choice in enumerate(actions):
            env = self.envs[i]
            action = self.legal[i][int(choice)]
            if action.kind == "restart":
                self.total_restarts += 1
            elif env.stage != STAGE_META:
                self.total_attempts += 1
            obs, reward, done, _trunc, info = env.step(action)
            phi_next = 0.0 if done else _potential(obs)
            # ⭐ 势函数塑形：``r' = r + γ·Φ(s') − Φ(s)``（Ng et al. 1999）。
            # 公式集中在一个函数里，保证塑形的 γ 与 GAE 的 γ 是同一个（审计 F12）。
            shaped = shaped_reward(reward, float(self.phi[i]), phi_next, bool(done),
                                   self.gamma, self.shaping)
            rewards[i] = shaped
            self.phi[i] = phi_next
            self.obs[i] = obs
            self.legal[i] = info["legal_actions"]
            if done:
                dones[i] = 1.0
                hp_ratio = env.raw_state.player.hp / max(1, env.raw_state.player.max_hp)
                finals.append((int(self.episode_id[i]), float(reward), float(hp_ratio)))
                self._spawn(i)
        return rewards, dones, finals

    def current_batch(self) -> dict[str, torch.Tensor]:
        return batch_encode(list(zip(self.obs, self.legal)))


def _potential(obs: Observation) -> float:
    living = [e for e in obs.enemies if e.hp > 0]
    if not living:
        return obs.player_hp / max(1, obs.player_max_hp)
    enemy = sum(e.hp / max(1, e.max_hp) for e in living) / len(living)
    return obs.player_hp / max(1, obs.player_max_hp) - enemy


def shaped_reward(reward: float, phi: float, phi_next: float, done: bool,
                  gamma: float, shaping: bool = True) -> float:
    """势函数塑形（``docs/13`` §10.4）：``r' = r + γ·Φ(s') − Φ(s)``。

    **终局的势取 0**，所以 ``done`` 时 ``Φ(s') = 0``。

    ⚠️ 审计 F12：旧实现把 ``γ`` 写死成 1（``phi_next - phi``），而 ``gamma``
    在配置里是可改的。两者一旦不同，塑形项与折扣就不是同一个 γ，
    **"势函数塑形不改变最优策略"的保证随之失效**，而且不会报错 ——
    只是训练曲线悄悄变差。所以公式单独成一个函数，两个 γ 只能从同一个参数来。
    """
    if not shaping:
        return float(reward)
    return float(reward) + gamma * (0.0 if done else phi_next) - float(phi)


# ==========================================================================
# 配置
# ==========================================================================
@dataclass
class PPOConfig:
    n_envs: int = 64
    rollout_len: int = 64
    total_updates: int = 200
    #: SL 重开预算（0 = NoSL）。上限见 sts2_sim.env.MAX_ATTEMPT_BUDGET_SIM
    attempt_budget: int = 0
    selection: str = "best"
    #: 训练角色。``docs/13`` §5：第一个交付范围是**单角色**。
    character: str = DEFAULT_TRAIN_CHARACTER
    #: 允许在**内置占位内容**上跑（仅单元测试 / 冒烟）。
    #: 默认 False —— 正式训练不允许静默退回占位内容（审计 F03）。
    allow_builtin: bool = False
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 512
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    aux_coef: float = 0.2
    max_grad_norm: float = 1.0
    shaping: bool = True
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    device: str = "auto"
    seed: int = 0
    eval_every: int = 25
    eval_episodes: int = 64
    log_every: int = 10


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ==========================================================================
# Checkpoint：**自证式**保存与加载（审计 F12）
# ==========================================================================
#: 观察协议版本 —— **从 `observe` 导入，不在这里另写一份**。
#: 两处各写一个常量必然漂移，而漂移的后果是旧权重在字段含义已变的输入上继续推理。
from sts2_sim.observe import OBSERVATION_PROTOCOL  # noqa: E402

#: SL 协议版本（``docs/13`` §9）。
SL_PROTOCOL = "sl-v1"


def checkpoint_payload(net: PolicyValueNet, config: PPOConfig,
                       optimizer=None, scheduler=None,
                       extra: dict | None = None) -> dict:
    """组装一个**可自证**的 checkpoint。

    审计 F12：旧 checkpoint 只存 ``state_dict`` 与 ``config`` —— 内容变化后
    同名 embedding 下标会指向另一张牌，加载方**无法察觉**。这里把内容、准入、
    词表与两个协议版本全部写进去，加载时逐项比对。
    """
    payload = {
        "state_dict": net.state_dict(),
        "config": vars(config) if hasattr(config, "__dict__") else dict(config),
        "content_fingerprint": eligibility.content_fingerprint(),
        "vocab_fingerprint": vocab_fingerprint(),
        "observation_protocol": OBSERVATION_PROTOCOL,
        "sl_protocol": SL_PROTOCOL,
    }
    from sts2_sim.content import CONTENT_SOURCE
    payload["content_source"] = CONTENT_SOURCE
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    #: 真正的"恢复训练"还需要训练侧随机状态（docs/13 §10.5）。
    payload["torch_rng"] = torch.get_rng_state()
    payload["numpy_rng"] = np.random.get_state()
    payload["python_rng"] = random.getstate()
    if extra:
        payload.update(extra)
    return payload


def verify_checkpoint(payload: dict, strict_content: bool = True) -> list[str]:
    """比对 checkpoint 与当前环境，返回**问题列表**（空 = 可以安全加载）。

    ``strict_content=True`` 时内容/词表不一致直接报错；不确定的调用方可以传
    ``False`` 只取警告 —— 但**不允许**静默忽略。
    """
    problems: list[str] = []
    if payload.get("observation_protocol") != OBSERVATION_PROTOCOL:
        problems.append(
            f"观察协议不匹配：checkpoint={payload.get('observation_protocol')!r} "
            f"当前={OBSERVATION_PROTOCOL!r}（Observation 字段语义已变）")
    if payload.get("sl_protocol") != SL_PROTOCOL:
        problems.append(
            f"SL 协议不匹配：checkpoint={payload.get('sl_protocol')!r} "
            f"当前={SL_PROTOCOL!r}")
    current_content = eligibility.content_fingerprint()
    if payload.get("content_fingerprint") != current_content:
        problems.append(
            f"内容指纹不匹配：checkpoint={payload.get('content_fingerprint')} "
            f"当前={current_content}")
    current_vocab = vocab_fingerprint()
    if payload.get("vocab_fingerprint") != current_vocab:
        problems.append(
            f"词表指纹不匹配：checkpoint={payload.get('vocab_fingerprint')} "
            f"当前={current_vocab}（embedding 下标可能张冠李戴）")
    if strict_content and problems:
        raise ValueError("checkpoint 与当前环境不兼容：\n  " + "\n  ".join(problems))
    return problems


def load_checkpoint(path, device: torch.device | str = "cpu",
                    strict_content: bool = True) -> dict:
    """加载 checkpoint 并**校验指纹**，返回原始 payload。

    调用方负责用 ``config`` 重建网络；本函数只保证"这些权重与当前内容对得上"。
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    verify_checkpoint(payload, strict_content=strict_content)
    return payload


def to_torch(batch: dict[str, np.ndarray], device: torch.device
             ) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device)
            for k, v in batch.items() if k != "n_actions"}


# ==========================================================================
# 评估
# ==========================================================================
def make_env(seed: int, rng: random.Random, attempt_budget: int = 0,
             character: str = DEFAULT_TRAIN_CHARACTER) -> SpireEnv:
    return SpireEnv(seed=seed, encounter=rng.choice(encounter_pool(character)),
                    deck=random_deck(rng, character),
                    player_hp=rng.randint(60, 80),
                    attempt_budget=attempt_budget,
                    # ⭐ 评估入口同样要把角色传下去（理由见 `CombatPool._spawn`）。
                    character=character)


def baseline_meta_action(info: dict) -> Action:
    """规则 bot 的 SL 元策略：**输了就重开，否则接受**（docs/04 §4.5b 的 (a) 方案）。"""
    kinds = {a.kind for a in info["legal_actions"]}
    if info["phase"] == "lost" and "restart" in kinds:
        return Action("restart")
    return Action("accept")


def _run_episode(env: SpireEnv, choose) -> dict:
    obs, info = env.reset()
    steps = 0
    while info["stage"] != "done" and steps < 4000:
        action = choose(obs, info)
        obs, _reward, done, _t, info = env.step(action)
        steps += 1
        if done:
            break
    return {
        "won": info["phase"] == "won",
        "hp_left": env.raw_state.player.hp,
        "turns": env.raw_state.turn,
        "attempts_used": env.attempts_used,
        "best_score": info.get("best_score") or 0.0,
    }


#: 环境故障的分类（``docs/13`` §7：未知机制要**单独计数并保存复现**）。
#: 分类的意义在于：`unknown_mechanic` 是"该实现但还没实现"（要补内容），
#: `data_gap` 是"内容表缺数据"（要修抽取器），`bug` 才是引擎写错了。
#: 三者混成一类时，一个数据缺口看起来就像引擎坏了。
FAULT_UNKNOWN_MECHANIC = "unknown_mechanic"
FAULT_DATA_GAP = "data_gap"
FAULT_BUG = "bug"


def classify_fault(exc: BaseException) -> str:
    """把异常归到三类之一。**按类型名判断，不 import 引擎**（避免耦合）。"""
    name = type(exc).__name__
    if name in ("UnsupportedMechanic", "UnknownCondition"):
        return FAULT_UNKNOWN_MECHANIC
    if isinstance(exc, (KeyError, IndexError)):
        return FAULT_DATA_GAP
    return FAULT_BUG


def _run_episode_guarded(env: SpireEnv, choose, seed: int,
                         faults: list[dict]) -> dict | None:
    """跑一局，**把环境故障与策略输局分开计数**（``docs/13`` §12.3）。

    审计 F12：旧实现只有"胜率"，环境异常（未支持的机制、协议错误、数据缺口）
    会和"策略打输了"混在同一个分母里 —— 于是一个**环境 bug 看起来像策略变差**。
    真机对拍期尤其危险：未知机制必须为零。所以这里捕获异常、单独计数，
    并把复现所需的种子、异常类型与文本留下来。
    """
    try:
        return _run_episode(env, choose)
    except Exception as exc:                          # noqa: BLE001
        faults.append({
            "seed": seed,
            "category": classify_fault(exc),
            "error": f"{type(exc).__name__}: {exc}",
        })
        return None


def evaluate(net: PolicyValueNet, device: torch.device,
             episodes: int = 64, seed_base: int = 900_000,
             attempt_budget: int = 0,
             character: str = DEFAULT_TRAIN_CHARACTER) -> dict:
    """固定种子集上的贪心评估。评估种子集与训练种子集不相交（docs/01 §1.3）。"""
    rng = random.Random(seed_base)
    envs = [make_env(seed_base + i, rng, attempt_budget, character)
            for i in range(episodes)]
    net.eval()

    def choose(obs, info):
        batch = to_torch(encode_observation(obs, info["legal_actions"]), device)
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        index, _logp, _value = net.act(batch, deterministic=True)
        return info["legal_actions"][int(index)]

    faults: list[dict] = []
    results = [r for r in (_run_episode_guarded(env, choose, seed_base + i, faults)
                           for i, env in enumerate(envs)) if r is not None]
    net.train()
    return _summarize(results, faults)


def evaluate_rulebot(episodes: int = 64, seed_base: int = 900_000,
                     attempt_budget: int = 0,
                     character: str = DEFAULT_TRAIN_CHARACTER) -> dict:
    """同一个评估集上的规则 bot（回归基线）。"""
    bot = RuleBot()
    rng = random.Random(seed_base)
    envs = [make_env(seed_base + i, rng, attempt_budget, character)
            for i in range(episodes)]

    def choose(obs, info):
        if info["stage"] == STAGE_META:
            return baseline_meta_action(info)
        return bot.act(obs, info["legal_actions"])

    faults: list[dict] = []
    results = [r for r in (_run_episode_guarded(env, choose, seed_base + i, faults)
                           for i, env in enumerate(envs)) if r is not None]
    return _summarize(results, faults)


def _summarize(results: list[dict], faults: list[dict] | None = None) -> dict:
    # ⚠️ 分母只算**成功跑完**的局：环境故障既不算赢也不算输（docs/13 §12.3）。
    n = max(1, len(results))
    faults = faults or []
    by_category: dict[str, int] = {}
    for fault in faults:
        category = fault.get("category", FAULT_BUG)
        by_category[category] = by_category.get(category, 0) + 1
    return {
        "win_rate": sum(r["won"] for r in results) / n,
        "avg_hp_left": sum(r["hp_left"] for r in results) / n,
        "avg_turns": sum(r["turns"] for r in results) / n,
        "avg_attempts": sum(r["attempts_used"] for r in results) / n,
        "n_episodes": len(results),
        "n_faults": len(faults),
        "faults_by_category": by_category,
        "fault_examples": faults[:5],
    }


def _summarize_with_faults(results: list[dict], faults: list[dict]) -> dict:
    return _summarize([r for r in results if r is not None], faults)


# ==========================================================================
# 训练
# ==========================================================================
def train(config: PPOConfig, verbose: bool = True) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)

    # ⭐ 内容门禁 + 版本指纹。审计 F12：checkpoint 只存权重与配置，内容变了以后
    # 同名 embedding 下标可能指向另一张牌 —— 必须把内容与词表一起钉死。
    eligibility.assert_trainable(config.character,
                                 allow_builtin=config.allow_builtin)
    fingerprint = eligibility.content_fingerprint()
    vocab = vocab_fingerprint()

    net = PolicyValueNet(config.d_model, config.n_layers, config.n_heads).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.total_updates), eta_min=config.lr * 0.1)

    pool = CombatPool(config.n_envs, base_seed=config.seed + 1,
                      shaping=config.shaping,
                      attempt_budget=config.attempt_budget,
                      selection=config.selection,
                      character=config.character,
                      allow_builtin=config.allow_builtin,
                      gamma=config.gamma)
    steps = config.n_envs * config.rollout_len
    history: list[dict] = []

    if verbose:
        from sts2_sim.content import CONTENT_SOURCE

        mode = (f"SL(K={config.attempt_budget},{config.selection})"
                if config.attempt_budget else "NoSL(K=0)")
        print(f"内容 {CONTENT_SOURCE} | 角色 {config.character} | "
              f"可抽卡池 {len(draft_pool(config.character))} 张 | "
              f"合格遭遇 {len(encounter_pool(config.character))} 组")
        print(f"内容指纹 {fingerprint['content_hash']} | "
              f"准入指纹 {fingerprint['admission_hash']} | "
              f"词表 {vocab['vocab_hash']}")
        print(f"设备 {device} | 参数 {net.n_parameters():,} | {mode} | "
              f"每轮 {steps:,} 步 × {config.total_updates} 轮 "
              f"= {steps * config.total_updates:,} 步")
        baseline = evaluate_rulebot(config.eval_episodes,
                                    attempt_budget=config.attempt_budget,
                                    character=config.character)
        print(f"[基线] 规则 bot：胜率 {baseline['win_rate']:.1%} "
              f"| 平均剩余 HP {baseline['avg_hp_left']:.1f} "
              f"| 平均回合 {baseline['avg_turns']:.1f} "
              f"| 平均重开 {baseline['avg_attempts']:.2f} "
              f"| 环境故障 {baseline.get('n_faults', 0)}")
        if baseline.get("fault_examples"):
            for fault in baseline["fault_examples"][:3]:
                print(f"      ⚠️ seed={fault['seed']} {fault['error']}")
        history.append({"update": -1, "tag": "rulebot", **baseline})

    started = time.perf_counter()
    for update in range(config.total_updates):
        # ---- 采样 -----------------------------------------------------
        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf, epid_buf = [], [], []
        episode_final: dict[int, float] = {}

        for _ in range(config.rollout_len):
            batch_np = pool.current_batch()
            batch = to_torch(batch_np, device)
            with torch.no_grad():
                action, logp, value = net.act(batch)
            obs_buf.append(batch_np)
            act_buf.append(action.cpu().numpy())
            logp_buf.append(logp.cpu().numpy())
            val_buf.append(value.cpu().numpy())
            epid_buf.append(pool.episode_id.copy())
            rewards, dones, finals = pool.step(action.cpu().numpy())
            rew_buf.append(rewards)
            done_buf.append(dones)
            for ep_id, _reward, hp_ratio in finals:
                episode_final[ep_id] = hp_ratio

        # ---- 自举 + GAE ----------------------------------------------
        with torch.no_grad():
            _a, _lp, last_value = net.act(to_torch(pool.current_batch(), device))
        values = np.stack(val_buf)                       # (T, B, 2)
        rewards = np.stack(rew_buf)                      # (T, B)
        dones = np.stack(done_buf)                       # (T, B)
        episode_ids = np.stack(epid_buf)                 # (T, B)

        advantages = np.zeros_like(rewards)
        last_gae = np.zeros(config.n_envs, dtype=np.float32)
        next_value = last_value[:, 0].cpu().numpy()
        for t in reversed(range(config.rollout_len)):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + config.gamma * next_value * not_done - values[t, :, 0]
            last_gae = delta + config.gamma * config.gae_lambda * not_done * last_gae
            advantages[t] = last_gae
            next_value = values[t, :, 0]
        returns = advantages + values[:, :, 0]

        # ---- 辅助任务标签：预测本局结束时的剩余 HP 比例 ----------------
        aux_target = np.full((config.rollout_len, config.n_envs), np.nan,
                             dtype=np.float32)
        aux_mask = np.zeros_like(aux_target)
        for t in range(config.rollout_len):
            for i in range(config.n_envs):
                ep_id = int(episode_ids[t, i])
                if ep_id in episode_final:
                    aux_target[t, i] = episode_final[ep_id]
                    aux_mask[t, i] = 1.0

        # ---- 展开成扁平 batch ----------------------------------------
        flat = {key: np.concatenate([obs_buf[t][key] for t in range(config.rollout_len)])
                for key in obs_buf[0] if key != "n_actions"}
        actions_flat = np.stack(act_buf).reshape(-1)
        logp_flat = np.stack(logp_buf).reshape(-1)
        adv_flat = advantages.reshape(-1)
        ret_flat = returns.reshape(-1)
        aux_flat = aux_target.reshape(-1)
        auxmask_flat = aux_mask.reshape(-1)

        adv_t = torch.as_tensor(adv_flat, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        actions_t = torch.as_tensor(actions_flat, device=device)
        logp_t = torch.as_tensor(logp_flat, device=device)
        ret_t = torch.as_tensor(ret_flat, device=device)
        aux_t = torch.as_tensor(np.nan_to_num(aux_flat, nan=0.0), device=device)
        auxmask_t = torch.as_tensor(auxmask_flat, device=device)

        total = actions_flat.shape[0]
        indices = np.arange(total)
        stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "aux": 0.0, "n": 0}
        for _epoch in range(config.epochs):
            np.random.shuffle(indices)
            for start in range(0, total, config.minibatch):
                chunk = indices[start:start + config.minibatch]
                mb = {k: torch.as_tensor(v[chunk], device=device) for k, v in flat.items()}
                logits, value = net(mb)
                log_probs = torch.log_softmax(logits, dim=-1)
                new_logp = log_probs.gather(-1, actions_t[chunk].unsqueeze(-1)).squeeze(-1)
                entropy = -(log_probs.exp() * log_probs).nan_to_num(0.0).sum(-1).mean()

                ratio = (new_logp - logp_t[chunk]).exp()
                unclipped = ratio * adv_t[chunk]
                clipped = torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * adv_t[chunk]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(value[:, 0], ret_t[chunk])
                aux_error = (value[:, 1] - aux_t[chunk]) ** 2
                aux_loss = (aux_error * auxmask_t[chunk]).sum() / (auxmask_t[chunk].sum() + 1e-8)
                loss = (policy_loss + config.value_coef * value_loss
                        - config.entropy_coef * entropy + config.aux_coef * aux_loss)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
                optimizer.step()

                stats["policy"] += float(policy_loss)
                stats["value"] += float(value_loss)
                stats["entropy"] += float(entropy)
                stats["aux"] += float(aux_loss)
                stats["n"] += 1
        scheduler.step()

        if verbose and (update % config.log_every == 0 or update == config.total_updates - 1):
            n = max(1, stats["n"])
            elapsed = time.perf_counter() - started
            done_steps = (update + 1) * steps
            print(f"[{update:4d}] 步 {done_steps:,} ({done_steps / elapsed:,.0f}/s) "
                  f"| π {stats['policy'] / n:+.4f} V {stats['value'] / n:.4f} "
                  f"H {stats['entropy'] / n:.3f} aux {stats['aux'] / n:.4f} "
                  f"| 重开 {pool.total_restarts:,}")

        if config.eval_every and (update + 1) % config.eval_every == 0:
            result = evaluate(net, device, config.eval_episodes,
                              attempt_budget=config.attempt_budget,
                              character=config.character)
            history.append({"update": update, **result})
            if verbose:
                print(f"      [评估] 胜率 {result['win_rate']:.1%} "
                      f"| 平均剩余 HP {result['avg_hp_left']:.1f} "
                      f"| 平均回合 {result['avg_turns']:.1f} "
                      f"| 平均重开 {result['avg_attempts']:.2f} "
                      f"| 环境故障 {result.get('n_faults', 0)}")

    final = evaluate(net, device, config.eval_episodes * 2,
                     attempt_budget=config.attempt_budget,
                     character=config.character)
    history.append({"update": config.total_updates, "tag": "final", **final})
    if verbose:
        print(f"\n[最终] 胜率 {final['win_rate']:.1%} | "
              f"平均剩余 HP {final['avg_hp_left']:.1f} | "
              f"平均重开 {final['avg_attempts']:.2f} | "
              f"环境故障 {final.get('n_faults', 0)}")
    return {"net": net, "history": history, "config": config,
            "content_fingerprint": fingerprint, "vocab_fingerprint": vocab}
