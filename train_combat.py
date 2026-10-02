"""战斗层 PPO 训练入口。

    python train_combat.py --content data/content/repo --updates 200 --n-envs 64
    python train_combat.py --smoke --allow-builtin   # 只验证管线，不产数据

产出：``checkpoints/combat_ppo.pt`` 与 ``checkpoints/combat_ppo.json``（训练曲线）。

⚠️ **内容必须显式指定**（审计 F03）：以前这里没有任何内容加载参数，
``_autoload_from_env()`` 定义了却没被调用，于是默认训练跑的是 12 张卡 /
3 只怪的内置占位内容 —— 训练"跑起来了"，但训练的是另一个游戏，且没有提示。
现在：``--content`` > 环境变量 ``STS2_CONTENT_DIR`` > 什么都不给就**报错退出**。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sts2_rl.ppo import PPOConfig, evaluate, evaluate_rulebot, train  # noqa: E402

CHECKPOINT_DIR = pathlib.Path("checkpoints")
DEFAULT_CONTENT_DIR = "data/content/repo"


def resolve_content(explicit: str | None, allow_builtin: bool) -> str | None:
    """决定从哪加载内容。返回 ``None`` = 用内置占位内容（仅测试/冒烟）。"""
    import os

    from sts2_sim.content import CONTENT_DIR_ENV

    path = explicit or os.environ.get(CONTENT_DIR_ENV)
    if not path and pathlib.Path(DEFAULT_CONTENT_DIR).is_dir():
        path = DEFAULT_CONTENT_DIR          # 本地默认目录，仍会打印实际路径
    if path:
        return path
    if allow_builtin:
        return None
    raise SystemExit(
        f"未指定内容目录，且 {DEFAULT_CONTENT_DIR} 不存在。\n"
        f"请用 --content DIR 或设置 {CONTENT_DIR_ENV}。"
        f"（只为跑通管线可加 --allow-builtin，但产出的数据没有训练意义）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="战斗层 PPO 训练")
    parser.add_argument("--content", default=None,
                        help=f"内容目录（默认 {DEFAULT_CONTENT_DIR}）")
    parser.add_argument("--allow-builtin", action="store_true",
                        help="允许在内置占位内容上跑（仅冒烟；产出的数据无意义）")
    parser.add_argument("--character", default="ironclad",
                        help="训练角色（docs/13 §5：第一版单角色）")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--rollout-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--no-shaping", action="store_true",
                        help="关掉势函数塑形（消融实验）")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sl-budget", type=int, default=0,
                        help="SL 重开预算（0=NoSL；模拟器上限 50，真机 ≤10）")
    parser.add_argument("--selection", default="best", choices=("best", "last"),
                        help="预算耗尽/接受时保留哪次试次的结果")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--out", default="combat_ppo")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟测试：少量轮次，验证管线能跑通并学习")
    args = parser.parse_args(argv)

    # ⭐ **先加载内容、再建词表与网络**：卡牌 embedding 按词表下标索引，
    # 顺序反了会让模型"张冠李戴"且不报错（docs/06 §6.4）。
    from sts2_sim.featurize import configure_content

    content_path = resolve_content(args.content, args.allow_builtin)
    if content_path:
        info = configure_content(content_path)
        print(f"[内容] {content_path} | 卡牌 {info.get('cards')} | "
              f"怪物 {info.get('monsters')} | 遭遇 {info.get('encounters')} | "
              f"词表 {info.get('n_cards')}")
    else:
        print("[内容] ⚠️ 内置占位内容（12 张卡 / 3 只怪）—— 仅供冒烟，不可用于训练")

    if args.smoke:
        args.updates = 24
        args.n_envs = 32
        args.rollout_len = 32
        args.eval_every = 8

    config = PPOConfig(
        n_envs=args.n_envs, rollout_len=args.rollout_len,
        total_updates=args.updates, lr=args.lr, entropy_coef=args.entropy,
        shaping=not args.no_shaping, device=args.device, seed=args.seed,
        d_model=args.d_model, n_layers=args.layers,
        attempt_budget=args.sl_budget, selection=args.selection,
        eval_episodes=args.eval_episodes, eval_every=args.eval_every,
        character=args.character, allow_builtin=args.allow_builtin,
    )

    started = time.perf_counter()
    result = train(config)
    elapsed = time.perf_counter() - started

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    model_path = CHECKPOINT_DIR / f"{args.out}.pt"
    # ⭐ checkpoint 必须带**内容/词表/协议指纹**（审计 F12）：
    # 内容变化后同名 embedding 下标可能指向另一张牌，只存权重等于埋了一个
    # 不会报错的错配。加载侧用 `load_checkpoint()` 校验。
    from sts2_rl.ppo import checkpoint_payload

    torch.save(checkpoint_payload(result["net"], config), model_path)

    curve_path = CHECKPOINT_DIR / f"{args.out}.json"
    curve_path.write_text(json.dumps({
        "config": vars(config),
        "content_source": content_path or "builtin",
        "content_fingerprint": result["content_fingerprint"],
        "vocab_fingerprint": result["vocab_fingerprint"],
        "history": result["history"],
        "elapsed_seconds": elapsed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n耗时 {elapsed:.1f}s | 模型 → {model_path} | 曲线 → {curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
