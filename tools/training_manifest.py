"""训练配置清单（``docs/13`` §7 / 工单 T00）。

**为什么必须有这个工具**：``docs/13`` §7 要求训练配置明确写出
"游戏构建、源码哈希、内容哈希、角色、解锁、进阶、幕范围、SL 协议、
允许内容和课程分布"。少了它，一个 checkpoint 无法回答
"我是在哪一版源码与哪一份内容上训出来的"—— 而内容一变，
卡牌 embedding 的下标就可能指向另一张牌（审计 F12），**而且不会报错**。

用法
----
    python -X utf8 tools/training_manifest.py                     # 生成并打印
    python -X utf8 tools/training_manifest.py --character silent
    python -X utf8 tools/training_manifest.py --check FILE        # 校验（CI 用）

产出写到 ``data/manifests/<内容哈希>.json``；同一份内容重复生成是幂等的
（``created_at`` 之外逐字段相同）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"
SOURCE = ROOT / "data" / "decompiled" / "sts2"
MANIFEST_DIR = ROOT / "data" / "manifests"

#: 课程标签（``docs/13`` §7：允许用受限内容做课程，但必须明确标注）
CURRICULUM_RESTRICTED = "restricted-simulation-curriculum"
CURRICULUM_FULL = "full-declared-scope"


def _sha256_files(root: Path, patterns: tuple[str, ...],
                  limit_suffix: tuple[str, ...] = ()) -> tuple[str, int]:
    """对目录下匹配的文件做**内容**哈希（相对路径 + 内容），返回 ``(hash, 文件数)``。

    相对路径也进哈希：否则"重命名一个文件"不会改变哈希，
    而重命名恰好是"内容清单换了"的一种。
    """
    digest = hashlib.sha256()
    count = 0
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    for path in sorted(set(files)):
        if limit_suffix and path.suffix not in limit_suffix:
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        count += 1
    return digest.hexdigest(), count


def _game_build() -> dict:
    """从反编译目录里读出程序集标识（不是"用户装的 Steam 构建号"）。

    ``AssemblyInfo.cs`` 里是 ``AssemblyInformationalVersion``；
    它**只能**用来说明"我读的是哪一份源码"，不能当成玩家的游戏版本
    （审计 F01 的说明里已经点过这一点）。
    """
    info = SOURCE / "Properties" / "AssemblyInfo.cs"
    csproj = SOURCE / "sts2.csproj"
    out: dict = {"assembly_info": None, "target_framework": None}
    if info.exists():
        import re
        text = info.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r'AssemblyInformationalVersion\s*\(\s*"([^"]+)"', text)
        if match:
            out["assembly_info"] = match.group(1)
    if csproj.exists():
        import re
        text = csproj.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"<TargetFramework>([^<]+)<", text)
        if match:
            out["target_framework"] = match.group(1)
    return out


def _tool_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import numpy
        out["numpy"] = numpy.__version__
    except Exception:                                     # noqa: BLE001
        out["numpy"] = None
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:                                     # noqa: BLE001
        out["torch"] = None
        out["cuda_available"] = None
    return out


def build_manifest(character: str = "ironclad", ascension: int = 0,
                   act_range: str = "act1",
                   sl_protocol: str = "sl-v1") -> dict:
    from sts2_sim import eligibility
    from sts2_sim import featurize
    from sts2_sim.content import CONTENT_SOURCE, ENCOUNTERS, CARD_DB
    from sts2_sim.env import MAX_ATTEMPT_BUDGET_SIM
    from sts2_sim.observe import OBSERVATION_PROTOCOL

    content_hash, content_files = _sha256_files(CONTENT, ("*.json",))
    source_hash, source_files = _sha256_files(SOURCE, ("*.cs",),
                                              limit_suffix=(".cs",))
    # 只哈希引擎自己实现的算子：**内核文件**也是"机制的唯一真相"的一部分。
    engine_files = ("core.py", "powers.py", "monster_ai.py", "hooks.py",
                    "orbs.py", "relics.py", "runeffects.py", "events.py",
                    "keywords.py", "mapgen.py", "rng.py", "content.py",
                    "observe.py", "featurize.py", "eligibility.py", "run.py",
                    "env.py", "relicbag.py", "acts.py", "pointodds.py")
    engine_digest = hashlib.sha256()
    for name in sorted(engine_files):
        path = ROOT / "sts2_sim" / name
        if path.exists():
            engine_digest.update(name.encode("utf-8"))
            engine_digest.update(hashlib.sha256(path.read_bytes()).digest())

    pool = eligibility.card_pool(character)
    admitted = eligibility.admitted_cards()
    by_rarity: dict[str, int] = {}
    for cid in pool:
        rarity = CARD_DB[cid].rarity
        by_rarity[rarity] = by_rarity.get(rarity, 0) + 1
    encounters = eligibility.training_encounters()

    fingerprint = eligibility.content_fingerprint()
    vocab = featurize.vocab_fingerprint()

    # 受限课程的判据：采样池覆盖率明显低于内容全量，或事件池不完整。
    # 这里**不写死结论**，只给出数字，让读的人自己判断（审计 F11）。
    total_pool_cards = sum(
        1 for cid in CARD_DB
        if CARD_DB[cid].rarity in eligibility.DRAFTABLE_RARITIES)
    coverage = len(pool) / max(1, total_pool_cards)
    curriculum = (CURRICULUM_FULL if coverage > 0.95 and not _event_gaps()
                  else CURRICULUM_RESTRICTED)

    return {
        "schema": "sts2-training-manifest/1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "game_build": _game_build(),
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "hash": source_hash,
            "files": source_files,
        },
        "content": {
            "path": str(CONTENT.relative_to(ROOT)),
            "loaded_from": CONTENT_SOURCE,
            "files_hash": content_hash,
            "files": content_files,
            "content_fingerprint": fingerprint,
            "vocab_fingerprint": vocab,
        },
        "engine": {
            "hash": engine_digest.hexdigest(),
            "files": list(engine_files),
        },
        "protocol": {
            "observation": OBSERVATION_PROTOCOL,
            "sl": sl_protocol,
        },
        "scope": {
            "character": character,
            "ascension": ascension,
            "act_range": act_range,
            "sl_budget_max_sim": MAX_ATTEMPT_BUDGET_SIM,
            "unlocks": "unknown",           # 待用户提供（docs/13 §15）
            "curriculum": curriculum,
            "curriculum_note": (
                "受限模拟课程：允许内容少于目标配置的全部可达内容，"
                "因此**不能**用它声称完成了真实分布（docs/13 §7）"
                if curriculum == CURRICULUM_RESTRICTED else
                "声明范围内内容齐全"),
        },
        "allowed_content": {
            "draft_pool": len(pool),
            "draft_pool_by_rarity": by_rarity,
            "draftable_cards_in_content": total_pool_cards,
            "draft_pool_coverage": round(coverage, 4),
            "cards_admitted": len(admitted),
            "encounters": {k: len(v) for k, v in encounters.items()},
            "encounters_total": {k: len(v) for k, v in ENCOUNTERS.items()},
        },
        "reject_reasons": eligibility.report(character)["reject_reasons"],
        "unknown_mechanic_policy": (
            "未知机制作为**环境故障**单独计数并保存复现，不当作玩家死亡/胜利/空房；"
            "最终真实分布评测要求未知机制为零（docs/13 §7）"),
        "tools": _tool_versions(),
    }


def _event_gaps() -> bool:
    """事件池是否还有缺口（有缺口就不算"完整范围"）。"""
    from sts2_sim import content
    usable = sum(1 for d in content.EVENT_DB.values() if d.usable)
    return usable < len(content.EVENT_DB)


def manifest_id(manifest: dict) -> str:
    """清单的**文件名**：把内容文件哈希 / 内容指纹 / 词表 / 引擎 / 协议一起算进去。

    ⚠️ 只用``content_fingerprint``当文件名是**错的**：它只覆盖卡牌与怪物，
    于是"只改了药水/事件/遗物"的两份内容会算出同一个文件名 ——
    后一份把前一份**覆盖**掉，而两份清单说的根本不是同一套内容。
    """
    parts = [
        manifest["content"]["files_hash"],
        str(manifest["content"]["content_fingerprint"]["content_hash"]),
        str(manifest["content"]["vocab_fingerprint"]["vocab_hash"]),
        manifest["engine"]["hash"],
        manifest["protocol"]["observation"],
        manifest["protocol"]["sl"],
        manifest["source"]["hash"],
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8"))
    return digest.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成训练配置清单")
    parser.add_argument("--character", default="ironclad")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--act-range", default="act1")
    parser.add_argument("--sl-protocol", default="sl-v1")
    parser.add_argument("--out", default=None,
                        help="输出路径（默认 data/manifests/<内容哈希>.json）")
    parser.add_argument("--check", default=None,
                        help="校验已有清单与当前环境是否一致")
    args = parser.parse_args(argv)

    from sts2_sim.featurize import configure_content

    if args.check:
        path = Path(args.check)
        saved = json.loads(path.read_text(encoding="utf-8"))
        configure_content(str(CONTENT))
        current = build_manifest(args.character, args.ascension,
                                 args.act_range, args.sl_protocol)
        problems = []
        for key in ("source", "content", "engine", "protocol", "scope",
                    "allowed_content"):
            if saved.get(key) != current.get(key):
                problems.append(key)
        if problems:
            print("❌ 清单与当前环境不一致，字段：" + "、".join(problems))
            for key in problems:
                print(f"  {key}:")
                print(f"    清单 = {json.dumps(saved.get(key), ensure_ascii=False)}")
                print(f"    当前 = {json.dumps(current.get(key), ensure_ascii=False)}")
            return 1
        print(f"✅ 清单 {path.name} 与当前源码/内容/引擎/协议完全一致")
        return 0

    configure_content(str(CONTENT))
    manifest = build_manifest(args.character, args.ascension, args.act_range,
                              args.sl_protocol)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{manifest_id(manifest)}.json"
    path = Path(args.out) if args.out else MANIFEST_DIR / name
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8")

    scope = manifest["scope"]
    content = manifest["content"]
    print("=" * 68)
    print("训练配置清单")
    print("=" * 68)
    print(f"  程序集        {manifest['game_build']['assembly_info']}")
    print(f"  源码哈希      {manifest['source']['hash'][:16]}… "
          f"（{manifest['source']['files']} 个 .cs）")
    print(f"  内容文件哈希  {content['files_hash'][:16]}… "
          f"（{content['files']} 个 json）")
    print(f"  内容指纹      {content['content_fingerprint']['content_hash']}")
    print(f"  准入门禁      {content['content_fingerprint']['admission_hash']}")
    print(f"  词表指纹      {content['vocab_fingerprint']['vocab_hash']}")
    print(f"  引擎哈希      {manifest['engine']['hash'][:16]}… "
          f"（{len(manifest['engine']['files'])} 个模块）")
    print(f"  协议          观察 {manifest['protocol']['observation']} / "
          f"SL {manifest['protocol']['sl']}")
    print(f"  范围          角色 {scope['character']} · 进阶 {scope['ascension']} · "
          f"{scope['act_range']}")
    print(f"  课程          {scope['curriculum']}")
    print(f"  可抽池        {manifest['allowed_content']['draft_pool']} 张"
          f"（可抽内容共 {manifest['allowed_content']['draftable_cards_in_content']} 张，"
          f"覆盖率 {manifest['allowed_content']['draft_pool_coverage']:.1%}）")
    print(f"  合格遭遇      {manifest['allowed_content']['encounters']}"
          f"（总 {manifest['allowed_content']['encounters_total']}）")
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
