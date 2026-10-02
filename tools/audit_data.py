"""数据审计：查重、对账、按磁盘实况重建 manifest。

这个工具是"先把导入的数据筛一遍"的落地。它回答三个问题，**每一条都能复现**：

  1. **有没有重复导入**？同一份数据被多个来源写进同一张表
     （实测踩过：encounters 87 条变成 261 条 = 87×3）。
  2. **有没有被静默丢掉的东西**？raw 里 115 只怪、归一化后 114 —— 少的那个
     是 `TestSubject`，一只真 Boss，只因为 codex 的 HP 是 null 就被整条丢掉。
  3. **manifest 说的是不是实话**？它曾经把**同一个路径**记了两遍、两条的
     records / bytes / sha256 全不一样（一条来自导出包、一条来自裸端点，
     后写的覆盖了前写的），还引用了一个根本不存在的 `export_eng.zip`。

用法
----
    python tools/audit_data.py            # 审计 + 重建 manifest
    python tools/audit_data.py --check    # 只审计，失败时退出码非 0（给 CI 用）

⚠️ 关于"重复"的判定：**按 key 查重是不够的**。cards/monsters/encounters 的主键
字段名分别是 `cid` / `eid` / `eid`，其余表才叫 `id`。一律用 `record["id"]` 查重
会得到 `id唯一=1`（全是 None）这种"零重复"的假象 —— 实测就是这么被骗过一次。
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RAW_ROOT = Path("data/raw/spire-codex")
CONTENT_ROOT = Path("data/content")

#: 各数据集的主键字段名。**不能假设都叫 `id`**。
PRIMARY_KEY: dict[str, str] = {
    "cards": "cid",
    "monsters": "eid",
    "encounters": "eid",
}
DEFAULT_KEY = "id"

#: 项目自己的元数据，不是游戏内容。
IGNORED = frozenset({"manifest"})

#: 这些内容表**不来自 codex**，而是从反编译源码抽取的（``tools/extract_monsters.py``）。
#: 拿它们去和 raw 对账会误报"raw 里没有对应数据集"。
#: ``acts_source`` 来自 ``tools/extract_acts.py``（幕/遭遇池/地图点计数），
#: codex 的 ``encounters.json`` 是**另一个东西**（按遭遇记怪物组成），不能对账。
SOURCE_DERIVED = frozenset({"monster_ai", "cards_source", "cards_source_audit",
                            "powers_source", "potions_source", "card_pools",
                            "relics_source", "monster_moves", "events_source",
                            "acts_source"})


def key_field(dataset: str) -> str:
    return PRIMARY_KEY.get(dataset, DEFAULT_KEY)


def norm_name(text: Any) -> str:
    """名字归一化，用于发现"同一张卡以不同 id 进了两次"。

    ⚠️ 归一化后会**故意**出现重复，这些是合法的，必须白名单掉：
    5 个角色的 ``Strike``/``Defend`` 是各自的独立卡（``strike_ironclad``…），
    ``FAKE_*`` 遗物是真遗物（``Anchor`` vs ``Anchor???``），
    ``Temporary Strength`` 是不同 power 类共用显示名。
    """
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SystemExit(f"{path} 不是合法 JSON：{exc}") from exc


def as_records(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for name in ("data", "items", "results", "records", "entries"):
            value = payload.get(name)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    return []


def datasets_in(path: Path) -> dict[str, list[dict]]:
    """把一个 .json 或 .zip 里的内容读成 ``{数据集: [记录]}``。"""
    found: dict[str, list[dict]] = {}
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if not member.lower().endswith(".json"):
                    continue
                try:
                    payload = json.loads(archive.read(member).decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                name = Path(member).stem.lower()
                records = as_records(payload)
                if records and name not in IGNORED:
                    found.setdefault(name, []).extend(records)
        return found
    if path.suffix == ".json":
        name = path.stem.lower()
        if name in IGNORED:
            return {}
        records = as_records(load_json(path))
        if records:
            found[name] = records
    return found


# ==========================================================================
# 检查项
# ==========================================================================
def find_duplicate_keys(dataset: str, records: list[dict]) -> dict[str, int]:
    """按主键查重。返回 ``{重复的 key: 次数}``。"""
    field = key_field(dataset)
    counts = collections.Counter(
        str(r.get(field)) for r in records if r.get(field) is not None)
    return {k: v for k, v in counts.items() if v > 1}


def find_duplicate_names(dataset: str, records: list[dict]) -> dict[str, list[str]]:
    """按名字查重，列出同名的所有 id（调用方需自行判断是否合法）。"""
    field = key_field(dataset)
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for record in records:
        grouped[norm_name(record.get("name"))].append(str(record.get(field)))
    return {k: v for k, v in grouped.items() if len(v) > 1 and k}


def audit_raw(raw_dir: Path) -> tuple[dict[str, dict], list[str]]:
    """审计原始目录：每个数据集只应有一个**权威**来源。"""
    problems: list[str] = []
    plain = sorted(p for p in raw_dir.glob("*.json") if p.stem.lower() not in IGNORED)
    archives = sorted(raw_dir.glob("*.zip"))

    report: dict[str, dict] = {}
    # 同名数据集既在裸端点又在导出包里 → 两份来源，导入时必须二选一。
    in_plain: set[str] = set()
    for path in plain:
        for name, records in datasets_in(path).items():
            in_plain.add(name)
            report[name] = {
                "authoritative": str(path), "source_kind": "endpoint",
                "records": len(records), "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
                "duplicates": find_duplicate_keys(name, records),
            }
            duplicates = find_duplicate_keys(name, records)
            if duplicates:
                problems.append(f"{path.name}: 主键重复 {list(duplicates)[:5]}")

    for path in archives:
        for name, records in datasets_in(path).items():
            if name in in_plain:
                continue        # 裸端点优先，导出包只补缺
            report[name] = {
                "authoritative": f"{path}::{name}.json", "source_kind": "archive",
                "records": len(records), "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
                "duplicates": find_duplicate_keys(name, records),
            }
            duplicates = find_duplicate_keys(name, records)
            if duplicates:
                problems.append(f"{path.name}::{name}: 主键重复 {list(duplicates)[:5]}")
    return report, problems


def audit_content(content_dir: Path, raw_report: dict[str, dict]) -> list[str]:
    """审计归一化结果：条数必须与原始来源对得上（**一个都不许少**）。"""
    problems: list[str] = []
    for path in sorted(content_dir.glob("*.json")):
        name = path.stem.lower()
        if name in IGNORED or name == "import_report":
            continue
        records = as_records(load_json(path))
        duplicates = find_duplicate_keys(name, records)
        if duplicates:
            problems.append(f"content/{path.name}: 主键重复 {list(duplicates)[:5]}")
        if name.endswith("_provisional"):
            continue
        if name in SOURCE_DERIVED:
            continue        # 源码抽取的表，不与 codex 对账
        raw = raw_report.get(name)
        if raw is None:
            problems.append(f"content/{path.name}: raw 里没有对应数据集")
        elif raw["records"] != len(records):
            problems.append(
                f"content/{path.name}: {len(records)} 条，raw 有 {raw['records']} 条"
                f"（差 {raw['records'] - len(records)}）")
    return problems


def game_version() -> dict:
    """读安装目录的 ``release_info.json``（游戏自己的版本戳）。

    有了它才能回答"社区数据是哪个版本的" —— API 侧**没有任何版本或日期信息**
    （无 `Last-Modified` / `ETag`，Cloudflare 只回请求时刻的 `Date`），
    只能靠**数值比对**反推。
    """
    path = Path(r"D:\Program Files (x86)\Steam\steamapps\common"
                r"\Slay the Spire 2\release_info.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def rebuild_manifest(raw_dir: Path, raw_report: dict[str, dict]) -> dict:
    """按**磁盘实况**重建 manifest：一个路径只留一条记录。

    旧 manifest 把同一个 `monsters.json` 记了两遍（一条 111 条/50KB、
    一条 115 条/183KB，sha256 不同），因为导出包与裸端点**写同一个路径**、
    各自追加了一条。任何"按路径索引"的读者都会随机拿到错的那条。

    ``fetched_at`` 与 ``codex_lag`` 是**必须留的溯源信息**：社区数据没有版本号，
    只能靠与源码比对来判断"它对应哪个游戏版本"（实测 codex 的卡牌数值落后
    游戏至少一个补丁 —— 见 ``codex_lag``）。
    """
    datasets = {}
    for name in sorted(raw_report):
        info = raw_report[name]
        datasets[name] = {
            "dataset": name,
            "source_kind": info["source_kind"],
            "authoritative": info["authoritative"],
            "records": info["records"],
            "duplicates": info["duplicates"],
        }
    files = {}
    for path in sorted(list(raw_dir.glob("*.json")) + list(raw_dir.glob("*.zip"))):
        if path.stem.lower() in IGNORED:
            continue
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
            "language": "zhs" if path.suffix == ".zip" else "eng",
            "fetched_at": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
    version = game_version()
    return {
        "source": "https://spire-codex.com/api",
        "license": "PolyForm Noncommercial 1.0.0",
        "note": (
            "由 tools/audit_data.py 按磁盘实况重建。裸端点（英文）是权威来源，"
            "导出包（中文，export_zhs.zip）只用于补裸端点没有的数据集；"
            "卡牌效果解析器吃英文描述，不要让中文覆盖英文。"
        ),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "codex_has_no_version": (
            "codex API 不提供版本号或数据日期（无 Last-Modified / ETag）。"
            "判断它对应哪个游戏版本只能靠与反编译源码比对数值。"
        ),
        "game_version": {
            "version": version.get("version"),
            "commit": version.get("commit"),
            "built_at": version.get("date"),
        },
        "codex_lag": (
            "实测分歧 37/651 个变量（5.7%）。已由 v0.110.0 补丁说明证实："
            "说明里的旧值 = codex 的值，新值 = 源码的值（Mangle 15(20)→20(26)、"
            "Biased Cognition 4(5)→5(6)、Relax 15(18)→16(18)…）。"
            "结论：**codex 的卡牌数值至少落后一个补丁**，以源码为准。"
        ),
        "files": files,
        "datasets": datasets,
    }


def cross_check_source_vs_codex(content_dir: Path) -> tuple[list[str], dict]:
    """**反向验证**：把源码抽取的结果与社区库对账。

    两个来源互相独立（一边是反编译的 C#，一边是社区整理的数据库），
    对不上就说明至少有一边错了 —— 这是目前手上最强的一致性证据。

    覆盖：怪物是否一一对应、HP 是否一致、每只怪是否都有出招状态机。
    """
    problems: list[str] = []
    stats: dict = {}
    monsters_path = content_dir / "monsters.json"
    ai_path = content_dir / "monster_ai.json"
    if not (monsters_path.exists() and ai_path.exists()):
        return problems, stats

    codex = {str(m.get("eid")): m for m in as_records(load_json(monsters_path))}
    ai = {str(a.get("eid")): a for a in as_records(load_json(ai_path))}
    stats["monsters_codex"] = len(codex)
    stats["monsters_from_source"] = len(ai)

    only_codex = sorted(set(codex) - set(ai))
    if only_codex:
        # `.Mocks` 与测试怪是正常的（源码里有、codex 里没有），反过来才是问题
        problems.append(f"codex 有但源码没抽出状态机：{only_codex}")

    missing_hp = sorted(eid for eid, a in ai.items()
                        if not a.get("hp") or a["hp"][0] is None)
    if missing_hp:
        problems.append(f"源码没给出 HP：{missing_hp}")

    mismatched: list[str] = []
    for eid, record in codex.items():
        source = ai.get(eid)
        if not source or not source.get("hp") or source["hp"][0] is None:
            continue
        if record.get("hp") is None:
            continue
        lo, hi = source["hp"]
        hi = lo if hi is None else hi
        if [lo, hi] != list(record["hp"]):
            mismatched.append(f"{eid}: 源码={(lo, hi)} codex={record['hp']}")
    stats["hp_checked"] = sum(
        1 for eid, r in codex.items()
        if r.get("hp") and (ai.get(eid) or {}).get("hp")
        and ai[eid]["hp"][0] is not None)
    if mismatched:
        problems.append(f"源码与 codex 的 HP 有 {len(mismatched)} 处不一致："
                        f"{mismatched[:5]}")
    return problems, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="数据审计与 manifest 重建")
    parser.add_argument("--check", action="store_true",
                        help="只审计，不写 manifest；有问题时退出码 2")
    parser.add_argument("--raw", default=str(RAW_ROOT / "repo"))
    parser.add_argument("--content", default=str(CONTENT_ROOT / "repo"))
    args = parser.parse_args(argv)

    raw_dir, content_dir = Path(args.raw), Path(args.content)
    if not raw_dir.exists():
        print(f"没有原始数据目录：{raw_dir}")
        return 3

    raw_report, problems = audit_raw(raw_dir)

    print(f"原始目录 {raw_dir}")
    print(f"{'数据集':16s} {'来源':9s} {'条数':>6s}  权威文件")
    for name in sorted(raw_report):
        info = raw_report[name]
        print(f"  {name:14s} {info['source_kind']:9s} {info['records']:6d}  "
              f"{info['authoritative']}")
    from_archive = [n for n, i in raw_report.items() if i["source_kind"] == "archive"]
    if from_archive:
        print(f"\n⚠️ 这些数据集只有中文导出包可用：{from_archive}")
        print("   英文版可以从 https://spire-codex.com/api/<名字> 取（裸端点）。")

    if content_dir.exists():
        problems += audit_content(content_dir, raw_report)
        cross_problems, stats = cross_check_source_vs_codex(content_dir)
        problems += cross_problems
        if stats:
            print(f"\n源码 vs 社区库对账："
                  f"codex {stats['monsters_codex']} 只 / 源码 {stats['monsters_from_source']} 只"
                  f"，HP 逐条比对 {stats['hp_checked']} 条")
    else:
        print(f"\n（跳过归一化结果：{content_dir} 不存在）")

    print()
    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for line in problems:
            print(f"  ❌ {line}")
    else:
        print("✅ 无重复、无静默丢失：每个数据集的条数都与来源对得上")

    if not args.check:
        manifest = rebuild_manifest(raw_dir, raw_report)
        target = raw_dir / "manifest.json"
        target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(f"\nmanifest 已按磁盘实况重建 → {target}")

    return 2 if (problems and args.check) else 0


if __name__ == "__main__":
    sys.exit(main())
