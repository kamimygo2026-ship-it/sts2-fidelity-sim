"""把 ``web_fetch`` 落盘的临时文件整理成标准原始数据（``docs/06`` 的 fetch 阶段）。

**为什么需要这个脚本**：agent 的 shell 没有出网权限，只能通过 ``web_fetch`` 抓数据；
它的完整响应会落在系统临时目录里。这个脚本把那些文件剥掉包装文本、校验 JSON、
写进 ``data/raw/spire-codex/<版本>/``，并更新 manifest。

在有网络的机器上应当优先用 ``tools/fetch_content.py``（一次拿全量 ZIP）。

    python tools/absorb_spill.py <落盘文件> monsters [--version 0.105.0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

RAW_ROOT = pathlib.Path("data/raw/spire-codex")

#: web_fetch 会在正文前加一段说明，剥掉它
PREFIX_MARKERS = ("External web content follows.", "\n\n")


def extract_json(text: str):
    """从包装文本里取出 JSON 正文。"""
    start_candidates = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not start_candidates:
        raise ValueError("文件里找不到 JSON 起始字符")
    start = min(start_candidates)
    body = text[start:]
    # 从后往前找最后一个闭合符号，容忍尾部的说明文字
    for end in range(len(body), 0, -1):
        chunk = body[:end].rstrip()
        if not chunk or chunk[-1] not in "]}":
            continue
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue
    raise ValueError("无法解析出完整 JSON（可能是被截断了）")


def salvage_array(text: str) -> tuple[list, int]:
    """从**被截断**的 JSON 数组里抢救出所有完整记录。

    web_fetch 的落盘有 100KB 上限，大端点（如 /api/monsters）会被硬截断。
    与其丢掉整份数据，不如保住已经完整的那些记录——但**绝不修补半条记录**：
    半个怪物比没有怪物危险得多。

    返回 ``(记录列表, 丢弃的尾巴长度)``。
    """
    start = text.find("[")
    if start < 0:
        raise ValueError("找不到 JSON 数组起始")
    depth = 0
    in_string = False
    escaped = False
    record_start = -1
    records: list[str] = []
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                record_start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and record_start >= 0:
                records.append(text[record_start:index + 1])
                record_start = -1
        elif char == "]" and depth == 0:
            break
    parsed = []
    for chunk in records:
        try:
            parsed.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    return parsed, len(text) - (start + sum(len(r) for r in records))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="整理 web_fetch 落盘的数据")
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("dataset", help="cards / monsters / encounters / relics ...")
    parser.add_argument("--version", default="unversioned")
    parser.add_argument("--repair", action="store_true",
                        help="响应被截断时，抢救完整记录（丢弃残缺的尾巴）")
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"找不到 {args.source}")
        return 3
    text = args.source.read_text(encoding="utf-8", errors="replace")
    salvaged = False
    try:
        payload = extract_json(text)
    except ValueError:
        if not args.repair:
            print(f"解析失败：JSON 不完整（文件 {len(text):,} 字符）")
            print("重试时加 --repair 以抢救完整记录。")
            return 2
        records, dropped = salvage_array(text)
        if not records:
            print("抢救失败：没有找到任何完整记录。")
            return 2
        payload = records
        salvaged = True
        print(f"⚠️ 响应被截断，已抢救 {len(records)} 条完整记录"
              f"（丢弃尾巴约 {dropped:,} 字符，未修补任何半条记录）")

    records = payload if isinstance(payload, list) else [payload]
    out_dir = RAW_ROOT / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(records, ensure_ascii=False).encode("utf-8")
    path = out_dir / f"{args.dataset}.json"
    path.write_bytes(body)

    manifest_path = out_dir / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {})
    manifest.setdefault("source", "https://spire-codex.com/api")
    manifest.setdefault("license", "PolyForm Noncommercial 1.0.0")
    manifest.setdefault("method", "web_fetch via agent（shell 无出网）")
    manifest["fetched_at"] = datetime.now(timezone.utc).isoformat()
    manifest.setdefault("datasets", {})[args.dataset] = {
        "file": str(path),
        "records": len(records),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "partial": salvaged,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    flag = "（**不完整**，仅含被截断前已完整的记录）" if salvaged else ""
    print(f"{args.dataset}: {len(records)} 条 → {path}（{len(body):,} 字节）{flag}")
    if records and isinstance(records[0], dict):
        print("首条记录的字段：" + ", ".join(sorted(records[0])[:20]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
