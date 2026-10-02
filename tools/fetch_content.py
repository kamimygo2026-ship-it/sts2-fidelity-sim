"""从社区数据源抓取《杀戮尖塔 2》内容数据（``docs/06`` 的 fetch 阶段）。

⚠️ **必须在你自己有网络的终端里运行**——agent 的 shell 没有出网权限。

用法
----
    python tools/fetch_content.py list          # 列出可用端点
    python tools/fetch_content.py export        # 首选：一次拿全量 ZIP
    python tools/fetch_content.py endpoints     # 备用：逐个端点抓 JSON
    python tools/fetch_content.py probe monsters  # 抓一条样本看 schema

产出落在 ``data/raw/spire-codex/<版本>/``，并写一份 ``manifest.json``
（来源 URL、抓取时间、sha256、字节数）——见 ``docs/06`` §6.3：
**原始数据必须原样归档**，将来对拍失败时才能回答"是我们解析错了，还是源头就错了"。

数据来源：https://spire-codex.com/api/  （无需 key，60 请求/分钟/IP）
许可证：PolyForm Noncommercial 1.0.0（仅限非商用，见 docs/05 风险 14）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://spire-codex.com/api"
RAW_ROOT = Path("data/raw/spire-codex")
#: 60 req/min/IP → 保守地每次请求间隔 1.2 秒
MIN_INTERVAL = 1.2

#: ⚠️ **必须带浏览器 UA**：服务端会拒绝默认的 `Python-urllib/3.x`（返回 403）。
#: 那个 403 是 HTTP 层错误——看起来像"没网"，其实只是被挡了 UA。
#:
#: ⚠️ 另一条更隐蔽的坑：**本机的 Schannel 系客户端（curl / Invoke-WebRequest / git）
#: 取不到 TLS 凭证**（SEC_E_NO_CREDENTIALS），所以"用 curl 试试"会得出"完全没网"的
#: 错误结论。Python 与 Node 用自带的 OpenSSL，**不受影响**。排查网络问题时别只测一种。
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

#: 逐端点抓取时用的端点（`docs/06` §6.9）
ENDPOINTS: tuple[str, ...] = (
    "cards", "relics", "monsters", "encounters", "events",
    "potions", "powers", "keywords", "characters", "ancients",
)

_last_request = 0.0


def _throttle() -> None:
    global _last_request
    wait = MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def get(path: str, *, timeout: float = 60.0, binary: bool = False,
        retries: int = 3) -> bytes:
    url = path if path.startswith("http") else f"{BASE}/{path.lstrip('/')}"
    last_error: Exception | None = None
    for attempt in range(retries):
        _throttle()
        try:
            request = urllib.request.Request(
                url, headers={"Accept": "*/*", "User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            print(f"  重试 {attempt + 1}/{retries}：{exc}")
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"无法获取 {url}：{last_error}")


def _write(out_dir: Path, name: str, payload: bytes) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_bytes(payload)
    return {
        "file": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def cmd_list(_args) -> int:
    print("首选（一次拿全量，不受 60/min 限制影响）：")
    print("  export          GET /exports/zhs  或  /exports/eng  → ZIP")
    print("\n备用端点：")
    for name in ENDPOINTS:
        print(f"  {name:14s} GET /api/{name}")
    print("\n其它：")
    print("  runs/shared/{hash}   GET /api/runs/shared/{hash}   # 社区 run 日志")
    return 0


def cmd_export(args) -> int:
    out_dir = RAW_ROOT / args.version
    name = f"export_{args.lang}.zip"
    print(f"下载全量导出（{args.lang}）…")
    payload = get(f"exports/{args.lang}", timeout=300.0, binary=True)
    info = _write(out_dir, name, payload)
    print(f"  {info['bytes']:,} 字节 → {info['file']}")
    _update_manifest(out_dir, {"exports": {args.lang: info}})
    print("\n下一步：解压后运行 importers 转换成我们的内容表（docs/06 §6.3）。")
    return 0


def cmd_endpoints(args) -> int:
    out_dir = RAW_ROOT / args.version
    recorded: dict[str, dict] = {}
    for name in ENDPOINTS:
        print(f"抓取 /api/{name} …")
        try:
            payload = get(name)
        except RuntimeError as exc:
            print(f"  跳过：{exc}")
            continue
        try:
            parsed = json.loads(payload)
            count = len(parsed) if isinstance(parsed, (list, dict)) else "?"
        except json.JSONDecodeError:
            count = "?"
        info = _write(out_dir, f"{name}.json", payload)
        recorded[name] = {**info, "records": count}
        print(f"  记录数 {count}，{info['bytes']:,} 字节")
    _update_manifest(out_dir, {"endpoints": recorded})
    return 0


def cmd_probe(args) -> int:
    """抓一条样本看 schema —— 构建 importer 之前先看清楚字段长什么样。"""
    print(f"抓取 /api/{args.endpoint} …")
    payload = get(args.endpoint)
    data = json.loads(payload)
    if isinstance(data, dict):
        for key in ("data", "items", "results"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if isinstance(data, list):
        print(f"共 {len(data)} 条，打印第 1 条：")
        sample = data[0]
    else:
        sample = data
    print(json.dumps(sample, ensure_ascii=False, indent=2)[:4000])
    return 0


def _update_manifest(out_dir: Path, section: dict) -> None:
    path = out_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest.setdefault("source", BASE)
    manifest.setdefault("license", "PolyForm Noncommercial 1.0.0")
    manifest["fetched_at"] = datetime.now(timezone.utc).isoformat()
    for key, value in section.items():
        manifest.setdefault(key, {}).update(value)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest → {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="抓取社区内容数据")
    parser.add_argument("--version", default="unversioned",
                        help="游戏版本号，用于目录命名（docs/06 §6.3 的版本锁定）")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    export = sub.add_parser("export")
    export.add_argument("--lang", default="zhs")
    sub.add_parser("endpoints")
    probe = sub.add_parser("probe")
    probe.add_argument("endpoint", help="例如 monsters / cards / encounters")
    args = parser.parse_args(argv)

    handlers = {"list": cmd_list, "export": cmd_export,
                "endpoints": cmd_endpoints, "probe": cmd_probe}
    try:
        return handlers[args.command](args)
    except RuntimeError as exc:
        print(f"\n失败：{exc}")
        print("检查网络连通性，或稍后重试（服务端限流 60 请求/分钟/IP）。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
