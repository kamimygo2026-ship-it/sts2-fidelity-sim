"""真机实验：StS2 的 save-load 是否保留洗牌结果？

这是**整个项目形式化基础的那个实验**（``docs/01`` §1.8 问题 1、§1.3 前提）。
结论决定 SL 目标是否成立：

  * 若保留 → 重开面对同一副牌序 → ``docs/01`` §1.4 的 meta-episode 成立
  * 若重掷 → "重开刷牌序"是另一回事 → ``docs/01``/``docs/04`` 必须改写

依赖：游戏运行中 + STS2MCP 类桥接 mod（localhost REST，默认端口 15526）。
本脚本只用标准库，无需安装任何东西。

⚠️ **本脚本尚未在真机上验证过**（作者环境无游戏本体）。接口形状按
   ``Gennadiyev/STS2MCP`` 的 README 与 ``docs/raw-full.md`` 编写，若字段不符
   请用 ``dump`` 先把原始 JSON 打出来看。

用法
----
    # 0. 自检（不需要游戏，验证比对逻辑）
    python tools/saveload_probe.py selftest

    # 1. 进入战斗后（任意回合都可以，但**不要**在战斗中途改变操作）
    python tools/saveload_probe.py dump before.json

    # 2. 在游戏里：保存并退出 → 继续游戏 → 回到同一场战斗
    #    （不要做任何其他操作）

    # 3. 再抓一次
    python tools/saveload_probe.py dump after.json

    # 4. 比对
    python tools/saveload_probe.py compare before.json after.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://localhost:15526/api/v1/singleplayer"

#: 与"牌序"直接相关的关键词——比对时单独高亮
SHUFFLE_KEYWORDS = ("draw", "pile", "hand", "deck", "shuffle", "discard",
                    "rng", "seed", "random")


def fetch_state(url: str, timeout: float = 5.0) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


# ==========================================================================
# 通用 JSON 差异（不知道 schema 也能用）
# ==========================================================================
def flatten(node: object, prefix: str = "") -> dict[str, object]:
    """把嵌套 JSON 压成 {路径: 叶子值}。"""
    flat: dict[str, object] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            flat.update(flatten(value, f"{prefix}[{index}]"))
    else:
        flat[prefix] = node
    return flat


def diff_states(before: object, after: object, limit: int = 40) -> dict[str, list]:
    flat_a, flat_b = flatten(before), flatten(after)
    keys = sorted(set(flat_a) | set(flat_b))
    same: list[str] = []
    changed: list[tuple[str, object, object]] = []
    for key in keys:
        a, b = flat_a.get(key, "<缺失>"), flat_b.get(key, "<缺失>")
        if a == b:
            same.append(key)
        else:
            changed.append((key, a, b))
    shuffle_related = [
        item for item in changed
        if any(word in item[0].lower() for word in SHUFFLE_KEYWORDS)
    ]
    return {
        "same": same,
        "changed": changed[:limit],
        "changed_total": len(changed),
        "shuffle_related": shuffle_related,
    }


def report(before: object, after: object) -> int:
    result = diff_states(before, after)
    print(f"相同字段 {len(result['same'])} 个；"
          f"不同字段 {result['changed_total']} 个\n")

    if result["changed"]:
        print("--- 差异（最多 40 条）---")
        for key, a, b in result["changed"]:
            print(f"  {key}\n      before = {a!r}\n      after  = {b!r}")
        print()

    if result["shuffle_related"]:
        print("--- ⚠️ 与牌序相关的差异 ---")
        for key, a, b in result["shuffle_related"]:
            print(f"  {key}: {a!r} → {b!r}")
        print()

    print("=" * 68)
    if result["changed_total"] == 0:
        print("结论：读档后状态**逐字段完全一致**。")
        print("  → save-load 保留了随机数状态。")
        print("  → ✅ SL 形式化成立：重开面对同一副牌序（docs/01 §1.4）。")
        return 0
    if result["shuffle_related"]:
        print("结论：读档后**牌序相关字段发生变化**。")
        print("  → ⚠️ StS2 可能在读档时重掷了随机数。")
        print("  → 这会让 SL 变成另一种博弈（重开刷牌序），")
        print("     docs/01 §1.4 与 docs/04 §4.5b 必须改写。")
        print("  → 建议：重复 3 次实验确认不是操作误差（比如读档回到了不同回合）。")
        return 2
    print("结论：有字段差异，但**与牌序无关**。")
    print("  → 可能是 UI/时间戳类字段（如 elapsed_time）。")
    print("  → 请检查上面的差异列表，确认没有语义字段（回合数、HP、手牌）在变。")
    print("  → 若回合数/HP/手牌不同，说明读档回到了不同的时点，实验不成立，需重做。")
    return 1


# ==========================================================================
# 自检
# ==========================================================================
def selftest() -> int:
    base = {
        "battle": {
            "turn": 2,
            "player": {"hp": 71, "max_hp": 80, "energy": 3},
            "draw_pile": [{"id": "CARD.STRIKE"}, {"id": "CARD.DEFEND"}],
            "hand": [{"id": "CARD.BASH"}],
        }
    }
    same = json.loads(json.dumps(base))
    rerolled = json.loads(json.dumps(base))
    rerolled["battle"]["draw_pile"] = [{"id": "CARD.DEFEND"}, {"id": "CARD.STRIKE"}]
    cosmetic = json.loads(json.dumps(base))
    cosmetic["elapsed_time"] = 1234

    failures = []
    if diff_states(base, same)["changed_total"] != 0:
        failures.append("完全相同的状态被判为不同")
    if not diff_states(base, rerolled)["shuffle_related"]:
        failures.append("牌序变化未被识别")
    dr = diff_states(base, cosmetic)
    if dr["changed_total"] != 1 or dr["shuffle_related"]:
        failures.append("纯外观字段差异被误判为牌序差异")

    if failures:
        print("selftest 失败：")
        for item in failures:
            print("  -", item)
        return 1
    print("selftest 通过：比对逻辑能区分「牌序变化」「纯外观变化」「无变化」三种情况。")
    return 0


# ==========================================================================
# CLI
# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="StS2 save-load 语义探针")
    parser.add_argument("--url", default=DEFAULT_URL, help="桥接 REST 端点")
    sub = parser.add_subparsers(dest="command", required=True)

    dump = sub.add_parser("dump", help="抓取当前状态并写入文件")
    dump.add_argument("out", type=Path)

    cmp_ = sub.add_parser("compare", help="比对两次抓取")
    cmp_.add_argument("before", type=Path)
    cmp_.add_argument("after", type=Path)

    sub.add_parser("selftest", help="自检比对逻辑（不需要游戏）")

    args = parser.parse_args(argv)

    if args.command == "selftest":
        return selftest()

    if args.command == "dump":
        try:
            state = fetch_state(args.url)
        except urllib.error.URLError as exc:
            print(f"无法连接 {args.url}\n  {exc}\n")
            print("检查清单：")
            print("  1. 游戏是否正在运行？")
            print("  2. 桥接 mod 是否已加载、且已进入单人对局？")
            print("  3. 端口是否为 15526？（用 netstat -ano | findstr 15526 确认）")
            return 3
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"已写入 {args.out}（{len(flatten(state))} 个叶子字段）")
        return 0

    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    return report(before, after)


if __name__ == "__main__":
    sys.exit(main())
