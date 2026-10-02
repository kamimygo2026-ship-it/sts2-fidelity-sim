"""审计：引擎**读到**的能力 id 与**注册了规则**的能力 id 是否对得上。

为什么值得单独一个工具（``docs/12`` 里那一类"静默错"）：

* **读到一个不存在的 id** = 拼写错。它不会报错，只是永远取到 0 —— 那一条逻辑
  **静默失效**（例如写成 ``power("focuss")``，"专注"就再也不加成了）。
* **读到了内容里存在的能力、但 ``powers.RULES`` 里没有它** = 这个能力**已经有行为**
  （行为写在消费者那一侧，例如 ``orbs._focused`` 读 ``focus``），可是门禁
  （``eligibility``）按 ``RULES`` 判断"能力未实现"，于是**所有施加它的卡都被拒**。
  这是反向的错：能力能用，卡却进不来。

判据只有一条：消费者已经读了它，就说明它的行为在引擎里存在，必须登记进 ``RULES``
并把源码出处写清楚；否则要么修拼写，要么补登记。

用法::

    python -X utf8 tools/audit_power_refs.py [--check]

``--check`` 有输出就以非零码退出（给 CI / 测试用）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 引擎里"取某个能力的层数"的所有写法。
READ_PATTERN = re.compile(
    r"\.(?:power|add_power|has_power)\(\s*[\"']([a-z0-9_]+)[\"']")


def read_power_ids(directory: Path) -> dict[str, list[str]]:
    """``{能力 id: [读它的模块…]}``。"""
    found: dict[str, list[str]] = {}
    for path in sorted(directory.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in READ_PATTERN.finditer(text):
            names = found.setdefault(match.group(1), [])
            if path.name not in names:
                names.append(path.name)
    return found


def known_power_ids() -> set[str]:
    """内容里存在的能力 id（以**源码抽取**为准）。"""
    records = json.loads((ROOT / "data" / "content" / "repo" / "powers_source.json")
                         .read_text(encoding="utf-8"))
    return {str(r["id"]) for r in records}


def audit() -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """返回 ``(拼写错的, 该登记但没登记的)``。"""
    from sts2_sim import powers

    reads = read_power_ids(ROOT / "sts2_sim")
    known = known_power_ids()
    typos = sorted((pid, mods) for pid, mods in reads.items() if pid not in known)
    unregistered = sorted((pid, mods) for pid, mods in reads.items()
                          if pid in known and pid not in powers.RULES)
    return typos, unregistered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="有输出就以非零码退出")
    args = parser.parse_args()

    typos, unregistered = audit()
    from sts2_sim import powers

    print(f"引擎读到 {len(read_power_ids(ROOT / 'sts2_sim'))} 个能力 id；"
          f"RULES 已登记 {len(powers.RULES)} 个")

    if typos:
        print(f"\n【拼写错】读到内容里不存在的 id（{len(typos)} 个）—— 该分支静默失效：")
        for pid, mods in typos:
            print(f"    {pid}  ← {', '.join(mods)}")
    if unregistered:
        print(f"\n【漏登记】行为已存在、但 RULES 里没有（{len(unregistered)} 个）"
              "—— 门禁会把所有施加它的卡拒掉：")
        for pid, mods in unregistered:
            print(f"    {pid}  ← {', '.join(mods)}")

    if not typos and not unregistered:
        print("\n全部一致 ✅（读到的一定登记过，登记的 id 一定存在）")
        return 0
    if args.check:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
