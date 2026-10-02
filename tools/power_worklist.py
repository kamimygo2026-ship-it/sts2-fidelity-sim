"""诊断：敌人在源码里给自己/玩家上了哪些能力（buff / debuff）。

按**出现位置**分类，因为这决定实现方式：

* ``AfterAddedToRoom`` / ``BeforeCombatStart`` —— 入场即带（固有属性）
* 某个 ``XxxMove`` 处理函数体内 —— 招式效果
* ``AfterDeath`` 等 —— 死亡触发

输出按"多少只怪在用"排序：这就是实现优先级的**数据驱动工单**
（``docs/09`` §5.5：覆盖率必须可数，缺口要落到具体对象）。
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
MONSTERS = ROOT / "data" / "decompiled" / "sts2" / "MegaCrit.Sts2.Core.Models.Monsters"
AI = ROOT / "data" / "content" / "repo" / "monster_ai.json"

APPLY = re.compile(r"PowerCmd\.Apply<(\w+?)Power\s*>")
#: 怪物类里的方法：`private async Task XxxMove(...)` / `public override async Task AfterAddedToRoom()`
METHOD = re.compile(r"(?:private|public|protected|internal)[\w\s]*?"
                    r"(?:async\s+)?(?:override\s+)?[\w<>?\[\],\s]+?\s+(\w+)\s*\(")


def sections(source: str) -> dict[str, str]:
    """粗切出每个方法的体（大括号配平）。够用来做归属判断。"""
    out: dict[str, str] = {}
    for match in re.finditer(r"\n\t(?:public|private|protected|internal)[^\n]*\n\t\{",
                            source):
        header = match.group(0)
        name = METHOD.search(header)
        if not name:
            continue
        start = match.end() - 1
        depth = 0
        for index in range(start, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    out[name.group(1)] = source[start:index]
                    break
    return out


def snake(name: str) -> str:
    name = name[:-5] if name.endswith("Power") else name
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def main() -> int:
    from sts2_sim.powers import IMPLEMENTED

    innate: collections.Counter = collections.Counter()
    moves: collections.Counter = collections.Counter()
    others: collections.Counter = collections.Counter()
    users: dict[str, set[str]] = collections.defaultdict(set)

    for path in sorted(MONSTERS.glob("*.cs")):
        source = path.read_text(encoding="utf-8", errors="replace")
        eid = snake(path.stem)
        for method, body in sections(source).items():
            for power in APPLY.findall(body):
                pid = snake(power)
                users[pid].add(eid)
                if method in ("AfterAddedToRoom", "BeforeCombatStart"):
                    innate[pid] += 1
                elif method.endswith("Move"):
                    moves[pid] += 1
                else:
                    others[pid] += 1

    # ⚠️ `innate_powers` 在 **monsters.json** 里（社区库提供的固有属性表），
    # 不在 `monster_ai.json`（那是出招状态机）。读错文件会让对账显示
    # "源码有、抽取结果没有" 覆盖全部条目 —— 一个看起来像重大缺失的假警报。
    monsters = json.loads(
        (ROOT / "data" / "content" / "repo" / "monsters.json").read_text(encoding="utf-8"))
    declared: collections.Counter = collections.Counter()
    for monster in monsters:
        for entry in monster.get("innate_powers") or []:
            pid = str(entry.get("power") or "").lower()
            if pid:
                declared[pid] += 1

    print(f"=== 敌方能力（共 {len(users)} 种）===")
    print(f"{'能力':32s} {'已实现':6s} {'固有':4s} {'招式':4s} {'其他':4s} {'怪数':4s}")
    rows = sorted(users, key=lambda p: (-len(users[p]), p))
    for pid in rows:
        flag = "✅" if pid in IMPLEMENTED else "❌"
        print(f"{pid:32s} {flag:6s} {innate[pid]:4d} {moves[pid]:4d} "
              f"{others[pid]:4d} {len(users[pid]):4d}")

    missing = [p for p in rows if p not in IMPLEMENTED]
    print(f"\n未实现：{len(missing)} 种")
    print("按受影响怪物数排序（这就是工单）：")
    for pid in missing:
        print(f"  {len(users[pid]):3d} 只  {pid}"
              f"   固有{innate[pid]} 招式{moves[pid]} 其他{others[pid]}")

    # 与已抽取的 innate_powers 对账：抽取器有没有漏
    print("\n=== 抽取器 innate_powers 与源码扫描对账 ===")
    only_source = {p: n for p, n in innate.items() if declared.get(p, 0) == 0}
    only_declared = {p: n for p, n in declared.items() if innate.get(p, 0) == 0}
    print(f"  源码有、抽取结果没有：{only_source}")
    print(f"  抽取结果有、源码没扫到：{only_declared}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
