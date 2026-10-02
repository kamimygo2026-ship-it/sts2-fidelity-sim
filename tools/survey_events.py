"""事件源码侦察：先量清楚 66 个事件各需要什么，再动手。

只做统计，不产出数据 —— 目的是回答"该先建什么子系统"。
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.extract_cards import (  # noqa: E402
    COMMAND_CALL, balanced, collect_locals, method_body, override_method_bodies)

SRC = Path("data/decompiled/sts2/MegaCrit.Sts2.Core.Models.Events")


def option_calls(source: str) -> list[tuple[str, str]]:
    """``[(选项 textKey, onChosen 表达式)]``。"""
    out = []
    for match in re.finditer(r"new EventOption\s*\(", source):
        args = balanced(source, match.end() - 1)[0]
        parts = [p.strip() for p in re.split(r",(?![^()<>]*[>)]\s*$)", args)]
        # 取最后一个字符串字面量作为 textKey（构造参数顺序有几种）
        keys = re.findall(r'"([^"]+)"', args)
        key = keys[-1] if keys else "?"
        # onChosen 通常是第 2 个参数
        chosen = parts[1] if len(parts) > 1 else "?"
        out.append((key, chosen))
    return out


def main() -> int:
    files = sorted(SRC.glob("*.cs"))
    print(f"事件源文件：{len(files)}")
    stats: collections.Counter = collections.Counter()
    commands: collections.Counter = collections.Counter()
    choice_events: list[str] = []
    combat_events: list[str] = []
    multi_stage: list[str] = []
    gated: list[str] = []
    rng_events: list[str] = []
    total_options = 0
    method_options = 0

    for path in files:
        source = path.read_text(encoding="utf-8", errors="replace")
        name = path.stem
        stats["total"] += 1
        if "EnterCombatWithoutExitingEvent" in source:
            combat_events.append(name)
        if source.count("SetEventState(") > 1:
            multi_stage.append(name)
        if "public override bool IsAllowed" in source:
            gated.append(name)
        if "base.Rng." in source or "Rng.Next" in source:
            rng_events.append(name)
        if "CardSelectCmd" in source:
            choice_events.append(name)

        bodies = override_method_bodies(source)
        for key, chosen in option_calls(source):
            total_options += 1
            token = chosen.strip()
            if re.fullmatch(r"\w+", token):
                method_options += 1
                body = bodies.get(token) or method_body(source, token) or ""
            else:
                body = token
            if "CardSelectCmd" in body:
                stats["option_needs_choice"] += 1
            for match in COMMAND_CALL.finditer(body):
                commands[f"{match.group(1)}.{match.group(2)}"] += 1

    print(f"选项总数：{total_options}（其中方法引用 {method_options}）")
    print(f"\n需要战斗交接的事件 {len(combat_events)}：{combat_events}")
    print(f"\n多阶段事件 {len(multi_stage)}：{multi_stage}")
    print(f"\n有进入条件（IsAllowed）{len(gated)}：{gated}")
    print(f"\n用事件内 RNG {len(rng_events)}：{rng_events}")
    print(f"\n含选牌（任意位置）{len(choice_events)}：{choice_events[:20]}")
    print(f"\n需要玩家选择的选项：{stats['option_needs_choice']}")
    print("\n选项体里的命令 Top 30：")
    for name, count in commands.most_common(30):
        print(f"  {count:4d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
