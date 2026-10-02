"""诊断：未实现的能力**分别需要哪个钩子**。

这一层决定"能不能立刻做"：
* 需要的钩子**引擎已有** → 可以逐个实现（纯体力活）
* 需要的钩子是引擎没有的 → 要先补钩子位（结构工作）

方法：读每个未实现能力的源码，看它 ``public override`` 了哪些方法 ——
那就是它要挂的钩子。
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
POWERS = ROOT / "data" / "decompiled" / "sts2" / "MegaCrit.Sts2.Core.Models.Powers"
CONTENT = ROOT / "data" / "content" / "repo"

OVERRIDE = re.compile(r"public\s+override\s+(?:async\s+)?[\w<>?\[\],\s]+?\s+(\w+)\s*\(")

#: 引擎**已有**的钩子/查询面（``docs/11`` §11.5 的钩子总线 + 查询接口）
HAVE = {
    "AfterEnergyReset", "AfterSideTurnStart", "AfterSideTurnEnd", "AfterDamageReceived",
    "AfterDamageGiven", "AfterCardPlayed", "AfterCardDrawn", "AfterDeath", "AfterApplied",
    "ModifyDamageMultiplicative", "ModifyDamageAdditive", "ModifyBlockAdditive",
    "ShouldClearBlock", "ShouldDraw", "ShouldFlush", "TryModifyPowerAmountReceived",
    "ModifyHpLostAfterOsty", "BeforeSideTurnStart", "AfterPlayerTurnStart", "AfterCombatEnd",
}

#: 引擎**没有**、但已经有对应机制的钩子（值个说明）
HAVE_BY_TABLE = {
    "ModifyDamageCap": "伤害上限（hard_to_kill 已用）",
    "ModifyOrbValue": "充能球数值（orbs.py 已处理 focus）",
    "ShouldPlayVfx": "纯表现",
    "IconBaseName": "纯表现",
    "GenerateAnimator": "纯表现",
}

#: 纯表现 / 多人用的虚方法 —— 模拟器不需要
COSMETIC = re.compile(
    r"(Vfx|Anim|Sfx|Icon|Title|Portrait|Node|Flash|Scale|HoverTip|Description|"
    r"LocString|AssetPath|Color|Sound|Music|Bark|Multiplayer|NetId|PlayerCount|"
    r"ShouldPlayVfx|DisplayAmount|ShowCounter|IsStackable|AllowNegative|"
    r"StackType|Type$|PackOdds|SpawnWeight|ShouldScale)")


def camel(pid: str) -> str:
    return "".join(part.capitalize() for part in pid.split("_")) + "Power"


def main() -> int:
    from sts2_sim import content
    from sts2_sim.powers import IMPLEMENTED

    content.load_content_dir(str(CONTENT))
    records = {str(r["cid"]).lower(): r for r in
               json.loads((CONTENT / "cards_source.json").read_text(encoding="utf-8"))}

    missing = sorted(set(content.POWERS) - set(IMPLEMENTED))
    # 谁在引用它（含残缺卡）
    users: dict[str, list[str]] = collections.defaultdict(list)
    for card in content.CARD_DB.values():
        for effect in card.effects:
            if effect.op == "apply_power" and effect.power in missing:
                users[effect.power].append(card.cid)

    needs = collections.Counter()
    ready: list[tuple[str, list[str]]] = []
    blocked: list[tuple[str, list[str]]] = []
    not_found: list[str] = []
    for pid in missing:
        path = POWERS / f"{camel(pid)}.cs"
        if not path.exists():
            hits = list(POWERS.glob(f"*{camel(pid)}.cs"))
            path = hits[0] if hits else None
        if path is None:
            not_found.append(pid)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hooks = [h for h in dict.fromkeys(OVERRIDE.findall(text))
                 if not COSMETIC.search(h) and h != "InitInternalData"]
        # 需要的新钩子 = 不在 HAVE / HAVE_BY_TABLE 里的
        new_hooks = [h for h in hooks if h not in HAVE and h not in HAVE_BY_TABLE]
        if not new_hooks:
            ready.append((pid, hooks))
        else:
            blocked.append((pid, new_hooks))
            for hook in new_hooks:
                needs[hook] += 1

    print("=" * 72)
    print(f"未实现能力 {len(missing)} 个：卡在哪")
    print("=" * 72)
    print(f"\n✅ **钩子已有、可以立刻做**：{len(ready)} 个")
    for pid, hooks in ready[:18]:
        uses = users.get(pid) or []
        tag = f"← {', '.join(uses[:2])}" if uses else ""
        print(f"    {pid:26s} {', '.join(hooks[:3])[:44]:46s} {tag}")
    if len(ready) > 18:
        print(f"    …还有 {len(ready) - 18} 个")

    print(f"\n❌ **需要引擎先补钩子**：{len(blocked)} 个，"
          f"涉及 {len(needs)} 种新钩子")
    for hook, count in needs.most_common(18):
        examples = [pid for pid, hooks in blocked if hook in hooks][:3]
        print(f"    {count:3d} 个能力  {hook:34s}（{', '.join(examples)}）")

    if not_found:
        print(f"\n⚠️ 源码里找不到类：{len(not_found)} 个 —— {not_found[:8]}")

    # 引用情况：这些能力被多少张**残缺卡**引用
    referenced = {pid: users[pid] for pid in missing if users.get(pid)}
    print(f"\n被残缺卡引用的未实现能力：{len(referenced)} 个")
    print("  也就是说：**其余 %d 个能力没有任何卡在用**（抽取器没抽到，或本就是内部能力）"
          % (len(missing) - len(referenced)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
