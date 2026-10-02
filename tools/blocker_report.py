"""诊断：卡牌 / 能力 / 药水**具体卡在哪**（``docs/09`` §20）。

只回答一个问题：**这些没法实现的东西，到底缺什么？**
按"缺的东西"聚合，而不是按对象罗列 —— 后者看不出该先做什么。
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONTENT = Path(__file__).resolve().parent.parent / "data" / "content" / "repo"


def main() -> int:
    from sts2_sim import content
    from sts2_sim.powers import IMPLEMENTED

    content.load_content_dir(str(CONTENT))
    records = {str(r["cid"]).lower(): r for r in
               json.loads((CONTENT / "cards_source.json").read_text(encoding="utf-8"))}
    potions = {str(r["pid"]).lower(): r for r in
               json.loads((CONTENT / "potions_source.json").read_text(encoding="utf-8"))}

    line = "=" * 72
    print(line)
    print("卡牌 / 能力 / 药水：卡在哪")
    print(line)

    # ---------------- 卡牌 ----------------
    incomplete = [c for c in content.CARD_DB.values() if c.effects_incomplete]
    print(f"\n【卡牌】{len(content.CARD_DB)} 张，残缺 {len(incomplete)}")
    buckets: dict[str, list[str]] = collections.defaultdict(list)
    blockers: collections.Counter = collections.Counter()
    for card in incomplete:
        record = records.get(card.cid) or {}
        if not card.effects and not card.triggers:
            reason = "A. 本来就没有可执行效果"
            buckets[reason].append(card.cid)
            if record.get("unsupported"):
                blockers[f"   （但源码里有：{record['unsupported'][0][:34]}）"] += 1
            continue
        missing = [p for p in content._missing_powers(record) if p not in IMPLEMENTED]
        if card.effect_source != "source":
            reason = "B. 源码解析不出效果（退回文本）"
        elif record.get("choice_commands"):
            reason = "C. 需要玩家选牌"
        elif missing:
            reason = "D. 引擎没实现它施加的能力"
        else:
            reason = "E. 其他解析缺口"
        buckets[reason].append(card.cid)
        for pid in missing:
            blockers[f"能力 {pid}"] += 1
        for command in (record.get("unsupported") or [])[:1]:
            blockers[f"命令 {command[:38]}"] += 1
    for reason in sorted(buckets):
        print(f"  {reason}: {len(buckets[reason])} 张"
              f"（例：{', '.join(sorted(buckets[reason])[:4])}）")
    print("  最大阻塞项：")
    for name, count in blockers.most_common(10):
        print(f"    {count:4d}  {name}")

    # ---------------- 能力 ----------------
    print(f"\n【能力】{len(content.POWERS)} 个，引擎已实现 {len(IMPLEMENTED)}，"
          f"未实现 {len(content.POWERS) - len(IMPLEMENTED)}")
    referenced: collections.Counter = collections.Counter()
    owner: dict[str, list[str]] = collections.defaultdict(list)
    for card in content.CARD_DB.values():
        for effect in card.effects:
            if effect.op == "apply_power" and effect.power:
                referenced[effect.power] += 1
                owner[effect.power].append(card.cid)
    trainable = [c for c in content.CARD_DB.values() if not c.effects_incomplete]
    trainable_refs = collections.Counter()
    for card in trainable:
        for effect in card.effects:
            if effect.op == "apply_power" and effect.power:
                trainable_refs[effect.power] += 1
    print(f"  可训练卡引用的未实现能力："
          f"{[p for p in trainable_refs if p not in IMPLEMENTED]}（必须为空）")
    blocking = [(pid, len(owner[pid])) for pid in referenced
                if pid not in IMPLEMENTED and pid in owner]
    blocking.sort(key=lambda kv: -kv[1])
    print(f"  被残缺卡引用的未实现能力：{len(blocking)} 种，"
          f"共涉及 {sum(n for _p, n in blocking)} 处引用")
    print("  影响最大的 12 个：")
    for pid, count in blocking[:12]:
        examples = ", ".join(sorted(owner[pid])[:3])
        print(f"    {count:3d} 张卡  {pid:26s}（{examples}）")

    # 只被"本来无效果"的卡引用 vs 被真有内容的卡引用
    no_effect = set(buckets["A. 本来就没有可执行效果"])
    real = [(pid, [c for c in owner[pid] if c not in no_effect])
            for pid, _n in blocking]
    real = [(pid, cards) for pid, cards in real if cards]
    print(f"  其中被**真有内容的卡**引用的：{len(real)} 种")

    # ---------------- 药水 ----------------
    bad_potions = [p for p in content.POTIONS.values() if p.effects_incomplete]
    print(f"\n【药水】{len(content.POTIONS)} 个，残缺 {len(bad_potions)}")
    reasons: collections.Counter = collections.Counter()
    for potion in bad_potions:
        record = potions.get(potion.pid) or {}
        # ⚠️ 判据顺序有讲究：**先看"有没有报了缺口"**，再看"有没有抽到效果"。
        # 早先反着写，于是"抽不到效果、但明确报了未支持命令"的药水
        # （`attack_potion` 报 `CardSelectCmd.FromChooseACardScreen`）
        # 被显示成"源码里就没抽到效果" —— 报告读起来像"这瓶药本来就没效果"，
        # 而事实是**我们知道它要做什么、只是还没实现**。结论对、理由错的报告
        # 比没有报告更糟。
        if record.get("choice_commands"):
            key = "需要玩家选牌"
        elif record.get("unsupported"):
            key = f"命令未支持：{record['unsupported'][0][:36]}"
        elif not potion.effects and not record.get("effects"):
            key = "源码里确实没有可执行效果"
        else:
            key = "其他"
        reasons[key] += 1
        print(f"    {potion.pid:26s} {potion.name[:14]:16s} {key}")
    for key, count in reasons.most_common():
        print(f"  {count:3d}  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
