"""内容填充度体检：一次性打印各类内容的**真实**覆盖情况。

存在的意义：README 里那些百分比很容易变成"自我安慰"。这个脚本把每一项都拆成
"总量 / 引擎真的能跑 / 还差什么"，并且**只报能验证的数字**。

用法
----
    python tools/content_status.py
    python tools/content_status.py --content data/content/repo
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="内容填充度体检")
    parser.add_argument("--content", default="data/content/repo")
    args = parser.parse_args(argv)

    from sts2_sim import content
    from sts2_sim import eligibility
    from sts2_sim import powers as power_rules
    summary = content.load_content_dir(args.content)
    coverage = content.engine_coverage()
    admission = eligibility.report("ironclad")

    line = "=" * 68

    print(line)
    print(f"内容目录：{summary.get('source')}")
    print(line)
    print("口径说明（docs/13 §7）：")
    print("  「可训练」= 引擎可忠实执行 **且依赖闭包可执行**，由 sts2_sim/eligibility.py")
    print("  统一判定；训练采样器用的是**同一个函数**，所以两边数字不可能对不上。")
    print("  状态的六层里只有前四层可自动判定；「源码用例」「真机对拍」一律为未通过。")

    # ---- 卡牌 ----
    total = coverage["cards_total"]
    usable = admission["cards_admitted"]
    pool_size = admission["draft_pool"]
    print(f"\n【卡牌】{total} 张 | 引擎可执行 {usable} "
          f"({usable / max(1, total):.1%}) | 静态标记残缺 {coverage['cards_incomplete']}")
    print(f"  铁甲战士**可抽池**（奖励/商店能出现的）：{pool_size} 张")
    print(f"  效果来源：反编译源码 {coverage['cards_effect_from_source']} 张 · "
          f"社区文本 {coverage['cards_effect_from_text']} 张")
    print("  拒绝原因（可归因，无『其它』桶）：")
    for reason, n in admission["reject_reasons"].items():
        print(f"    {n:4d}  {reason}")
    if admission["cards_closure_only_failures"]:
        print(f"    其中 {admission['cards_closure_only_failures']} 张**自身没问题**，"
              f"只是它生成的牌不合格（闭包缺口）")

    # 按角色统计可用率（与采样器同源）
    by_character: dict[str, list[int]] = {}
    for pool, members in content.CARD_POOLS.items():
        ok = inc = 0
        for cid in members:
            card = content.CARD_DB.get(cid)
            if card is None:
                continue
            if eligibility.card_admission(cid).admitted:
                ok += 1
            else:
                inc += 1
        if ok + inc:
            by_character[pool] = [ok, inc]
    print("  按卡池（合格 / 不合格）：")
    for pool in sorted(by_character, key=lambda p: -by_character[p][0]):
        ok, inc = by_character[pool]
        print(f"    {pool:22s} 合格 {ok:3d} / 不合格 {inc:3d}"
              f"  ({ok / max(1, ok + inc):5.1%})")

    # 残缺原因分类
    reasons: collections.Counter = collections.Counter()
    for card in content.CARD_DB.values():
        if not card.effects_incomplete:
            continue
        if not card.effects and not card.triggers:
            reasons["本来就无效果（诅咒/状态/任务牌）"] += 1
        elif card.effect_source != "source":
            reasons["效果只从社区文本解析（源码解析不了）"] += 1
        else:
            reasons["源码解析出残缺（引擎缺算子/需选牌/依赖运行时）"] += 1
    print("  静态残缺原因：")
    for reason, n in reasons.most_common():
        print(f"    {n:4d}  {reason}")

    # ---- 敌人 ----
    with_ai = sum(1 for e in content.ENEMY_DB.values()
                  if getattr(e, "ai_record", None))
    print(f"\n【敌人】{coverage['monsters_total']} 只 | 出招状态机 {with_ai} 只挂上")
    ai_path = Path(args.content) / "monster_ai.json"
    if ai_path.exists():
        import json
        ai = json.loads(ai_path.read_text(encoding="utf-8"))
        machine_kinds = collections.Counter()
        for record in ai:
            if not record.get("states"):
                machine_kinds["无状态机（单招）"] += 1
                continue
            if record.get("random_states"):
                machine_kinds["含随机分支"] += 1
            elif record.get("cycle"):
                machine_kinds["固定循环"] += 1
            else:
                machine_kinds["条件分支"] += 1
        print(f"  出招模式：{dict(machine_kinds)}")
    print(f"  固有属性：{coverage['monsters_with_innate_powers']} 只有 · "
          f"{coverage['monsters_with_unimplemented_innate']} 只仍有未实现的能力")
    if coverage["unimplemented_innate_ranked"]:
        print("    未实现的固有属性 Top 8：")
        for pid, n in coverage["unimplemented_innate_ranked"][:8]:
            print(f"      {n:3d} 只  {pid}")

    # ---- 能力（buff / debuff）----
    print(f"\n【能力 buff/debuff】社区库+源码 {len(content.POWERS)} 个 | "
          f"引擎已实现 {len(power_rules.IMPLEMENTED)} 个")
    # ⚠️ 关键区分：只有**可训练卡**引用到的未实现能力才是真问题 ——
    # 卡在训练集里、费用照扣、效果什么也不做。只在残缺卡上出现的
    # 未实现能力**不影响训练**（那些卡已经被排除）。
    trainable_missing: collections.Counter = collections.Counter()
    excluded_missing: collections.Counter = collections.Counter()
    for card in content.CARD_DB.values():
        bucket = excluded_missing if card.effects_incomplete else trainable_missing
        for effect in card.effects:
            if (effect.op == "apply_power" and effect.power
                    and effect.power not in power_rules.IMPLEMENTED):
                bucket[effect.power] += 1
    print(f"  ⚠️ 可训练卡引用的未实现能力：{len(trainable_missing)} 种"
          f"（必须为 0，否则那些卡会静默无效）")
    print(f"     残缺卡引用的未实现能力：{len(excluded_missing)} 种"
          f"（已随卡排除，不影响训练）")
    if trainable_missing:
        for pid, n in trainable_missing.most_common(10):
            print(f"       {n:3d} 次  {pid}")

    # 每个已实现能力有没有**行为测试**兜底
    test_text = "\n".join(
        p.read_text(encoding="utf-8")
        for p in sorted((Path(__file__).resolve().parent.parent / "tests").glob("*.py")))
    covered = sorted(pid for pid in power_rules.IMPLEMENTED if f'"{pid}"' in test_text
                     or f"'{pid}'" in test_text)
    uncovered = sorted(set(power_rules.IMPLEMENTED) - set(covered))
    print(f"  有测试覆盖的能力：{len(covered)}/{len(power_rules.IMPLEMENTED)}")
    # ⭐ "实现了但打不到"：只报"已实现 N 个"会造成错觉 ——
    # 能力写完了，但没有任何东西施加它（招式效果缺失 / 施加它的卡残缺 / 条件性施加）。
    unreachable = coverage.get("powers_implemented_but_unreachable") or []
    print(f"  其中**当前能被打到**的："
          f"{coverage.get('powers_reachable')}/{len(power_rules.IMPLEMENTED)}")
    if unreachable:
        print(f"    实现了但打不到（{len(unreachable)} 个，多为条件性/阶段性施加）："
              f"{', '.join(unreachable)}")
    if uncovered:
        print(f"    没有直接测试的：{', '.join(uncovered)}")

    # ---- 遗物 ----
    print(f"\n【遗物】{coverage['relics_total']} 个 | 有已复刻行为 "
          f"{coverage['relics_with_behavior']} | 完全复刻 "
          f"{coverage['relics_fully_modeled']}")
    print(f"  未实现的钩子：{coverage['unimplemented_relic_hook_kinds']} 种 / "
          f"{coverage['unimplemented_relic_hooks']} 处")
    # ⭐ 按类别拆开（审计 F11）：只有 commands/query 才是**真的缺行为**；
    # bookkeeping 只写遗物自己的私有字段与状态显示，本身不改变游戏状态。
    kinds = coverage.get("relic_hook_gap_kinds") or {}
    if kinds:
        print(f"    其中真的缺行为 {coverage['relic_hook_gaps_effectful']} 处"
              f"（commands {kinds.get('commands', 0)} / query {kinds.get('query', 0)}"
              f" / unknown {kinds.get('unknown', 0)}）"
              f" · 仅记账 {coverage['relic_hook_gaps_bookkeeping']} 处"
              f"（只写自己的计数器/状态显示，**本身不改变数值**）")
        print(f"    仍有真缺口的遗物 {coverage['relics_with_effectful_gaps']} 个"
              f" · 只剩记账型缺口的 {coverage['relics_bookkeeping_only_gaps']} 个")
    print(f"  未采纳的数值钩子（条件算不出）："
          f"{len(summary.get('relics_unresolved_numeric') or [])} 条")
    print("    未实现钩子 Top 10（按受影响遗物数）：")
    for hook, n in coverage["unimplemented_relic_hook_ranked"][:10]:
        print(f"      {n:4d} 个遗物  {hook}")

    # ---- 药水 / 事件 ----
    print(f"\n【药水】{summary.get('potions')} 个 | 可用 {summary.get('potions_usable')} "
          f"| 残缺 {summary.get('potions_incomplete')}")
    events_total = summary.get("events") or 0
    events_usable = summary.get("events_usable") or 0
    events_pool = summary.get("events_pool") or 0
    if events_total:
        print(f"\n【事件】{events_total} 个 | 引擎可跑 {events_usable} "
              f"| 第 1 幕事件池 {events_pool}")
        blocked = summary.get("events_blocked_examples") or []
        if blocked:
            print(f"  有缺口（前 10）：{', '.join(blocked)}")
        # 缺口分类：事件级的不可用理由
        import collections as _collections

        from sts2_sim import content as _content

        reasons: _collections.Counter = _collections.Counter()
        for definition in _content.EVENT_DB.values():
            for reason in definition.reasons:
                reasons[reason.split("：")[0]] += 1
        if reasons:
            print("  缺口分布（按事件数）：")
            for reason, count in reasons.most_common(10):
                print(f"    {count:4d}  {reason}")

    # ---- 汇总 ----
    print(f"\n{line}")
    print("总览")
    print(line)
    print(f"  卡牌  {usable:3d}/{total:<3d} 引擎可执行（可抽池 {pool_size}）")
    print(f"  敌人  {coverage['monsters_total']:3d}/{coverage['monsters_total']:<3d} "
          f"出招可跑（{coverage['monsters_with_unimplemented_innate']} 只固有属性不全）")
    print(f"  能力  {len(power_rules.IMPLEMENTED):3d}/{len(content.POWERS):<3d} 已实现")
    print(f"  遗物  {coverage['relics_with_behavior']:3d}/{coverage['relics_total']:<3d} "
          f"有行为（{coverage['relics_fully_modeled']} 完全复刻）")
    print(f"  药水  {summary.get('potions_usable'):3d}/{summary.get('potions'):<3d} 可用")
    print(f"  事件  {events_usable:3d}/{events_total:<3d} 可跑"
          f"（第 1 幕事件池 {events_pool}）")
    print(f"  合格遭遇  {admission['encounters']}")
    print("  ⚠️ 以上全部属于「可执行 / 闭包可执行」两层；")
    print("     「源码用例」只有 tests/ 里那一部分，「真机对拍」为 0。")
    print("     不得用这些数字声称数值与真机一致（审计 F11）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
