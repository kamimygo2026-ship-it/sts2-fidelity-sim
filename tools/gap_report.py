"""生成**残缺清单**：把引擎还没做到位的卡牌 / 敌人 / 药水列成可读文档。

设计意图
--------
"还差多少"必须能落到**具体对象**上：哪张卡、叫什么、卡面上写的是什么、
缺的是哪一步。只报百分比等于没说 —— 那份清单下一轮一定被忘掉。

中文文本取自真机中文导出（``data/raw/spire-codex/repo/export_zhs.zip``），
**只用它的名字与描述**（用来让人看懂这是哪张卡），缺口一律由源码侧判定。

用法
----
    python tools/gap_report.py                 # 写到 docs/10-残缺清单.md
    python tools/gap_report.py --stdout        # 直接打印
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"
ZHS = ROOT / "data" / "raw" / "spire-codex" / "repo" / "export_zhs.zip"
OUT_DEFAULT = ROOT / "docs" / "10-残缺清单.md"


def clean(text: str) -> str:
    """去掉卡面里的富文本标记，但**保留标记里携带的数值**。

    ⚠️ 直接 ``re.sub(r"\\[/?\\w+\\]")`` 会把 ``[energy:1]`` 整段删掉，
    于是"获得[energy:1]"变成"获得" —— 数字丢了，读的人以为卡面本身没写值。
    """
    import re
    text = str(text or "")
    # 带参数的图标标记：`[energy:1]` → `1能量`、`[star:3]` → `3星辰`
    labels = {"energy": "能量", "star": "星辰", "hp": "生命", "gold": "金币"}
    for token in ("energy", "star", "hp", "gold"):
        text = re.sub(rf"\[{token}:(\d+)\]", rf"\1{labels[token]}", text)
    text = re.sub(r"\[(\w+):(\d+)\]", r"\2", text)      # 其它带参标记只留数字
    text = re.sub(r"\[/?\w+\]", "", text)               # 纯样式标记直接去掉
    return " ".join(text.split()).replace("|", "／")


def load_zhs(name: str) -> dict[str, dict]:
    if not ZHS.exists():
        return {}
    with zipfile.ZipFile(ZHS) as archive:
        if name not in archive.namelist():
            return {}
        items = json.loads(archive.read(name).decode("utf-8"))
    key = {"cards.json": "id", "monsters.json": "id", "potions.json": "id",
           "relics.json": "id"}.get(name, "id")
    return {str(item.get(key, "")).lower(): item for item in items}


def card_reason(card, record: dict | None, missing_powers: list[str]) -> tuple[str, str]:
    """返回 ``(分类, 具体缺口)``。**按源码侧证据判定**，不看社区文本。"""
    from sts2_sim import content

    if record is None:
        return "源码里没有对应记录", "抽取器没产出这张卡"
    if not card.effects and not card.triggers:
        return ("本来就无效果（诅咒 / 状态 / 任务牌）",
                "靠「不可打出 + 占手牌」起作用，属于正确建模")
    if record.get("unsupported"):
        commands = record["unsupported"]
        return "源码里有引擎没有的命令", "、".join(str(c) for c in commands[:3])
    if record.get("choice_commands"):
        return "需要玩家选牌", f"{record['choice_commands']} 处选牌命令"
    if missing_powers:
        names = "、".join(content.POWERS.get(p, {}).name if hasattr(
            content.POWERS.get(p), "name") else p for p in missing_powers[:3])
        return "引擎没实现这个能力", f"{names}（{', '.join(missing_powers[:3])}）"
    reason = content._source_reject_reason(record)
    if reason:
        return "源码效果采用了但被拒绝", reason
    if card.effect_source != "source":
        return "效果只从社区文本解析", "源码解析不出可执行的效果"
    # ⚠️ 不许有「其它」桶（审计 F11）：落到这里就把 `eligibility` 的
    # 具体拒绝原因原样带出来，让每张卡都能归因到一条可行动的缺口。
    from sts2_sim import eligibility
    reasons = eligibility.card_reasons(card.cid)
    if reasons:
        return "引擎侧准入未通过", "、".join(reasons[:4])
    return "闭包缺口（它生成的牌不合格）", "、".join(
        eligibility.closure_gaps(card.cid)[:4]) or "见 eligibility.card_admission()"


def probe_monsters() -> dict[str, str]:
    """实际跑一遍每只怪的出招，返回 ``{eid: 异常}``。

    跑 12 个回合是为了尽量走到条件分支与随机分支 —— 只跑一回合会漏掉
    "第二回合才触发的条件"。
    """
    from sts2_sim import content, core

    failures: dict[str, str] = {}
    deck = list(content.STARTING_DECK)
    for eid in sorted(content.ENEMY_DB):
        if not content.ENEMY_DB[eid].moves:
            failures[eid] = "没有出招"
            continue
        try:
            state = core.start_combat(deck, [eid], seed=11)
            events: list[str] = []
            for _ in range(12):
                state.discard.extend(state.hand)
                state.hand = []
                core._run_enemy_turn(state, events)
                if not state.player.alive() or not state.enemies[0].alive():
                    break
                state.turn += 1
                core.start_player_turn(state, events)
        except Exception as exc:  # noqa: BLE001
            failures[eid] = f"{type(exc).__name__}: {exc}"
    return failures


def probe_move_effects() -> list[tuple[str, str, str]]:
    """找出「意图是 buff/debuff，但效果列表为空」的招式。

    ⚠️ 这就是**静默洞**的形状：意图告诉玩家"它要上 buff"，而引擎里这招什么也不做。
    玩家按错误的威胁评估做决策，模型学到的是"这只怪那回合很安全"。

    只挑 buff/debuff 意图：``unknown`` 里混着大量真·空招
    （``nothing`` / ``sleep`` / ``stun``），那些是**正确**的空。

    ⚠️ 读的是**引擎加载后的状态**（``content.ENEMY_DB``），不是 ``monsters.json``
    的社区库原文：招式效果已经改成从源码抽取并覆盖了，读原文会报出已经修好的旧缺口 ——
    一份对不上现实的报告比没有报告更糟。
    """
    from sts2_sim import content

    suspicious: list[tuple[str, str, str]] = []
    for enemy in content.ENEMY_DB.values():
        for move in enemy.moves:
            if move.effects:
                continue
            if move.intent in ("buff", "debuff"):
                suspicious.append((enemy.eid, move.mid, move.intent))
    return sorted(suspicious)


def build() -> str:
    from sts2_sim import content
    from sts2_sim.powers import IMPLEMENTED

    summary = content.load_content_dir(str(CONTENT))
    zhs_cards = load_zhs("cards.json")
    zhs_monsters = load_zhs("monsters.json")
    zhs_potions = load_zhs("potions.json")

    records = {str(r["cid"]).lower(): r for r in
               json.loads((CONTENT / "cards_source.json").read_text(encoding="utf-8"))}

    groups: dict[str, list[tuple]] = collections.defaultdict(list)
    for card in sorted(content.CARD_DB.values(), key=lambda c: c.cid):
        if not card.effects_incomplete:
            continue
        record = records.get(card.cid)
        missing = [p for p in content._missing_powers(record)] if record else []
        missing = [p for p in missing if p not in IMPLEMENTED]
        category, detail = card_reason(card, record, missing)
        entry = zhs_cards.get(card.cid, {})
        groups[category].append((
            entry.get("name") or card.name, card.cid, card.cost, card.card_type,
            clean(entry.get("description", "")), detail))

    lines: list[str] = []
    lines.append("# 10 · 残缺清单")
    lines.append("")
    lines.append("> 本文件由 `python tools/gap_report.py` **生成**，不要手改。")
    lines.append(">")
    lines.append("> 判定依据一律是**源码侧**证据（`data/decompiled/sts2/`）：")
    lines.append("> 抽不出效果、引擎缺算子、缺能力、需要选牌动作。")
    lines.append("> 中文名与卡面文字取自真机中文导出，**只用于让人看懂这是哪张卡**，")
    lines.append("> 不用作数值依据（`docs/09` §5.2 铁律）。")
    lines.append("")

    total = sum(len(v) for v in groups.values())
    # ⚠️ 这一节的所有数字都必须**现算**（审计 F11）：这里以前写着
    # `len(ENEMY_DB) - 2` 与硬编码的缺口 `2`，代码改完之后它照旧这么印。
    from sts2_sim import eligibility
    admission = eligibility.report(content.DEFAULT_CHARACTER)
    broken = probe_monsters()          # 真正跑一遍出招，不猜
    lines.append("## 一、总览")
    lines.append("")
    lines.append("口径（`docs/13` §7）：**可执行 + 闭包可执行**，由 "
                 "`sts2_sim/eligibility.py` 统一判定，与训练采样器同源。")
    lines.append("「源码用例」只有 `tests/` 里那一部分，「真机对拍」为 **0**。")
    lines.append("")
    lines.append("| 内容 | 总数 | 可执行 | 缺口 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 卡牌 | {admission['cards_total']} | "
                 f"{admission['cards_admitted']} | "
                 f"{admission['cards_total'] - admission['cards_admitted']} |")
    lines.append(f"| 敌人（出招跑通） | {len(content.ENEMY_DB)} | "
                 f"{len(content.ENEMY_DB) - len(broken)} | {len(broken)} |")
    lines.append(f"| 药水 | {summary.get('potions')} | "
                 f"{summary.get('potions_usable')} | "
                 f"{summary.get('potions_incomplete')} |")
    lines.append(f"| 事件 | {summary.get('events')} | "
                 f"{summary.get('events_usable')} | "
                 f"{(summary.get('events') or 0) - (summary.get('events_usable') or 0)} |")
    lines.append(f"| 遗物（有已复刻行为） | {len(content.RELICS)} | "
                 f"{content.engine_coverage()['relics_with_behavior']} | "
                 f"{len(content.RELICS) - content.engine_coverage()['relics_with_behavior']} |")
    lines.append("")
    lines.append("卡牌拒绝原因（可归因，无「其它」桶）：")
    lines.append("")
    lines.append("| 原因 | 张数 |")
    lines.append("|---|---|")
    for reason, count in admission["reject_reasons"].items():
        lines.append(f"| `{reason}` | {count} |")
    lines.append("")
    lines.append("卡牌残缺按原因分组：")
    lines.append("")
    lines.append("| 原因 | 张数 |")
    lines.append("|---|---|")
    for category, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {category} | {len(items)} |")
    lines.append("")

    for category, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"## 二 · {category}（{len(items)} 张）")
        lines.append("")
        lines.append("| 中文名 | id | 费 | 类型 | 卡面写的是什么 | 缺口 |")
        lines.append("|---|---|---|---|---|---|")
        for name, cid, cost, ctype, description, detail in items:
            lines.append(f"| {name} | `{cid}` | {cost} | {ctype} | "
                         f"{description} | {detail} |")
        lines.append("")

    # ---- 敌人 ----
    # ⚠️ `broken` 已在总览之前算好（那两行**必须跑出来**，不能硬编码：
    # 硬编码的"已知问题"会随着修复过期，而清单上还挂着它 —— 或者反过来，
    # 新坏掉的怪不会出现在清单里）。

    lines.append("## 三 · 敌人")
    lines.append("")
    lines.append(f"### 3.1 出招状态机跑不通（{len(broken)} 只）")
    lines.append("")
    if broken:
        lines.append("跑法：每只怪单独开一场战斗、推进 12 个回合，"
                     "任何异常都算缺口。")
        lines.append("")
        lines.append("| 中文名 | id | 症状 |")
        lines.append("|---|---|---|")
        for eid, why in broken.items():
            entry = zhs_monsters.get(eid, {})
            lines.append(f"| {entry.get('name') or eid} | `{eid}` | {clean(why)} |")
    else:
        lines.append("全部 115 只都能跑通。")
    lines.append("")

    lines.append("### 3.2 槽位名依赖（`SlotName`）")
    lines.append("")
    lines.append("真机的槽位名由**遭遇**定义（`EncounterModel.Slots` + "
                 "`(怪物, 槽位名)` 配对），不是按位置推的：")
    lines.append("")
    lines.append("| 遭遇 | 槽位名 |")
    lines.append("|---|---|")
    lines.append("| `ExoskeletonsNormal` | `first` / `second` / `third` / `fourth` |")
    lines.append("| `BowlbugsNormal` | `first` / `middle` / `last` |")
    lines.append("| 蠕虫群 | `wriggler1` / `wriggler2` / `wriggler3` / `wriggler4` |")
    lines.append("")
    ai = json.loads((CONTENT / "monster_ai.json").read_text(encoding="utf-8"))
    slot_users = []
    for record in ai:
        conditions = set()
        for branches in (record.get("conditionals") or {}).values():
            for branch in branches:
                conditions.add(str(branch.get("condition") or ""))
        for branch in (record.get("conditional_initials") or []):
            conditions.add(str(branch.get("condition") or ""))
        if any("SlotName" in c for c in conditions):
            slot_users.append(record["eid"])
    lines.append(f"受影响：**{len(slot_users)} 只** —— "
                 + "、".join(f"`{eid}`" for eid in slot_users))
    lines.append("")

    lines.append("### 3.3 固有属性没实现")
    lines.append("")
    coverage = content.engine_coverage()
    lines.append(f"{coverage['monsters_with_innate_powers']} 只怪有固有属性，"
                 f"其中 **{coverage['monsters_with_unimplemented_innate']} 只**"
                 f"含有引擎没实现的能力。按受影响怪物数排序：")
    lines.append("")
    lines.append("| 能力 | 影响的怪 |")
    lines.append("|---|---|")
    for pid, count in coverage["unimplemented_innate_ranked"]:
        lines.append(f"| `{pid}` | {count} |")
    lines.append("")
    lines.append("涉及的具体怪：")
    lines.append("")
    lines.append("| 中文名 | id | 缺的能力 |")
    lines.append("|---|---|---|")
    for enemy in sorted(content.ENEMY_DB.values(), key=lambda e: e.eid):
        absent = sorted({pid for pid, _ in enemy.innate_powers if pid not in IMPLEMENTED})
        if not absent:
            continue
        entry = zhs_monsters.get(enemy.eid, {})
        lines.append(f"| {entry.get('name') or enemy.name} | `{enemy.eid}` | "
                     f"{', '.join('`' + p + '`' for p in absent)} |")
    lines.append("")

    lines.append("### 3.4 ⭐ 招式效果缺失：意图说有，数据里没有")
    lines.append("")
    lines.append("**怪物招式表原本来自社区库**，而社区库对很多招式只记了意图、没记效果 ——")
    lines.append("那些怪在我的模拟器里**那一回合什么也不做**，而意图还显示着 buff/debuff。")
    lines.append("")
    lines.append("✅ **已修**：招式效果改为从源码抽取（`tools/extract_moves.py`），"
                 "按「状态机里的处理函数名」对上招式，覆盖社区库版本。")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 招式效果来自**源码** | {summary.get('monster_move_effects_from_source')} 条 |")
    lines.append(f"| 仍用社区库（源码抽不干净） | "
                 f"{summary.get('monster_move_effects_from_codex')} 条 |")
    lines.append(f"| 抽出来但落不进 DSL | "
                 f"{summary.get('monster_move_effects_rejected')} 条 |")
    lines.append("")
    suspicious = probe_move_effects()
    lines.append(f"「意图是 buff/debuff，但效果列表为空」的招式共 **{len(suspicious)} 条**，"
                 f"涉及 {len({entry[0] for entry in suspicious})} 只怪（完整清单）：")
    lines.append("")
    lines.append("| 怪 | 招式 | 意图 |")
    lines.append("|---|---|---|")
    for eid, mid, intent in suspicious:
        entry = zhs_monsters.get(eid, {})
        lines.append(f"| {entry.get('name') or eid} | `{mid}` | {intent} |")
    lines.append("")
    lines.append("修法：招式效果必须**从源码抽取**（和卡牌/遗物/药水同一条管线），")
    lines.append("不能继续用社区库 —— 它把\"这只怪那一回合做什么\"整段漏掉，")
    lines.append("而漏掉的恰好是玩家最需要预判的部分。")
    lines.append("")

    # ---- 药水 ----
    lines.append("## 四 · 药水")
    lines.append("")
    lines.append("| 中文名 | id | 状态 |")
    lines.append("|---|---|---|")
    for pid, potion in sorted(content.POTIONS.items()):
        if not potion.effects_incomplete:
            continue
        entry = zhs_potions.get(pid, {})
        lines.append(f"| {entry.get('name') or potion.name} | `{pid}` | "
                     f"{clean(entry.get('description', '')) or '效果残缺'} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成残缺清单")
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args(argv)
    text = build()
    if args.stdout:
        print(text)
        return 0
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"写出 {args.out}（{len(text.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
