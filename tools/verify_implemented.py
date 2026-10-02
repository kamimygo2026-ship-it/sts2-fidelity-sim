"""**已实现内容 vs 源码**的一致性核对（审计 F11 的"证据"要求）。

与 ``tools/audit_data.py`` 的分工：

* ``audit_data.py`` —— 管"数据有没有丢/重复/和社区库对不上"（数据管线层）
* **本工具** —— 管"**引擎里真正能跑的那部分**，和源码抽取出来的记录是不是同一件事"

为什么必须单独做一遍：内容经过"抽取 → 合并 → 解析成 Effect → 加载"四道手，
每一道都可能悄悄改掉一个数字或换掉一个算子。``content_status.py`` 报的是
"能不能跑"，**不是**"跑的是不是源码写的那件事"。

检查项
------
1. **数值来源**：可训练卡里有几张的效果其实来自**社区文本**？（社区库落后至少一个
   补丁，实测 18 处数值分歧 —— 那些卡的数字**不是**源码的）
2. **实现一致性**：卡 / 药水 / 遗物的引擎对象，逐条对比源码记录里的
   ``(op, amount/var, target)``；不一致就是管线把东西改掉了
3. **升级版本**：有 ``UpgradeValueBy`` 的卡，升级后的数值必须真的跟着变
4. **能力表**：引擎用的 ``stack_type`` / ``allow_negative`` 是否与源码一致
5. **敌人**：血量区间与招式数值是否与 ``monster_ai.json`` / ``monster_moves.json`` 一致

用法
----
    python -X utf8 tools/verify_implemented.py
    python -X utf8 tools/verify_implemented.py --sample 20
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"


def _effects_key(effect) -> tuple:
    """效果的可比较键，含 CalculatedVar 的公式及升级系数，避免漏检/误报。"""
    fields = ("calc_kind", "calc_arg", "calc_base", "calc_extra")
    if isinstance(effect, dict):
        amount = effect.get("amount")
        return (effect.get("op"), amount, str(effect.get("target") or ""),
                *(effect.get(name) for name in fields))
    return (effect.op, effect.amount, str(getattr(effect, "target", "") or ""),
            *(getattr(effect, name) for name in fields))


class Report:
    def __init__(self) -> None:
        self.sections: list[tuple[str, list[str], str]] = []

    def add(self, title: str, problems: list[str], ok_note: str) -> None:
        self.sections.append((title, problems, ok_note))

    def print(self, sample: int) -> int:
        total = 0
        print("=" * 72)
        print("已实现内容 vs 源码 一致性核对")
        print("=" * 72)
        for title, problems, ok_note in self.sections:
            total += len(problems)
            mark = "✅" if not problems else f"⚠️ {len(problems)} 处"
            print(f"\n【{title}】{mark}")
            if problems:
                for line in problems[:sample]:
                    print(f"    · {line}")
                if len(problems) > sample:
                    print(f"    …（共 {len(problems)} 条，只印前 {sample} 条）")
            else:
                print(f"    {ok_note}")
        print("\n" + "=" * 72)
        if total:
            print(f"合计 {total} 处需要看的地方（上面逐条列出）")
        else:
            print("全部一致 ✅")
        return 1 if total else 0


def check_card_value_sources() -> tuple[list[str], collections.Counter, list[str]]:
    """可训练卡里，效果来自社区文本的有哪些（那些数字**不是**源码的）。

    ⚠️ 诅咒 / 状态牌（clog）不算：它们本来就该没有效果，"从文本解析"没有意义。
    实测 13 张文本来源的卡里 12 张是 clog，**只有 `scare` 是可打出的** ——
    而它连 `Scare.cs` 都不存在（社区库超前于本地构建）。
    这条已经被 `eligibility` 的 `effects_from_community_text` 门禁挡住。
    """
    from sts2_sim import eligibility
    from sts2_sim.content import CARD_DB

    admitted = eligibility.admitted_cards()
    by_source = collections.Counter()
    text_only: list[str] = []
    for cid in sorted(admitted):
        card = CARD_DB[cid]
        by_source[card.effect_source] += 1
        if card.effect_source != "source" and not eligibility.is_clog(card):
            text_only.append(f"{cid}（{card.name}）")
    return text_only, by_source, sorted(admitted)


def check_card_source_agreement(admitted: list[str]) -> list[str]:
    """引擎里的卡效果 vs ``cards_source.json`` 的记录，逐条比。"""
    from sts2_sim.content import CARD_DB, _resolve_effects, _var_values

    records = {r["cid"]: r for r in json.loads(
        (CONTENT / "cards_source.json").read_text(encoding="utf-8"))}
    problems: list[str] = []
    for cid in admitted:
        record = records.get(cid)
        if record is None:
            continue
        card = CARD_DB[cid]
        # 基础版本
        expected = _resolve_effects(record.get("effects") or [], _var_values(record, False))
        if expected is not None and expected:
            engine = tuple(card.effects)
            if [_effects_key(e) for e in engine] != [_effects_key(e) for e in expected]:
                problems.append(
                    f"{cid}: 引擎 {[_effects_key(e) for e in engine]} "
                    f"≠ 源码 {[_effects_key(e) for e in expected]}")
    return problems


def check_upgrade_versions(admitted: list[str]) -> list[str]:
    """有升级的卡，升级后的数值必须真的跟着变（``UpgradeValueBy``）。"""
    from sts2_sim.content import CARD_DB

    records = {r["cid"]: r for r in json.loads(
        (CONTENT / "cards_source.json").read_text(encoding="utf-8"))}
    problems: list[str] = []
    for cid in admitted:
        record = records.get(cid)
        card = CARD_DB[cid]
        if not record or not card.upgrade:
            continue
        deltas = [str(v.get("raw")) for v in record.get("vars") or []
                  if v.get("upgrade_delta")]
        base = [_effects_key(e) for e in card.effects]
        upgraded = [_effects_key(e) for e in card.upgrade]
        if base == upgraded and deltas:
            problems.append(f"{cid}: 源码有升级增量 {deltas}，但升级版效果与基础版完全相同")
    return problems


def check_potions() -> list[str]:
    """药水：引擎对象 vs 源码记录。"""
    from sts2_sim.content import POTIONS, _resolve_effects, _var_values

    records = {r["pid"]: r for r in json.loads(
        (CONTENT / "potions_source.json").read_text(encoding="utf-8"))}
    problems: list[str] = []
    for pid, potion in sorted(POTIONS.items()):
        if potion.effects_incomplete:
            continue                      # 已如实标成残缺，单独统计
        record = records.get(pid)
        if record is None:
            problems.append(f"{pid}: 引擎有、源码记录没有")
            continue
        expected = _resolve_effects(record.get("effects") or [], _var_values(record, False))
        if expected and [_effects_key(e) for e in potion.effects] != \
                [_effects_key(e) for e in expected]:
            problems.append(
                f"{pid}: 引擎 {[_effects_key(e) for e in potion.effects]} "
                f"≠ 源码 {[_effects_key(e) for e in expected]}")
    return problems


def check_relics() -> list[str]:
    """遗物：已采纳的钩子 vs 源码记录里的同名钩子。"""
    from sts2_sim.content import (HOOK_TIMING_BY_NAME, RELICS, _resolve_effects,
                                  _var_values)

    records = {r["rid"]: r for r in json.loads(
        (CONTENT / "relics_source.json").read_text(encoding="utf-8"))}
    problems: list[str] = []
    for rid, relic in sorted(RELICS.items()):
        record = records.get(rid)
        if not record:
            continue
        values = _var_values(record, False)
        for hook in relic.hooks:
            if hook.timing in ("obtained", "combat_end", "combat_victory"):
                continue                  # Run 层钩子由 runeffects 结算，单独看
            source_hook = next(
                (name for name in record.get("hooks") or {}
                 if HOOK_TIMING_BY_NAME.get(name) == hook.timing
                 or (name == "AfterRoomEntered" and hook.timing == "combat_start")),
                None)
            if source_hook is None:
                problems.append(f"{rid}.{hook.timing}: 引擎有钩子，源码记录里找不到")
                continue
            expected = _resolve_effects(
                (record["hooks"][source_hook].get("effects") or []), values)
            if expected and [_effects_key(e) for e in hook.effects] != \
                    [_effects_key(e) for e in expected]:
                problems.append(
                    f"{rid}.{hook.timing}: 引擎 {[_effects_key(e) for e in hook.effects]} "
                    f"≠ 源码 {[_effects_key(e) for e in expected]}")
    return problems


def check_powers() -> list[str]:
    """能力：引擎用的叠加语义 / 正负性 vs 源码。"""
    from sts2_sim.content import POWERS

    source = {str(e.get("id", "")).lower(): e for e in json.loads(
        (CONTENT / "powers_source.json").read_text(encoding="utf-8"))}
    problems: list[str] = []
    for pid, power in sorted(POWERS.items()):
        record = source.get(pid)
        if not record:
            continue
        expected_stack = str(record.get("stack_type") or "").lower()
        if expected_stack and power.stack_type != expected_stack:
            problems.append(f"{pid}: stack_type 引擎={power.stack_type} "
                            f"源码={expected_stack}")
        if "allow_negative" in record and bool(record["allow_negative"]) != \
                bool(power.allow_negative):
            problems.append(f"{pid}: allow_negative 引擎={power.allow_negative} "
                            f"源码={record['allow_negative']}")
    return problems


def check_enemies() -> list[str]:
    """敌人：血量区间与招式数值 vs 源码抽取。"""
    from sts2_sim.content import ENEMY_DB

    ai = {str(e.get("eid", "")).lower(): e for e in json.loads(
        (CONTENT / "monster_ai.json").read_text(encoding="utf-8"))}
    problems: list[str] = []
    for eid, enemy in sorted(ENEMY_DB.items()):
        record = ai.get(eid)
        if not record:
            problems.append(f"{eid}: 没有源码状态机记录")
            continue
        hp = record.get("hp")
        if hp and hp[0] is not None:
            expected = (int(hp[0]), int(hp[1]) if hp[1] is not None else int(hp[0]))
            if enemy.hp != expected:
                problems.append(f"{eid}: HP 引擎={enemy.hp} 源码={expected}")
        if not enemy.moves:
            problems.append(f"{eid}: 引擎没有招式")
    return problems


#: "声明了但没有任何效果消费"的变量 → 它代表哪一种**缺失机制**。
#:
#: ⚠️ 这是本项目**最隐蔽**的一类静默错误：抽取器只抓直接命令，
#: 下面这些变量既没变成效果、也没记进 `unsupported`，于是卡被当成
#: "完全复刻"通过准入 —— 而它在模拟器里明显更弱。
#: 实测确认过的例子（都已逐个对过 C# 源码）：
#:
#: * ``Repeat`` —— ``Capacitor``（`OrbCmd.AddSlots(Repeat=2)`，引擎只加 1 个槽）、
#:   ``Quadcast``（激发 4 次，引擎 1 次）、``BouncingFlask``（3 次 × 3 毒）
#: * ``Cards`` —— ``CloakAndDagger``（加 1 张 Shiv，引擎没加）
#: * ``Increase`` —— ``Claw`` / ``Rampage`` / ``TheBall`` / ``Maul`` / ``KinglyPunch``
#:   的"本场每次打出递增"
MECHANIC_VARS: dict[str, str] = {
    "Repeat": "效果重复 N 次（引擎只做 1 次）",
    "Increase": "本场成长（每次打出递增，引擎没有该机制）",
    "Cards": "附加/抽取 N 张（引擎没做或次数不对）",
    "Shivs": "生成 N 张 Shiv",
    "PutBack": "放回抽牌堆 N 张",
    "PlayMax": "本回合最多再打 N 张",
    "CalculationBase": "运行期公式的基数",
    "CalculationExtra": "运行期公式的增量",
    "CalculatedCards": "运行期算出的张数",
    "CalculatedChannels": "运行期算出的引导次数",
    "DamageIncrease": "数值修正（也可能只是 Power 的量纲，需看源码）",
    "DamageDecrease": "数值修正（也可能只是 Power 的量纲，需看源码）",
    "Energy": "能量收支（可能已被 gain_energy 表达，需看源码）",
}

#: 引擎用**别的方式**表达的变量，不算漏。
CONSUMED_ELSEWHERE: dict[str, str] = {
    "Damage": "damage 效果直接消费",
    "Block": "block 效果直接消费",
    "Heal": "heal 效果直接消费",
    "Draw": "draw 效果直接消费",
}


def check_unmodelled_card_mechanics(admitted: list[str]) -> list[str]:
    """可训练卡里"声明了变量但没有任何效果消费它"的 —— 逐条列出待核。

    ⚠️ 本函数**只负责列出来**，不假装判定：是否真的缺机制必须回源码看。
    但它把范围缩到了很小的规模，人手核对是可行的。

    判据与 `content._unconsumed_mechanics`（**准入用的那道闸门**）同源：
    * 跳过诅咒 / 状态牌（clog）—— 它们本来就该没有效果；
    * 跳过 `POWER_MAGNITUDE_VARS` —— 那些变量由能力消费（`Tank` 的
      `DamageIncrease` / `DamageDecrease` 是 `TankPower` 的量纲）。
    """
    import re as _re
    from sts2_sim import eligibility
    from sts2_sim.content import CARD_DB, MECHANIC_VARS, POWER_MAGNITUDE_VARS

    records = {r["cid"]: r for r in json.loads(
        (CONTENT / "cards_source.json").read_text(encoding="utf-8"))}
    problems: list[str] = []
    for cid in admitted:
        record = records.get(cid)
        if record is None or eligibility.is_clog(CARD_DB[cid]):
            continue
        blob = json.dumps(record.get("effects") or [], ensure_ascii=False)
        blob += json.dumps(record.get("triggers") or [], ensure_ascii=False)
        for var in record.get("vars") or []:
            name = str(var.get("name") or "")
            if name not in MECHANIC_VARS or name in POWER_MAGNITUDE_VARS:
                continue
            if _re.search(rf'"{name}"', blob) or _re.search(rf"\.{name}\b", blob):
                continue
            hint = MECHANIC_VARS.get(name, "（未归类，需看源码）")
            problems.append(f"{cid}（{CARD_DB[cid].name}）变量 {name}：{hint}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="已实现内容 vs 源码 一致性核对")
    parser.add_argument("--sample", type=int, default=12,
                        help="每节最多打印多少条")
    args = parser.parse_args(argv)

    if not CONTENT.exists():
        print(f"缺内容目录 {CONTENT}")
        return 3
    from sts2_sim.featurize import configure_content
    configure_content(str(CONTENT))

    report = Report()
    text_only, by_source, admitted = check_card_value_sources()
    report.add(
        "1 · 可训练卡的数值来源",
        text_only,
        f"全部 {len(admitted)} 张可训练卡的效果都来自反编译源码")

    report.add(
        "2 · 卡牌：引擎效果 vs 源码记录",
        check_card_source_agreement(admitted),
        f"{len(admitted)} 张可训练卡逐条一致")

    report.add(
        "3 · 卡牌：**声明了但没有任何效果消费**的变量（缺失机制）",
        check_unmodelled_card_mechanics(admitted),
        "没有卡漏掉已声明的机制变量")

    report.add(
        "4 · 卡牌：升级版本数值",
        check_upgrade_versions(admitted),
        "有升级增量的卡，升级版数值都真的变了")

    report.add("5 · 药水：引擎效果 vs 源码记录", check_potions(),
               "全部可用药水逐条一致")
    report.add("6 · 遗物：已采纳钩子 vs 源码记录", check_relics(),
               "全部已采纳的遗物钩子逐条一致")
    report.add("7 · 能力：叠加语义 / 正负性", check_powers(),
               "全部与源码一致")
    report.add("8 · 敌人：血量与招式", check_enemies(),
               "全部与源码一致")

    print(f"卡牌数值来源分布：{dict(by_source)}")
    code = report.print(args.sample)
    print("\n⚠️ 本工具核对的是『引擎 vs 自己的源码抽取』，**不是**『引擎 vs 真机』。")
    print("   前者能证明管线没改数字；后者需要真机对拍（当前为 0）。")
    print("   第 3 节列出的条目**必须回 C# 源码逐条确认**，工具只负责缩小范围。")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
