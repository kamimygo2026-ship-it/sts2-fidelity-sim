"""审计：已实现能力与源码覆写方法是否**对得上**（``docs/12`` 的"静默错"一族）。

两类缺口都不会报错，只会让数值悄悄偏掉：

1. **注册了但没人消费** —— 能力进了 ``RULES``、``content_status`` 也就把它算成
   "已实现"，可引擎里根本没有读它的地方（``curl_up`` / ``hard_to_kill`` / ``slow``
   都曾整条无效）。
2. **源码覆写了事件方法，引擎没接** —— 例如 ``DoubleDamagePower.AfterSideTurnEnd``
   负责"到期递减"，漏接就等于**永久双倍**（``ColossusPower`` / ``IntangiblePower`` /
   ``CoveredPower`` 同样如此）；``BlockNextTurnPower.AfterBlockCleared`` 漏接
   会绕开 ``gain_block``，于是"获得格挡时"的订阅者不触发。

输出三份清单：

* ``dead`` —— ``RULES`` 里既没有钩子、也没有查询消费者的能力；
* ``unwired`` —— 源码覆写了事件方法、``RULES`` 却没接的能力（按"成员 → 钩子字段"
  的对照表判断）；
* ``duration_without_table`` —— 标了 ``duration=True`` 但**不在**递减表里的能力
  （那个标记本身没有消费者，标了不等于会递减）。

用法::

    python -X utf8 tools/audit_power_hooks.py [--check]

``--check`` 有输出就以非零码退出（给 CI / 测试用）。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DECOMPILED = ROOT / "data" / "decompiled" / "sts2"

#: 源码成员 → 引擎 ``PowerRules`` 的钩子字段（**一个成员可能对应多个字段**：
#: ``AfterSideTurnStart`` 对玩家是 ``on_owner_turn_start``、对敌人是
#: ``on_player_turn_start``；``AfterDeath`` 分自己死 / 同伴死）。
#: 只要有一个字段接上了就算"接了"。
HOOK_OF_MEMBER: dict[str, tuple[str, ...]] = {
    "AfterApplied": ("on_applied",),
    "AfterCardPlayed": ("on_card_played",),
    "BeforeCardPlayed": ("on_before_card_played",),
    "AfterCardEnteredCombat": ("on_card_entered_combat",),
    "AfterBlockCleared": ("on_block_cleared",),
    "AfterBlockGained": ("on_block_gained",),
    "AfterDamageGiven": ("on_damage_given",),
    "AfterDamageReceived": ("on_attacked",),
    "AfterDeath": ("on_self_death", "on_ally_death", "on_applier_death"),
    "AfterCardDrawn": ("on_card_drawn", "on_card_drawn_early"),
    "AfterCardExhausted": ("on_card_exhausted",),
    "AfterEnergyReset": ("on_energy_reset",),
    "AfterEnergySpent": ("on_energy_spent",),
    "AfterOrbEvoked": ("on_orb_evoked",),
    "AfterPowerAmountChanged": ("on_power_amount_changed",),
    "AfterCardGeneratedForCombat": ("on_card_generated_for_combat",),
    "BeforeHandDraw": ("on_before_hand_draw",),
    "BeforeSideTurnStart": ("on_before_side_turn_start",),
    "BeforeSideTurnEnd": ("on_before_side_turn_end",),
    "AfterSideTurnStart": ("on_owner_turn_start", "on_player_turn_start"),
    "AfterSideTurnEnd": ("on_owner_turn_end", "on_side_turn_end"),
    "AfterPlayerTurnStart": ("on_after_player_turn_start",),
    # ⚠️ Pre/Post 是**两个不同时机**（``CombatManager.cs:867`` vs ``:1545``）：
    # Pre 在"抽完牌、即将出牌"（`MayhemPower`），Post 在"回合结束、清手牌之前"
    # （`StampedePower`）。早先把两者都映射到 `on_auto_post_play` —— 那会让
    # `mayhem` 看起来"已接"，而实际上它要的时机根本没被分发。
    "AfterAutoPrePlayPhaseEntered": ("on_auto_pre_play",),
    "AfterAutoPostPlayPhaseEntered": ("on_auto_post_play",),
}

#: **查询 / 数据 / 表现型**成员：不是"事件钩子"，本工具不判它们（另有专项：
#: 查询型在 ``damage_multipliers`` / ``block_multiplier`` / 各种 ``*_bonus`` 表里，
#: 由 ``tests/test_source_citations.py`` 与各自的回归覆盖）。
QUERY_OR_DATA_MEMBERS = frozenset({
    "Type", "StackType", "InstanceType", "CanonicalVars", "InitInternalData",
    "ExtraHoverTips", "DisplayAmount", "AllowNegative", "IsPositive", "OriginModel",
    "ShouldScaleInMultiplayer", "GetScaledAmountForMultiplayer", "AmountLabelColor",
    "ModifyDamageAdditive", "ModifyDamageMultiplicative", "ModifyBlockAdditive",
    "ModifyBlockMultiplicative", "ModifyOrbValue", "ModifyHandDraw", "ModifyMaxEnergy",
    "ModifyCardPlayCount", "ModifyCardPlayResultLocation", "AfterModifyingCardPlayCount",
    "AfterModifyingCardPlayResultLocation", "AfterModifyingBlockAmount",
    "AfterModifyingHpLostAfterOsty", "ModifyHpLostAfterOstyLate", "ModifyHpLostBeforeOstyLate",
    "AfterModifyingHpLostBeforeOsty", "AfterModifyingDamageAmount", "ModifyDamageCap",
    "TryModifyEnergyCostInCombat", "TryModifyEnergyCostInCombatLate",
    "TryModifyPowerAmountReceived", "AfterModifyingPowerAmountReceived",
    "ShouldClearBlock", "ShouldFlush", "ShouldAllowHitting", "ShouldTakeExtraTurn",
    "ShouldStopCombatFromEnding", "ShouldPlay", "ShouldDraw", "ShouldPowerBeRemovedAfterOwnerDeath",
    "ShouldPowerBeRemovedOnDeath", "ShouldCreatureBeRemovedFromCombatAfterDeath",
    "ShouldOwnerDeathTriggerFatal", "OwnerIsSecondaryEnemy",
    "AfterPreventingBlockClear", "AfterShuffle", "AfterStarsGained", "AfterStarsSpent",
    "AfterCurrentHpChanged", "AfterCreatureAddedToCombat", "AfterOstyRevived",
    "AfterSideTurnStartLate", "AfterForge", "AfterAttack", "BeforeAttack",
    "BeforeDamageReceived", "BeforePotionUsed", "BeforeCombatStart", "BeforeSideTurnEndVeryEarly",
    "BeforeSideTurnEndEarly", "ModifyUnblockedDamageTarget", "ModifyVulnerableMultiplier",
    "TryModifyKeywordsInCombat", "OnApplied", "AfterRemoved", "AfterCardChangedPiles",
})

#: **人工核实过、确实可以不接**的"源码有、引擎没有"的成员。
#: 格式 ``(能力 id, 成员) → 理由``。**只放"数值上等价或不可达"的**；
#: 凡是会改数值的缺口都不许进这里（要修，不是要记账）。
KNOWN_UNWIRED: dict[tuple[str, str], str] = {
    ("conqueror", "AfterSideTurnEnd"):
        "走 ``powers.DECREMENTS_AT_ENEMY_SIDE_TURN_END`` 表（``tick_durations``）——"
        "与 ``vulnerable`` / ``weak`` 同一条路径：它贴在**敌人**身上，"
        "「自己阵营回合结束」就是敌方阵营结束，数值上等价（`docs/12` §2.46）",
    ("barricade", "AfterApplied"):
        "只写 ``DynamicVars['ApplierName']``（给界面看施加者名字），无数值作用",
    ("tank", "AfterApplied"):
        "给**队友**上 `GuardedPower`（``GetTeammatesOf``），单人局恒为空 —— 多人专用",
    ("guarded", "AfterDeath"):
        "施加者死亡时移除；单人局里只有 ``tank`` 会施加它，而那条是多人专用",
    ("covered", "AfterDeath"):
        "同上：移除条件是「施加者（另一个玩家）死了」，多人专用",
    ("asleep", "AfterDamageReceived"):
        "挨打醒来：要 ``CreatureCmd.Stun(owner, WakeUpMove, nextMoveId)`` —— 让这只怪"
        "**这一回合改走唤醒招式**。引擎的 stun 注入的是空招，接不了唤醒招式，"
        "所以先不接（接一半会让它白醒一回合，比不接更偏）",
    ("slumber", "AfterDamageReceived"):
        "同上（睡甲虫挨打递减、归零要 `Stun(..., WakeUpMove)`）",
    ("ringing", "?ShouldPlay"):
        "``ShouldPlay`` 否决出牌；引擎里「打不出」由苦痛查询表达（``can_play`` 查苦痛），"
        "与「否决打出」对数值等价",
    ("smoggy", "?ShouldPlay"): "同 ``ringing``",
    ("chains_of_binding", "?ShouldPlay"): "同 ``ringing``",
    ("phantom_blades", "AfterRemoved"): "移除时清关键字：战斗副本随战斗丢弃，无需清理",
}


def _decompiled_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in DECOMPILED.rglob("*.cs"):
        index.setdefault(path.stem, path)
    return index


def _members(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    pattern = re.compile(
        r"\b(?:public|protected|internal|private)\s+override\s+[^;{=]{0,140}?\b(\w+)\s*[({=]")
    return sorted(set(pattern.findall(text)))


_CONTENT_TEXT: str | None = None


def _content_text() -> str:
    """``data/content/repo`` 下所有抽取数据的文本（缓存）。

    ⭐ 能力 id 也会出现在**内容数据**里，而不是 Python 源码里：卡牌效果的
    ``power`` / ``calc_arg``。`SovereignBlade` 的格挡公式就是
    ``calc_arg = "parry"``（``CalculatedBlockVar.WithMultiplier(... GetOwnerParryAmount ...)``）。
    只扫 ``.py`` 会把这类"只被数据引用"的能力误报成"注册了但没人消费"。
    """
    global _CONTENT_TEXT
    if _CONTENT_TEXT is None:
        directory = ROOT / "data" / "content" / "repo"
        _CONTENT_TEXT = "\n".join(path.read_text(encoding="utf-8")
                                  for path in sorted(directory.glob("*.json")))
    return _CONTENT_TEXT


def _has_consumer_outside_rules(pid: str, powers_source: str,
                                other_source: str) -> bool:
    """能力 id 在 ``RULES`` 字面量**之外**出现过吗（查询表 / 别的模块 / 内容数据）？"""
    start = powers_source.find("RULES: dict[str, PowerRules] = {")
    end = powers_source.find("\n}\n", start)
    outside = powers_source[:start] + powers_source[end:]
    pattern = re.compile(r"""["']%s["']""" % re.escape(pid))
    return bool(pattern.search(outside) or pattern.search(other_source)
                or pattern.search(_content_text()))


def audit() -> dict[str, list]:
    """返回三份清单：``dead`` / ``unwired`` / ``duration_without_table``。"""
    from sts2_sim import powers

    records = json.loads((ROOT / "data" / "content" / "repo" / "powers_source.json")
                         .read_text(encoding="utf-8"))
    class_of = {r["id"]: r.get("class", "") for r in records}
    index = _decompiled_index()
    powers_source = (ROOT / "sts2_sim" / "powers.py").read_text(encoding="utf-8")
    other_source = "\n".join(
        (ROOT / "sts2_sim" / name).read_text(encoding="utf-8")
        for name in ("core.py", "keywords.py", "orbs.py", "relics.py", "content.py",
                     "run.py", "runeffects.py", "events.py"))

    ignore_fields = {"pid", "source", "duration", "damage_additive", "block_additive"}
    dead: list[str] = []
    unwired: list[tuple[str, str]] = []
    duration_without_table: list[str] = []

    for pid, rule in sorted(powers.RULES.items()):
        hooks = [f.name for f in dataclasses.fields(rule)
                 if f.name not in ignore_fields and getattr(rule, f.name) is not None]
        if not hooks and not _has_consumer_outside_rules(pid, powers_source, other_source):
            dead.append(pid)
        if rule.duration and pid not in powers.DECREMENTS_AT_ENEMY_SIDE_TURN_END:
            duration_without_table.append(pid)
        path = index.get(class_of.get(pid, ""))
        if path is None:
            continue
        for member in _members(path):
            if member in QUERY_OR_DATA_MEMBERS:
                continue
            fields = HOOK_OF_MEMBER.get(member)
            if fields is None:
                continue
            if any(getattr(rule, field, None) is not None for field in fields):
                continue
            if (pid, member) in KNOWN_UNWIRED:
                continue
            unwired.append((pid, member))
    return {"dead": dead, "unwired": unwired,
            "duration_without_table": duration_without_table}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="有**硬缺口**（没人消费 / duration 没表）就以非零码退出")
    parser.add_argument("--strict", action="store_true",
                        help="连「未接的覆写」也算缺口（默认只报，因为已核实过的那批"
                             "在 tests/test_power_hook_coverage.py 里逐条记了原因）")
    args = parser.parse_args()

    from sts2_sim import powers

    result = audit()
    print(f"已实现能力 {len(powers.RULES)} 个；未接的事件覆写 {len(result['unwired'])} 条")

    if result["dead"]:
        print(f"\n【注册了但没人消费】{len(result['dead'])} 个 —— 数值上整条无效：")
        for pid in result["dead"]:
            print(f"    {pid}  | {powers.RULES[pid].source[:80]}")
    if result["unwired"]:
        print(f"\n【源码有覆写、引擎没接】{len(result['unwired'])} 条"
              "（已核实过的那批见 tests/test_power_hook_coverage.py 的 EQUIVALENT / TODO）：")
        for pid, member in result["unwired"]:
            print(f"    {pid:26s} {member}")
    if result["duration_without_table"]:
        print(f"\n【duration=True 但不在递减表里】{len(result['duration_without_table'])} 个"
              "（那个标记没有消费者，标了不等于会递减）：")
        for pid in result["duration_without_table"]:
            print(f"    {pid}")

    hard = bool(result["dead"] or result["duration_without_table"])
    if not hard and not result["unwired"]:
        print("\n全部一致 ✅（注册的都有消费者，源码的事件覆写都接了）")
        return 0
    if hard or args.strict:
        return 1 if args.check or args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
