"""已实现能力 vs 源码覆写方法的**覆盖率护栏**（``tools/audit_power_hooks.py``）。

为什么需要它：``content_status`` 的"能力 188/267 已实现"只统计**注册**，
而实测发现三类"注册了但没接上"的缺口 ——

* ``curl_up`` / ``hard_to_kill`` / ``slow`` **整条无效**（引擎里没有消费者）；
* ``double_damage`` / ``colossus`` / ``intangible`` / ``covered`` / ``panache``
  缺**到期时机**（永久生效 = 静默变强）；
* ``galvanic`` 缺"打出镀能牌要挨打"的**代价**那一半（白送 = 静默变弱）。

本文件把审计输出钉成回归：**不许出现新的未接覆写**。

* ``EQUIVALENT`` —— 已逐条回源码核过：引擎在**别处**（查询表 / 计数器 /
  另一个调用点 / 多人专用）表达了同一件事，附上「在哪」；
* ``TODO`` —— 已核实**确实是缺口**、但需要新机制（否决出牌 / 施加者死亡分发 /
  唤醒招式），留作后续轮次，**不许再悄悄多**。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 引擎在别处表达（已核过源码）。
EQUIVALENT: dict[tuple[str, str], str] = {
    ("vulnerable", "AfterSideTurnEnd"): "`DECREMENTS_AT_ENEMY_SIDE_TURN_END`（易伤/虚弱/脆弱同一张表）",
    ("weak", "AfterSideTurnEnd"): "同上",
    ("frail", "AfterSideTurnEnd"): "同上",
    ("tainted", "AfterSideTurnEnd"): "同上（TaintedPower 也是 duration）",
    ("free_attack", "BeforeCardPlayed"): "`COST_ZERO_POWERS`（攻击牌费用归零）",
    ("free_power", "BeforeCardPlayed"): "`COST_ZERO_POWERS`",
    ("free_skill", "BeforeCardPlayed"): "`COST_ZERO_POWERS`",
    ("veilpiercer", "BeforeCardPlayed"): "`COST_ZERO_POWERS`（虚无牌）",
    ("clarity", "AfterSideTurnStart"): "`HAND_DRAW_BONUS_POWERS`（每回合 +1 张）",
    ("draw_cards_next_turn", "AfterSideTurnStart"): "`HAND_DRAW_BONUS_POWERS`",
    ("tools_of_the_trade", "AfterPlayerTurnStart"): "`HAND_DRAW_BONUS_POWERS`",
    ("tyranny", "AfterPlayerTurnStart"): "`HAND_DRAW_BONUS_POWERS`",
    ("echo_form", "AfterApplied"): "`CARD_PLAY_COUNT_POWERS` + `cards_played_started_this_turn`",
    ("echo_form", "AfterCardPlayed"): "同上",
    ("echo_form", "BeforeSideTurnStart"): "同上（按回合隔离的计数）",
    ("void_form", "AfterApplied"): "`COST_ZERO_POWERS` + `cards_played_started_this_turn`",
    ("void_form", "AfterCardPlayed"): "同上",
    ("void_form", "BeforeSideTurnStart"): "同上",
    ("juggling", "AfterApplied"): "`attacks_played_this_turn`（真机也是同一个量）",
    ("juggling", "AfterSideTurnEnd"): "同上（按回合隔离）",
    ("skittish", "AfterSideTurnEnd"): "`power_used_turn` 回合戳，等价于「本回合已用」",
    ("possess_strength", "AfterPowerAmountChanged"): "`powers.possess_steal`（从 `add_power` 调用）",
    ("possess_speed", "AfterPowerAmountChanged"): "同上",
    ("vicious", "AfterPowerAmountChanged"): "`on_power_applied`（施加者是自己 + 易伤 + 量 > 0）",
    ("vital_spark", "AfterPowerAmountChanged"): "苦痛的量按**当前层数**读（引擎的 affliction 只存名字），"
                                                "重新贴一遍是空操作",
    ("underworld", "AfterDamageGiven"): "`powers.on_ally_damage_given`（专门的第二条分发）",
    ("star_next_turn", "AfterEnergyReset"): "`on_owner_turn_start`（引擎的回合开始同一步）",
    ("block_next_turn", "AfterBlockCleared"): "`on_owner_turn_start`（`start_player_turn` 里**清格挡之后**才分发）",
    ("doom", "BeforeSideTurnEnd"): "`dies_to_doom` 在敌方侧回合结束时检查（core.py:1579）",
    ("doom", "AfterSideTurnEnd"): "`dies_to_doom` 在玩家回合结束时检查（core.py:2008）",
    ("plating", "BeforeSideTurnStart"): "`on_owner_turn_start`（每次获得格挡时重算层数）",
    ("plating", "AfterApplied"): "只写展示用的层数，数值由 `_plating_tick` 给",
    ("hardened_shell", "BeforeSideTurnStart"): "`power_flags['hardened_shell_received']` 的累计与清零",
    ("shrink", "AfterApplied"): "只写施加者名字（展示）",
    ("covered", "AfterApplied"): "多人（给另一个玩家挂 `InterceptPower`），单人局恒为空",
    ("guarded", "AfterApplied"): "只写施加者名字（展示）",
    ("demon_form", "AfterApplied"): "只写施加者名字（展示）",
    ("reaper_form", "AfterApplied"): "只写施加者名字（展示）",
    ("serpent_form", "AfterApplied"): "只写施加者名字（展示）",
    ("ritual", "AfterApplied"): "记「是不是敌人刚贴的」（决定首回合是否跳过），已由 `_ritual` 的回合戳表达",
    ("illusion", "AfterApplied"): "只写展示变量",
    ("hellraiser", "AfterSideTurnEnd"): "每回合 9 次限流：引擎未实现，已在 RULES 说明里写明",
    ("surrounded", "AfterDeath"): "重算朝向：引擎用 `power_facing` + `back_attack_*` 一次性表达，不做死亡重算",
    ("surrounded", "BeforeCardPlayed"): "同上",
    ("debilitate", "AfterSideTurnEnd"): "`DECREMENTS_AT_ENEMY_SIDE_TURN_END`（`duration=True`；"
                                         "它只可能贴在敌人身上，写的是 `participants.Contains(Owner)`）",
    ("no_block", "AfterSideTurnEnd"): "同上（`if (side == CombatSide.Enemy) Decrement`，与递减表逐字同义）",
}

#: 已核实**确实是缺口**、待补机制（不许再新增）。
#:
#: 现在是**空的** —— 曾经记在这里的三类都补上了：
#: * ``constrict`` / ``hex`` / ``shrink`` 的 ``AfterDeath`` → ``on_applier_death``（施加者死亡）；
#: * ``chains_of_binding.BeforeCardPlayed`` → 接进 ``ShouldPlay`` 的查询路径（Bound）；
#: * ``asleep`` / ``slumber`` 的 ``AfterDamageReceived`` → ``Stun(creature, stunMove, nextMoveId)``
#:   的三段语义（唤醒动作 / 下一招固定 / 这一回合不动）。
TODO: dict[tuple[str, str], str] = {}


class TestPowerHookCoverage(unittest.TestCase):
    def test_no_unwired_overrides_beyond_the_reviewed_snapshot(self):
        from tools.audit_power_hooks import audit

        allowed = set(EQUIVALENT) | set(TODO)
        unexpected = [item for item in audit()["unwired"] if item not in allowed]
        self.assertEqual(
            unexpected, [],
            "出现了**新的**「源码有覆写、引擎没接」条目"
            "（先回源码核实，再决定修还是记账）："
            f"{unexpected}")

    def test_no_dead_registrations(self):
        """注册了却没有消费者的能力 = 数值上整条无效（``curl_up`` 曾经如此）。"""
        from tools.audit_power_hooks import audit

        self.assertEqual(audit()["dead"], [])

    def test_duration_flag_is_backed_by_the_table(self):
        """``duration=True`` 这个标记本身没有消费者 —— 必须在递减表里才算数。"""
        from tools.audit_power_hooks import audit

        self.assertEqual(audit()["duration_without_table"], [])


if __name__ == "__main__":
    unittest.main()
