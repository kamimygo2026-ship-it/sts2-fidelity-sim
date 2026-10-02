"""报告：**没进训练集的内容**按根因族聚类，并写出 `docs/14`。

`tools/blocker_report.py` 回答"每一条卡在哪"，`tools/gap_report.py` 写逐条清单
（`docs/10`），本工具回答第三个问题：**它们其实是同一件事**。

  * 卡牌：把源码抽取给的阻塞串归成**机制族**（选牌动作、动态公式、生成物入手、
    召唤、附魔…），给出每族卡住多少张与代表卡。
  * 能力：未实现的能力按**它重写了哪些钩子**聚类 —— 同一族往往一次补一个钩子位
    就能整批落地（对应 `docs/11` §11.2 的钩子分类）。
  * 遗物：按缺口**类别**（命令 / 查询 / 记账）与**时机**聚类。
  * 事件 / 药水：按不可跑的原因聚类。

用法：

    python -X utf8 tools/gap_clusters.py                      # 打印
    python -X utf8 tools/gap_clusters.py --write docs/14-未纳入内容与共性分析.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DECOMPILED = ROOT / "data" / "decompiled" / "sts2"

#: 卡牌的 `unsupported` 串 → 机制族。顺序即优先级（一个串只算进第一个命中的族，
#: 因为"选牌"通常是更根上的原因：没有选牌动作，生成物就无法落到具体牌上）。
CARD_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("选牌动作（没有 `CardSelectCmd` 语义）",
     ("选牌结果", "需要玩家选择", "CardSelectCmd", "从牌组里选", "select_card")),
    ("动态数值公式（`CalculatedVar` / `DynamicVars` 组合）",
     ("CalculatedVar", "量由运行期公式决定", "WithHitCount", "DynamicVars")),
    ("生成物放入手牌 / 抽牌堆",
     ("CreateInHand", "CreateInDrawPile", "AddGeneratedCard", "AddGeneratedCards")),
    ("自动打出", ("AutoPlay",)),
    ("召唤物（`OstyCmd` / 随从）", ("Summon", "OstyCmd")),
    ("锻造（`ForgeCmd`）", ("Forge",)),
    ("附魔（Enchantment）", ("nchant",)),
    ("药水生成", ("TryToProcure", "PotionCmd")),
    ("升级 / 降级牌", ("upgrade_card", "downgrade_card", "UpgradeCard")),
    ("未建模的钩子 / 触发器", ("未扫描的钩子", "OnChosen", "trigger")),
)

#: 能力重写的成员名 → 机制族（对应 `docs/11` §11.2 的钩子分类）。
POWER_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("数值修正（`Modify*`：伤害 / 格挡 / 费用 / 抽取）",
     ("ModifyDamage", "ModifyBlock", "ModifyEnergy", "ModifyCard", "ModifyHand",
      "ModifyXValue", "ModifyOrb", "ModifyPower", "ModifyHp", "ModifyAmount",
      "ModifyCost", "ModifyDraw", "ModifyReward")),
    ("回合级时机（`After`/`Before` × `Turn`）",
     ("TurnStart", "TurnEnd", "SideTurnStart", "SideTurnEnd")),
    ("卡牌动作响应（Played / Drawn / Exhausted / Generated）",
     ("CardPlayed", "CardDrawn", "CardExhausted", "CardAdded", "CardDiscarded",
      "CardGenerated", "AfterCard")),
    ("伤害 / 生命事件（Damaged / Death / Heal）",
     ("Damaged", "DamageReceived", "Death", "Heal", "Hp")),
    ("查询 / 决策型（`Should*`）", ("Should",)),
    ("能力施加响应（Applied / Removed）", ("PowerApplied", "Applied", "Removed")),
    ("充能球（`Orb*`）", ("Orb",)),
    ("其它时机钩子（`After*` / `Before*` / `On*`）", ("After", "Before", "On")),
    ("纯数据：只声明类型 / 栈规则（效果由别处驱动）",
     ("Type", "StackType", "OriginModel", "IsPositive", "IsVisible", "Amount")),
    ("没有重写任何成员（要单独看它怎么起作用）", ()),
)

#: 事件的不可跑原因 → 族（归并成"人话"，但保留原始计数）。
EVENT_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("附魔系统（选牌用途 / 附魔语义）", ("nchant", "附魔")),
    ("选项抽取缺口（页面上没有可用选项）",
     ("选项没抽出来", "没有可点的选项", "没有效果也没有去向")),
    ("进入条件解析不出（`IsAllowed` 写法未支持）", ("进入条件算不出",)),
    ("事件内战斗交接", ("事件战斗",)),
    ("选项 lambda / 闭包带参数", ("lambda 带参数",)),
    ("效果的量抽不出", ("效果的量抽不出",)),
    ("跨页动态变量改写", ("跨页累计",)),
    ("具体命令缺口（`效果缺口：…`）", ("效果缺口：",)),
)

#: 应当**排除而不是实现**的东西：表现层 / 多人 / 废弃内容。
#: 判据是"删掉它不影响任何数值与决策"（`docs/09` 铁律 5：能排除的就排除）。
EXCLUDE_MARKERS = (
    "Layout.RemoveNodesOnPortrait", "SetPortrait", "HoverTip", "Animation",
    "IsMultiplayer", "MultiplayerOnly", "Deprecated", "deprecated",
    "CompleteQuest", "MimicRestSiteHeal",
)

#: 能力实现**分批**用的判定依据（"这个能力现在做得了吗"）。
#:
#: * :data:`POWER_DATA_ONLY_MEMBERS`：不产生行为的成员（类型 / 栈 / 悬浮提示…），
#:   实现这类能力只需注册 + 施加语义。
#: * :data:`POWER_SUPPORTED_MEMBERS`：引擎**已经有**的钩子位 / 修正槽
#:   （对应 `PowerRules` 的字段与 `hooks.HOOKS`）。
#: * :data:`POWER_PLANNED_SLOTS`：还没做但**很小**的槽位（一处调用点即可）。
#: 三者之外的成员 = 这个能力要等新基建。
POWER_DATA_ONLY_MEMBERS = frozenset({
    "Type", "StackType", "ExtraHoverTips", "CanonicalVars", "DisplayAmount",
    "InstanceType", "AllowNegative", "InitInternalData", "PowerType",
})
POWER_SUPPORTED_MEMBERS = frozenset({
    "AfterCardPlayed", "AfterSideTurnEnd", "BeforeCardPlayed", "AfterApplied",
    "AfterPlayerTurnStart", "AfterCardDrawn", "AfterCardDrawnEarly",
    "ModifyDamageAdditive", "ModifyBlockAdditive", "AfterDamageGiven",
    "AfterEnergyReset", "AfterDeath", "AfterBlockGained", "AfterCardExhausted",
    "AfterPowerApplied",
    # ⭐ 这几个**总线已经在分发**（`hooks.HOOKS` 里有声明、core 里有调用点），
    # 只是早先漏在了这份名单外 —— 名单错一次，就会把能做的工作误判成"要等基建"。
    "BeforeSideTurnStart", "AfterDamageReceived",
    "AfterOwnerTurnStart", "AfterOwnerTurnEnd",
    # ⭐ `AfterSideTurnStart` 也**已经能用**：它对应 `on_owner_turn_start`
    # （`powers.on_turn_start` 的文档就写着 `Hook.AfterSideTurnStart`，而 core 在
    # **双方**回合开始各触发一次：core.py:1412 敌人 / core.py:1559 玩家）。
    # 源码里的 `participants.Contains(base.Owner)` 正是"自家阵营回合开始"，
    # 与"按拥有者分发"一一对应。
    "AfterSideTurnStart",
    # ⭐ 下面三个**管线已经存在**（现算过调用点与签名），但当前把"那一个能力"
    # 写死在函数里，实现时只需把它改成一张小表：
    #   * `ModifyCardPlayCount` / `AfterModifyingCardPlayCount`
    #     → `powers.card_play_count(state, card)`（拿得到 **card**，所以"牌型/拥有者"
    #       这类条件可以就地判断）+ `on_card_play_count_modified`
    #   * `ModifyMaxEnergy` → `powers.max_energy_bonus(state)`（拿得到 **player**）
    "ModifyCardPlayCount", "AfterModifyingCardPlayCount", "ModifyMaxEnergy",
    # ⭐ 又一批**已经接上**的时机（本轮起逐批补的调用点）。名单不跟着更新，
    # 报告就会把"已经能做"的能力一直挂在"要等基建"里 —— 而人只看这张表。
    "AfterCardEnteredCombat",       # `_add_generated_card` 里分发（第 24 轮）
    "AfterBlockCleared",            # `start_player_turn` / `_run_enemy_turn`（第 18 轮）
    "AfterEnergySpent",             # `play_card` 扣费之后（第 19 轮）
    "AfterOrbEvoked",               # `orbs.evoke` 之后（第 19 轮）
    "AfterCardGeneratedForCombat",  # `_add_generated_card`（第 8 轮）
    "BeforeHandDraw",               # `start_player_turn` 抽牌前（第 8 轮）
    "AfterAutoPrePlayPhaseEntered",  # `start_player_turn` 末尾（第 29 轮）
    "AfterAutoPostPlayPhaseEntered",  # 回合结束（早先）
    "AfterStarsSpent", "AfterStarsGained",   # 第 31 轮
    "ModifyCardPlayResultLocation", "AfterModifyingCardPlayResultLocation",  # 第 16 轮
    "TryModifyEnergyCostInCombat", "TryModifyEnergyCostInCombatLate",  # 第 19 轮
    "ModifyOrbValue", "ShouldClearBlock", "ShouldFlush", "AfterPowerAmountChanged",
})
POWER_PLANNED_SLOTS = frozenset({"AfterRemoved", "BeforeSideTurnEnd"})

#: **子系统级**的身体缺口：出现它们就说明这个能力要等一整个子系统，
#: 不能按"补一个小调用点"排期（成员缺口会被它们盖过，见 `buff_waves`）。
SUBSYSTEM_BLOCKERS = frozenset({
    "Osty 召唤物", "锻造（ForgeCmd）", "附魔", "随从",
})

#: 这些成员即使"管线存在"也算 Wave B：**管线当下拿不到判断条件所需的信息**。
#: 典型是 `ModifyDamageMultiplicative`：签名只有 `(attacker, defender)`，
#: 而源码条件要 `props.IsPoweredAttack()` 与 `cardSource != null` —— 拿不到就没法判，
#: 硬用无条件倍率就是**静默变强**。修法是让数值修正能带谓词（同一件事）。
POWER_PREDICATE_PIPELINE_MEMBERS = frozenset({
    "ModifyDamageMultiplicative", "ModifyUnblockedDamageTarget",
    "TryModifyEnergyCostInCombat", "TryModifyEnergyCostInCombatLate",
    "ModifyCardPlayResultLocation", "AfterModifyingCardPlayResultLocation",
})

#: **方法体级**的阻塞判据：成员受支持 ≠ 能做 —— 方法体里可能调用了引擎没有的**原语**。
#: 例：`EntropyPower` 只重写 `AfterPlayerTurnStart`（成员受支持），但体内是
#: `CardSelectCmd.FromHand` + `CardCmd.Transform`（选牌动作，Wave B）；
#: `FurnacePower` 体内是 `ForgeCmd.Forge`。
#: token → 阻塞项名（与 `POWER_INFRA_COST` 的键对应，便于合并统计）。
POWER_BODY_BLOCKERS: tuple[tuple[str, str], ...] = (
    ("CardSelectCmd", "选牌动作"), ("CardSelectorPrefs", "选牌动作"),
    ("ForgeCmd.Forge", "锻造（ForgeCmd）"),
    ("OstyCmd", "Osty 召唤物"),
    # ⚠️ 不止 `OstyCmd`：`CalcifyPower` 的判据是 ``dealer?.Monster is Osty``
    # （"随从打出的伤害才加成"），只认 `OstyCmd` 会让它落到"条件化数值修正"那个
    # 泛泛的桶里 —— 而它真正缺的是**随从子系统**（`docs/09` 铁律 5：先修对的缺口）。
    ("is Osty", "Osty 召唤物"),
    ("Enchant", "附魔"),
    ("AutoPlay", "自动打出"),
    # ⚠️ `CreateInHand` / `AddGeneratedCard` **已经从这张表里去掉**：引擎早就能把
    # 生成的牌放进手牌 / 抽牌堆 / 弃牌堆（`add_copies_to_hand`、`_add_generated_card`
    # 支持 top/random），留着只会把 `calamity` / `soulbound` 误判成"要等基建"。
    ("OrbCmd.EvokeLast", "唤起最后一个球（新算子）"),
    ("PotionCmd", "药水生成"),
    ("CardCmd.Transform", "战斗内变形"),
    ("ExtraTurn", "额外回合"),
    ("PowerCmd.ModifyAmount", "能力数值修改"),
    ("ApplyKeyword", "实例级关键字（ApplyKeyword）"),
    ("ApplySingleTurnRetain", "单回合保留"),
    ("LoseBlock", "失去格挡算子"),
    ("Stun", "击晕算子"),
) 

#: Wave B 的基建**成本分级**（人工判断，只影响"先做哪个"，不影响数字）。
#: 判据：**小** = 现有管线加一个槽位/调用点；**中** = 要改一条既有管线的语义
#: （能量 / 抽牌 / 打牌次数 / 生成物落位）；**大** = 需要新子系统（随从 Osty、
#: 附魔、锻造）；**排除** = 表现层与多人，不该做。
POWER_INFRA_COST: dict[str, tuple[str, str]] = {
    "ModifyDamageMultiplicative": ("小", "伤害管线已有倍率表，补 per-power 注册即可"),
    "AfterPowerAmountChanged": ("小", "只需一个调用点（docs/12 G09 已记它只在 apply_power 派发）"),
    "AfterCombatEnd": ("小", "战斗结束时机已有（遗物侧在用），补能力侧槽位"),
    "ModifyMaxEnergy": ("小", "能量上限管线已有"),
    "BeforeSideTurnStart": ("小", "回合开始时机已有，补能力侧槽位"),
    "BeforeDeath": ("小", "死亡路径已集中到 mark_dead，补「死前」钩子"),
    "AfterStarsGained": ("小", "星资源管线已有"),
    "AfterStarsSpent": ("小", "星资源管线已有"),
    "ModifyOrbValue": ("小", "充能球数值管线已有"),
    "AfterCurrentHpChanged": ("小", "生命变化是集中的"),
    "AfterEnergySpent": ("小", "能量消耗是集中的"),
    "AfterModifyingBlockAmount": ("小", "格挡结算点已有"),
    "AfterCardEnteredCombat": ("小", "卡进入战斗有明确入口"),
    "ShouldClearBlock": ("小", "查询型，Blur 一个能力用"),
    "AfterPreventingBlockClear": ("小", "同上，成对出现"),
    "ModifyHandDraw": ("中", "要改抽牌数量的语义"),
    "BeforeHandDraw": ("中", "与 ModifyHandDraw 成对，一起做"),
    "TryModifyEnergyCostInCombatLate": ("中", "要改费用计算管线"),
    "TryModifyEnergyCostInCombat": ("中", "同上"),
    "ModifyCardPlayCount": ("中", "「打两次」要改打牌管线（Burst/EchoForm 一族）"),
    "AfterModifyingCardPlayCount": ("中", "同上"),
    "ModifyCardPlayResultLocation": ("中", "要改「打出的牌去哪」（消耗/回手）"),
    "AfterModifyingCardPlayResultLocation": ("中", "同上"),
    "AfterCardGeneratedForCombat": ("中", "生成物入口要收敛成一处"),
    "AfterAttack": ("中", "**已接**：按「伤害事件」的（painful_stabs / skittish / suck）走 "
                          "on_damage_given / on_attacked；按「指令」的（gigantification）走 "
                          "powers.on_attack_command_finished —— 新能力先判自己属于哪一种"),
    "BeforeAttack": ("中", "半接：gigantification 用「伤害惰性捕获」等价表达；"
                           "要看指令对象本身的能力还没接"),
    "AfterAutoPrePlayPhaseEntered": ("中", "与自动打牌阶段有关（docs/13 B2 执行帧）"),
    "AfterModifyingHpLostAfterOsty": ("大", "Osty（召唤物）子系统未实现"),
    "ModifyHpLostAfterOstyLate": ("大", "同上"),
    "AfterForge": ("大", "锻造（ForgeCmd）未实现"),
    "GetScaledAmountForMultiplayer": ("排除", "多人"),
    "ShouldPlayVfx": ("排除", "表现层"),
    "IsVisibleInternal": ("排除", "表现层"),
    "AfterOstyRevived": ("大", "Osty（召唤物）子系统未实现"),
    "ShouldTakeExtraTurn": ("中", "额外回合语义未建模"),
    "AfterTakingExtraTurn": ("中", "同上"),
    "ShouldAllowHitting": ("中", "目标可打性查询"),
    "ShouldCreatureBeRemovedFromCombatAfterDeath": ("中", "死亡后是否离场的查询"),
    "ShouldPowerBeRemovedAfterOwnerDeath": ("小", "死亡清理点已有"),
    "ModifyUnblockedDamageTarget": ("中", "要改未格挡伤害的落点"),
    "AfterCreatureAddedToCombat": ("小", "加入战斗有明确入口"),
    "AfterDamageReceived": ("小", "受到伤害时机已有（遗物侧在用）"),
    "AfterBlockCleared": ("小", "格挡清空有明确位置，补一个调用点"),
    "实例级关键字（ApplyKeyword）": ("中", "卡实例目前没有关键字槽（关键字只在 CardDef 上）"),
    "条件化数值修正": ("中", "additive/multiplicative 目前只有开关语义，"
                            "要带谓词（牌标签 / ValueProp / 拥有者）才能表达源码条件"),
    # ⚠️ 伪成员：不是源码成员名，而是「该排除」的桶（见 POWER_MULTIPLAYER_MARKERS）。
    "多人专用（GetTeammates 一族）": ("排除", "单人局里「别的玩家」恒为空集 → 效果恒等于"
                                                 "原值；按 `docs/09` 铁律 5 排除而非近似"),
    "不可达（源码里没有施加者）": ("排除", "整个反编译树里除自身文件与自动生成的类型表外"
                                        "**无人引用**，内容侧（卡/遗物/药水/事件）也搜不到 ——"
                                        " 没有施加者就没有可观测行为，实现它是白写"),
    "卡牌消费者（能力本身是纯标记）": ("中", "行为写在**卡牌**里（卡去查这个能力），"
                                          "引擎缺「按能力改写这张牌的目标/数值」的槽位"),
    "卡牌动态变量 → 能力实例（SetDamage 一族）": ("中", "卡牌 `OnPlay` 里 "
                                                  "`Apply<X>()?.SetDamage(DynamicVars[...])` / "
                                                  "`SetBlock(...)`：抽取器没产出这个算子，"
                                                  "硬实现会让升级后的卡沿用能力自带常量（静默变弱）"),
    "Run 侧金币与地图历史": ("大", "`PlayerCmd.LoseGold` / `MarkLootStolen` 属于 Run 层，"
                                "战斗引擎拿不到玩家金币与地图历史"),
}

#: 「多人专用」这个**伪成员**的名字（`POWER_INFRA_COST` 的键之一）。
#: 单独领出来是因为它只用于渲染排除桶，`blocker_examples` 要跳过它。
MULTIPLAYER_BLOCKER = "多人专用（GetTeammates 一族）"

#: **源码里没有施加者**的能力：整个反编译树只有它自己和自动生成的类型表提到它，
#: 内容侧（卡牌 / 遗物 / 药水 / 事件）一处引用都没有。核过 `docs/14` §7 的 Wave A
#: 清单后发现 `gravity` 属于这一类 —— 与其实现一个永远不会被贴上的能力，
#: 不如把它明确标成"不可达"。
UNREACHABLE_POWERS: dict[str, str] = {
    "gravity": "除 `GravityPower.cs` 与 `AbstractModelSubtypes.cs`（生成物）外无人引用 "
               "`GravityPower`；relics / potions / events / cards 的抽取结果里也搜不到",
}
UNREACHABLE_BLOCKER = "不可达（源码里没有施加者）"

#: Wave A 的**假阳性**：工具只看"重写的成员能不能表达"，下面这些能力成员都命中，
#: 但缺的东西**不在成员清单里**（行为在卡牌里 / 缺一个抽取算子 / 缺 Run 层）。
#: 格式 ``pid → (类别, 证据)``，类别会作为"缺失成员"列进 Wave B 表。
WAVE_A_FALSE_POSITIVES: dict[str, tuple[str, str]] = {
    "fan_of_knives": (
        "卡牌消费者（能力本身是纯标记）",
        "`Shiv.cs` 的 `TargetType` getter：`if (HasFanOfKnives) return TargetType.AllEnemies;`"),
    "parry": (
        "卡牌消费者（能力本身是纯标记）",
        "类注释写明「不做任何事」：`SovereignBlade` 卡去查它并改自己的格挡"),
    "seeking_edge": (
        "卡牌消费者（能力本身是纯标记）",
        "类注释写明行为在 `SovereignBlade` 一族卡里（`SovereignBlade` 本身还没被抽出来）"),
    "the_hunt": (
        "卡牌消费者（能力本身是纯标记）",
        "类注释：这个能力只是给玩家看的指示器，真正的行为在 `TheHunt` 卡里"),
    "the_bomb": (
        "卡牌动态变量 → 能力实例（SetDamage 一族）",
        "`TheBomb.OnPlay`：`Apply<TheBombPower>(..., Turns).SetDamage(DynamicVars[\"BombDamage\"])`"
        "（升级 +10 改的是**卡牌**变量）"),
    "thievery": (
        "Run 侧金币与地图历史",
        "`ThieveryPower.Steal()` 走 `PlayerCmd.LoseGold`，还要 `MarkLootStolen` 记账"),
    "sword_sage": (
        "卡牌消费者（能力本身是纯标记）",
        "`TryAddReplays` 只对 ``card is SovereignBlade`` 生效，而 `SovereignBlade` 自己"
        "还没被抽出来（`unsupported` 里有 `AfterCardChangedPiles` 与运行期 `CalculatedBlock`）"
        "—— 实现了也没有牌吃"),
    "nightmare": (
        "外部注入的实例数据",
        "行为要 `Data.selectedCard`（由**卡牌** `Nightmare.OnPlay` 用 `SetSelectedCard` 塞进来），"
        "而那个 setter 抽取器认不出（`unsupported` 里如实记了）—— 能力侧缺的不是钩子"),
}

#: **多人专用**的源码标记：出现这些就说明该能力的行为建立在"其他玩家"之上
#: （`GetTeammates` 返回队友、`PlatformUtil.GetPlayerName` 取玩家名、
#: `RunManager.Instance.NetService` 取网络平台句柄）。单人局里队友集合恒为空，
#: 效果恒等于原值 —— 属于「能排除就排除」，不该排进基建批次白花一轮。
#:
#: ⚠️ **只收这三个**：裸 `NetId` 太宽 —— `heist` / `swipe` 只是用它写地图历史
#: （`MarkLootReturned`），真正的缺口在 Run 侧奖励，不是多人。
POWER_MULTIPLAYER_MARKERS = ("GetTeammates", "GetPlayerName", "NetService")

#: §7.2 的**人工核实结论**（工具报嫌疑，人回源码与引擎里逐条确认后落到这里）。
#: 格式：``pid → (结论, 证据)``。结论只有三种：
#: ``已核实：结构上成立`` / ``已核实：残余简化`` / ``确认变强（需修）``。
SILENT_BUFF_VERDICTS: dict[str, tuple[str, str]] = {
    "strength": (
        "已核实：结构上成立",
        "`compute_damage` 全仓库只有**一个**调用点（`deal_damage`，core.py:725），"
        "而 `unpowered=True` 的分支在它之**前**就返回（core.py:722）—— "
        "所以「`props.IsPoweredAttack()`」这一条由调用路径保证；"
        "「`base.Owner != dealer`」由 `attacker.power(pid)` 按**经手者**查表保证。"),
    "dexterity": (
        "已核实：结构上成立",
        "`gain_block` 里 `unpowered=True` 直接跳过 `compute_block`（core.py:607-609），"
        "对应源码的 `props.IsPoweredCardOrMonsterMoveBlock()`；"
        "「按拥有者查表」对应 `cardSource.Owner == base.Owner`。"),
    "vigor": (
        "已核实：残余简化（顺序执行下等价）",
        "源码还把加成**绑定到下一条命令/同一张牌**"
        "（`internalData.commandToModify` 两处判断），引擎的语义是"
        "「下一次吃力量的攻击 +层数、用后移除」。单体顺序执行下两者结果相同，"
        "**但引擎没有对应用例**，也没有实现那种绑定。"),
}

#: §7.2 的残余问题：上表核实完之后**仍然没有答案**的地方（留给下一批做用例）。
SILENT_BUFF_OPEN_QUESTIONS: tuple[str, ...] = (
    "「powered 但**不是攻击**」的伤害/格挡在真机里存在吗？"
    "若存在（例如某张技能牌造成非 Unpowered 伤害），它现在会吃到力量 —— "
    "需要一条**反例用例**把这条边界钉住。",
    "怪物招式的格挡走的是 `IsPoweredCardOrMonsterMoveBlock()`，"
    "引擎目前只按 `unpowered` 标志区分；怪物的非 Unpowered 格挡是否吃玩家敏捷需要确认。",
)


def classify(text: str, table) -> str:
    for family, tokens in table:
        if any(token in text for token in tokens):
            return family
    return "其它（一次性命令，长尾）"


def load_content(directory: Path):
    from sts2_sim import content
    return content, content.load_content_dir(directory)


def card_clusters(content) -> dict:
    """卡牌：按**主因**聚类（命令级根因优先，其次能力 / 机制变量 / 文本）。"""
    from sts2_sim import eligibility

    records = {r["cid"]: r for r in json.loads(
        (ROOT / "data" / "content" / "repo" / "cards_source.json")
        .read_text(encoding="utf-8"))}
    families: dict[str, list[str]] = defaultdict(list)
    reasons: Counter[str] = Counter()
    excluded: list[str] = []
    for cid in content.CARD_DB:
        admission = eligibility.card_admission(cid)
        if admission.admitted:
            continue
        record = records.get(cid, {})
        detail = admission.reasons()
        for reason in detail:
            reasons[reason.split(":")[0]] += 1
        unsupported = [str(x) for x in (record.get("unsupported") or [])]
        if any(marker in "; ".join(unsupported) for marker in EXCLUDE_MARKERS) \
                or str(record.get("class", "")).startswith("Deprecated"):
            excluded.append(cid)
            continue
        unconsumed = (content._unconsumed_mechanics(record)
                      if hasattr(content, "_unconsumed_mechanics") else [])
        triggers = record.get("triggers") or {}
        if unsupported:
            family = classify("; ".join(unsupported), CARD_FAMILIES)
        elif unconsumed:
            family = "声明了机制变量但没接到效果（`Repeat` / `Increase` / `Cards`…）"
        elif triggers:
            family = "手牌 / 时点触发器未建模（`on_turn_end_in_hand` 等）"
        elif any(r.startswith("unimplemented_power") for r in detail):
            family = "施加了未实现的能力（能力缺口）"
        elif any(r.startswith("effects_from_community_text") for r in detail):
            family = "效果只来自社区文本（源码抽不出可执行效果）"
        elif "no_effects" in detail:
            family = "源码里本来就没有可执行效果（诅咒 / 状态 / 任务牌）"
        elif not record:
            family = "只来自社区数据（没进源码抽取）"
        else:
            family = "其它（需逐条看）"
        families[family].append(cid)
    blocked = sum(1 for cid in content.CARD_DB
                  if not eligibility.card_admission(cid).admitted)
    return {"total": len(content.CARD_DB), "blocked": blocked, "reasons": reasons,
            "families": dict(sorted(families.items(), key=lambda kv: -len(kv[1]))),
            "excluded": excluded}


def power_clusters(content) -> dict:
    """能力：未实现的能力按**重写的钩子**聚类，并统计引用它的卡数。"""
    from sts2_sim import powers

    records = json.loads((ROOT / "data" / "content" / "repo" / "powers_source.json")
                         .read_text(encoding="utf-8"))
    missing = [r for r in records if r["id"] not in powers.RULES]
    by_class = {r["id"]: r.get("class", "") for r in records}
    by_type = {r["id"]: r.get("type") for r in records}

    implemented = set(powers.RULES)
    class_to_pid = {r.get("class", ""): r["id"] for r in records}
    cards_by_power: dict[str, list[str]] = defaultdict(list)
    for cid, definition in content.CARD_DB.items():
        for effect in (*definition.effects, *definition.upgrade):
            if effect.power:
                cards_by_power[str(effect.power)].append(cid)

    index: dict[str, Path] = {}
    for path in DECOMPILED.rglob("*.cs"):
        index.setdefault(path.stem, path)
    override_re = re.compile(
        r"\b(?:public|protected|internal|private)\s+override\s+[^;{=]{0,140}?\b(\w+)\s*[({=]")
    families: dict[str, list[str]] = defaultdict(list)
    members_by_power: dict[str, list[str]] = {}
    conditional_modifier: dict[str, bool] = {}
    body_blockers: dict[str, list[str]] = {}
    multiplayer: dict[str, str] = {}
    for record in missing:
        path = index.get(by_class.get(record["id"]) or "")
        members: list[str] = []
        source_text = ""
        if path is not None:
            source_text = path.read_text(encoding="utf-8-sig", errors="replace")
            members = sorted(set(override_re.findall(source_text)))
        members = [m for m in members
                   if not method_is_presentation(source_text, m)]
        members_by_power[record["id"]] = members
        conditional_modifier[record["id"]] = any(
            numeric_modifier_is_conditional(source_text, member)
            for member in ("ModifyDamageAdditive", "ModifyBlockAdditive"))
        blockers_here = {name for token, name in POWER_BODY_BLOCKERS
                         if token in source_text}
        # ⭐ 「依赖另一个**还没实现**的能力」也是一种阻塞：源码里
        # `PowerCmd.Apply<DoubleDamagePower>(...)` 需要那个能力先能用。
        # 只看成员会把它误判成 Wave A（成员都支持，但被依赖的能力不在表里）。
        for klass in set(re.findall(r"Apply<(\w+)>", source_text)):
            dependency = class_to_pid.get(klass)
            if dependency and dependency not in implemented:
                blockers_here.add(f"依赖未实现的能力：{dependency}")
        body_blockers[record["id"]] = sorted(blockers_here)
        # ⭐ **多人专用**：源码里出现 "队友 / 玩家名 / 网络平台" 这三样之一的，
        # 它的行为在单人局里恒等于原值（队友集合为空）—— 该排除，不该进基建批次。
        hit = next((m for m in POWER_MULTIPLAYER_MARKERS if m in source_text), "")
        if hit:
            multiplayer[record["id"]] = hit
        families[classify(" ".join(members), POWER_FAMILIES)].append(record["id"])
    return {"total": len(records), "implemented": len(powers.RULES),
            "families": dict(sorted(families.items(), key=lambda kv: -len(kv[1]))),
            "cards_by_power": cards_by_power,
            "members_by_power": members_by_power,
            "conditional_modifier": conditional_modifier,
            "body_blockers": body_blockers,
            "multiplayer": multiplayer,
            "missing": [r["id"] for r in missing],
            "by_type": by_type}


def silent_buff_suspects(content) -> list[dict]:
    """已实现的能力里，**用无条件开关表达了带条件的源码**的嫌疑名单。

    ⚠️ 这是**启发式审计**，不是判决：命中只说明"值得逐条回源码看"。
    判据：规则里 `damage_additive` / `block_additive` 为真（无条件加层数），
    而源码里的 `ModifyDamageAdditive` / `ModifyBlockAdditive` 里有 `if` 守卫。

    为什么必须有这条检查：丢掉条件的能力**只是伤害高几点**，没有报错、没有崩溃，
    卡牌照样能打 —— 这类"静默变强"是本项目最危险的一类 bug（`docs/09` 铁律 R1、
    `docs/12` §2.6 的遗物同类教训）。
    """
    from sts2_sim import powers

    records = json.loads((ROOT / "data" / "content" / "repo" / "powers_source.json")
                         .read_text(encoding="utf-8"))
    class_of = {r["id"]: r.get("class", "") for r in records}
    index: dict[str, Path] = {}
    for path in DECOMPILED.rglob("*.cs"):
        index.setdefault(path.stem, path)

    suspects: list[dict] = []
    for pid, rules in powers.RULES.items():
        flags = [(member, getattr(rules, field, False))
                 for member, field in (("ModifyDamageAdditive", "damage_additive"),
                                       ("ModifyBlockAdditive", "block_additive"))]
        if not any(flag for _member, flag in flags):
            continue
        path = index.get(class_of.get(pid, ""))
        if path is None:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for member, flag in flags:
            if flag and numeric_modifier_is_conditional(text, member):
                suspects.append({"pid": pid, "member": member,
                                 "source": f"{class_of.get(pid, '')}.{member}"})
    return suspects


def method_is_presentation(text: str, member: str) -> bool:
    """这个方法是不是**只有表现层**（VFX / 闪一下 / 什么都没做）？

    为什么需要它：``AfterRemoved`` 这类成员在"形态"能力里**只是关掉特效**
    （``ReaperFormPower.AfterRemoved`` 只有 ``Vfx?.SetActive(false)``），
    把它算成"缺失的机制槽位"会把已经能做的能力误判成要等基建。
    判据：方法体里除了 VFX/闪效类调用之外没有任何状态改动。
    """
    match = re.search(
        rf"(?:public|protected|internal)\s+override\s+[\w<>?,\[\] ]+\s+{re.escape(member)}\s*\(",
        text)
    if match is None:
        return False
    body = text[match.start():]
    body = body[:body.find("\n\tpublic", 1) if "\n\tpublic" in body[1:] else 1200]
    presentation_tokens = ("Vfx", "Flash()", "SetActive", "AddChild", "IsValid()",
                           "Task.CompletedTask", "return;")
    stripped = body
    for token in presentation_tokens:
        stripped = stripped.replace(token, "")
    # 去掉空白、括号、修饰符之后如果只剩骨架，就是纯表现
    leftover = re.sub(r"[\s{}()\[\];=?.!,+\-*/<>&|]|public|protected|private|override|async|Task|void|bool|decimal|int|await|return|true|false|null|new|this|base", "", stripped)
    return len(leftover) < 30


def numeric_modifier_is_conditional(text: str, member: str) -> bool:
    """源码里的这个数值修正成员**带条件**吗？

    ⚠️ 这一条是"静默变强"的判据：引擎的 `damage_additive` / `block_additive` 现在是
    **开关**（`True` = 无条件把层数加进去），而真机的
    `ModifyDamageAdditive` 几乎都带着谓词：

        if (base.Owner != dealer) return 0m;
        if (!props.IsPoweredAttack()) return 0m;
        if (!card.Tags.Contains(CardTag.Shiv)) return 0m;

    用开关去表达带条件的源码 = **丢掉条件 = 能力变强**，而症状只是"伤害高了几点"，
    没有任何报错。所以这类能力算 Wave B：要先让修正槽能带谓词。
    """
    match = re.search(
        rf"(?:public|protected|internal)\s+override\s+decimal\s+{re.escape(member)}\s*\(", text)
    if match is None:
        return False
    body = text[match.start():match.start() + 2400]
    # 只看方法体前若干行的判断（截断了也不会漏掉开头的守卫）
    return "if (" in body or "if(" in body


def buff_waves(powers_: dict) -> dict:
    """把**未实现的 buff** 分成两波：现有基建能做的 / 需要新基建的。

    “能做”的判据是：它重写的成员全部落在"纯数据 + 引擎已有钩子 + 计划中的小槽位"里，
    **并且**它的数值修正是无条件的（带谓词的修正没法用开关表达，见
    :func:`numeric_modifier_is_conditional`）。
    输出的 `blockers` 反过来就是下一波要补的**基建清单**（按影响的能力数排序）——
    这份清单比"还要实现 127 个能力"有用得多，因为补一个特性往往解锁一组能力。
    """
    ok = POWER_DATA_ONLY_MEMBERS | POWER_SUPPORTED_MEMBERS | POWER_PLANNED_SLOTS
    wave_a: list[str] = []
    wave_b: dict[str, list[str]] = {}
    excluded: dict[str, str] = {}
    blockers: dict[str, list[str]] = defaultdict(list)
    for pid, members in powers_["members_by_power"].items():
        if powers_["by_type"].get(pid) != "buff":
            continue
        if pid in powers_.get("multiplayer", {}):
            # 多人专用 → 单独一桶（成本「排除」），不要占着某个成员说"补个注册即可"。
            excluded[pid] = powers_["multiplayer"][pid]
            blockers["多人专用（GetTeammates 一族）"].append(pid)
            continue
        if pid in UNREACHABLE_POWERS:
            # 源码里没有任何施加者 → 排除（不是"待做"）。
            excluded[pid] = "不可达"
            blockers[UNREACHABLE_BLOCKER].append(pid)
            continue
        if pid in WAVE_A_FALSE_POSITIVES:
            # 成员都命中，但缺的东西**不在成员清单里**（行为在卡里 / 缺抽取算子 / 缺 Run 层）。
            member = WAVE_A_FALSE_POSITIVES[pid][0]
            wave_b[pid] = [member]
            blockers[member].append(pid)
            continue
        # ⭐ **子系统级的身体缺口优先于成员缺口**：`necro_mastery` 的成员
        # `AfterCurrentHpChanged` 看着只是"补个小调用点"，可方法体里的判据是
        # ``creature.Monster is Osty && PetOwner == Owner`` —— 真正缺的是**随从子系统**。
        # 只报成员会让人以为一轮能做完，实际要动的是整个子系统。
        subsystem = [b for b in powers_["body_blockers"].get(pid, ())
                     if b in SUBSYSTEM_BLOCKERS]
        if subsystem:
            wave_b[pid] = subsystem
            for member in subsystem:
                blockers[member].append(pid)
            continue
        missing = [m for m in members if m not in ok]
        if not missing:
            # ⚠️ **先看方法体里的原语缺口**（随从 / 锻造 / 选牌…），再看"数值修正带条件"。
            # 反过来的话，`calcify` 会被归到"条件化数值修正"（它确实带谓词），
            # 而它真正缺的是**随从 Osty 子系统** —— 修错了地方白花一轮。
            missing = list(powers_["body_blockers"].get(pid, ()))
        if not missing and powers_["conditional_modifier"].get(pid):
            # 成员都支持、方法体也没有缺失原语，但数值修正带条件 → 用开关表达会变强
            missing = ["条件化数值修正"]
        if missing:
            wave_b[pid] = missing
            for member in missing:
                blockers[member].append(pid)
        else:
            wave_a.append(pid)
    by_impact = sorted(wave_a, key=lambda p: -len(powers_["cards_by_power"].get(p, ())))
    return {
        "wave_a": by_impact,
        "wave_b": wave_b,
        "excluded": excluded,
        "blockers": dict(sorted(blockers.items(), key=lambda kv: -len(kv[1]))),
    }


def relic_clusters(content) -> dict:
    """遗物：按缺口类别（命令 / 查询 / 记账）与时机聚类。"""
    kinds: Counter[str] = Counter()
    timings: Counter[str] = Counter()
    by_kind: dict[str, list[str]] = defaultdict(list)
    for rid, definition in content.RELICS.items():
        for hook, kind in (getattr(definition, "unmodeled_hook_kinds", ()) or ()):
            kinds[str(kind)] += 1
            timings[str(hook)] += 1
            if len(by_kind[str(kind)]) < 6:
                by_kind[str(kind)].append(rid)
    return {"kinds": kinds, "timings": timings, "by_kind": dict(by_kind)}


def blocker_examples(waves: dict, limit: int = 14, max_lines: int = 22) -> dict:
    """给 Wave B 的每个缺失成员附上一段**真实源码**，省掉下一轮的来回找。

    为什么值得放进报告：Wave B 的工作量不在"写 Python"，而在"把源码里的条件逐条读懂"。
    把"这个成员长什么样、哪个能力在用"直接摆在清单里，下一轮就能照着实现，
    而不是先花半天在 3538 个 `.cs` 里翻。

    每个成员取**第一个**用到它的能力的方法体（截断），其余只列能力名。
    """
    from sts2_sim import powers as power_module  # noqa: F401  (保持与实现同源)

    index: dict[str, Path] = {}
    for path in DECOMPILED.rglob("*.cs"):
        index.setdefault(path.stem, path)
    records = json.loads((ROOT / "data" / "content" / "repo" / "powers_source.json")
                         .read_text(encoding="utf-8"))
    class_of = {r["id"]: r.get("class", "") for r in records}

    out: dict[str, dict] = {}
    for member, pids in list(waves["blockers"].items())[:limit]:
        if member in (MULTIPLAYER_BLOCKER, UNREACHABLE_BLOCKER):
            continue                      # 排除桶没有"方法体原文"可看，跳过
        example_pid = pids[0]
        path = index.get(class_of.get(example_pid, ""))
        snippet = ""
        if path is not None:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            # 抓该成员的方法体（从签名行到下一个方法/类结束）
            match = re.search(
                rf"((?:public|protected|internal|private)[^\n;{{=]*\b{re.escape(member)}\s*[\({{=][^\n]*\n"
                rf"(?:.*?\n)??)", text)
            if match:
                start = match.start()
                body = text[start:start + 2600]
                snippet = "\n".join(body.splitlines()[:max_lines])
        out[member] = {"powers": pids, "example": example_pid,
                       "class": class_of.get(example_pid, ""), "snippet": snippet}
    return out


def event_clusters(content) -> dict:
    """事件：按不可跑原因聚类（原始原因 + 归并后的族）。

    注意区分两个计数：**原因条数**（一个事件可以有多条原因）与**涉及事件数**。
    """
    reasons: Counter[str] = Counter()
    families: Counter[str] = Counter()
    events_per_family: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[str]] = defaultdict(list)
    for eid, definition in content.EVENT_DB.items():
        if getattr(definition, "usable", False):
            continue
        for reason in (getattr(definition, "reasons", ()) or ("（未记录原因）",)):
            key = str(reason)
            reasons[key] += 1
            family = classify(key, EVENT_FAMILIES)
            families[family] += 1
            events_per_family[family].add(eid)
            if len(examples[family]) < 5 and eid not in examples[family]:
                examples[family].append(eid)
    return {"reasons": reasons, "families": families,
            "events_per_family": {k: len(v) for k, v in events_per_family.items()},
            "examples": dict(examples)}


def potion_clusters(content) -> dict:
    """药水：按阻塞原因聚类（直接用抽取器记下的 `unsupported`）。"""
    reasons: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for record in json.loads((ROOT / "data" / "content" / "repo" / "potions_source.json")
                             .read_text(encoding="utf-8")):
        name = str(record.get("pid") or record.get("id") or "?")
        for item in record.get("unsupported", ()) or ():
            family = classify(str(item), CARD_FAMILIES)
            reasons[family] += 1
            if len(examples[family]) < 4 and name not in examples[family]:
                examples[family].append(name)
    return {"reasons": reasons, "examples": dict(examples)}


def _pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:.0f}%" if whole else "—"


def render(content, summary) -> str:
    cards = card_clusters(content)
    powers_ = power_clusters(content)
    relics = relic_clusters(content)
    events = event_clusters(content)
    potions = potion_clusters(content)

    lines: list[str] = []
    add = lines.append
    add("# 14 · 未纳入内容与共性分析")
    add("")
    add("> 本文件由 `python -X utf8 tools/gap_clusters.py --write docs/14-未纳入内容与共性分析.md`")
    add("> **生成**，不要手改。逐条清单在 [`docs/10`](10-残缺清单.md)；")
    add("> 本文只做一件事：把「没进来的东西」按**根因族**归类，看它们其实是同一件事。")
    add("")
    add("## 1. 总览")
    add("")
    add("| 内容 | 总数 | 已纳入 | 未纳入 | 未纳入占比 |")
    add("|---|---|---|---|---|")
    add(f"| 卡牌（引擎可执行） | {cards['total']} | {cards['total'] - cards['blocked']} "
        f"| {cards['blocked']} | {_pct(cards['blocked'], cards['total'])} |")
    add(f"| 能力（引擎已实现） | {powers_['total']} | {powers_['implemented']} "
        f"| {powers_['total'] - powers_['implemented']} "
        f"| {_pct(powers_['total'] - powers_['implemented'], powers_['total'])} |")
    add(f"| 遗物（有行为） | {len(content.RELICS)} | "
        f"{sum(1 for d in content.RELICS.values() if getattr(d, 'has_behavior', False))} "
        f"| {sum(1 for d in content.RELICS.values() if not getattr(d, 'has_behavior', False))} | — |")
    add(f"| 药水（可用） | {summary.get('potions', '—')} | {summary.get('potions_usable', '—')} "
        f"| {summary.get('potions_incomplete', '—')} | — |")
    add(f"| 事件（可跑） | {summary.get('events', '—')} | {summary.get('events_usable', '—')} "
        f"| {summary.get('events_blocked', '—')} | — |")
    add("")
    add("> ⚠️ **口径**：这里是内容**准入**，与「与真机一致」无关 ——")
    add("> 全部内容的真机对拍仍然是 **0**。")
    add("")

    add("## 2. 卡牌：主因聚类")
    add("")
    add(f"未进训练集的 {cards['blocked']} 张卡，按**可操作性**取主因（一张卡可能同时有多个原因，"
        "只算最主要的那个）：")
    add("")
    add("| 主因族 | 张数 | 占未纳入 | 代表卡 |")
    add("|---|---|---|---|")
    for family, items in cards["families"].items():
        add(f"| {family} | {len(items)} | {_pct(len(items), cards['blocked'])} "
            f"| {', '.join(items[:4])} |")
    if cards["excluded"]:
        add(f"| （建议排除而不是实现：表现层 / 废弃） | {len(cards['excluded'])} | "
            f"{_pct(len(cards['excluded']), cards['blocked'])} | "
            f"{', '.join(cards['excluded'][:4])} |")
    add("")
    add("准入原因可叠加，各自计数：")
    add("")
    for reason, count in cards["reasons"].most_common():
        add(f"* `{reason}` —— {count}")
    add("")

    add("## 3. 能力：按**重写的钩子**聚类")
    add("")
    add(f"{powers_['total']} 个能力里未实现 {powers_['total'] - powers_['implemented']} 个。"
        "按它在源码里重写的方法分类（这就是「要补哪个钩子位」）：")
    add("")
    add("| 钩子族 | 能力数 | 被多少张卡引用 | 例子 |")
    add("|---|---|---|---|")
    for family, items in powers_["families"].items():
        touched = sum(len(powers_["cards_by_power"].get(pid, ())) for pid in items)
        add(f"| {family} | {len(items)} | {touched} | {', '.join(items[:5])} |")
    add("")
    add("> 「纯数据」那一族**不需要新钩子位** —— 它们的效果由施加它的卡或别的能力驱动，")
    add("> 补上只是注册 + 施加语义，成本最低。真正要动钩子总线的是前三族。")
    add("")

    add("## 4. 遗物：按缺口类别与时机")
    add("")
    add(f"缺口类别：{' · '.join(f'`{k}` {v}' for k, v in relics['kinds'].most_common())}")
    add("")
    add("| 缺口时机 | 处数 |")
    add("|---|---|")
    for hook, count in relics["timings"].most_common(12):
        add(f"| `{hook}` | {count} |")
    add("")

    add("## 5. 事件：按不可跑原因聚类")
    add("")
    add("| 原因族 | 原因条数 | 涉及事件数 | 例子 |")
    add("|---|---|---|---|")
    for family, count in events["families"].most_common():
        add(f"| {family} | {count} | {events['events_per_family'].get(family, 0)} "
            f"| {', '.join(events['examples'].get(family, [])[:4])} |")
    add("")
    add("原始原因（前 12 条）：")
    add("")
    for reason, count in events["reasons"].most_common(12):
        add(f"* {count} × {reason}")
    add("")

    add("## 6. 药水：按阻塞原因聚类")
    add("")
    add("| 原因族 | 处数 | 例子 |")
    add("|---|---|---|")
    for family, count in potions["reasons"].most_common():
        add(f"| {family} | {count} | {', '.join(potions['examples'].get(family, [])[:4])} |")
    add("")

    waves = buff_waves(powers_)
    add("## 7. buff 能力：两波填充计划")
    add("")
    add("判据是「这个能力现在做得了吗」：把它重写的成员对一遍"
        "**纯数据成员 + 引擎已有钩子位 + 计划中的小槽位**。全部命中 = 现在就能做。")
    add("")
    add(f"**Wave A（现有基建可做，{len(waves['wave_a'])} 个）**，按被卡引用数排序：")
    add("")
    add("```")
    for start in range(0, len(waves["wave_a"]), 6):
        add("  " + ", ".join(waves["wave_a"][start:start + 6]))
    add("```")
    add("")
    add(f"**Wave B（需先补基建，{len(waves['wave_b'])} 个）**。要补的特性、解锁的能力数与成本：")
    add("")
    cost_count: Counter[str] = Counter()
    for member in waves["blockers"]:
        cost_count[POWER_INFRA_COST.get(member, ("未评估", ""))[0]] += 1
    add("成本分布：" + " · ".join(f"**{k}** {v} 项"
                                 for k, v in cost_count.most_common()))
    add("")
    add("| 缺失的成员（源码侧） | 解锁能力数 | 成本 | 说明 | 例子 |")
    add("|---|---|---|---|---|")
    for member, pids in waves["blockers"].items():
        cost, note = POWER_INFRA_COST.get(member, ("未评估", "——"))
        add(f"| `{member}` | {len(pids)} | {cost} | {note} | {', '.join(pids[:4])} |")
    add("")
    add("> 读法：**先补左边那一列，而不是逐个啃能力**。先做「小」的（现有管线加槽位），"
        "再做「中」的（改一条管线的语义），「大」的留给对应子系统批次，"
        "「排除」的不要做。")
    add("")
    add(f"⚠️ 成本「排除」的那几行**共** {len(waves.get('excluded', {}))} 个能力，**不在** "
        f"Wave B 的 {len(waves['wave_b'])} 个里面 —— 它们不该做，列出来是为了下一轮"
        "不要再评估一遍。两条判据：① 源码里出现 `GetTeammates` / `GetPlayerName` / "
        "`NetService`（单人局里效果恒等于原值）；② 整个反编译树与内容侧都没有施加者"
        "（不可达）。")
    add("")
    add("### 7.1 每个缺失成员的源码长相（下一轮的现成输入）")
    add("")
    add("Wave B 的难点不在写 Python，而在把源码条件逐条读懂。下面把每个成员"
        "**第一个用到它的能力**的方法体原文附上（截断），其余能力只列名。")
    add("")
    for member, info in blocker_examples(waves).items():
        cost, note = POWER_INFRA_COST.get(member, ("未评估", "——"))
        add(f"<details><summary><code>{member}</code> · 成本 {cost} · "
            f"{len(info['powers'])} 个能力（例：{info['example']} / {info['class']}）"
            f"{' · ' + note if note and note != '——' else ''}</summary>")
        add("")
        add(f"其余使用者：{', '.join(info['powers'][1:]) or '（无）'}")
        add("")
        if info["snippet"]:
            add("```csharp")
            add(info["snippet"])
            add("```")
        else:
            add("（没抓到方法体原文，需要人工看源码）")
        add("")
        add("</details>")
        add("")

    suspects = silent_buff_suspects(content)
    add("### 7.2 静默变强嫌疑（**已实现**的能力，需逐条回源码确认）")
    add("")
    add("引擎的 `damage_additive` / `block_additive` 是**开关**（无条件加层数），"
        "而真机的 `ModifyDamageAdditive` / `ModifyBlockAdditive` 几乎都带谓词：")
    add("")
    add("```csharp")
    add("if (base.Owner != dealer) return 0m;       // 只有自己的伤害")
    add("if (!props.IsPoweredAttack()) return 0m;   // 只有「吃力量」的攻击")
    add("if (!card.Tags.Contains(CardTag.Shiv)) return 0m;")
    add("```")
    add("")
    add("用开关表达带条件的源码 = **丢掉条件 = 能力变强**，症状只是「伤害高几点」，"
        "没有报错。工具报的是**线索**；下面每条都已回源码与引擎逐条核实。")
    add("")
    if suspects:
        add("| 能力 | 源码成员 | 核实结论 |")
        add("|---|---|---|")
        for item in suspects:
            verdict, evidence = SILENT_BUFF_VERDICTS.get(
                item["pid"], ("**未核实**", "——"))
            add(f"| `{item['pid']}` | `{item['source']}` | {verdict} |")
        add("")
        for item in suspects:
            verdict, evidence = SILENT_BUFF_VERDICTS.get(item["pid"])
            if evidence and evidence != "——":
                add(f"* `{item['pid']}` —— {evidence}")
        add("")
        add("**仍未回答的问题**（下一批要做成用例）：")
        add("")
        for question in SILENT_BUFF_OPEN_QUESTIONS:
            add(f"* {question}")
    else:
        add("（没有嫌疑条目 ✅）")
    add("")
    add("")

    add("## 8. 共性分析")
    add("")
    head = list(cards["families"].items())[:4]
    head_cards = sum(len(v) for _k, v in head)
    add(f"1. **卡牌缺口集中在前 4 族**：`{head[0][0]}` 等 4 族共 {head_cards} 张，"
        f"占未纳入卡牌的 {_pct(head_cards, cards['blocked'])}。剩下十几族是长尾，"
        "多数是各自只有一张卡用的冷门命令。")
    add(f"2. **能力缺口是卡牌缺口的根**：{cards['reasons'].get('unimplemented_power', 0)} 处"
        "准入拒绝是「施加了引擎没有的能力」，而能力那一侧只有 "
        f"{powers_['total'] - powers_['implemented']} 个待实现、且前三族就占 "
        f"{sum(len(v) for _k, v in list(powers_['families'].items())[:3])} 个 —— "
        "**补一个钩子位往往整族落地**，不是 170 份独立工作。")
    add("3. **跨内容类型共享同一批机制原语**（数字由本次扫描算出）：")
    add("")
    def family_count(table: dict, *keys: str) -> int:
        return sum(len(table.get(k, ())) for k in keys)

    select_cards = family_count(cards["families"], "选牌动作（没有 `CardSelectCmd` 语义）")
    select_potions = potions["reasons"].get("选牌动作（没有 `CardSelectCmd` 语义）", 0)
    enchant_events = events["families"].get("附魔系统（选牌用途 / 附魔语义）", 0)
    gen_cards = family_count(cards["families"], "生成物放入手牌 / 抽牌堆",
                             "自动打出", "召唤物（`OstyCmd` / 随从）")
    trigger_cards = family_count(cards["families"],
                                 "手牌 / 时点触发器未建模（`on_turn_end_in_hand` 等）")
    combat_events = events["families"].get("事件内战斗交接", 0)
    relic_gaps = sum(relics["kinds"].values())
    add("| 机制原语 | 卡住哪些内容 |")
    add("|---|---|")
    add(f"| **选牌动作**（从候选里挑 N 张，含生成物选牌） | 卡牌 {select_cards} 张"
        f" + 药水 {select_potions} 处 + 事件附魔族 {enchant_events} 条 + 部分遗物 |")
    add(f"| **动态数值公式**（`CalculatedVar` / `WithHitCount` / 变量组合） "
        f"| 卡牌 {family_count(cards['families'], '动态数值公式（`CalculatedVar` / `DynamicVars` 组合）')} 张 |")
    add(f"| **生成物落位**（入手 / 入抽牌堆 / 自动打出 / 召唤） | 卡牌 {gen_cards} 张，药水若干 |")
    add(f"| **附魔系统** | 事件 {enchant_events} 条原因（选牌用途 `enchant` 与附魔语义） |")
    add(f"| **手牌 / 时点触发器** | 卡牌 {trigger_cards} 张（`on_turn_end_in_hand` 等） |")
    add(f"| **事件内战斗交接 + 执行帧栈** | 事件 {combat_events} 条，且是嵌套选择 / 自动打牌的共同前置 |")
    add(f"| **遗物条件建模**（取模计数 / 随机目标 / 卡牌标签） | 遗物 {relic_gaps} 处缺口里的多数 |")
    add("| **解锁状态与修正器**（`UnlockState` / `Modifier`） | 跨全部内容类型，目前一律当「未解锁 = 排除」 |")
    add("")
    add("4. **遗物缺口不是均匀分布的**：时机上 `AfterObtained` 一处就占 "
        f"{relics['timings'].most_common(1)[0][1]} 处（一次性获得型效果），"
        f"而类别上 `commands` {relics['kinds'].get('commands', 0)} 处 > "
        f"`query` {relics['kinds'].get('query', 0)} 处 > "
        f"`bookkeeping` {relics['kinds'].get('bookkeeping', 0)} 处 —— "
        "说明一半以上是「钩子位有了但不会执行那条命令」，而不是「没有这个时机」。")
    #: 三类"应当排除而不是实现"的清单（表现层 / 多人 / 废弃 / 任务系统）。
    deprecated_potions = [pid for pid in content.POTIONS if "deprecated" in pid.lower()]
    deprecated_relics = [rid for rid in content.RELICS if "deprecated" in rid.lower()]
    dead_events = [eid for eid, d in content.EVENT_DB.items()
                   if not getattr(d, "usable", False)
                   and any(("CompleteQuest" in r or "Deprecated" in r
                            or "恒假" in r or "IsAllowed" in r)
                           for r in (getattr(d, "reasons", ()) or ()))]
    add("5. **三类性质必须分开看**：")
    add("")
    add("   * **缺机制原语**（一次实现解锁一批）：上面的表 —— 这是主线，也是唯一值得投入架构的。")
    add(f"   * **缺内容体量**（机械劳动、风险低）：{powers_['total'] - powers_['implemented']} 个能力、"
        f"{summary.get('potions_incomplete', '—')} 瓶药水、"
        f"{sum(1 for d in content.RELICS.values() if not getattr(d, 'has_behavior', False))} 个遗物 —— "
        "逐条照源码填即可，但要按「能力 → 卡牌」的顺序做，否则填完能力才发现卡还卡在别处。")
    add(f"   * **不该做**（应当明确排除）：表现层与多人（`Layout.*` / `SetPortrait` / `HoverTip`）、"
        f"废弃内容（`Deprecated*`）、任务系统（`PlayerCmd.CompleteQuest`）。"
        f"本次扫描识别出：卡牌 {len(cards['excluded'])} 张、药水 {len(deprecated_potions)} 瓶、"
        f"遗物 {len(deprecated_relics)} 个、恒假/废弃事件 {len(dead_events)} 个。"
        "排除它们不需要理由，保留它们只会让缺口数字虚高。")
    add("")
    add("## 9. 复现")
    add("")
    add("```powershell")
    add("$env:PYTHONUTF8='1'")
    add("python -X utf8 tools/gap_clusters.py            # 打印聚类")
    add("python -X utf8 tools/gap_clusters.py --write docs/14-未纳入内容与共性分析.md")
    add("python -X utf8 tools/blocker_report.py          # 逐条卡在哪")
    add("python -X utf8 tools/gap_report.py              # 逐条清单 → docs/10")
    add("```")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="未纳入内容的根因聚类")
    parser.add_argument("--content", default="data/content/repo")
    parser.add_argument("--write", default=None, help="写出 markdown（例如 docs/14-….md）")
    args = parser.parse_args()

    content, summary = load_content(ROOT / args.content)
    if args.write:
        target = ROOT / args.write
        target.write_text(render(content, summary), encoding="utf-8")
        print(f"写出 {target}（{len(render(content, summary).splitlines())} 行）")
        return 0

    cards = card_clusters(content)
    powers_ = power_clusters(content)
    relics = relic_clusters(content)
    events = event_clusters(content)
    print("=" * 74)
    print("未纳入内容的根因聚类（数字全部现算）")
    print("=" * 74)
    print(f"\n【卡牌】{cards['total']} 张中未进训练集 {cards['blocked']} 张")
    for family, items in cards["families"].items():
        print(f"  {len(items):4d} 张  {family}")
        print(f"        例：{', '.join(items[:5])}")
    print(f"\n【能力】{powers_['total']} 个，已实现 {powers_['implemented']}")
    for family, items in powers_["families"].items():
        touched = sum(len(powers_["cards_by_power"].get(pid, ())) for pid in items)
        print(f"  {len(items):4d} 个  被 {touched:3d} 张卡引用  {family}")
    print(f"\n【遗物】缺口类别：{dict(relics['kinds'].most_common())}")
    print(f"  缺口时机：{dict(relics['timings'].most_common(8))}")
    print(f"\n【事件】原因族：{dict(events['families'].most_common())}")
    waves = buff_waves(powers_)
    print(f"\n【buff 能力分波】Wave A（现在可做）{len(waves['wave_a'])} 个｜"
          f"Wave B（需补基建）{len(waves['wave_b'])} 个")
    print("  要补的基建（解锁能力数）：")
    for member, pids in list(waves["blockers"].items())[:10]:
        print(f"    {len(pids):3d}  {member}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
