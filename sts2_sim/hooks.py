"""**钩子总线**：把"什么时机、谁触发、在哪调用"集中成一张声明表（``docs/11`` §11.3）。

为什么要有这个模块
------------------
真机的钩子是**一个入口分发给所有模型**：``Hook.AfterDeath(...)`` 只有一处调用，
``Hook.cs`` 内部遍历所有 model，谁重写了就调谁。漏调用在结构上不可能。

引擎早期是在 ``core.py`` 各处**手写**调用（``power_rules.on_attacked(...)``），
于是"某条路径忘了通知"成了最常见的 bug 形态。审计时一次抓到 **4 条死亡路径**
绕过钩子（末日处决 / 招式自爆 / 中毒掉血），症状全是**静默**的 ——
怪还在按部就班地打，只是"该发生的事没发生"。

这个模块把三件事变成**可查询、可测试**的：

1. :data:`HOOKS` —— 每个钩子的声明：真机对应哪个回调、触发时遍历谁、在哪些函数里调用
2. :func:`fire` —— **唯一**的触发入口，调用方只说"发生了什么"
3. :func:`trace` —— 记录一次战斗里触发过哪些钩子（测试与排错用）

加一个新钩子的正确姿势
----------------------
1. 在 :data:`HOOKS` 里加一条声明（写清真机出处与触发时机）
2. 在 ``PowerRules`` 上加钩子位
3. 在核心流程里 ``fire("新钩子", ...)``

``tests/test_hook_bus.py`` 会检查：声明过的钩子必须有调用点、
调用点用到的名字必须声明过、遍历范围不能写错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

#: 触发时遍历谁。
#: * ``"owner"``  —— 调用方指定的那一个单位（如"被打的那个"）
#: * ``"player"`` —— 玩家
#: * ``"enemies"``—— 全部敌人
#: * ``"both"``   —— 玩家 + 全部敌人（苦痛族挂玩家、`enrage` 挂敌人，两边都要覆盖）
#: * ``"each"``   —— 调用方给一个列表，逐个触发
#: * ``"relics"`` —— 拥有者的**遗物**（``TimingHook``：还没实现的能力钩子）
HOLDER_SCOPES = frozenset({"owner", "player", "enemies", "both", "each", "relics"})


@dataclass(frozen=True)
class HookSpec:
    """一个钩子的声明。"""

    name: str
    #: 真机对应的回调（``Hook.Xxx`` 或 ``AbstractModel.Xxx``）—— 铁律 R1 的源码出处
    source: str
    #: 触发时遍历谁，见 :data:`HOLDER_SCOPES`
    scope: str
    #: 在核心流程的哪一步触发（人话，用来对照真机的调用顺序）
    timing: str
    #: ``core.py`` 里调用它的函数名（**至少有这个函数**，由测试校验）
    call_sites: tuple[str, ...]
    #: 额外传给处理器的实参名（调用方必须提供，否则报错而不是静默传 None）
    args: tuple[str, ...] = ()
    #: 是否由 `on_any_death` 这类"父钩子"内部转发（转发型的 call_site 写父钩子）
    forwarded_by: str | None = None


#: **钩子总表**。顺序按真机的回合流程排，便于对照。
HOOKS: dict[str, HookSpec] = {
    spec.name: spec
    for spec in (
        HookSpec(
            name="on_energy_reset",
            source="Hook.AfterEnergyReset",
            scope="owner",
            timing="能量重置**之后**（放前面会被重置覆盖）",
            call_sites=("start_player_turn",),
        ),
        HookSpec(
            name="on_owner_turn_start",
            source="Hook.AfterSideTurnStart",
            scope="owner",
            timing="拥有者所在阵营回合开始（抽牌、能量重置都已完成）",
            call_sites=("start_player_turn", "_run_enemy_turn"),
        ),
        HookSpec(
            name="on_player_turn_start",
            source="Hook.AfterSideTurnStart（订阅者是敌人）",
            scope="enemies",
            timing="**玩家**阵营回合开始 —— 拥有者是敌人的能力也在这时触发"
                    "（`RampartPower`：炮台在玩家回合开始时拿格挡）",
            call_sites=("start_player_turn",),
        ),
        HookSpec(
            name="on_card_drawn",
            source="Hook.AfterCardDrawn",
            scope="both",
            timing="玩家抽到一张牌之后",
            call_sites=("draw_cards",),
            args=("card",),
        ),
        HookSpec(
            name="on_card_drawn_early",
            source="Hook.AfterCardDrawnEarly（``Hook.AfterCardDrawn`` 的**第一遍**遍历，"
                   "``Hook.cs:202-210``）",
            scope="both",
            timing="玩家抽到一张牌之后、``on_card_drawn`` **之前**（牌已经在手牌里）",
            call_sites=("draw_cards",),
            args=("card",),
        ),
        HookSpec(
            name="on_card_played",
            source="Hook.AfterCardPlayed",
            scope="both",
            timing="玩家打出一张牌、效果结算**之后**",
            call_sites=("step",),
            args=("card_type", "card"),
        ),
        HookSpec(
            name="on_before_card_played",
            source="Hook.BeforeCardPlayed（``CardModel.cs:1926``，在 ``OnPlay`` 之前）",
            scope="both",
            timing="一张牌**开始结算之前**（牌已进打出区、效果一条都还没跑）",
            call_sites=("step", "_autoplay_sly"),
            args=("card",),
        ),
        HookSpec(
            name="on_power_applied",
            source="Hook.AfterPowerAmountChanged（``PowerCmd.Apply`` 那条路径；"
                   "``AbstractModel.AfterPowerAmountChanged(choiceContext, power, amount, applier, cardSource)``）",
            scope="both",
            timing="某个能力被施加时，**通知双方所有模型**（带施加者与这一次的量）",
            call_sites=("_apply_one",),
            args=("pid", "applier", "applied"),
        ),
        HookSpec(
            name="on_power_amount_changed",
            source="Hook.AfterPowerAmountChanged",
            scope="both",
            timing="**任意**能力的层数发生变化之后（施加 / 修改 / 递减都会通知）",
            call_sites=("add_power",),
            args=("pid", "amount", "applier", "changed_owner"),
        ),
        HookSpec(
            name="on_before_death",
            source="Hook.BeforeDeath（``CreatureCmd.cs`` 的 ``Kill`` 路径，先于 ``AfterDeath``）",
            scope="both",
            timing="某个单位**即将死亡**（死亡还没处理：血量尚在、遗物与能力都还没结算）",
            call_sites=("mark_dead",),
            args=("dead",),
        ),
        HookSpec(
            name="on_applier_death",
            source="Hook.AfterDeath（第三段作用域：``creature == base.Applier``）",
            scope="both",
            timing="**把这条能力贴到我身上的那个单位**死了（``ConstrictPower`` / "
                   "``ShrinkPower`` / ``HexPower`` 的结束条件）",
            call_sites=("on_any_death",),
            args=("dead",),
        ),
        HookSpec(
            name="on_card_entered_combat",
            source="Hook.AfterCardEnteredCombat（``CardPileCmd.cs:515``；转念/转化走 "
                   "``CardCmd.cs:447``）",
            scope="both",
            timing="一张牌**刚进入战斗**（此前没有任何牌堆，``oldPile == null``）之后 —— "
                   "只认「新造出来的牌」，同战斗内换牌堆**不**触发",
            call_sites=("_add_generated_card",),
            args=("card",),
        ),
        HookSpec(
            name="on_card_generated_for_combat",
            source="Hook.AfterCardGeneratedForCombat",
            scope="both",
            timing="战斗内**生成**一张牌之后（`CardPileCmd.AddGeneratedCardToCombat`）",
            call_sites=("_add_generated_card",),
            args=("card", "creator"),
        ),
        HookSpec(
            name="on_before_hand_draw",
            source="Hook.BeforeHandDraw",
            scope="both",
            timing="玩家回合**抽手牌之前**（`StartTurn` 里快照完 AmountOnTurnStart 之后）",
            call_sites=("start_player_turn",),
            args=(),
        ),
        HookSpec(
            name="on_after_shuffle",
            source="Hook.AfterShuffle（``CardPileCmd.cs:1131``，在 ``Shuffle`` 末尾）",
            scope="both",
            timing="**洗完牌之后**（弃牌堆并回抽牌堆、整体 ``StableShuffle(Rng.Shuffle)``）"
                   "—— 抽牌抽到空堆时也会洗，所以这个时机在抽牌循环**中间**也会出现。"
                   "`StratagemPower`（计策）在这里让玩家从抽牌堆挑牌入手，"
                   "会让抽牌挂起（`core.DrawFrame`）",
            call_sites=("shuffle_piles",),
            args=(),
        ),
        HookSpec(
            name="on_before_side_turn_end",
            source="Hook.BeforeSideTurnEnd",
            scope="both",
            timing="某一方回合结束、**结算回合结束效果之前**（能力侧的时机）",
            call_sites=("_run_enemy_turn", "step"),
            args=("side",),
        ),
        HookSpec(
            name="on_energy_spent",
            source="Hook.AfterEnergySpent（签名 ``AfterEnergySpent(CardModel card, int amount)``）",
            scope="both",
            timing="**打出卡牌**、费用刚扣掉之后（药水/效果扣能量不经过这里——它们没有卡对象）",
            call_sites=("step",),
            args=("card", "amount"),
        ),
        HookSpec(
            name="on_block_cleared",
            source="Hook.AfterBlockCleared（``CombatManager.cs:768``）",
            scope="both",
            timing="每个**开始回合**的单位清完格挡之后（玩家第 1 回合不清格挡也照样分发）",
            call_sites=("start_player_turn", "_run_enemy_turn"),
            args=("creature",),
        ),
        HookSpec(
            name="on_orb_evoked",
            source="Hook.AfterOrbEvoked（``OrbCmd.cs:147``）",
            scope="both",
            timing="激发一个充能球之后，带上**这次激发的目标**"
                   "（源码是 ``targets = await evokedOrb.Evoke(...)`` 的返回值）",
            call_sites=("evoke",),
            args=("orb", "targets"),
        ),
        HookSpec(
            name="on_before_side_turn_start",
            source="Hook.BeforeSideTurnStart",
            scope="both",
            timing="某一方回合开始、**清格挡之前**（能力侧的时机；遗物侧是 "
                    "``relic_before_side_turn_start``）",
            call_sites=("start_player_turn", "_run_enemy_turn"),
            args=("side",),
        ),
        HookSpec(
            name="on_stars_spent",
            source="Hook.AfterStarsSpent（``CardModel.cs:1842``，`SpendStars` 里 "
                   "``amount > 0`` 才分发）",
            scope="both",
            timing="打出的牌**真的花掉了星**之后（能量扣完、星也扣完之后）",
            call_sites=("step",),
            args=("amount",),
        ),
        HookSpec(
            name="on_stars_gained",
            source="Hook.AfterStarsGained（``PlayerCmd.cs:95``）",
            scope="both",
            timing="玩家**获得星**之后（`PlayerCmd.GainStars` 的唯一出口）",
            call_sites=("gain_stars",),
            args=("amount",),
        ),
        HookSpec(
            name="on_auto_pre_play",
            source="Hook.AfterAutoPrePlayPhaseEntered（``CombatManager.cs:867``，在 ``Play`` 阶段之前）",
            scope="both",
            timing="玩家回合**抽完牌、即将进入出牌阶段**（`AutoPrePlay`）—— "
                   "`MayhemPower` 在这里自动打出抽牌堆顶的牌",
            call_sites=("start_player_turn",),
            args=(),
        ),
        HookSpec(
            name="on_auto_post_play",
            source="Hook.AfterAutoPostPlayPhaseEntered（``CombatManager.cs:1545``，"
                   "在 ``BeforeSideTurnEnd`` 之前）",
            scope="both",
            timing="玩家结束回合、进入 ``AutoPostPlay`` 阶段 —— **清手牌之前**",
            call_sites=("step",),
        ),
        HookSpec(
            name="on_card_exhausted",
            source="Hook.AfterCardExhausted（唯一调用者是 ``CardCmd.Exhaust``，``CardCmd.cs:246``）",
            scope="both",
            timing="一张牌**被消耗之后**（牌已经进了消耗堆）",
            call_sites=("_finish_played_card", "_run_hand_triggers",
                        "_autoplay_sly", "_resolve_selection"),
            args=("card", "caused_by_ethereal"),
        ),
        HookSpec(
            name="on_block_gained",
            source="Hook.AfterBlockGained（``CreatureCmd.GainBlock`` 末尾，``CreatureCmd.cs:699``）",
            scope="owner",
            timing="某个单位**获得格挡之后**（格挡已经加上、数值是全部修正之后的量）",
            call_sites=("gain_block", "_orb_gain_block"),
            args=("gained", "card"),
        ),
        HookSpec(
            name="on_side_turn_end",
            source="Hook.AfterSideTurnEnd（带 ``side`` 参数的那次分发）",
            scope="both",
            timing="**任意一方**阵营回合结束 —— ``side`` 说明结束的是哪一方"
                    "（`FlameBarrierPower` 靠它判断「不是自己这边结束就移除」）",
            call_sites=("step", "_end_enemy_side_turn"),
            args=("side",),
        ),
        HookSpec(
            name="on_side_turn_end_late",
            source="Hook.AfterSideTurnEnd 的**第二遍**遍历（``AbstractModel.AfterSideTurnEndLate``）"
                   "—— ``Hook.cs`` 在同一个 ``AfterSideTurnEnd`` 里先遍历全部模型的 "
                   "``AfterSideTurnEnd``，再遍历全部模型的 ``AfterSideTurnEndLate``",
            scope="both",
            timing="任意一方阵营回合结束、``AfterSideTurnEnd`` 的**第一遍全部跑完之后**"
                    "（``DisintegrationPower`` 的「回合末自伤」就在这里，"
                    "早于它跑会让自伤发生在其它回合末效果之前）",
            call_sites=("step", "_end_enemy_side_turn"),
            args=("side",),
        ),
        HookSpec(
            name="on_after_player_turn_start",
            source="Hook.AfterPlayerTurnStart（``CombatManager.cs:910`` 的 SetupPlayerTurn）",
            scope="both",
            timing="**玩家**回合开始、抽牌**之后**、``AfterSideTurnStart`` **之前**"
                    "（`InfernoPower` / `CrimsonMantlePower` 挂在这里）",
            call_sites=("start_player_turn",),
        ),
        HookSpec(
            name="on_attacked",
            source="Hook.AfterDamageReceived",
            scope="owner",
            timing="**受到伤害之后**（扣血已完成）。普通伤害与掉血**都**走这里",
            call_sites=("deal_damage", "deal_raw_damage"),
            # ``card`` = 真机的 ``cardSource``：这次伤害由哪张牌造成（没有就是 ``None``）。
            # `RupturePower` 用它区分"卡牌自伤"与"其它掉血"，两者的发力量时机不同。
            args=("attacker", "unblocked", "from_card", "powered", "card"),
        ),
        HookSpec(
            name="on_damage_given",
            source="Hook.AfterDamageGiven",
            scope="owner",
            timing="**造成伤害之后**（攻击方视角）",
            call_sites=("deal_damage",),
            # ``target`` = 真机的 ``Creature target``：这一笔伤害打在谁身上。
            # `EnvenomPower` / `ConcoctPower`（"打出有效攻击就给目标上毒"）与
            # `MonarchsGazePower`（给目标上"力量下降"）全靠它 —— 少了它，
            # 这三个能力只能变成"给自己上毒"或者干脆不实现。
            args=("unblocked", "powered", "was_fully_blocked", "target"),
        ),
        HookSpec(
            name="on_applied",
            source="Hook.AfterApplied（配套的 ``BeforeApplied`` 带 ``amount``："
                   "``AbstractModel.BeforeApplied(Creature, decimal amount, …)``）",
            scope="owner",
            timing="某个能力**刚被施加**（只触发被施加的那一个，不遍历全部）",
            call_sites=("_apply_one",),
            # ``applied`` = **这一次**施加了多少层（真机 ``BeforeApplied`` 的 ``amount``）。
            # 处理器收到的第二个位置实参是施加**之后的总层数**，两者不同：
            # `TemporaryStrengthPower` 要的是"这一次"的量，拿总层数会越叠越多。
            args=("pid", "applied"),
        ),
        HookSpec(
            name="on_owner_turn_end",
            source="Hook.AfterSideTurnEnd",
            scope="owner",
            timing="拥有者所在阵营回合结束",
            call_sites=("on_turn_end", "_end_enemy_side_turn"),
        ),
        HookSpec(
            name="on_self_death",
            source="Hook.AfterDeath（触发者是自己）",
            scope="owner",
            timing="**自己**死亡之后（`StockPower` 补人 / `SurprisePower` 召唤）",
            call_sites=("on_any_death",),
            forwarded_by="on_any_death",
        ),
        HookSpec(
            name="on_ally_death",
            source="Hook.AfterDeath（触发者是队友）",
            scope="enemies",
            timing="**同阵营队友**死亡之后（`CrabRagePower` 暴怒 / `RavenousPower` 吞食）",
            call_sites=("on_any_death",),
            forwarded_by="on_any_death",
        ),
        # ---- 遗物侧（同一套源码回调，同一张总线）----------------------------
        # 遗物与能力都重写 `AbstractModel` 的同一批虚方法，所以它们在总线上
        # 应当**同名同义**。下面五条是遗物目前实现的全部战斗内时机；
        # 其余 110 种（`AfterObtained` / `AfterCombatEnd` …）还没做，
        # 与能力的缺口一样如实记在报告里。
        HookSpec(
            name="relic_combat_start",
            source="Hook.BeforeCombatStart + Hook.AfterRoomEntered（战斗房间）",
            scope="relics",
            timing="第 1 回合开始**之前**（真机 `CombatManager.cs:594`）",
            call_sites=("start_combat", "_apply_start_of_combat_relics"),
        ),
        HookSpec(
            name="relic_before_side_turn_start",
            source="Hook.BeforeSideTurnStart",
            scope="relics",
            timing="回合开始、**清格挡之前**",
            call_sites=("start_player_turn",),
        ),
        HookSpec(
            name="relic_energy_reset",
            source="Hook.AfterEnergyReset",
            scope="relics",
            timing="能量重置之后（与能力的同名时机一致）",
            call_sites=("start_player_turn",),
        ),
        HookSpec(
            name="relic_after_player_turn_start",
            source="Hook.AfterPlayerTurnStart",
            scope="relics",
            timing="抽牌之后、`AfterSideTurnStart` 之前",
            call_sites=("start_player_turn",),
        ),
        HookSpec(
            name="relic_side_turn_start",
            source="Hook.AfterSideTurnStart",
            scope="relics",
            timing="回合开始流程的最后一步（提灯 / 赤備正在这里）",
            call_sites=("start_player_turn",),
        ),
        HookSpec(
            name="relic_after_card_played",
            source="Hook.AfterCardPlayed",
            scope="relics",
            timing="玩家**打出任意一张牌之后**（`GamePiece` 只看能力牌、"
                    "`Permafrost` / `LostWisp` / `RainbowRing` 同理）",
            call_sites=("_fire_after_card_played",),
            # 牌型由调用方传入 → `relics._applies` 的 `card_type` 守卫求值；
            # 本回合已打出的张数 → `every_n_turn` 守卫（"每 N 张攻击牌触发一次"）求值。
            # 两个都必须传：算不出的守卫会被**静默跳过**（安全方向，但遗物等于没接）。
            args=("card_type", "played_counts"),
        ),
        # ⚠️ 能力侧的 `on_side_turn_end` 暂**不**给遗物复用：那条时机上的遗物
        # 钩子都带"取模计数 / 随机目标"，抽取器还建不出来 —— 采纳就是无条件生效。
        # 证据与启用前置条件见 `content.HOOK_TIMING` 的注释与 docs/12 §2.6「遗物口径」。
        # ---- Run 层（作用于牌组 / 金币 / 最大生命，不走战斗内分发）----------
        HookSpec(
            name="relic_obtained",
            source="Hook.AfterObtained",
            scope="relics",
            timing="**拾取遗物时**（`CreatureCmd.GainMaxHp` / 加诅咒 / 转化…）",
            call_sites=("apply_run_hook",),
        ),
        HookSpec(
            name="relic_combat_end",
            source="Hook.AfterCombatEnd",
            scope="relics",
            timing="**战斗结束**（无论胜负，回血 / 加金币）",
            call_sites=("apply_run_hook",),
        ),
        HookSpec(
            name="relic_combat_victory",
            source="Hook.AfterCombatVictory",
            scope="relics",
            timing="**战斗胜利**（燃烧之血在这里回血）",
            call_sites=("apply_run_hook",),
        ),
    )
}

#: 遗物时机名 → 总线上的钩子名。两边一一对应，由测试校验。
RELIC_TIMING_TO_HOOK: dict[str, str] = {
    "combat_start": "relic_combat_start",
    "before_side_turn_start": "relic_before_side_turn_start",
    "after_energy_reset": "relic_energy_reset",
    "after_player_turn_start": "relic_after_player_turn_start",
    "after_side_turn_start": "relic_side_turn_start",
    "after_card_played": "relic_after_card_played",
    # Run 层：不走战斗内分发，由 `relics.apply_run_hook` 结算
    "obtained": "relic_obtained",
    "combat_end": "relic_combat_end",
    "combat_victory": "relic_combat_victory",
}


@dataclass
class HookTrace:
    """一次战斗里触发过哪些钩子（排错与测试用）。

    两种语义**都要**，它们回答不同的问题：

    * **分发过**（``names`` / ``count``）—— "核心流程走到这个时机了吗"。
      断言它等于在测**流程**：少一个时机就是某条路径没通知。
    * **有人响应**（``responded``）—— "这个能力真的被触发了吗"。
      断言它等于在测**能力**：比如"被打之后滑溜有没有减层"。
    """

    fired: list[tuple[str, str, bool]] = field(default_factory=list)

    def record(self, hook: str, holder: str, ran: bool) -> None:
        self.fired.append((hook, holder, ran))

    def names(self) -> list[str]:
        return [name for name, _holder, _ran in self.fired]

    def count(self, hook: str) -> int:
        return sum(1 for name, _h, _r in self.fired if name == hook)

    def responders(self) -> list[tuple[str, str]]:
        return [(name, holder) for name, holder, ran in self.fired if ran]

    def count_responded(self, hook: str) -> int:
        return sum(1 for name, _h, ran in self.fired if name == hook and ran)


#: 当前是否在记录（``None`` = 不记录）。测试用 :func:`trace` 打开。
_ACTIVE_TRACE: HookTrace | None = None


class trace:  # noqa: N801 —— 当成上下文管理器用，小写更顺手
    """``with trace() as t: ...`` —— 记录这段代码里触发的钩子。"""

    def __enter__(self) -> HookTrace:
        global _ACTIVE_TRACE
        self._previous = _ACTIVE_TRACE
        _ACTIVE_TRACE = HookTrace()
        return _ACTIVE_TRACE

    def __exit__(self, *exc: object) -> None:
        global _ACTIVE_TRACE
        _ACTIVE_TRACE = self._previous


def holders_for(spec: HookSpec, state: Any, ctx: dict) -> list:
    """按声明的范围算出这一跳要遍历哪些单位。"""
    player = getattr(state, "player", None)
    enemies = list(getattr(state, "enemies", ()) or ())
    if spec.scope == "owner":
        owner = ctx.get("owner")
        return [owner] if owner is not None else []
    if spec.scope == "player":
        return [player] if player is not None else []
    if spec.scope == "enemies":
        return [e for e in enemies if e.alive()]
    if spec.scope == "both":
        return [x for x in [player, *enemies] if x is not None and x.alive()]
    if spec.scope == "each":
        return [x for x in (ctx.get("each") or ()) if x is not None]
    if spec.scope == "relics":
        # 遗物的钩子：拥有者 = 玩家，遍历它带的遗物
        return [player] if player is not None else []
    raise KeyError(f"未声明的遍历范围 {spec.scope!r}（{spec.name}）")


def fire(name: str, state: Any, events: list[str], handler: Callable[..., Any],
         **ctx: Any) -> None:
    """触发一个钩子。``handler(holder, amount, context)`` 返回 ``REMOVE`` 时移除该能力。

    ⚠️ 钩子名必须是 :data:`HOOKS` 里声明过的 —— 拼错名字会**立刻报错**，
    而不是安静地什么都不做。这正是这个模块存在的意义：
    以前是每个分发函数各写一份遍历逻辑，漏了哪条路径没人知道。
    """
    spec = HOOKS.get(name)
    if spec is None:
        raise KeyError(f"未声明的钩子 {name!r}（要加钩子请先在 hooks.HOOKS 里声明）")
    missing = [key for key in spec.args if key not in ctx]
    if missing:
        raise TypeError(f"钩子 {name} 缺少实参 {missing}"
                        f"（真机 {spec.source} 需要它们，少传会让对应能力永不触发）")
    # 把钩子名与 state/events 放进上下文：处理器要按 `hook` 去查
    # `PowerRules` 上的字段，并按 `state`/`events` 构造 `PowerContext`。
    ctx["hook"] = name
    ctx["state"] = state
    ctx["events"] = events
    for holder in holders_for(spec, state, ctx):
        # ⚠️ 分发**一定**记，响应与否另记一个标志：
        # 钩子会遍历"玩家 + 全部敌人"，多数持有者没有对应能力；
        # 只记响应者会让"流程有没有走到这个时机"没法断言（测试里的怪常常没有能力）。
        ran = bool(handler(holder, ctx))
        if _ACTIVE_TRACE is not None:
            _ACTIVE_TRACE.record(name, getattr(holder, "name", "?"), ran)


def declared(name: str) -> bool:
    return name in HOOKS


def audit() -> list[str]:
    """自检：声明与 ``PowerRules`` 的钩子位、与遗物时机是否一致。返回问题列表。

    三个方向都要查 —— 任何一个对不上，对应的机制都会**静默失效**：

    1. ``PowerRules`` 有钩子位但没声明 → ``fire`` 会拒绝这个名字（好在会报错）
    2. 声明了但没有钩子位 → 永远不会被调用
    3. 遗物时机没有对应的总线钩子 → 那个时机的遗物全部不生效
    """
    from . import powers, relics

    problems: list[str] = []
    fields = {f for f in powers.PowerRules.__dataclass_fields__ if f.startswith("on_")}
    declared_hooks = {name for name in HOOKS if name.startswith("on_")}
    for field_name in sorted(fields - declared_hooks):
        problems.append(f"钩子位 {field_name} 没有在 hooks.HOOKS 里声明")
    for hook_name in sorted(declared_hooks - fields):
        problems.append(f"hooks.HOOKS 声明了 {hook_name}，但 PowerRules 没有这个钩子位")
    for timing in relics.TIMING_ORDER:
        hook = RELIC_TIMING_TO_HOOK.get(timing)
        if hook is None:
            problems.append(f"遗物时机 {timing} 没有对应的总线钩子")
        elif hook not in HOOKS:
            problems.append(f"遗物时机 {timing} 指向未声明的钩子 {hook}")
    for spec in HOOKS.values():
        if spec.scope not in HOLDER_SCOPES:
            problems.append(f"{spec.name}: 未知遍历范围 {spec.scope}")
        if not spec.source:
            problems.append(f"{spec.name}: 缺少源码出处（铁律 R1）")
        if not spec.call_sites:
            problems.append(f"{spec.name}: 没有任何调用点")
    return problems

