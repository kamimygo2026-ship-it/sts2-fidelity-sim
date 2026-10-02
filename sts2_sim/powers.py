"""能力（power）的行为框架 —— ``docs/09`` L2。

引擎原来只把能力当**数字**存着，另在 ``core.py`` 里硬编码检查
``strength`` / ``weak`` / ``vulnerable`` / ``energy`` 四个。这条路走不通：
真机有 **265 个能力**，其中 61 个挂在 ``AfterSideTurnEnd``、29 个挂在
``AfterCardPlayed``、19 个挂在 ``AfterSideTurnStart``（按 265 个能力类
实际重写的钩子统计）。

本模块把"能力的钩子行为"集中到一张**注册表**里，与源码同名对应：

===================  ============================================================
真机钩子              本模块
===================  ============================================================
``AfterEnergyReset``  ``on_energy_reset``   玩家回合能量重置后
``AfterSideTurnStart````on_owner_turn_start`` 拥有者所在阵营回合开始
``AfterSideTurnEnd``  ``on_owner_turn_end``   拥有者所在阵营回合结束
（数值修正）           ``block_additive`` / ``damage_additive`` / …
===================  ============================================================

设计原则（与 ``docs/09`` 一致）：

1. **没实现的能力不算实现**。``IMPLEMENTED`` 是显式白名单；不在里面的能力
   对引擎而言就是"不生效"，由 ``content.engine_coverage()`` 报出来，
   相关卡牌标记 ``effects_incomplete`` 并排除出训练集。
2. **层数语义必须与真机一致**：能力是"回合结束递减"还是"触发后立刻移除"，
   差别会让战斗时长整体偏掉。每个规则都注明源码依据。
3. **不做近似**。实现不了就留空并让覆盖率说话，不写"差不多"的逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from functools import partial
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:                       # pragma: no cover
    from .core import Combatant, CombatState

#: 处理器返回值：保留能力 / 立刻移除整个能力。
KEEP = "keep"
REMOVE = "remove"

#: 回合结束递减的能力。
#:
#: ⚠️ **条件是 `side == CombatSide.Enemy`，不是"拥有者阵营"** —— 三个能力的源码
#: 完全一致（`VulnerablePower` / `WeakPower` / `FrailPower` 的
#: ``AfterSideTurnEnd`` 里都是 ``if (side == CombatSide.Enemy) TickDownDuration``）。
#:
#: 另外两条同表但写法不同的（`docs/12` §2.29）：
#: * `NoBlockPower`：``if (side == CombatSide.Enemy) await PowerCmd.Decrement(this)``
#:   —— 与上面三条**逐字同义**（都是"敌方阵营回合结束减 1"）。
#: * `DebilitatePower`：``if (participants.Contains(base.Owner)) await PowerCmd.Decrement(this)``
#:   —— 写的是"拥有者所在阵营结束"。它在单人局里只可能被 `Debilitate` 卡贴到**敌人**
#:   身上（全仓库只有这一处 `Apply<DebilitatePower>`），敌人的"自己阵营结束"就是敌方
#:   阵营结束，两者等价；哪天有内容把它贴给玩家，这条要拆成单独的时机。
#:
#: 差别是有后果的：玩家身上的易伤若在**玩家回合结束**就递减，那么"敌人这回合打你
#: 之前先给你上易伤"这类联动会少算一整个回合的量。正确时机是**敌方阵营回合结束时**，
#: 对场上所有持有者一起递减。
DECREMENTS_AT_ENEMY_SIDE_TURN_END = frozenset({"vulnerable", "weak", "frail",
                                                "debilitate", "no_block"})


@dataclass
class PowerContext:
    """钩子上下文：能力拥有者、战斗状态、事件日志。"""

    state: "CombatState"
    owner: "Combatant"
    events: list[str] = field(default_factory=list)
    #: ⭐ **当前派发的是哪一个能力实例**（S01）。``None`` = 这个能力不分实例。
    #: 处理器里的 ``PowerCmd.Decrement(this)`` / ``Remove(this)`` 作用的就是它：
    #: 两个 `the_bomb` 各自倒计时，不能减到"最后一个"身上。
    instance: object | None = None
    #: ⭐ **这个实例自己的动态变量**（``SetDamage`` / ``SetBlock`` 写进来的）。
    #: 分实例能力**必须**读它而不是 ``owner.power_vars[pid]`` —— 后者按设计指向
    #: "最新实例"，两个炸弹会互相偷数值（一张升级一张没升级时静默算错）。
    instance_vars: dict = field(default_factory=dict)

    def opponents_of(self, combatant: "Combatant") -> list["Combatant"]:
        """对方阵营的存活单位。"""
        if combatant is self.state.player:
            return [e for e in self.state.enemies if e.alive()]
        return [self.state.player] if self.state.player.alive() else []

    def is_player(self, combatant: "Combatant") -> bool:
        return combatant is self.state.player


@dataclass(frozen=True)
class PowerRules:
    """一个能力的行为。``source`` 写明源码依据，便于对照检查。"""

    pid: str
    source: str
    on_energy_reset: Callable[[PowerContext, int], str | None] | None = None
    on_owner_turn_start: Callable[[PowerContext, int], str | None] | None = None
    on_owner_turn_end: Callable[[PowerContext, int], str | None] | None = None
    on_attacked: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterSideTurnStart`` 但**触发条件是"玩家阵营回合开始"**（``RampartPower``）。
    #: 拥有者是敌人，所以不能挂在 ``on_owner_turn_start`` 上（那要等它自己的回合）。
    on_player_turn_start: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterCardPlayed``：玩家打出牌时触发（``EnragePower`` 只看技能牌）。
    on_card_played: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterEnergySpent``：**卡牌费用**刚扣掉时触发（``OrbitPower``）。
    #: 参数：``card``（花了钱的那张牌）、``amount``（花掉多少，X 费卡是全部能量）。
    on_energy_spent: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterOrbEvoked``：一个充能球被激发之后触发（``ThunderPower``）。
    #: 参数：``orb``（被激发的球）、``targets``（这次激发打到的单位）。
    on_orb_evoked: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterBlockCleared``：某个单位的格挡**被清空之后**触发（``ToricToughnessPower`` /
    #: ``SelfFormingClayPower``）。参数：``creature``（刚清空格挡的那个单位）——
    #: 源码的判据是 ``creature == base.Owner``，所以要传下来，不能只按持有者分发。
    on_block_cleared: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterCardEnteredCombat``：一张牌**刚进入战斗**（新造出来的牌）之后触发。
    #: 参数：``card``。苦痛族（``tangled`` / ``smoggy`` / ``ringing`` / ``hex`` /
    #: ``vital_spark`` / ``galvanic``）靠它给**战斗中途生成的牌**补贴苦痛 ——
    #: 少了这一半，"生成一张攻击牌"在 ``tangled`` 下不会被缠住（引擎比真机**弱**）。
    on_card_entered_combat: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterDeath`` 的**第三段作用域**：**施加者**死了（``creature == base.Applier``）。
    #: 与 ``on_self_death`` / ``on_ally_death`` 分开：那两个是"自己死 / 队友死"，
    #: 这一段是"把这条能力贴到我身上的那个单位死了" —— ``constrict`` / ``shrink`` /
    #: ``hex`` 的结束条件都是它。参数：``dead``（死掉的那个单位）。
    on_applier_death: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterAutoPrePlayPhaseEntered``（``CombatManager.cs:867``）：玩家回合的
    #: **AutoPrePlay 阶段**——也就是说"抽完牌、马上要进入出牌阶段"那一步。
    #: `MayhemPower` 在这里自动打出抽牌堆顶的 Amount 张。
    #: ⚠️ 与 ``on_auto_post_play``（回合**结束**时的 AutoPostPlay）是两个不同的时机。
    on_auto_pre_play: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterStarsSpent``（``CardModel.cs:1842``）：**真的花了星**之后触发
    #: （``amount > 0`` 才分发）。`ChildOfTheStarsPower` 挂在这里。
    #: 参数：``amount``（这次花了几颗星）。
    on_stars_spent: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterStarsGained``（``PlayerCmd.cs:95``）：获得星之后触发。
    #: `BlackHolePower` 的一半挂在这里。
    on_stars_gained: Callable[[PowerContext, int], str | None] | None = None
    #: ``BeforeDeath``：**即将死亡**（还来得及做事，``mark_dead`` 的第一件事）。
    #: `SwipePower` / `HeistPower` 在这里把偷来的东西还给玩家。参数：``dead``。
    on_before_death: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterDeath``：**同阵营**其他单位死亡时触发（``CrabRagePower``）。
    on_ally_death: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterAttack``：拥有者**造成**伤害后触发（``PainfulStabsPower``）。
    on_damage_given: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterDeath`` 但**触发者是拥有者自己**（``StockPower`` / ``SurprisePower``）。
    #: 与 ``on_ally_death`` 分开：那两个是"别人死了我触发"。
    on_self_death: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterApplied``：能力**刚被施加时**触发一次（``TangledPower`` 就是在这时候
    #: 给所有攻击牌贴苦痛的）。
    on_applied: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterCardDrawn``：玩家抽到牌时触发（``ChainsOfBindingPower`` 贴 ``Bound``）。
    on_card_drawn: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterCardDrawnEarly``：``Hook.AfterCardDrawn`` 的**第一遍**遍历
    #: （``HellraiserPower`` 在这里把抽到的攻击牌自动打出去）。
    on_card_drawn_early: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterCardExhausted``：**有牌被消耗之后**触发（``FeelNoPainPower`` / ``DarkEmbracePower``）。
    #: 参数：``card``（被消耗的那张牌）、``caused_by_ethereal``（是不是虚无造成的）。
    on_card_exhausted: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterBlockGained``：拥有者**获得格挡之后**触发（``JuggernautPower``）。
    #: 参数：``gained``（全部修正之后的格挡量）、``card``（来源卡牌，非卡牌来源为 ``None``）。
    on_block_gained: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterSideTurnEnd``（带 ``side`` 的那次分发）：**任意一方**回合结束都触发，
    #: 与 ``on_owner_turn_end``（只在自己这边结束时）区分（``FlameBarrierPower``）。
    on_side_turn_end: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterSideTurnEndLate``：``Hook.AfterSideTurnEnd`` 的**第二遍**遍历
    #: （``Hook.cs`` 先跑完所有模型的 ``AfterSideTurnEnd``，再跑所有模型的
    #: ``AfterSideTurnEndLate``）。``DisintegrationPower`` 的"回合末自伤"挂在这里。
    #: 参数与 :data:`on_side_turn_end` 相同：``side``（结束的是哪一方）。
    on_side_turn_end_late: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterPlayerTurnStart``：**玩家**回合开始、抽牌之后（``InfernoPower`` /
    #: ``CrimsonMantlePower`` 的"每回合开始掉血"）。
    on_after_player_turn_start: Callable[[PowerContext, int], str | None] | None = None
    #: ``BeforeCardPlayed``：一张牌**开始结算之前**（``JugglingPower`` 数第 3 张攻击牌、
    #: ``RupturePower`` 登记"这张牌正在被打出"）。参数：``card``。
    on_before_card_played: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterPowerAmountChanged``：某个能力被施加时**通知双方所有模型**。
    #: 参数：``pid`` / ``applier``（施加者）/ ``applied``（这一次的层数）。
    #: ``ViciousPower`` 靠它实现"我给别人上易伤就抽牌"。
    on_power_applied: Callable[[PowerContext, int], str | None] | None = None
    #: ``BeforeSideTurnStart``：某一方回合开始、**清格挡之前**（``AggressionPower``）。
    #: 参数：``side``。
    on_before_side_turn_start: Callable[[PowerContext, int], str | None] | None = None
    #: ``AfterAutoPostPlayPhaseEntered``：玩家结束回合进入 AutoPostPlay 阶段
    #: （手牌还完整时；``StampedePower`` 在这里自动打出攻击牌）。
    on_auto_post_play: Callable[[PowerContext, int], str | None] | None = None
    block_additive: bool = False
    damage_additive: bool = False
    duration: bool = False
    #: ``BeforeSideTurnEnd``：某一方回合结束**之前**（``TheBombPower`` / ``HailstormPower``）。
    on_before_side_turn_end: object = None
    #: ``BeforeHandDraw``：玩家回合**抽手牌之前**（``CallOfTheVoidPower`` 等生成牌）。
    on_before_hand_draw: object = None
    #: ``AfterCardGeneratedForCombat``：战斗内生成一张牌之后。
    on_card_generated_for_combat: object = None
    #: ``AfterPowerAmountChanged``：**任意**能力的层数发生变化之后。
    on_power_amount_changed: object = None
    #: ``AfterShuffle``：**洗完牌之后**（``CardPileCmd.cs:1131``）。
    #: `StratagemPower`（计策）在这里从抽牌堆挑牌入手 —— 它会让抽牌挂起，
    #: 因为真机那是一个 ``await``（抽牌循环停住，选完再继续抽）。
    on_after_shuffle: object = None


# ==========================================================================
# 具体能力
# ==========================================================================
def _energy_next_turn(ctx: PowerContext, amount: int) -> str:
    """``EnergyNextTurnPower.AfterEnergyReset``：获得 N 点能量，然后**移除自身**。"""
    ctx.state.energy += amount
    ctx.events.append(f"{ctx.owner.name} 获得 {amount} 点能量（上回合留存）")
    return REMOVE


def _draw_cards_next_turn(ctx: PowerContext, amount: int) -> str:
    """``DrawCardsNextTurnPower.AfterHandDraw``：额外抽 N 张，然后移除自身。"""
    from .core import draw_cards
    draw_cards(ctx.state, amount, ctx.events)
    return REMOVE


def _block_next_turn(ctx: PowerContext, amount: int) -> str:
    """``BlockNextTurnPower.AfterBlockCleared``：获得 N 点格挡（Unpowered），然后移除。

    ⚠️ 必须走 ``CreatureCmd.GainBlock`` 的**同一条通道**（``core.gain_block``）：
    旧实现直接 ``ctx.owner.block += amount``，于是"获得格挡时"的订阅者
    （``JuggernautPower`` 板甲）**静默不触发** —— 格挡到手了，反伤却没打。
    """
    from . import core
    core.gain_block(ctx.state, ctx.owner, amount, ctx.events, unpowered=True,
                    label="下回合格挡")
    return REMOVE


def _star_next_turn(ctx: PowerContext, amount: int) -> str:
    """``StarNextTurnPower``：获得 N 颗星。"""
    ctx.state.stars = getattr(ctx.state, "stars", 0) + amount
    return REMOVE


def _poison(ctx: PowerContext, amount: int) -> str:
    """``PoisonPower.AfterSideTurnStart`` → ``Trigger()``：中毒者回合开始时掉血。

    真机 ``Trigger`` 是**循环**：每次迭代造成 ``Amount`` 点伤害后**立刻递减 1**，
    所以第 i 次自然是 ``Amount - i``（``CalculateTotalDamageNextTurn`` 只是给 UI
    预览用的求和，机制上与循环等价）。

    触发次数 ``TriggerCount = min(层数, 1 + 对方加速层数)`` —— 默认 1 次，
    ``AccelerantPower`` 才会让它多触发。

    伤害带 ``Unblockable | Unpowered``：**不格挡、不受力量/虚弱影响**。

    ⚠️ **递减这一步不能省**：只算总伤害而不递减的话，中毒会每回合都打满层数、
    永不衰减 —— 总伤害高得离谱，而且"中毒是无视格挡的持续消耗"这个战术定位
    会完全变样。实测漏掉过一次。
    """
    from .core import deal_raw_damage
    triggers = amount
    for opponent in ctx.opponents_of(ctx.owner):
        triggers = min(amount, 1 + opponent.power("accelerant"))
    total = sum(max(0, amount - index) for index in range(max(0, triggers)))
    if total:
        deal_raw_damage(ctx.state, ctx.owner, total, ctx.events, "中毒")
    if triggers > 0 and ctx.owner.alive():
        ctx.owner.add_power("poison", -triggers)      # 对应 `PowerCmd.Decrement`
    return KEEP


def _confused(ctx: PowerContext, amount: int) -> str:
    """ConfusedPower.AfterCardDrawn：每次抽牌以 CombatEnergyCosts.NextInt(4) 改费用。"""
    if ctx.owner is not ctx.state.player:
        return KEEP
    card = ctx.card
    definition = card.definition()
    # CardEnergyCost 构造将 X 费的 Canonical 设为 0：仍掷随机数，但实际花费仍为 X。
    if not definition.is_x_cost and definition.cost < 0:
        return KEEP
    card.combat_cost_override = ctx.state.hidden.rng.next_index("combat_energy_costs", 4)
    # GetWithModifiers 按加入顺序计算；新本场 Absolute 覆盖此前的临时免费。
    # 后续再施加的免费仍可覆盖它，临时免费到期后恢复本场费用。
    card.free_this_turn = False
    return KEEP


def _demise(ctx: PowerContext, amount: int) -> str:
    """DemisePower.AfterSideTurnEnd：本阵营结束时穿透格挡掉血，不递减层数。

    源码的 dealer/cardSource 都为 null；复用伤害通道才能保留 Buffer 与死亡钩子。
    """
    from .core import deal_raw_damage
    owner_side = "player" if ctx.is_player(ctx.owner) else "enemy"
    if ctx.side == owner_side:
        deal_raw_damage(ctx.state, ctx.owner, amount, ctx.events, "凋亡")
    return KEEP


def _disintegration(ctx: PowerContext, amount: int) -> str:
    """``DisintegrationPower.AfterSideTurnEndLate``：自己阵营回合末对自己造成 Amount 点伤害。

    ::

        // DisintegrationPower.cs:19-26
        public override async Task AfterSideTurnEndLate(..., CombatSide side,
                                                        IEnumerable<Creature> participants)
        {
            if (participants.Contains(base.Owner))
                await CreatureCmd.Damage(choiceContext, base.Owner, base.Amount,
                                         ValueProp.Unpowered, base.Owner);
        }

    ⚠️ 三个细节，少一个都会静默偏掉：

    * 判据是"**自己这一方**结束"（``participants.Contains(Owner)``）—— 与
      :func:`_demise` 同形，所以要把 ``side`` 与拥有者阵营比对；
    * 是 ``ValueProp.Unpowered``：力量**不**加成、**不**吃易伤倍率，但**格挡可以吸收**
      —— 必须走 :func:`core.deal_damage` 的 ``unpowered=True`` 通道，
      用"不经格挡的掉血"（``deal_raw_damage``）会让格挡白给；
    * ``dealer`` 是**拥有者自己**（源码第 5 个实参 ``base.Owner``），不是 null。
      层数**不递减**（源码里没有 ``Decrement``）。
    """
    from .core import deal_damage
    owner_side = "player" if ctx.is_player(ctx.owner) else "enemy"
    if ctx.side != owner_side or amount <= 0:
        return KEEP
    deal_damage(ctx.state, ctx.owner, ctx.owner, amount, ctx.events, unpowered=True)
    return KEEP


def sloth_cards_played(combatant: "Combatant") -> int:
    """``SlothPower.DisplayAmount``：本回合在怠惰限制下**已经打出**的牌数。

    真机的私有字段 ``_cardsPlayedThisTurn`` 被 ``DisplayAmount`` 覆盖显示出去，
    所以它是玩家可见状态（UI 上显示的是"已出几张"，不是层数）。
    """
    return combatant.power_vars.get("sloth", {}).get("played", 0)


def _sloth_card_played(ctx: PowerContext, amount: int) -> str:
    """``SlothPower.BeforeCardPlayed``：拥有者每打出**自己的一张牌**，私有计数 +1。

    ::

        if (cardPlay.Card.Owner != base.Owner.Player) return Task.CompletedTask;
        _cardsPlayedThisTurn++;

    ⚠️ 判据是"牌主 == 能力拥有者的玩家"。引擎里只有玩家有牌堆，所以
    "能力不在玩家身上"（``base.Owner.Player == null``）就等价于不计数 ——
    不能只按 ``holder`` 是玩家就无脑计数，那样敌人身上的怠惰也会累加。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    if getattr(ctx, "card", None) is None:
        return KEEP
    flags = ctx.owner.power_vars.setdefault("sloth", {})
    flags["played"] = flags.get("played", 0) + 1
    return KEEP


def _sloth_reset(ctx: PowerContext, amount: int) -> str:
    """``SlothPower.BeforeSideTurnStart``：**拥有者参与的那一方**回合开始时计数归零。

    ::

        if (!participants.Contains(base.Owner)) return Task.CompletedTask;
        _cardsPlayedThisTurn = 0;

    ⚠️ 归零的是"这一方回合开始"，不是"拥有者自己的回合"—— 单人里两者相同，
    但判据要照抄 ``participants``，否则以后多人会错。归零**不动**层数：
    层数是上限（``Amount``），归零的是私有进度。
    """
    owner_side = "player" if ctx.is_player(ctx.owner) else "enemy"
    if ctx.side != owner_side:
        return KEEP
    flags = ctx.owner.power_vars.get("sloth")
    if flags:
        flags["played"] = 0
    return KEEP


#: **能力级** ``ShouldPlay`` 覆写者：``pid → 判据``。
#:
#: 真机 ``AbstractModel.ShouldPlay`` 是**同一批**虚方法：卡牌覆写它（``Normality`` /
#: ``Enthralled``，见 ``core.CARD_SHOULD_PLAY``）、能力也覆写它（``SlothPower``）。
#: ``Hook.ShouldPlay`` 遍历所有 listener，**任一**返回 false 就拒绝这次出牌。
#: 引擎两张表分开只是"来源不同"，判据语义必须一致（同一个 ``card`` + ``autoPlayType``）。
def _stratagem(ctx: PowerContext, amount: int) -> str:
    """``StratagemPower.AfterShuffle``：洗完牌后从**抽牌堆**挑 ``Amount`` 张入手。

    ::

        if (player != base.Owner.Player) return;
        foreach (CardModel item in await CardSelectCmd.FromCombatPile(
                     ctx, PileType.Draw.GetPile(Owner.Player), Owner.Player,
                     new CardSelectorPrefs(SelectionScreenPrompt, base.Amount)))
            await CardPileCmd.Add(item, PileType.Hand);

    三条都不能省：

    * 候选来自**抽牌堆**（``PileType.Draw``）—— 而抽牌堆顺序是 L3 隐藏信息，
      所以候选顺序走 :meth:`PendingSelection.candidates` 里那条**规范排序**
      （按 ``(cid, upgraded)``），绝不把隐藏顺序编码进观测；
    * 是**移动入手**（``CardPileCmd.Add``），**不**派发"抽到牌"的钩子 ——
      `ConfusedPower` / `SpeedsterPower` 那类"抽牌时"的能力不该被它触发；
    * 能力**不移除**、层数也不减（源码里没有 ``PowerCmd.Remove`` / ``Decrement``）。
    """
    if ctx.owner is not ctx.state.player or amount <= 0:
        return KEEP
    from .core import open_pile_selection
    open_pile_selection(ctx.state, ctx.events, purpose="to_hand",
                        source_pile="draw", amount=amount,
                        actor=ctx.state.player)
    return KEEP


def _foregone_conclusion(ctx: PowerContext, amount: int) -> str:
    """``ForegoneConclusionPower.BeforeHandDraw``：抽牌前挑牌入手，然后**移除自己**。

    ::

        if (player == base.Owner.Player) {
            await CardPileCmd.ShuffleIfNecessary(choiceContext, base.Owner.Player);
            await CardPileCmd.Add(await CardSelectCmd.FromCombatPile(
                choiceContext, PileType.Draw.GetPile(Owner.Player), Owner.Player,
                new CardSelectorPrefs(SelectionScreenPrompt, base.Amount)), PileType.Hand);
            await PowerCmd.Remove(this);
        }

    四件事按源码顺序：

    1. ``ShuffleIfNecessary`` —— **抽牌堆为空才洗**；洗牌又会派发 ``AfterShuffle``，
       所以 ``StratagemPower`` 可能在这里**先**挂住选择（嵌套 ``await``）；
    2. 从抽牌堆挑 ``Amount`` 张；
    3. ``CardPileCmd.Add`` 入手（**移动**，不派发"抽到牌"的钩子）；
    4. ``PowerCmd.Remove(this)`` —— **整个移除**这个能力，不是减层。

    ⚠️ 两个顺序不能错：

    * 移除必须在**选完之后**（挂起时能力还在），由
      ``PendingSelection.remove_power_after`` 承载；
    * 洗牌被别的能力挂住时，本能力的"选牌 + 移除"要**排队**等它
      （``core.PowerTask``），不能丢 —— 丢了这张牌就白打了，而且不报错。
    """
    if ctx.owner is not ctx.state.player or amount <= 0:
        return KEEP
    from . import core
    state = ctx.state
    # ① `ShuffleIfNecessary`：只有抽牌堆空才洗。
    if not state.draw_pile and core.shuffle_piles(state, ctx.events):
        state.pending_tasks.append(core.PowerTask(
            pid=ctx.pid, kind="select_to_hand", amount=amount,
            remove=True, actor=state.player))
        return KEEP
    # ②③④ 挑牌 → 入手 → （选完）整个移除自己
    if core.open_pile_selection(state, ctx.events, purpose="to_hand",
                                source_pile="draw", amount=amount,
                                actor=state.player):
        state.pending.remove_power_after = ctx.pid
        return KEEP
    # 没有候选：真机的 `Add` 是空操作，但 `PowerCmd.Remove` **无条件**执行。
    core._remove_power_whole(state, state.player, ctx.pid, ctx.events)
    return KEEP


def _entropy(ctx: PowerContext, amount: int) -> str:
    """``EntropyPower.AfterPlayerTurnStart``：挑 ``Amount`` 张手牌**随机转化**。

    ::

        if (player != base.Owner.Player) return;
        CardSelectorPrefs prefs = new CardSelectorPrefs(TransformSelectionPrompt, base.Amount);
        List<CardModel> list = (await CardSelectCmd.FromHand(ctx, player, prefs, null, this)).ToList();
        foreach (CardModel item in list) {
            await CardCmd.TransformToRandom(item, player.RunState.Rng.CombatCardSelection);
        }

    三条：

    * 时机是**玩家回合开始、抽牌之后**（``AfterPlayerTurnStart``）；
    * 候选是**手牌**；选出来的牌被**原位**换成随机的另一张 —— 候选池见
      :func:`core.transformation_options`（真机 ``CardFactory`` 的过滤链），
      随机流是 ``combat_card_selection``；
    * 能力**不移除**（源码里没有 ``PowerCmd.Remove``）：层数就是每回合能转几张。

    ⚠️ 挂起时整个"回合开始的剩余步骤"都要等（真机是同一个 ``await``），
    由 ``core.CombatState.turn_start_step`` 记到第几步。
    """
    if ctx.owner is not ctx.state.player or amount <= 0:
        return KEEP
    from .core import open_pile_selection
    open_pile_selection(ctx.state, ctx.events, purpose="transform",
                        source_pile="hand", amount=amount,
                        actor=ctx.state.player)
    return KEEP


def _nightmare(ctx: PowerContext, amount: int) -> str:
    """``NightmarePower.BeforeHandDraw``：把选中的那张牌**克隆 Amount 份入手**，然后移除自己。

    ::

        if (player == base.Owner.Player) {
            CardModel card = GetInternalData<Data>().selectedCard;
            for (int i = 0; i < base.Amount; i++) {
                CardModel card2 = card.CreateClone();
                await CardPileCmd.AddGeneratedCardToCombat(card2, PileType.Hand, base.Owner.Player);
            }
            await PowerCmd.Remove(this);
        }

    四条：

    * 只在**拥有者是玩家**时触发；
    * 克隆的是 `SetSelectedCard` 存下来的**快照**（`ctx.instance_vars`），
      不是手牌里那张牌 —— 原卡后来被升级 / 转化都不该改到它；
    * 走 `AddGeneratedCardToCombat`：会派发 `AfterCardGeneratedForCombat`
      与 `AfterCardEnteredCombat`（`ArsenalPower` 那类"生成牌就…"的能力要能看到，
      苦痛族也要给这些新牌补贴标记）；
    * 最后 `PowerCmd.Remove(this)` —— **只移除这一个实例**（S01）：两张 `Nightmare`
      各自兑现一次，不能一次清光。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    snapshot = ctx.instance_vars.get("selected_card")
    if snapshot is None:
        # 真机在 `selectedCard == null` 时**根本不会 Apply**（`Apply` 与
        # `SetSelectedCard` 都在同一个 `if (selectedCard != null)` 里）。
        # 引擎的效果表是顺序的、没有条件，所以会出现"实例在、载荷为空" ——
        # 如实移除它，而不是每回合空转一次。
        ctx.events.append("梦魇没有选中的卡，移除")
        return REMOVE
    from .core import _add_generated_card
    for _ in range(max(0, amount)):
        _add_generated_card(ctx.state, snapshot.clone(), "hand")
    ctx.events.append(f"梦魇生成 {amount} 张 {snapshot.definition().name}")
    return REMOVE


def _sloth_allows(state: "CombatState", card, auto: bool) -> bool:
    """``SlothPower.ShouldPlay``：``已出牌数 < 层数`` 才放行。

    ::

        if (card.Owner.Creature != base.Owner) return true;
        return _cardsPlayedThisTurn < base.Amount;   // SlothPower.cs:22-29

    ⚠️ 两条不能"顺手优化"的地方：

    * 上限比的是 **``<``**：层数 3 时第 4 张被拒、前 3 张放行；
    * ``autoPlayType`` 被源码丢弃（参数名 ``_``）—— **自动打出同样受限制**，
      而 ``Enthralled`` 是放行自动打出的。两张表不能共用一条"自动打出免检"。

    牌主那一半（``card.Owner.Creature != base.Owner``）由调用点保证：
    ``core.card_playable`` 只在玩家出牌时查询，而能力挂在玩家身上时
    任何手牌的主人都等于它。
    """
    player = state.player
    amount = player.power("sloth")
    if amount <= 0:
        return True
    return sloth_cards_played(player) < amount


POWER_SHOULD_PLAY: dict[str, "Callable[[CombatState, object, bool], bool]"] = {
    "sloth": _sloth_allows,
}


def should_play_allows(state: "CombatState", card, auto: bool) -> bool:
    """``Hook.ShouldPlay`` 的**能力侧**：任一能力否决就返回 False。

    ``auto`` 是这次出牌是不是自动打出（真机 ``AutoPlayType != null``）。
    两种覆写者对它态度不同：``Normality`` 不看它（自动打出也照样锁），
    ``SlothPower`` 也不看它（参数名是 ``_``）。所以**不要**在这里统一过滤自动打出。
    """
    player = getattr(state, "player", None)
    if player is None:
        return True
    for pid, rule in POWER_SHOULD_PLAY.items():
        if player.power(pid) > 0 and not rule(state, card, auto):
            return False
    return True


def _plating(ctx: PowerContext, amount: int) -> str:
    """``PlatingPower.BeforeSideTurnEndEarly``：回合结束获得 N 点格挡。

    ⚠️ **是 ``Unpowered`` 的格挡**：不吃敏捷、也不被脆弱打折。

    递减在**回合开始**（``AfterSideTurnStart``），不是这里 —— 见 :func:`_plating_tick`。
    """
    ctx.owner.block += amount
    ctx.events.append(f"{ctx.owner.name} 获得 {amount} 点格挡（镀层）")
    return KEEP


def _plating_tick(ctx: PowerContext, amount: int) -> str:
    """``PlatingPower.AfterSideTurnStart``：回合开始减 1 层。

    真机带**两个豁免**（照抄，别简化）::

        if (participants.Contains(Owner)
            && (Owner.Player == null || Owner.Player.PlayerCombatState.TurnNumber != 1)
            && (Owner.Side != CombatSide.Enemy || combatState.RoundNumber != 1))

    即：玩家在第 1 回合不减、敌人在第 1 轮不减。少了递减，
    镀层会变成"永远每回合 N 点格挡"；有了递减，总量才是 N+(N-1)+…+1。
    """
    if amount <= 0:
        return KEEP
    turn = getattr(ctx.state, "turn", 1)
    is_enemy = ctx.owner is not getattr(ctx.state, "player", None)
    if is_enemy:
        if turn > 1:                       # 敌方：`RoundNumber != 1`
            ctx.owner.add_power("plating", -1)
    elif turn != 1:                        # 玩家：`TurnNumber != 1`
        ctx.owner.add_power("plating", -1)
    return KEEP


def _regeneration(ctx: PowerContext, amount: int) -> str:
    """``RegenPower.BeforeSideTurnEndEarly``：回合结束回复 N 点生命，**然后递减 1 层**。

    ⚠️ 治疗与递减是**同一步**里的两件事（源码里紧挨着）：

    .. code-block:: csharp

       await CreatureCmd.Heal(base.Owner, base.Amount);
       await PowerCmd.Decrement(this);

    所以 N 层再生的总治疗量是 ``N + (N-1) + … + 1``。少了递减那一行，
    "5 层再生"会从"总共回 15 点"变成"**永远每回合回 5 点**"。
    """
    healed = min(amount, ctx.owner.max_hp - ctx.owner.hp)
    if healed > 0:
        ctx.owner.hp += healed
        ctx.events.append(f"{ctx.owner.name} 回复 {healed} 点生命（再生）")
    ctx.owner.add_power("regen", -1)
    return KEEP


def _thorns(ctx: PowerContext, amount: int) -> str:
    """``ThornsPower.AfterDamageReceived``：被攻击后反弹 N 点伤害。

    ⚠️ 真机的反伤发生在**受到攻击伤害之后**，且不受攻击者格挡影响。
    当前引擎只在玩家被敌人攻击时触发（敌人互相攻击的情况本作很少见）。
    """
    attacker = getattr(ctx, "attacker", None)
    if attacker is None or not attacker.alive():
        return KEEP
    from .core import deal_raw_damage
    deal_raw_damage(ctx.state, attacker, amount, ctx.events, "荆棘")
    return KEEP


def _no_draw_expire(ctx: PowerContext, amount: int) -> str:
    """``NoDrawPower.AfterSideTurnEnd``：拥有者阵营回合结束时**整个移除**。

    ⚠️ 它是 ``StackType.Single``（只记有无），不是递减 —— 当成递减会让它多留一回合。
    """
    return REMOVE


def _retain_hand_expire(ctx: PowerContext, amount: int) -> str:
    """``RetainHandPower.AfterSideTurnEnd``：回合结束**递减 1**（不是移除）。"""
    if amount <= 1:
        return REMOVE
    ctx.owner.add_power("retain_hand", -1)
    return KEEP


def _territorial(ctx: PowerContext, amount: int) -> str:
    """``TerritorialPower.AfterSideTurnEnd``：拥有者阵营回合结束时获得 ``Amount`` 点力量。"""
    ctx.owner.add_power("strength", amount)
    ctx.events.append(f"{ctx.owner.name} 获得 {amount} 点力量（领地意识）")
    return KEEP


def _asleep(ctx: PowerContext, amount: int) -> str:
    """``AsleepPower.AfterSideTurnEnd``：倒计时；归零时怪**醒来**（``WakeUpMove``）。

    真机：``PowerCmd.Decrement(this)``，归零后调用 ``WakeUpMove`` —— 而拉瓦金的
    ``WakeUpMove`` 只有动画/音效（**不**移除镀层），镀层是在**睡着倒计时到 1**
    的那一步由 ``BeforeSideTurnEndVeryEarly`` 移除的（见 :func:`_asleep_plating`）。
    引擎把"醒来"表达成**移除该能力**：怪的 AI 用 ``HasPower<AsleepPower>()``
    分支决定是睡还是打（``SLEEP_BRANCH``），所以移除能力就等于醒来。
    """
    if amount <= 1:
        ctx.events.append(f"{ctx.owner.name} 醒来")
        return REMOVE
    ctx.owner.add_power("asleep", -1)
    return KEEP


def _asleep_plating(ctx: PowerContext, amount: int) -> str:
    """``AsleepPower.BeforeSideTurnEndVeryEarly``：倒计时到 1 且还有镀层 → 移除镀层。

    源码::

        if (participants.Contains(base.Owner) && base.Amount <= 1
            && base.Owner.HasPower<PlatingPower>())
            await PowerCmd.Remove(base.Owner.GetPower<PlatingPower>());

    也就是"醒来**前一步**就把镀层扔掉" —— 少了它，拉瓦金醒来时还带着镀层
    （比真机硬，而且刚好卡在"醒来"这个节点上，看起来像正常的强化）。
    """
    if _side_of(ctx) != _owner_side(ctx) or amount > 1:
        return KEEP
    if ctx.owner.power("plating") > 0:
        ctx.owner.add_power("plating", -ctx.owner.power("plating"))
        ctx.events.append(f"{ctx.owner.name} 的镀层随长眠消散")
    return KEEP


def _asleep_wake(ctx: PowerContext, amount: int) -> str:
    """``AsleepPower.AfterDamageReceived``：**掉血**就醒 —— 移除镀层、击晕（唤醒）并移除自己。

    源码::

        if (target == base.Owner && result.UnblockedDamage != 0) {
            if (base.Owner.HasPower<PlatingPower>()) Remove(Plating);
            monster.IsAwake = true;
            await CreatureCmd.Stun(base.Owner, monster.WakeUpMove, "SLASH_MOVE");
            await PowerCmd.Remove(this);
        }

    ⚠️ 判据是"**掉了血**"（``UnblockedDamage != 0``）：被完全格挡打不醒。
    ``Stun(..., WakeUpMove, "SLASH_MOVE")`` = 这一回合走唤醒动作、**下一招固定猛击**。
    """
    if getattr(ctx, "unblocked", 0) <= 0:
        return KEEP
    if ctx.owner.power("plating") > 0:
        ctx.owner.add_power("plating", -ctx.owner.power("plating"))
        ctx.events.append(f"{ctx.owner.name} 的镀层被击碎")
    stun(ctx.owner, ctx.state, ctx.events, action="remove_plating", follow_up="slash")
    return REMOVE


def _slumber(ctx: PowerContext, amount: int) -> str:
    """``SlumberPower.AfterSideTurnEnd``：同样是倒计时，归零即醒。

    真机归零后调 ``SlumberingBeetle.WakeUpMove`` —— 那里面有"有镀层就移除"。
    """
    if amount <= 1:
        ctx.events.append(f"{ctx.owner.name} 醒来")
        if ctx.owner.power("plating") > 0:
            ctx.owner.add_power("plating", -ctx.owner.power("plating"))
            ctx.events.append(f"{ctx.owner.name} 的镀层随苏醒消散")
        return REMOVE
    ctx.owner.add_power("slumber", -1)
    return KEEP


def _slumber_wake(ctx: PowerContext, amount: int) -> str:
    """``SlumberPower.AfterDamageReceived``：**掉血**就递减，归零则醒（击晕 + 唤醒动作）。

    源码::

        if (target == base.Owner && result.UnblockedDamage != 0) {
            await PowerCmd.Decrement(this);
            if (base.Amount <= 0) await CreatureCmd.Stun(owner, beetle.WakeUpMove, "ROLL_OUT_MOVE");
        }

    醒来动作里唯一的数值部分是"移除镀层"（``remove_plating``）；下一招固定滚撞。
    """
    if getattr(ctx, "unblocked", 0) <= 0:
        return KEEP
    ctx.owner.add_power("slumber", -1)
    if ctx.owner.power("slumber") > 0:
        return KEEP
    stun(ctx.owner, ctx.state, ctx.events, action="remove_plating",
         follow_up="roll_out")
    return REMOVE
    """``VigorPower``：下一次攻击额外造成 N 点伤害，**攻击后移除**。

    这里只声明为"伤害加法"，消耗时机由 ``core`` 在攻击结算后处理 ——
    与 ``strength`` 的区别正是"用一次就没了"。
    """
    return KEEP


def _flutter(ctx: PowerContext, amount: int) -> str:
    """``FlutterPower.AfterDamageReceived``：受到**有效攻击伤害**就减 1 层。

    ``ModifyDamageMultiplicative`` 那半边（×0.50）在
    :data:`DAMAGE_MULTIPLIERS` 里；这里只管层数。
    归零时真机会把拥有者**击晕**（盗贼跳虫落地的机制），那一步需要 AI 状态机配合，
    本引擎尚未实现 —— 已在缺口清单里列明，不静默略过。
    """
    if getattr(ctx, "unblocked", 0) != 0 and getattr(ctx, "powered", True):
        ctx.owner.add_power("flutter", -1)
        ctx.events.append(f"{ctx.owner.name} 的滑翔减到 {ctx.owner.power('flutter')}")
    return KEEP


def _shrink_tick(ctx: PowerContext, amount: int) -> str:
    """``ShrinkPower.AfterSideTurnEnd``：拥有者回合结束减 1 层（负层数表示永久）。

    负数 = ``IsInfinite``（``AllowNegative`` 为 true，且 ``Amount < 0`` 时
    堆叠类型变成 ``Single``）—— 那种情况下**不递减**。
    """
    if amount > 0:
        ctx.owner.add_power("shrink", -1)
    return KEEP


def _surrounded_face(ctx: PowerContext, amount: int) -> str:
    """``SurroundedPower`` 的朝向更新（``BeforeCardPlayed`` / ``BeforePotionUsed``）。

    真机按"玩家这回合打的是**左边还是右边**的敌人"翻转朝向；被包围者只在
    **背后**被打时吃 ×1.5。引擎用一张固定的朝向标记表达，翻面条件照抄：
    朝右时若打的是带 ``back_attack_left`` 的敌人 → 翻到左，反之亦然。
    """
    target_facing = getattr(ctx, "target_facing", "")
    if not target_facing:
        return KEEP
    if ctx.owner.power_facing == "right" and target_facing == "left":
        ctx.owner.power_facing = "left"
    elif ctx.owner.power_facing == "left" and target_facing == "right":
        ctx.owner.power_facing = "right"
    return KEEP


def _hatch_tick(ctx: PowerContext, amount: int) -> str:
    """``HatchPower.AfterSideTurnEnd``：拥有者回合结束减 1 层（纯倒计时）。"""
    if amount > 0:
        ctx.owner.add_power("hatch", -1)
        ctx.events.append(f"{ctx.owner.name} 的孵化剩 {ctx.owner.power('hatch')}")
    return KEEP


def _nemesis(ctx: PowerContext, amount: int) -> str:
    """``NemesisPower.AfterSideTurnEnd``：**隔回合**给自己上 1 层无形，再隔回合撤掉。

    ::

        _shouldApplyIntangible = !_shouldApplyIntangible;
        if (_shouldApplyIntangible) Apply<IntangiblePower>(owner, 1);
        else if (owner.HasPower<IntangiblePower>()) Remove(...);

    所以复仇之影是"挨打一回合、免伤一回合"交替。少写一半（只加不减）会让它**永久免伤**。
    """
    toggle = ctx.owner.power_flags.get("nemesis", 0)
    if toggle:
        ctx.owner.power_flags["nemesis"] = 0
        if ctx.owner.power("intangible") > 0:
            ctx.owner.powers.pop("intangible", None)
            ctx.events.append(f"{ctx.owner.name} 失去了无形")
    else:
        ctx.owner.power_flags["nemesis"] = 1
        ctx.owner.add_power("intangible", 1)
        ctx.events.append(f"{ctx.owner.name} 获得无形")
    return KEEP


def _constrict(ctx: PowerContext, amount: int) -> str:
    """``ConstrictPower.AfterSideTurnEnd``：拥有者回合结束受到 ``Amount`` 点**无源**伤害。

    ``CreatureCmd.Damage(..., ValueProp.Unpowered, base.Owner)`` —— 施加者是自己，
    所以**不吃任何加成**（无力量、无易伤），但**可以格挡**。
    """
    from . import core
    core.deal_damage(ctx.state, None, ctx.owner, amount, ctx.events, unpowered=True)
    return KEEP


def _painful_stabs(ctx: PowerContext, amount: int) -> str:
    """``PainfulStabsPower.AfterAttack``：**自己**打出有效攻击时，往玩家弃牌堆塞 1 张伤口。

    每命中一次塞一张（真机遍历 ``command.Results`` 里的每一次命中）。
    ``ShouldCreatureBeRemovedFromCombatAfterDeath`` 为 false —— 死后仍留在场上结算。
    """
    if getattr(ctx, "unblocked", 0) <= 0 and not getattr(ctx, "powered", True):
        return KEEP
    from . import core
    for _ in range(max(1, amount)):
        core._apply_effects(ctx.state, [core.Effect(op="add_card", amount=1,
                                                    card="wound", target="self",
                                                    pile="discard")],
                            ctx.state.player, -1, ctx.events)
    return KEEP


def _personal_hive(ctx: PowerContext, amount: int) -> str:
    """``PersonalHivePower.AfterDamageReceived``：被**有效攻击**打中时往玩家抽牌堆塞 ``Amount`` 张眩晕。

    ⚠️ 位置是 ``CardPilePosition.Random``（**随机插进抽牌堆**），不是放牌堆顶 ——
    放顶部会让"下一次抽牌必抽到眩晕"，强度完全不同。
    """
    if getattr(ctx, "from_card", False) is False:
        return KEEP
    from . import core
    for _ in range(max(1, amount)):
        core._apply_effects(ctx.state, [core.Effect(op="add_card", amount=1,
                                                    card="dazed", target="self",
                                                    pile="draw", position="random")],
                            ctx.state.player, -1, ctx.events)
    return KEEP


def _imbalanced(ctx: PowerContext, amount: int) -> str:
    """``ImbalancedPower.AfterDamageGiven``：**自己打出的伤害被完全格挡**时，自己失衡。

    ::

        if (dealer == base.Owner && result.WasFullyBlocked)
        {
            if (!(base.Owner.Monster is BowlbugRock bowlbugRock))
                await CreatureCmd.Stun(base.Owner);
            else bowlbugRock.IsOffBalance = true;   // 盛碗虫改为标记失衡
        }

    盛碗虫不直接晕，而是置 ``IsOffBalance`` —— 它的 AI 靠这个标记转向
    ``DIZZY_MOVE``（见 ``BowlbugRock`` 的条件分支）。这里两种都做：
    置标记 + 记一笔，AI 侧由 ``IsOffBalance`` 读取。
    """
    if getattr(ctx, "was_fully_blocked", False) is False:
        return KEEP
    ctx.owner.power_flags["off_balance"] = 1
    ctx.events.append(f"{ctx.owner.name} 的攻击被完全格挡 → 失衡")
    return KEEP


def _suck(ctx: PowerContext, amount: int) -> str:
    """``SuckPower.AfterAttack``：**自己**打出有效攻击且造成掉血 → +``Amount`` 力量。

    真机按"命中的每一组里有没有掉血"计数（多段攻击可能多组），
    这里按每次掉血计一次 —— 引擎每次伤害是独立调用，与真机的分组
    在**单段/多段**两种情形下结果一致。
    """
    if getattr(ctx, "unblocked", 0) <= 0 or not getattr(ctx, "powered", True):
        return KEEP
    ctx.owner.add_power("strength", amount)
    ctx.events.append(f"{ctx.owner.name} 汲取力量 +{amount}")
    return KEEP


def _shriek(ctx: PowerContext, amount: int) -> str:
    """``ShriekPower.AfterDamageReceived``：掉血后**血量 ≤ 层数**即被击晕并移除自身。

    ``Amount`` 是**血量阈值**（不是伤害量）：怪被打到残血就晕一回合。
    层数可为负（``AllowNegative``），负值时阈值 ≤ 0，永远不会触发。
    """
    if getattr(ctx, "unblocked", 0) <= 0:
        return KEEP
    if ctx.owner.hp > amount:
        return KEEP
    stun(ctx.owner, ctx.state, ctx.events, source="尖啸")
    return REMOVE


def _plow(ctx: PowerContext, amount: int) -> str:
    """``PlowPower.AfterDamageReceived``：掉血后**血量 ≤ 层数**→ 清除临时力量并击晕。

    真机把身上所有 ``TemporaryStrengthPower`` 清掉（那些是"本回合临时力量"），
    再 ``CreatureCmd.Stun``。引擎里临时力量用 ``temporary_strength`` 表示。
    """
    if getattr(ctx, "unblocked", 0) <= 0 or ctx.owner.hp > amount:
        return KEEP
    ctx.owner.powers.pop("temporary_strength", None)
    stun(ctx.owner, ctx.state, ctx.events, source="犁地")
    return REMOVE


def _ravenous(ctx: PowerContext, amount: int) -> str:
    """``RavenousPower.AfterDeath``：**同阵营队友阵亡** → 自身被击晕并 +``Amount`` 力量。

    ::

        if (!wasRemovalPrevented && target != base.Owner
            && target.Side == base.Owner.Side && !base.Owner.IsDead)
        { ... await CreatureCmd.Stun(base.Owner, StunnedMove);
              await PowerCmd.Apply<StrengthPower>(..., base.Amount, ...); }

    即"吞食同伴"：吃一口，晕一回合，永久加力量。
    """
    stun(ctx.owner, ctx.state, ctx.events, source="吞食")
    ctx.owner.add_power("strength", amount)
    ctx.events.append(f"{ctx.owner.name} 吞食同伴：力量 +{amount}")
    return KEEP


def _stock(ctx: PowerContext, amount: int) -> str:
    """``StockPower.AfterDeath``：**自己死亡时**在原地补一只（层数 -1）。

    ::

        if (!wasRemovalPrevented && target == base.Owner && base.Amount > 0)
        {
            Axebot axebot = ...; axebot.StockAmount = base.Amount - 1;
            await CreatureCmd.Add(axebot, base.CombatState, base.Owner.Side, base.Owner.SlotName);
        }

    所以"库存"是**同一种怪的连续增援**；``ShouldStopCombatFromEnding`` 为 true ——
    它还会补人时，战斗不能因为"场上没敌人了"而结束。
    """
    if amount <= 0:
        return KEEP
    spawned = spawn_enemy(ctx.state, "axebot", ctx.events)
    if spawned is not None:
        spawned.add_power("stock", amount - 1)
        ctx.events.append(f"{ctx.owner.name} 的库存剩 {amount - 1}")
    return KEEP


def _surprise(ctx: PowerContext, amount: int) -> str:
    """``SurprisePower.AfterDeath``：**自己死亡时**召唤胖地精与鬼祟地精。

    ``ShouldStopCombatFromEnding = true`` —— 死的瞬间战斗不能判定为胜利，
    否则玩家会在援军出现前就"赢"了。
    """
    spawn_enemy(ctx.state, "fat_gremlin", ctx.events)
    spawn_enemy(ctx.state, "sneaky_gremlin", ctx.events)
    return KEEP


def _infested(ctx: PowerContext, amount: int) -> str:
    """``InfestedPower.AfterDeath``：**自己死亡时**召唤 4 只**眩晕状态**的扭动虫。

    ``wriggler.StartStunned = true`` → 它们入场时处于沉睡/眩晕（先挨打一回合）。
    四只分别占 ``wriggler1..4`` 槽位。
    """
    for _ in range(4):
        spawned = spawn_enemy(ctx.state, "wriggler", ctx.events)
        if spawned is not None:
            spawned.power_flags["stunned"] = 1
    return KEEP


def _hardened_shell(ctx: PowerContext, amount: int) -> str:
    """``HardenedShellPower.AfterDamageReceived``：累计本回合已经掉的血（供上限用），且**每回合重置**。

    ``ModifyHpLostBeforeOstyLate`` 那半边在 :func:`hp_loss_cap` 里：
    上限 = ``Amount - 本回合已掉血``。所以硬壳不是"每次免伤"，而是
    "一回合总共最多掉 ``Amount`` 点" —— 用累计值而不是单次值。
    """
    if getattr(ctx, "unblocked", 0) <= 0:
        return KEEP
    received = ctx.owner.power_flags.get("hardened_shell_received", 0)
    ctx.owner.power_flags["hardened_shell_received"] = received + ctx.unblocked
    return KEEP


def _hardened_shell_reset(ctx: PowerContext, amount: int) -> str:
    """``HardenedShellPower.BeforeSideTurnStart``：**每个回合开始**清空累计掉血。"""
    ctx.owner.power_flags["hardened_shell_received"] = 0
    return KEEP


def _paper_cuts(ctx: PowerContext, amount: int) -> str:
    """``PaperCutsPower.AfterDamageGiven``：自己打出有效攻击且造成掉血 → 玩家**失去上限生命**。

    ::

        if (dealer == base.Owner && target.IsPlayer
            && props.IsPoweredAttack() && result.UnblockedDamage > 0)
            await CreatureCmd.LoseMaxHp(choiceContext, target, base.Amount, isFromCard: false);

    ⚠️ 是 ``LoseMaxHp``（**永久的**最大生命），不是伤害 ——
    不吃格挡、不吃力量，也不随战斗结束恢复。这是纸割最狠的地方。
    """
    if getattr(ctx, "unblocked", 0) <= 0 or not getattr(ctx, "powered", True):
        return KEEP
    player = ctx.state.player
    player.max_hp = max(1, player.max_hp - amount)
    if player.hp > player.max_hp:
        player.hp = player.max_hp
    ctx.events.append(f"{player.name} 失去 {amount} 点最大生命（纸割）")
    return KEEP


def _tangled(ctx: PowerContext, amount: int) -> str:
    """``TangledPower``：给玩家**所有攻击牌**贴 ``Entangled``（费用 +``Amount``）。

    贴上去的瞬间生效（``AfterApplied`` 遍历 ``AllCards``）；
    玩家回合结束时移除自身，并在 ``AfterRemoved`` 里清掉所有 ``Entangled``。
    """
    afflict_cards(ctx.state, "entangled", ctx.events, card_type="attack")
    return KEEP


def _tangled_clear(ctx: PowerContext, amount: int) -> str:
    """``TangledPower.AfterSideTurnEnd``：拥有者回合结束移除自身。"""
    return REMOVE


def _ringing(ctx: PowerContext, amount: int) -> str:
    """``RingingPower``：给玩家**所有**牌贴 ``Ringing``（打不出）。回合结束移除。"""
    afflict_cards(ctx.state, "ringing", ctx.events)
    return KEEP


def _ringing_clear(ctx: PowerContext, amount: int) -> str:
    """``RingingPower.AfterSideTurnEnd``：拥有者回合结束移除自身。"""
    return REMOVE


def _smoggy(ctx: PowerContext, amount: int) -> str:
    """``SmoggyPower.AfterCardPlayed``：玩家打出**技能牌**后，其余技能牌全部贴上 ``Smog``。

    ⚠️ 触发条件是"打出的牌是技能牌"（``cardPlay.Card.Type != CardType.Skill`` 直接 return），
    而且贴的是**技能牌**。写成"所有牌"会让玩家连攻击牌都打不出，是另一种游戏。
    """
    if getattr(ctx, "card_type", "") != "skill":
        return KEEP
    afflict_cards(ctx.state, "smog", ctx.events, card_type="skill")
    return KEEP


def _smoggy_clear(ctx: PowerContext, amount: int) -> str:
    """``SmoggyPower.AfterSideTurnEnd``：拥有者回合结束清掉所有 ``Smog``（能力本身留着）。"""
    cleared = clear_affliction(ctx.state, "smog")
    if cleared:
        ctx.events.append(f"烟雾散去（{cleared} 张）")
    return KEEP


def _hex(ctx: PowerContext, amount: int) -> str:
    """``HexPower``：给玩家所有牌贴 ``Hexed`` → 这些牌获得**虚无**（``Ethereal``）。

    源码里关键字是 ``TryModifyKeywordsInCombat`` 动态给的（不是改卡牌本身），
    所以能力一移除，虚无就自动消失 —— 引擎用 :func:`affliction_keywords` 同样表达。
    """
    afflict_cards(ctx.state, "hexed", ctx.events)
    return KEEP


def _chains_of_binding(ctx: PowerContext, amount: int) -> str:
    """``ChainsOfBindingPower``：**每抽一张牌**就贴 ``Bound``（每回合最多 ``Amount`` 张）。

    真机按"本回合已经贴过几张"计数（``CardAfflictedEntry`` 历史），
    所以是"每回合前 N 张抽到的牌被束缚"，不是"所有牌"。
    引擎用 ``power_flags`` 记本回合已贴数，回合开始归零。
    """
    used = ctx.owner.power_flags.get("bound_this_turn", 0)
    if used >= amount:
        return KEEP
    state = ctx.state
    card = getattr(ctx, "card", None)
    if card is None:
        return KEEP
    if card.affliction is None:
        card.affliction = "bound"
        ctx.owner.power_flags["bound_this_turn"] = used + 1
        ctx.events.append(f"{card.definition().name} 被束缚")
    return KEEP


def _panache_reset(ctx: PowerContext, amount: int) -> str:
    """``PanachePower.AfterSideTurnEnd``：``participants.Contains(Owner)`` → ``CardsLeft`` 复位 5。

    源码::

        if (!participants.Contains(base.Owner)) return;
        base.DynamicVars["CardsLeft"].BaseValue = 5m;

    ⚠️ 少了它，"打 4 张、回合结束、下回合再打 1 张"也会触发一次 ——
    进度**跨回合累计** = 每回合白送一次群伤（静默变强）。
    """
    ctx.owner.power_flags["panache_cards_left"] = 5
    return KEEP


def _chains_mark_played(ctx: PowerContext, amount: int) -> str:
    """``ChainsOfBindingPower.BeforeCardPlayed``：这张被打出的牌**带着 Bound** → 翻标志。

    源码::

        if (card.IsDupe) return;                       // 复制品不算
        if (card.Owner.Creature != base.Owner) return;
        if (!(card.Affliction is Bound)) return;
        data.boundCardPlayed = true;

    翻了这个标志之后，``ShouldPlay`` 就会把**其余**的 Bound 牌挡掉
    （见 :func:`affliction_blocks_play`）—— 所以"第一张能打、后面的打不出"。
    """
    card = getattr(ctx, "card", None)
    if card is None or getattr(card, "affliction", None) != "bound":
        return KEEP
    ctx.owner.power_flags["bound_played_this_turn"] = 1
    return KEEP


def _chains_reset(ctx: PowerContext, amount: int) -> str:
    """``ChainsOfBindingPower.BeforeSideTurnEnd``：拥有者阵营回合结束**之前**复位。

    真机在这里做两件事（都要）：

    * ``boundCardPlayed = false`` —— 每回合的"只能打一张 Bound"重新开张；
    * **清掉所有 ``Bound`` 苦痛** —— 束缚只持续到这一回合结束。

    只做第一件的话，牌上的 ``Bound`` 会无限累积（下回合抽到的新牌又叠一层）。
    """
    if _side_of(ctx) != _owner_side(ctx):
        return KEEP
    ctx.owner.power_flags["bound_played_this_turn"] = 0
    # 每回合"最多贴 Amount 张"的计数也要复位（真机按本回合的
    # `CardAfflictedEntry` 历史过滤）。
    ctx.owner.power_flags["bound_this_turn"] = 0
    cleared = clear_affliction(ctx.state, "bound")
    if cleared:
        ctx.events.append(f"束缚解除：{cleared} 张牌的 Bound 被清除")
    return KEEP


def _vital_spark(ctx: PowerContext, amount: int) -> str:
    """``VitalSparkPower``：给玩家**所有技能牌**贴 ``Tainted``。

    玩家**打出**一张带 ``Tainted`` 的牌时 → 给玩家自己上 ``Amount`` 层 ``TaintedPower``
    （``TaintedPower.ModifyDamageAdditive``：玩家受到的**有效攻击**伤害 +Amount）。
    所以"生命火花"是"用技能牌就要多挨打"。

    真机在 ``BeforeCombatStart`` 贴一次、``AfterCardEnteredCombat`` 补新进场的牌；
    引擎对应 ``on_applied``（贴上的那一刻在场的牌）+ ``on_card_entered_combat``
    （之后新生成的技能牌）—— 两半都要，少后一半会让"战斗中生成的技能牌"
    不带沾染（引擎比真机**弱**）。
    """
    afflict_cards(ctx.state, "tainted", ctx.events, card_type="skill")
    return KEEP


def _vital_spark_played(ctx: PowerContext, amount: int) -> str:
    """``VitalSparkPower.AfterCardPlayed``：打出的牌带 ``Tainted`` → 给玩家上 ``TaintedPower``。"""
    card = getattr(ctx, "card", None)
    if card is None or card.affliction != "tainted":
        return KEEP
    ctx.state.player.add_power("tainted", amount)
    ctx.events.append(f"{ctx.state.player.name} 沾染（受伤 +{amount}）")
    return KEEP


def _tainted_hurt(ctx: PowerContext, amount: int) -> str:
    """``TaintedPower.ModifyDamageAdditive``：拥有者受到的**有效攻击**伤害 +``Amount``。

    数值修正走伤害管线（``DAMAGE_ADDITIVE`` 之外的单列），所以这里不在钩子里改数 ——
    放在 :func:`tainted_damage_bonus` 里由 ``compute_damage`` 查询。
    """
    return KEEP


def tainted_damage_bonus(target: "Combatant") -> int:
    """``TaintedPower.ModifyDamageAdditive``：目标受伤 +``Amount``。"""
    return target.power("tainted")


def _galvanic(ctx: PowerContext, amount: int) -> str:
    """``GalvanicPower``：给玩家**所有能力牌**贴 ``Galvanized``。"""
    afflict_cards(ctx.state, "galvanized", ctx.events, card_type="power")
    return KEEP


def _possess_return(ctx: PowerContext, amount: int) -> str:
    """``PossessStrengthPower.AfterDeath``：拥有者死亡时把偷来的还回去。"""
    owed = ctx.owner.powers_debts.get("possess_strength", 0)
    if owed:
        ctx.state.player.add_power("strength", owed)
        ctx.events.append(f"{ctx.state.player.name} 取回 {owed} 点力量")
        ctx.owner.powers_debts["possess_strength"] = 0
    owed = ctx.owner.powers_debts.get("possess_speed", 0)
    if owed:
        ctx.state.player.add_power("dexterity", owed)
        ctx.events.append(f"{ctx.state.player.name} 取回 {owed} 点敏捷")
        ctx.owner.powers_debts["possess_speed"] = 0
    return KEEP


def possess_steal(holder: "Combatant", state: "CombatState",
                  pid: str, amount: int, events: list[str]) -> None:
    """玩家获得力量/敏捷时，拥有 ``possess_*`` 的敌人把这份加成**拿走**。

    ``PossessStrengthPower.AfterPowerAmountChanged`` 的条件是
    ``applier == base.Owner``（也就是"这次加成是这只怪给的"）—— 引擎里
    敌人给玩家上力量只发生在极少数招式上，这里按同一条件实现：
    由 ``_apply_one`` 在给玩家加力量/敏捷且来源是敌人时调用。
    """
    key = "strength" if pid == "possess_strength" else "dexterity"
    took = holder.power(pid)
    if took <= 0 or amount <= 0:
        return
    stolen = min(took * amount, state.player.power(key))
    if stolen <= 0:
        return
    state.player.add_power(key, -stolen)
    holder.add_power(key, stolen)
    holder.powers_debts[pid] = holder.powers_debts.get(pid, 0) + stolen
    events.append(f"{holder.name} 窃取了 {stolen} 点{key}")


def _tender(ctx: PowerContext, amount: int) -> str:
    """``TenderPower.AfterCardPlayed``：玩家每打出一张牌，拥有者 **−1 力量 −1 敏捷**。

    ⚠️ 是**临时**削弱：``AfterSideTurnEnd`` 会按"本回合打了几张"**等量还回去**。
    只实现前半段（只减不加）会让它变成一个永久削弱的超强 debuff；
    只实现后半段则什么也不发生。两边必须成对。
    """
    player = ctx.state.player
    player.add_power("strength", -1)
    player.add_power("dexterity", -1)
    played = player.power_flags.get("tender_played", 0) + 1
    player.power_flags["tender_played"] = played
    ctx.events.append(f"{player.name} 被削：力量 −1、敏捷 −1（本回合第 {played} 张）")
    return KEEP


def _tender_restore(ctx: PowerContext, amount: int) -> str:
    """``TenderPower.AfterSideTurnEnd``：按本回合出牌数把力量/敏捷**还回去**，计数清零。"""
    player = ctx.state.player
    played = player.power_flags.get("tender_played", 0)
    if played > 0:
        player.add_power("strength", played)
        player.add_power("dexterity", played)
        ctx.events.append(f"{player.name} 恢复力量/敏捷 +{played}")
        player.power_flags["tender_played"] = 0
    return KEEP


def revive(combatant: "Combatant", state: "CombatState", events: list[str],
           amount: int | None = None, source: str = "") -> None:
    """``ShouldCreatureBeRemovedFromCombatAfterDeath`` → false：**死了但留在场上**。

    真机的三个例子（``IllusionPower`` / ``ReattachPower`` / ``AdaptablePower``）都是
    同一个形状：死后不被移除、期间**打不到**（``ShouldAllowHitting`` → false）、
    到自己的回合按各自的招式回血复活。

    ``amount=None`` 表示回满（幻影的 ``REVIVE_MOVE`` 回 ``MaxHp - CurrentHp``）。
    """
    combatant.reviving = True
    combatant.revive_amount = amount
    suffix = f"（{source}）" if source else ""
    events.append(f"{combatant.name} 倒下但仍在场上，即将复活{suffix}")


def perform_revive(combatant: "Combatant", state: "CombatState",
                   events: list[str]) -> None:
    """复活中的单位轮到自己时回血并恢复（由 ``_run_enemy_turn`` 调用）。"""
    amount = getattr(combatant, "revive_amount", None)
    heal = combatant.max_hp - combatant.hp if amount is None else amount
    combatant.hp = min(combatant.max_hp, combatant.hp + max(0, heal))
    combatant.reviving = False
    combatant.revive_amount = None
    events.append(f"{combatant.name} 复活，回复 {heal} 点生命")


def _illusion(ctx: PowerContext, amount: int) -> str:
    """``IllusionPower.AfterDeath``：**自己死亡时**不被移除，下回合**回满血**复活。

    ::

        await CreatureCmd.TriggerAnim(base.Owner, "StunTrigger", 0f);
        GetInternalData<Data>().isReviving = true;
        MoveState state = new MoveState("REVIVE_MOVE", ReviveMove, new HealIntent()) { ... };
        base.Owner.Monster.SetMoveImmediate(state);

    并且 ``ShouldAllowHitting`` 返回 false —— 复活期间**打不到它**。
    另外它被施加时会自带 ``MinionPower``（幻影属于次级敌人）。
    """
    revive(ctx.owner, ctx.state, ctx.events, amount=None, source="幻影")
    return KEEP


def _reattach(ctx: PowerContext, amount: int) -> str:
    """``ReattachPower.AfterDeath``：**其他节还活着**时不死，回合结束回 ``Amount`` 点。

    ``ShouldOwnerDeathTriggerFatal`` = "其他节全死了" —— 也就是**最后一节才算真死**。
    所以虫子是"必须一次把每节都打掉"的机制；少了它，打死一节就等于打赢了。
    """
    from . import core
    others = [e for e in state_enemies(ctx.state)
              if e is not ctx.owner and e.power("reattach") > 0]
    if others and all(o.hp <= 0 and not getattr(o, "reviving", False) for o in others):
        # 其他节都真死了 → 这一节也是最后一节，允许真死
        return KEEP
    revive(ctx.owner, ctx.state, ctx.events, amount=amount, source="再附着")
    return KEEP


def _adaptable(ctx: PowerContext, amount: int) -> str:
    """``AdaptablePower.AfterDeath``：试验体倒下后进入**死亡态**再复活。

    ```csharp
    if (!wasRemovalPrevented && creature == base.Owner && creature.Monster is TestSubject testSubject)
    { GetInternalData<Data>().isReviving = true; await testSubject.TriggerDeadState(); }
    ```

    ``ShouldStopCombatFromEnding`` 为 true：它在死亡态时战斗不能结束。
    引擎把"死亡态"表达成复活（回满血）—— 试验体的三阶段形态差异属于
    出招状态机的内容，见 ``TestSubject``。
    """
    revive(ctx.owner, ctx.state, ctx.events, amount=None, source="适应")
    return KEEP


def _steam_eruption(ctx: PowerContext, amount: int) -> str:
    """``SteamEruptionPower.AfterDeath``：瀑布巨人倒地后进入**"即将爆炸"**态。

    .. code-block:: csharp

       if (!wasRemovalPrevented && creature == base.Owner)
           await ((WaterfallGiant)creature.Monster).TriggerAboutToBlowState();

    配合 ``ShouldCreatureBeRemovedFromCombatAfterDeath → false`` 与
    ``ShouldStopCombatFromEnding → true``：它没被移除、战斗也不能结束，
    下一回合走 ``ABOUT_TO_BLOW_MOVE → EXPLODE_MOVE``（自爆）。

    引擎用 ``MoveStateMachine.set_immediate`` 表达这次状态注入 ——
    少了它，巨人会在倒地瞬间直接消失，"临死反扑"整段没了。
    """
    machine = getattr(ctx.owner, "machine", None)
    if machine is not None and "ABOUT_TO_BLOW_MOVE" in getattr(machine, "states", {}):
        machine.set_immediate("ABOUT_TO_BLOW_MOVE")
    revive(ctx.owner, ctx.state, ctx.events, amount=1, source="蒸汽喷发")
    return KEEP


def _ritual(ctx: PowerContext, amount: int) -> str:
    """``RitualPower.AfterSideTurnEnd``：拥有者回合结束时获得 ``Amount`` 点力量。

    ⚠️ 真机有一个**"刚上上去就跳过第一次"**的规则（照抄，别简化）::

        public override Task AfterApplied(...)
        {
            if (base.Owner.IsEnemy) WasJustAppliedByEnemy = true;
        }
        public override async Task AfterSideTurnEnd(...)
        {
            if (participants.Contains(base.Owner))
            {
                if (WasJustAppliedByEnemy) { WasJustAppliedByEnemy = false; return; }
                await PowerCmd.Apply<StrengthPower>(..., base.Amount, ...);
            }
        }

    因为仪式是**在敌人自己回合里**上的（`IncantationMove`），而同一个回合的
    `AfterSideTurnEnd` 紧接着就会跑 —— 不跳过的话，"上仪式的那一回合"就白赚一层，
    觉醒邪教徒的第一个回合会多 1 点力量。
    """
    applied_turn = ctx.owner.power_applied_turn.get("ritual")
    if applied_turn == ctx.state.turn:
        ctx.owner.power_applied_turn.pop("ritual", None)
        ctx.events.append(f"{ctx.owner.name} 的仪式本回合刚获得，推迟触发")
        return KEEP
    ctx.owner.add_power("strength", amount)
    ctx.events.append(f"{ctx.owner.name} 的仪式生效：力量 +{amount}")
    return KEEP


def _high_voltage(ctx: PowerContext, amount: int) -> str:
    """``HighVoltagePower.AfterSideTurnEnd``：拥有者回合结束时获得 ``Amount`` 点力量。

    与 :func:`_ritual` 是同一个效果，**但没有"刚上上去跳过第一次"那条规则** ——
    高压是卡牌给的（玩家自己上），仪式是怪给自己上的。
    """
    ctx.owner.add_power("strength", amount)
    ctx.events.append(f"{ctx.owner.name} 的高压生效：力量 +{amount}")
    return KEEP


def _escape_artist(ctx: PowerContext, amount: int) -> str:
    """``EscapeArtistPower.AfterSideTurnEnd``：**纯视觉倒计时**，没有战斗效果。

    源码注释写得很直白：*"Just a visual timer for when ThievingHopper will escape."*
    递减是为了让层数与画面一致；真正的"逃跑"在别处（`ThievingHopper` 自己的招式）。
    照实实现，不硬塞一个不存在的效果。
    """
    if amount > 1:
        ctx.owner.add_power("escape_artist", -1)
    return KEEP


def _battleworn_dummy_time_limit(ctx: PowerContext, amount: int) -> str:
    """``BattlewornDummyTimeLimitPower.AfterSideTurnEnd``：倒计时归零即**逃跑**。

    ::

        if (base.Amount > 1) { await PowerCmd.Decrement(this); return; }
        if (... Encounter is BattlewornDummyEventEncounter ...) RanOutOfTime = true;
        await CreatureCmd.Escape(base.Owner);

    即 ``Amount`` 层 = 还能打几个回合；打完就脱离战斗（不是"死"，是"走了"）。
    """
    if amount > 1:
        ctx.owner.add_power("battleworn_dummy_time_limit", -1)
        ctx.events.append(f"{ctx.owner.name} 的时限剩 {amount - 1} 回合")
        return KEEP
    escape(ctx.owner, ctx.state, ctx.events)
    return REMOVE


def escape(combatant: "Combatant", state: "CombatState",
           events: list[str]) -> None:
    """``CreatureCmd.Escape``：脱离战斗。

    ⚠️ 与"死亡"**不是一回事**：逃跑的怪不算被击杀（不给遗物/金币的击杀判定，
    也不进"本回合死了几只"的统计）。引擎里用 ``hp = 0`` + 一个显式标记表达，
    战斗结算按标记区分。
    """
    combatant.hp = 0
    combatant.escaped = True
    events.append(f"{combatant.name} 逃离了战斗")


def _slippery_dec(ctx: PowerContext, amount: int) -> str:
    """``SlipperyPower.AfterDamageReceived``：受到**未格挡**伤害就减 1 层。

    配合 ``ModifyHpLostAfterOsty``（掉血压到最多 1 点）就是"滑溜"：
    前 N 次被打每次最多掉 1 点血。
    """
    if getattr(ctx, "unblocked", 0) >= 1 and amount > 0:
        ctx.owner.add_power("slippery", -1)
        ctx.events.append(f"{ctx.owner.name} 的滑溜减到 {ctx.owner.power('slippery')}")
    return KEEP


def _skittish(ctx: PowerContext, amount: int) -> str:
    """``SkittishPower.AfterAttack``：**每回合一次**，被卡牌攻击且掉了血 → 获得 N 格挡。

    三个条件同时成立才触发（源码）：

    * 这一回合还没因此拿过格挡（``HasGainedBlockThisTurn``）
    * 伤害带 ``ValueProp.Move``（**招式**伤害；中毒/反伤这类不算）
    * ``command.ModelSource is CardModel``（来源必须是**卡牌**，怪物互殴不算）
    * 且这次攻击**真的掉了血**（``UnblockedDamage != 0``）

    豁免条件少写一个都会让它变成"每被打一次就给格挡"，强度差好几倍。
    """
    if getattr(ctx, "unblocked", 0) == 0:
        return KEEP
    if not getattr(ctx, "from_card", False):
        return KEEP
    if ctx.owner.power_used_turn.get("skittish") == ctx.state.turn:
        return KEEP                            # 本回合已经拿过了
    ctx.owner.power_used_turn["skittish"] = ctx.state.turn
    ctx.owner.block += amount                  # `ValueProp.Unpowered`
    ctx.events.append(f"{ctx.owner.name} 的胆怯触发：获得 {amount} 点格挡")
    return KEEP


def _rampart(ctx: PowerContext, amount: int) -> str:
    """``RampartPower.AfterSideTurnStart``：**玩家回合开始时**，己方炮台获得 N 点格挡。

    ::

        if (side != CombatSide.Player || ...) return;
        foreach (Creature c in base.CombatState.Enemies.Where(c => c.Monster is TurretOperator))
            await CreatureCmd.GainBlock(c, base.Amount, ValueProp.Unpowered, null);

    注意是"**所有**炮台"（同场可能不止一只），且格挡是 ``Unpowered``。
    """
    from . import core
    for enemy in state_enemies(ctx.state):
        if getattr(enemy, "eid", "") == "turret_operator" and enemy.alive():
            enemy.block += amount
            ctx.events.append(f"{enemy.name} 获得 {amount} 点格挡（壁垒）")
    return KEEP


def state_enemies(state: "CombatState") -> list:
    return list(getattr(state, "enemies", ()) or ())


def _enrage(ctx: PowerContext, amount: int) -> str:
    """``EnragePower.AfterCardPlayed``：玩家打出**技能牌**时，拥有者获得 ``Amount`` 点力量。

    ::

        if (cardPlay.Card.Type == CardType.Skill)
            await PowerCmd.Apply<StrengthPower>(..., base.Owner, base.Amount, ...);

    ⚠️ 只看 ``CardType.Skill``：攻击牌与能力牌**不触发**。
    一律触发会让"愤怒"从"别打技能牌"变成"别出牌"，是另一种游戏。
    """
    if getattr(ctx, "card_type", "") != "skill":
        return KEEP
    ctx.owner.add_power("strength", amount)
    ctx.events.append(f"{ctx.owner.name} 因技能牌而愤怒：力量 +{amount}")
    return KEEP


def _crab_rage(ctx: PowerContext, amount: int) -> str:
    """``CrabRagePower.AfterDeath``：**同阵营的队友**死亡时，+6 力量、+99 格挡，然后移除自身。

    ::

        if (creature != base.Owner && creature.Side == base.Owner.Side)
        {
            await PowerCmd.Apply<StrengthPower>(..., 6, ...);
            await CreatureCmd.GainBlock(base.Owner, 99m, null);
            await PowerCmd.Remove(this);
        }

    ``Amount`` 层数**不参与**数值 —— 6 与 99 是写死的 ``DynamicVars``。
    一次性效果：触发完自己就消失。
    """
    ctx.owner.add_power("strength", 6)
    ctx.owner.block += 99
    ctx.events.append(f"{ctx.owner.name} 因同伴阵亡而暴怒：力量 +6、格挡 +99")
    return REMOVE


def _minion() -> None:
    """``MinionPower`` 没有钩子 —— 它的语义在**死亡处理**里（见 :func:`kill_secondary_teammates`）。"""


def kill_secondary_teammates(state: "CombatState", dead: "Combatant",
                             events: list[str]) -> None:
    """``CreatureCmd`` 的死亡处理：**主敌人死了、且剩下的队友全是次级敌人 → 队友一起死**。

    .. code-block:: csharp

       if (creature.Side == CombatSide.Enemy)
           if (isPrimaryEnemy && teammates.Count != 0
               && teammates.All(t => t.IsSecondaryEnemy))
               await Kill(teammates);

    其中 ``IsSecondaryEnemy`` = 身上有 ``OwnerIsSecondaryEnemy`` 的能力
    （``MinionPower`` / ``IllusionPower``）。这条规则是"打首领，小怪跟着倒"的机制；
    漏了它，带小怪的 Boss 战会变成"必须把小怪一只只清完"。
    """
    enemies = state_enemies(state)
    dead_index = enemies.index(dead) if dead in enemies else -1
    if dead_index < 0:
        return
    was_primary = dead.power("minion") <= 0 and dead.power("illusion") <= 0
    if not was_primary:
        return
    teammates = [e for e in enemies if e is not dead and e.alive()]
    if not teammates:
        return
    if not all(t.power("minion") > 0 or t.power("illusion") > 0 for t in teammates):
        return
    for teammate in teammates:
        teammate.hp = 0
        events.append(f"{teammate.name} 随首领倒下")


# ==========================================================================
# 查询型接口（引擎在关键路径上问"这个能力现在是不是在生效"）
# ==========================================================================
def blocks_draw(combatant: "Combatant") -> bool:
    """``NoDrawPower.ShouldDraw`` → ``false``：这一回合不抽牌。"""
    return combatant.power("no_draw") > 0


#: ``ShouldFlush(player)`` 返回 ``false`` 的能力：回合结束**不清空**手牌。
#: ``RetainHandPower``（保留手牌）与 ``WellLaidPlansPower``（未雨绸缪）
#: 是**两个不同的能力**，但判据完全一样（``player != Owner.Player → true``）。
HAND_FLUSH_PREVENTERS = ("retain_hand", "well_laid_plans")


def retains_hand(combatant: "Combatant") -> bool:
    """``Hook.ShouldFlush`` → ``false``：回合结束不清空手牌（任一持有者即可）。"""
    return any(combatant.power(pid) > 0 for pid in HAND_FLUSH_PREVENTERS)


#: **苦痛 → 行为的查表**。苦痛类自己不含逻辑（源码里几乎是空类），
#: 效果全在对应能力的方法里：
#:
#: ================== ==========================================================
#: 苦痛                源码里的行为
#: ================== ==========================================================
#: ``entangled``      ``TangledPower.TryModifyEnergyCostInCombat``：攻击牌费用 **+Amount**
#: ``ringing``        ``RingingPower.ShouldPlay`` → false（打不出）
#: ``smog``           ``SmoggyPower.ShouldPlay`` → false（打不出）
#: ``bound``          ``ChainsOfBindingPower``：打出后标记 ``boundCardPlayed``
#: ``hexed``          ``HexPower.TryModifyKeywordsInCombat``：获得 **虚无**（Ethereal）
#: ``tainted``        ``TaintedPower``：受伤 +Amount
#: ``galvanized``     ``GalvanizedPower``：能力牌相关
#: ================== ==========================================================
AFFLICTION_COST_POWERS: tuple[tuple[str, str], ...] = (
    # (苦痛, 施加它的能力)；能力的 Amount 就是加价
    ("entangled", "tangled"),
)

#: 被苦痛禁止打出的对应关系：``(苦痛, 能力)``。
AFFLICTION_BLOCK_POWERS: tuple[tuple[str, str], ...] = (
    ("ringing", "ringing"),
    ("smog", "smoggy"),
)

#: 苦痛带来的关键字：``(苦痛, 能力, 关键字)``。
AFFLICTION_KEYWORDS: tuple[tuple[str, str, str], ...] = (
    ("hexed", "hex", "ethereal"),
)


def affliction_cost_bonus(state: "CombatState", card) -> int:
    """苦痛给这张牌加多少费（``TangledPower.TryModifyEnergyCostInCombat``）。

    条件是"牌带 ``Entangled`` **且** 拥有者是玩家"。加价来自能力的 ``Amount``。
    """
    if card.affliction is None:
        return 0
    player = getattr(state, "player", None)
    if player is None:
        return 0
    bonus = 0
    for affliction, power in AFFLICTION_COST_POWERS:
        if card.affliction == affliction and player.power(power) > 0:
            bonus += player.power(power)
    return bonus


def affliction_blocks_play(state: "CombatState", card) -> str | None:
    """苦痛是否让这张牌打不出（``ShouldPlay`` → false）。返回原因或 ``None``。"""
    if card.affliction is None:
        return None
    player = getattr(state, "player", None)
    if player is None:
        return None
    for affliction, power in AFFLICTION_BLOCK_POWERS:
        if card.affliction == affliction and player.power(power) > 0:
            return power
    # ⭐ ``bound`` 是**有条件**的：``ChainsOfBindingPower.ShouldPlay`` 返回
    # ``!boundCardPlayed`` —— 这一回合**已经打过一张**被束缚的牌之后，
    # 其余的 Bound 牌才打不出（第一张永远能打）。
    # 条件是"本回合的状态"，所以不能塞进上面那张无条件表里。
    if (card.affliction == "bound" and player.power("chains_of_binding") > 0
            and player.power_flags.get("bound_played_this_turn")):
        return "chains_of_binding"
    return None


def affliction_keywords(state: "CombatState", card) -> set[str]:
    """苦痛带来的关键字（``HexPower.TryModifyKeywordsInCombat``：虚无）。"""
    keywords: set[str] = set()
    if card.affliction is None:
        return keywords
    player = getattr(state, "player", None)
    if player is None:
        return keywords
    for affliction, power, keyword in AFFLICTION_KEYWORDS:
        if card.affliction == affliction and player.power(power) > 0:
            keywords.add(keyword)
    return keywords


def afflict_cards(state: "CombatState", affliction: str, events: list[str],
                  card_type: str | None = None, only_unafflicted: bool = True,
                  piles: tuple[str, ...] = ("hand", "draw", "discard")) -> int:
    """给玩家**本场战斗里的牌**贴苦痛（``CardCmd.Afflict``）。

    ``card_type`` 限定卡牌类型（``tangled`` 只贴攻击牌、``smoggy`` 只贴技能牌、
    ``galvanic`` 只贴能力牌）。返回贴了多少张。
    """
    changed = 0
    for pile_name in piles:
        pile = getattr(state, pile_name, None) or []
        for card in pile:
            if only_unafflicted and card.affliction is not None:
                continue
            if card_type is not None and card.definition().card_type != card_type:
                continue
            card.affliction = affliction
            changed += 1
    if changed:
        events.append(f"{changed} 张牌被施加苦痛「{affliction}」")
    return changed


def clear_affliction(state: "CombatState", affliction: str) -> int:
    """清掉某种苦痛（能力被移除时真机会逐个 ``CardCmd.ClearAffliction``）。"""
    cleared = 0
    for pile_name in ("hand", "draw", "discard", "exhaust", "play_area"):
        for card in getattr(state, pile_name, None) or []:
            if card.affliction == affliction:
                card.affliction = None
                cleared += 1
    return cleared


def hp_loss_cap(combatant: "Combatant") -> int | None:
    """``ModifyHpLostAfterOsty`` / ``ModifyHpLostAfterOstyLate``：掉血上限。

    ``None`` = 没有上限。四个来源，语义各不相同：

    * ``intangible`` —— 掉血压到 **1**（无形）
    * ``slippery``   —— 同上，但每受一次未格挡伤害减 1 层（滑溜）
    * ``hardened_shell`` —— 每回合的**总掉血预算**：``Amount`` 减去本回合已掉的血。
      不是"每次压到 1"，而是"一回合总共最多掉这么多"，所以要用累计值。
    * ``buffer`` —— 压到 **0**（`docs/12` §2.30）。源码在 ``…AfterOstyLate`` 这个
      **晚段**（``BufferPower.cs:14``），而无形在 ``…AfterOsty``；晚段在后，
      所以**缓冲赢**：同时有缓冲和无形时，这一次掉血被完全免掉、缓冲减 1 层、
      无形不减（无形的递减是回合末的 duration）。
    """
    # ⭐ `BufferPower` 在最前面：它是晚段，结果覆盖无形/滑溜/硬壳。
    # ⚠️ 但**不能**在这里递减层数 —— 本函数是纯查询，被三处伤害路径调用
    # （其中两处在"伤害被完全格挡"时也会走到）；递减放在
    # :func:`apply_hp_loss_cap` 里，只在**真的压住了掉血**时发生。
    if combatant.power("buffer") > 0:
        return 0
    if combatant.power("intangible") > 0 or combatant.power("slippery") > 0:
        return 1
    shell = combatant.power("hardened_shell")
    if shell > 0:
        received = combatant.power_flags.get("hardened_shell_received", 0)
        return max(0, shell - received)
    return None


def apply_hp_loss_cap(target: "Combatant", hp_loss: int,
                      events: list[str]) -> int:
    """把一次掉血按 :func:`hp_loss_cap` 压上限，并处理"这次抵消**消耗了一层**"。

    ``BufferPower`` 是唯一"用一次少一层"的上限来源：

    ::

        public override async Task AfterModifyingHpLostAfterOsty()
        {
            await PowerCmd.Decrement(this);
        }

    ⚠️ 递减必须挂在这里（"真的压住了"），不能挂进 :func:`hp_loss_cap`：
    那个函数在三处伤害路径上都会调，其中有两处在掉血为 0（被完全格挡 / 掉血本来就是 0）
    的时候也会调 —— 挂错地方会让缓冲被**空放**掉。反过来，只压上限不递减
    会让缓冲变成**永久免疫**（静默变强）。

    ⚠️ 递减量是 **1 层**（``Decrement``），不是"按被免掉的伤害量"。缓冲是
    "免掉 N **次**掉血"，不是"免掉 N 点血"。
    """
    if hp_loss <= 0:
        return hp_loss
    cap = hp_loss_cap(target)
    if cap is None or hp_loss <= cap:
        return hp_loss
    if cap == 0 and target.power("buffer") > 0:
        # ``cap == 0`` 还可能来自"硬壳预算用光"，所以这里要再确认缓冲在场。
        target.add_power("buffer", -1)
        events.append(f"{target.name} 的缓冲抵消了 {hp_loss} 点掉血"
                      f"（剩 {target.power('buffer')} 层）")
    return cap


def caps_hp_loss(combatant: "Combatant") -> bool:
    """兼容旧名：有没有掉血上限。**新代码请用** :func:`hp_loss_cap`。"""
    return hp_loss_cap(combatant) is not None


# ==========================================================================
# 运行期公式（`docs/12` §2.33）
# ==========================================================================
#: 引擎**能算**的运行期公式种类。与 ``content.ENGINE_CALC_KINDS`` 是同一张表
#: 的两处表述（``tests/test_calc_formulas.py`` 钉住它们一致）——
#: 抽取器只对"认得出来的形状"写 ``calc_kind``，这里决定**引擎算不算得对**。
CALC_KINDS: frozenset[str] = frozenset({
    "own_block", "target_block", "draw_pile_count", "exhaust_count",
    "target_power", "self_power", "all_cards_tag", "card_plays_total",
    # `Synchronize` / `CompileDriver`：**不同种类**的球数（不是球总数）。
    "distinct_orb_types",
    # `Hang`：``Math.Max(下限, 目标身上这条能力的层数)`` ——
    # ``calc_arg`` 编码成 ``"<pid>:<下限>"``。
    "target_power_floor",
})


def calc_amount(state, owner, target, effect) -> int:
    """算一条**运行期公式**效果的最终量（`docs/12` §2.33 / §2.34）。

    真机算式（``CalculatedVar.WithMultiplier`` 的 docstring）::

        CalculatedVar = CalculationBase + ExtraDamage × multiplier(lambda)

    例子（都回源码核过）：

    * ``BodySlam``：``CalculationBase(0) + ExtraDamage(1) × 当前格挡`` → 伤害 = 格挡；
    * ``PerfectedStrike``：``6 + 2 × 名字里带 Strike 的牌数``；
    * ``MindBlast``：``0 + 1 × 抽牌堆张数``；
    * ``Bully``：``4 + 2 × 被打者身上的易伤层数``；
    * ``Synchronize``：``0 + 1 × **不同种类**的球数``（``group by orb.Id``）；
    * ``Hang``：``max(2, 目标身上的"绞刑"层数)`` —— 这条**没有** base/extra，
      整个量就是那个表达式（``calc_base=0, calc_extra=1``）。

    ``multiplier`` 的**形状**由抽取器判定（认不出就不生成 ``calc_kind``，
    整张卡按"量由运行期公式决定"排除），所以这里只做求值：
    遇到表里没有的 ``calc_kind`` **直接报错**，不静默算成 0。
    """
    kind = str(getattr(effect, "calc_kind", "") or "")
    arg = str(getattr(effect, "calc_arg", "") or "")
    if kind not in CALC_KINDS:
        raise ValueError(f"未实现的运行期公式 {kind!r}")
    if kind == "own_block":
        multiplier = owner.block
    elif kind == "target_block":
        multiplier = target.block if target is not None else 0
    elif kind == "draw_pile_count":
        multiplier = len(state.draw_pile)
    elif kind == "exhaust_count":
        multiplier = len(state.exhaust)
    elif kind == "target_power":
        multiplier = target.power(arg) if target is not None else 0
    elif kind == "self_power":
        # 源码写的是 ``Math.Max(0, …GetPowerAmount<X>())``：负层数当 0。
        multiplier = max(0, owner.power(arg))
    elif kind == "all_cards_tag":
        multiplier = sum(
            1 for pile in (state.hand, state.draw_pile, state.discard,
                           state.exhaust, state.play_area)
            for card in pile
            if arg in tuple(getattr(card.definition(), "tags", ()) or ()))
    elif kind == "card_plays_total":
        # `History.CardPlaysFinished.Count()`：**整场战斗**打完的牌数（不是本回合）。
        multiplier = int(getattr(state, "card_plays_finished_this_combat", 0))
    elif kind == "distinct_orb_types":
        # `Synchronize` / `CompileDriver`：``Orbs group by orb.Id`` → **种数**。
        # 用球总数会把"3 个闪电球"算成 3 而不是 1（静默变强）。
        multiplier = len(set(getattr(state, "orbs", ()) or ()))
    elif kind == "target_power_floor":
        # `Hang`：``Math.Max(2, cardPlay.Target.GetPowerAmount<HangPower>())``
        pid, _, floor = arg.partition(":")
        layers = target.power(pid) if target is not None else 0
        multiplier = max(int(floor or 0), layers)
        if pid == "hang":
            # Hang.OnPlay：新增层数还要受 999999999 上限约束；只做翻倍会溢出。
            multiplier = min(multiplier, max(0, 999999999 - layers))
    else:                                            # pragma: no cover - 上面已拦
        raise ValueError(f"未实现的运行期公式 {kind!r}")
    total = int(getattr(effect, "calc_base", 0) or 0) \
        + int(getattr(effect, "calc_extra", 0) or 0) * int(multiplier)
    return max(0, total)


#: 阻止回合开始清格挡的能力（``ShouldClearBlock`` → ``false``）。
#:
#: ⚠️ 真机 `Creature.AfterTurnStart(side)` **双方阵营都会清格挡**：
#: 玩家回合开始清玩家的、敌方回合开始清敌人的。少了敌方那一次，
#: 会给自己加格挡的怪（`guardbot` 每回合 15、`aeonglass` 33）格挡会**无限累积**，
#: 实测 4 个回合后从 15 涨到 60 —— 那只怪直接变成打不死的。
BLOCK_CLEAR_PREVENTERS = frozenset({"barricade", "burrowed", "blur"})


def blocks_block_clear(combatant: "Combatant") -> bool:
    """``Hook.ShouldClearBlock`` → 有没有哪个能力阻止清格挡。

    真机是"任一 model 返回 false 就不清"：``BarricadePower``（玩家）、
    ``BurrowedPower``（敌人）、``BlurPower``（玩家，**带期限** —— 每回合递减 1，
    见 `_blur_tick`）、``SturdyClamp``（遗物）。遗物那条走
    :func:`sts2_sim.relics`，这里只管能力。
    """
    return any(combatant.power(pid) > 0 for pid in BLOCK_CLEAR_PREVENTERS)


def type_for_amount(amount: int, kind: str, stack_type: str,
                    allow_negative: bool) -> str:
    """``PowerModel.GetTypeForAmount``：**带负值时正负性会翻转**。

    源码逻辑（照抄，别自作主张简化）：

    * ``Counter`` 且有 ``AllowNegative``、数量为负 → 算 Debuff
      （"力量被削成负数"是负面效果）
    * 不允许负值、原本是 Debuff、数量为负 → 算 Buff

    这个判定是 Artifact 是否抵消的依据，弄错会让"给敌人上负力量"
    在敌人有神器时被错误地抵消（或该抵消的没抵消）。
    """
    if stack_type == "counter" and allow_negative and amount < 0:
        return "debuff"
    if not allow_negative and kind == "debuff" and amount < 0:
        return "buff"
    return kind


def negates_debuff(owner: "Combatant", pid: str, amount: int) -> bool:
    """``ArtifactPower.TryModifyPowerAmountReceived``：神器抵消一次负面能力。

    条件三者同时成立：目标是拥有者、施加的是 **Debuff**、且该能力**可见**
    （``IsVisible``）。引擎没有"隐藏能力"的概念，一律当可见 ——
    这是个已知的近似，写在这里而不是藏着。
    """
    if owner.power("artifact") <= 0:
        return False
    if amount == 0:
        return False
    from .content import POWERS
    definition = POWERS.get(pid)
    if definition is None:
        return False
    return type_for_amount(amount, definition.kind,
                           definition.stack_type,
                           definition.allow_negative) == "debuff"


# ==========================================================================
# ⭐ 铁甲战士（Ironclad）能力 —— 全部逐一对照反编译源码
# ==========================================================================
def _owners_side_active(ctx: PowerContext) -> bool:
    """真机 ``base.CombatState.CurrentSide == base.Owner.Side`` 的引擎写法。

    ``RupturePower`` / ``InfernoPower`` 都有这个条件："**只有自己回合**掉血才算数"。
    玩家回合（含回合结束的清手牌阶段）``state.phase == "player"``，
    敌方回合 ``== "enemy"``。
    """
    player = getattr(ctx.state, "player", None)
    phase = getattr(ctx.state, "phase", "player")
    if ctx.owner is player:
        return phase == "player"
    return phase == "enemy"


def _gain_block(state: "CombatState", owner: "Combatant", amount: int,
                events: list[str], label: str) -> int:
    """走 :func:`core.gain_block`（``CreatureCmd.GainBlock`` 的唯一入口）。

    绕开它会让 ``AfterBlockGained``（板甲）与"获得格挡时"的其他机制**静默不触发**。
    """
    from . import core
    return core.gain_block(state, owner, amount, events, unpowered=True,
                           label=label)


def _demon_form(ctx: PowerContext, amount: int) -> str:
    """``DemonFormPower.AfterSideTurnStart``：拥有者所在阵营回合开始 → +``Amount`` 力量。

    ::

        if (participants.Contains(base.Owner))
            await PowerCmd.Apply<StrengthPower>(..., base.Owner, base.Amount, base.Owner, null);

    ``participants`` 是"这一方所有单位"，所以玩家这么挂就是自己回合开始加。
    引擎的 ``on_owner_turn_start`` 正是"拥有者所在阵营回合开始"，一一对应。
    """
    ctx.owner.add_power("strength", amount)
    ctx.events.append(f"{ctx.owner.name} 获得 {amount} 点力量（恶魔形态）")
    return KEEP


#: "每回合按层数再施加一条能力"的族（``AfterSideTurnStart`` 里
#: ``Apply<Xxx>(base.Owner, ±base.Amount, base.Owner, null)`` 的那些）。
#:
#: ⚠️ 与 :data:`_TEMPORARY_ATTRIBUTE_POWERS` **不是**一回事：那一族在拥有者阵营
#: 回合结束时把属性**还回去**（属性只在当回合存在）；这一族是**每回合**再施加一次、
#: **永不撤销** —— ``biased_cognition`` 每回合扣专注、``wraith_form`` 每回合扣敏捷、
#: ``neurosurge`` 每回合给自己上末日。两者不能共用一个实现。
#:
#: 同族一共 9 条（源码扫描 ``AfterSideTurnStart``/``AfterSideTurnEnd`` 里
#: ``Apply<Xxx>(base.Owner, ±base.Amount, …)`` 的全集，已由
#: ``tests/test_periodic_and_multiplier_powers.py`` 钉住），其中 6 条各有额外条件、
#: 早先已单独实现，暂不并表：``demon_form``（+力量）、``prep_time``（+活力）、
#: ``shadow_step``（+双倍伤害后**整条移除**）、``high_voltage`` / ``territorial``
#: （回合**结束** +力量）、``ritual``（同上，外加"刚贴上去跳过第一次"）。
_PERIODIC_ATTRIBUTE_POWERS: dict[str, tuple[str, int]] = {
    # pid: (每回合施加的能力 pid, Sign)
    "biased_cognition": ("focus", -1),
    "wraith_form": ("dexterity", -1),
    "neurosurge": ("doom", 1),
}


def _periodic_attribute(ctx: PowerContext, amount: int, pid: str) -> str:
    """``AfterSideTurnStart``：拥有者所在阵营回合开始，施加 ``Sign × Amount`` 层。

    ::

        if (participants.Contains(base.Owner))
            await PowerCmd.Apply<XxxPower>(choiceContext, base.Owner, Sign * base.Amount,
                                           base.Owner, null);

    ``participants.Contains(base.Owner)`` 由引擎的 ``on_owner_turn_start``
    （拥有者**所在阵营**回合开始）一一对应。

    ⚠️ 取法分两条：正向走 :func:`apply_power_to`（真机的 ``Apply`` 路径 —— 会走
    ``BeforeApplied``/``AfterApplied``，也会被神器抵消），负向走 ``add_power``
    （引擎约定：减层只有这一条路，``apply_power_to`` 对负数直接报错）。
    与 ``_demon_form`` / ``_prep_time`` 的取法一致。
    """
    attr, sign = _PERIODIC_ATTRIBUTE_POWERS[pid]
    if sign > 0:
        apply_power_to(ctx.state, ctx.owner, attr, sign * amount, ctx.events,
                       applier=ctx.owner)
    else:
        ctx.owner.add_power(attr, sign * amount)
        ctx.events.append(f"{ctx.owner.name} 失去 {amount} 点"
                          f"{_TEMPORARY_ATTRIBUTE_CN[attr]}（{pid}）")
    return KEEP


def _borrowed_time_expire(ctx: PowerContext, amount: int) -> str:
    """``BorrowedTimePower.AfterSideTurnEnd``：拥有者所在阵营回合结束**整条移除**。

    ::

        if (participants.Contains(base.Owner))
            await PowerCmd.Remove(this);

    ⚠️ 与 ``debilitate`` / ``no_block`` 那种"递减一层"不同：这里是**一次全清**。
    加价那一半在 :func:`energy_cost_reduction`（同一个 ``TryModifyEnergyCostInCombat``
    阶段）里，源码出处写在那个函数的 docstring 里。
    """
    ctx.events.append(f"{ctx.owner.name} 的借来的时间结束（打牌加价解除）")
    return REMOVE


def _the_gambit(ctx: PowerContext, amount: int) -> str:
    """``TheGambitPower.AfterDamageReceived``：掉血即**处决拥有者**并移除自己。

    ::

        if (target == base.Owner && props.IsPoweredAttack() && result.UnblockedDamage > 0)
        {
            await PowerCmd.Remove(this);
            Flash();
            await CreatureCmd.Kill(base.Owner);
        }

    ⚠️ 三个条件都不能省：

    * ``target == base.Owner`` —— 由分发保证（钩子按"被打的那个单位"派发，
      拥有者就是被打者），不是"谁持有谁就在别人挨打时触发"；
    * ``props.IsPoweredAttack()`` —— ``Unpowered`` 的掉血（中毒、自伤、反伤）
      **不触发**；引擎的 ``powered`` 实参就是这一位；
    * ``result.UnblockedDamage > 0`` —— **被完全格挡不算**。只看"被打了"
      会让一张格挡流卡组随手打出孤注一掷就自尽。

    死亡走 :func:`core.mark_dead`（``CreatureCmd.Kill`` 的唯一入口），
    这样 ``BeforeDeath`` / ``AfterDeath`` 那些钩子不会漏。
    """
    if int(getattr(ctx, "unblocked", 0) or 0) <= 0:
        return KEEP
    if not getattr(ctx, "powered", True):
        return KEEP
    from . import core                     # 延迟导入：core 反过来要导入本模块
    ctx.events.append(f"{ctx.owner.name} 的孤注一掷被触发")
    core.mark_dead(ctx.state, ctx.owner, ctx.events, cause="孤注一掷")
    return REMOVE


#: "**施加者**打出一张牌就在我身上结算一次"的两个能力（`docs/12` §2.31）。
#:
#: 两条源码**逐字同构**（只差"结算成什么"）：
#:
#: ``StranglePower`` / ``OblivionPower``（都是 ``InstanceType.InstancedPerApplier``）::
#:
#:     BeforeCardPlayed(cardPlay):
#:         if (Applier?.Player == null) return;
#:         if (cardPlay.Card.Owner != Applier.Player) return;
#:         data.amountsForPlayedCards.Add(cardPlay.Card, base.Amount);
#:     AfterCardPlayed(cardPlay):
#:         if (data.amountsForPlayedCards.Remove(cardPlay.Card, out var value))
#:             // strangle: Damage(base.Owner, value, Unblockable|Unpowered, null, null)
#:             // oblivion: Apply<DoomPower>(base.Owner, value, base.Applier, null)
#:
#: 条件里的 ``Applier`` 就是"把这条能力贴到敌人身上的玩家"，引擎里记在
#: ``Combatant.powers_applier[pid]`` —— 单人对局里只有玩家会施加它们
#: （``Strangle`` / ``Oblivion`` 两张卡都只能打敌人），所以"记一个施加者"够用，
#: 不需要真的实例化多条能力。
#:
#: ⚠️ 记的是 **BeforeCardPlayed 那一刻的层数**（``Add(card, Amount)``），
#: 不是 ``AfterCardPlayed`` 时的层数：能力在牌结算过程中被叠层时，
#: 这一次结算用的是**记录时**的值。
_APPLIER_CARD_PLAY_POWERS: frozenset[str] = frozenset({"strangle", "oblivion"})


def _applier_card_played_record(ctx: PowerContext, amount: int, pid: str) -> str:
    """``BeforeCardPlayed``：把"这张牌要按多少层结算"记在拥有者身上。

    钥匙用牌的 ``uid``（与 ``SubroutinePower`` 同一个理由：同一个身份、
    SL/快照下稳定），存在 ``power_flags`` 里。
    """
    card = getattr(ctx, "card", None)
    if card is None:
        return KEEP
    applier = ctx.owner.powers_applier.get(pid)
    if applier is None or applier is not ctx.state.player:
        return KEEP
    ctx.owner.power_flags[f"{pid}_card_{card.uid}"] = amount
    return KEEP


def _strangle_hit(ctx: PowerContext, amount: int) -> str:
    """``StranglePower.AfterCardPlayed``：对拥有者造成**不可格挡且无修正**的等量伤害。

    ⚠️ 是 ``Unblockable | Unpowered``（源码原样）：不吃格挡、也不吃力量/易伤/虚弱。
    写成普通伤害会让"勒杀"被易伤放大、被格挡吃掉 —— 两个方向都错。
    """
    card = getattr(ctx, "card", None)
    if card is None:
        return KEEP
    value = ctx.owner.power_flags.pop(f"strangle_card_{card.uid}", 0)
    if value <= 0:
        return KEEP
    from . import core
    core.deal_damage(ctx.state, None, ctx.owner, value, ctx.events,
                     unblockable=True, unpowered=True)
    return KEEP


def _oblivion_hit(ctx: PowerContext, amount: int) -> str:
    """``OblivionPower.AfterCardPlayed``：给拥有者上等量**末日**（施加者是原施加者）。"""
    card = getattr(ctx, "card", None)
    if card is None:
        return KEEP
    value = ctx.owner.power_flags.pop(f"oblivion_card_{card.uid}", 0)
    if value <= 0:
        return KEEP
    apply_power_to(ctx.state, ctx.owner, "doom", value, ctx.events,
                   applier=ctx.owner.powers_applier.get("oblivion"))
    return KEEP


def _strangle_expire(ctx: PowerContext, amount: int) -> str:
    """``StranglePower.AfterSideTurnEnd``：``participants.Contains(Owner)`` → 整条移除。"""
    return REMOVE


def _oblivion_expire(ctx: PowerContext, amount: int) -> str:
    """``OblivionPower.AfterSideTurnEnd``：条件是 ``side == CombatSide.Player``。

    ⚠️ 与同族的 ``strangle`` **不同**：它看的是**玩家阵营**结束（不是拥有者阵营），
    所以挂在 ``on_side_turn_end`` 上而不是 ``on_owner_turn_end``。
    拥有者是敌人，玩家回合结束时移除 —— 也就是"本回合打出的牌才吃末日"。
    """
    if getattr(ctx, "side", "") != "player":
        return KEEP
    return REMOVE


def _rage(ctx: PowerContext, amount: int) -> str:
    """``RagePower.AfterCardPlayed``：玩家打出**攻击牌** → 获得 ``Amount`` 点格挡。

    ::

        if (cardPlay.Card.Owner == base.Owner.Player && cardPlay.Card.Type == CardType.Attack)
            await CreatureCmd.GainBlock(base.Owner, base.Amount, ValueProp.Unpowered, null);

    三个条件逐字照抄：牌属于**拥有者本人**、类型是攻击牌、格挡是 ``Unpowered``
    （不吃敏捷、不被脆弱打折）。拥有者是敌人时 ``Owner.Player`` 为 null，
    永远不相等 —— 所以必须显式判"拥有者是玩家"，不能只看牌的类型。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    if getattr(ctx, "card_type", "") != "attack":
        return KEEP
    _gain_block(ctx.state, ctx.owner, amount, ctx.events, "暴怒")
    return KEEP


def _rage_expire(ctx: PowerContext, amount: int) -> str:
    """``RagePower.AfterSideTurnEnd``：``participants.Contains(Owner)`` → ``PowerCmd.Remove``。

    ⚠️ 层数是 ``Counter`` 却**不递减**，而是"自己这边回合结束就整个移除"。
    当成递减（每回合 -1）会让暴怒多留好几个回合。
    """
    return REMOVE


def _feel_no_pain(ctx: PowerContext, amount: int) -> str:
    """``FeelNoPainPower.AfterCardExhausted``：**自己的**牌被消耗 → 获得 ``Amount`` 点格挡。

    ::

        if (card.Owner.Creature == base.Owner)
            await CreatureCmd.GainBlock(base.Owner, base.Amount, ValueProp.Unpowered, null);

    ⚠️ **任何原因**的消耗都算：打出带 Exhaust 的牌、``Corruption`` 让技能牌消耗、
    选牌效果消耗手牌/抽牌堆的牌、虚无牌回合结束被消耗 —— 所以钩子挂在
    ``CardCmd.Exhaust`` 这个唯一出口上，而不是某一处调用点。
    引擎里所有战斗牌都属于玩家，所以"牌属于拥有者"等价于"拥有者是玩家"。
    """
    if getattr(ctx, "card", None) is None:
        return KEEP
    if ctx.owner is not ctx.state.player:
        return KEEP
    _gain_block(ctx.state, ctx.owner, amount, ctx.events, "无痛")
    return KEEP


def _dark_embrace(ctx: PowerContext, amount: int) -> str:
    """``DarkEmbracePower.AfterCardExhausted``：**非虚无**造成的消耗 → **立刻**抽 ``Amount`` 张。

    ⚠️ 虚无（Ethereal）造成的消耗**不在这里抽**，只记个数（源码里的
    ``Data.etherealCount``），留到 ``AfterSideTurnEnd`` 再一起抽。源码注释写明了原因：
    回合结束时虚无牌是在**清手牌之前**被消耗的，此刻抽到的牌会被同一次清手牌弃掉。
    少了这个延迟，黑暗拥抱在回合结束时等于什么都没抽到。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    if getattr(ctx, "caused_by_ethereal", False):
        flags = ctx.owner.power_flags
        flags["dark_embrace_ethereal"] = flags.get("dark_embrace_ethereal", 0) + 1
        return KEEP
    from .core import draw_cards
    draw_cards(ctx.state, amount, ctx.events)
    return KEEP


def _dark_embrace_flush(ctx: PowerContext, amount: int) -> str:
    """``DarkEmbracePower.AfterSideTurnEnd``：把虚无牌欠下的抽牌一次补上，计数归零。

    ``CardPileCmd.Draw(choiceContext, base.Amount * data.etherealCount, ...)`` ——
    是"层数 × 虚无牌数"，不是"每张抽一次"。
    """
    count = ctx.owner.power_flags.pop("dark_embrace_ethereal", 0)
    if count:
        from .core import draw_cards
        draw_cards(ctx.state, amount * count, ctx.events)
    return KEEP


def _juggernaut(ctx: PowerContext, amount: int) -> str:
    """``JuggernautPower.AfterBlockGained``：**自己获得格挡**后，对随机敌人造成 ``Amount`` 点伤害。

    ::

        if (!(amount <= 0m) && creature == base.Owner) {
            var target = base.Owner.Player.RunState.Rng.CombatTargets.NextItem(hittableEnemies);
            await CreatureCmd.Damage(new ThrowingPlayerChoiceContext(), target, base.Amount,
                                     ValueProp.Unpowered, base.Owner);
        }

    ⚠️ 目标走 ``Rng.CombatTargets`` 这条**独立随机流**（与"弹跳药瓶"同一条）。
    用主随机流会让同种子下的随机序列与真机分叉 —— 这类错误在单测里看不出来。
    ⚠️ 伤害是 ``Unpowered``：不吃力量、不吃易伤；但**不是** ``Unblockable``，可以被格挡。
    """
    if getattr(ctx, "gained", 0) <= 0:
        return KEEP
    hittable = [e for e in state_enemies(ctx.state) if e.alive() and e.targetable()]
    if not hittable:
        return KEEP
    from . import core
    index = ctx.state.hidden.rng.next_index("combat_targets", len(hittable))
    core.deal_damage(ctx.state, ctx.owner, hittable[index], amount, ctx.events,
                     unpowered=True)
    return KEEP


def _flame_barrier(ctx: PowerContext, amount: int) -> str:
    """``FlameBarrierPower.AfterDamageReceived``：被**有效攻击**打中 → 反伤 ``Amount`` 点。

    ::

        if (target == base.Owner && dealer != null && props.IsPoweredAttack())
            await CreatureCmd.Damage(choiceContext, dealer, base.Amount, ValueProp.Unpowered, base.Owner);

    ⚠️ 条件里**没有**"有没有掉血"：被完全格挡照样反伤（这一点与荆棘相同）。
    ⚠️ 反伤是 ``Unpowered``：不吃拥有者的力量，也不吃攻击者的易伤。
    """
    attacker = getattr(ctx, "attacker", None)
    if attacker is None:
        return KEEP
    if not getattr(ctx, "powered", True):
        return KEEP
    from . import core
    core.deal_damage(ctx.state, ctx.owner, attacker, amount, ctx.events,
                     unpowered=True)
    return KEEP


def _flame_barrier_expire(ctx: PowerContext, amount: int) -> str:
    """``FlameBarrierPower.AfterSideTurnEnd``：**不是自己这边**的回合结束就移除。

    ::

        if (base.Owner.Side != side) await PowerCmd.Remove(this);

    ``AfterSideTurnEnd`` 对**双方阵营**都会分发，所以火焰屏障覆盖的是"敌人的整个回合"，
    并在**敌方**回合结束时消失（而不是自己回合结束时）。少了这条，屏障会一直留着；
    写成"自己回合结束就移除"，敌人这一回合的反伤就全没了。
    """
    side = getattr(ctx, "side", "")
    owner_side = "player" if ctx.owner is ctx.state.player else "enemy"
    if side and side != owner_side:
        return REMOVE
    return KEEP


#: 临时属性族（``TemporaryStrengthPower`` / ``TemporaryDexterityPower`` /
#: ``TemporaryFocusPower``）的**同构表**：``pid -> (内部属性, Sign, 施加文案, 到期文案)``。
#:
#: ⭐ 三个父类**逐字同构**（``Temporary{Strength,Dexterity,Focus}Power.cs``），
#: 唯一差别是 ``InternallyAppliedPower``（``StrengthPower`` / ``DexterityPower`` /
#: ``FocusPower``）与子类的 ``OriginModel``，所以引擎用**一份实现 + 这张表**，
#: 而不是复制 21 份 —— 复制的那份迟早只有一半跟着源码改（`docs/16` §2.1）。
#:
#: ``Sign`` 是子类 ``IsPositive`` 的符号：``protected override bool IsPositive => false``
#: 的那 7 个（``CrushUnder`` / ``DarkShackles`` / ``DyingStar`` / ``EnfeeblingTouch`` /
#: ``HyperbeamFocusDown`` / ``PiercingWail`` / ``ShacklingPotion``）是 -1。
#: 文案只进事件日志，用来在对拍/轨迹里认人。
_TEMPORARY_ATTRIBUTE_POWERS: dict[str, tuple[str, int, str, str]] = {
    # pid: (内部属性, Sign, 施加文案, 到期文案)
    # ---- Wave A 之前就在表里的三条（同族的头三个入口）----
    "setup_strike": ("strength", 1, "（本回合）", "（临时力量到期）"),
    "mangle": ("strength", -1, "（本回合）", "（重击到期）"),
    "monarchs_gaze_strength_down": ("strength", -1, "（君威）", "（君威到期）"),
    # ---- 卡牌 13 张（每张只引用这一条能力）----
    "anticipate": ("dexterity", 1, "（本回合）", "（预判到期）"),
    "coordinate": ("strength", 1, "（本回合）", "（协同到期）"),
    "crush_under": ("strength", -1, "（本回合）", "（碾碎到期）"),
    "dark_shackles": ("strength", -1, "（本回合）", "（暗镣到期）"),
    "dying_star": ("strength", -1, "（本回合）", "（陨星到期）"),
    "enfeebling_touch": ("strength", -1, "（本回合）", "（衰弱之触到期）"),
    "fade": ("dexterity", 1, "（本回合）", "（隐没到期）"),
    "feeding_frenzy": ("strength", 1, "（本回合）", "（狂食到期）"),
    "focused_strike": ("focus", 1, "（本回合）", "（聚能打击到期）"),
    "hotfix": ("focus", 1, "（本回合）", "（热修到期）"),
    "hyperbeam_focus_down": ("focus", -1, "（本回合）", "（超光束到期）"),
    "piercing_wail": ("strength", -1, "（本回合）", "（刺耳尖啸到期）"),
    "synchronize": ("focus", 1, "（本回合）", "（同步到期）"),
    # ---- 药水 3 个 + 遗物 2 个（同一条父类行为，别的入口）----
    "flex_potion": ("strength", 1, "（本回合）", "（灵活药水到期）"),
    "shackling_potion": ("strength", -1, "（本回合）", "（镣铐药水到期）"),
    "speed_potion": ("dexterity", 1, "（本回合）", "（迅捷药水到期）"),
    "helical_dart": ("dexterity", 1, "（本回合）", "（螺旋镖到期）"),
    "reptile_trinket": ("strength", 1, "（本回合）", "（鳞饰到期）"),
}

#: 内部属性的中文名（只用于事件日志）。
_TEMPORARY_ATTRIBUTE_CN: dict[str, str] = {
    "strength": "力量", "dexterity": "敏捷", "focus": "专注"}


def _temporary_attribute_applied(ctx: PowerContext, amount: int, pid: str) -> str:
    """``Temporary*Power.BeforeApplied`` 的公共实现。

    ::

        await PowerCmd.Apply<XxxPower>(choiceContext, target, Sign * amount,
                                       applier, cardSource, silent: true);

    ``on_applied`` 对"拥有者身上任何能力被施加"都会分发，所以必须自己判 ``pid``。

    ⚠️ 加的是 ``ctx.applied``（**这一次**施加的层数），不是处理器收到的 ``amount``
    （那是施加后的**总层数**）。用总层数的话两张 3 层会加 3+6=9 而不是 6。

    ⚠️ **只有首次施加走这里**：真机的 ``BeforeApplied`` 只在"新造一个能力实例"那条
    路径上调用（``PowerCmd.cs:136``）；叠加走 ``ModifyAmount``（``PowerCmd.cs:119``），
    那条路径**不调** ``BeforeApplied``、只发 ``AfterPowerAmountChanged``（``:249``），
    由 :func:`_temporary_attribute_changed` 同步。两处都做就是**翻倍**。
    """
    if getattr(ctx, "pid", "") != pid:
        return KEEP
    if not getattr(ctx, "fresh", True):
        return KEEP
    attr, sign, label, _expire_label = _TEMPORARY_ATTRIBUTE_POWERS[pid]
    applied = getattr(ctx, "applied", amount)
    ctx.owner.add_power(attr, sign * applied)
    ctx.events.append(f"{ctx.owner.name} 获得 {sign * applied:+d} 点"
                      f"{_TEMPORARY_ATTRIBUTE_CN[attr]}{label}")
    return KEEP


def _temporary_attribute_changed(ctx: PowerContext, amount: int, pid: str) -> str:
    """``Temporary*Power.AfterPowerAmountChanged`` 的公共实现：层数被改时同步内部属性。

    ::

        if (!(amount == (decimal)base.Amount) && power == this)
            await PowerCmd.Apply<XxxPower>(choiceContext, base.Owner,
                                           (decimal)Sign * amount, applier, cardSource, silent: true);

    ``amount`` 是这一次的**变动量**（``PowerCmd.cs:249`` 传 ``modifiedOffset``），
    而条件里的 ``base.Amount`` 是变动**之后**的总层数 —— 所以这条成立
    ⟺ "本来就非零、又叠了一层"（``delta == new_total`` ⟺ ``old_total == 0``），
    与"首次施加走 ``BeforeApplied``"正好互补、不重叠。

    少了它，叠加的那部分只改临时能力的层数、不改内部属性（表现为"第二瓶灵活药水
    白喝"）；递减同理，撤不掉对应的属性。

    ⚠️ 取值口径：位置实参 ``amount`` 是引擎总线给的"**持有者这条能力的当前层数**"
    （= 源码的 ``base.Amount``），**变动量**在 ``ctx.amount`` 里（与 ``_shroud`` /
    ``_sleight_of_flesh`` 同一口径）。把两者用反了会让判据恒真或恒假 ——
    恒假时"叠加不加属性"，恒真时"首次施加被加两遍"。

    ``changed_owner`` 是"谁的层数变了"，必须判：总线会遍历玩家与全部敌人。
    条件照抄源码（而不是自己写成"old_total != 0"），免得以后源码变了看不出来。
    """
    if getattr(ctx, "pid", "") != pid:
        return KEEP
    if getattr(ctx, "changed_owner", None) is not ctx.owner:
        return KEEP
    delta = int(getattr(ctx, "amount", 0) or 0)
    if delta == amount:                      # 源码：``if (!(amount == base.Amount))``
        return KEEP
    attr, sign, label, _expire_label = _TEMPORARY_ATTRIBUTE_POWERS[pid]
    ctx.owner.add_power(attr, sign * delta)
    ctx.events.append(f"{ctx.owner.name} 获得 {sign * delta:+d} 点"
                      f"{_TEMPORARY_ATTRIBUTE_CN[attr]}{label}")
    return KEEP


def _temporary_attribute_expire(ctx: PowerContext, amount: int, pid: str) -> str:
    """``Temporary*Power.AfterSideTurnEnd`` 的公共实现。

    ::

        if (participants.Contains(base.Owner)) {
            Flash();
            await PowerCmd.Remove(this);
            await PowerCmd.Apply<XxxPower>(choiceContext, base.Owner, -Sign * base.Amount, base.Owner, null);
        }

    ``on_owner_turn_end`` 只在拥有者**所在阵营**回合结束时分发，等价于
    ``participants.Contains(base.Owner)``（``AfterSideTurnEnd`` 对双方阵营都分发，
    子类靠这个判断过滤）。

    ⚠️ 撤销量是**当前的层数** ``base.Amount``（处理器第二个位置实参 ``amount``）：
    两次"重击"叠到 20 层时，回合结束要还 20 点，而不是每次还 10。写成"每次还自己的
    初始值"会让力量凭空多出来。
    """
    attr, sign, _apply_label, expire_label = _TEMPORARY_ATTRIBUTE_POWERS[pid]
    ctx.owner.add_power(attr, -sign * amount)
    ctx.events.append(f"{ctx.owner.name} 获得 {-sign * amount:+d} 点"
                      f"{_TEMPORARY_ATTRIBUTE_CN[attr]}{expire_label}")
    return REMOVE


def _setup_strike(ctx: PowerContext, amount: int) -> str:
    """``SetupStrikePower``（``IsPositive`` 默认 true）：立刻 +``Amount`` 点力量。"""
    return _temporary_attribute_applied(ctx, amount, "setup_strike")


def _mangle(ctx: PowerContext, amount: int) -> str:
    """``ManglePower.IsPositive => false``：立刻 -``Amount`` 点力量。"""
    return _temporary_attribute_applied(ctx, amount, "mangle")


def _setup_strike_changed(ctx: PowerContext, amount: int) -> str:
    """``SetupStrikePower`` 层数被改（叠加/递减）时同步力量。"""
    return _temporary_attribute_changed(ctx, amount, "setup_strike")


def _mangle_changed(ctx: PowerContext, amount: int) -> str:
    """``ManglePower`` 层数被改（叠加/递减）时同步力量。"""
    return _temporary_attribute_changed(ctx, amount, "mangle")


def _setup_strike_expire(ctx: PowerContext, amount: int) -> str:
    """``SetupStrikePower`` 的回合结束：撤掉临时力量（-1 × 层数）。"""
    return _temporary_attribute_expire(ctx, amount, "setup_strike")


def _mangle_expire(ctx: PowerContext, amount: int) -> str:
    """``ManglePower`` 的回合结束：把扣掉的力量还回去（+1 × 层数）。"""
    return _temporary_attribute_expire(ctx, amount, "mangle")


def _self_damage_applied(ctx: PowerContext, pid: str) -> str:
    """``InfernoPower`` / ``CrimsonMantlePower`` 的 ``SelfDamage`` 初值 0、**每打出一张 +1**。

    源码里这个自增发生在**卡牌**侧（``CrimsonMantle.cs:28`` /
    ``Inferno.cs:23``：``PowerCmd.Apply<...>(...).IncrementSelfDamage()``），
    引擎的卡牌记录只表达得出 ``apply_power``，表达不了"顺手把能力上的变量 +1"。
    因为这两张能力只能由各自的卡施加，所以"被施加的次数"与真机的 ``SelfDamage`` 完全相等 ——
    在这里按 ``on_applied`` 计数，是同一个量而不是近似。
    """
    if getattr(ctx, "pid", "") != pid:
        return KEEP
    flags = ctx.owner.power_flags
    key = f"{pid}_self_damage"
    flags[key] = flags.get(key, 0) + 1
    return KEEP


def _crimson_mantle_applied(ctx: PowerContext, amount: int) -> str:
    """``CrimsonMantle.cs:28`` 的 ``IncrementSelfDamage()``。"""
    return _self_damage_applied(ctx, "crimson_mantle")


def _inferno_applied(ctx: PowerContext, amount: int) -> str:
    """``Inferno.cs:23`` 的 ``IncrementSelfDamage()``。"""
    return _self_damage_applied(ctx, "inferno")


def _self_damage(pid: str, ctx: PowerContext) -> int:
    """这次"回合开始掉血"掉多少：等于该能力被施加过的次数（``SelfDamage``）。"""
    return ctx.owner.power_flags.get(f"{pid}_self_damage", 0)


def _crimson_mantle_turn_start(ctx: PowerContext, amount: int) -> str:
    """``CrimsonMantlePower.AfterPlayerTurnStart``：先按 ``SelfDamage`` 掉血，再获得格挡。

    ::

        await CreatureCmd.Damage(choiceContext, base.Owner, damageVar.BaseValue, damageVar.Props, base.Owner);
        await CreatureCmd.GainBlock(base.Owner, base.Amount, ValueProp.Unpowered, null);

    ``SelfDamage`` 带 ``Unblockable | Unpowered``：不吃格挡、不受任何修正。
    格挡是 ``Unpowered``：不吃敏捷、不被脆弱打折。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    from . import core
    self_damage = _self_damage("crimson_mantle", ctx)
    if self_damage > 0:
        core.deal_damage(ctx.state, None, ctx.owner, self_damage, ctx.events,
                         unpowered=True, unblockable=True)
    _gain_block(ctx.state, ctx.owner, amount, ctx.events, "猩红披风")
    return KEEP


def _inferno_turn_start(ctx: PowerContext, amount: int) -> str:
    """``InfernoPower.AfterPlayerTurnStart``：按 ``SelfDamage`` 掉血。

    掉血本身会走 ``AfterDamageReceived``，于是**同一条能力的下半截**
    （``_inferno_retaliate``）会立刻对全体敌人开火 —— 这是炼狱的核心联动。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    from . import core
    self_damage = _self_damage("inferno", ctx)
    if self_damage > 0:
        core.deal_damage(ctx.state, None, ctx.owner, self_damage, ctx.events,
                         unpowered=True, unblockable=True)
    return KEEP


def _inferno_retaliate(ctx: PowerContext, amount: int) -> str:
    """``InfernoPower.AfterDamageReceived``：**自己回合**掉血 → 对**所有**可击中敌人造成 ``Amount`` 点伤害。

    ::

        if (target != base.Owner || result.UnblockedDamage <= 0
            || base.Owner.CombatState.CurrentSide != base.Owner.Side) return;
        await CreatureCmd.Damage(choiceContext, base.CombatState.HittableEnemies,
                                 base.Amount, ValueProp.Unpowered, base.Owner);

    三个条件一个都不能少：受伤的是自己、**真的掉了血**（被格挡不算）、
    而且是**自己回合**（`_owners_side_active`）。
    """
    if getattr(ctx, "unblocked", 0) <= 0:
        return KEEP
    if not _owners_side_active(ctx):
        return KEEP
    from . import core
    for enemy in state_enemies(ctx.state):
        if enemy.alive() and enemy.targetable():
            core.deal_damage(ctx.state, ctx.owner, enemy, amount, ctx.events,
                             unpowered=True)
    return KEEP


def _rupture_pending_key(card) -> str | None:
    """破裂按**卡牌实例**记账的键（对应真机的 ``playedCards`` 那张字典）。

    用 ``CardInstance.uid`` 而不是 ``id()``：uid 是普通整数，快照/恢复
    （``copy.deepcopy``）之后仍然指向同一张牌，而 ``id()`` 会变。
    """
    return None if card is None else f"rupture_pending_{card.uid}"


def _rupture_begin(ctx: PowerContext, amount: int) -> str:
    """``RupturePower.BeforeCardPlayed``：登记"这张牌正在被打出"，攒力量先记 0。

    ::

        if (cardPlay.Card.Owner.Creature != base.Owner) return;
        if (base.CombatState.CurrentSide != base.Owner.Side) return;
        GetInternalData<Data>().playedCards.Add(cardPlay.Card, 0);

    引擎里所有战斗牌都属于玩家，所以第一条等价于"拥有者是玩家"。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    if not _owners_side_active(ctx):
        return KEEP
    key = _rupture_pending_key(card)
    if key is not None:
        ctx.owner.power_flags[key] = 0
    return KEEP


def _rupture_damage(ctx: PowerContext, amount: int) -> str:
    """``RupturePower.AfterDamageReceived``：自己回合掉血 → 攒/发 ``Amount`` 点力量。

    ::

        if (target == base.Owner && result.UnblockedDamage > 0
            && base.CombatState.CurrentSide == base.Owner.Side) {
            if (cardSource == null || !playedCards.ContainsKey(cardSource))
                await PowerCmd.Apply<StrengthPower>(..., base.Amount, ...);
            else playedCards[cardSource] += base.Amount;
        }

    ⚠️ "是不是正在打出的那张牌造成的伤害"决定了**发力量的时机**：
    卡牌造成的自伤要等这张牌结算完才发（``AfterCardPlayed``），
    否则"先掉血、后结算伤害"的牌（例如重击前自伤的牌）会被这份力量凭空加强。
    """
    if getattr(ctx, "unblocked", 0) <= 0:
        return KEEP
    if not _owners_side_active(ctx):
        return KEEP
    card = getattr(ctx, "card", None)
    key = _rupture_pending_key(card)
    flags = ctx.owner.power_flags
    if key is not None and key in flags:
        flags[key] += amount
        return KEEP
    ctx.owner.add_power("strength", amount)
    ctx.events.append(f"{ctx.owner.name} 因受伤获得 {amount} 点力量（破裂）")
    return KEEP


def _rupture_end(ctx: PowerContext, amount: int) -> str:
    """``RupturePower.AfterCardPlayed``：把这张牌欠下的力量一次发下去。

    ::

        if (cardPlay.Card.Owner.Creature == base.Owner
            && playedCards.Remove(cardPlay.Card, out var value))
            await PowerCmd.Apply<StrengthPower>(..., value, ...);
    """
    card = getattr(ctx, "card", None)
    key = _rupture_pending_key(card)
    if key is None:
        return KEEP
    pending = ctx.owner.power_flags.pop(key, 0)
    if pending:
        ctx.owner.add_power("strength", pending)
        ctx.events.append(f"{ctx.owner.name} 因受伤获得 {pending} 点力量（破裂）")
    return KEEP


def _vicious(ctx: PowerContext, amount: int) -> str:
    """``ViciousPower.AfterPowerAmountChanged``：**自己**给别人上易伤 → 抽 ``Amount`` 张。

    ::

        if (!(amount <= 0m) && applier == base.Owner && power is VulnerablePower)
            await CardPileCmd.Draw(choiceContext, base.Amount, base.Owner.Player);

    三个条件照抄：这次施加的量 **> 0**（"减少易伤"不算）、施加者是拥有者自己、
    施加的必须是 ``VulnerablePower``。
    """
    if getattr(ctx, "pid", "") != "vulnerable":
        return KEEP
    if getattr(ctx, "applied", 0) <= 0:
        return KEEP
    if getattr(ctx, "applier", None) is not ctx.owner:
        return KEEP
    from .core import draw_cards
    draw_cards(ctx.state, amount, ctx.events)
    return KEEP


def _juggling(ctx: PowerContext, amount: int) -> str:
    """``JugglingPower.BeforeCardPlayed``：本回合第 **3** 张攻击牌 → 复制 ``Amount`` 张进手牌。

    ::

        if (cardPlay.Card.Owner == base.Owner.Player && cardPlay.Card.Type == CardType.Attack) {
            data.attacksPlayedThisTurn++;
            if (data.attacksPlayedThisTurn == 3)
                for (int i = 0; i < base.Amount; i++)
                    await CardPileCmd.AddGeneratedCardToCombat(cardPlay.Card.CreateClone(), PileType.Hand, ...);
        }

    真机的私有计数器在 ``AfterApplied`` 里用"本回合已开始的出牌记录"初始化，
    之后每张攻击牌 +1 —— 与"本回合第几张攻击牌"完全相等，所以这里直接读
    ``state.attacks_played_this_turn``（同一个量，而且是**一张牌只数一次**）。

    ⚠️ 只认 ``CardType.Attack``，且触发条件是 ``== 3``（不是"每 3 张"）。
    """
    from . import core
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    if getattr(card.definition(), "card_type", "") != "attack":
        return KEEP
    if getattr(ctx.state, "attacks_played_this_turn", 0) != 3:
        return KEEP
    for _ in range(max(0, amount)):
        # `CreateClone()` + `AddGeneratedCardToCombat(..., PileType.Hand, ...)`
        clone = core.CardInstance(cid=card.cid, upgraded=card.upgraded)
        ctx.state.hand.append(clone)
        ctx.state.added_cards += 1
    ctx.events.append(f"{ctx.owner.name} 的戏法触发：复制 {amount} 张"
                      f"{card.definition().name} 入手牌")
    return KEEP


def is_upgradable(definition) -> bool:
    """``CardModel.IsUpgradable`` 在引擎里**可判定**的形式：这张牌有升级版效果。

    引擎的升级完全由 ``CardDef.upgrade``（抽取器按 ``upgrade_delta`` 重新解出的
    那一套效果）驱动（见 ``core.step`` 里 ``definition.upgrade if card.upgraded``），
    所以"有升级版"就是"升级真的会改变这张牌"。
    """
    return bool(getattr(definition, "upgrade", ()) or ())


def _aggression(ctx: PowerContext, amount: int) -> str:
    """``AggressionPower.BeforeSideTurnStart``：从弃牌堆随机捞 ``Amount`` 张攻击牌并升级。

    ::

        var source = PileType.Discard.GetPile(...).Cards.Where(c => c.Type == CardType.Attack);
        var picked = source.ToList().UnstableShuffle(Rng.CombatCardSelection).Take(base.Amount);
        foreach (var card in picked) { await CardPileCmd.Add(card, PileType.Hand);
                                       if (card.IsUpgradable) CardCmd.Upgrade(card); }

    * 只从**弃牌堆**里捞，且只捞**攻击牌**；
    * 走 ``combat_card_selection`` 这条独立随机流做 ``UnstableShuffle``
      （源码的 ``UnstableShuffle`` 就是标准 Fisher–Yates，与
      ``RngSet.shuffle`` 的算法一致）；
    * 升级是**战斗内**的（卡面写的是"for the rest of combat"），
      所以只改战斗副本的 ``upgraded``，**不**走 ``mark_permanent_upgrade``
      那套回写。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    if getattr(ctx, "side", "") != "player":
        return KEEP
    candidates = [c for c in ctx.state.discard
                  if getattr(c.definition(), "card_type", "") == "attack"]
    if not candidates:
        return KEEP
    ctx.state.hidden.rng.shuffle(candidates, "combat_card_selection")
    for card in candidates[:max(0, amount)]:
        ctx.state.discard.remove(card)
        ctx.state.hand.append(card)
        upgraded = ""
        if not card.upgraded and is_upgradable(card.definition()):
            card.upgraded = True
            upgraded = "（已升级）"
        ctx.events.append(f"{card.definition().name} 被侵略捞回手牌{upgraded}")
    return KEEP


def _stampede(ctx: PowerContext, amount: int) -> str:
    """``StampedePower.AfterAutoPostPlayPhaseEntered``：随机自动打出 ``Amount`` 张手牌攻击牌。

    ::

        for (int i = 0; i < base.Amount; i++) {
            var items = hand.Cards.Where(c => c.Type == CardType.Attack
                                              && !c.Keywords.Contains(CardKeyword.Unplayable));
            var card = base.Owner.Player.RunState.Rng.Shuffle.NextItem(items);
            if (card != null) await CardCmd.AutoPlay(choiceContext, card, null);
        }

    ⚠️ 每打出一张就**重新查询手牌**（所以两张不会是同一张，也不会低估数量）；
    选牌走 ``shuffle`` 流（``Rng.Shuffle.NextItem``），目标是 ``AutoPlay`` 自己
    从 ``combat_targets`` 里随机 —— 两条流不能混用。
    """
    from . import core, keywords as keyword_rules
    if ctx.owner is not ctx.state.player:
        return KEEP
    for _ in range(max(0, amount)):
        options = [c for c in ctx.state.hand
                   if getattr(c.definition(), "card_type", "") == "attack"
                   and not keyword_rules.is_unplayable(c.definition())]
        if not options:
            return KEEP
        index = ctx.state.hidden.rng.next_index("shuffle", len(options))
        # ⚠️ 自动打出可能**挂在选牌上**（`autoplay_card` 返回 False）：
        # 那时效果链还没跑完，**必须立刻停手** —— 继续打下一张会让两张牌
        # 同时挂在同一个 `pending` 上（真机是 `await`，天然不会发生）。
        if not core.autoplay_card(ctx.state, options[index], ctx.events):
            return KEEP
    return KEEP


# ---- 数值修正型（无自身钩子，由核心流程查询）------------------------------
def _hellraiser(ctx: PowerContext, amount: int) -> str:
    """``HellraiserPower.AfterCardDrawnEarly``：抽到带 ``Strike`` 标签的牌 → 自动打出。

    ::

        if (card.Owner.Creature != base.Owner || !card.Tags.Contains(CardTag.Strike)) return;
        if (base.Owner.CombatState.HittableEnemies.All(c => c.HpDisplay.IsInfinite())) {
            if (data.infiniteAutoPlaysThisTurn >= 9) { flag = false; …报提示… }
            data.infiniteAutoPlaysThisTurn++;
        } else ResetInfiniteAutoPlayData();
        if (flag) { data.autoPlayingCards.Add(card); await CardCmd.AutoPlay(…); … }

    ``card.Owner.Creature != base.Owner`` —— 引擎里战斗牌都属于玩家，故等价于"拥有者是玩家"。

    ⚠️ **未实现的那一半**：``HittableEnemies.All(HpDisplay.IsInfinite())`` 那个分支
    只在"全场敌人都是**无限血量**"时成立，它存在的目的是给无限血量的练功木桩
    限流（每回合最多自动打出 9 张）。引擎的内容表里**没有任何无限血量的敌人**
    （``ENEMY_DB`` 全部有有限 HP，``tools`` 侧也以 ``hp is None`` 判"缺血量"），
    所以 ``All(...)`` 在有任何可打敌人时恒为假 —— 走到的一直是 ``else`` 那一支。
    报告里如实记为未实现项，不假装支持无限血量。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    if "Strike" not in (card.definition().tags or ()):
        return KEEP
    from . import core
    core.autoplay_card(ctx.state, card, ctx.events)
    return KEEP


def _one_two_punch_expire(ctx: PowerContext, amount: int) -> str:
    """``OneTwoPunchPower.AfterSideTurnEnd``：``participants.Contains(Owner)`` → 整个移除。"""
    return REMOVE


#: ``ModifyCardPlayCount`` 的能力：``pid → (需要的牌型 或 "*")``。
#:
#: 真机每个能力各写一个 ``ModifyCardPlayCount``，条件都是
#: "``card.Owner.Creature != Owner`` 就放行"（拥有者关系由"能力挂在谁身上"保证）
#: 再加上一条牌型判断。引擎把这两半压成一张表。
#:
#: ⚠️ **与层数无关**：这些能力永远只 ``+1``；"还能生效几张"由
#: ``AfterModifyingCardPlayCount`` 每次递减 1 表达（``EchoFormPower`` 例外，
#: 它靠"本回合已开始的牌数 < 层数"这条闸门，且不递减）。
CARD_PLAY_COUNT_POWERS: tuple[tuple[str, str], ...] = (
    ("one_two_punch", "attack"),      # 攻击牌多打一次
    ("burst", "skill"),               # 技能牌多打一次（BurstPower）
    ("signal_boost", "power"),        # 能力牌多打一次（SignalBoostPower）
    ("duplication", "*"),             # 任意牌多打一次（DuplicationPower）
    ("echo_form", "*"),               # 前 Amount 张牌多打一次（EchoFormPower）
)

#: ``AfterModifyingCardPlayCount`` → ``PowerCmd.Decrement`` 的能力
#: （**只**在真的改过次数时递减，否则打两张不相干的牌就把层数耗光了）。
CARD_PLAY_COUNT_DECREMENT: frozenset[str] = frozenset({
    "one_two_punch", "burst", "signal_boost", "duplication"})

#: 回合结束时**整个移除**的能力（``AfterSideTurnEnd → PowerCmd.Remove``）。
CARD_PLAY_COUNT_EXPIRE_AT_TURN_END: frozenset[str] = frozenset({"burst", "duplication"})


def card_play_count(state: "CombatState", card) -> int:
    """``Hook.ModifyCardPlayCount``：这张牌要结算几次（``GeneratePlayCount``）。

    ::

        int playCount = GetEnchantedReplayCount() + 1;
        playCount = Hook.ModifyCardPlayCount(combatState, this, playCount, target, out models);
        await Hook.AfterModifyingCardPlayCount(combatState, this, models);

    名单与判据见 :data:`CARD_PLAY_COUNT_POWERS`。``EchoFormPower`` 额外有一条闸门：

    ::

        int num = History.CardPlaysStarted.Count(e => e.Actor == Owner
                    && e.CardPlay.IsFirstInSeries && e.HappenedThisTurn(CombatState));
        if (num >= base.Amount) return playCount;

    —— 也就是"每回合**前 Amount 张**牌打两次"。计数器在引擎里是
    ``state.cards_played_started_this_turn``，且**在算完次数之后**才 +1
    （真机 ``CardPlayStarted`` 是在 playCount 算完、进循环时登记的）。
    """
    player = getattr(state, "player", None)
    if player is None:
        return 1
    count = 1
    card_type = getattr(card.definition(), "card_type", "")
    for pid, required in CARD_PLAY_COUNT_POWERS:
        amount = player.power(pid)
        if amount <= 0:
            continue
        if required != "*" and card_type != required:
            continue
        if pid == "echo_form":
            started = getattr(state, "cards_played_started_this_turn", 0)
            if started >= amount:
                continue
        count += 1
    return count


def on_card_play_count_modified(state: "CombatState", card) -> None:
    """``Hook.AfterModifyingCardPlayCount``：只通知**改过次数**的那些模型。

    真机遍历的是 ``modifyingModels``（``Hook.cs`` 里 ``if (modifiers.Contains(modifier))``），
    所以"这次没被加倍的牌"**不会**扣层数 —— 否则打两张无关的牌就把
    "下一次多打一次"白白耗光。名单见 :data:`CARD_PLAY_COUNT_DECREMENT`。
    """
    player = getattr(state, "player", None)
    if player is None:
        return
    card_type = getattr(card.definition(), "card_type", "")
    for pid, required in CARD_PLAY_COUNT_POWERS:
        if pid not in CARD_PLAY_COUNT_DECREMENT:
            continue
        amount = player.power(pid)
        if amount <= 0:
            continue
        if required != "*" and card_type != required:
            continue
        player.add_power(pid, -1)


#: `GuardedPower` / `ColossusPower` 的 ``DamageDecrease`` 动态变量字面量：
#: 两处都是 ``new DynamicVar("DamageDecrease", 0.5m)``
#: （`GuardedPower.cs:30`、`ColossusPower.cs:23`，已逐条回源码核对）。
GUARDED_DAMAGE_DECREASE = 0.5
COLOSSUS_DAMAGE_DECREASE = 0.5

#: ``ModifyHandDraw`` 的能力表：``pid → 修正方式``。
#:
#: 真机每个能力各重写一次 ``ModifyHandDraw``，条件都是
#: ``if (player != base.Owner.Player) return count;``（拥有者关系由挂载对象保证），
#: 区别只在**怎么改这个累计值**：
#:
#: * ``ClarityPower``：``count + 1m``（**写死 1**，与层数无关）
#: * ``DemesnePower`` / ``MachineLearningPower`` / ``ToolsOfTheTradePower`` /
#:   ``TyrannyPower``：``count + base.Amount``
#: * ``MindRotPower``（``MindRotPower.cs:20``）：``Math.Max(0m, count - base.Amount)``
#:   —— **带截零**，所以它不能当成"负数加数"混进求和：
#:   基础 5、层数 9、同时有 ``clarity`` 时，谁是链上第一个决定结果 ——
#:   ``clarity`` 先 → ``max(0, 5+1-9) = 0``；``mind_rot`` 先 → ``max(0, 5-9)+1 = 1``。
#:   两个数不同，所以**顺序也必须按真机来**（见 :func:`modify_hand_draw`）。
#:
#: 顺序依据 ``CombatState.IterateHookListeners``（``CombatState.cs:411``）：
#: 逐个 creature 先加**它自己的能力**（按 ``Creature.Powers`` 列表 = 施加顺序），
#: 再（若是玩家）加**它的遗物**。引擎的遗物侧是按守卫聚合求和
#: （``relics.hand_draw_bonus``），现有遗物全是加法、可交换，聚合与逐个等价 ——
#: 但**必须排在能力之后**，源码里遗物就在能力之后遍历。
HAND_DRAW_MODIFIERS: dict[str, str] = {
    "clarity": "fixed_one",
    "demesne": "amount",
    "machine_learning": "amount",
    "tools_of_the_trade": "amount",
    "tyranny": "amount",
    "mind_rot": "subtract_clamped",
}


def _apply_hand_draw_modifier(kind: str, count: int, amount: int) -> int:
    """按源码逐条施加一个 ``ModifyHandDraw`` 修正。"""
    if kind == "fixed_one":
        return count + 1
    if kind == "amount":
        return count + amount
    if kind == "subtract_clamped":
        # MindRotPower.cs:20 `return Math.Max(0m, count - (decimal)base.Amount);`
        return max(0, count - amount)
    raise KeyError(f"未声明的抽牌修正方式 {kind!r}")


def modify_hand_draw(state: "CombatState", count: int) -> int:
    """``Hook.ModifyHandDraw``：从基础张数出发，按源码顺序**链式**修正。

    真机（``CombatManager.cs:905``）::

        decimal handDraw = Hook.ModifyHandDraw(state, player, 5m, out modifiers);

    而 ``Hook.ModifyHandDraw`` 是"遍历全部 listener，把上一个人的返回值喂给下一个人"，
    listener 顺序 = ``Creature.Powers`` 列表 = **能力被施加的顺序**。所以这里按
    ``player.powers`` 的插入顺序遍历（dict 保序，``add_power`` 是唯一入口），
    **不是**按 :data:`HAND_DRAW_MODIFIERS` 的声明顺序 —— 那会让
    ``clarity`` 与 ``mind_rot`` 的先后与真机相反，抽到的张数直接差 1。
    敌人的同名能力都带 ``player != Owner.Player`` 的早退，对玩家抽牌没有影响。

    ⚠️ 不要退回"把修正量求和再加上去"：``mind_rot`` 在链中间带截零，
    求和会把截零丢掉（见 :data:`HAND_DRAW_MODIFIERS` 的注释）。
    药水 / 卡牌的额外抽牌走 ``core.draw_cards``，**不经过这里**（源码也不经过）。
    """
    player = getattr(state, "player", None)
    if player is None:
        return count
    for pid, amount in list(player.powers.items()):
        kind = HAND_DRAW_MODIFIERS.get(pid)
        if kind is None or amount <= 0:
            continue
        count = _apply_hand_draw_modifier(kind, count, amount)
    return count


def hand_draw_bonus(state: "CombatState") -> int:
    """**加法型** ``ModifyHandDraw`` 能力的净和（诊断 / 回归用）。

    ⚠️ 它**不是**实际抽牌张数：``mind_rot`` 那种带 ``Math.Max(0, …)`` 的减益
    不在这个和里（算进去也会因截零失真）。实际抽牌张数一律走
    :func:`modify_hand_draw`，``core.start_player_turn`` 用的是那一个。
    """
    player = getattr(state, "player", None)
    if player is None:
        return 0
    total = 0
    for pid, amount in list(player.powers.items()):
        kind = HAND_DRAW_MODIFIERS.get(pid)
        if kind is None or amount <= 0 or kind == "subtract_clamped":
            continue
        total += 1 if kind == "fixed_one" else amount
    return total


#: ``ModifyCardPlayResultLocation`` 的能力：``pid → 判据``。
#: 默认落点是**弃牌堆**；这些能力把它改成别处（目前两项都改成**抽牌堆顶**）。
#: 条件的另一半（``card.Owner.Creature == base.Owner``）由调用路径保证：
#: 只有玩家在打牌。
CARD_PLAY_RESULT_POWERS: tuple[tuple[str, str], ...] = (
    ("rebound", "any"),            # 任意牌：原本进弃牌堆 → 抽牌堆顶，然后递减 1 层
    ("nostalgia", "attack_skill"),  # 攻击/技能牌，且本回合这类牌数 < 层数
)


def card_play_result_location(state: "CombatState", card,
                              default: str = "discard") -> tuple[str, tuple[str, ...]]:
    """``Hook.ModifyCardPlayResultLocation``：打出的牌最终落到哪个牌堆。

    返回 ``(落点, 改写过落点的能力)``；``default`` 是引擎算出的默认落点
    （弃牌堆；消耗牌与腐败的技能牌在调用点就已走消耗，不会进这里）。

    ::

        // ReboundPower：原本进弃牌堆 → 改到抽牌堆**顶**
        if (card.Owner.Creature != base.Owner || location.pileType != PileType.Discard) return location;
        location.pileType = PileType.Draw; location.position = CardPilePosition.Top;
        // NostalgiaPower：再加两条 —— 牌型 ∈ {Attack, Skill}、本回合这类牌数 < 层数
    """
    player = getattr(state, "player", None)
    if player is None or default != "discard":
        return default, ()
    definition = card.definition()
    card_type = getattr(definition, "card_type", "")
    pile = default
    modifiers: list[str] = []
    for pid, requirement in CARD_PLAY_RESULT_POWERS:
        amount = player.power(pid)
        if amount <= 0:
            continue
        if requirement == "attack_skill":
            if card_type not in ("attack", "skill"):
                continue
            # 源码数的是"本回合已开始的**攻击/技能**牌数"（`CardPlaysStarted` + 牌型过滤）
            started = getattr(state, "attack_skill_plays_started_this_turn", 0)
            if started >= amount:
                continue
        pile = "draw"
        modifiers.append(pid)
    # ⭐ `FeralPower.ModifyCardPlayResultLocation`：**0 费打出的攻击牌** → **回手牌**。
    # 四条判据照抄源码：拥有者是玩家（由调用路径保证）、牌型是攻击、
    # **这次出牌实际花了 0 能量**（``resources.EnergyValue <= 0``）、
    # 还没用完本回合的额度（``zeroCostAttacksPlayed < Amount``）。
    # （``card.IsDupe`` 那一条引擎没有"复制品"概念，不适用。）
    feral = player.power("feral")
    if (feral > 0 and card_type == "attack"
            and getattr(state, "current_play_energy_spent", -1) == 0
            and player.power_flags.get("feral_returned", 0) < feral):
        pile = "hand"
        modifiers.append("feral")
    return pile, tuple(modifiers)


def on_card_play_result_location(state: "CombatState", card, pile: str,
                                 modifiers: tuple[str, ...] = ()) -> None:
    """``Hook.AfterModifyingCardPlayResultLocation``：只通知**真的改过落点**的能力。

    ``ReboundPower`` 靠它递减 1 层 —— 漏掉"只通知改过的"会让弹回在
    "这次没生效"时也掉层数（与 `AfterModifyingCardPlayCount` 同一个坑）。
    """
    player = getattr(state, "player", None)
    if player is None:
        return
    for pid in modifiers:
        if pid == "rebound" and pile == "draw":
            player.add_power("rebound", -1)
        elif pid == "feral":
            # `FeralPower.AfterModifyingCardPlayResultLocation`：额度 +1
            # （只有**真的改过落点**的这次才算）。
            player.power_flags["feral_returned"] = \
                player.power_flags.get("feral_returned", 0) + 1


#: ``TryModifyEnergyCostInCombat``（**非 Late 段**）：把**能力牌**的费用**减** ``Amount``。
#: 源码：``if (card.Type != Power || originalCost <= 0) return false;
#:        modifiedCost = max(0, originalCost - base.Amount);``（``CuriousPower.cs``）
COST_REDUCTION_POWERS: tuple[tuple[str, str], ...] = (
    ("curious", "power"),
)

#: ``TryModifyEnergyCostInCombatLate``（**Late 段**）：把符合条件的牌费用**清零**。
#: 每一项 = ``(pid, 条件)``；条件是牌型 / ``ethereal``（按关键字）/ ``void_form``
#: （按"本回合已打完的牌数 < 层数"）。条件的另一半
#: （``card.Owner.Creature == base.Owner`` 与 ``card.Pile ∈ {Hand, Play}``）
#: 由调用路径保证：``core.play_cost`` 只在"玩家正在打算出的牌"上调用。
COST_ZERO_POWERS: tuple[tuple[str, str], ...] = (
    ("free_attack", "attack"),
    ("free_power", "power"),
    ("free_skill", "skill"),
    ("veilpiercer", "ethereal"),
    ("corruption", "skill"),
    ("void_form", "void_form"),
)


def energy_cost_reduction(state: "CombatState", card, cost: int) -> int:
    """``Hook.TryModifyEnergyCostInCombat``（非 Late 段）：按表加价 / 减费，**下限 0**。

    ⚠️ **加价与减费是同一个阶段**（真机里都是 ``TryModifyEnergyCostInCombat``，
    只是不同模型各自返回修改后的费用）：

    * ``BorrowedTimePower``：``if (card.Owner.Creature != base.Owner) return false;
      modifiedCost = originalCost + (decimal)base.Amount; return true;``
      —— 拥有者的**所有牌**（不判牌型）都 +``Amount``。
    * ``CuriousPower``：能力牌 ``-Amount``，且自带 ``max(0, …)`` 下限。

    引擎按"先加价、再减费"应用（与表里的顺序一致）。两者在单人局里不会同时出现在
    玩家身上（``curious`` 是怪物侧的能力，而这里只看 ``state.player`` 的能力），
    所以这个顺序不影响任何现有内容；写清楚是为了以后加新条目时不必重新推导。
    """
    player = getattr(state, "player", None)
    if player is None:
        return cost
    card_type = getattr(card.definition(), "card_type", "")
    extra = player.power("borrowed_time")
    if extra > 0:
        cost += extra
    for pid, required in COST_REDUCTION_POWERS:
        amount = player.power(pid)
        if amount <= 0 or cost <= 0:
            continue
        if required != "*" and card_type != required:
            continue
        cost = max(0, cost - amount)
    return cost


def void_form_applies(state: "CombatState", card) -> bool:
    """``VoidFormPower.ShouldSkip(card)``：本回合**已打完**的牌数 < 层数 → 这张免费。

    它同时覆写了 ``TryModifyEnergyCostInCombatLate``（清能量费）与
    ``TryModifyStarCost``（**清星费**），判据完全一样 —— 所以两边共用这一个函数，
    免得出现"能量免费了但还要花星"这种半吊子。
    """
    player = getattr(state, "player", None)
    if player is None:
        return False
    amount = player.power("void_form")
    if amount <= 0:
        return False
    # 计数在 `AfterCardPlayed`（非自动打出、且是 `IsLastInSeries`）里 +1，
    # 引擎用 `cards_played_started_this_turn`（每张牌只算一次）对齐。
    return getattr(state, "cards_played_started_this_turn", 0) < amount


def energy_cost_zeroed(state: "CombatState", card) -> bool:
    """``Hook.TryModifyEnergyCostInCombatLate``（Late 段）：是否有一项把费用清零。

    ⚠️ **阶段顺序**：Late 段跑在非 Late 段**之后**，所以一旦这里成立，
    前面加过/减过的费用一并作废（"腐败 + 纠缠 = 免费"就是这个顺序的结果）。
    """
    player = getattr(state, "player", None)
    if player is None:
        return False
    definition = card.definition()
    card_type = getattr(definition, "card_type", "")
    keywords = tuple(getattr(definition, "keywords", ()) or ())
    for pid, required in COST_ZERO_POWERS:
        amount = player.power(pid)
        if amount <= 0:
            continue
        if required == "void_form":
            # `VoidFormPower`：本回合**已打完**的牌数 < 层数时免费
            # （与清星费共用同一个判据，见 `void_form_applies`）。
            if void_form_applies(state, card):
                return True
            continue
        if required == "ethereal":
            if "Ethereal" in keywords:
                return True
            continue
        if card_type == required:
            return True
    return False


def _card_play_count_expire(ctx: PowerContext, amount: int) -> str:
    """``BurstPower`` / ``DuplicationPower`` 的 ``AfterSideTurnEnd``：整个移除。

    ::

        if (participants.Contains(base.Owner)) await PowerCmd.Remove(this);
    """
    if _side_of(ctx) != _owner_side(ctx):
        return KEEP
    return REMOVE


def _unmovable_keys(flags: dict[str, int], turn: int) -> list[str]:
    return [k for k in flags
            if k.startswith("unmovable_block_") and k.startswith(f"unmovable_block_{turn}_")]


def block_multiplier(state: "CombatState", owner: "Combatant", card) -> Decimal:
    """``Hook.ModifyBlock`` 的乘法阶段里，能力贡献的倍率（目前只有 ``UnmovablePower``）。

    ::

        ModifyBlockMultiplicative(target, block, props, cardSource, cardPlay):
            if (target.IsMonster) return 1m;                       // 怪物的格挡不翻倍
            if (!props.IsCardOrMonsterMove()) return 1m;           // 非卡牌/招式来源不翻倍
            if (cardSource != null && cardSource.Owner.Creature != base.Owner) return 1m;
            int num = History.BlockGained.Count(e => e.HappenedThisTurn(combatState)
                        && e.CardPlay != null && e.CardPlay.Player.Creature == base.Owner
                        && e.Props.IsCardOrMonsterMove() && e.CardPlay != cardPlay);
            return num >= base.Amount ? 1m : 2m;

    ⚠️ 三个容易漏的点，全都照抄了：

    * ``num`` 数的是**本回合已经拿过格挡的"别的出牌"** —— 同一张牌自己拿两次只算一次
      （``e.CardPlay != cardPlay``），所以多段格挡的牌两次都翻倍；
    * 只有**卡牌来源**（``card is not None``）才参与：能力 / 药水 / 充能球给的格挡
      在真机里 ``Props`` 不含 ``Move``，不翻倍（引擎里它们都是 ``unpowered``，
      在 ``gain_block`` 那一侧就已经被排除了）；
    * 计数按**回合**隔离（``HappenedThisTurn``）—— 引擎用回合号做键前缀表达，
      不需要额外的"回合开始清零"钩子。

    ⭐ 还有一条**乘法**（`docs/12` §2.29）：
    ``NoBlockPower.ModifyBlockMultiplicative`` —— 拥有者**由卡牌**获得的格挡 ×0：

    ::

        if (target != base.Owner) return 1m;
        if (props.HasFlag(ValueProp.Unpowered)) return 1m;
        if (cardSource == null) return 1m;
        return 0m;

    三条条件在引擎里的落点：``target != base.Owner`` 由调用侧保证
    （``gain_block`` 只算**获得格挡的那个人**的能力）；``Unpowered`` 走的是
    ``gain_block(unpowered=True)`` 那条**不调本函数**的分支；``cardSource == null``
    就是这里的 ``card is None``。所以剩下的条件只有 "``card is not None``"。
    与 ``unmovable`` 同时存在时乘积仍然是 0（0 × 2），乘法阶段按顺序相乘。
    """
    factor = Decimal(1)
    amount = owner.power("unmovable")
    if amount > 0 and card is not None and getattr(owner, "eid", None) is None:
        flags = owner.power_flags
        turn = getattr(state, "turn", 0)
        current = f"unmovable_block_{turn}_{card.uid}"
        num = sum(1 for key in _unmovable_keys(flags, turn) if key != current)
        factor = Decimal(2) if num < amount else Decimal(1)
    if owner.power("no_block") > 0 and card is not None:
        return Decimal(0)
    return factor


def block_gained(state: "CombatState", owner: "Combatant", card) -> None:
    """``CombatManager.History.BlockGained``：记下"这张牌本回合拿过格挡"。

    只有**卡牌来源**要记（真机的计数条件里 ``e.CardPlay != null``），
    并且顺手把**旧回合**的键清掉 —— 键里带回合号，
    所以"本回合第几次"不需要额外的清零时机。
    """
    if card is None or owner.power("unmovable") <= 0:
        return
    if getattr(owner, "eid", None) is not None:
        return
    flags = owner.power_flags
    turn = getattr(state, "turn", 0)
    for key in [k for k in flags if k.startswith("unmovable_block_")
                and k not in _unmovable_keys(flags, turn)]:
        flags.pop(key, None)
    flags[f"unmovable_block_{turn}_{card.uid}"] = 1


def corruption_applies_to(state: "CombatState", card) -> bool:
    """``CorruptionPower``：拥有者的**技能牌** —— 费用归零、打出后消耗。

    ::

        TryModifyEnergyCostInCombatLate:   if (card.Owner.Creature != base.Owner || card.Type != CardType.Skill) return false; modifiedCost = 0;
        ModifyCardPlayResultLocation:      if (card.Owner.Creature != base.Owner || card.Type != CardType.Skill) return location; location.pileType = PileType.Exhaust;

    两个钩子的条件是**同一对**（牌属于拥有者 + 是技能牌），所以合成一个查询。
    """
    player = getattr(state, "player", None)
    if player is None or player.power("corruption") <= 0:
        return False
    return getattr(card.definition(), "card_type", "") == "skill"


#: ``ModifyMaxEnergy`` 的能力：``(pid, 符号)``。
#:
#: 真机 ``Hook.ModifyMaxEnergy`` 把 ``amount`` 依次喂给每个 listener，
#: 每个能力各重写一次、条件都是 ``if (player != base.Owner.Player) return amount;``
#: （拥有者关系由挂载对象保证）：
#:
#: * ``PyrePower``：``MaxEnergy => Hook.ModifyMaxEnergy(...)``（``PlayerCombatState.cs:101``；
#:   能量重置就是 ``Energy = MaxEnergy``，``PlayerCombatState.ResetEnergy``）。
#: * ``FriendshipPower``：``return amount + (decimal)base.Amount;``
#: * ``WasteAwayPower``（``WasteAwayPower.cs:22``）：``return amount - (decimal)base.Amount;``
#:
#: ⚠️ **这里不做 ``max(0, …)``**：``PlayerCombatState.MaxEnergy`` 只是
#: ``(int)Hook.ModifyMaxEnergy(...)``，``ResetEnergy`` 是 ``Energy = MaxEnergy``，
#: 而 ``Energy`` 的 setter 没有截零（只有 ``GainEnergy`` / ``LoseEnergy`` 才 clamp，
#: ``PlayerCombatState.cs:172-188``）。凭直觉补一个截零，会让"能量上限被压到负数"
#: 这个真机行为在模拟器里消失。整条链是纯加减法（线性），求和与逐个链式等价。
MAX_ENERGY_POWERS: tuple[tuple[str, int], ...] = (
    ("pyre", 1),
    ("friendship", 1),
    ("waste_away", -1),
)


def max_energy_bonus(state: "CombatState") -> int:
    """``Hook.ModifyMaxEnergy``：能力对**每回合能量上限**的净修正（可为负）。

    名单见 :data:`MAX_ENERGY_POWERS`。调用点是 ``start_player_turn`` 的能量重置
    （与遗物侧 ``relic_rules.energy_bonus`` 相加）。施加 ``waste_away`` 时
    **不**改变当前能量，只影响下一次重置 —— 源码里 ``PowerCmd.Apply`` 与
    ``MaxEnergy`` 是两条路径，Apply 不会顺手扣当前能量。
    """
    player = getattr(state, "player", None)
    if player is None:
        return 0
    return sum(sign * player.power(pid) for pid, sign in MAX_ENERGY_POWERS)


def _danse_macabre(ctx: PowerContext, amount: int) -> str:
    """``DanseMacabrePower.BeforeCardPlayed``：打出的牌**实际费用 ≥ 2** → +``Amount`` 格挡。

    ::

        if (cardPlay.Card.Owner.Creature == base.Owner
            && cardPlay.Card.EnergyCost.GetResolved() >= base.DynamicVars.Energy.IntValue)
            await CreatureCmd.GainBlock(base.Owner, base.Amount, ValueProp.Unpowered, null);

    ⚠️ 两个细节不能省：

    * 闸门是 ``EnergyVar(2)``（源码 ``CanonicalVars``），**不是**层数；
    * 比的是 ``GetResolved()``（**修正后**的费用），所以腐败把技能牌清零之后
      就不再触发 —— 用卡面费用会在腐败下**多给**格挡。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    from . import core                     # 延迟导入：core 反过来要导入本模块
    if core.play_cost(ctx.state, card) < 2:
        return KEEP
    _gain_block(ctx.state, ctx.owner, amount, ctx.events, "死亡之舞")
    return KEEP


# ==========================================================================
# ⭐ Wave A：按"被卡引用"补齐的能力（每条都注明反编译源码出处）
#
# 这一节与 `core.py` 的钩子总线配合：钩子位在 `hooks.HOOKS` 里声明、
# 由 `core` 的真实流程触发，处理器只管"收到事件之后做什么"。
# ==========================================================================
def apply_power_to(state: "CombatState", target: "Combatant", pid: str,
                   amount: int, events: list[str],
                   applier: "Combatant | None" = None) -> bool:
    """``PowerCmd.Apply<T>(choiceContext, target, amount, applier, cardSource)`` 的引擎入口。

    **能力/遗物内部**要给某个单位上能力时走这里，而不是直接
    ``target.add_power(...)`` —— 直接加会漏掉真机同一条路径上的两件事：

    1. ``ArtifactPower.TryModifyPowerAmountReceived``：负面能力被神器**整条抵消**
       （且神器减 1 层）。少这一步，带神器的怪会被随便上满 debuff。
    2. ``Hook.AfterApplied`` / ``Hook.AfterPowerAmountChanged``：能力刚被施加时的
       通知。``TangledPower`` 这类"贴上瞬间生效"的能力靠它，``ViciousPower``
       靠后者判"是谁上的"。

    与 ``core._apply_one`` 的 ``apply_power`` 分支**语义完全一致**，区别只是
    调用方（那边是卡牌算子，这里是能力/遗物）。返回是否真的加上了。

    ⚠️ **负数会直接报错，不会静默什么都不做**。真机里"减层"是另一条路径
    （``PowerCmd.Decrement`` / 带负值的 ``Apply`` 配 ``silent: true``），
    而这里早先是 ``if amount <= 0: return False`` —— 于是"想来减层"的调用
    会**悄无声息地无效**（写这个函数的当下就踩到了：独白回合结束要给
    ``-StrengthApplied`` 力量，用本函数撤不掉，测试才发现）。
    减层请用 ``target.add_power(pid, -n)``。

    ⚠️ **施加**走这里、**改层数**走 ``add_power``：只有本函数会分发
    ``AfterApplied`` / ``AfterPowerAmountChanged`` 里的"施加"语义。
    规则里需要"给谁上一条能力"时要用本函数（``add_power`` 会静默跳过
    ``AfterApplied``，而 `tangled` / `hex` 这些能力正是在那里生效的）。
    """
    if amount < 0:
        raise ValueError(
            f"apply_power_to 不接受负数（{pid} {amount}）：减层请用 add_power(pid, -n)"
            "（真机的减层走 PowerCmd.Decrement / silent 路径，不走施加路径）")
    if amount == 0 or not target.alive():
        return False
    if negates_debuff(target, pid, amount):
        target.add_power("artifact", -1)
        events.append(f"{target.name} 的神器抵消了 {pid}"
                      f"（剩 {target.power('artifact')}）")
        return False
    # ``fresh``：这条能力原本不存在（真机 ``PowerCmd.Apply`` 的"新实例"路径）。
    # 叠加走 ``ModifyAmount``，那条路径不发 ``AfterApplied`` —— 见 ``on_applied``。
    fresh = target.power(pid) == 0
    target.add_power(pid, amount, applier=applier)
    target.power_applied_turn[pid] = state.turn
    on_applied(target, state, events, pid, amount, fresh=fresh)
    on_power_applied(state, events, pid, applier, amount)
    events.append(f"{target.name} 获得 {pid} {amount:+d}")
    return True


def hittable_enemies(state: "CombatState") -> list:
    """``CombatState.HittableEnemies``：**打得到**的敌人（复活中的打不到）。

    引擎里 ``living_enemies()`` 把"复活中"的单位也算活着（战斗不能在它复活前结束），
    而真机的 ``HittableEnemies`` 用 ``IsHittable`` 把它们排除掉。两个集合**不等价**，
    直接用活着的那份会让"随机打一个敌人"在复活窗口里选到打不到的目标。
    """
    return [e for e in state.enemies
            if e.alive() and (not hasattr(e, "targetable") or e.targetable())]


# ---- AfterEnergyReset -----------------------------------------------------
def _genesis(ctx: PowerContext, amount: int) -> str:
    """``GenesisPower.AfterEnergyReset``：能量重置后获得 ``Amount`` 颗星。"""
    if ctx.owner is not ctx.state.player:
        return KEEP
    from . import core
    core.gain_stars(ctx.state, amount, ctx.events)
    return KEEP


def _radiance(ctx: PowerContext, amount: int) -> str:
    """``RadiancePower.AfterEnergyReset``：能量 +``EnergyVar(1)``，然后递减 1 层。

    ::

        await PlayerCmd.GainEnergy(base.DynamicVars.Energy.IntValue, player);
        await PowerCmd.Decrement(this);

    ⚠️ 加的是**动态变量的字面量 1**（``new EnergyVar(1)``），不是层数 ——
    按层数加会让 3 层光辉一次给 3 点能量。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    ctx.state.energy += 1
    ctx.events.append("能量 +1（光辉）")
    if amount <= 1:
        return REMOVE
    ctx.owner.add_power("radiance", -1)
    return KEEP


def _lightning_rod(ctx: PowerContext, amount: int) -> str:
    """``LightningRodPower.AfterEnergyReset``：引导 **1** 个闪电球，然后递减 1 层。

    源码注释说明了为什么放在 ``AfterEnergyReset`` 而不是 ``BeforeSideTurnStart``：
    要让"为了腾槽位而被激发的球"（等离子给的能量、冰球给的格挡）在能量重置
    /清格挡**之后**才结算，否则那笔收益会被覆盖掉。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    from . import orbs as orb_rules
    orb_rules.channel(ctx.state, "lightning", ctx.events)
    if amount <= 1:
        return REMOVE
    ctx.owner.add_power("lightning_rod", -1)
    return KEEP


def _spinner(ctx: PowerContext, amount: int) -> str:
    """``SpinnerPower.AfterEnergyReset``：引导 ``Amount`` 个玻璃球。"""
    if ctx.owner is not ctx.state.player:
        return KEEP
    from . import orbs as orb_rules
    for _ in range(max(0, amount)):
        orb_rules.channel(ctx.state, "glass", ctx.events)
    return KEEP


# ---- AfterSideTurnStart（拥有者所在阵营）--------------------------------
def _coolant(ctx: PowerContext, amount: int) -> str:
    """``CoolantPower.AfterSideTurnStart``：按**球的种类数** × ``Amount`` 获得格挡。

    ::

        int num = (from orb in OrbQueue.Orbs group orb by orb.Id).Count();
        await CreatureCmd.GainBlock(base.Owner, num * base.Amount, ValueProp.Unpowered, null);

    ⚠️ 数的是 ``group by orb.Id`` 的**种类**数，不是球的个数 —— 3 个闪电球只算 1 种。
    写成 "Count()" 会让 3 个球给 3 倍格挡，这是最容易漏的一条。
    格挡是 ``Unpowered``：不吃敏捷、不被脆弱打折。
    """
    kinds = len({orb.oid for orb in getattr(ctx.state, "orbs", ())})
    if kinds <= 0:
        return KEEP
    _gain_block(ctx.state, ctx.owner, kinds * amount, ctx.events, "冷却剂")
    return KEEP


def _countdown(ctx: PowerContext, amount: int) -> str:
    """``CountdownPower.AfterSideTurnStart``：给**随机**一个可打的敌人上 ``Amount`` 层末日。

    ::

        var creature = base.Owner.Player.RunState.Rng.CombatTargets.NextItem(HittableEnemies);
        if (creature != null) await PowerCmd.Apply<DoomPower>(..., creature, base.Amount, base.Owner, null);

    随机目标走 ``Rng.CombatTargets`` 这条独立流（与板甲/闪电球同一条），
    换成主随机流会让同种子下的随机序列与真机分叉。
    """
    targets = hittable_enemies(ctx.state)
    if not targets:
        return KEEP
    index = ctx.state.hidden.rng.next_index("combat_targets", len(targets))
    apply_power_to(ctx.state, targets[index], "doom", amount, ctx.events,
                   applier=ctx.owner)
    return KEEP


def _noxious_fumes(ctx: PowerContext, amount: int) -> str:
    """``NoxiousFumesPower.AfterSideTurnStart``：给**全体可打的敌人**上 ``Amount`` 层中毒。

    ::

        await PowerCmd.Apply<PoisonPower>(..., base.CombatState.HittableEnemies, base.Amount, base.Owner, null);

    目标是 ``HittableEnemies``（集合），不是主目标 —— 写成单体会让这张牌
    从 AoE 变成单点，强度差一整档。
    """
    for enemy in hittable_enemies(ctx.state):
        apply_power_to(ctx.state, enemy, "poison", amount, ctx.events,
                       applier=ctx.owner)
    return KEEP


def _prep_time(ctx: PowerContext, amount: int) -> str:
    """``PrepTimePower.AfterSideTurnStart``：拥有者所在阵营回合开始 → 获得 ``Amount`` 层活力。"""
    apply_power_to(ctx.state, ctx.owner, "vigor", amount, ctx.events,
                   applier=ctx.owner)
    return KEEP


def _has_keyword(state: "CombatState", card, keyword: str) -> bool:
    """``card.Keywords.Contains(CardKeyword.X)`` = **卡面关键字 + 战斗内贴上的关键字**。

    ⚠️ 两者都要查：``HexPower`` 是靠 ``TryModifyKeywordsInCombat`` 动态给"虚无"的
    （引擎用 ``affliction_keywords`` 表达），只读卡面会让"被诅咒的牌不算虚无"，
    于是 ``PagestormPower`` / ``SpiritOfAshPower`` 在这类牌上**静默不触发**。
    """
    from . import keywords as keyword_rules
    if keyword in keyword_rules.keywords_of(card.definition()):
        return True
    return keyword in affliction_keywords(state, card)


# ---- AfterCardDrawn -------------------------------------------------------
def _own_card_drawn(ctx: PowerContext) -> bool:
    """``AfterCardDrawn`` 里 ``card.Owner.Creature == base.Owner`` 的引擎写法。

    引擎里**所有**战斗牌都属于玩家（``cardPlay.Card.Owner`` 恒为玩家），
    所以"这张牌属于拥有者"等价于"拥有者是玩家"。拥有者是敌人时
    ``base.Owner.Player`` 为 null，源码里那个等式恒为假 —— 两者等价。
    """
    return ctx.owner is ctx.state.player


def _automation(ctx: PowerContext, amount: int) -> str:
    """``AutomationPower.AfterCardDrawn``：每抽一张牌计数 -1，数到 0 → +``Amount`` 能量并复位。

    ::

        data.cardsLeft--;                       // 初值 10（`new DynamicVar("BaseCards", 10m)`）
        if (data.cardsLeft <= 0) {
            await PlayerCmd.GainEnergy(base.Amount, base.Owner.Player);
            data.cardsLeft = 10;
        }

    ⚠️ 计数是"每 10 张牌"，**不是** 10 层能力；复位成 10 而不是 0。
    状态按真机的 ``InitInternalData`` 存在能力自己的实例数据里，引擎用
    ``power_flags``（同一个拥有者一份，与真机的实例计数等价）。
    """
    if not _own_card_drawn(ctx):
        return KEEP
    flags = ctx.owner.power_flags
    left = flags.get("automation_cards_left", 10) - 1
    if left <= 0:
        ctx.state.energy += amount
        ctx.events.append(f"能量 +{amount}（自动化：第 10 张牌）")
        left = 10
    flags["automation_cards_left"] = left
    return KEEP


def _cacophony(ctx: PowerContext, amount: int) -> str:
    """``CacophonyPower.AfterCardDrawn``：数到第 33 张牌 → 随机一个可打敌人受 ``Amount`` 伤害。

    ::

        base.DynamicVars.Cards.BaseValue--;      // `CardsVar(33)`
        if (base.DynamicVars.Cards.IntValue <= 0) {
            var enemy = ...Rng.CombatTargets.NextItem(base.CombatState.HittableEnemies);
            base.DynamicVars.Cards.BaseValue = 33m;
            if (enemy != null) await CreatureCmd.Damage(..., enemy, base.Amount, ValueProp.Unpowered, base.Owner);
        }

    ⚠️ 伤害是 ``Unpowered``（不吃力量/易伤，但**可被格挡**），目标走
    ``Rng.CombatTargets`` 独立随机流。
    """
    if not _own_card_drawn(ctx):
        return KEEP
    flags = ctx.owner.power_flags
    left = flags.get("cacophony_cards_left", 33) - 1
    if left <= 0:
        flags["cacophony_cards_left"] = 33
        targets = hittable_enemies(ctx.state)
        if targets:
            from . import core
            index = ctx.state.hidden.rng.next_index("combat_targets", len(targets))
            core.deal_damage(ctx.state, ctx.owner, targets[index], amount,
                             ctx.events, unpowered=True)
    else:
        flags["cacophony_cards_left"] = left
    return KEEP


def _pagestorm(ctx: PowerContext, amount: int) -> str:
    """``PagestormPower.AfterCardDrawn``：抽到带 ``Ethereal`` 的牌 → 再抽 ``Amount`` 张。

    ``card.Keywords.Contains(CardKeyword.Ethereal)`` 读的是**战斗内的关键字集合**
    （``TryModifyKeywordsInCombat`` 给的关键字也算，例如 ``hex`` 贴的虚无），
    所以这里必须查"静态关键字 + 苦痛给的关键字"两处，不能只看卡面。
    """
    if not _own_card_drawn(ctx):
        return KEEP
    card = getattr(ctx, "card", None)
    if card is None or not _has_keyword(ctx.state, card, "Ethereal"):
        return KEEP
    from .core import draw_cards
    draw_cards(ctx.state, amount, ctx.events)
    return KEEP


def _iteration(ctx: PowerContext, amount: int) -> str:
    """``IterationPower.AfterCardDrawn``：**本回合第一次**抽到状态牌 → 再抽 ``Amount`` 张。

    ::

        if (card.Owner.Creature == base.Owner && card.Type == CardType.Status) {
            int num = History.Entries.OfType<CardDrawnEntry>()
                .Count(e => e.HappenedThisTurn(base.CombatState) && e.Actor == base.Owner
                            && e.Card.Type == CardType.Status);
            if (num <= 1) await CardPileCmd.Draw(choiceContext, base.Amount, base.Owner.Player);
        }

    ⚠️ ``num <= 1`` 里的这一张**就是刚抽到的这张**，所以语义是"本回合第一张状态牌"，
    而不是"已经抽过一张之后再抽一张"。写成"每张状态牌都抽"会让状态牌变成抽牌机。
    """
    if not _own_card_drawn(ctx):
        return KEEP
    card = getattr(ctx, "card", None)
    if card is None or getattr(card.definition(), "card_type", "") != "status":
        return KEEP
    flags = ctx.owner.power_flags
    turn = getattr(ctx.state, "turn", 1)
    if flags.get("iteration_turn") != turn:
        flags["iteration_turn"] = turn
        flags["iteration_status_drawn"] = 0
    flags["iteration_status_drawn"] = flags.get("iteration_status_drawn", 0) + 1
    if flags["iteration_status_drawn"] <= 1:
        from .core import draw_cards
        draw_cards(ctx.state, amount, ctx.events)
    return KEEP


# ---- AfterDamageGiven -----------------------------------------------------
def _attack_target(ctx: PowerContext):
    """``AfterDamageGiven`` 的公共前置：这一笔伤害是不是**拥有者打出的有效攻击**。

    ``dealer == base.Owner`` 由钩子的作用域天然保证（``on_damage_given`` 只通知
    攻击方自己），``props.IsPoweredAttack()`` 与 ``result.UnblockedDamage > 0``
    仍要显式判 —— 后者是"被完全格挡就不触发"。
    """
    if not getattr(ctx, "powered", True):
        return None
    if getattr(ctx, "unblocked", 0) <= 0:
        return None
    return getattr(ctx, "target", None)


def _poison_on_attack(ctx: PowerContext, amount: int, label: str) -> str:
    """``EnvenomPower`` / ``ConcoctPower`` 共用的"有效攻击 → 给目标上毒"。"""
    target = _attack_target(ctx)
    if target is None:
        return KEEP
    apply_power_to(ctx.state, target, "poison", amount, ctx.events,
                   applier=ctx.owner)
    return KEEP


def _envenom(ctx: PowerContext, amount: int) -> str:
    """``EnvenomPower.AfterDamageGiven``（``EnvenomPower.cs:20-26``）：有效攻击掉血 → 目标中毒。"""
    return _poison_on_attack(ctx, amount, "剧毒")


def _concoct(ctx: PowerContext, amount: int) -> str:
    """``ConcoctPower.AfterDamageGiven``（``ConcoctPower.cs:21-27``）：同 ``EnvenomPower``。"""
    return _poison_on_attack(ctx, amount, "调配")


def _concoct_expire(ctx: PowerContext, amount: int) -> str:
    """``ConcoctPower.AfterSideTurnEnd``：``base.Owner.Side != side`` → 整个移除。

    ⚠️ 条件是"结束的**不是**自己这边"，所以它覆盖敌人的整个回合、
    并在**敌方**回合结束时消失（与 ``FlameBarrierPower`` 同一个写法）。
    写成"自己回合结束就移除"会让它当回合就没了。
    """
    side = getattr(ctx, "side", "")
    owner_side = "player" if ctx.owner is ctx.state.player else "enemy"
    if side and side != owner_side:
        return REMOVE
    return KEEP


def _monarchs_gaze(ctx: PowerContext, amount: int) -> str:
    """``MonarchsGazePower.AfterDamageGiven``：有效攻击 → 给目标上 ``Amount`` 层力量下降。

    ::

        if (dealer == base.Owner && props.IsPoweredAttack())
            await PowerCmd.Apply<MonarchsGazeStrengthDownPower>(..., target, base.Amount, base.Owner, null);

    ⚠️ 这里**没有** ``UnblockedDamage > 0``（被完全格挡照样上力量下降）——
    与同族的 ``EnvenomPower`` 条件不同，照抄源码。
    """
    if not getattr(ctx, "powered", True):
        return KEEP
    target = getattr(ctx, "target", None)
    if target is None:
        return KEEP
    apply_power_to(ctx.state, target, "monarchs_gaze_strength_down", amount,
                   ctx.events, applier=ctx.owner)
    return KEEP


def _monarchs_gaze_down(ctx: PowerContext, amount: int) -> str:
    """``MonarchsGazeStrengthDownPower``（``TemporaryStrengthPower`` 且 ``IsPositive => false``）。

    与 ``ManglePower`` 完全同构：施加时 -``Amount`` 力量，拥有者回合结束移除并还回去。
    归并进 :data:`_TEMPORARY_ATTRIBUTE_POWERS`，不再单独实现。
    """
    return _temporary_attribute_applied(ctx, amount, "monarchs_gaze_strength_down")


def _monarchs_gaze_down_changed(ctx: PowerContext, amount: int) -> str:
    """``MonarchsGazeStrengthDownPower`` 层数被改（叠加/递减）时同步力量。"""
    return _temporary_attribute_changed(ctx, amount, "monarchs_gaze_strength_down")


def _monarchs_gaze_down_expire(ctx: PowerContext, amount: int) -> str:
    """``TemporaryStrengthPower.AfterSideTurnEnd``：移除并把扣掉的力量还回去。"""
    return _temporary_attribute_expire(ctx, amount, "monarchs_gaze_strength_down")


#: 能力行为注册表。**不在表里的能力 = 引擎没实现**（由覆盖率报告出来）。
def _hailstorm(ctx: PowerContext, amount: int) -> str:
    """``HailstormPower.BeforeSideTurnEnd``：自家阵营回合结束前，有**冰球**就对全体可打敌人造成 ``Amount``。

    ::

        int num = Owner.Player.PlayerCombatState.OrbQueue.Orbs.Count(o => o is FrostOrb);
        if (num >= DynamicVars["FrostOrbs"].IntValue)          // FrostOrbs = 1
            await CreatureCmd.Damage(..., CombatState.HittableEnemies, base.Amount, ValueProp.Unpowered, Owner);

    ⚠️ 三条都不能省：闸门是"**至少 1 个冰球**"（`FrostOrbs` 常量 = 1，不是层数）、
    目标是 ``HittableEnemies``（复活中的打不到）、伤害是 ``Unpowered``
    （不吃力量、不被易伤放大）。
    """
    if _side_of(ctx) != _owner_side(ctx):
        return KEEP
    frost = sum(1 for orb in getattr(ctx.state, "orbs", ())
                if getattr(orb, "oid", "") == "frost")
    if frost < 1:
        return KEEP
    from . import core
    for enemy in hittable_enemies(ctx.state):
        core.deal_damage(ctx.state, ctx.owner, enemy, amount, ctx.events,
                         unpowered=True)
    return KEEP


def _sneaky(ctx: PowerContext, amount: int) -> str:
    """``SneakyPower.AfterCardPlayed``：**别人**打出攻击牌时，自己获得 ``Amount`` 格挡。

    ::

        if (cardPlay.Card.Owner.Creature != base.Owner && cardPlay.Card.Type == CardType.Attack)
            await CreatureCmd.GainBlock(base.Owner, base.Amount, ValueProp.Unpowered, null, fast: true);

    ⚠️ 条件是 ``!=``（**不是**自己打出的牌）—— 这是挂在敌人身上的能力：
    玩家一打攻击牌，它涨格挡。写成"自己打出"会让它完全反向。
    """
    if ctx.owner is ctx.state.player:
        return KEEP                     # 战斗牌都属于玩家，`!= owner` = 持有者是敌人
    if getattr(ctx, "card_type", "") != "attack":
        return KEEP
    _gain_block(ctx.state, ctx.owner, amount, ctx.events, "鬼祟")
    return KEEP


def _haunt(ctx: PowerContext, amount: int) -> str:
    """``HauntPower.AfterCardPlayed``：打出 ``Soul`` 时，对**随机**一个可打敌人造成 ``Amount`` 点不可格挡伤害。

    ::

        if (cardPlay.Card is Soul && cardPlay.Card.Owner.Creature == base.Owner) {
            var creature = Owner.Player.RunState.Rng.CombatTargets.NextItem(HittableEnemies);
            await CreatureCmd.Damage(..., creature, base.Amount, ValueProp.Unblockable | ValueProp.Unpowered, null, null, null);
        }

    三个要点：只有 ``Soul`` 触发、随机目标走 ``combat_targets`` 专用流、
    伤害同时是 ``Unblockable``（不吃格挡）与 ``Unpowered``（不吃力量）。
    攻击者为 ``null``：源码传的就是 null，所以**不该**触发攻击者的"造成伤害"类能力。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    if getattr(card, "cid", None) != "soul":
        return KEEP
    targets = hittable_enemies(ctx.state)
    if not targets:
        return KEEP
    index = ctx.state.hidden.rng.next_index("combat_targets", len(targets))
    from . import core
    core.deal_damage(ctx.state, None, targets[index], amount, ctx.events,
                     unpowered=True, unblockable=True)
    return KEEP


def _reaper_form(ctx: PowerContext, amount: int) -> str:
    """``ReaperFormPower.AfterDamageGiven``：自己造成的有效攻击伤害 → 给目标 ``总伤害 × 层数`` 层末日。

    ::

        if (dealer == base.Owner && props.IsPoweredAttack() && result.TotalDamage > 0)
            await PowerCmd.Apply<DoomPower>(..., target, result.TotalDamage * base.Amount, base.Owner, null);

    ⚠️ 用的是 ``result.TotalDamage``（**含被格挡的那部分**），不是掉血量 ——
    用 ``unblocked`` 会让"打在格挡上"的攻击少上很多末日。这也是为什么
    ``on_damage_given`` 必须把 ``total`` 传下来。
    """
    total = int(getattr(ctx, "total", 0) or 0)
    if ctx.owner is not ctx.state.player:
        return KEEP
    if not getattr(ctx, "powered", False) or total <= 0:
        return KEEP
    target = getattr(ctx, "target", None)
    if target is None:
        return KEEP
    apply_power_to(ctx.state, target, "doom", total * amount, ctx.events,
                   applier=ctx.owner)
    return KEEP


def _afterimage(ctx: PowerContext, amount: int) -> str:
    """``AfterimagePower.BeforeCardPlayed``：记下"这张牌开始结算时的层数"。

    ::

        if (cardPlay.Card.Owner.Creature != base.Owner) return;
        GetInternalData<Data>().amountsForPlayedCards.Add(cardPlay.Card, base.Amount);

    ⚠️ 为什么必须"开场记快照、结束用快照值"（源码注释写得很清楚）：

    * 这样**打出残影本体**的那次不会用"加完之后的新层数"再多给一次；
    * 在残影生效**之前**就开始结算的牌不会触发。

    引擎用 ``power_flags`` 按牌 uid 存这份快照，与真机的实例数据等价。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    ctx.owner.power_flags[f"afterimage:{getattr(card, 'uid', id(card))}"] = amount
    return KEEP


def _afterimage_payoff(ctx: PowerContext, amount: int) -> str:
    """``AfterimagePower.AfterCardPlayed``：按**快照层数**给格挡（Unpowered）。"""
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    key = f"afterimage:{getattr(card, 'uid', id(card))}"
    snapshot = ctx.owner.power_flags.pop(key, None)
    if snapshot is None or int(snapshot) <= 0:
        return KEEP
    _gain_block(ctx.state, ctx.owner, int(snapshot), ctx.events, "残影")
    return KEEP


def _serpent_form(ctx: PowerContext, amount: int) -> str:
    """``SerpentFormPower.BeforeCardPlayed``：记下"这张牌开始结算时的层数"。"""
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    ctx.owner.power_flags[f"serpent_form:{getattr(card, 'uid', id(card))}"] = amount
    return KEEP


def _serpent_form_payoff(ctx: PowerContext, amount: int) -> str:
    """``SerpentFormPower.AfterCardPlayed``：按快照值对**随机**可打敌人造成 Unpowered 伤害。

    ::

        if (cardPlay.Card.Owner == base.Owner.Player && amountsForPlayedCards.Remove(cardPlay.Card, out var damage) && damage > 0) {
            var creature = Owner.Player.RunState.Rng.CombatTargets.NextItem(HittableEnemies);
            if (creature != null) await CreatureCmd.Damage(choiceContext, creature, damage, ValueProp.Unpowered, base.Owner);
        }

    与残影同一套快照语义；随机目标同样走 ``combat_targets`` 专用流。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    key = f"serpent_form:{getattr(card, 'uid', id(card))}"
    snapshot = ctx.owner.power_flags.pop(key, None)
    if snapshot is None or int(snapshot) <= 0:
        return KEEP
    targets = hittable_enemies(ctx.state)
    if not targets:
        return KEEP
    index = ctx.state.hidden.rng.next_index("combat_targets", len(targets))
    from . import core
    core.deal_damage(ctx.state, ctx.owner, targets[index], int(snapshot), ctx.events,
                     unpowered=True)
    return KEEP


def _reflect(ctx: PowerContext, amount: int) -> str:
    """``ReflectPower.AfterDamageReceived``：**被有效攻击打掉格挡**时，把格挡量反弹给攻击者。

    ::

        if (target == base.Owner && result.BlockedDamage > 0 && props.IsPoweredAttack() && dealer != null)
            await CreatureCmd.Damage(choiceContext, dealer, result.BlockedDamage, ValueProp.Unpowered, base.Owner);

    ⚠️ 反弹的是 ``BlockedDamage``（**被格挡掉的那部分**，不是掉血量），且必须同时满足
    "有效攻击 + 有来源" —— 少一条都会凭空多出伤害。反弹本身是 ``Unpowered``。
    """
    attacker = getattr(ctx, "attacker", None)
    blocked = int(getattr(ctx, "blocked", 0) or 0)
    if ctx.owner is not ctx.state.player:
        return KEEP
    if blocked <= 0 or not getattr(ctx, "powered", False) or attacker is None:
        return KEEP
    from . import core
    core.deal_damage(ctx.state, ctx.owner, attacker, blocked, ctx.events,
                     unpowered=True)
    return KEEP


def _reflect_expire(ctx: PowerContext, amount: int) -> str:
    """``ReflectPower.AfterSideTurnStart``：自家回合开始递减 1 层。"""
    if ctx.owner is not ctx.state.player:
        return KEEP
    if amount <= 1:
        return REMOVE
    ctx.owner.add_power("reflect", -1)
    return KEEP


def _loop(ctx: PowerContext, amount: int) -> str:
    """``LoopPower.AfterPlayerTurnStart``：按层数把**最左边那个球**的被动触发 N 次。

    ::

        if (player == base.Owner.Player && OrbQueue.Orbs.Count != 0)
            for (int i = 0; i < base.Amount; i++)
                await OrbCmd.Passive(choiceContext, OrbQueue.Orbs[0], null);

    ⚠️ 三条：触发的是 ``Orbs[0]``（**最左边**，不是随机、不是全部）、
    次数是**层数**、没有球时整个不触发。``OrbCmd.Passive`` 还会带上球自己的
    跨回合状态更新（暗球涨激发值、玻璃球衰减），见 :func:`orbs.passive`。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    from . import orbs as orb_rules
    for _ in range(amount):
        if not ctx.state.orbs:
            break
        orb_rules.passive(ctx.state, ctx.state.orbs[0], ctx.events)
    return KEEP


def _hibernate(ctx: PowerContext, amount: int) -> str:
    """``HibernatePower.AfterPlayerTurnStart``：玩家回合开始 → 自己递减 1 层。"""
    if ctx.owner is not ctx.state.player:
        return KEEP
    if amount <= 1:
        return REMOVE
    ctx.owner.add_power("hibernate", -1)
    return KEEP


def _corrosive_wave(ctx: PowerContext, amount: int) -> str:
    """``CorrosiveWavePower.AfterCardDrawn``：自己抽到一张牌 → 全体可打敌人各 ``Amount`` 层中毒。

    ::

        if (card.Owner.Creature != base.Owner) return;
        await PowerCmd.Apply<PoisonPower>(..., CombatState.HittableEnemies, base.Amount, base.Owner, null);

    ⚠️ **不看** ``fromHandDraw``：回合开始抽的 5 张牌同样触发（与 ``SpeedsterPower`` 相反）。
    目标是集合 ``HittableEnemies``（复活中的打不到）。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    for enemy in hittable_enemies(ctx.state):
        apply_power_to(ctx.state, enemy, "poison", amount, ctx.events,
                       applier=ctx.owner)
    return KEEP


def _speedster(ctx: PowerContext, amount: int) -> str:
    """``SpeedsterPower.AfterCardDrawn``：**非**手牌抽取的抽牌 → 全体可打敌人受 ``Amount`` 点 Unpowered 伤害。

    ::

        if (!fromHandDraw && card.Owner.Creature == base.Owner && CurrentSide == Owner.Side)
            await CreatureCmd.Damage(..., HittableEnemies, base.Amount, ValueProp.Unpowered, base.Owner);

    ⚠️ ``!fromHandDraw`` 是这条能力的**全部意义**：回合开始的手牌抽取不触发，
    只有效果抽牌（战斗内抽牌）才触发。少了这个标志，它每回合白打一发。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    if getattr(ctx, "from_hand_draw", False):
        return KEEP
    from . import core
    for enemy in hittable_enemies(ctx.state):
        core.deal_damage(ctx.state, ctx.owner, enemy, amount, ctx.events,
                         unpowered=True)
    return KEEP


def _monologue_begin(ctx: PowerContext, amount: int) -> str:
    """``MonologuePower.BeforeCardPlayed``：记下**开始结算时**该给的力量。

    ::

        GetInternalData<Data>().amountsForPlayedCards.Add(cardPlay.Card,
            base.DynamicVars.Strength.IntValue);      // PowerVar<StrengthPower>(1)

    ⚠️ 记的是 ``DynamicVars.Strength``（源码 ``CanonicalVars`` 里写死的 **1**），
    **不是**能力层数 —— 用层数会让多层的独白凭空变强。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    ctx.owner.power_flags[f"monologue:{getattr(card, 'uid', id(card))}"] = 1
    return KEEP


def _monologue_payoff(ctx: PowerContext, amount: int) -> str:
    """``MonologuePower.AfterCardPlayed``：按快照给力量，并累计"这一回合给了多少"。

    :: 

        await PowerCmd.Apply<StrengthPower>(..., base.Owner, value, base.Owner, null, silent: true);
        DynamicVars["StrengthApplied"].BaseValue += DynamicVars.Strength.IntValue;

    ``StrengthApplied`` 是用来在回合结束时**精确撤回**的（见 :func:`_monologue_expire`）。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    key = f"monologue:{getattr(card, 'uid', id(card))}"
    snapshot = ctx.owner.power_flags.pop(key, None)
    if snapshot is None or int(snapshot) <= 0:
        return KEEP
    apply_power_to(ctx.state, ctx.owner, "strength", int(snapshot), ctx.events,
                   applier=ctx.owner)
    flags = ctx.owner.power_flags
    flags["monologue_applied"] = flags.get("monologue_applied", 0) + int(snapshot)
    return KEEP


def _monologue_expire(ctx: PowerContext, amount: int) -> str:
    """``MonologuePower.AfterSideTurnEnd``：自家回合结束 → 先移除自己，再撤回给过的力量。

    ⚠️ 撤回的是**累计给过的量**（``StrengthApplied``），不是层数：
    这一回合打了 3 张牌就该撤 3 点，撤层数会在多张牌时留下残留力量。
    """
    if _side_of(ctx) != _owner_side(ctx):
        return KEEP
    applied = int(ctx.owner.power_flags.pop("monologue_applied", 0) or 0)
    if applied:
        # ⚠️ 这里**必须**用 `add_power` 而不是 `apply_power_to`：
        # 真机是 `PowerCmd.Apply<StrengthPower>(..., -StrengthApplied, ..., silent: true)`，
        # 而引擎的 `apply_power_to` 是"施加"语义、不接受负数（实测静默无效）。
        # `silent: true` 本身也意味着**不**走施加类反馈，所以直接减层是对的。
        ctx.owner.add_power("strength", -applied)
        ctx.events.append(f"{ctx.owner.name} 失去 {applied} 点力量（独白结束）")
    return REMOVE


def _panache(ctx: PowerContext, amount: int) -> str:
    """``PanachePower.AfterCardPlayed``：每打 **5** 张牌，对全体可打敌人造成 ``Amount`` 点 Unpowered 伤害。

    ::

        if (data.alreadyApplied) {
            DynamicVars["CardsLeft"].BaseValue--;
            if (DynamicVars["CardsLeft"].IntValue <= 0) {
                await CreatureCmd.Damage(..., HittableEnemies, base.Amount, ValueProp.Unpowered, base.Owner);
                DynamicVars["CardsLeft"].BaseValue = 5m;
            }
        }
        data.alreadyApplied = true;

    ⚠️ ``alreadyApplied`` 的作用是**别把潘趣自己算进去**（源码注释：*"so we don't count
    the Panache card towards itself"*）—— 第一次收到事件（就是打出潘趣本体那一次）
    只翻标志、不计数。计划里把它理解成"每 5 张"就会少算一张。
    计数与"还差几张"存在 ``power_flags`` 里，等价于真机的实例数据。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    flags = ctx.owner.power_flags
    if flags.get("panache_applied"):
        left = int(flags.get("panache_cards_left", 5)) - 1
        if left <= 0:
            from . import core
            for enemy in hittable_enemies(ctx.state):
                core.deal_damage(ctx.state, ctx.owner, enemy, amount, ctx.events,
                                 unpowered=True)
            left = 5
        flags["panache_cards_left"] = left
    else:
        flags["panache_applied"] = 1
    return KEEP


def _pale_blue_dot(ctx: PowerContext, amount: int) -> str:
    """``PaleBlueDotPower.AfterCardPlayed``：本回合打满 **5** 张牌 → 获得 ``Amount`` 层"下回合抽牌"。

    ::

        if (!data.alreadyActivatedThisTurn && AttacksPlayedThisTurn >= 5) {
            data.alreadyActivatedThisTurn = true;
            await PowerCmd.Apply<DrawCardsNextTurnPower>(..., base.Owner, base.Amount, base.Owner, null);
        }

    ``AttacksPlayedThisTurn`` 数的是**本回合结算完的牌**（``CardPlaysFinished``，
    名字里的 Attacks 是误导），所以用引擎的 ``cards_played_started_this_turn`` 对齐；
    它在 ``AfterCardPlayed`` 触发时已经包含当前这张牌。
    每回合只触发一次，回合结束复位（见 :func:`_pale_blue_dot_reset`）。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    flags = ctx.owner.power_flags
    if flags.get("pale_blue_dot_used"):
        return KEEP
    if getattr(ctx.state, "cards_played_started_this_turn", 0) < 5:
        return KEEP
    flags["pale_blue_dot_used"] = 1
    apply_power_to(ctx.state, ctx.owner, "draw_cards_next_turn", amount, ctx.events,
                   applier=ctx.owner)
    return KEEP


def _pale_blue_dot_reset(ctx: PowerContext, amount: int) -> str:
    """``PaleBlueDotPower.AfterSideTurnEnd``：自家回合结束 → 复位"本回合已触发"。"""
    if _side_of(ctx) == _owner_side(ctx):
        ctx.owner.power_flags.pop("pale_blue_dot_used", None)
    return KEEP


def _storm_begin(ctx: PowerContext, amount: int) -> str:
    """``StormPower.BeforeCardPlayed``：**能力牌**开始结算时记下层数快照。"""
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    if getattr(card.definition(), "card_type", "") != "power":
        return KEEP
    ctx.owner.power_flags[f"storm:{getattr(card, 'uid', id(card))}"] = amount
    return KEEP


def _storm_payoff(ctx: PowerContext, amount: int) -> str:
    """``StormPower.AfterCardPlayed``：按快照引导这么多**闪电球**。

    ::

        if (cardPlay.Card.Owner == base.Owner.Player && amountsForPlayedCards.Remove(cardPlay.Card, out var lightning)
            && lightning > 0)
            for (int i = 0; i < lightning; i++)
                await OrbCmd.Channel<LightningOrb>(choiceContext, base.Owner.Player);

    ⚠️ 只有**能力牌**会进快照（``BeforeCardPlayed`` 里判了 ``CardType.Power``），
    所以打攻击/技能牌时字典里根本没有它 —— 用"任意牌都引导"会让风暴变成万能引擎。
    与残影/蛇形同一套快照语义（打出风暴本体那次用的是**加层前**的层数）。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    key = f"storm:{getattr(card, 'uid', id(card))}"
    snapshot = ctx.owner.power_flags.pop(key, None)
    if snapshot is None or int(snapshot) <= 0:
        return KEEP
    from . import orbs as orb_rules
    for _ in range(int(snapshot)):
        orb_rules.channel(ctx.state, "lightning", ctx.events)
    return KEEP


def _spirit_of_ash(ctx: PowerContext, amount: int) -> str:
    """``SpiritOfAshPower.BeforeCardPlayed``：打出**虚无（Ethereal）**牌时 +``Amount`` 格挡（Unpowered）。

    ::

        if (cardPlay.Card.Owner == base.Owner.Player && cardPlay.Card.Keywords.Contains(CardKeyword.Ethereal))
            await CreatureCmd.GainBlock(base.Owner, base.Amount, ValueProp.Unpowered, null);

    ⚠️ 判的是**牌的 Keywords 里有没有 Ethereal**（不是"会不会被消耗"、也不是牌型）。
    引擎里关键字来自内容表 ``CardDef.keywords``。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    keywords = tuple(getattr(card.definition(), "keywords", ()) or ())
    if "Ethereal" not in keywords:
        return KEEP
    _gain_block(ctx.state, ctx.owner, amount, ctx.events, "灰烬之魂")
    return KEEP


def _the_sealed_throne(ctx: PowerContext, amount: int) -> str:
    """``TheSealedThronePower.BeforeCardPlayed``：自己每打一张牌 → +``Amount`` 星。

    ::

        if (cardPlay.Card.Owner == base.Owner.Player)
            await PlayerCmd.GainStars(base.Amount, base.Owner.Player);

    没有别的条件 —— 任意牌都触发（这是"封印王座"给摄政王的星资源引擎）。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    from . import core
    core.gain_stars(ctx.state, amount, ctx.events)
    return KEEP


#: ``BeforeHandDraw`` 一族：从**角色卡池**里生成牌塞进手牌时的过滤条件。
#:
#: 真机走 ``CardFactory.GetDistinctForCombat(..., Rng.CombatCardGeneration)`` ——
#: 「distinct」= 同一次生成里不重复，「combat」= 生成的是战斗内副本。
#: 随机数走 ``combat_card_generation`` 这条**专用流**（与卡牌奖励分开，
#: 见 `rng.py` 的说明）。
def amount_on_turn_start(combatant, pid: str) -> int:
    """``PowerModel.AmountOnTurnStart``：这个能力**回合开始时**的层数。

    ⚠️ 与"当前层数"不同：回合中途被改层数时，读当前值会算错
    （``HelloWorldPower`` 按它生成牌、``DrawCardsNextTurnPower`` 用它判"还有没有"）。
    快照在 ``core.start_player_turn`` 的第 1 步完成（对应 ``Creature.cs:682``）。
    """
    return int(getattr(combatant, "power_amount_on_turn_start", {}).get(pid, 0))


def generate_cards_to_hand(state: "CombatState", count: int, events: list[str],
                           *, pool: str | None = None, rarity: str | None = None,
                           card_type: str | None = None,
                           exclude_basic_and_ancient: bool = False,
                           exclude: tuple[str, ...] = (),
                           free: bool = False,
                           upgrade: bool = False) -> list[str]:
    """从卡池里随机抽 ``count`` 张（**互不重复**）塞进手牌，返回卡 id 列表。

    对应真机的 ``CardFactory.GetDistinctForCombat`` + ``CardPileCmd.AddGeneratedCardToCombat``：

      * **互不重复**（distinct）：同一次生成里不会出现两张同名牌；
      * 走 ``combat_card_generation`` 专用流（用 ``shuffle`` 会与真机分叉，而且
        会让"用药水生成牌"影响到洗牌序列）；
      * 生成的牌进**手牌**，所以受手牌上限约束（``add_to_hand`` 负责）；
      * ``free=True`` —— 调用方（``CardModel.SetToFreeThisTurn()``）要求"生成的牌
        本回合免费"。**必须在这里置标记**，不能生成之后再按 cid 回查手牌：
        手牌满时 ``add_to_hand`` 会把牌改放弃牌堆，回查就找不到它，
        而真机那张牌**在弃牌堆里也带着 0 费**（修饰符跟着卡实例走）。
    """
    from . import content as content_module
    from . import eligibility as gate
    from .core import CardInstance, _add_generated_card

    pool_name = pool or content_module.character_pool(
        getattr(state, "character", None) or content_module.DEFAULT_CHARACTER)
    #: 空池名 = 调用方要"固定卡"（飞刀 / 横扫凝视），走 `add_copies_to_hand`。
    if not pool_name:
        return []
    members = list(content_module.CARD_POOLS.get(pool_name, ()))
    if not members:                                  # pragma: no cover - 内容缺失
        return []
    candidates = []
    for cid in members:
        definition = content_module.CARD_DB.get(cid)
        if definition is None:
            continue
        if rarity is not None and definition.rarity != rarity:
            continue
        if card_type is not None and definition.card_type != card_type:
            continue
        if exclude_basic_and_ancient and definition.rarity in ("basic", "ancient"):
            continue
        if exclude and cid in exclude:
            # ``where !(c is JackOfAllTrades)``：真机防"把自己生成出来"。
            continue
        candidates.append(cid)
    # ⭐⭐ **生成的牌必须过准入门禁**（外部复核 R3）。
    #
    # 原来这里直接抽整个卡池，于是 `creative_ai`（引擎实现了、也被准入）会发出
    # `biased_cognition` 这类**引擎执行不了**的牌 —— 实测种子 8 就是它。
    # 那等于"模拟器凭空给玩家一张它自己算不对的牌"，而且报告里一切正常。
    #
    # 按 `docs/13` §7 的口径：受限课程**预先缩池**并标注分布变化，
    # 而不是发出来之后静默删掉效果或重抽（那种做法把"缺内容"藏了起来）。
    admitted = gate.admitted_cards()
    kept = [cid for cid in candidates if cid in admitted]
    if len(kept) != len(candidates):
        events.append(
            f"生成池按准入门禁缩小：{len(candidates)} → {len(kept)} 张"
            f"（{len(candidates) - len(kept)} 张引擎执行不了）")
    candidates = kept
    if not candidates:
        return []
    stream = state.hidden.rng
    stream.shuffle(candidates, "combat_card_generation")
    generated: list[str] = []
    for cid in candidates[:max(0, count)]:
        # 同上：走"生成"出口，好让 `AfterCardGeneratedForCombat` 生效。
        # ``upgrade`` —— 真机 `CosmicConcoction` 是"逐张 `CardCmd.Upgrade` 之后
        # 才进手牌"，所以生成出来的就是**升级版实例**。
        instance = CardInstance(cid, upgraded=upgrade)
        if free:
            # ``CardModel.SetToFreeThisTurn()`` —— 真机是"先生成实例、再把它设成
            # 本回合免费"，所以免费标记属于**这个实例**，不写进卡牌定义。
            instance.free_this_turn = True
        _add_generated_card(state, instance, "hand")
        generated.append(cid)
    return generated


def _generate_for_power(ctx: PowerContext, amount: int, *, pool: str | None = None,
                        rarity: str | None = None, card_type: str | None = None,
                        exclude_basic_and_ancient: bool = False) -> None:
    generate_cards_to_hand(ctx.state, amount, ctx.events, pool=pool, rarity=rarity,
                           card_type=card_type,
                           exclude_basic_and_ancient=exclude_basic_and_ancient)


def _call_of_the_void(ctx: PowerContext, amount: int) -> str:
    """``CallOfTheVoidPower.BeforeHandDraw``：抽牌前把 ``Amount`` 张**非基础/非远古**的池内牌塞进手牌。

    ::

        var pool = Owner.Player.Character.CardPool.GetUnlockedCards(...)
                     .Where(c => c.Rarity != CardRarity.Basic && c.Rarity != CardRarity.Ancient);
        CardFactory.GetDistinctForCombat(..., Rng.CombatCardGeneration) → AddGeneratedCardToCombat(Hand)

    ⚠️ 过滤的是**稀有度**（排除 Basic / Ancient），不是牌型 ——
    照抄这条才能让"虚空召唤"不会塞进打击/防御。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    _generate_for_power(ctx, amount, exclude_basic_and_ancient=True)
    return KEEP


def _creative_ai(ctx: PowerContext, amount: int) -> str:
    """``CreativeAiPower.BeforeHandDraw``：抽牌前把 ``Amount`` 张**能力牌**塞进手牌。"""
    if ctx.owner is not ctx.state.player:
        return KEEP
    _generate_for_power(ctx, amount, card_type="power")
    return KEEP


def _hello_world(ctx: PowerContext, amount: int) -> str:
    """``HelloWorldPower.BeforeHandDraw``：按 **``AmountOnTurnStart``** 生成等量**普通**牌。

    ::

        if (player == base.Owner.Player && base.AmountOnTurnStart >= 1)
            CardFactory.GetDistinctForCombat(..., where c.Rarity == CardRarity.Common,
                                             base.AmountOnTurnStart, Rng.CombatCardGeneration)

    ⚠️ 张数用的是 ``AmountOnTurnStart``（回合开始时的层数），**不是**当前层数；
    而且 ``>= 1`` 才动手 —— 0 层时不该"生成 0 张"变成空操作日志噪声。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    snapshot = amount_on_turn_start(ctx.owner, "hello_world")
    if snapshot < 1:
        return KEEP
    _generate_for_power(ctx, snapshot, rarity="common")
    return KEEP


def _spectrum_shift(ctx: PowerContext, amount: int) -> str:
    """``SpectrumShiftPower.BeforeHandDraw``：抽牌前把 ``Amount`` 张**无色**牌塞进手牌。

    :: 

        CardFactory.GetDistinctForCombat(player, ModelDb.CardPool<ColorlessCardPool>()..., base.Amount, ...)

    ⚠️ 池子是 **ColorlessCardPool**，不是角色池 —— 用错池子会发出角色牌。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    _generate_for_power(ctx, amount, pool="ColorlessCardPool")
    return KEEP


def add_copies_to_hand(state: "CombatState", cid: str, count: int,
                       events: list[str]) -> list[str]:
    """把 ``count`` 张**指定的**牌塞进手牌（``CardFactory`` 的固定卡生成）。

    与 :func:`generate_cards_to_hand` 的区别：这里**不抽随机** ——
    飞刀（``Shiv.CreateInHand``）与横扫凝视（``CreateCard<SweepingGaze>``）
    生成的永远是同一张牌。张数不受"池子大小"限制（可以有 5 张飞刀）。
    """
    from .content import CARD_DB
    from .core import CardInstance, _add_generated_card
    made: list[str] = []
    if cid not in CARD_DB:
        return made
    for _ in range(max(0, count)):
        # ⚠️ 走 `_add_generated_card` 而不是 `add_to_hand`：前者是"生成"的**唯一**
        # 出口，会触发 `AfterCardGeneratedForCombat`（军械库/烟囱/化废为宝靠它）。
        _add_generated_card(state, CardInstance(cid), "hand")
        made.append(cid)
    return made


def _sentry_mode(ctx: PowerContext, amount: int) -> str:
    """``SentryModePower.BeforeHandDraw``：抽牌前塞 ``Amount`` 张 ``SweepingGaze``。

    ``combatState.CreateCard<SweepingGaze>(Owner.Player)`` —— 固定卡，不抽随机。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    add_copies_to_hand(ctx.state, "sweeping_gaze", amount, ctx.events)
    return KEEP


def _infinite_blades(ctx: PowerContext, amount: int) -> str:
    """``InfiniteBladesPower.BeforeHandDraw``：抽牌前塞 ``Amount`` 张**飞刀**。

    ``await Shiv.CreateInHand(base.Owner.Player, base.Amount, combatState);``
    —— 固定生成、不抽随机，且同样受手牌上限约束。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    add_copies_to_hand(ctx.state, "shiv", amount, ctx.events)
    return KEEP


def _arsenal(ctx: PowerContext, amount: int) -> str:
    """``ArsenalPower.AfterCardGeneratedForCombat``：自己生成的牌 → +``Amount`` 力量。

    ::

        if (creator != null && creator.Creature == base.Owner)
            await PowerCmd.Apply<StrengthPower>(..., base.Owner, base.Amount, base.Owner, null);

    ⚠️ 条件是 **``creator == base.Owner``**（这张牌是"我"生成的），
    别人生成的牌不该给力量。
    """
    creator = getattr(ctx, "creator", None)
    if creator is not ctx.owner:
        return KEEP
    apply_power_to(ctx.state, ctx.owner, "strength", amount, ctx.events,
                   applier=ctx.owner)
    return KEEP


def _pillar_of_creation(ctx: PowerContext, amount: int) -> str:
    """``PillarOfCreationPower.AfterCardGeneratedForCombat``：自己生成的牌 → +``Amount`` 格挡（Unpowered）。"""
    creator = getattr(ctx, "creator", None)
    if creator is not ctx.owner:
        return KEEP
    _gain_block(ctx.state, ctx.owner, amount, ctx.events, "创生之柱")
    return KEEP


def _smokestack(ctx: PowerContext, amount: int) -> str:
    """``SmokestackPower.AfterCardGeneratedForCombat``：生成的牌是**状态牌**时才群伤。

    ::

        if (card.Type == CardType.Status && creator != null && creator.Creature == base.Owner)
            await CreatureCmd.Damage(..., HittableEnemies, base.Amount, ValueProp.Unpowered, base.Owner);

    ⚠️ 两道条件都要：**牌型是 Status** + 是自己生成的。少第一道会让"生成任何牌"
    都触发群伤（强度差一个量级）。
    """
    card = getattr(ctx, "card", None)
    creator = getattr(ctx, "creator", None)
    if creator is not ctx.owner or card is None:
        return KEEP
    if getattr(card.definition(), "card_type", "") != "status":
        return KEEP
    from . import core
    for enemy in hittable_enemies(ctx.state):
        core.deal_damage(ctx.state, ctx.owner, enemy, amount, ctx.events,
                         unpowered=True)
    return KEEP


def _trash_to_treasure(ctx: PowerContext, amount: int) -> str:
    """``TrashToTreasurePower.AfterCardGeneratedForCombat``：生成**状态牌** → 引导 ``Amount`` 个**随机**充能球。

    ::

        if (card.Type == CardType.Status && creator != null && creator.Creature == base.Owner)
            for (int i = 0; i < base.Amount; i++) {
                var orb = OrbModel.GetRandomOrb(Owner.Player.RunState.Rng.CombatOrbGeneration);
                await OrbCmd.Channel(..., orb, Owner.Player);
            }

    ⚠️ 随机球走 ``combat_orbs`` 专用流（与"混沌"同一条），不是战斗主随机。
    """
    card = getattr(ctx, "card", None)
    creator = getattr(ctx, "creator", None)
    if creator is not ctx.owner or card is None:
        return KEEP
    if getattr(card.definition(), "card_type", "") != "status":
        return KEEP
    from . import orbs as orb_rules
    kinds = sorted(orb_rules.ORB_DEFS)
    if not kinds:
        return KEEP
    for _ in range(amount):
        index = ctx.state.hidden.rng.next_index("combat_orbs", len(kinds))
        orb_rules.channel(ctx.state, kinds[index], ctx.events)
    return KEEP


def _shroud(ctx: PowerContext, amount: int) -> str:
    """``ShroudPower.AfterPowerAmountChanged``：**自己**给别人上**末日**时，+``Amount`` 格挡。

    ::

        if (applier == base.Owner && power is DoomPower)
            await CreatureCmd.GainBlock(base.Owner, base.Amount, ValueProp.Unpowered, null);

    ⚠️ 两道条件：``applier == base.Owner``（是"我"上的）与 **被改的是末日**。
    递减（``Decrement`` 的 ``applier: null``）**不该**给格挡 —— 漏掉第一条就会
    "每次末日跳一下都白拿格挡"。
    """
    applier = getattr(ctx, "applier", None)
    if applier is not ctx.owner:
        return KEEP
    if getattr(ctx, "pid", "") != "doom":
        return KEEP
    _gain_block(ctx.state, ctx.owner, amount, ctx.events, "裹尸布")
    return KEEP


def power_type_for_amount(pid: str, amount: int) -> str:
    """``PowerModel.GetTypeForAmount``：**按这次的变动量**判这个能力是增益还是减益。

    ::

        if (StackType == Counter && AllowNegative && customAmount < 0) return Debuff;
        if (!AllowNegative && Type == Debuff && customAmount < 0)     return Buff;
        return Type;

    ⚠️ "把减益层数**减**下去"对玩家是**好事**（源码判成 Buff），所以按 ``pid`` 的静态
    类型判会让"给敌人掉虚弱"也触发减益类能力 —— 差一个反向。
    """
    from .content import POWERS
    definition = POWERS.get(pid)
    if definition is None:
        return ""
    kind = getattr(definition, "kind", "")
    if (getattr(definition, "stack_type", "") == "counter"
            and getattr(definition, "allow_negative", False) and amount < 0):
        return "debuff"
    if (not getattr(definition, "allow_negative", False) and kind == "debuff"
            and amount < 0):
        return "buff"
    return kind


def _sleight_of_flesh(ctx: PowerContext, amount: int) -> str:
    """``SleightOfFleshPower.AfterPowerAmountChanged``：自己给**敌人**上**减益**时，对该敌人造成 ``Amount`` 点伤害。

    ::

        if (amount != 0m && power.GetTypeForAmount(amount) == PowerType.Debuff
            && power.Owner.IsEnemy && applier == base.Owner && !(power is ITemporaryPower))
            await CreatureCmd.Damage(..., power.Owner, base.Amount, ValueProp.Unpowered, base.Owner);

    ⚠️ 四条条件缺一不可：变动非零、**是减益**、被上的是**敌人**、**是自己上的**。
    少"减益"这条会让"给敌人上任何能力都掉血"。
    """
    from .content import POWERS
    applier = getattr(ctx, "applier", None)
    changed_owner = getattr(ctx, "changed_owner", None)
    pid = getattr(ctx, "pid", "")
    if applier is not ctx.owner or int(getattr(ctx, "amount", 0) or 0) == 0:
        return KEEP
    if changed_owner is None or changed_owner is ctx.state.player:
        return KEEP                              # 被上能力的必须是**敌人**
    if power_type_for_amount(pid, int(getattr(ctx, "amount", 0) or 0)) != "debuff":
        return KEEP
    from . import core
    core.deal_damage(ctx.state, ctx.owner, changed_owner, amount, ctx.events,
                     unpowered=True)
    return KEEP


def _owner_side(ctx: PowerContext) -> str:
    """拥有者属于哪一方（``"player"`` / ``"enemy"``）。"""
    return "player" if ctx.owner is ctx.state.player else "enemy"


#: ``Hook.AfterCombatEnd``：战斗结束时给**额外奖励**的能力。
#: ``(能力 id, 奖励种类)`` —— 种类由 ``sts2_sim.run`` 翻译成 Run 层效果：
#:
#: * ``gold`` —— ``RoyaltiesPower``：``room.AddExtraReward(GoldReward(Amount))``
#: * ``remove_card_choices`` —— ``ForbiddenGrimoirePower``：
#:   ``AddExtraReward(CardRemovalReward))`` × Amount（**玩家各选一张**移除）
#: * ``upgrade_random_deck_cards`` —— ``ImprovementPower``：
#:   ``Rng.CombatCardSelection.NextItem`` 随机挑可升级的牌，挑一张移出候选
#:
#: ⚠️ 这三条都作用在**牌组 / 金币**上，那些归 Run 层（``docs/13`` §3.2 的分层契约），
#: 所以战斗侧只产出**描述**，怎么落地由 :meth:`sts2_sim.run.RunEnv._apply_after_combat_end_powers`
#: 决定 —— 这样 ``powers.py`` 不需要 import Run 层，反之亦然。
AFTER_COMBAT_END_POWERS: tuple[tuple[str, str], ...] = (
    ("royalties", "gold"),
    ("forbidden_grimoire", "remove_card_choices"),
    ("improvement", "upgrade_random_deck_cards"),
)


def after_combat_end(state: "CombatState", events: list[str]) -> list[dict]:
    """``Hook.AfterCombatEnd``：战斗结束时发额外奖励的能力，返回**奖励描述**。

    真机签名是 ``AfterCombatEnd(CombatRoom room)``，能力在里面对房间追加奖励
    （``room.AddExtraReward(player, reward)``）。引擎里"房间/奖励队列"在 Run 层，
    所以这里只回答"谁该给什么、给多少"，由 Run 层去发。
    """
    player = getattr(state, "player", None)
    if player is None:
        return []
    out: list[dict] = []
    for pid, kind in AFTER_COMBAT_END_POWERS:
        amount = player.power(pid)
        if amount <= 0:
            continue
        out.append({"kind": kind, "amount": int(amount), "power": pid})
        events.append(f"战斗结束奖励：{pid} → {kind} ×{amount}")
    return out


def afflict_card(state: "CombatState", card, affliction: str,
                 events: list[str]) -> bool:
    """给**一张**牌贴苦痛（``CardCmd.Afflict``），已带苦痛的牌跳过。

    对应源码里 ``AfterCardEnteredCombat`` 那一半：新进场的牌如果
    ``card.Affliction == null`` 就贴。返回是否贴上了。

    ⚠️ **已知简化（与 ``afflict_cards`` 同一个）**：真机的苦痛带**逐卡快照的量**
    （``Afflict<T>(card, base.Amount)``），引擎的 ``CardInstance.affliction`` 只存名字，
    消费者读的是**当前层数**。层数在战斗中途变过时两者会分叉 —— 这一点写在
    ``docs/09`` 的口径里，不在这里偷偷"顺手修"（改它要动费用计算与关键字查询）。
    """
    if card is None or getattr(card, "affliction", None) is not None:
        return False
    card.affliction = affliction
    events.append(f"{card.definition().name} 被施加苦痛「{affliction}」")
    return True


def _entered_affliction(ctx: PowerContext, affliction: str,
                        card_type: str | None = None,
                        require_skill_played: bool = False) -> str:
    """苦痛族 ``AfterCardEnteredCombat`` 的公共条件：**新牌**按类型补贴苦痛。

    源码里六个能力这一半的判据都是同一个形状（``tangled`` / ``ringing`` / ``hex``
    还多一条"所有者是玩家"）：牌刚进战斗、还没有苦痛、类型对得上。
    ``smoggy`` 额外要求"本回合已经打出过技能牌"（``CardPlaysStarted`` 过滤技能牌）。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    if card_type is not None and card.definition().card_type != card_type:
        return KEEP
    if require_skill_played and getattr(ctx.state, "skills_played_this_turn", 0) <= 0:
        return KEEP
    afflict_card(ctx.state, card, affliction, ctx.events)
    return KEEP


def _tangled_entered(ctx: PowerContext, amount: int) -> str:
    """``TangledPower.AfterCardEnteredCombat``：新进场的**攻击牌**贴 ``Entangled``。"""
    return _entered_affliction(ctx, "entangled", card_type="attack")


def _smoggy_entered(ctx: PowerContext, amount: int) -> str:
    """``SmoggyPower.AfterCardEnteredCombat``：本回合打过技能牌时，新进场的**技能牌**贴 ``Smog``。"""
    return _entered_affliction(ctx, "smog", card_type="skill",
                               require_skill_played=True)


def _ringing_entered(ctx: PowerContext, amount: int) -> str:
    """``RingingPower.AfterCardEnteredCombat``：新进场的**任意**牌贴 ``Ringing``。"""
    return _entered_affliction(ctx, "ringing")


def _hex_entered(ctx: PowerContext, amount: int) -> str:
    """``HexPower.AfterCardEnteredCombat``：新进场的**任意**牌贴 ``Hexed``。"""
    return _entered_affliction(ctx, "hexed")


def _vital_spark_entered(ctx: PowerContext, amount: int) -> str:
    """``VitalSparkPower.AfterCardEnteredCombat``：新进场的**技能牌**贴 ``Tainted``。"""
    return _entered_affliction(ctx, "tainted", card_type="skill")


def _galvanic_entered(ctx: PowerContext, amount: int) -> str:
    """``GalvanicPower.AfterCardEnteredCombat``：新进场的**能力牌**贴 ``Galvanized``。"""
    return _entered_affliction(ctx, "galvanized", card_type="power")


def grant_keyword(state: "CombatState", card, keyword: str,
                  events: list[str] | None = None) -> bool:
    """``CardCmd.ApplyKeyword(card, keyword)``：给**一张牌实例**加战斗内关键字。

    真机的关键字有"卡牌固有"（``CardDef.Keywords``）与"战斗内施加"两层，
    这个函数只管后者 —— 引擎存在 ``CardInstance.keywords``，战斗结束随副本丢弃。
    """
    if card is None:
        return False
    current = tuple(getattr(card, "keywords", ()) or ())
    if keyword in current:
        return False
    card.keywords = current + (keyword,)
    if events is not None:
        events.append(f"{card.definition().name} 获得关键字 {keyword}（本场战斗）")
    return True


def _mark_shivs_retain(state: "CombatState", events: list[str]) -> int:
    """给**本场战斗里所有飞刀**加 ``Retain``（``PhantomBladesPower`` 的两半都用它）。

    真机遍历的是 ``PlayerCombatState.AllCards``；引擎对应"抽牌堆 + 手牌 +
    弃牌堆 + 出牌区"（消耗堆里的牌已经离场，加了也没有意义）。
    """
    marked = 0
    for pile_name in ("hand", "draw_pile", "discard", "play_area"):
        for card in list(getattr(state, pile_name, None) or []):
            if "Shiv" not in tuple(getattr(card.definition(), "tags", ()) or ()):
                continue
            marked += 1 if grant_keyword(state, card, "Retain", events) else 0
    return marked


def _phantom_blades_apply(ctx: PowerContext, amount: int) -> str:
    """``PhantomBladesPower.AfterApplied``：贴上的瞬间给在场所有飞刀加 ``Retain``。"""
    marked = _mark_shivs_retain(ctx.state, ctx.events)
    if marked:
        ctx.events.append(f"幻影之刃：{marked} 张飞刀获得保留")
    return KEEP


def _phantom_blades_entered(ctx: PowerContext, amount: int) -> str:
    """``PhantomBladesPower.AfterCardEnteredCombat``：**新进场的飞刀**也加 ``Retain``。"""
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    if "Shiv" not in tuple(getattr(card.definition(), "tags", ()) or ()):
        return KEEP
    grant_keyword(ctx.state, card, "Retain", ctx.events)
    return KEEP


def _side_of(ctx: PowerContext) -> str:
    """本次钩子通知的是哪一方（``Hook.BeforeSideTurnEnd`` 的 ``side`` 参数）。"""
    return str(getattr(ctx, "side", ""))


# ==========================================================================
# Wave B · 第十五批：把"注册了但没人消费"的能力接上（源码逐条对齐）
# ==========================================================================
def _curl_up_mark(ctx: PowerContext, amount: int) -> str:
    """``CurlUpPower.AfterDamageReceived``：记下"是哪张**牌**打中了它"。

    源码只看三件事（连"掉没掉血"都不看 —— 参数是 ``DamageResult _``）：

    * ``target == base.Owner``（挨打的是自己）；
    * ``props.IsPoweredAttack()``；
    * ``cardSource != null``，且**已经记过的那张牌不算"另一张"**
      （同一张牌的多段伤害只记第一次）。

    记下来的用途在 :func:`_curl_up_play`：那张牌**结算完**才蜷缩。
    """
    card = getattr(ctx, "card", None)
    if not getattr(ctx, "powered", True) or card is None:
        return KEEP
    flags = ctx.owner.power_flags
    if not flags.get("curl_up_card", 0):
        flags["curl_up_card"] = getattr(card, "uid", 0)
    return KEEP


def _curl_up_play(ctx: PowerContext, amount: int) -> str:
    """``CurlUpPower.AfterCardPlayed``：被打中的那张牌**结算完** → 获得 Amount 点格挡并移除。

    ⚠️ 时机是"牌结算完"而不是"挨打时"：所以它挡的是**下一张**牌 ——
    写成挨打当下就给格挡，等于把"蜷缩"提前了一个动作。
    """
    card = getattr(ctx, "card", None)
    if card is None:
        return KEEP
    if ctx.owner.power_flags.get("curl_up_card", 0) != getattr(card, "uid", 0):
        return KEEP
    from . import core
    core.gain_block(ctx.state, ctx.owner, amount, ctx.events, unpowered=True,
                    label="蜷缩")
    return REMOVE


def _slow_count(ctx: PowerContext, amount: int) -> str:
    """``SlowPower.AfterCardPlayed``：**每打出一张牌**累计 1 层 ``SlowAmount``。

    源码不看是谁打的（``AfterCardPlayed`` 对所有出牌都分发）—— 玩家打牌，
    怪物身上的"迟缓"就涨一格，它**受到的**攻击伤害就多 10%。
    """
    if ctx.owner.power_flags.get("slow_amount", 0) < 0:
        return KEEP
    ctx.owner.power_flags["slow_amount"] = ctx.owner.power_flags.get("slow_amount", 0) + 1
    return KEEP


def _slow_reset(ctx: PowerContext, amount: int) -> str:
    """``SlowPower.AfterSideTurnStart``：拥有者**自己**的回合开始 → ``SlowAmount`` 归零。

    也就是"玩家这一回合打了几张牌"只在这一回合内有效。
    """
    ctx.owner.power_flags["slow_amount"] = 0
    return KEEP


def _rebound_expire(ctx: PowerContext, amount: int) -> str:
    """``ReboundPower.AfterSideTurnEnd``：``participants.Contains(Owner)`` → 整条移除。

    ⚠️ 这是"没用上就作废"：`Rebound` 只在这一回合有效，没改写落点也要消失。
    少了它，弹回会**跨回合留着**（玩家凭空多一次"下张牌回抽牌堆顶"）。
    """
    return REMOVE


def _galvanic_played(ctx: PowerContext, amount: int) -> str:
    """``GalvanicPower.AfterCardPlayed``：打出带 ``Galvanized`` 的牌 → 拥有者受伤。

    源码::

        if (cardPlay.Card.Affliction is Galvanized)
            await CreatureCmd.Damage(…, base.Owner.Creature, base.Amount,
                                     ValueProp.Unpowered | ValueProp.Move, null, null);

    ⚠️ 这是"镀能"的**代价**（打能力牌要挨打），少了它这个能力只剩好处 ——
    引擎比真机**弱**，而且是"白送"的那种弱。伤害来源是 ``null``（无源），
    所以不吃力量/易伤，但**可以被格挡**。
    """
    card = getattr(ctx, "card", None)
    if card is None or getattr(card, "affliction", None) != "galvanized":
        return KEEP
    from . import core
    core.deal_damage(ctx.state, None, ctx.owner, amount, ctx.events, unpowered=True)
    return KEEP


def _ends_with_applier(ctx: PowerContext, amount: int) -> str:
    """``AfterDeath``：**施加者**死了 → 移除这条能力。

    源码（``ConstrictPower`` / ``ShrinkPower`` 都一样）::

        if (!wasRemovalPrevented && creature == base.Applier)
            await PowerCmd.Remove(this);

    引擎里施加者记在 ``Combatant.powers_applier[pid]``（``add_power`` 写入），
    所以规则要靠 ``ctx.pid`` 查"我自己是谁"。
    """
    applier = ctx.owner.powers_applier.get(getattr(ctx, "pid", ""))
    dead = getattr(ctx, "dead", None)
    return REMOVE if applier is not None and applier is dead else KEEP


def _hex_applier_death(ctx: PowerContext, amount: int) -> str:
    """``HexPower.AfterDeath``：施加者死了 → 移除自己**并清掉所有 ``Hexed``**。

    ⚠️ 两件事缺一不可：能力移除但苦痛还留在牌上，那些牌会**继续带虚无**
    （真机的 ``AfterRemoved`` 就是负责清苦痛的）。
    """
    if _ends_with_applier(ctx, amount) != REMOVE:
        return KEEP
    cleared = clear_affliction(ctx.state, "hexed")
    if cleared:
        ctx.events.append(f"六角消失：{cleared} 张牌的 Hexed 被清除")
    return REMOVE


def _mayhem(ctx: PowerContext, amount: int) -> str:
    """``MayhemPower.AfterAutoPrePlayPhaseEntered``：**自动打出抽牌堆顶的 Amount 张**。

    源码::

        if (player == base.Owner.Player)
            await CardPileCmd.AutoPlayFromDrawPile(choiceContext, base.Owner.Player,
                                                   base.Amount, CardPilePosition.Top,
                                                   forceExhaust: false);

    ``CardPilePosition.Top`` = 从**顶**开始抽着打（引擎里抽牌堆顶是列表尾部）；
    ``forceExhaust: false`` = 不额外消耗，落点由卡牌自己决定（带 ``Exhaust`` 的照旧消耗）。
    打出的每一张都走 ``CardCmd.AutoPlay`` 那条路（不付能量、``Before/AfterCardPlayed`` 照常分发）。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    from . import core
    for _ in range(max(1, amount)):
        if not ctx.state.draw_pile:
            break
        card = ctx.state.draw_pile.pop()            # 抽牌堆顶 = 列表尾部
        ctx.events.append(f"混乱：自动打出抽牌堆顶的 {card.definition().name}")
        # 挂在选牌上就停手（理由见 `_stampede` 的注释）。
        if not core.autoplay_card(ctx.state, card, ctx.events):
            break
    return KEEP


def _master_planner(ctx: PowerContext, amount: int) -> str:
    """``MasterPlannerPower.AfterCardPlayed``：玩家打出的**技能牌**获得 ``Sly``（战斗内）。

    源码::

        if (cardPlay.Card.Owner != base.Owner.Player) return;
        if (cardPlay.Card.Type != CardType.Skill) return;
        CardCmd.ApplyKeyword(cardPlay.Card, CardKeyword.Sly);

    ⚠️ 这是**卡实例级**关键字（打完才加，作用于这张牌**之后**被弃掉时）——
    所以要写进 ``CardInstance.keywords``，而读的那一侧（``_autoplay_sly`` 的判据）
    也必须看实例关键字，否则加了等于没加。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    if card.definition().card_type != "skill":
        return KEEP
    grant_keyword(ctx.state, card, "Sly", ctx.events)
    return KEEP


def _feral_applied(ctx: PowerContext, amount: int) -> str:
    """``FeralPower.AfterApplied``：把"本回合**已经**打过几张 0 费攻击牌"当作已用额度。

    源码::

        SetZeroCostAttacksPlayed(History.CardPlaysStarted.Count(e =>
            e.CardPlay.Card.Type == Attack && e.CardPlay.Player == Owner.Player
            && e.CardPlay.Resources.EnergyValue == 0 && e.HappenedThisTurn(CombatState)));

    少了它，战斗中途贴上"野性"会把**贴上之前**打过的 0 费攻击也当成额度，
    于是在本回合多发几张牌（静默变强）。
    """
    ctx.owner.power_flags["feral_returned"] = int(
        getattr(ctx.state, "zero_cost_attacks_played_this_turn", 0))
    return KEEP


def _feral_reset(ctx: PowerContext, amount: int) -> str:
    """``FeralPower.AfterSideTurnStart``：拥有者阵营回合开始 → 本回合已用额度归零。"""
    ctx.owner.power_flags["feral_returned"] = 0
    return KEEP


def one_for_all_bonus(attacker: "Combatant", card, state: "CombatState") -> int:
    """``OneForAllPower.ModifyDamageAdditive``：**0 费打出的攻击牌** +Amount。

    源码::

        if (!props.IsPoweredAttack()) return 0m;
        if (cardSource == null) return 0m;
        if (cardSource.Owner.Creature != base.Owner) return 0m;
        if (cardPlay?.Card.EnergyCost.CostsX ?? cardSource.EnergyCost.CostsX) return 0m;
        if (cardPlay != null && cardPlay.Resources.EnergySpent != 0) return 0m;
        if (cardPlay == null && cardSource.EnergyCost.GetWithModifiers(CostModifiers.All) != 0) return 0m;
        return base.Amount;

    也就是"这一张**实际没花能量**的攻击牌"（免费/0 费减费之后都算），
    X 费牌**不算**（真机专门排除了它）。
    """
    if card is None or state is None or card.definition().is_x_cost:
        return 0
    in_play = getattr(state, "current_play_energy_spent", -1)
    if in_play >= 0:
        # 正在这次出牌里：看**实际花了多少**（减费/免费能力都算进去）。
        if in_play != 0:
            return 0
    else:
        # 不在出牌里（延迟伤害这类）：退回到"这张牌的**带修正费用**是不是 0"。
        from . import core
        if core.play_cost(state, card) != 0:
            return 0
    return attacker.power("one_for_all")


def star_cost_zeroed(state: "CombatState", card) -> bool:
    """``Hook.ModifyStarCost``：有没有能力把这张牌的**星费**清零。

    真机只有 ``VoidFormPower`` 覆写它（``BrilliantScarf`` 是遗物，另走遗物侧），
    而且判据与它清零**能量**费用时**完全一样**（同一个 ``ShouldSkip(card)``）——
    所以这里直接复用 :func:`void_form_applies`，不另抄一份条件
    （抄两份的下场是"能量免费了但还要花星"这种半吊子）。
    """
    return void_form_applies(state, card)


def on_stars_spent(state: "CombatState", amount: int,
                   events: list[str]) -> None:
    """``Hook.AfterStarsSpent``（``CardModel.cs:1842``）：**真的花了星**之后分发。

    真机只在 ``amount > 0`` 时调（``SpendStars`` 里那个 if）——
    `ChildOfTheStarsPower` 因此不需要自己判 0，但引擎这边照抄那个 if 更保险。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_stars_spent", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx), amount=amount)


def on_stars_gained(state: "CombatState", amount: int,
                    events: list[str]) -> None:
    """``Hook.AfterStarsGained``（``PlayerCmd.cs:95``）：获得星之后分发。"""
    from . import hooks as hook_bus
    hook_bus.fire("on_stars_gained", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx), amount=amount)


def _child_of_the_stars(ctx: PowerContext, amount: int) -> str:
    """``ChildOfTheStarsPower.AfterStarsSpent``：**花星** → 获得 ``Amount × 星数`` 格挡。

    源码::

        if (amount > 0 && spender == base.Owner.Player)
            await CreatureCmd.GainBlock(base.Owner, base.Amount * amount,
                                        ValueProp.Unpowered, null);

    ``Unpowered``：这份格挡不吃敏捷、也不被脆弱打折（与"按打出的牌给格挡"那类不同）。
    """
    spent = int(getattr(ctx, "amount", 0) or 0)
    if spent <= 0 or ctx.owner is not ctx.state.player:
        return KEEP
    from . import core
    core.gain_block(ctx.state, ctx.owner, amount * spent, ctx.events,
                    unpowered=True, label="星辰之子")
    return KEEP


def _black_hole_hit(ctx: PowerContext, amount: int) -> str:
    """``BlackHolePower`` 的两半共用的群伤：对全体可打敌人造成 ``Amount`` 点 Unpowered 伤害。"""
    from . import core
    for enemy in hittable_enemies(ctx.state):
        core.deal_damage(ctx.state, ctx.owner, enemy, amount, ctx.events,
                         unpowered=True)
    return KEEP


def _black_hole_gained(ctx: PowerContext, amount: int) -> str:
    """``BlackHolePower.AfterStarsGained``：获得星（``amount > 0``）→ 群伤。"""
    if int(getattr(ctx, "amount", 0) or 0) <= 0:
        return KEEP
    if ctx.owner is not ctx.state.player:
        return KEEP
    return _black_hole_hit(ctx, amount)


def _black_hole_played(ctx: PowerContext, amount: int) -> str:
    """``BlackHolePower.AfterCardPlayed``：**花了星的那张牌打完** → 群伤。

    源码注释解释了为什么不用 ``AfterStarsSpent``：
    *"stars are spent at the beginning of the card play, but Black Hole should
    trigger after the card is played"* —— 星在出牌**开头**扣，而黑洞要等
    这张牌**结算完**才炸。判据是 ``Resources.StarsSpent > 0`` **且**
    ``CardPlay.IsLastInSeries``（多打一次的牌只在**最后一遍**触发）。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    if int(getattr(ctx.state, "current_play_stars_spent", 0) or 0) <= 0:
        return KEEP
    if not getattr(ctx, "is_last_in_series", False):
        return KEEP
    return _black_hole_hit(ctx, amount)


def _calamity_mark(ctx: PowerContext, amount: int) -> str:
    """``CalamityPower.BeforeCardPlayed``：记下"这张**攻击牌**开始结算时的层数"。

    与 `SubroutinePower` 同一个套路（源码都用 ``Dictionary<CardModel,int>``）：
    按**牌身份**记层数，避免同一张牌多打时重复触发。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    if card.definition().card_type != "attack":
        return KEEP
    ctx.owner.power_flags[f"calamity_card_{card.uid}"] = amount
    return KEEP


def _calamity_play(ctx: PowerContext, amount: int) -> str:
    """``CalamityPower.AfterCardPlayed``：那张攻击牌结算完 → 生成 Amount 张随机攻击牌进手牌。

    源码::

        if (!amountsForPlayedCards.Remove(cardPlay.Card, out var _)) return;
        list = CardFactory.GetForCombat(player, 牌池里 Type == Attack …, Amount,
                                        Rng.CombatCardGeneration);
        foreach (item in list) await CardPileCmd.AddGeneratedCardToCombat(item, Hand, player);

    ⚠️ 三点都要照抄：**只有记录过的牌**（也就是攻击牌）才触发、从**角色卡池**里抽
    **攻击牌**、走 ``combat_card_generation`` 随机流。少任何一条都会静默偏掉。
    """
    card = getattr(ctx, "card", None)
    if card is None:
        return KEEP
    recorded = ctx.owner.power_flags.pop(f"calamity_card_{card.uid}", None)
    if not recorded:
        return KEEP
    generate_cards_to_hand(ctx.state, max(1, int(recorded)), ctx.events,
                           card_type="attack")
    return KEEP


def _consuming_shadow(ctx: PowerContext, amount: int) -> str:
    """``ConsumingShadowPower.AfterSideTurnEnd``：拥有者回合结束 → 激发**最后一个球** Amount 次。

    源码::

        if (participants.Contains(base.Owner) && OrbQueue.Orbs.Count != 0)
            for (i < Amount) await OrbCmd.EvokeLast(choiceContext, player);

    ``EvokeLast`` = 激发 ``Orbs.Last()``（**最右边**那个）并把它移出队列 ——
    引擎的 ``orbs.evoke(state, len(state.orbs) - 1, ...)`` 正是同一件事。
    ⚠️ 判据是"进入循环**之前**队列非空"（不是每激发一次判一次）。
    """
    if ctx.owner is not ctx.state.player or not ctx.state.orbs:
        return KEEP
    from . import orbs as orb_rules
    for _ in range(max(1, amount)):
        if not ctx.state.orbs:
            break
        orb_rules.evoke(ctx.state, len(ctx.state.orbs) - 1, ctx.events)
    return KEEP


def _soulbound(ctx: PowerContext, amount: int) -> str:
    """``SoulboundPower.AfterCardGeneratedForCombat``：**施加者**造出 ``Soul`` → 再塞 Amount 张。

    源码::

        if (creator != null && creator.Creature == base.Applier && card is Soul && !IsAddingSoul) {
            IsAddingSoul = true;
            CardPileCmd.AddGeneratedCardsToCombat(Soul.Create(owner, Amount, state),
                                                  PileType.Draw, owner, Random);
            IsAddingSoul = false;
        }

    三个条件照抄：creator 是**把这条能力贴到我身上的那个玩家**（真机 ``base.Applier``，
    引擎记在 ``powers_applier``）、造出来的牌是 ``Soul``、并且**不是自己造的**
    （``IsAddingSoul`` 防自激 —— 少了它就是无限递归）。
    落点是**抽牌堆的随机位置**（``CardPilePosition.Random`` → ``Rng.Shuffle``）。
    """
    card = getattr(ctx, "card", None)
    creator = getattr(ctx, "creator", None)
    if card is None or getattr(card, "cid", "") != "soul":
        return KEEP
    applier = ctx.owner.powers_applier.get("soulbound")
    if applier is None or creator is not applier:
        return KEEP
    if ctx.owner.power_flags.get("soulbound_adding"):
        return KEEP                          # 防自激（真机 `IsAddingSoul`）
    from .core import CardInstance, _add_generated_card
    ctx.owner.power_flags["soulbound_adding"] = 1
    try:
        for _ in range(max(1, amount)):
            _add_generated_card(ctx.state, CardInstance("soul"), "draw",
                                position="random")
    finally:
        ctx.owner.power_flags["soulbound_adding"] = 0
    return KEEP


def slow_multiplier(holder: "Combatant") -> Decimal | None:
    """``SlowPower.ModifyDamageMultiplicative``：``1 + 0.1 × SlowAmount``。

    条件：``target == base.Owner``（打的是**它**）且 ``props.IsPoweredAttack()``
    （由调用路径保证：``Unpowered`` 的伤害走不到 ``compute_damage``）。
    """
    if holder.power("slow") <= 0:
        return None
    counted = holder.power_flags.get("slow_amount", 0)
    return Decimal(1) + Decimal("0.1") * counted


def damage_cap(holder: "Combatant") -> int | None:
    """``Hook.ModifyDamageCap``：每次伤害的上限（``HardToKillPower``）。

    源码::

        public override decimal ModifyDamageCap(Creature? target, …)
        {
            if (target != base.Owner) return decimal.MaxValue;
            return base.Amount;
        }

    ⚠️ 这是伤害管线的**第 3 步**（加法 → 乘法 → **钳制** → 落地），
    少这一步，"难以击杀"的怪会被一刀秒 —— 而它整条设计就是这个上限。
    ``target != Owner`` 时返回 ``decimal.MaxValue``（不设限），所以取**最小值**。
    """
    amount = holder.power("hard_to_kill")
    return amount if amount > 0 else None


def _double_damage_tick(ctx: PowerContext, amount: int) -> str:
    """``DoubleDamagePower.AfterSideTurnEnd``：``participants.Contains(Owner)`` → 递减 1 层。

    双倍伤害只持续到**自己回合结束**。少接这一条，`shadow_step` 给的
    "下回合双倍"会**永久留着** —— 静默变强，而且日志一切正常。
    """
    ctx.owner.add_power("double_damage", -1)
    return KEEP


def _colossus_tick(ctx: PowerContext, amount: int) -> str:
    """``ColossusPower.AfterSideTurnEnd``：``side == Enemy`` → 递减 1 层。

    ⚠️ 判据是"**敌方**阵营回合结束"，不是"拥有者阵营"（与易伤/虚弱/脆弱同一条），
    所以不能用 ``on_owner_turn_end`` 糊过去 —— 玩家身上的巨像减伤要在敌人回合
    打完之后才掉。
    """
    if _side_of(ctx) != "enemy":
        return KEEP
    ctx.owner.add_power("colossus", -1)
    return KEEP


def _intangible_tick(ctx: PowerContext, amount: int) -> str:
    """``IntangiblePower.AfterSideTurnEnd``：``side == Enemy`` → 递减 1 层。

    少接这一条，"无形"会永久把掉血压在 1 —— 整场战斗免疫，而且不报错。
    """
    if _side_of(ctx) != "enemy":
        return KEEP
    ctx.owner.add_power("intangible", -1)
    return KEEP


def _covered_expire(ctx: PowerContext, amount: int) -> str:
    """``CoveredPower.AfterSideTurnEnd``：``side == Enemy`` → 整条移除。"""
    return REMOVE if _side_of(ctx) == "enemy" else KEEP


# ==========================================================================
# Wave B · 第十二批：`set_power_var`（能力实例变量）+ `AfterBlockCleared`
# ==========================================================================
#: ``TheBombPower.CanonicalVars`` 的 ``DamageVar(40m, Unpowered)`` —— 卡牌没通过
#: ``SetDamage`` 告诉我们数值时的兜底（正常路径上 ``set_power_var`` 会写进来）。
THE_BOMB_DEFAULT_DAMAGE = 40


def _the_bomb(ctx: PowerContext, amount: int) -> str:
    """``TheBombPower.BeforeSideTurnEnd``：自家阵营回合结束前递减；数到 1 就炸。

    源码::

        if (!participants.Contains(base.Owner)) return;
        if (base.Amount > 1) { await PowerCmd.Decrement(this); return; }
        await CreatureCmd.Damage(…, base.CombatState.HittableEnemies,
                                 base.DynamicVars.Damage, base.Owner);
        await PowerCmd.Remove(this);

    ⚠️ 伤害量来自**能力实例变量** ``Damage``：卡牌 ``OnPlay`` 用
    ``SetDamage(DynamicVars["BombDamage"])`` 写进来（升级 +10 也跟着变）。
    只读 ``CanonicalVars`` 的 40 会让升级过的炸弹打出 40 而不是 50 —— 静默变弱。
    """
    if _owner_side(ctx) != _side_of(ctx):
        return KEEP
    if amount > 1:
        # ⭐ **定向递减**：减的是**本实例**（`ctx.instance`）。两个炸弹各自倒计时，
        # 减到"最后一个"身上会让先放的那个永远不炸（而日志正常）。
        ctx.owner.add_power("the_bomb", -1, instance=ctx.instance)
        return KEEP
    from . import core
    # ⚠️ 读**本实例**的变量：一张升级过、一张没升级的两颗炸弹伤害不同，
    # 共用一个 `power_vars` 会让后者用前者的数值。
    damage = ctx.instance_vars.get("Damage", THE_BOMB_DEFAULT_DAMAGE)
    for enemy in hittable_enemies(ctx.state):
        core.deal_damage(ctx.state, ctx.owner, enemy, damage, ctx.events,
                         unpowered=True)
    ctx.events.append(f"炸弹引爆：全体 {damage} 点（Unpowered）")
    return REMOVE


def _toric_toughness(ctx: PowerContext, amount: int) -> str:
    """``ToricToughnessPower.AfterBlockCleared``：自己的格挡被清空 → 按**记录的数值**
    给格挡（Unpowered），然后递减 1 层。

    源码::

        public void SetBlock(decimal block) { base.DynamicVars.Block.BaseValue = block; }
        // AfterBlockCleared:
        if (creature == base.Owner) {
            await CreatureCmd.GainBlock(base.Owner, base.DynamicVars.Block, null);
            await PowerCmd.Decrement(this);
        }

    ⚠️ ``SetBlock`` 收到的是**上一次 ``GainBlock`` 的返回值**（卡牌把"实际获得的格挡"
    写进来），不是卡面那个 5 —— 抽取器把它标成 ``from="last_block_gain"``。
    少这一层，这张卡在"有敏捷/脆弱/倍率"的局面里数值全错。
    """
    if getattr(ctx, "creature", None) is not ctx.owner:
        return KEEP
    from . import core
    # ⭐ 读**本实例**的变量（`SetBlock` 只写它自己那一份）。
    block = ctx.instance_vars.get("Block", 0)
    core.gain_block(ctx.state, ctx.owner, block, ctx.events, unpowered=True,
                    label="环面韧性")
    ctx.owner.add_power("toric_toughness", -1, instance=ctx.instance)
    return KEEP


def _self_forming_clay(ctx: PowerContext, amount: int) -> str:
    """``SelfFormingClayPower.AfterBlockCleared``：自己的格挡被清空 → 获得 Amount 点
    格挡（``ValueProp.Unpowered``）并整条移除。

    ⚠️ 施加者是**遗物** ``SelfFormingClay``（``AfterDamageReceived`` 且真的掉血时贴），
    那个遗物还没抽出来 —— 能力本身先按源码实现，等遗物侧接上就会自动生效。
    """
    if getattr(ctx, "creature", None) is not ctx.owner:
        return KEEP
    from . import core
    core.gain_block(ctx.state, ctx.owner, amount, ctx.events, unpowered=True,
                    label="自成形黏土")
    return REMOVE



# ==========================================================================
# Wave B · 第八批：把「出牌计数 / 回合开始 / 同伴伤害」这三类实例数据接上
# ==========================================================================
#: ``RollingBoulderPower.CanonicalVars`` 里的 ``DamageVar(5m, Unpowered)``：
#: 每回合把 Amount 加上这个值（``SetAmount(base.Amount + base.DynamicVars.Damage.IntValue)``）。
#: 它是**能力自己的常量**，不是卡牌动态变量 —— 升级卡牌改的是起始层数。
ROLLING_BOULDER_INCREMENT = 5
#: ``WitheringPresencePower`` 的 ``CardsLeft`` 初值（``CanonicalVars``，常量 6）。
WITHERING_PRESENCE_CARDS = 6


def _subroutine_record(ctx: PowerContext, amount: int) -> str:
    """``SubroutinePower.BeforeCardPlayed``：记下"这张**能力牌**开始结算时的层数"。

    源码用 ``Dictionary<CardModel,int>`` 按**牌模型**记层数，理由写在注释里：
    避免"在它被贴上之前就开始结算的牌"也拿能量，也避免同一张牌多打时重复给。
    引擎用牌的 ``uid`` 当键（同一个身份，SL/快照下稳定），存在 ``power_flags``。
    """
    card = getattr(ctx, "card", None)
    if card is None or ctx.owner is not ctx.state.player:
        return KEEP
    if str(getattr(card.definition(), "card_type", "")) != "power":
        return KEEP
    ctx.owner.power_flags[f"subroutine_card_{card.uid}"] = amount
    return KEEP


def _subroutine_energy(ctx: PowerContext, amount: int) -> str:
    """``SubroutinePower.AfterCardPlayed``：结算完那张能力牌 → 还回记录时的层数点能量。

    ⚠️ 用的是**记录时**的层数，不是当前层数（``Remove(card, out var energy)``）；
    没记录过（不是能力牌 / 在贴上之前就打出的牌）→ 一点也不给。
    """
    card = getattr(ctx, "card", None)
    if card is None:
        return KEEP
    energy = ctx.owner.power_flags.pop(f"subroutine_card_{card.uid}", 0)
    for _ in range(max(0, energy)):
        ctx.state.energy += 1
    if energy > 0:
        ctx.events.append(f"子程序：返还 {energy} 点能量")
    return KEEP


def _rolling_boulder(ctx: PowerContext, amount: int) -> str:
    """``RollingBoulderPower.AfterPlayerTurnStart``：对**所有可打的敌人**造成 Amount 点
    Unpowered 伤害，然后把 Amount 加 5（越滚越大），永不消耗。

    ``CreatureCmd.Damage(..., HittableEnemies, base.Amount, ValueProp.Unpowered, Owner)``
    —— 伤害来源是拥有者，所以荆棘/受击类能力照样响应（但数值不吃力量/易伤）。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    from . import core
    for enemy in hittable_enemies(ctx.state):
        core.deal_damage(ctx.state, ctx.owner, enemy, amount, ctx.events,
                         unpowered=True)
    ctx.owner.add_power("rolling_boulder", ROLLING_BOULDER_INCREMENT)
    ctx.events.append(f"滚石：伤害提升到 "
                      f"{ctx.owner.power('rolling_boulder')}")
    return KEEP


def _withering_presence(ctx: PowerContext, amount: int) -> str:
    """``WitheringPresencePower.AfterCardPlayed``：**目标玩家**每打一张牌，
    ``CardsLeft`` 减 1；减到 0 就往他手里塞 1 张 ``Wither`` 并把计数复位成 6。

    ⚠️ ``Amount`` 不参与这段逻辑（源码只用它当层数显示），真正驱动的是
    ``DynamicVars["CardsLeft"]``。真机比的是 ``cardPlay.Card.Owner == base.Target.Player``：
    这条能力挂在怪物身上，``Target`` 就是玩家 —— 单人局里"玩家打出的牌"与它等价。
    """
    card = getattr(ctx, "card", None)
    if card is None:
        return KEEP
    flags = ctx.owner.power_flags
    left = flags.get("withering_cards_left", WITHERING_PRESENCE_CARDS)
    left -= 1
    if left > 0:
        flags["withering_cards_left"] = left
        return KEEP
    flags["withering_cards_left"] = WITHERING_PRESENCE_CARDS
    add_copies_to_hand(ctx.state, "wither", 1, ctx.events)
    ctx.events.append("凋零笼罩：塞入 1 张「凋零」，计数复位")
    return KEEP


def _shadow_step(ctx: PowerContext, amount: int) -> str:
    """``ShadowStepPower.AfterSideTurnStart``：拥有者阵营回合开始 → 给自己
    ``Amount`` 层**双倍伤害**，然后整条移除。

    ``participants.Contains(base.Owner)`` 由引擎的 ``on_owner_turn_start``
    （拥有者所在阵营回合开始）天然表达。
    """
    if ctx.owner is not ctx.state.player:
        return KEEP
    ctx.owner.add_power("double_damage", amount)
    ctx.events.append(f"影步：获得 {amount} 层双倍伤害")
    return REMOVE


def _underworld(ctx: PowerContext, amount: int) -> str:
    """``UnderworldPower.AfterDamageGiven``：**同伴**（同阵营的非自己、非自己随从）
    打出有效攻击且总伤害 > 0 → 给被打的目标上 ``总伤害 × Amount`` 层**末日**。

    ``props.IsPoweredAttack()`` = 引擎的 ``powered``；``result.TotalDamage`` =
    ``ctx.total``（**含被格挡的部分**，用 ``unblocked`` 会少上很多层）。
    源码还有一条 ``dealer.PetOwner != base.Owner.Player``（别把"自己的随从"算成同伴）——
    引擎没有随从子系统（Osty 未实现），该条在单人局里恒成立。
    """
    dealer = getattr(ctx, "dealer", None)
    target = getattr(ctx, "target", None)
    total = int(getattr(ctx, "total", 0) or 0)
    if not getattr(ctx, "powered", True) or total <= 0:
        return KEEP
    if dealer is None or target is None:
        return KEEP
    if dealer is ctx.owner or not _same_side(ctx.owner, dealer, ctx.state):
        return KEEP
    target.add_power("doom", total * amount, applier=ctx.owner)
    ctx.events.append(f"冥界：{target.name} 被追加 {total * amount} 层末日")
    return KEEP


def _underworld_expire(ctx: PowerContext, amount: int) -> str:
    """``UnderworldPower.AfterSideTurnEnd``：``side == Enemy`` → 整条移除。"""
    return REMOVE if _side_of(ctx) == "enemy" else KEEP


#: ``OrbitPower`` 的触发阈值（``private const int _energyIncrement = 4``，
#: 同时也是 ``CanonicalVars`` 里的 ``EnergyVar(4)``）：每累计花掉 4 点能量触发一次，
#: 每次回**层数**点能量。
ORBIT_ENERGY_PER_TRIGGER = 4


def _orbit(ctx: PowerContext, amount: int) -> str:
    """``OrbitPower.AfterEnergySpent``：自己**打出卡牌**花掉能量后累计，每满 4 点回
    ``Amount`` 点能量。

    源码::

        data.energySpent += amount;
        int triggers = data.energySpent / 4 - data.triggerCount;
        if (triggers > 0) { await PlayerCmd.GainEnergy(base.Amount * triggers, ...); ... }

    ⚠️ 按 ``triggerCount``（**已触发次数**）而不是按"花掉的能量取模"记账：
    层数在两次触发之间变了也不会补发/漏发。
    """
    spent = int(getattr(ctx, "amount", 0) or 0)
    if spent <= 0 or ctx.owner is not ctx.state.player:
        return KEEP
    flags = ctx.owner.power_flags
    total = flags.get("orbit_energy_spent", 0) + spent
    flags["orbit_energy_spent"] = total
    triggers = total // ORBIT_ENERGY_PER_TRIGGER - flags.get("orbit_triggers", 0)
    if triggers > 0:
        flags["orbit_triggers"] = flags.get("orbit_triggers", 0) + triggers
        ctx.state.energy += amount * triggers
        ctx.events.append(f"环绕：花满能量 → 回 {amount * triggers} 点")
    return KEEP


def block_additive_modifiers(owner: "Combatant", state: "CombatState | None" = None,
                             card=None) -> list[tuple[str, int]]:
    """``ModifyBlockAdditive`` 里**带条件**的那一族（今天只有 ``FastenPower``）。

    源码（``FastenPower.cs:17-30``）::

        if (base.Owner != target) return 0m;                      // 只有自己获得格挡才算
        if (!props.IsPoweredCardOrMonsterMoveBlock()) return 0m;   // 由"非 unpowered"保证
        if (cardSource != null && !cardSource.Tags.Contains(CardTag.Defend)) return 0m;
        return base.Amount;

    ⚠️ 最后一条是**标签**条件：卡牌来源必须是带 ``Defend`` 标签的牌；
    ``cardSource == null``（怪物招式 / 充能球这类没有牌来源的格挡）反而**照样加**。
    少这一条会让"连接"变成"所有格挡都 +Amount"（静默变强）。
    """
    if state is None:
        return []
    tags: tuple = ()
    if card is not None:
        definition = card.definition() if hasattr(card, "definition") else None
        tags = tuple(getattr(definition, "tags", ()) or ())
    out: list[tuple[str, int]] = []
    for holder in (state.player, *state.enemies):
        if not holder.alive() or holder is not owner:
            continue
        amount = holder.power("fasten")
        if amount <= 0:
            continue
        if card is not None and "Defend" not in tags:
            continue
        out.append(("fasten", amount))
    return out


def _blur_tick(ctx: PowerContext, amount: int) -> str:
    """``BlurPower.AfterSideTurnStart``：拥有者阵营回合开始 → 递减 1 层。

    "格挡不清空"那一半不在这里：它由 :data:`BLOCK_CLEAR_PREVENTERS`（``ShouldClearBlock``
    查询）表达，时机在**清格挡那一步**（``Creature.AfterTurnStart``），比这里早。
    源码自带的 ``AfterPreventingBlockClear`` 覆写是**空实现**（只判 ``this != preventer``
    就返回），所以不需要为它加钩子。
    """
    ctx.owner.add_power("blur", -1)
    return KEEP


def shadowmeld_multiplier(owner: "Combatant") -> Decimal:
    """``ShadowmeldPower.ModifyBlockMultiplicative``：拥有者获得的格挡 ×``2^Amount``。

    源码只有一条判据 —— ``if (base.Owner != target) return 1m;`` —— **不看 props、
    也不看来源**。所以 Unpowered 的来源（药水 / 充能球）同样要乘：调用方
    :func:`sts2_sim.core.gain_block` 在 ``unpowered`` 分支里也乘这个倍率，
    只把它塞进 ``compute_block`` 会漏掉那一条（引擎比真机**弱**）。
    """
    amount = owner.power("shadowmeld")
    if amount <= 0:
        return Decimal(1)
    return Decimal(2) ** amount


def _shadowmeld_expire(ctx: PowerContext, amount: int) -> str:
    """``ShadowmeldPower.AfterSideTurnEnd``：``participants.Contains(Owner)`` → 整条移除。"""
    return REMOVE


def _thunder(ctx: PowerContext, amount: int) -> str:
    """``ThunderPower.AfterOrbEvoked``：激发的球是**闪电**且属于自己时，
    对**这次激发的目标**再造成 ``Amount`` 点 Unpowered 伤害。

    源码 ``CreatureCmd.Damage(..., livingTargets, base.Amount, Unpowered, base.Owner)``
    —— 只打**还活着**的目标（激发本身可能已经把它打死了），且伤害不带力量/易伤。
    """
    orb = getattr(ctx, "orb", None)
    if orb is None or getattr(orb, "oid", "") != "lightning":
        return KEEP
    from . import core
    for target in getattr(ctx, "targets", ()) or ():
        if target.alive():
            core.deal_damage(ctx.state, ctx.owner, target, amount, ctx.events,
                             unpowered=True)
    return KEEP


def on_energy_spent(state: "CombatState", card, amount: int,
                    events: list[str]) -> None:
    """``Hook.AfterEnergySpent``：**打出卡牌**付掉能量之后分发（``OrbitPower`` 用）。

    真机的钩子签名是 ``AfterEnergySpent(CardModel card, int amount)`` —— 只有"卡牌的
    费用"才有卡对象，所以药水/效果扣能量不在这里；调用点是 ``play_card`` 的扣费之后。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_energy_spent", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx),
                  card=card, amount=amount)


def on_orb_evoked(state: "CombatState", orb, targets, events: list[str]) -> None:
    """``Hook.AfterOrbEvoked``（``OrbCmd.cs:147``）：激发一个球之后，
    把**这次激发的目标**一起分发（``ThunderPower`` 用）。"""
    from . import hooks as hook_bus
    hook_bus.fire("on_orb_evoked", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx),
                  orb=orb, targets=tuple(targets or ()))


def on_block_cleared(creature: "Combatant", state: "CombatState",
                     events: list[str]) -> None:
    """``Hook.AfterBlockCleared``（``CombatManager.cs:768``）：某个单位的格挡被清空之后。

    真机对**每个开始回合的单位**都分发一次（``foreach (creature in
    creaturesStartingTurn) Hook.AfterBlockCleared(state, creature)``），
    而且**不看它到底有没有格挡**（玩家第 1 回合不清格挡也照样分发）。
    所以 ``creature`` 必须传下去：``ToricToughnessPower`` / ``SelfFormingClayPower``
    的判据都是 ``creature == base.Owner``。
    """
    _fire(creature, state, events, "on_block_cleared", creature=creature)


def on_before_death(state: "CombatState", dead: "Combatant",
                    events: list[str]) -> None:
    """``Hook.BeforeDeath``：某个单位**即将死亡**（``mark_dead`` 的第一件事）。

    与 ``on_any_death``（死后那三段落）分开：这一段要"死之前还来得及做事"——
    `SwipePower` / `HeistPower` 在这里把偷来的东西还给玩家。
    参数 ``dead`` 是那个要死的单位，规则自己比 ``ctx.owner is ctx.dead``。

    ⚠️ 必须先于"标记已死"分发：规则可能要读它身上的能力层数/实例数据，
    标记之后再读就得靠"死亡也保留数据"这种额外约定。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_before_death", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx), dead=dead)


def on_card_entered_combat(state: "CombatState", events: list[str], card) -> None:
    """``Hook.AfterCardEnteredCombat``（``CardPileCmd.cs:515``）：一张牌**刚进入战斗**。

    真机的判据是 ``oldPile == null && cardPile.IsCombatPile`` —— 也就是
    **新造出来的牌**（此前没有任何牌堆）。同一场战斗内"从弃牌堆回手牌"这类
    **换牌堆**不触发（``CardPileCmd.cs:372`` 的注释专门点了这一条）。
    引擎的等价物是 :func:`sts2_sim.core._add_generated_card`（生成牌的**唯一**出口）。

    苦痛族六个能力靠它给**战斗中途生成的牌**补上苦痛；少了这一半，
    "生成一张攻击牌"在 ``tangled`` 下不会被缠住（引擎比真机**弱**，而且不报错）。
    """
    _fire(state.player, state, events, "on_card_entered_combat", card=card)


def on_auto_pre_play(state: "CombatState", events: list[str]) -> None:
    """``Hook.AfterAutoPrePlayPhaseEntered``（``CombatManager.cs:867``）。

    真机的阶段顺序是 ``Start → AutoPrePlay → Play → AutoPostPlay``：
    这个时机在**抽完牌、即将进入出牌阶段**那一刻（``RunAutoPrePlayPhase``
    把阶段设成 `AutoPrePlay`、跑完这个钩子、再设成 `Play`）。
    `MayhemPower` 在这里自动打出抽牌堆顶的 Amount 张。

    ⚠️ 别和 :func:`on_auto_post_play` 混：那个是**回合结束**进入 `AutoPostPlay`
    的时机（`StampedePower` 用），两者差着整个出牌阶段。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_auto_pre_play", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx))


def on_ally_damage_given(state: "CombatState", dealer: "Combatant | None",
                         target: "Combatant | None", total: int,
                         powered: bool, events: list[str]) -> None:
    """``Hook.AfterDamageGiven`` 的**第二条调用点**：通知"**不是经手者**的持有者"。

    为什么不能挂在 :func:`on_damage_given` 上：那一条按真机的 ``scope=owner``
    只通知**经手者自己**的能力（``PainfulStabsPower`` / ``SuckPower`` 都是
    "自己打出去"才算）。而 ``UnderworldPower`` 的判据是 ``dealer.Side == Owner.Side
    && dealer != Owner`` —— **别人**打的伤害我要响应，所以必须再通知一遍全体，
    并把 ``dealer`` 放进上下文（见 ``docs/13`` §19 的"补调用点"这一类）。
    """
    if dealer is None or target is None:
        return
    for holder in (state.player, *state.enemies):
        if not holder.alive() or holder.power("underworld") <= 0:
            continue
        if holder is dealer or not _same_side(holder, dealer, state):
            continue
        context = PowerContext(state=state, owner=holder, events=events)
        context.dealer = dealer
        context.target = target
        context.total = total
        context.powered = powered
        if _underworld(context, holder.power("underworld")) == REMOVE:
            holder.powers.pop("underworld", None)


RULES: dict[str, PowerRules] = {
    rules.pid: rules for rules in (
        PowerRules("fan_of_knives", "FanOfKnivesPower：标记由 Shiv.TargetType/OnPlay "
                   "消费；core.card_target 与出牌执行共同实现动态群攻"),
        PowerRules("energy_next_turn", "EnergyNextTurnPower.AfterEnergyReset",
                   on_energy_reset=_energy_next_turn),
        PowerRules("draw_cards_next_turn", "DrawCardsNextTurnPower.AfterHandDraw",
                   on_energy_reset=_draw_cards_next_turn),
        PowerRules("block_next_turn", "BlockNextTurnPower.AfterSideTurnStart",
                   on_owner_turn_start=_block_next_turn),
        PowerRules("star_next_turn", "StarNextTurnPower.AfterSideTurnStart",
                   on_owner_turn_start=_star_next_turn),
        PowerRules("poison", "PoisonPower.AfterSideTurnStart",
                   on_owner_turn_start=_poison),
        # ---- KnowledgeDemon 的四种「知识诅咒」（docs/17 P01–P04，W01）--------
        PowerRules("mind_rot", "MindRotPower.ModifyHandDraw："
                   "Math.Max(0, count - Amount)（MindRotPower.cs:20）；"
                   "链式修正在 powers.modify_hand_draw"),
        PowerRules("waste_away", "WasteAwayPower.ModifyMaxEnergy：amount - Amount"
                   "（WasteAwayPower.cs:22）；链式修正在 powers.max_energy_bonus"
                   "（可为负 —— 源码此处不截零）"),
        PowerRules("sloth", "SlothPower：BeforeCardPlayed 累计私有出牌数、"
                   "BeforeSideTurnStart 归零、ShouldPlay 用「已出牌数 < 层数」限制出牌"
                   "（SlothPower.cs:22-51）",
                   on_before_card_played=_sloth_card_played,
                   on_before_side_turn_start=_sloth_reset),
        PowerRules("disintegration", "DisintegrationPower.AfterSideTurnEndLate："
                   "自己阵营结束时对自己造成 Amount 点 Unpowered 伤害"
                   "（dealer 是自己，格挡可吸收），层数不递减",
                   on_side_turn_end_late=_disintegration),
        # ---- W02（docs/17 P05–P08）------------------------------------------
        PowerRules("stratagem", "StratagemPower.AfterShuffle：洗完牌后从抽牌堆挑"
                   "Amount 张**移入**手牌（不是抽牌）；能力保留、层数不减。"
                   "抽牌循环会在洗牌处挂起，续跑见 core.DrawFrame",
                   on_after_shuffle=_stratagem),
        PowerRules("foregone_conclusion", "ForegoneConclusionPower.BeforeHandDraw："
                   "抽牌前 ShuffleIfNecessary → 从抽牌堆挑 Amount 张入手 → "
                   "**整个移除**自己（选完才移除；洗牌被别人挂住时排队等）",
                   on_before_hand_draw=_foregone_conclusion),
        PowerRules("nightmare", "NightmarePower.BeforeHandDraw（Instanced）：把卡牌"
                   "`SetSelectedCard` 存的快照克隆 Amount 份入手，然后**只移除这一个实例**"
                   "（S01 的实例表 + 定向 Remove）",
                   on_before_hand_draw=_nightmare),
        PowerRules("entropy", "EntropyPower.AfterPlayerTurnStart：从手牌挑 Amount 张"
                   "**原位随机转化**（CardCmd.TransformToRandom；候选池按 CardFactory "
                   "的过滤链 + 引擎准入预缩池）；能力不移除，层数 = 每回合转几张",
                   on_after_player_turn_start=_entropy),
        # ---- W03（docs/17 P09–P13）------------------------------------------
        PowerRules("parry", "ParryPower：**纯标记、无自身行为**（源码类的注释写明"
                   "「doesn't actually do anything on its own」）——"
                   "`SovereignBlade.GainsBlock` / `CalculatedBlockVar` 读它的层数"
                   "（`GetOwnerParryAmount` → `self_power` 运行期公式），层数 = 格挡"),
        PowerRules("seeking_edge", "SeekingEdgePower：**纯标记**（StackType=Single）——"
                   "`SovereignBlade.TargetType` 读它把单体改成全体；"
                   "`core.card_target` 是唯一判据，伤害落点跟着一起改"),
        PowerRules("confused", "ConfusedPower.AfterCardDrawn：拥有者抽牌时随机设为 0..3 费，"
                   "持续本场战斗；X 费消耗随机数但仍花全部能量",
                   on_card_drawn=_confused),
        PowerRules("demise", "DemisePower.AfterSideTurnEnd：本阵营结束受到 Amount "
                   "点 Unblockable | Unpowered 伤害，不递减",
                   on_side_turn_end=_demise),
        PowerRules("plating", "PlatingPower：回合结束获得 Amount 点格挡（Unpowered），"
                              "回合开始减 1 层（玩家第 1 回合 / 敌人第 1 轮豁免）",
                   on_owner_turn_end=_plating,
                   on_owner_turn_start=_plating_tick),
        PowerRules("regen", "RegenPower.BeforeSideTurnEndEarly：回复 Amount 点生命后递减 1 层",
                   on_owner_turn_end=_regeneration),
        PowerRules("thorns", "ThornsPower.AfterDamageReceived",
                   on_attacked=_thorns),
        # `no_draw` / `retain_hand`：行为在引擎的抽牌与清手牌路径上查询
        # （`blocks_draw` / `retains_hand`），这里只登记"回合结束怎么办"。
        PowerRules("no_draw", "NoDrawPower：不抽牌；AfterSideTurnEnd 整个移除",
                   on_owner_turn_end=_no_draw_expire),
        PowerRules("retain_hand", "RetainHandPower：不清手牌；AfterSideTurnEnd 递减",
                   on_owner_turn_end=_retain_hand_expire),
        # 查询型，行为在伤害管线/`AfterSideTurnEnd` 上：
        PowerRules("intangible",
                   "IntangiblePower.ModifyHpLostAfterOsty：掉血压到 1；"
                   "敌方回合结束递减（`AfterSideTurnEnd` 的 `side == Enemy`）",
                   on_side_turn_end=_intangible_tick),
        PowerRules("doom", "DoomPower.IsOwnerDoomed：生命 ≤ 层数时处决"),
        # 神器：行为在 `apply_power` 的路径上查询（`negates_debuff`），
        # 没有自己的回合钩子。
        PowerRules("artifact", "ArtifactPower.TryModifyPowerAmountReceived：抵消一次 Debuff"),
        PowerRules("territorial", "TerritorialPower.AfterSideTurnEnd：+Amount 力量",
                   on_owner_turn_end=_territorial),
        PowerRules("asleep",
                   "AsleepPower：**掉血就醒**（移除镀层 + 击晕走唤醒动作 + 下一招固定猛击）；"
                   "否则回合结束递减，归零醒来；倒计时到 1 时先扔掉镀层"
                   "（`BeforeSideTurnEndVeryEarly`）",
                   on_owner_turn_end=_asleep,
                   on_attacked=_asleep_wake,
                   on_before_side_turn_end=_asleep_plating),
        PowerRules("slumber",
                   "SlumberPower：**掉血递减**，归零则醒（击晕走唤醒动作 + 下一招固定滚撞）；"
                   "回合结束也递减，归零醒来（`WakeUpMove` 里的「移除镀层」同样照做）",
                   on_owner_turn_end=_slumber,
                   on_attacked=_slumber_wake),
        # 查询型，行为在伤害管线上：
        PowerRules("hard_to_kill",
                   "HardToKillPower.ModifyDamageCap：打**它**的伤害每次最多 Amount 点"
                   "（管线第 3 步「钳制」；消费者 `powers.damage_cap`）"),
        PowerRules("slow",
                   "SlowPower：每打出一张牌累计 1 格，拥有者**受到的**有效攻击"
                   " ×(1 + 0.1 × 已出牌数)；拥有者自己回合开始归零"
                   "（消费者 `powers.slow_multiplier`）",
                   on_card_played=_slow_count,
                   on_owner_turn_start=_slow_reset),
        PowerRules("curl_up",
                   "CurlUpPower：被**牌**打中时记下那张牌，等它**结算完** → 获得"
                   " Amount 点格挡（Unpowered）并移除",
                   on_attacked=_curl_up_mark,
                   on_card_played=_curl_up_play),
        # 查询型，行为在"清格挡"那一步：
        PowerRules("barricade", "BarricadePower.ShouldClearBlock → false：格挡永不清除"),
        PowerRules("burrowed",
                   "BurrowedPower.ShouldClearBlock → false：这只怪的格挡永不清除"),
        # ⭐ 敌方自我强化类（回合结束加力量）
        PowerRules("ritual", "RitualPower.AfterSideTurnEnd：回合结束 +Amount 力量"
                             "（刚获得的那一回合跳过）",
                   on_owner_turn_end=_ritual),
        PowerRules("high_voltage", "HighVoltagePower.AfterSideTurnEnd：回合结束 +Amount 力量",
                   on_owner_turn_end=_high_voltage),
        # ⭐ 倒计时类
        PowerRules("escape_artist", "EscapeArtistPower：纯视觉倒计时，无战斗效果",
                   on_owner_turn_end=_escape_artist),
        PowerRules("battleworn_dummy_time_limit",
                   "BattlewornDummyTimeLimitPower：倒计时归零 → CreatureCmd.Escape",
                   on_owner_turn_end=_battleworn_dummy_time_limit),
        # ⭐ 防御类
        PowerRules("slippery", "SlipperyPower.ModifyHpLostAfterOsty：掉血压到 1；"
                               "受未格挡伤害减 1 层",
                   on_attacked=_slippery_dec),
        PowerRules("skittish", "SkittishPower.AfterAttack：每回合一次，被卡牌打出未格挡伤害"
                               "→ 获得 Amount 点格挡（Unpowered）",
                   on_attacked=_skittish),
        PowerRules("rampart", "RampartPower.AfterSideTurnStart：玩家回合开始时，"
                              "己方炮台获得 Amount 点格挡（Unpowered）",
                   on_player_turn_start=_rampart),
        # ⭐ 触发类
        PowerRules("enrage", "EnragePower.AfterCardPlayed：玩家打出技能牌 → +Amount 力量",
                   on_card_played=_enrage),
        PowerRules("crab_rage", "CrabRagePower.AfterDeath：同阵营队友阵亡 → "
                                "+6 力量 +99 格挡，然后移除自身",
                   on_ally_death=_crab_rage),
        PowerRules("minion", "MinionPower：次级敌人（主敌人死亡时一并倒下）"),
        # ⭐ 减伤族（乘法阶段在 `DAMAGE_MULTIPLIERS` 里）
        PowerRules("flutter", "FlutterPower：受有效攻击伤害 ×0.50，每次有效伤害减 1 层",
                   on_attacked=_flutter),
        PowerRules("soar", "SoarPower.ModifyDamageMultiplicative：受有效攻击伤害 ×0.50"),
        PowerRules("shrink", "ShrinkPower：**自己造成**的伤害 ×0.70，回合结束减 1 层"
                             "（层数为负 = 永久）；**施加者死了**整条移除",
                   on_owner_turn_end=_shrink_tick,
                   on_applier_death=_ends_with_applier),
        PowerRules("surrounded",
                   "SurroundedPower：只在**背后**被打时 ×1.5（朝向由打哪一侧决定）"),
        # ⭐ 纯标记能力：自身没有钩子，供别的机制查询（`SurroundedPower` 查它们）
        PowerRules("back_attack_left", "BackAttackLeftPower：纯标记（从左侧攻击）"),
        PowerRules("back_attack_right", "BackAttackRightPower：纯标记（从右侧攻击）"),
        # ⭐ 倒计时 / 开关
        PowerRules("hatch", "HatchPower.AfterSideTurnEnd：回合结束减 1 层",
                   on_owner_turn_end=_hatch_tick),
        PowerRules("nemesis", "NemesisPower.AfterSideTurnEnd：隔回合切换无形",
                   on_owner_turn_end=_nemesis),
        PowerRules("constrict", "ConstrictPower.AfterSideTurnEnd：回合结束受到"
                                " Amount 点无源伤害（Unpowered，可格挡）；"
                                "**施加者死了**整条移除（否则那条蛇死了还在缠）",
                   on_owner_turn_end=_constrict,
                   on_applier_death=_ends_with_applier),
        PowerRules("painful_stabs", "PainfulStabsPower.AfterAttack：自己打出有效攻击"
                                    "→ 往玩家弃牌堆塞伤口",
                   on_damage_given=_painful_stabs),
        PowerRules("personal_hive", "PersonalHivePower.AfterDamageReceived：被有效攻击"
                                    "打中 → 往玩家抽牌堆**随机**塞眩晕",
                   on_attacked=_personal_hive),
        # ⭐ 眩晕 / 召唤族（共用 `powers.stun` 与 `powers.spawn_enemy` 两个原语）
        PowerRules("imbalanced", "ImbalancedPower.AfterDamageGiven：自己打出的伤害被"
                                 "**完全格挡** → 自身失衡（盛碗虫转 DIZZY_MOVE）",
                   on_damage_given=_imbalanced),
        PowerRules("suck", "SuckPower.AfterAttack：自己打出有效攻击且造成掉血"
                           "→ +Amount 力量",
                   on_damage_given=_suck),
        PowerRules("shriek", "ShriekPower.AfterDamageReceived：掉血后血量 ≤ 层数"
                             "→ 击晕自己并移除",
                   on_attacked=_shriek),
        PowerRules("plow", "PlowPower.AfterDamageReceived：掉血后血量 ≤ 层数"
                           "→ 清除临时力量并击晕自己",
                   on_attacked=_plow),
        PowerRules("ravenous", "RavenousPower.AfterDeath：队友阵亡 → 击晕自己"
                               "并 +Amount 力量（吞食同伴）",
                   on_ally_death=_ravenous),
        PowerRules("stock", "StockPower.AfterDeath：自己死亡时补一只（层数 -1）",
                   on_self_death=_stock),
        PowerRules("surprise", "SurprisePower.AfterDeath：自己死亡时召唤"
                               "胖地精 + 鬼祟地精",
                   on_self_death=_surprise),
        PowerRules("infested", "InfestedPower.AfterDeath：自己死亡时召唤 4 只"
                               "眩晕的扭动虫",
                   on_self_death=_infested),
        # ⭐ 掉血上限 / 削上限
        PowerRules("hardened_shell", "HardenedShellPower：每回合掉血上限 = Amount − 本回合已掉血",
                   on_attacked=_hardened_shell,
                   on_owner_turn_start=_hardened_shell_reset),
        PowerRules("paper_cuts", "PaperCutsPower.AfterDamageGiven：自己打出有效攻击且掉血"
                                 "→ 玩家**永久失去** Amount 点最大生命",
                   on_damage_given=_paper_cuts),
        # ⭐ 苦痛族（战斗内给卡牌贴标记；行为在 `affliction_*` 查询里）
        PowerRules("tangled", "TangledPower：给所有攻击牌贴 Entangled（费用 +Amount），"
                              "拥有者回合结束移除；**战斗中途生成的攻击牌**也贴"
                              "（`AfterCardEnteredCombat`）",
                   on_applied=_tangled,
                   on_card_entered_combat=_tangled_entered,
                   on_owner_turn_end=_tangled_clear),
        PowerRules("ringing", "RingingPower：给所有牌贴 Ringing（打不出），回合结束移除；"
                              "**新进场的牌**也贴",
                   on_applied=_ringing,
                   on_card_entered_combat=_ringing_entered,
                   on_owner_turn_end=_ringing_clear),
        PowerRules("smoggy", "SmoggyPower：玩家打出技能牌后给其余技能牌贴 Smog（打不出），"
                             "回合结束清；**新进场的技能牌**在本回合打过技能牌时也贴",
                   on_card_played=_smoggy,
                   on_card_entered_combat=_smoggy_entered,
                   on_owner_turn_end=_smoggy_clear),
        PowerRules("hex", "HexPower：给所有牌贴 Hexed → 获得**虚无**；新进场的牌也贴；"
                          "**施加者死了**就整条移除并清掉所有 Hexed",
                   on_applied=_hex,
                   on_card_entered_combat=_hex_entered,
                   on_applier_death=_hex_applier_death),
        PowerRules("chains_of_binding",
                   "ChainsOfBindingPower：每抽一张牌贴 Bound（每回合最多 Amount 张）；"
                   "`BeforeCardPlayed` 记「本回合已打过一张 Bound」→ `ShouldPlay` 把**其余**"
                   "Bound 牌挡掉（第一张永远能打）；`BeforeSideTurnEnd` 复位计数并"
                   "**清掉所有 Bound**",
                   on_card_drawn=_chains_of_binding,
                   on_before_card_played=_chains_mark_played,
                   on_before_side_turn_end=_chains_reset),
        PowerRules("vital_spark", "VitalSparkPower：给所有技能牌贴 Tainted；"
                                  "打出带 Tainted 的牌 → 玩家受伤 +Amount；"
                                  "**新进场的技能牌**也贴",
                   on_applied=_vital_spark,
                   on_card_entered_combat=_vital_spark_entered,
                   on_card_played=_vital_spark_played),
        PowerRules("tainted", "TaintedPower.ModifyDamageAdditive：拥有者受到的"
                              "有效攻击伤害 +Amount"),
        # ============ Wave B · 第十四批（卡实例级关键字：`ApplyKeyword`）========
        PowerRules("phantom_blades",
                   "PhantomBladesPower：`AfterApplied` / `AfterCardEnteredCombat` 给"
                   "**飞刀**加 ``Retain``（卡实例级关键字），"
                   "`ModifyDamageAdditive` 让**本回合第一张飞刀** +Amount"
                   "（数 `CardPlaysFinished` 里的飞刀，正在打的这张还没登记）",
                   on_applied=_phantom_blades_apply,
                   on_card_entered_combat=_phantom_blades_entered),
        PowerRules("possess_strength",
                   "PossessStrengthPower：窃取玩家获得的力量，死亡时归还",
                   on_self_death=_possess_return),
        PowerRules("possess_speed",
                   "PossessSpeedPower：窃取玩家获得的敏捷，死亡时归还",
                   on_self_death=_possess_return),
        PowerRules("tender", "TenderPower：玩家每打一张牌 −1 力量/敏捷，"
                             "拥有者回合结束按出牌数还回",
                   on_card_played=_tender,
                   on_owner_turn_end=_tender_restore),
        # ⭐ 复活族（死后留在场上、期间打不到、回合到了回血复活）
        PowerRules("illusion", "IllusionPower.AfterDeath：死后不被移除，下回合回满血复活",
                   on_self_death=_illusion),
        PowerRules("reattach", "ReattachPower.AfterDeath：其他节还活着时不死，"
                               "回合结束回 Amount 点",
                   on_self_death=_reattach),
        PowerRules("adaptable", "AdaptablePower.AfterDeath：试验体倒下后进入死亡态并复活",
                   on_self_death=_adaptable),
        PowerRules("steam_eruption",
                   "SteamEruptionPower.AfterDeath：瀑布巨人倒地后进入「即将爆炸」态",
                   on_self_death=_steam_eruption),
        # 数值修正型：每层 +1（真机在 Hook.ModifyDamage* 里累加层数）
        PowerRules("strength", "StrengthPower：伤害加法阶段 +层数",
                   damage_additive=True),
        PowerRules("dexterity", "DexterityPower：格挡加法阶段 +层数",
                   block_additive=True),
        PowerRules("vigor", "VigorPower：下一次攻击 +层数（用后移除）",
                   damage_additive=True),
        # ⭐ 铁甲战士（Ironclad）—— 逐条对照反编译源码实现
        PowerRules("rage", "RagePower.AfterCardPlayed：玩家打出攻击牌 → +Amount 格挡"
                            "（Unpowered）；AfterSideTurnEnd 整个移除（不递减）",
                   on_card_played=_rage,
                   on_owner_turn_end=_rage_expire),
        PowerRules("feel_no_pain", "FeelNoPainPower.AfterCardExhausted：自己的牌被消耗"
                                   " → +Amount 格挡（Unpowered）",
                   on_card_exhausted=_feel_no_pain),
        PowerRules("dark_embrace", "DarkEmbracePower.AfterCardExhausted：非虚无的消耗立刻抽"
                                   " Amount 张；虚无的消耗记到 AfterSideTurnEnd 一并抽",
                   on_card_exhausted=_dark_embrace,
                   on_owner_turn_end=_dark_embrace_flush),
        PowerRules("juggernaut", "JuggernautPower.AfterBlockGained：自己获得格挡后对"
                                 "随机敌人（Rng.CombatTargets）造成 Amount 点伤害（Unpowered）",
                   on_block_gained=_juggernaut),
        PowerRules("flame_barrier",
                   "FlameBarrierPower.AfterDamageReceived：被有效攻击打中 → 反伤 Amount"
                   "（Unpowered，被完全格挡也反）；AfterSideTurnEnd 在**对方**回合结束时移除",
                   on_attacked=_flame_barrier,
                   on_side_turn_end=_flame_barrier_expire),
        PowerRules("crimson_mantle",
                   "CrimsonMantlePower.AfterPlayerTurnStart：先按 SelfDamage 掉血"
                   "（Unblockable|Unpowered），再 +Amount 格挡（Unpowered）",
                   on_applied=_crimson_mantle_applied,
                   on_after_player_turn_start=_crimson_mantle_turn_start),
        PowerRules("inferno", "InfernoPower.AfterPlayerTurnStart：按 SelfDamage 掉血；"
                               "AfterDamageReceived：自己回合掉血 → 对全体敌人造成 Amount 伤害（Unpowered）",
                   on_applied=_inferno_applied,
                   on_after_player_turn_start=_inferno_turn_start,
                   on_attacked=_inferno_retaliate),
        PowerRules("setup_strike",
                   "SetupStrikePower（TemporaryStrengthPower，IsPositive=true）："
                   "施加时 +Amount 力量，层数变化时同步，AfterSideTurnEnd 移除并 −Amount",
                   on_applied=_setup_strike,
                   on_power_amount_changed=_setup_strike_changed,
                   on_owner_turn_end=_setup_strike_expire),
        PowerRules("mangle", "ManglePower（TemporaryStrengthPower，IsPositive=false）："
                              "施加时 −Amount 力量，层数变化时同步，AfterSideTurnEnd 移除并 +Amount",
                   on_applied=_mangle,
                   on_power_amount_changed=_mangle_changed,
                   on_owner_turn_end=_mangle_expire),
        PowerRules("demon_form", "DemonFormPower.AfterSideTurnStart：自己这边回合开始 +Amount 力量",
                   on_owner_turn_start=_demon_form),
        # ============ Wave C · 第一批（每回合施加族 + 两个 duration 型减益）=========
        PowerRules("biased_cognition",
                   "BiasedCognitionPower.AfterSideTurnStart：拥有者所在阵营回合开始，"
                   "给自己 −Amount 点专注（**永不撤销**，每回合都扣）",
                   on_owner_turn_start=partial(_periodic_attribute,
                                               pid="biased_cognition")),
        PowerRules("wraith_form",
                   "WraithFormPower.AfterSideTurnStart：拥有者所在阵营回合开始，"
                   "给自己 −Amount 点敏捷（**永不撤销**，每回合都扣）",
                   on_owner_turn_start=partial(_periodic_attribute, pid="wraith_form")),
        PowerRules("neurosurge",
                   "NeurosurgePower.AfterSideTurnStart：拥有者所在阵营回合开始，"
                   "给自己 +Amount 层末日（`Apply<DoomPower>`，走施加路径）",
                   on_owner_turn_start=partial(_periodic_attribute, pid="neurosurge")),
        PowerRules("debilitate",
                   "DebilitatePower：被拥有者（敌人）的**易伤倍率**抬高 "
                   "（`amount + (amount−1)`：1.5 → 2.0）、施加者的**虚弱倍率**压低 "
                   "（`amount − (1−amount)`：0.75 → 0.5）；"
                   "拥有者所在阵营回合结束递减（duration，见递减表）",
                   duration=True),
        PowerRules("no_block",
                   "NoBlockPower：拥有者**由卡牌**获得的格挡 ×0（`ModifyBlockMultiplicative`："
                   "非卡牌来源 / Unpowered 一律不乘）；敌方阵营回合结束递减（duration）",
                   duration=True),
        # ============ Wave C · 第二批（掉血阶段 / 战斗内加价 / 掉血处决）=========
        PowerRules("buffer",
                   "BufferPower.ModifyHpLostAfterOstyLate：拥有者受到的掉血压到 **0**"
                   "（晚段，覆盖无形的「压到 1」），每抵消一次 `AfterModifyingHpLostAfterOsty` "
                   "递减 1 层；消费者是 `powers.apply_hp_loss_cap`（三处掉血路径共用）"),
        PowerRules("borrowed_time",
                   "BorrowedTimePower.TryModifyEnergyCostInCombat：拥有者打出的**所有牌**"
                   "费用 +Amount（与 `CuriousPower` 同一阶段，见 `energy_cost_reduction`）；"
                   "拥有者所在阵营回合结束整条移除",
                   on_owner_turn_end=_borrowed_time_expire),
        PowerRules("hang",
                   "HangPower.ModifyDamageMultiplicative：被打者身上有「绞刑」、"
                   "且这张牌是 `Hang` 时，该牌的伤害 × 层数（消费者是 "
                   "`powers.damage_multipliers`）；卡牌侧的层数是 "
                   "`max(2, 当前层数)`（运行期公式 `target_power_floor`）"),
        PowerRules("the_gambit",
                   "TheGambitPower.AfterDamageReceived：拥有者受到**有效攻击**且**掉了血**"
                   "（`UnblockedDamage > 0`）→ 移除自己并 `CreatureCmd.Kill` 拥有者"
                   "（走 `core.mark_dead`）",
                   on_attacked=_the_gambit),
        # ============ Wave C · 第三批（施加者的牌 → 在我身上结算）=========
        PowerRules("strangle",
                   "StranglePower（InstancedPerApplier）：施加者（玩家）打出一张牌 → "
                   "BeforeCardPlayed 记下**当时**的层数、AfterCardPlayed 对拥有者（敌人）"
                   "造成等量 `Unblockable|Unpowered` 伤害；拥有者所在阵营回合结束整条移除",
                   on_before_card_played=partial(_applier_card_played_record,
                                                 pid="strangle"),
                   on_card_played=_strangle_hit,
                   on_owner_turn_end=_strangle_expire),
        PowerRules("oblivion",
                   "OblivionPower（InstancedPerApplier）：施加者（玩家）打出一张牌 → "
                   "AfterCardPlayed 给拥有者（敌人）上等量末日（施加者=原施加者）；"
                   "**玩家**阵营回合结束整条移除（`side == CombatSide.Player`）",
                   on_before_card_played=partial(_applier_card_played_record,
                                                 pid="oblivion"),
                   on_card_played=_oblivion_hit,
                   on_side_turn_end=_oblivion_expire),
        PowerRules("pyre", "PyrePower.ModifyMaxEnergy：每回合能量上限 +Amount"),
        PowerRules("friendship",
                   "FriendshipPower.ModifyMaxEnergy：拥有者每回合能量上限 +Amount"
                   "（源码只判 player != base.Owner.Player，由挂载对象保证）"),
        PowerRules("danse_macabre",
                   "DanseMacabrePower.BeforeCardPlayed：打出的牌**修正后**费用 ≥ 2"
                   "（EnergyVar(2)）→ +Amount 格挡（Unpowered）",
                   on_before_card_played=_danse_macabre),
        # ================= Wave A · 批次 3（`BeforeSideTurnEnd` 时机 + 伤害侧） =====
        PowerRules("hailstorm",
                   "HailstormPower.BeforeSideTurnEnd：自家回合结束前，只要场上有**冰球**"
                   "（`FrostOrbs` ≥ 1）就对全体可打敌人造成 Amount 点 Unpowered 伤害",
                   on_before_side_turn_end=_hailstorm),
        PowerRules("sneaky",
                   "SneakyPower.AfterCardPlayed：**别人**打出攻击牌 → 自己 +Amount 格挡"
                   "（Unpowered；条件是 `!=` owner，挂在敌人身上）",
                   on_card_played=_sneaky),
        PowerRules("haunt",
                   "HauntPower.AfterCardPlayed：打出 `Soul` → 随机一个可打敌人受"
                   " Amount 点 Unblockable+Unpowered 伤害（Rng.CombatTargets）",
                   on_card_played=_haunt),
        PowerRules("reaper_form",
                   "ReaperFormPower.AfterDamageGiven：自己的有效攻击造成伤害 → 给目标"
                   " `总伤害 × 层数` 层末日（AtterRemoved 只是 VFX 清理）",
                   on_damage_given=_reaper_form),
        # ================= Wave A · 批次 4（快照族 + 反伤） ======================
        PowerRules("afterimage",
                   "AfterimagePower：BeforeCardPlayed 记下「开始结算时的层数」，"
                   "AfterCardPlayed 按**那个快照值**给格挡（Unpowered）——"
                   "避免「残影本体那一次」重复结算",
                   on_before_card_played=_afterimage,
                   on_card_played=_afterimage_payoff),
        PowerRules("serpent_form",
                   "SerpentFormPower：与残影同一套快照语义，AfterCardPlayed 按快照值对"
                   "**随机**可打敌人造成 Unpowered 伤害（Rng.CombatTargets）",
                   on_before_card_played=_serpent_form,
                   on_card_played=_serpent_form_payoff),
        PowerRules("reflect",
                   "ReflectPower：AfterDamageReceived 被有效攻击打掉格挡时，把"
                   " **被格挡的伤害**反弹给攻击者（Unpowered）；"
                   "AfterSideTurnStart 递减 1 层",
                   on_attacked=_reflect,
                   on_owner_turn_start=_reflect_expire),
        # ================= Wave A · 批次 5（`ModifyCardPlayCount` 族） ==========
        # 次数修改走 `powers.card_play_count` 的表（见 `CARD_PLAY_COUNT_POWERS`），
        # 所以这里只需要声明"回合结束时要不要整个移除"。
        PowerRules("burst",
                   "BurstPower.ModifyCardPlayCount：**技能牌**多打一次；"
                   "AfterModifyingCardPlayCount 递减 1 层；AfterSideTurnEnd 整个移除",
                   on_side_turn_end=_card_play_count_expire),
        PowerRules("duplication",
                   "DuplicationPower.ModifyCardPlayCount：**任意牌**多打一次；"
                   "AfterModifyingCardPlayCount 递减 1 层；AfterSideTurnEnd 整个移除",
                   on_side_turn_end=_card_play_count_expire),
        PowerRules("signal_boost",
                   "SignalBoostPower.ModifyCardPlayCount：**能力牌**多打一次；"
                   "AfterModifyingCardPlayCount 递减 1 层（不随回合结束移除）"),
        PowerRules("echo_form",
                   "EchoFormPower.ModifyCardPlayCount：本回合**已开始的牌数 < 层数**时"
                   "任意牌多打一次（不递减；计数不含当前这张牌）"),
        # ================= Wave A · 批次 6（抽牌 / 球 / 递减） ==================
        PowerRules("loop",
                   "LoopPower.AfterPlayerTurnStart：有球时按层数触发**最左边那个球**"
                   "的被动（OrbCmd.Passive，含暗球累积/玻璃衰减）",
                   on_after_player_turn_start=_loop),
        PowerRules("hibernate",
                   "HibernatePower.AfterPlayerTurnStart：玩家回合开始递减 1 层",
                   on_after_player_turn_start=_hibernate),
        PowerRules("corrosive_wave",
                   "CorrosiveWavePower.AfterCardDrawn：自己抽到牌（**含**手牌抽取）→"
                   "全体可打敌人各 Amount 层中毒；AfterSideTurnEnd 整个移除",
                   on_card_drawn=_corrosive_wave,
                   on_side_turn_end=_card_play_count_expire),
        PowerRules("speedster",
                   "SpeedsterPower.AfterCardDrawn：**非**手牌抽取的抽牌 → 全体可打敌人"
                   "受 Amount 点 Unpowered 伤害",
                   on_card_drawn=_speedster),
        # ================= Wave A · 批次 7（实例计数 / 快照 + 撤回） ============
        PowerRules("monologue",
                   "MonologuePower：BeforeCardPlayed 记下 `Strength`(=1) 快照，"
                   "AfterCardPlayed 给这么多力量并累计 StrengthApplied；"
                   "AfterSideTurnEnd 移除自己并**撤回累计值**",
                   on_before_card_played=_monologue_begin,
                   on_card_played=_monologue_payoff,
                   on_side_turn_end=_monologue_expire),
        PowerRules("panache",
                   "PanachePower.AfterCardPlayed：每 5 张牌（**不含潘趣本体那一次**）"
                   "对全体可打敌人造成 Amount 点 Unpowered 伤害；"
                   "**拥有者回合结束把进度复位 5**（跨回合累计 = 每回合白送一次群伤）",
                   on_card_played=_panache,
                   on_owner_turn_end=_panache_reset),
        PowerRules("pale_blue_dot",
                   "PaleBlueDotPower.AfterCardPlayed：本回合打满 5 张牌 → 获得 Amount 层"
                   "「下回合抽牌」；每回合只触发一次，AfterSideTurnEnd 复位",
                   on_card_played=_pale_blue_dot,
                   on_side_turn_end=_pale_blue_dot_reset),
        # ================= Wave A · 批次 8（快照 / 关键字 / 星资源） ============
        PowerRules("storm",
                   "StormPower：BeforeCardPlayed 对**能力牌**记层数快照，"
                   "AfterCardPlayed 按快照引导等量闪电球",
                   on_before_card_played=_storm_begin,
                   on_card_played=_storm_payoff),
        PowerRules("spirit_of_ash",
                   "SpiritOfAshPower.BeforeCardPlayed：打出带 **Ethereal** 关键字的牌 →"
                   " +Amount 格挡（Unpowered）",
                   on_before_card_played=_spirit_of_ash),
        PowerRules("the_sealed_throne",
                   "TheSealedThronePower.BeforeCardPlayed：自己每打一张牌 → +Amount 星",
                   on_before_card_played=_the_sealed_throne),
        # ================= Wave B · 首批（`ModifyHandDraw`） ====================
        # 抽牌张数修正走 `powers.hand_draw_bonus` 的表（见 `HAND_DRAW_BONUS_POWERS`），
        # 所以这五个只需要注册；`clarity` 是"恒 +1"，其余是"每层 +1"。
        PowerRules("clarity",
                   "ClarityPower.ModifyHandDraw：每回合抽牌数 **+1**（写死 1，不是层数）"),
        PowerRules("demesne",
                   "DemesnePower.ModifyHandDraw：每回合抽牌数 +Amount"),
        PowerRules("machine_learning",
                   "MachineLearningPower.ModifyHandDraw：每回合抽牌数 +Amount"),
        PowerRules("tools_of_the_trade",
                   "ToolsOfTheTradePower.ModifyHandDraw：每回合抽牌数 +Amount"),
        PowerRules("tyranny",
                   "TyrannyPower.ModifyHandDraw：每回合抽牌数 +Amount"),
        # ================= Wave B · 第二批（`BeforeHandDraw`） ==================
        PowerRules("call_of_the_void",
                   "CallOfTheVoidPower.BeforeHandDraw：抽牌前把 Amount 张"
                   "**非基础/非远古**的池内牌塞进手牌（Rng.CombatCardGeneration）",
                   on_before_hand_draw=_call_of_the_void),
        PowerRules("creative_ai",
                   "CreativeAiPower.BeforeHandDraw：抽牌前把 Amount 张**能力牌**"
                   "塞进手牌（互不重复）",
                   on_before_hand_draw=_creative_ai),
        PowerRules("hello_world",
                   "HelloWorldPower.BeforeHandDraw：按 **AmountOnTurnStart** 生成等量"
                   "**普通**牌塞进手牌（0 层则不动手）",
                   on_before_hand_draw=_hello_world),
        PowerRules("spectrum_shift",
                   "SpectrumShiftPower.BeforeHandDraw：抽牌前把 Amount 张**无色**牌"
                   "塞进手牌（ColorlessCardPool）",
                   on_before_hand_draw=_spectrum_shift),
        PowerRules("sentry_mode",
                   "SentryModePower.BeforeHandDraw：抽牌前塞 Amount 张 `SweepingGaze`"
                   "（固定卡，不抽随机）",
                   on_before_hand_draw=_sentry_mode),
        PowerRules("infinite_blades",
                   "InfiniteBladesPower.BeforeHandDraw：抽牌前塞 Amount 张**飞刀**"
                   "（Shiv.CreateInHand，固定卡）",
                   on_before_hand_draw=_infinite_blades),
        # ============ Wave B · 第三批（带条件的数值修正，机制在 damage_* 两张表里） ====
        PowerRules("accuracy",
                   "AccuracyPower.ModifyDamageAdditive：**自己**造成的伤害 + 有牌来源 +"
                   " 牌带 `Shiv` 标签 → +Amount"),
        PowerRules("leadership",
                   "LeadershipPower.ModifyDamageAdditive：**同阵营的别人**造成的伤害"
                   " → +Amount"),
        PowerRules("covered",
                   "CoveredPower.ModifyDamageMultiplicative：被打的是自己 → ×0（免疫）；"
                   "**敌方回合结束整条移除**（`AfterSideTurnEnd` 的 `side == Enemy`）",
                   on_side_turn_end=_covered_expire),
        PowerRules("guarded",
                   "GuardedPower.ModifyDamageMultiplicative：被打的是自己 →"
                   " ×DamageDecrease(0.5)"),
        PowerRules("colossus",
                   "ColossusPower.ModifyDamageMultiplicative：被打的是自己且**攻击者"
                   "身上有易伤** → ×DamageDecrease(0.5)；"
                   "**敌方回合结束递减 1 层**（`AfterSideTurnEnd` 的 `side == Enemy`）",
                   on_side_turn_end=_colossus_tick),
        PowerRules("double_damage",
                   "DoubleDamagePower.ModifyDamageMultiplicative：自己打出的**卡牌**"
                   " 伤害 → ×2（怪物招式不算）；**自己回合结束递减 1 层**"
                   "（少了这一条，`shadow_step` 给的双倍会永久留着）",
                   on_owner_turn_end=_double_damage_tick),
        # ============ Wave B · 第五批（能量费用修改，机制在 COST_* 两张表里） ========
        PowerRules("rebound",
                   "ReboundPower.ModifyCardPlayResultLocation：本来进弃牌堆的牌 →"
                   " **抽牌堆顶**；改过之后递减 1 层；"
                   "**自己回合结束整条移除**（没用上就作废）",
                   on_owner_turn_end=_rebound_expire),
        PowerRules("galvanic",
                   "GalvanicPower：给所有能力牌（含新进场的）贴 Galvanized；"
                   "打出带 Galvanized 的牌 → 拥有者受 Amount 点 Unpowered 伤害"
                   "（这是它的代价）",
                   on_applied=_galvanic,
                   on_card_entered_combat=_galvanic_entered,
                   on_card_played=_galvanic_played),
        PowerRules("nostalgia",
                   "NostalgiaPower.ModifyCardPlayResultLocation：**攻击/技能牌**且"
                   "本回合这类牌数 < 层数时 → 抽牌堆顶"),
        PowerRules("shroud",
                   "ShroudPower.AfterPowerAmountChanged：**自己**上**末日**时 → +Amount 格挡",
                   on_power_amount_changed=_shroud),
        PowerRules("sleight_of_flesh",
                   "SleightOfFleshPower.AfterPowerAmountChanged：自己给**敌人**上**减益**"
                   "时 → 对该敌人造成 Amount 点 Unpowered 伤害",
                   on_power_amount_changed=_sleight_of_flesh),
        PowerRules("free_attack",
                   "FreeAttackPower.TryModifyEnergyCostInCombatLate：**攻击牌**费用归零"),
        PowerRules("free_power",
                   "FreePowerPower.TryModifyEnergyCostInCombatLate：**能力牌**费用归零"),
        PowerRules("free_skill",
                   "FreeSkillPower.TryModifyEnergyCostInCombatLate：**技能牌**费用归零"),
        PowerRules("veilpiercer",
                   "VeilpiercerPower.TryModifyEnergyCostInCombatLate：**虚无牌**费用归零"),
        PowerRules("curious",
                   "CuriousPower.TryModifyEnergyCostInCombat：**能力牌**费用 −Amount"
                   "（下限 0；跑在 Late 段之前）"),
        PowerRules("void_form",
                   "VoidFormPower.TryModifyEnergyCostInCombatLate：本回合**前 Amount 张**"
                   "牌费用归零（之后再打就要付费）"),
        PowerRules("arsenal",
                   "ArsenalPower.AfterCardGeneratedForCombat：**自己生成**的牌 →"
                   " +Amount 力量",
                   on_card_generated_for_combat=_arsenal),
        PowerRules("pillar_of_creation",
                   "PillarOfCreationPower.AfterCardGeneratedForCombat：自己生成的牌 →"
                   " +Amount 格挡（Unpowered）",
                   on_card_generated_for_combat=_pillar_of_creation),
        PowerRules("smokestack",
                   "SmokestackPower.AfterCardGeneratedForCombat：自己生成的**状态牌** →"
                   " 全体可打敌人受 Amount 点 Unpowered 伤害",
                   on_card_generated_for_combat=_smokestack),
        PowerRules("trash_to_treasure",
                   "TrashToTreasurePower.AfterCardGeneratedForCombat：自己生成的"
                   "**状态牌** → 引导 Amount 个随机充能球（Rng.CombatOrbs）",
                   on_card_generated_for_combat=_trash_to_treasure),
        PowerRules("tracking",
                   "TrackingPower.ModifyDamageMultiplicative：自己打出的卡牌伤害、"
                   "且**目标身上有虚弱** → ×(1 + Amount/100)"),
        # ============ Wave B · 第七批（攻击指令族：`BeforeAttack` / `AfterAttack`）===
        # 引擎没有 ``AttackCommand`` 对象，用"这条伤害真的算出来了没有 + 是**哪张牌**
        # 建的指令"来等价表达。逐条指令结算的调用点是 ``_run_card_plays`` /
        # ``autoplay_card`` / ``_resolve_selection``（真机每个 playIndex 建一条指令）。
        PowerRules("lethality",
                   "LethalityPower.ModifyDamageMultiplicative：本回合**第一张**攻击牌的"
                   "有效攻击 → ×(1 + Amount/100)；判据是源码里 `CardPlaysStarted` 的"
                   "攻击牌计数 == 1（当前这张**已计入**），引擎用 "
                   "`state.attacks_played_this_turn`（逐次打出 +1）"),
        PowerRules("gigantification",
                   "GigantificationPower.BeforeAttack/AfterAttack：捕获**下一条**攻击指令，"
                   "它的伤害 ×3（源码是字面量 `3m`，与层数无关；层数只表示还能用几次），"
                   "指令结算完递减 1 层"),
        PowerRules("corruption", "CorruptionPower：技能牌费用为 0"
                                  "（TryModifyEnergyCostInCombatLate）且打出后消耗"
                                  "（ModifyCardPlayResultLocation）"),
        PowerRules("cruelty", "CrueltyPower.ModifyVulnerableMultiplier：自己造成的"
                               "**有效攻击**打在别人身上时，易伤倍率 1.5 → 1.5+Amount/100"),
        PowerRules("tank", "TankPower.ModifyDamageMultiplicative：自己受到的有效攻击 ×1.5"
                            "（GuardedPower 那一半是多人专用：GetTeammatesOf 在单人局恒为空）"),
        PowerRules("unmovable", "UnmovablePower.ModifyBlockMultiplicative：每回合**前 Amount 次**"
                                 "由卡牌获得的格挡 ×2（按出牌去重、按回合隔离）"),
        PowerRules("one_two_punch",
                   "OneTwoPunchPower.ModifyCardPlayCount：拥有者的攻击牌**多打出一次**"
                   "（``playCount + 1``，与层数无关）；``AfterModifyingCardPlayCount`` "
                   "每次递减 1 → 层数 = 还能生效几次；AfterSideTurnEnd 整个移除",
                   on_owner_turn_end=_one_two_punch_expire),
        PowerRules("hellraiser", "HellraiserPower.AfterCardDrawnEarly：抽到带 Strike 标签的牌"
                                  " → 自动打出（无限血量敌人的每回合 9 次限流未实现，"
                                  "引擎没有无限血量的敌人）",
                   on_card_drawn_early=_hellraiser),
        PowerRules("rupture", "RupturePower：自己回合掉血 → +Amount 力量；"
                               "若是**正在打出的那张牌**造成的，则等这张牌结算完再发",
                   on_before_card_played=_rupture_begin,
                   on_attacked=_rupture_damage,
                   on_card_played=_rupture_end),
        PowerRules("vicious", "ViciousPower.AfterPowerAmountChanged：**自己**给别人上"
                               " Amount>0 的易伤 → 抽 Amount 张",
                   on_power_applied=_vicious),
        PowerRules("juggling", "JugglingPower.BeforeCardPlayed：本回合第 **3** 张攻击牌"
                                " → 复制 Amount 张进手牌（计数在 AfterApplied 初始化、"
                                "回合结束清零）",
                   on_before_card_played=_juggling),
        PowerRules("aggression", "AggressionPower.BeforeSideTurnStart：从弃牌堆按"
                                  " combat_card_selection 随机捞 Amount 张攻击牌进手牌并升级",
                   on_before_side_turn_start=_aggression),
        PowerRules("stampede", "StampedePower.AfterAutoPostPlayPhaseEntered：回合结束、"
                                "清手牌之前，按 shuffle 流随机自动打出 Amount 张手牌攻击牌",
                   on_auto_post_play=_stampede),
        # ================= Wave A · 批次 1：纯标记 + 已有钩子 =================
        # ⭐ 纯标记：自身无行为，由别的机制查询（源码注释明写 "doesn't actually do
        # anything on its own"）。`_poison` 读它来多算触发次数。
        PowerRules("accelerant", "AccelerantPower：纯标记 —— PoisonPower 查询它"
                                  "（TriggerCount = 1 + accelerant）"),
        # ⭐ AfterEnergyReset 族
        PowerRules("genesis", "GenesisPower.AfterEnergyReset：能量重置后获得 Amount 颗星",
                   on_energy_reset=_genesis),
        PowerRules("radiance", "RadiancePower.AfterEnergyReset：能量 +1（EnergyVar(1)），"
                                "然后递减 1 层",
                   on_energy_reset=_radiance),
        PowerRules("lightning_rod",
                   "LightningRodPower.AfterEnergyReset：引导 1 个闪电球，然后递减 1 层",
                   on_energy_reset=_lightning_rod),
        PowerRules("spinner", "SpinnerPower.AfterEnergyReset：引导 Amount 个玻璃球",
                   on_energy_reset=_spinner),
        # ⭐ AfterSideTurnStart 族（`participants.Contains(Owner)` = 拥有者所在阵营，
        # 与引擎的 `on_owner_turn_start` 一一对应）
        PowerRules("coolant", "CoolantPower.AfterSideTurnStart：按**球种类数** × Amount "
                               "获得格挡（Unpowered）",
                   on_owner_turn_start=_coolant),
        PowerRules("countdown", "CountdownPower.AfterSideTurnStart：给随机一个可打敌人"
                                 "上 Amount 层末日（Rng.CombatTargets）",
                   on_owner_turn_start=_countdown),
        PowerRules("noxious_fumes", "NoxiousFumesPower.AfterSideTurnStart：给全体可打敌人"
                                     "上 Amount 层中毒",
                   on_owner_turn_start=_noxious_fumes),
        PowerRules("prep_time", "PrepTimePower.AfterSideTurnStart：拥有者回合开始获得"
                                 " Amount 层活力",
                   on_owner_turn_start=_prep_time),
        # ================= Wave A · 批次 2：AfterCardDrawn / AfterDamageGiven ===
        PowerRules("automation", "AutomationPower.AfterCardDrawn：每 10 张牌 → +Amount 能量",
                   on_card_drawn=_automation),
        PowerRules("cacophony", "CacophonyPower.AfterCardDrawn：每 33 张牌 → 随机一个"
                                 "可打敌人受 Amount 点伤害（Unpowered）",
                   on_card_drawn=_cacophony),
        PowerRules("pagestorm", "PagestormPower.AfterCardDrawn：抽到带 Ethereal 的牌"
                                 " → 再抽 Amount 张",
                   on_card_drawn=_pagestorm),
        PowerRules("iteration", "IterationPower.AfterCardDrawn：本回合第一张**状态牌**"
                                 " → 再抽 Amount 张",
                   on_card_drawn=_iteration),
        PowerRules("envenom", "EnvenomPower.AfterDamageGiven：有效攻击且掉血 → "
                               "给目标上 Amount 层中毒",
                   on_damage_given=_envenom),
        PowerRules("concoct", "ConcoctPower.AfterDamageGiven：同上；"
                               "AfterSideTurnEnd 在**对方**回合结束时整个移除",
                   on_damage_given=_concoct,
                   on_side_turn_end=_concoct_expire),
        PowerRules("monarchs_gaze",
                   "MonarchsGazePower.AfterDamageGiven：有效攻击 → 给目标上"
                   " Amount 层力量下降（被完全格挡也触发）",
                   on_damage_given=_monarchs_gaze),
        PowerRules("monarchs_gaze_strength_down",
                   "MonarchsGazeStrengthDownPower（TemporaryStrengthPower，"
                   "IsPositive=false）：施加时 −Amount 力量，层数变化时同步，"
                   "拥有者回合结束移除并 +Amount",
                   on_applied=_monarchs_gaze_down,
                   on_power_amount_changed=_monarchs_gaze_down_changed,
                   on_owner_turn_end=_monarchs_gaze_down_expire),
        # ============ Wave B · 第八批（实例数据 + 同伴伤害） ====================
        PowerRules("subroutine",
                   "SubroutinePower.Before/AfterCardPlayed：记下**能力牌**开始结算时的层数，"
                   "结算完还回等量能量（按牌身份记录，避免多打重复给）",
                   on_before_card_played=_subroutine_record,
                   on_card_played=_subroutine_energy),
        PowerRules("rolling_boulder",
                   "RollingBoulderPower.AfterPlayerTurnStart：对全部可打敌人造成 Amount 点"
                   "Unpowered 伤害，然后 Amount += 5（能力自带的 DamageVar 常量）",
                   on_after_player_turn_start=_rolling_boulder),
        PowerRules("withering_presence",
                   "WitheringPresencePower.AfterCardPlayed：目标玩家每打一张牌 "
                   "`CardsLeft` −1；到 0 塞 1 张「凋零」进手牌并复位为 6",
                   on_card_played=_withering_presence),
        PowerRules("shadow_step",
                   "ShadowStepPower.AfterSideTurnStart：拥有者阵营回合开始 → 给自己 "
                   "Amount 层双倍伤害并整条移除",
                   on_owner_turn_start=_shadow_step),
        PowerRules("underworld",
                   "UnderworldPower.AfterDamageGiven：**同伴**的有效攻击造成总伤害 > 0 → "
                   "给目标 `总伤害 × Amount` 层末日；敌方回合结束整条移除",
                   on_side_turn_end=_underworld_expire),
        # ============ Wave B · 第九批（**消费者已经在引擎里**，补登记） ========
        # 这一类最阴：行为早就写好了（消费者直接读这个能力的层数），但 ``RULES``
        # 里没有它 —— 于是门禁按"能力未实现"把**所有施加它的卡**都拒了。
        # `tools/audit_power_refs.py` 专门查这个（读到却没登记 / 读到不存在的 id）。
        PowerRules("focus",
                   "FocusPower.ModifyOrbValue：自己球的所有数值 +Amount，"
                   "**下限 0**（`AllowNegative` 为真，负专注要把值钳在 0）；"
                   "消费者是 `orbs._focused`，Plasma 声明 `focus_scaled=False` 因此不受影响"),
        # ============ Wave B · 第十批（三个新调用点：能量消耗 / 格挡条件 / 激发球）===
        PowerRules("orbit",
                   "OrbitPower.AfterEnergySpent：自己打牌花掉能量后累计，每满 4 点"
                   "回 Amount 点能量（按**已触发次数**记账，层数中途变化不补不欠）",
                   on_energy_spent=_orbit),
        PowerRules("fasten",
                   "FastenPower.ModifyBlockAdditive：自己获得的格挡 +Amount，"
                   "但**卡牌来源必须带 `Defend` 标签**（没有牌来源的格挡照样加；"
                   "`AfterModifyingBlockAmount` 是空实现，不影响数值）"),
        PowerRules("thunder",
                   "ThunderPower.AfterOrbEvoked：激发的球是**闪电**时，"
                   "对这次激发的目标再造成 Amount 点 Unpowered 伤害",
                   on_orb_evoked=_thunder),
        # ============ Wave B · 第十一批（格挡侧：不清格挡查询 / 倍率） ============
        PowerRules("blur",
                   "BlurPower：`ShouldClearBlock` 对自己返回 false（回合开始**不清格挡**），"
                   "`AfterSideTurnStart` 递减 1 层（`AfterPreventingBlockClear` 是空实现）",
                   on_owner_turn_start=_blur_tick),
        PowerRules("shadowmeld",
                   "ShadowmeldPower.ModifyBlockMultiplicative：自己获得的格挡 ×2^Amount"
                   "（源码**不看 props**，所以 Unpowered 来源也乘），"
                   "自己阵营回合结束整条移除",
                   on_owner_turn_end=_shadowmeld_expire),
        PowerRules("well_laid_plans",
                   "WellLaidPlansPower.ShouldFlush：对拥有者返回 false → 回合结束"
                   "**不清空手牌**（与 `RetainHandPower` 判据相同，是两个能力；"
                   "消费者是 `powers.retains_hand`）"),
        # ============ Wave B · 第十二批（`set_power_var` + `AfterBlockCleared`）====
        PowerRules("the_bomb",
                   "TheBombPower.BeforeSideTurnEnd：自家阵营回合结束前递减；剩 1 层时"
                   "对全体可打敌人造成 `Damage` 点 Unpowered 伤害并移除 —— 伤害量来自"
                   "**能力实例变量**（卡牌 `SetDamage(BombDamage)`，升级 +10 跟着变）",
                   on_before_side_turn_end=_the_bomb),
        PowerRules("toric_toughness",
                   "ToricToughnessPower.AfterBlockCleared：自己的格挡被清空 → 按"
                   "`SetBlock(上次实际获得的格挡)` 记录的数值给格挡（Unpowered），"
                   "并递减 1 层",
                   on_block_cleared=_toric_toughness),
        PowerRules("self_forming_clay",
                   "SelfFormingClayPower.AfterBlockCleared：自己的格挡被清空 → 获得"
                   " Amount 点格挡（Unpowered）并整条移除"
                   "（施加者是遗物 `SelfFormingClay`，遗物侧还没抽出来）",
                   on_block_cleared=_self_forming_clay),
        # ============ Wave B · 第十三批（`AfterCombatEnd`：Run 层额外奖励）=======
        # 三个能力都在战斗结束时给**额外奖励**，消费者是 `powers.after_combat_end`
        # （由 `RunEnv._apply_after_combat_end_powers` 在战斗胜利后调用）。
        PowerRules("royalties",
                   "RoyaltiesPower.AfterCombatEnd：`room.AddExtraReward(GoldReward(Amount))` "
                   "—— 战斗结束额外发 Amount 金币（卡牌 `Royalties` 给 30，升级 +10）"),
        PowerRules("forbidden_grimoire",
                   "ForbiddenGrimoirePower.AfterCombatEnd：`AddExtraReward(CardRemovalReward)` "
                   "× Amount —— 战斗结束让玩家**各选一张**移除（引擎走挂起选牌，不替玩家选）"),
        PowerRules("improvement",
                   "ImprovementPower.AfterCombatEnd：把牌组里**可升级**的牌随机升级 Amount 张"
                   "（源码 `Rng.CombatCardSelection.NextItem`，挑一张移出候选，不重复）"),
        # ============ Wave B · 第十六批（AutoPrePlay 时机 + 卡实例级 Sly）========
        PowerRules("mayhem",
                   "MayhemPower.AfterAutoPrePlayPhaseEntered：玩家**抽完牌、即将进入"
                   "出牌阶段**时，从抽牌堆顶自动打出 Amount 张"
                   "（`CardPileCmd.AutoPlayFromDrawPile(..., Top, forceExhaust: false)`）",
                   on_auto_pre_play=_mayhem),
        PowerRules("master_planner",
                   "MasterPlannerPower.AfterCardPlayed：玩家打出的**技能牌**获得"
                   " ``Sly``（`CardCmd.ApplyKeyword`，卡实例级关键字 → 之后被弃掉时自动打出）",
                   on_card_played=_master_planner),
        # ============ Wave B · 第十七批（"这一张是不是 0 费打的"）==============
        # `Resources.EnergyValue`（本次出牌实际花的能量）是这两个能力的**唯一判据**；
        # 引擎记在 `state.current_play_energy_spent`（出牌结束时置回 -1）。
        PowerRules("feral",
                   "FeralPower.ModifyCardPlayResultLocation：**0 费打出的攻击牌**回到"
                   "**手牌**（每回合 Amount 次，改过才 +1；回合开始归零）；"
                   "`AfterApplied` 用「本回合已打过的 0 费攻击数」初始化额度",
                   on_applied=_feral_applied,
                   on_owner_turn_start=_feral_reset),
        PowerRules("one_for_all",
                   "OneForAllPower.ModifyDamageAdditive：**0 费打出的攻击牌** +Amount"
                   "（X 费牌不算；不在出牌里时按「这张牌的带修正费用 == 0」判）"),
        # ============ Wave B · 第十八批（星资源：花星 / 得星）==============
        PowerRules("child_of_the_stars",
                   "ChildOfTheStarsPower.AfterStarsSpent：**花星** → 获得"
                   " `Amount × 花掉的星` 点格挡（Unpowered，不吃敏捷/脆弱）",
                   on_stars_spent=_child_of_the_stars),
        PowerRules("black_hole",
                   "BlackHolePower：**获得星**（amount > 0）→ 对全体可打敌人造成 Amount 点"
                   "Unpowered 伤害；**花了星的那张牌结算完**（`IsLastInSeries`）也群伤"
                   "（源码特意不用 AfterStarsSpent：星在出牌开头就扣了，黑洞要等牌打完）",
                   on_stars_gained=_black_hole_gained,
                   on_card_played=_black_hole_played),
        # ============ Wave B · 第十九批（生成物 / 充能球 / 造牌）==============
        PowerRules("calamity",
                   "CalamityPower：记下打出的**攻击牌**的层数，那张牌结算完 → 从角色卡池"
                   "生成 `Amount` 张随机**攻击牌**进手牌（`Rng.CombatCardGeneration`）",
                   on_before_card_played=_calamity_mark,
                   on_card_played=_calamity_play),
        PowerRules("consuming_shadow",
                   "ConsumingShadowPower.AfterSideTurnEnd：拥有者回合结束 → "
                   "`OrbCmd.EvokeLast`（激发**最右边**那个球）Amount 次",
                   on_owner_turn_end=_consuming_shadow),
        PowerRules("soulbound",
                   "SoulboundPower.AfterCardGeneratedForCombat：**施加者**造出 ``Soul`` → "
                   "再往抽牌堆**随机位置**塞 Amount 张（`IsAddingSoul` 防自激）",
                   on_card_generated_for_combat=_soulbound),
        # ============ 纯 duration 型：回合结束递减 1，无其他行为
        PowerRules("vulnerable", "VulnerablePower：受伤 ×1.5，回合结束递减",
                   duration=True),
        PowerRules("weak", "WeakPower：造成伤害 ×0.75，回合结束递减",
                   duration=True),
        PowerRules("frail", "FrailPower：获得格挡 ×0.75，回合结束递减",
                   duration=True),
    )
}

# ==========================================================================
# 临时属性族（``Temporary{Strength,Dexterity,Focus}Power``）的批量登记
# ==========================================================================
# ⭐ 这些子类的源码**逐字同构**（``AnticipatePower.cs`` 之类整份只有
# ``OriginModel``，部分多一行 ``IsPositive``），三个事件方法全部继承父类
# （``Temporary{Strength,Dexterity,Focus}Power.cs`` 的 ``BeforeApplied`` /
# ``AfterPowerAmountChanged`` / ``AfterSideTurnEnd``）。所以这里不写 18 份
# 字面量，而是按 :data:`_TEMPORARY_ATTRIBUTE_POWERS` 批量登记：**表里多一行
# 就等于源码里多一个同构子类**，行为一字不改（`docs/16` §2.1）。
#
# ``setup_strike`` / ``mangle`` / ``monarchs_gaze_strength_down`` 是最早的三个
# 入口，已在上面的字面量里单独登记（保留原来的 source 文案），这里跳过。
_TEMPORARY_PARENT_OF_ATTRIBUTE = {
    "strength": "TemporaryStrengthPower",
    "dexterity": "TemporaryDexterityPower",
    "focus": "TemporaryFocusPower",
}
for _temp_pid, (_temp_attr, _temp_sign, _temp_apply, _temp_expire) in (
        _TEMPORARY_ATTRIBUTE_POWERS.items()):
    if _temp_pid in RULES:
        continue
    RULES[_temp_pid] = PowerRules(
        _temp_pid,
        f"{_temp_pid}Power : {_TEMPORARY_PARENT_OF_ATTRIBUTE[_temp_attr]}"
        f"（子类只覆写 OriginModel{' 与 IsPositive' if _temp_sign < 0 else ''}）："
        f"BeforeApplied 施加 {_temp_sign:+d}×Amount 点{_TEMPORARY_ATTRIBUTE_CN[_temp_attr]}，"
        "AfterPowerAmountChanged 在层数被改时同步，"
        f"AfterSideTurnEnd（本阵营）移除并 {-_temp_sign:+d}×当前层数",
        on_applied=partial(_temporary_attribute_applied, pid=_temp_pid),
        on_power_amount_changed=partial(_temporary_attribute_changed, pid=_temp_pid),
        on_owner_turn_end=partial(_temporary_attribute_expire, pid=_temp_pid))
del _temp_pid, _temp_attr, _temp_sign, _temp_apply, _temp_expire

#: 引擎**已实现**的能力集合。``content.engine_coverage()`` 用它判断卡牌可用性。
IMPLEMENTED: frozenset[str] = frozenset(RULES)

#: 数值修正的查表：哪些能力在格挡/伤害的加法阶段贡献"每层 +1"。
BLOCK_ADDITIVE: frozenset[str] = frozenset(
    pid for pid, rules in RULES.items() if rules.block_additive)
DAMAGE_ADDITIVE: frozenset[str] = frozenset(
    pid for pid, rules in RULES.items() if rules.damage_additive)

#: 乘法修正的查表：``(能力, 作用于哪一方, 倍率)``。
#:
#: 真机这些全是 ``ModifyDamageMultiplicative``，**累乘**（不是加法），
#: 所以放在同一张表里按顺序乘。``IsPoweredAttack()`` = 带 ``Move`` 且非
#: ``Unpowered``（``ValuePropExtensions.cs:5-12``）—— 正好是引擎调用
#: ``compute_damage`` 的那一段，所以这里不必再判一次。
#:
#: ============== ======== ====== ============================================
#: 能力            作用方   倍率   源码
#: ============== ======== ====== ============================================
#: ``weak``       攻击方   0.75   ``WeakPower``
#: ``vulnerable`` 防御方   1.50   ``VulnerablePower``
#: ``flutter``    防御方   0.50   ``FlutterPower.ModifyDamageMultiplicative``
#: ``soar``       防御方   0.50   ``SoarPower.ModifyDamageMultiplicative``
#: ``shrink``     攻击方   0.70   ``ShrinkPower``（``(100-30)/100``）
#: ============== ======== ====== ============================================
DAMAGE_MULTIPLIERS: tuple[tuple[str, str, str], ...] = (
    ("weak", "attacker", "0.75"),
    ("vulnerable", "defender", "1.5"),
    ("flutter", "defender", "0.50"),
    ("soar", "defender", "0.50"),
    ("shrink", "attacker", "0.70"),
)


def _same_side(first, second, state: "CombatState") -> bool:
    """两个单位是否同一阵营（真机 ``Creature.Side``）。"""
    if first is None or second is None:
        return False
    return (first is state.player) == (second is state.player)


def damage_additive_modifiers(attacker: "Combatant | None",
                              defender: "Combatant",
                              state: "CombatState | None" = None,
                              card=None,
                              powered: bool = True) -> list[tuple[str, int]]:
    """**带条件**的加法修正（``ModifyDamageAdditive`` 那条加法阶段）。

    真机 `Hook.ModifyDamageInternal` 遍历**所有模型**，每个模型自己判条件：

    * ``AccuracyPower``：``base.Owner == dealer``（**自己**造成的伤害）+ 有效攻击 +
      有牌来源 + ``card.Tags.Contains(CardTag.Shiv)`` → ``+Amount``
    * ``LeadershipPower``：``base.Owner != dealer``（**别人**打的）+ 同阵营 +
      有效攻击 → ``+Amount``（挂在敌人身上 = 队友的攻击加伤）

    ⚠️ 与 :data:`DAMAGE_ADDITIVE`（无条件、只看攻击方）分开：那一份是"每层 +1"的
    简单族、已对拍，保持原样不动；这一份是**逐条照抄条件**的。
    """
    if attacker is None or state is None:
        return []
    out: list[tuple[str, int]] = []
    card_tags: tuple = ()
    if card is not None:
        definition = card.definition() if hasattr(card, "definition") else None
        card_tags = tuple(getattr(definition, "tags", ()) or ())
    for holder in (state.player, *state.enemies):
        if not holder.alive():
            continue
        amount = holder.power("accuracy")
        if (amount > 0 and holder is attacker and card is not None
                and "Shiv" in card_tags):
            out.append(("accuracy", amount))
        amount = holder.power("phantom_blades")
        if (amount > 0 and holder is attacker and card is not None
                and "Shiv" in card_tags and state is not None):
            # `PhantomBladesPower.ModifyDamageAdditive`：**本回合第一张飞刀** +Amount。
            # 源码数 ``CardPlaysFinished`` 里本回合的飞刀 —— 正在打的这张还没登记，
            # 所以第一张吃加成、第二张起不吃（`num > 0 → 0m`）。
            finished = getattr(state, "tag_plays_finished_this_turn", {}) or {}
            if finished.get("Shiv", 0) <= 0:
                out.append(("phantom_blades", amount))
        amount = holder.power("one_for_all")
        if amount > 0 and holder is attacker:
            # `OneForAllPower.ModifyDamageAdditive`：**0 费打出的攻击牌** +Amount
            # （X 费牌不算；条件见 :func:`one_for_all_bonus`）。
            bonus = one_for_all_bonus(holder, card, state)
            if bonus:
                out.append(("one_for_all", bonus))
        amount = holder.power("leadership")
        if (amount > 0 and holder is not attacker
                and _same_side(holder, attacker, state)):
            out.append(("leadership", amount))
    return out


def damage_multipliers(attacker: "Combatant | None",
                       defender: "Combatant",
                       card=None,
                       state: "CombatState | None" = None) -> list[tuple[str, str]]:
    """乘法阶段要乘哪些倍率，返回 ``[(来源, 倍率)]``。

    固定倍率来自 :data:`DAMAGE_MULTIPLIERS`；**条件倍率**单独算：

    * ``SurroundedPower``：只有从背后打才 ×1.5。
    * ``CrueltyPower.ModifyVulnerableMultiplier``：把**易伤那一项**的倍率抬高
      ``Amount/100``（源码是 ``amount + base.Amount / 100m``，加在倍率上而不是再乘一次）。
      条件：被打的不是残酷的拥有者本人、且这次是**有效攻击** —— 后者由
      "走到 ``compute_damage``" 天然保证（``Unpowered`` 的伤害不走这里）。
    * ``TankPower.ModifyDamageMultiplicative``：拥有者受到的有效攻击 ×1.5。
    * ``DebilitatePower``（`docs/12` §2.29）：**两处**都要改，方向相反 ——
      被打者身上有 ``debilitate`` 时把**易伤**那一项抬高
      （``amount + (amount − 1)``：1.5 → 2.0），攻击者身上有 ``debilitate`` 时把
      **虚弱**那一项压低（``amount − (1 − amount)``：0.75 → 0.5）。
      源码里这是 ``DebilitatePower`` 的两个**公开方法**（不是 override），
      由 ``VulnerablePower`` / ``WeakPower`` 在算倍率时回头调用它 ——
      所以只在 ``vulnerable``/``weak`` 这两项里生效，别的倍率不受影响。
    """
    out: list[tuple[str, str]] = []
    for pid, side, factor in DAMAGE_MULTIPLIERS:
        holder = attacker if side == "attacker" else defender
        if holder is None or holder.power(pid) <= 0:
            continue
        if pid == "vulnerable":
            if attacker is not None and attacker is not defender:
                # `CrueltyPower`：只有玩家能持有它，而"拥有者 == 被打的目标"时源码
                # 明确返回原倍率 —— 所以"攻击方身上的残酷"与源码条件等价。
                bonus = attacker.power("cruelty")
                if bonus:
                    factor = str(Decimal(factor) + Decimal(bonus) / Decimal(100))
            # `DebilitatePower.ModifyVulnerableMultiplier`：由**被打者**持有。
            # 源码只判 `target == base.Owner` 与"有效攻击"（后者已由走到这里保证），
            # 没有攻击者存在与否的条件。
            if defender.power("debilitate") > 0:
                factor = str(Decimal(factor) + (Decimal(factor) - Decimal(1)))
        if pid == "weak" and attacker is not None \
                and attacker.power("debilitate") > 0:
            # `DebilitatePower.ModifyWeakMultiplier`：由**施加者**持有
            # （源码 `dealer.GetPower<DebilitatePower>()`）。
            factor = str(Decimal(factor) - (Decimal(1) - Decimal(factor)))
        out.append((pid, factor))
    # `SurroundedPower.ModifyDamageMultiplicative`：被包围者只在**背后**被打时 ×1.5。
    if defender.power("surrounded") > 0 and attacker is not None:
        facing = defender.power_facing
        behind = (facing == "right" and attacker.power("back_attack_left") > 0) or \
                 (facing == "left" and attacker.power("back_attack_right") > 0)
        if behind:
            out.append(("surrounded", "1.5"))
    # `TankPower.ModifyDamageMultiplicative`：`if (target != base.Owner) return 1;`
    # 所以只有"拥有者自己被打"才 ×1.5。另一半（给队友挂 `GuardedPower` 让他们 ×0.5）
    # 是**多人专用**：`GetTeammatesOf` 在单人局恒为空，引擎只模拟单人，故无可实现部分。
    if defender.power("tank") > 0:
        out.append(("tank", "1.5"))
    # ⭐ 带条件的倍率族（`ModifyDamageMultiplicative` 逐条照抄）。
    # 与上面几个的区别：它们的条件要看**谁打的**（dealer）、**有没有牌来源**（cardSource）、
    # 以及**被打的是不是自己**（target == base.Owner）。
    # `props.IsPoweredAttack()` 一律成立 —— `Unpowered` 的伤害走不到 `compute_damage`。
    card_tags: tuple = ()
    card_type = ""
    if card is not None:
        definition = card.definition() if hasattr(card, "definition") else None
        card_tags = tuple(getattr(definition, "tags", ()) or ())
        card_type = getattr(definition, "card_type", "")
    holders = [defender, attacker] if attacker is not None else [defender]
    if state is not None:
        holders = [state.player, *state.enemies]
    for holder in holders:
        if not holder.alive():
            continue
        amount = holder.power("covered")
        if amount > 0 and holder is defender:
            # `CoveredPower`：被打的是自己 → **×0**（完全免疫）。
            out.append(("covered", "0"))
        amount = holder.power("guarded")
        if amount > 0 and holder is defender:
            # `GuardedPower`：被打的是自己 → ×``DamageDecrease``（源码是动态变量的字面量）。
            out.append(("guarded", str(GUARDED_DAMAGE_DECREASE)))
        amount = holder.power("colossus")
        if (amount > 0 and holder is defender and attacker is not None
                and attacker.power("vulnerable") > 0):
            # `ColossusPower`：被打的是自己 **且攻击者身上有易伤** → ×``DamageDecrease``。
            out.append(("colossus", str(COLOSSUS_DAMAGE_DECREASE)))
        amount = holder.power("double_damage")
        if (amount > 0 and holder is attacker and card is not None):
            # `DoubleDamagePower`：``dealer == base.Owner`` 且 **cardSource != null**
            # （怪物招式不算）→ ×2。
            out.append(("double_damage", "2"))
        amount = holder.power("slow")
        if amount > 0 and holder is defender:
            # `SlowPower.ModifyDamageMultiplicative`：打的是**它**、且是有效攻击
            # （`Unpowered` 走不到 `compute_damage`）→ ×(1 + 0.1 × 本回合已出牌数)。
            factor = slow_multiplier(holder)
            if factor is not None:
                out.append(("slow", str(factor)))
        amount = holder.power("tracking")
        if (amount > 0 and holder is attacker and card is not None
                and defender.power("weak") > 0):
            # `TrackingPower`：自己打的、有牌来源、**目标身上有虚弱** → ×(1 + Amount/100)。
            out.append(("tracking", str(Decimal(1) + Decimal(amount) / Decimal(100))))
        # `LethalityPower`：本回合**第一张**攻击牌的有效攻击 → ×(1 + Amount/100)。
        # 源码判据（``LethalityPower.cs:40``）：
        #   num  = CardPlaysStarted 里"本回合 + 牌型=Attack + 玩家是自己"的条数；
        #   num2 = 1（这张牌正在 `PileType.Play` 里）；`num > num2 → 1m`。
        # 也就是说 num 必须 == 1 —— 而**当前这一次打出已经登记**（CardPlayStarted 在
        # 伤害之前），所以引擎用 `attacks_played_this_turn`（逐次打出 +1）恒等表达。
        # `CurrentPlayIndex > 0`（多打一次的第二遍）也由同一个计数覆盖：那时它是 2。
        amount = holder.power("lethality")
        if (amount > 0 and holder is attacker and card is not None
                and card_type == "attack"
                and getattr(state, "attacks_played_this_turn", 0) == 1):
            out.append(("lethality", str(Decimal(1) + Decimal(amount) / Decimal(100))))
        # `HangPower.ModifyDamageMultiplicative`（`docs/12` §2.34）：被打者身上有
        # "绞刑"、且**这张牌就是 `Hang`** → ×``base.Amount``（层数）。
        # 源码三条件：``target == base.Owner``（被打的是它自己，由下面这段
        # "只有防御方触发的倍率"表达）、``props.IsPoweredAttack()``（走到
        # `compute_damage` 天然成立）、``cardSource is Hang``（引擎比 ``card.cid``）。
        # ⚠️ 判"哪张牌"必须用 `card`，不能只看"谁身上有能力" ——
        # 少了它，**所有**打这个敌人的攻击都会被 ×层数（静默变强好几倍）。
        if (card is not None and getattr(card, "cid", "") == "hang"
                and holder is defender and defender.power("hang") > 0):
            # ⚠️ 必须判 ``holder is defender``：这个循环会遍历玩家与**所有**敌人，
            # 少了它，一个敌人身上的绞刑会按持有者个数各乘一次
            # （实测：2 层被算成 ×4，Hang 打 40 而不是 20）。
            out.append(("hang", str(defender.power("hang"))))
        # `GigantificationPower`：**下一条**攻击指令 ×3。
        # 源码两半：`BeforeAttack` 把第一条"自己打的、攻击牌建的、有效攻击"的指令存进
        # `Data.commandToModify`（已捕获就不再捕获）；`ModifyDamageMultiplicative` 在
        # `commandToModify == null || cardSource == 捕获的那张牌` 时给 `3m`。
        # 引擎没有指令对象，用**牌的 `uid`** 当身份：第一条真的算到伤害的攻击牌把它
        # 捕获住，同一张牌的多段伤害都 ×3，别的牌（例如它自动打出的牌）不沾光。
        # ⚠️ 分歧记录：真机在**建指令**时就捕获（所以"攻击牌但一次有效伤害都没造成"
        # 也会吃掉一层），引擎改成"真的算到伤害才捕获"。当前内容里所有"没有可解析
        # damage 效果"的攻击牌都因 `unsupported` 进不了可抽池，所以该分歧在准入内容上
        # **不可观测**；将来解析了这类牌要回到 `BeforeAttack` 的时机重新对齐。
        amount = holder.power("gigantification")
        if (amount > 0 and holder is attacker and card is not None
                and card_type == "attack"):
            flags = holder.power_flags
            uid = getattr(card, "uid", 0)
            captured = flags.get("gigantification_card", 0)
            if not captured:
                flags["gigantification_card"] = captured = uid
            if captured == uid:
                out.append(("gigantification", "3"))
    return out


def on_attack_command_finished(state: "CombatState", card=None) -> None:
    """``Hook.AfterAttack``：**一条**攻击指令结算完 → `GigantificationPower` 结账。

    源码 ``GigantificationPower.AfterAttack``：``if (command == commandToModify)``
    时才清空捕获并 ``PowerCmd.Decrement``。引擎的等价物是"捕获到的正是这张牌"。
    调用点是**逐条指令**（真机每个 playIndex 建一条指令）：``_run_card_plays`` 的每次
    打出、``autoplay_card``、以及挂起后续跑的 ``_resolve_selection`` —— 写成"每张牌
    一次"会让"多打一次"的攻击牌少递减一层（静默变强）。
    """
    if state is None:
        return
    uid = getattr(card, "uid", None)
    if uid is None:
        return
    player = getattr(state, "player", None)
    holders = [player] if player is not None else []
    holders.extend(getattr(state, "enemies", ()) or ())
    for holder in holders:
        if holder.power("gigantification") <= 0:
            continue
        flags = holder.power_flags
        if flags.get("gigantification_card", 0) != uid:
            continue
        flags["gigantification_card"] = 0
        holder.add_power("gigantification", -1)



# ==========================================================================
# 引擎调用入口
# ==========================================================================
def _dispatch(combatant: "Combatant", ctx: dict) -> int:
    """钩子总线的回调：按 ``ctx["hook"]`` 找规则字段并调用。

    返回**实际调用了几个处理器** —— 总线用它决定要不要记轨迹
    （钩子会遍历"玩家 + 全部敌人"，多数持有者没有对应能力，
    全记下来轨迹里全是噪声，反而看不出"这一跳谁响应了"）。
    """
    hook = ctx["hook"]
    fired = 0
    for pid in list(combatant.powers):
        rule = RULES.get(pid)
        if rule is None:
            continue                      # 未实现的能力：不生效，由覆盖率报告
        handler = getattr(rule, hook, None)
        if handler is None:
            continue
        # ⭐ **分实例能力要逐实例派发**（S01）：真机每个实例都是独立的 model 对象，
        # 同一个钩子它们各收一次，处理器里的 ``base.Amount`` 也是**实例自己的**层数。
        # 只派发一次（拿总和）会让两个炸弹共用一个倒计时 —— 而日志完全正常。
        instances = list(combatant.power_instances.get(pid) or ())
        for instance in (instances if instances else [None]):
            if instance is not None and instance not in (
                    combatant.power_instances.get(pid) or ()):
                continue                  # 前一个实例已经把自己移除（`Remove(this)`）
            context = PowerContext(state=ctx["state"], owner=combatant,
                                   events=ctx["events"])
            # ⭐ 把 ``pid`` 也放进去：规则有时要知道"**我自己**是哪条能力"
            # （``_ends_with_applier`` 要查 ``powers_applier[pid]``）。
            context.pid = pid
            context.instance = instance
            context.instance_vars = (instance.vars if instance is not None
                                     else combatant.power_vars.get(pid) or {})
            for key, value in ctx.items():
                if key in ("state", "events", "hook", "owner", "each"):
                    continue
                setattr(context, key, value)
            fired += 1
            amount = (instance.amount if instance is not None
                      else combatant.power(pid))
            if handler(context, amount) == REMOVE:
                if instance is not None:
                    # `Remove(this)`：只摘掉**这一个**实例（`Nightmare` 两个实例
                    # 各自兑现一次，不能一次清光）。
                    combatant.remove_power_instance(pid, instance)
                else:
                    combatant.powers.pop(pid, None)
    return fired


def _fire(combatant: "Combatant", state: "CombatState", events: list[str],
          hook: str, **extra) -> None:
    """触发某个钩子（走 :mod:`sts2_sim.hooks` 的声明表）。

    ``hook`` 是 ``PowerRules`` 上的字段名，也是 ``hooks.HOOKS`` 里的键 ——
    两边由 ``hooks.audit()`` 与测试保证一一对应。
    """
    from . import hooks as hook_bus
    hook_bus.fire(hook, state, events, _dispatch,
                  owner=combatant, **extra)


def on_energy_reset(combatant: "Combatant", state: "CombatState",
                    events: list[str]) -> None:
    """玩家回合能量重置后（``Hook.AfterEnergyReset``）。

    ⚠️ 时机很重要：真机是在**能量已经重置为初始值之后**才加这一笔，
    放在重置之前会被覆盖掉。
    """
    _fire(combatant, state, events, "on_energy_reset")


def on_turn_start(combatant: "Combatant", state: "CombatState",
                  events: list[str]) -> None:
    """拥有者所在阵营的回合开始（``Hook.AfterSideTurnStart``）。"""
    _fire(combatant, state, events, "on_owner_turn_start")


def on_turn_end(combatant: "Combatant", state: "CombatState",
                events: list[str]) -> None:
    """拥有者所在阵营的回合结束（``Hook.AfterSideTurnEnd``）。

    ⚠️ 这里**只跑钩子，不递减 duration**：易伤 / 虚弱 / 脆弱的递减时机是
    "敌方阵营回合结束"（见 ``DECREMENTS_AT_ENEMY_SIDE_TURN_END``），
    由 :func:`tick_durations` 单独负责。
    """
    _fire(combatant, state, events, "on_owner_turn_end")


def on_power_amount_changed(state: "CombatState", pid: str, amount: int,
                            applier=None, owner=None) -> None:
    """**任意**能力的层数变化之后（``Hook.AfterPowerAmountChanged``）。

    真机两个调用点（``PowerCmd.cs:160`` 施加、``:249`` 修改）**都**在非零变动时通知，
    ``PowerCmd.Decrement`` 走的是后者的 ``applier: null`` 形态 —— 所以递减也会通知。

    ⚠️ **重入保护**：处理器本身可能再改层数（例如给目标再上一层末日），
    真机靠"通知是在改完之后发的"自然收敛；引擎加一个显式标志，防止自激循环。
    """
    if getattr(state, "_in_power_amount_change", False):
        return
    state._in_power_amount_change = True
    try:
        from . import hooks as hook_bus
        hook_bus.fire("on_power_amount_changed", state, [],
                      lambda holder, ctx: _dispatch(holder, ctx),
                      pid=pid, amount=amount, applier=applier, changed_owner=owner)
    finally:
        state._in_power_amount_change = False


def on_card_generated_for_combat(state: "CombatState", events: list[str],
                                 card, creator=None) -> None:
    """战斗内**生成**一张牌之后（``Hook.AfterCardGeneratedForCombat``）。

    ``creator`` 是"谁生成了这张牌"（``Player?``）。真机里这个钩子只对
    **玩家生成**的牌有意义（``creator.Creature == base.Owner``），所以引擎总是
    传玩家 —— 但这不影响条件判定，因为 `ArsenalPower` 这类还会判牌型。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_card_generated_for_combat", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx),
                  card=card, creator=creator)


def on_before_hand_draw(state: "CombatState", events: list[str]) -> None:
    """玩家回合**抽手牌之前**（``Hook.BeforeHandDraw``）。

    ``CallOfTheVoidPower`` / ``CreativeAiPower`` / ``HelloWorldPower`` /
    ``SentryModePower`` / ``SpectrumShiftPower`` / ``InfiniteBladesPower`` 都在这里
    把生成的牌塞进手牌 —— 所以时机必须在**抽牌之前**（它们是"这回合手牌的一部分"）。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_before_hand_draw", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx))


def on_after_shuffle(state: "CombatState", events: list[str]) -> bool:
    """**洗完牌之后**（``Hook.AfterShuffle``，``CardPileCmd.cs:1131``）。

    返回 ``True`` = 有处理者开了玩家选择（``StratagemPower`` 从抽牌堆挑牌入手），
    调用方必须**停在原地**：真机这里是一个 ``await``，抽牌循环会停住 ——
    洗牌之后**该抽的那几张要等选完再抽**。返回 ``False`` = 没人挂起，继续原流程。

    ⚠️ 抽到空堆时的洗牌也会走到这里，所以这个时机在抽牌循环**中间**出现，
    不是"回合开始时洗一次"那种粗粒度。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_after_shuffle", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx))
    return state.pending is not None


def on_before_side_turn_end(state: "CombatState", events: list[str],
                            side: str) -> None:
    """**任意一方**阵营回合结束**之前**（``Hook.BeforeSideTurnEnd``）。

    ``TheBombPower``（倒计时到 1 层就炸）与 ``HailstormPower``（有冰球就下冰雹）
    都挂在这个时机上 —— 它们必须在"回合真的结束"之前结算，放到
    ``AfterSideTurnEnd`` 会让"这回合该发生的伤害"晚一整个回合。

    ⚠️ 与 :func:`on_side_turn_end` 一样是**全体**分发并把 ``side`` 传下去：
    源码里的判据是 ``participants.Contains(Owner)``（"是自家阵营结束吗"）。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_before_side_turn_end", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx), side=side)


def on_side_turn_end(state: "CombatState", events: list[str],
                     side: str) -> None:
    """**任意一方**阵营回合结束（``Hook.AfterSideTurnEnd`` 的全体分发）。

    与 :func:`on_turn_end` 的区别是**谁被通知**：那个只通知"结束的这一方的单位"，
    这个通知**双方**并把 ``side`` 传下去 —— ``FlameBarrierPower`` 正是靠
    "``Owner.Side != side`` 就移除"来判断自己该不该消失。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_side_turn_end", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx), side=side)


def on_side_turn_end_late(state: "CombatState", events: list[str],
                          side: str) -> None:
    """``AfterSideTurnEndLate``：``AfterSideTurnEnd`` 的**第二遍**遍历。

    真机 ``Hook.AfterSideTurnEnd``（``Hook.cs``）在**同一个函数**里跑两遍：
    先遍历全部 listener 的 ``AfterSideTurnEnd``，收尾之后再遍历全部 listener 的
    ``AfterSideTurnEndLate``。所以调用点必须紧跟 :func:`on_side_turn_end`
    —— 位置错了（例如放到 ``tick_durations`` 之后、或者和第一遍交错）
    ``DisintegrationPower`` 的自伤就会插到别的回合末效果中间。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_side_turn_end_late", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx), side=side)


def on_after_player_turn_start(state: "CombatState",
                               events: list[str]) -> None:
    """**玩家**回合开始、抽牌之后（``Hook.AfterPlayerTurnStart``）。

    ⚠️ 与 :func:`on_player_turn_start` 不是一回事：那个对应
    ``AfterSideTurnStart``（跑在抽牌**之后**），这个对应 ``AfterPlayerTurnStart``
    （跑在抽牌之后、``AfterSideTurnStart`` **之前**）。
    ``InfernoPower`` / ``CrimsonMantlePower`` 的"每回合开始掉血"挂在这里。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_after_player_turn_start", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx))


def on_card_exhausted(combatant: "Combatant", state: "CombatState",
                      events: list[str], card, caused_by_ethereal: bool) -> None:
    """一张牌**被消耗之后**（``Hook.AfterCardExhausted``）。

    ``card`` 是被消耗的那张牌本体（``FeelNoPainPower`` 靠它判"是不是自己的牌"，
    ``DarkEmbracePower`` 靠 ``caused_by_ethereal`` 决定立刻抽还是攒到回合结束）。
    遍历**双方**：真机的 ``AfterCardExhausted`` 会通知所有模型。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_card_exhausted", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx),
                  card=card, caused_by_ethereal=caused_by_ethereal)


def on_block_gained(combatant: "Combatant", state: "CombatState",
                    events: list[str], gained: int, card=None) -> None:
    """某个单位**获得格挡之后**（``Hook.AfterBlockGained``）。

    ``gained`` 是**全部修正之后**的格挡量（真机传的也是 ``modifiedAmount``），
    所以"加了 0 点"这类情况也能被看见（``JuggernautPower`` 自己会过滤）。
    ``card`` 是这次格挡的来源卡牌（``CreatureCmd.GainBlock(..., cardPlay?.Card)``），
    非卡牌来源为 ``None`` —— ``UnmovablePower`` 的"每回合前 N 次由卡牌获得的格挡"
    要靠它区分来源。
    """
    _fire(combatant, state, events, "on_block_gained", gained=gained, card=card)


def on_before_card_played(state: "CombatState", events: list[str],
                          card) -> None:
    """一张牌**开始结算之前**（``Hook.BeforeCardPlayed``）。

    遍历双方：真机这一个钩子也是全体模型分发。``JugglingPower``（戏法）靠它数
    "本回合第几张攻击牌"，``RupturePower``（破裂）靠它登记"这张牌正在被打出"，
    好在结算结束时才把攒下的力量发下去。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_before_card_played", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx), card=card)


def on_power_applied(state: "CombatState", events: list[str], pid: str,
                     applier: "Combatant | None", applied: int) -> None:
    """某个能力被施加时，**通知双方所有模型**（``Hook.AfterPowerAmountChanged``）。

    与 :func:`on_applied` 的区别是作用域与实参：那个只通知"被施加的那个单位"
    且不带施加者；这个带 ``applier`` 与这一次的量。
    ``ViciousPower``（凶恶）的整个机制就是"**我**给别人上易伤 → 抽牌"，
    所以必须知道"是谁施加的"。

    ⚠️ 覆盖范围：引擎里**只有** ``apply_power`` 算子（卡牌/药水/遗物效果）这一条
    增加能力的路径；真机的 ``PowerCmd.Apply`` 之外还有 ``ModifyAmount`` / ``Decrement``
    两条，它们只会**减少**层数，而 ``ViciousPower`` 明确要求 ``amount > 0``，
    所以那两条路径对它没有影响。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_power_applied", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx),
                  pid=pid, applier=applier, applied=applied)


def on_before_side_turn_start(state: "CombatState", events: list[str],
                              side: str) -> None:
    """某一方回合开始、**清格挡之前**（``Hook.BeforeSideTurnStart``）。

    ``AggressionPower``（侵略）在这里"从弃牌堆随机捞一张攻击牌并升级" ——
    放在 ``on_owner_turn_start``（抽牌之后）会与真机的手牌顺序不同。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_before_side_turn_start", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx), side=side)


def on_auto_post_play(state: "CombatState", events: list[str]) -> None:
    """玩家进入 ``AutoPostPlay`` 阶段（``Hook.AfterAutoPostPlayPhaseEntered``）。

    时机是"结束回合的最开始、手牌还完整的时候"：
    ``CombatManager.EndPlayerTurnPhaseOneInternal``（``CombatManager.cs:1545``）
    先分发这个钩子，之后才是 ``BeforeSideTurnEnd`` 与 ``DoTurnEnd``（手牌触发器 + 清手牌）。
    ``StampedePower``（踩踏）在这里随机自动打出一张手牌里的攻击牌。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_auto_post_play", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx))


def tick_durations(state: "CombatState", events: list[str]) -> None:
    """敌方阵营回合结束时，对**场上所有单位**递减易伤 / 虚弱 / 脆弱。"""
    everyone = [state.player] + list(state.enemies)
    for combatant in everyone:
        for pid in list(combatant.powers):
            if pid in DECREMENTS_AT_ENEMY_SIDE_TURN_END and combatant.power(pid) > 0:
                combatant.add_power(pid, -1)


def on_attacked(defender: "Combatant", attacker: "Combatant",
                state: "CombatState", events: list[str],
                unblocked: int = 0, from_card: bool = False,
                powered: bool = True, card=None,
                blocked: int = 0) -> None:
    """受到伤害之后（``Hook.AfterDamageReceived``）。

    ``unblocked`` / ``from_card`` / ``powered`` 是钩子的**实参**，不是可选装饰：
    ``SlipperyPower`` 要看"掉没掉血"，``SkittishPower`` 还要看"来源是不是卡牌"。
    只传一个"被打了"的布尔，这两个能力就都实现不了 —— 而它们恰好是
    "滑溜的墨宝"和"胆怯的花园幽灵鳗"的核心机制。

    ``card`` 是真机的 ``cardSource``（这次伤害由哪张牌造成）。``attacker`` 可以是
    ``None``：自伤没有来源，但真机**照样**分发这个钩子（``CreatureCmd.cs:416``）。

    ``blocked`` 是 ``DamageResult.BlockedDamage``（**被格挡掉的量**）——
    ``ReflectPower`` 反弹的就是它，不是掉血量。少了这个参数，反伤只能瞎猜。
    """
    _fire(defender, state, events, "on_attacked",
          attacker=attacker, unblocked=unblocked, from_card=from_card,
          powered=powered, card=card, blocked=blocked)


def on_player_turn_start(state: "CombatState", events: list[str]) -> None:
    """**玩家**阵营回合开始 → 拥有者是敌人的能力也在这时触发（``RampartPower``）。

    拥有者是敌人，所以挂不到 ``on_owner_turn_start`` 上 —— 那个要等它自己的回合。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_player_turn_start", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx))


def on_card_played(state: "CombatState", events: list[str],
                   card_type: str, card=None,
                   is_last_in_series: bool = True) -> None:
    """玩家打出一张牌之后（``Hook.AfterCardPlayed``）。

    遍历**玩家 + 全部敌人**：苦痛族（``tangled`` / ``ringing`` / ``smoggy`` /
    ``hex`` / ``chains_of_binding``）是加在**玩家**身上的 debuff（源码里读的是
    ``base.Owner.Player``），而 ``enrage`` 是加在**敌人**身上、同样关心
    "玩家打了什么牌"。只遍历一方，另一半就会静默不触发。

    ``card_type`` 是 ``"skill"`` / ``"attack"`` / ``"power"`` ——
    ``EnragePower`` **只认技能牌**，``SmoggyPower`` 也只看技能牌。
    ``card`` 是打出的那张牌：``VitalSparkPower`` 要看它有没有带 ``Tainted``。
    ``is_last_in_series`` 是 ``CardPlay.IsLastInSeries``：多打一次的牌只在**最后
    一遍**为真（`BlackHolePower` 用它保证"花了星的牌只炸一次"）。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_card_played", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx),
                  card_type=card_type, card=card,
                  is_last_in_series=is_last_in_series)


def on_applied(combatant: "Combatant", state: "CombatState", events: list[str],
               pid: str, applied: int, fresh: bool = True) -> None:
    """某个能力**刚被施加**（``Hook.AfterApplied``）。

    只有被施加的那一个能力触发，不是遍历全部 —— 真机的 ``AfterApplied``
    也是逐个模型回调自己的。``TangledPower`` 靠它"贴上时立刻给所有攻击牌贴苦痛"。

    ``applied`` 是**这一次**施加的层数（真机 ``BeforeApplied`` 的 ``amount``）。
    处理器拿到的第二个位置实参是施加后的总层数，两者不能混用：
    ``TemporaryStrengthPower``（临时力量 / 重击）加的是"这一次"的量，
    用总层数会变成 3、6、9… 越叠越多。

    ⚠️ ``fresh`` 是"这条能力**原本不存在**（这次是新造实例）"。真机的
    ``AfterApplied`` / ``BeforeApplied`` 只在 ``PowerCmd.Apply`` 的**新实例**那条
    路径上调用（``PowerCmd.cs:136/159``）；叠加走 ``ModifyAmount``（``:119``）**不调**
    它们，只发 ``AfterPowerAmountChanged``（``:249``）。引擎里这条钩子两个路径都会发
    （有些能力按"被打出的次数"计数，例如 ``CrimsonMantlePower`` 的 ``SelfDamage``，
    叠加也必须发），所以把路径信息作为 ``fresh`` 透传，让需要区分的能力自己判。
    """
    _fire(combatant, state, events, "on_applied", pid=pid, applied=applied,
          fresh=fresh)


def on_draw_card(state: "CombatState", card, events: list[str],
                 from_hand_draw: bool = False) -> None:
    """玩家**抽到一张牌**之后（``Hook.AfterCardDrawn``）。

    遍历玩家 + 敌人：``ChainsOfBindingPower`` 挂在玩家身上、贴玩家的牌。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_card_drawn", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx), card=card,
                  from_hand_draw=from_hand_draw)


def on_draw_card_early(state: "CombatState", card, events: list[str],
                       from_hand_draw: bool = False) -> None:
    """玩家抽到一张牌之后、``on_card_drawn`` **之前**（``Hook.AfterCardDrawnEarly``）。

    真机的 ``Hook.AfterCardDrawn`` 对模型列表**遍历两遍**（``Hook.cs:202-213``）：
    第一遍调 ``AfterCardDrawnEarly``、第二遍才调 ``AfterCardDrawn``。
    ``HellraiserPower``（地狱使者）挂在第一遍上 —— 它要在别的"抽牌时"能力
    之前把这张 Strike 打出去。
    """
    from . import hooks as hook_bus
    hook_bus.fire("on_card_drawn_early", state, events,
                  lambda holder, ctx: _dispatch(holder, ctx), card=card,
                  from_hand_draw=from_hand_draw)


def on_damage_given(attacker: "Combatant", state: "CombatState",
                    events: list[str], unblocked: int,
                    powered: bool = True,
                    was_fully_blocked: bool = False,
                    target: "Combatant | None" = None,
                    total: int = 0) -> None:
    """**拥有者造成了伤害之后**（``Hook.AfterDamageGiven``）。

    ``unblocked`` 是这次实际掉的血，``powered`` 是"有效攻击"
    （带 ``Move`` 且非 ``Unpowered``），``was_fully_blocked`` 是"被完全挡下"
    （``DamageResult.WasFullyBlocked``）。``PainfulStabsPower`` 看前两个，
    ``ImbalancedPower`` 只看最后一个 —— 少传一个，对应的能力就永远不触发。

    ``target`` 是**这一笔伤害打在谁身上**（真机 ``Creature target``）。
    ``EnvenomPower`` / ``ConcoctPower``（有效攻击 → 给目标上毒）与
    ``MonarchsGazePower``（给目标上力量下降）需要它。

    ``total`` 是 ``DamageResult.TotalDamage``（**含被格挡的那部分**）——
    ``ReaperFormPower`` 按它算末日层数；用 ``unblocked`` 会让"打在格挡上"的
    攻击少上很多层。
    """
    _fire(attacker, state, events, "on_damage_given",
          unblocked=unblocked, powered=powered, total=total,
          was_fully_blocked=was_fully_blocked, target=target)


def on_any_death(state: "CombatState", dead: "Combatant",
                 events: list[str]) -> None:
    """有单位死亡之后：先触发**自己的** ``AfterDeath``，再触发队友的，最后结算连锁倒下。

    这是**唯一**的死亡钩子入口（``core.mark_dead`` 调它），
    两种作用域都在这里转发：``on_self_death``（死者自己）与
    ``on_ally_death``（同阵营队友）。
    """
    from . import hooks as hook_bus

    def dispatch(holder, ctx):
        _dispatch(holder, ctx)

    # 1) 死者自己的能力（`StockPower` 补人 / `SurprisePower` 召唤 / `InfestedPower`）
    hook_bus.fire("on_self_death", state, events, dispatch, owner=dead)
    # 2) 队友的能力（`CrabRagePower` 暴怒 / `RavenousPower` 吞食）
    hook_bus.fire("on_ally_death", state, events, dispatch,
                  each=[e for e in (getattr(state, "enemies", ()) or ())
                        if e is not dead])
    # 3) ⭐ **施加者**死了（``AfterDeath`` 的第三段作用域）：``constrict`` / ``shrink`` /
    #    ``hex`` 的结束条件都是 ``creature == base.Applier``。这一段必须通知**全体**
    #    （玩家与敌人身上都可能挂着"某个单位施加的"减益），由规则自己比对施加者。
    hook_bus.fire("on_applier_death", state, events, dispatch, dead=dead)
    kill_secondary_teammates(state, dead, events)


def stun(combatant: "Combatant", state: "CombatState",
         events: list[str], source: str = "", *,
         action: str = "", follow_up: str = "") -> None:
    """``CreatureCmd.Stun``：让一个单位**下一回合不动**（只跳过一次）。

    ⚠️ 是"跳过**一次**"，不是永久眩晕：引擎里用一次性标志位，
    ``_choose_intent`` 消费掉它之后怪物恢复正常出招。
    写成"移除意图"会让被晕的怪从此再不行动。

    ⭐ 真机还有一个三参数的重载 ``Stun(creature, stunMove, nextMoveId)``
    （``asleep`` / ``slumber`` 的"挨打醒来"走它）：

    * ``stunMove`` —— 这一回合要执行的**唤醒动作**；真机里 ``WakeUpMove`` 几乎全是
      动画/音效，唯一的数值动作是"有镀层就移除它"，引擎用 ``action`` 按名字派发
      （``"remove_plating"``，见 ``core._run_stun_action``）；
    * ``nextMoveId`` —— **下一回合**固定走哪一招（拉瓦金 `SLASH_MOVE`、睡甲虫
      `ROLL_OUT_MOVE`），引擎记进 ``combatant.stun_follow_up`` 并在出招时兑现一次。

    玩家的 stun 不适用（真机 ``StunInternal`` 对玩家直接抛异常）。
    """
    combatant.power_flags["stunned"] = 1
    if action:
        combatant.stun_action = action
    if follow_up:
        combatant.stun_follow_up = follow_up
    suffix = f"（{source}）" if source else ""
    events.append(f"{combatant.name} 被击晕，下回合无法行动{suffix}")


def spawn_enemy(state: "CombatState", eid: str, events: list[str],
                hp: int | None = None) -> object | None:
    """``CreatureCmd.Add<T>(combatState, slotName)``：**战斗中召唤**一个敌人。

    与 ``start_combat`` 里的建怪逻辑保持一致：血量从 ``up_front`` 流掷点、
    挂上真机状态机、施加固有属性。``eid`` 不在怪物表里就返回 ``None``
    （调用方负责上报，不静默生成一只空怪）。
    """
    from . import core
    from .content import ENEMY_DB

    definition = ENEMY_DB.get(eid)
    if definition is None or definition.hp is None:
        return None
    lo, hi = definition.hp
    if hp is None:
        hp = int(state.hidden.rng.randint(lo, hi, "up_front"))
    ai_record = getattr(definition, "ai_record", None) or {}
    index = len(state.enemies)
    enemy = core.EnemyState(name=f"{definition.name}#{index}", hp=hp, max_hp=hp,
                            eid=eid, machine=core.build_machine(ai_record or None))
    # ⭐ 召唤也要有**空闲槽位**（真机 `EncounterModel.GetNextOpenSlot` 语义）——
    # 用 `index` 当槽位名会让"第 3 只怪"在有人死后仍然叫 third，
    # 而 `Exoskeleton` 这类怪的条件就是按槽位名分支的（审计 F05）。
    enemy.slot_name = core.next_open_slot(state)
    state.enemies.append(enemy)
    core.init_enemy_ai(enemy, index)
    for pid, amount in definition.innate_powers:
        if amount:
            enemy.add_power(pid, amount)
    core._choose_intent(state, enemy)
    events.append(f"{definition.name} 加入战斗（槽位 {enemy.slot_name}）")
    return enemy


def dies_to_doom(combatant: "Combatant") -> bool:
    """``DoomPower.IsOwnerDoomed``：层数 ≥ 当前生命时被处决。"""
    amount = combatant.power("doom")
    return amount > 0 and combatant.hp <= amount


def coverage() -> dict:
    """能力实现覆盖率（给 ``content.engine_coverage()`` 用）。"""
    from .content import POWERS
    implemented = sorted(set(POWERS) & IMPLEMENTED)
    return {
        "powers_total": len(POWERS),
        "powers_implemented": len(implemented),
        "powers_unimplemented": len(POWERS) - len(implemented),
        "implemented": implemented,
    }
