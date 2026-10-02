"""规则引擎：状态、隐藏信息隔离、SL 快照语义、动作与掩码。

设计约束（``docs/02`` §2.2）：

1. ``restore`` 必须**保留隐藏 RNG 状态**——这是 SL 语义的物理基础。
   重掷只允许通过显式的 ``reroll_hidden()``，且仅供测试。
2. 状态可深拷贝（快照 / 重开 / 并行 worker / 回放）。
3. ``step`` 是纯函数：不依赖全局、不读墙钟、不读文件。
"""

from __future__ import annotations

import copy
import itertools
from decimal import Decimal
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from .content import CARD_DB, ENEMY_DB, STARTING_DECK, CardDef, Effect, EnemyDef, MoveDef
from .rng import RngSet

PLAYER_MAX_HP = 80
BASE_ENERGY = 3
CARDS_PER_TURN = 5
#: 手牌上限（真机 ``CardPile.MaxCardsInHand``）。`Anointed` 的"补满手牌"要用它。
MAX_HAND_SIZE = 10


# ==========================================================================
# 实体
# ==========================================================================
#: 卡牌实例的唯一 id 来源。``uid`` 参与相等性比较，而牌堆操作用的是
#: ``list.remove(card)``（靠 ``==`` 找第一个相等的元素）—— 若多张卡 uid 相同，
#: 移除的可能**不是**目标对象。
#: ⚠️ 实测踩过：生成的卡（`add_card`）一律用默认 uid=0，于是"选中的那张"
#: 与"被移除的那张"可能不是同一张。uid 不进观测（不泄漏隐藏信息），
#: 所以全局自增是安全的。
_UID_COUNTER = itertools.count(1)


def next_uid() -> int:
    return next(_UID_COUNTER)


@dataclass
class CardInstance:
    cid: str
    upgraded: bool = False
    uid: int = field(default_factory=next_uid)
    #: **苦痛**（``AfflictionModel``）：战斗内贴在这张牌上的标记。
    #: 标记本身不含逻辑，真正的行为在对应能力里查询（``tangled`` 改费用、
    #: ``smoggy`` / ``ringing`` 禁打出、``hex`` 给虚无）。
    #: 只在**本场战斗**有效 —— 与附魔（永久）不同，战斗结束自然消失。
    affliction: str | None = None
    #: ⭐ **永久卡牌关联**（``docs/13`` §3.2）：这张战斗内副本克隆自
    #: 永久牌组里的哪一张（那个实例的 ``uid``）。``None`` = 战斗内生成的牌。
    #:
    #: 有了它，"战斗里发生的**永久**改牌（升级/附魔）"才有明确的回写目标，
    #: 而不是靠 ``cid`` 去猜 —— 同名的两张牌改错了哪一张，是**静默**的错误。
    #: 与 ``uid`` 一样属于环境内部信息，不进观测。
    link_uid: int | None = None
    #: ⭐ **实例级关键字**（``CardCmd.ApplyKeyword(card, CardKeyword.X)``）。
    #: 真机可以在**战斗内**给某一张牌单独加关键字（``PhantomBladesPower`` 给飞刀
    #: 加 ``Retain``），而 ``CardDef.keywords`` 是卡牌的固有属性、改不得。
    #: 战斗结束随战斗副本一起丢弃 ✓（永久牌组里的是另一批实例）。
    keywords: tuple[str, ...] = ()
    #: ⭐ **本回合可免费打出**（``CardModel.SetToFreeThisTurn()``，CardModel.cs:1267）。
    #:
    #: 真机实现是 ``EnergyCost.SetThisTurnOrUntilPlayed(0)`` —— 往
    #: ``CardEnergyCost._localModifiers`` 里塞一条
    #: ``LocalCostModifier(0, Absolute, EndOfTurn | WhenPlayed)``：
    #:
    #: * **Absolute** 语义是"**替换**卡面费用"，不是"再减一笔"。所以它必须作用在
    #:   :meth:`effective_cost`（卡面费用那一层），而**不是** :func:`play_cost`
    #:   的最后一步 —— 否则"免费牌 + 纠缠（苦痛加价）"会变成 0 费，
    #:   真机是 ``0 + Amount``（``Hook.TryModifyEnergyCostInCombat`` 跑在
    #:   LocalCostModifier **之后**）。
    #: * 清除条件是 ``EndOfTurn | WhenPlayed``：回合结束（``EndOfTurnCleanup``，
    #:   ``CombatManager.cs:1697/1813``）或**这张牌被打出后**
    #:   （``AfterCardPlayedCleanup``，``CardModel.cs:2007``）—— 谁先到算谁。
    #: * X 费卡的 Canonical 在 CardEnergyCost 构造里归零，局部修饰可以写入；
    #:   但 GetWithModifiers 的 CostsX 分支跳过修饰，GetAmountToSpend 仍取全部能量。
    free_this_turn: bool = False
    #: ⭐ **下一次打出后强制消耗**（``CardModel.ExhaustOnNextPlay``，
    #: 由 ``CardPileCmd.AutoPlayFromDrawPile(…, forceExhaust: true)`` 设置，
    #: 使用者是 ``Havoc``）。丢掉它，那张牌打完会进弃牌堆、还能再被抽到 ——
    #: 比真机强，而且日志完全正常。
    exhaust_on_next_play: bool = False
    #: ConfusedPower.AfterCardDrawn → CardEnergyCost.SetThisCombat：本场的绝对费用。
    #: 不随出牌/回合末清除；战斗副本重建时不复制，避免随机费用写进永久牌组。
    combat_cost_override: int | None = None

    def definition(self) -> CardDef:
        return CARD_DB[self.cid]

    def clone(self) -> "CardInstance":
        """复制这张牌的**形态**（新 uid），对应真机 ``CardModel.CreateClone()``。

        用途是 `NightmarePower.SetSelectedCard`：把选中的卡存成快照，
        下回合按 ``Amount`` 复制若干份入手。字段一个一个抄而不是 ``copy.copy``：
        **uid 必须换新**（同名副本在牌堆里要靠 uid 区分），其余形态
        （升级 / 苦痛 / 实例关键字 / 本场费用覆盖）照抄。
        """
        return CardInstance(
            cid=self.cid, upgraded=self.upgraded,
            affliction=self.affliction, link_uid=self.link_uid,
            keywords=tuple(self.keywords), free_this_turn=self.free_this_turn,
            exhaust_on_next_play=self.exhaust_on_next_play,
            combat_cost_override=self.combat_cost_override)

    def cost(self) -> int:
        return self.definition().cost

    def playable(self) -> bool:
        """卡牌**本身**能否打出（不看苦痛）。

        ⚠️ **不可打出有两个来源，都必须判**（``CardModel.CanPlay``，CardModel.cs:1734）：

        1. ``CardKeyword.Unplayable`` 关键字 —— 真机直接给出
           ``UnplayableReason.HasUnplayableKeyword``。**这一条与费用无关**：
           实测 ``normality``（费用 -1）与 ``debris``（1 费）都是正/负费各异的
           不可打出牌，只看费用会把它们判反。
        2. **负费用**（历史表示法，burn / decay 这类）。``is_x_cost`` 是权威例外
           （``Cascade`` 是 X 费却写 ``base(-1, …)``）。

        还要"有效果"——没效果的空卡打出去只会白扣能量。
        ⚠️ 这是**模拟器自己的约定**，不是真机的：``CardModel.CanPlay``
        只看关键字 / 资源 / ``ShouldPlay`` / ``IsPlayable``，**不要求"有效果"**。
        唯一的例外见 :func:`card_playable`（``Enthralled`` 覆写了 ``ShouldPlay``
        并显式允许自己被打出）。
        """
        from . import keywords as keyword_rules
        d = self.definition()
        if keyword_rules.is_unplayable(d):
            return False
        if d.is_x_cost:
            return len(d.effects) > 0
        return d.cost >= 0 and len(d.effects) > 0

    def effective_cost(self, energy: int) -> int:
        """打出这张牌**实际**要花多少能量。

        ⚠️ X 费卡花的是**全部**能量 —— 真机把 X 定义为"当前能量"。
        所以 `whirlwind` / `skewer` 这类卡不是 0 费（它们的构造参数恰好是 0），
        `cascade` 也不是不可打出（它的构造参数是 -1）。
        两个方向都错得很远：一个是白嫖 9 张牌，一个是把能用的牌锁死。

        苦痛带来的费用修正由 :func:`cost_with_affliction` 施加（``Entangled`` +1）。
        """
        if self.definition().is_x_cost:
            # X 费卡的 GetAmountToSpend 直接取全部能量，不读局部费用修饰。
            # 当成免费会让 X 变成"当前能量"却
            # 一点能量都不花 —— 静默变强，而且日志完全正常。
            return energy
        if self.free_this_turn:
            return 0
        if self.combat_cost_override is not None:
            return self.combat_cost_override
        return self.definition().cost


def _allows_negative(name: str) -> bool:
    """能力是否允许为负——**以真机能力表为准**（``allow_negative`` 字段）。"""
    from .content import POWERS
    definition = POWERS.get(name)
    if definition is not None:
        return definition.allow_negative
    return name in Combatant.NEGATIVE_POWERS


def _instance_type(name: str) -> str:
    """能力的 ``InstanceType``（真机 ``PowerModel.InstanceType``）。

    ``none`` = 同类叠层（默认）；``instanced`` = 每次施加新实例；
    ``instancedperapplier`` = 每个施加者一个实例。取值来自源码抽取
    （``content.POWERS[pid].instance_type``），引擎不硬编码。
    """
    from .content import POWERS
    definition = POWERS.get(name)
    return definition.instance_type if definition is not None else "none"


@dataclass
class PowerInstance:
    """一个**能力实例**（真机 ``PowerModel`` 的实例）。

    只有 ``InstanceType != None`` 的能力才有多个；``TheBombPower`` 的源码注释是
    最好的说明 —— "放第二个炸弹，你要的是**另一个从 3 开始倒数**的炸弹，
    而不是把第一个的层数变成 6"。

    ⚠️ 载荷（``vars``）必须**随实例**走：两个炸弹的伤害可以不同（一张升级过、
    一张没有），共用一个 ``power_vars[pid]`` 会让第二个炸弹用第一个的数值 —— 静默。
    """

    pid: str
    amount: int = 0
    #: 这个实例自己的动态变量（``SetDamage`` / ``SetBlock`` 写的就是它）。
    vars: dict = field(default_factory=dict)
    #: 施加者（真机 ``PowerModel.Applier``）；``instancedperapplier`` 靠它归组。
    applier: object | None = None


@dataclass
class Combatant:
    name: str
    hp: int
    max_hp: int
    block: int = 0
    powers: dict[str, int] = field(default_factory=dict)
    #: 某个能力**是在第几回合被加上的**。``RitualPower`` 要"跳过刚获得的那一次
    #: 回合结束"，没有这个记录就只能靠猜。
    power_applied_turn: dict[str, int] = field(default_factory=dict)
    #: 某个能力**在本回合已经触发过**（"每回合一次"型，``SkittishPower``）。
    power_used_turn: dict[str, int] = field(default_factory=dict)
    #: ``CreatureCmd.Escape`` 的标记：**脱离战斗 ≠ 死亡**（不算击杀、不进统计）。
    escaped: bool = False
    #: ``SurroundedPower`` 的朝向（``"left"`` / ``"right"``）——
    #: 被包围者只在**背后**被打时才吃 ×1.5。
    power_facing: str = "right"
    #: 能力的通用整数标志位（开关型状态），例如 ``nemesis`` 的"这回合给不给无形"。
    #: 与 ``power_used_turn``（每回合一次）分开：那个存的是**回合号**，这个是开关。
    power_flags: dict[str, int] = field(default_factory=dict)
    #: ⭐ **能力实例变量**（``pid → {变量名: 值}``）。对应真机的
    #: ``PowerModel.DynamicVars`` 里那些"由卡牌在施加时告诉能力"的量：
    #: ``TheBombPower.SetDamage(卡牌 BombDamage)``（爆炸多少伤害）、
    #: ``ToricToughnessPower.SetBlock(实际获得的格挡)``。
    #: 与 ``power_flags`` 分开是因为它是**能力自己的动态变量**（有名字、可能有默认值），
    #: 而 ``power_flags`` 是引擎内部的开关位。
    power_vars: dict[str, dict[str, int]] = field(default_factory=dict)
    #: ⭐ **能力的施加者**（``pid → 施加它的单位``），对应真机 ``PowerModel.Applier``。
    #: 一小撮减益的结束条件是"**施加它的那个单位**死了"
    #: （``ConstrictPower`` / ``ShrinkPower`` / ``HexPower`` 的 ``AfterDeath``
    #: 都是 ``if (creature == base.Applier)``）—— 不记施加者，那类能力永远不结束。
    powers_applier: dict[str, "Combatant"] = field(default_factory=dict)
    #: ⭐ **分实例能力的实例表**（``pid → [PowerInstance, …]``，S01）。
    #: 真机 ``InstanceType != None`` 的能力每次施加**新建实例**（各自 Amount /
    #: DynamicVars / Applier）；:attr:`powers` 里那个数字是它们的**总和镜像**，
    #: 所以观察层与既有查询一行都不用改。见 :class:`PowerInstance`。
    power_instances: dict[str, list] = field(default_factory=dict)
    #: ``possess_strength`` / ``possess_speed`` 偷来的数值**欠账**：
    #: 拥有者死亡时要按账目还回去。不记这笔账，偷来的力量就永久留下了。
    powers_debts: dict[str, int] = field(default_factory=dict)
    #: 死亡**已经处理过**（``mark_dead`` 的统一入口跑过了）。
    #: 不能用 ``hp > 0`` 代替：伤害路径是先扣血再调 ``mark_dead``，
    #: 那时 hp 已经是 0 了。
    death_processed: bool = False
    #: 反向引用：真机 ``Creature.CombatState``。能力层要"改层数时通知别人"
    #: （``Hook.AfterPowerAmountChanged``），而 `add_power` 是全仓库的**唯一**改层数
    #: 入口 —— 没有这个引用就得把 state 一路透传 20 多处。
    #: ⚠️ 只在 `start_combat` 里赋值；它不参与观测（`observe.py` 只读显式字段）。
    state: object | None = None

    def alive(self) -> bool:
        return self.hp > 0

    def power(self, name: str) -> int:
        return self.powers.get(name, 0)

    #: 内置兜底：没有真机能力表时哪些能力允许为负。
    #: ⚠️ 真机有 **5 个** allow_negative 能力（``dexterity / focus / shriek / shrink /
    #: strength``）。这里只是占位兜底，加载真机内容后以 ``content.POWERS`` 为准——
    #: 硬编码会漏（实测我第一版就漏了 focus / shriek / shrink）。
    NEGATIVE_POWERS = frozenset({"strength", "dexterity"})

    def add_power(self, name: str, amount: int, applier=None,
                  instance: "PowerInstance | None" = None) -> None:
        """改层数（真机 ``PowerCmd.Apply`` 与 ``PowerCmd.Decrement`` 的共同落点）。

        ⚠️ 每改一次都要发 ``Hook.AfterPowerAmountChanged``：真机的两个调用点
        （``PowerCmd.cs:160`` 施加、``:249`` 修改）都在**非零**变动时通知，
        而 ``Decrement`` 走的就是 ``ModifyAmount(..., applier: null)``
        （``PowerCmd.Decrement``）—— 所以递减也必须通知，且 ``applier=None``。

        ⭐ 施加者要**记下来**（真机 ``PowerModel.Applier``）：一堆减益
        （``ConstrictPower`` / ``ShrinkPower`` / ``HexPower``）的结束条件是
        "**施加它的那个单位**死了"。只把 applier 透传给
        ``AfterPowerAmountChanged`` 而不存，那类能力就永远等不到结束条件。

        ⭐⭐ **归零时的顺序**：真机是 ``SetAmount(newAmount)`` →
        ``Hook.AfterPowerAmountChanged``（``PowerCmd.cs:241/249``）→
        **之后**才 ``if (power.ShouldRemoveDueToAmount()) await Remove(power)``
        （``:251``）。引擎原来是"先 pop 再通知"，于是"把自己扣到 0"的能力
        **自己收不到这次通知**（总线按持有者的能力表分发，能力已经不在表里了）——
        临时属性族正是靠这次通知把内部属性还回去，先 pop 就让它永远不还
        （``fade`` 叠到 0 时敏捷永久留下）。所以这里先按真机落上数值（含 0 / 负数）、
        通知完再按 ``ShouldRemoveDueToAmount`` 的同一判据摘掉；
        通知期间被别的处理器重新加上去（>0）就不再摘 —— 与真机判据一致。
        """
        if _instance_type(name) != "none":
            # ⭐ **分实例能力**（`content.POWERS[pid].instance_type`，S01）：
            # 每次施加新建实例、载荷各归各的，总和镜像回 `powers[pid]`。
            self._apply_power_instance(name, amount, applier, instance)
            return
        before = self.power(name)
        total = before + amount
        removed = total == 0 or (total < 0 and not _allows_negative(name))
        self.powers[name] = total
        if amount > 0 and applier is not None:
            # 重新施加会覆盖施加者（真机 `Applier` 是可变的）。
            self.powers_applier[name] = applier
        delta = self.power(name) - before
        if delta and self.state is not None:
            from . import powers as power_rules
            power_rules.on_power_amount_changed(
                self.state, name, delta, applier=applier, owner=self)
        # ``ShouldRemoveDueToAmount``：钩子**之后**再判一次（可能已被重新加上）
        if removed and (self.power(name) == 0
                        or (self.power(name) < 0 and not _allows_negative(name))):
            self.powers.pop(name, None)
            self.powers_applier.pop(name, None)

    # ---- 分实例能力（InstanceType != None，S01）--------------------------
    def _apply_power_instance(self, name: str, amount: int, applier,
                              instance: "PowerInstance | None") -> None:
        """分实例能力的施加 / 定向递增递减 / 移除。

        三种情形：

        * ``instance`` 给定 —— **定向**操作。能力处理器里的
          ``PowerCmd.Decrement(this)`` / ``Remove(this)`` 作用的就是
          **它自己那一个**实例（`TheBombPower` 的两个炸弹各自倒计时）；
        * 没给实例且 ``amount > 0`` —— 施加。``instanced`` 每次**新建**一个；
          ``instancedperapplier`` 先找**同一施加者**的实例，找到就叠上去
          （`OblivionPower`：两个玩家互不干扰，同一个人连放两张要叠加）；
        * 没给实例且 ``amount < 0`` —— 从**最后一个实例**扣（调用方没指定时无法
          更精确；真机的 `Decrement` 永远带 `this`，所以这条只服务"没有实例概念的
          调用点"，例如测试或药水）。

        ⚠️ 每改一次都要发 `Hook.AfterPowerAmountChanged`（与普通能力同一条规则），
        否则"层数变化时触发"的能力对分实例能力整条失效。
        """
        instances = self.power_instances.setdefault(name, [])
        before = self.power(name)
        if instance is not None:
            instance.amount += amount
            if instance.amount <= 0 and not _allows_negative(name):
                if instance in instances:
                    instances.remove(instance)
        elif amount > 0:
            target = None
            if _instance_type(name) == "instancedperapplier" and applier is not None:
                target = next((item for item in instances
                               if item.applier is applier), None)
            if target is not None:
                target.amount += amount
            else:
                fresh = PowerInstance(pid=name, amount=amount, applier=applier)
                instances.append(fresh)
                # `set_power_var` 写的是"刚创建的那个实例"（真机 `Apply(...).SetXxx()`），
                # 所以 `power_vars[pid]` 指向**最新实例**的 vars 字典（同一个对象）。
                self.power_vars[name] = fresh.vars
                if applier is not None:
                    self.powers_applier[name] = applier
        else:
            remaining = -amount
            while remaining > 0 and instances:
                last = instances[-1]
                take = min(remaining, last.amount)
                last.amount -= take
                remaining -= take
                if last.amount <= 0:
                    instances.remove(last)
        self._sync_instance_mirror(name)
        delta = self.power(name) - before
        if delta and self.state is not None:
            from . import powers as power_rules
            power_rules.on_power_amount_changed(
                self.state, name, delta, applier=applier, owner=self)

    def _sync_instance_mirror(self, name: str) -> None:
        """把实例表的**总和**写回 ``powers[pid]``。

        观察层、`power()`、合法动作、内容门禁读的都是 ``powers`` —— 有了镜像，
        S01 就不需要改那些地方一行代码。
        """
        instances = self.power_instances.get(name) or []
        total = sum(item.amount for item in instances)
        if total == 0 and not _allows_negative(name):
            self.powers.pop(name, None)
            self.power_instances.pop(name, None)
            self.powers_applier.pop(name, None)
            return
        self.powers[name] = total

    def power_instance(self, name: str) -> "PowerInstance | None":
        """``instanced`` 能力的**最新**实例（`Remove(this)` / 定向递减的落点）。"""
        instances = self.power_instances.get(name) or []
        return instances[-1] if instances else None

    def remove_power_instance(self, name: str, instance: "PowerInstance") -> None:
        """移除**指定的一个**实例（真机 ``PowerCmd.Remove(this)``）。

        ⚠️ 不能走 `add_power(name, -amount)`：那会从**最后一个**实例扣，
        当两个实例的层数不同时减错对象（而且不报错）。
        """
        instances = self.power_instances.get(name) or []
        before = self.power(name)
        if instance in instances:
            instances.remove(instance)
        self._sync_instance_mirror(name)
        delta = self.power(name) - before
        if delta and self.state is not None:
            from . import powers as power_rules
            power_rules.on_power_amount_changed(
                self.state, name, delta, applier=None, owner=self)


@dataclass
class Intent:
    """敌人意图——**公开信息**（屏幕上直接显示）。见 docs/01 §1.2 的 L1。"""

    kind: str
    mid: str
    value: int
    times: int


@dataclass
class EnemyState(Combatant):
    eid: str = ""
    intent: Intent | None = None
    move_history: list[str] = field(default_factory=list)
    #: 真机出招状态机（``monster_ai.MoveStateMachine``）。有它就用它——
    #: 这是"机制与原版一致"的关键；没有才退回旧的近似逻辑。
    machine: object | None = None
    #: ⭐ ``CreatureCmd.Stun(creature, stunMove, nextMoveId)`` 的两个附加参数：
    #: ``stun_action`` 是这一回合要执行的唤醒动作（按名字派发，见
    #: :func:`_run_stun_action`），``stun_follow_up`` 是**下一回合**固定要走的招式 id。
    stun_action: str = ""
    stun_follow_up: str = ""
    #: 下一回合**强制**走的招式 id（眩晕的后续招式在这里兑现一次后清空）。
    forced_move: str = ""
    #: **复活中**（``ShouldCreatureBeRemovedFromCombatAfterDeath`` → false）。
    #: 血量已经归零，但**留在场上**、不会被移除、也**打不到**；
    #: 到它的回合会按能力指定的量回血并恢复。
    #: 少了这个标记，"死了又活"的怪（幻影 / 再附着 / 试验体）会直接被移出战斗。
    reviving: bool = False
    #: 复活时回多少血（``None`` = 回满，``reattach`` 是回 ``Amount``）。
    revive_amount: int | None = None
    #: ⭐ **遭遇槽位名**（审计 F05）。真机是遭遇定义给的（"first" / "second" /
    #: "third" / "fourth" / "fifth"，``BowlbugsNormal`` 例外地是 first/middle/last）。
    #: 空 = 按位置序数现算。把它当"第几个敌人"是错的：``Exoskeleton`` 的条件就是
    #: ``SlotName == "third"``，三只怪里第 3 只被判成 second 就永远走错分支。
    slot_name: str = ""
    #: ⭐ **怪物专属持久标志**（``HasAmalgamDied`` / ``IsOffBalance`` /
    #: ``HasBeetleCharged``…）。源码里是实例私有字段，由招式副作用写入；
    #: 出厂值为 ``False``（C# 默认值），但**显式记录**，这样"缺值"能被发现。
    ai_flags: dict[str, bool] = field(default_factory=dict)
    #: 专属计数器（`_curseOfKnowledgeCounter` 这类）。缺失时条件求值报错。
    ai_counters: dict[str, int] = field(default_factory=dict)

    def definition(self) -> EnemyDef:
        return ENEMY_DB[self.eid]

    def alive(self) -> bool:
        # ⚠️ 复活中的单位**算活着**：否则战斗会在它复活前就判定结束。
        return self.hp > 0 or self.reviving

    def targetable(self) -> bool:
        """能不能被玩家选中 / 被打到（``ShouldAllowHitting`` → false 时不能）。"""
        return self.hp > 0 and not self.reviving


# ==========================================================================
# 隐藏信息：绝不可进入观测
# ==========================================================================
class Hidden:
    """**所有 L3 级（未揭示）信息的容器。**

    ``docs/01`` §1.2：这个类里的任何字段都不能出现在 ``Observation`` 里，
    也不能被特征化路径 import。CI 里有静态检查（见 tests/test_anticheat.py）。
    """

    __slots__ = ("rng", "move_queue", "future_drops")

    def __init__(self, rng: RngSet) -> None:
        self.rng = rng
        self.move_queue: list[str] = []      # 敌人未来行动队列（L3）
        self.future_drops: list[str] = []    # 未来掉落表（L3）


# ==========================================================================
# 战斗状态
# ==========================================================================
@dataclass
class PendingSelection:
    """等待玩家选牌：效果执行到这里**挂起**，选完再续跑。

    真机的效果序列是 `抽牌 → 选牌 → 弃牌`，所以选牌必须**落在序列的正确位置**
    上。用"挂起 + 续跑"而不是"先跑完再补一刀"，顺序才不会错
    （``acrobatics`` 是"抽 3 再弃 1"，顺序反了会变成"弃 1 再抽 3"）。

    候选牌**每次从牌堆现算**，不存下标快照 —— 选一张之后下标会移位，
    存快照就会选错牌（而观察层看到的永远是正确的候选，错的是引擎）。
    """

    purpose: str                  # discard | exhaust | to_hand | to_draw
    source_pile: str              # hand | discard | draw
    remaining: int                # 还要选几张
    position: str = ""            # to_draw 时的 `top` 等
    rest: tuple = ()              # 选完之后要继续跑的效果
    target_index: int = 0
    played_card: object | None = None
    #: ⭐ **这次挂起发生在谁的执行里**（S02）。卡牌路径是玩家（`play_card` 传
    #: `state.player`）；怪物出招是那只怪。续跑时 `_apply_effects` 的 `source`
    #: 必须用它 —— 怪物效果的 `target="enemy"` 是**以怪物为视角**的，
    #: 用玩家当 source 会把"打玩家"变成"打它自己"（且完全不会报错）。
    actor: object | None = None
    #: ⭐ **显式候选**（S02 / `choose_one`）：这些卡是 `CombatState.CreateCard`
    #: 出来的**临时实例**，不在任何牌堆里。非空时 :meth:`candidates` 直接返回它。
    offered: tuple = ()
    #: 与 :attr:`offered` **一一对应**的效果组：选中第 i 张就执行第 i 组
    #: （真机是那张卡的 `IChoosable.OnChosen`）。存在这里而不是靠 cid 回查，
    #: 因为同一个 cid 在不同轮次有不同载荷（`Disintegration` 的 6/7/8）。
    choice_effects: tuple = ()
    #: ⭐ **选完之后要整个移除的能力 pid**（真机 `PowerCmd.Remove(this)`）。
    #: 不能在挂起那一刻移除：真机是**选完才** `await PowerCmd.Remove`，
    #: 提前移除会让"选牌界面还开着、能力已经没了"（`ForegoneConclusionPower`）。
    remove_power_after: str = ""
    #: 打出时消耗的能量（X 费卡的 X），续跑时要带过去
    x_value: int = 0
    #: 打出这张牌时已经产生的事件（续跑时要接上同一批）
    prefix_events: list[str] = field(default_factory=list)
    #: 被打出的牌是否已经移出手牌（挂起时）—— 完成时要落进正确的牌堆
    card_consumed: bool = False
    #: ⭐ 这张牌**还要再结算几次**（``OneTwoPunchPower`` 让攻击牌多打一次）。
    #: 挂起（选牌）时把剩余次数带过去，续跑时接着打完 —— 少了它，
    #: "多打一次"的牌一旦带选牌效果就只结算一次，而日志一切正常。
    plays_remaining: int = 0
    #: ⭐ 落点**快照**（``(牌堆, 改写过落点的能力)``）。真机在"开始结算之前"
    #: 就把 ``resultLocation`` 算好存进局部变量（``CardModel.cs:1882``，早于
    #: ``History.CardPlayStarted``），一直用到整张牌结算完。挂起时要把这个
    #: 快照带去续跑：重算会把自己算进"本回合已开始的牌数"里，
    #: `NostalgiaPower` 于是少改写一张（且不会报错）。
    result_location: object | None = None

    def candidates(self, state: "CombatState") -> list[CardInstance]:
        """候选牌。**顺序 = 动作下标**，观测侧必须用同一个顺序。

        ⚠️ 顺序必须是**公开且稳定**的（``docs/13`` §8.1）。

        这一条曾经两边不一致：观测按 ``(cid, upgraded)`` 排序给出候选，
        而动作下标是这里的**原始顺序** —— 实测手牌
        ``[bash, strike, strike, defend, defend]`` 在观测里变成
        ``[bash, defend, defend, strike, strike]``，于是"选第 1 张"在
        观测里是 strike、在引擎里是 strike …… 一换牌序就会**选错牌**，
        而且不会报错。修法是让顺序只有一处定义：就这里。

        * ``hand`` —— 手牌顺序本身就是玩家看到并排列的顺序，直接用；
        * ``discard`` —— 玩家能翻开查看弃牌堆，且同名牌不可区分，规范排序即可；
        * ``draw`` —— **抽牌堆顺序是 L3 隐藏信息**，绝不能成为候选排列。
          按 ``(cid, upgraded)`` 规范排序，相同牌之间的相对顺序保持稳定
          （它们对玩家完全不可区分，因此不泄漏任何信息）。
        """
        if self.source_pile == "hand":
            return list(state.hand)
        if self.offered:
            # ⭐ 显式候选（`choose_one`）：顺序**就是**界面上的顺序，不排序 ——
            # 排序会让"左/右"与动作下标错位，选中的卡与看到的卡不是同一张。
            return list(self.offered)
        pile = state.discard if self.source_pile == "discard" else state.draw_pile
        return sorted(pile, key=lambda card: (card.cid, card.upgraded))


@dataclass
class EnemyActionFrame:
    """敌方回合**挂在玩家选择上**时的进度（S02 的最小形态）。

    真机的怪物动作是 async 序列：`CurseOfKnowledgeMove` 在循环里 `await` 玩家的
    二选一，await 期间整个战斗状态原样停在那里。引擎的 `step` 是同步的，
    所以把"跑到哪了"存成**数据**，由 :func:`_resume_enemy_turn` 接着跑。

    ⚠️ 只记"哪个敌人、哪一招、招式的收尾标记"三件事：该招**剩余的效果**
    已经在 `PendingSelection.rest` 里（`_apply_effects` 挂起时存好的），
    重复存两份迟早会漂移。
    """

    #: 正在出招的敌人在 `state.enemies` 里的下标。
    enemy_index: int
    #: 招式 id（`Move.mid`）—— 用来在恢复时核对"还是不是同一招"。
    move_mid: str
    #: `Move.kills_self`：招式自带的自爆，正常流程在效果跑完后执行；
    #: 挂起续跑时必须补上，否则自爆怪少死一次（而日志一切正常）。
    kills_self: bool = False


@dataclass
class DrawFrame:
    """**抽牌循环中途**的帧：抽到空堆洗牌时，`AfterShuffle` 的处理者要玩家选牌。

    真机 ``CardPileCmd.DrawInternal`` 的循环里是 ``await ShuffleIfNecessary(...)``
    —— ``await`` 期间整个抽牌（以及调用它的 ``SetupPlayerTurn``）停在那里。
    引擎的 ``draw_cards`` 是同步的，所以洗牌触发选择时要把"**还要抽几张**"存下来，
    由 :func:`_resume_after_selection` 接着抽。

    ⚠️ 不存这个帧的后果是"先把牌抽完、再让玩家选" —— 手牌张数与顺序都和真机不同，
    而且没有任何报错（`StratagemPower` 的整条价值就在"洗牌后从抽牌堆捞牌"）。
    """

    #: 还没抽的张数（洗牌那一刻的剩余量）。
    #: ``-1`` = **整段手牌抽取都还没开始**（挂起发生在 ``BeforeHandDraw``，
    #: 那时连 ``ModifyHandDraw`` 都还没算 —— ``ForegoneConclusionPower`` 就是这种）。
    remaining: int
    #: 这一轮抽牌是不是"回合开始的手牌抽取"（`Hook.AfterCardDrawn` 要看它）。
    from_hand_draw: bool = False
    #: ⭐ 选完之后还要不要回到**玩家回合开始的剩余步骤**：
    #: `start_player_turn` 里抽牌之后还有 `AfterPlayerTurnStart`、
    #: `AfterSideTurnStart`、球的回合开始、`AutoPrePlay` —— 少一步都是整段流程缺失。
    resume_turn_start: bool = False


def open_pile_selection(state: CombatState, events: list[str], *,
                        purpose: str, source_pile: str, amount: int,
                        position: str = "",
                        actor: "Combatant | None" = None) -> bool:
    """开一次"从某个牌堆选 N 张"的挂起（``CardSelectCmd.FromCombatPile`` 那一族）。

    返回 ``True`` = 已挂起（``state.pending`` 已设置）；``False`` = 没有候选、
    **当场跳过**（真机在没有合法候选时也不向玩家提问）。

    ⚠️ 空候选必须当场跳过：挂一个"没有候选"的选择会让 ``legal_actions()`` 返回
    **空列表** —— 环境既没有动作可发也不会自己往前走，训练直接炸成
    "存在没有任何合法动作的样本"（`_apply_effects` 里同一条注释）。
    """
    pending = PendingSelection(purpose=purpose, source_pile=source_pile,
                               remaining=max(1, int(amount)), position=position,
                               actor=actor)
    if not pending.candidates(state):
        events.append(f"跳过选择（{purpose}：没有候选）")
        return False
    state.pending = pending
    events.append(f"等待选择 {amount} 张牌（{purpose}）")
    return True


#: 真机 ``CardRarity`` 里"允许作为转化结果"的三档（枚举值 2/3/4）。
#: 判据是 ``(uint)(rarity - 8) > 1u``（``CardFactory.cs:193``）：原卡不是
#: ``Status``(8) / ``Curse``(9) 时，只允许 Common/Uncommon/Rare。
TRANSFORM_RARITIES = frozenset({"common", "uncommon", "rare"})
#: 原卡是这三类**稀有度**时，候选池换成 ``ColorlessCardPool``。
#: 出处 ``CardFactory.cs:172``：判据是 ``original.Type != CardType.Quest``（**卡类型**）
#: 与 ``Rarity != Event/Ancient/Token``（**稀有度**）—— 两个维度都要判。
#: ⚠️ 把 Status/Curse 也塞进来是错的：它们的候选来自**自己的池**
#: （StatusCardPool / CurseCardPool），实测会多发一堆状态牌。
TRANSFORM_COLORLESS_FROM = frozenset({"event", "ancient", "token"})


def _card_pool_of(cid: str) -> str:
    """这张卡属于哪个卡池（真机 ``CardModel.Pool`` 的引擎近似）。

    真机是卡对象上的属性；引擎只有"池 → cid 列表"的正向表，所以反查一次。
    找不到返回空串（调用方按"空池"处理，由它决定报错还是拒绝）。
    """
    from . import content
    for name, members in content.CARD_POOLS.items():
        if cid in members:
            return name
    return ""


def transformation_options(state: CombatState, card: CardInstance) -> list[str]:
    """``CardFactory.GetDefaultTransformationOptions``：这张牌能转成哪些牌。

    过滤链照抄源码（``CardFactory.cs:170-212``）：

    1. **候选池**：原卡的 ``Pool``；原卡稀有度 ∈ Quest/Event/Ancient/Token 时
       换成 ``ColorlessCardPool``；
    2. 原卡**不是** Status/Curse 时，候选只允许 Common/Uncommon/Rare（枚举 2/3/4）；
    3. 战斗内 → 只要 ``CanBeGeneratedInCombat``；
    4. **排除原卡自己的 id**；
    5. 玩家数过滤（``FilterForPlayerCount``）—— 单人 profile 下是空操作。

    ⚠️ 这里**只算真机规则**，"引擎准入缩池"由调用方
    :func:`transform_card_random` 做（`docs/13` §7：受限课程预先缩池并记账，
    **不许**"发出来再重抽"）。真机空候选会抛异常
    （``All transformation options provided are invalid!``），引擎交调用方如实拒绝。
    """
    from . import content
    definition = card.definition()
    rarity = definition.rarity
    # ⭐ 换无色池的判据是**两个维度**（`CardFactory.cs:172`）：
    # `CardType.Quest` 看的是卡**类型**，Event/Ancient/Token 看的是**稀有度**。
    colorless = (definition.card_type == "quest"
                 or rarity in TRANSFORM_COLORLESS_FROM)
    pool_name = "ColorlessCardPool" if colorless else _card_pool_of(card.cid)
    options: list[str] = []
    for cid in content.CARD_POOLS.get(pool_name, ()):
        candidate = content.CARD_DB.get(cid)
        if candidate is None:
            continue
        if (rarity not in ("status", "curse")
                and candidate.rarity not in TRANSFORM_RARITIES):
            continue
        if not candidate.can_be_generated_in_combat:
            continue
        if cid == card.cid:
            continue
        options.append(cid)
    return options


def transform_card_random(state: CombatState, card: CardInstance,
                          events: list[str]) -> "CardInstance | None":
    """``CardCmd.TransformToRandom``：把这张牌**原位**换成随机的另一张。

    顺序照抄 ``CardCmd.Transform``（``CardCmd.cs:371-407``）：

    * 候选池 → 用 ``Rng.CombatCardSelection``（引擎的 ``combat_card_selection``
      流）抽一张；
    * **新建卡实例**（真机 ``CreateCard``）：不继承升级 / 附魔 / 苦痛；
    * **保留原来的位置**（真机先 ``RemoveFromCurrentPile()``，再按
      ``(堆类型, 下标)`` 排好序插回去）；
    * 新卡是"刚进入战斗的牌" → 派发 ``AfterCardEnteredCombat``
      （``CardPileCmd.cs:515`` 的判据 ``oldPile == null``），但**不**计入
      ``added_cards``（一换一，总张数不变）。

    返回新卡；``None`` = 这次转化被拒绝（候选为空 / 牌不在任何牌堆里）。
    """
    from . import content, eligibility
    from . import powers as power_rules
    raw = transformation_options(state, card)
    admitted = set(eligibility.admitted_cards())
    options = [cid for cid in raw if cid in admitted]
    if len(options) != len(raw):
        # 受限课程**预先缩池**并把缩池写进日志（`docs/13` §7 的门禁口径）。
        events.append(f"转化候选按准入缩池：{len(raw)} → {len(options)}")
    if not options:
        events.append(f"⚠️ 转化候选池为空（{card.definition().name}），拒绝这次转化")
        return None
    picked = options[state.hidden.rng.next_index("combat_card_selection",
                                                 len(options))]
    replacement = CardInstance(picked)
    for pile in (state.hand, state.draw_pile, state.discard, state.exhaust):
        if card in pile:
            pile[pile.index(card)] = replacement
            break
    else:
        events.append(f"⚠️ 被转化的牌不在任何牌堆里（{card.definition().name}）")
        return None
    events.append(f"转化 {card.definition().name} → {replacement.definition().name}")
    power_rules.on_card_entered_combat(state, events, replacement)
    return replacement


@dataclass(frozen=True)
class PowerTask:
    """选完之后还要跑的**能力动作**（S02 的续跑单元之一）。

    为什么不是普通的"效果"：`ForegoneConclusionPower.BeforeHandDraw` 是
    「洗牌 → 从抽牌堆选 N 张入手 → **整个移除这个能力**」三段，
    最后一段不是效果（`PowerCmd.Remove(this)`），而且第一段（洗牌）本身
    可能被**另一个**能力（`StratagemPower.AfterShuffle`）挂住 —— 嵌套。
    把待办存成数据，:`func:`_resume_after_selection` 就能一层层接着跑。
    """

    #: 待办属于哪个能力（用于记账与"选完移除"）。
    pid: str
    #: 动作种类：目前只有 ``select_to_hand``（从抽牌堆选 ``amount`` 张入手）。
    kind: str
    amount: int
    #: 这个动作跑完要不要**整个移除**该能力（真机 ``await PowerCmd.Remove(this)``）。
    remove: bool = False
    #: 这次挂起发生在谁的执行里（玩家）。
    actor: object | None = None


@dataclass
class CombatState:
    player: Combatant
    enemies: list[EnemyState]
    hand: list[CardInstance]
    draw_pile: list[CardInstance]      # 内部有序；观测层默认只给 bag
    discard: list[CardInstance]
    exhaust: list[CardInstance]
    play_area: list[CardInstance]
    hidden: Hidden
    turn: int = 1
    energy: int = BASE_ENERGY
    #: 每回合能量上限（角色定义给的）。屏幕上是 "3/3" 这种显示，属于公开信息。
    max_energy: int = BASE_ENERGY
    phase: str = "player"              # player | enemy | won | lost
    added_cards: int = 0               # 战斗中新增的卡（Anger 等），用于守恒校验
    #: 本场战斗属于哪个角色。**能力生成牌要用它选卡池**
    #: （`CallOfTheVoidPower` / `CreativeAiPower` / `HelloWorldPower` 走
    #: `Character.CardPool`；用错角色的池子会发出别的角色的牌）。
    character: str = "ironclad"
    ascension: int = 0                 # 当前进阶等级（决定怪物血量/伤害）
    #: ⭐ **最近一次实际获得的格挡**（``CreatureCmd.GainBlock`` 的返回值）。
    #: ``ToricToughnessPower.SetBlock(blockAmount)`` 拿的就是它 ——
    #: 卡面上写的是 5，实际可能是 5×脆弱0.75×倍率 = 7；用卡面值会静默算错。
    last_block_gain: int = 0
    #: ⭐ **本回合已经打出（含自动打出）的攻击牌张数**。
    #: 对应真机的 ``CombatManager.History.CardPlaysStarted`` 按回合/按牌型筛选出来的
    #: 那一份记录 —— ``JugglingPower``（戏法）的私有计数器与它恒等（源码在
    #: ``AfterApplied`` 里用"本回合已开始的攻击牌"初始化，之后每张 +1）。
    #: 玩家回合开始时清零（真机是 ``JugglingPower.AfterSideTurnEnd`` 清零，
    #: 两者对"本回合第几张攻击牌"是同一个量）。
    attacks_played_this_turn: int = 0
    #: **本回合已开始结算的牌数**（真机 ``History.CardPlaysStarted`` 里
    #: ``Actor == 拥有者 && CardPlay.IsFirstInSeries && HappenedThisTurn`` 的计数）。
    #: ``EchoFormPower`` 用它表达"每回合前 Amount 张牌打两次"。
    #: ⚠️ 记的是"**开始**结算的牌"，不是结算次数：一张牌打两次只算 1
    #: （真机靠 ``PlayIndex == 0`` 过滤）。
    cards_played_started_this_turn: int = 0
    #: 其中**攻击/技能**的张数（`NostalgiaPower` 只数这两类：
    #: 源码 `(uint)(type - 1) <= 1u` 覆盖 Attack=1 / Skill=2）。
    attack_skill_plays_started_this_turn: int = 0
    #: 「星」资源（摄政王角色；`PlayerCmd.GainStars`）。0 时对其它角色无影响。
    stars: int = 0
    #: ⭐ **本回合已打出的技能牌张数**（逐次打出 +1，与 `attacks_played_this_turn`
    #: 同一个时机）。`SmoggyPower.AfterCardEnteredCombat` 的判据是源码里的
    #: ``CardPlaysStarted.Any(type == Skill)`` —— 少了这个计数器，"本回合打过技能牌吗"
    #: 就判不出来，只能近似成"打过牌"（那是**静默变强**）。
    skills_played_this_turn: int = 0
    #: ⭐ **本回合已打出的 0 费攻击牌张数**（``CardPlay.Resources.EnergyValue == 0``
    #: 的攻击牌）。`FeralPower.AfterApplied` 用它初始化"本回合已经返还过几张" ——
    #: 少了它，战斗中途贴上"野性"会**多发**几张（把它贴上之前打的也算进去）。
    zero_cost_attacks_played_this_turn: int = 0
    #: ⭐ **当前这次出牌实际花掉的能量**（``CardPlay.Resources.EnergyValue`` /
    #: ``EnergySpent``）。`FeralPower`（0 费攻击牌回手牌）与 `OneForAllPower`
    #: （0 费攻击牌 +Amount）都靠它判"这一张是不是免费打的"。
    #: ``-1`` 表示"当前**不在**一次出牌的结算里"（此时按卡牌的**带修正费用**判）。
    current_play_energy_spent: int = -1
    #: ⭐ **整场战斗**打完的出牌数（``History.CardPlaysFinished.Count()``）。
    #: 与 ``cards_played_started_this_turn`` 不同：那个按回合清零。
    #: `GoldAxe` 的"伤害 = 本场打出的牌数"（运行期公式 `card_plays_total`，
    #: `docs/12` §2.33）读这个 —— 少了它只能按本回合算，**静默算错**。
    card_plays_finished_this_combat: int = 0
    #: ⭐ **当前这次出牌花掉的星**（``CardPlay.Resources.StarsSpent``）。
    #: `BlackHolePower` 的 ``AfterCardPlayed`` 分支（"花了星的那张牌打完 → 群伤"）用它。
    current_play_stars_spent: int = 0
    #: X 星费卡这次花掉的星（``LastStarsSpent`` / 结算用）。
    x_stars_value: int = 0
    #: ⭐ **按标签统计"本回合结算完的出牌"**：``标签 → 张数``。
    #: 真机 ``History.CardPlaysFinished`` 按 ``CardPlay.Card.Tags`` 过滤
    #: （``PhantomBladesPower`` 数的是"本回合已经打完的飞刀"）。
    #: 用**标签**而不是写死"飞刀"：同一条判据将来还要给别的标签用。
    tag_plays_finished_this_turn: dict[str, int] = field(default_factory=dict)
    #: 充能球（缺陷角色）。列表顺序 = 屏幕上的槽位顺序（**左到右**），
    #: 满槽时新球会把**最左边**那个挤掉（激发），见 ``orbs.channel``。
    orbs: list = field(default_factory=list)
    orb_slots: int = 3
    #: 非空 = 正在等待玩家选牌（效果执行挂起中）。
    pending: "PendingSelection | None" = None
    #: ⭐ **S02 执行帧（敌方回合的中途）**：非空 = 敌方回合挂在一次玩家选择上、
    #: 还没跑完。真机是 async 序列（在怪物动作里 `await` 玩家的选择），引擎是同步的，
    #: 所以把"跑到哪了"存成**数据** —— `snapshot()` 走 `copy.deepcopy`，
    #: 闭包与生成器进不去快照，SL 会直接废掉。
    #: 第一个真实用例是 `KnowledgeDemon.CurseOfKnowledgeMove`（三轮二选一）。
    enemy_action: "EnemyActionFrame | None" = None
    #: ⭐ **抽牌循环中途的帧**（S02 的第二种）：抽到空堆洗牌时触发的能力
    #: （`StratagemPower`）要玩家选牌，抽牌必须停在那里等 —— 见 :class:`DrawFrame`。
    draw_frame: "DrawFrame | None" = None
    #: ⭐ **能力动作的待办队列**（S02 的第三种）：`ForegoneConclusionPower` 的
    #: 「洗牌 → 选牌入手 → 移除自己」三段里，洗牌可能先被另一个能力挂住。
    #: 见 :class:`PowerTask`。
    pending_tasks: list = field(default_factory=list)
    #: ⭐ **最近一次"记住这张卡"的选择结果**（``select_card(purpose="remember")``）。
    #: 真机 `Nightmare.OnPlay` 把选中的卡交给
    #: ``PowerCmd.Apply<NightmarePower>(…).SetSelectedCard(card)``；引擎先存在这里，
    #: 紧跟着的 ``set_power_var(from="remembered_card")`` 再取走。
    #: 存的是**克隆快照**（:meth:`CardInstance.clone`），所以原卡之后被升级 /
    #: 转化 / 消耗都不影响这份载荷。
    remembered_card: "CardInstance | None" = None
    #: ⭐ **回合开始流程跑到第几步**（`_finish_player_turn_start` 的续跑下标）。
    #: 非 0 = 回合开始的后半段挂在一次玩家选择上（`EntropyPower` 在
    #: `AfterPlayerTurnStart` 要挑牌转化），剩下的步骤等选完再跑 ——
    #: 真机那里是一个 `await`，整个 `SetupPlayerTurn` 都停着。
    turn_start_step: int = 0
    #: X 费卡的 X（打出时消耗的能量）。挂起时要带过去续跑。
    x_value: int = 0
    #: 本场战斗携带的遗物 id。**引擎不硬编码遗物**，一切由内容层的
    #: ``RELICS`` 驱动（``docs/09``：源码是机制的唯一真相，引擎不是数据）。
    relics: tuple[str, ...] = ()
    #: 房间种类（``monster`` / ``elite`` / ``boss``）—— 决定哪些遗物生效
    #: （``sling_of_courage`` 只在精英战、``pantograph`` 只在 Boss 战）。
    room: str = "monster"
    #: ⭐ **战斗内发生的永久改牌**（``docs/13`` §3.2「退场」一行）：
    #: 键 = 被改的**永久牌** uid（:attr:`CardInstance.link_uid`），
    #: 值 = ``{"op": "upgrade" | "enchant", "value": ...}``。
    #:
    #: 临时改牌（本场战斗升级、临时费用）**不进这里** —— 它们随战斗消失。
    #: 没有这份记录，"永久"与"临时"就只能靠效果实现者自己记得回写，
    #: 而"忘了回写"是静默的（升级在下一场战斗里凭空消失）。
    permanent_card_changes: dict[int, dict] = field(default_factory=dict)
    #: `AmountOnTurnStart`：**回合开始时**各能力的层数快照（真机 `Creature.cs:682`
    #: 在 `BeforeTurnStart` 里给每个能力存一份 `power.AmountOnTurnStart = Amount`）。
    #: 有些能力读的是"回合开始时的层数"而不是当前层数（`HelloWorldPower` 按它生成
    #: 牌，`DrawCardsNextTurnPower` / `SummonNextTurnPower` 用它判"这回合还有没有"）。
    #: 少了它，这些能力只能读当前值 —— 中途被改层数就会错。
    power_amount_on_turn_start: dict = field(default_factory=dict)
    #: 遗物的**战斗内**计数（``docs/13`` §3.2「遗物与持久计数」）。
    #: 退场时由 Run 层按需回写成持久计数；战斗内的临时计数随战斗丢弃。
    relic_counters: dict[tuple[str, str], int] = field(default_factory=dict)
    #: ⭐ **药水槽**（``docs/13`` §3.2 入场行："药水与槽位 | 共用玩家库存"）。
    #: 战斗**不新开一份库存**：Run 层把自己的槽位传进来，战斗里用掉的药水
    #: 通过 :func:`use_potion_at` 直接写回，所以"用掉一瓶下战斗不会回来"
    #: 是结构上成立的，而不是靠调用方记得同步。
    potions: list[str | None] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    def selecting(self) -> bool:
        return self.pending is not None

    # ---- 汇总 --------------------------------------------------------
    def all_piles(self) -> list[CardInstance]:
        return (self.hand + self.draw_pile + self.discard
                + self.exhaust + self.play_area)

    def living_enemies(self) -> list[EnemyState]:
        return [e for e in self.enemies if e.alive()]

    def finished(self) -> bool:
        return self.phase in ("won", "lost")


# ==========================================================================
# 快照
# ==========================================================================
@dataclass
class Snapshot:
    """完整状态副本（**含隐藏 RNG**）。SL 重开靠它实现。"""

    state: CombatState

    def restore_into(self, target_holder: list[CombatState]) -> CombatState:
        clone = copy.deepcopy(self.state)
        target_holder[0] = clone
        return clone


def snapshot(state: CombatState) -> Snapshot:
    return Snapshot(copy.deepcopy(state))


def restore(snap: Snapshot) -> CombatState:
    """恢复状态，**保留隐藏 RNG 状态**。SL 语义的物理基础（docs/01 §1.6 测试 2）。"""
    return copy.deepcopy(snap.state)


def reroll_hidden(state: CombatState, new_seed: int) -> None:
    """⚠️ **仅供测试**：保持所有可见状态不变，重新掷隐藏随机状态。

    用于"投影不变性"与"归因测试"——隐藏信息怎么变，观测都必须纹丝不动。
    真实游戏里不存在这个操作：它对应"如果当初的洗牌是另一副顺序，但屏幕
    上显示的东西完全一样"。
    """
    state.hidden = Hidden(RngSet(new_seed))


# ==========================================================================
# 动作
# ==========================================================================
@dataclass(frozen=True, slots=True)
class Action:
    kind: str                 # play_card | end_turn | select_card | use_potion
    hand_index: int = -1
    target: int = -1          # 敌人下标；-1 表示无需目标
    #: ⭐ **药水槽位**（``use_potion`` 用）。审计 F09：``use_potion()`` 一直
    #: 只是个可调用函数，合法动作与策略动作词表里**没有药水动作** ——
    #: 于是"手里有药水"这件事对策略而言不存在。
    #: 单独一个字段而不是复用 ``hand_index``：两者下标空间不同，
    #: 复用会让"第 0 张牌"和"第 0 瓶药水"在编码里无法区分。
    slot: int = -1

    def __repr__(self) -> str:  # 便于日志
        if self.kind == "end_turn":
            return "end_turn"
        if self.kind == "select_card":
            return f"select_card(hand={self.hand_index})"
        if self.kind == "use_potion":
            return f"use_potion(slot={self.slot},target={self.target})"
        return f"play_card(hand={self.hand_index},target={self.target})"


def potion_actions(state: "CombatState") -> list[Action]:
    """药水动作（``PotionModel.CanUse`` + ``OnUse``）。

    ``docs/13`` §8.2：药水属于统一动作协议的一部分。审计 F09 的缺口是
    "``use_potion()`` 可调用，但合法动作与词表里没有药水动作" ——
    策略看不到药水，自然学不会用。
    """
    from .content import POTIONS

    actions: list[Action] = []
    for slot, pid in enumerate(state.potions):
        if not pid:
            continue
        potion = POTIONS.get(pid)
        if potion is None or potion.effects_incomplete:
            continue                    # 残缺药水不产生动作（不能静默变强/变弱）
        if potion.usage == "automatic":
            continue                    # 自动触发的药水不能主动使用
        if "enemy" in potion.target_type:
            for index, enemy in enumerate(state.enemies):
                if enemy.alive() and enemy.targetable():
                    actions.append(Action("use_potion", -1, index, slot))
        else:
            actions.append(Action("use_potion", -1, -1, slot))
    return actions


#: **卡牌级 ``ShouldPlay`` 覆盖方**：``cid → 是否允许这次出牌``。
#:
#: 真机 ``CardModel.ShouldPlay(card, autoPlayType)`` 是**虚方法**，卡牌可以覆写它，
#: 于是"手里有这张牌时别的牌打不出"这类限制有了归处。引擎原先只实现了**能力**侧
#: 的等价物（``affliction_blocks_play``：束缚 / 烟雾），**卡牌侧整批没实现** ——
#: 后果是这两张**诅咒**在模拟器里变成纯收益（没有任何代价）：
#:
#: * ``Normality``（Normality.cs:40-52）：本回合打出 **≥3** 张后否决**所有**出牌
#:   （它不区分自动打出）。
#: * ``Enthralled``（Enthralled.cs:21-41）：手里的**其它**牌一律打不出；
#:   自己可以打出，**自动打出**不受限制（``autoPlayType != None`` 时放行）。
#:
#: ⚠️ 判据里的"在手里"（``base.Pile.Type == Hand``）由调用方保证：
#: 只有**手牌**里的覆写者才否决，所以 :func:`hand_should_play_allows` 遍历的是
#: ``state.hand`` —— 这两张牌一旦进了弃牌堆/消耗堆就不再产生影响。
CARD_SHOULD_PLAY: dict[str, "Callable[[CombatState, CardInstance, bool], bool]"] = {}


def _normality_allows(state: CombatState, card: CardInstance, auto: bool) -> bool:
    """``Normality.ShouldPlay``：本回合已打出 ``CardsPlayedThisTurn >= 3`` → 否决。

    计数出处：``Normality.CardsPlayedThisTurn`` = ``History.CardPlaysStarted``
    里"本回合、本人"的条目数（Normality.cs:33）—— 与引擎的
    ``state.cards_played_started_this_turn`` 是同一个量（见 ``_note_card_play_started``）。
    """
    return state.cards_played_started_this_turn < 3


def _enthralled_allows(state: CombatState, card: CardInstance, auto: bool) -> bool:
    """``Enthralled.ShouldPlay``：自己放行；自动打出放行；其余一律否决。"""
    if card.cid == "enthralled":
        return True
    return bool(auto)


CARD_SHOULD_PLAY.update({"normality": _normality_allows,
                         "enthralled": _enthralled_allows})


def hand_should_play_allows(state: CombatState, card: CardInstance, *,
                            auto: bool) -> bool:
    """手里所有覆写了 ``ShouldPlay`` 的牌是否**都**允许这次出牌。

    ``auto=True`` 表示这次是**自动打出**（``AutoPlayType != None``）——
    只有 ``Normality`` 那种不看 ``autoPlayType`` 的覆写者才会否决自动打出。
    """
    for held in state.hand:
        rule = CARD_SHOULD_PLAY.get(held.cid)
        if rule is not None and not rule(state, card, auto):
            return False
    return True


def should_play_allows(state: CombatState, card: CardInstance, *,
                       auto: bool) -> bool:
    """``Hook.ShouldPlay``：**全部** listener 的否决（卡牌级 + 能力级）。

    真机只有一个入口 —— ``Hook.ShouldPlay`` 遍历所有模型，**任一**返回 false 就拒绝，
    所以引擎也只能有一个组合点：手动出牌（:func:`card_playable`）与三条自动打出路径
    （``_autoplay_sly`` × 1、:func:`autoplay_card` × 1）都用它。
    各写一遍"卡牌级 + 能力级"的话，迟早有一条路径只查了一半。

    ⚠️ ``auto`` 要传下去：``Normality`` 与 ``SlothPower`` 都不看它，
    但 ``Enthralled`` 只看它（自动打出免检）。
    """
    from . import powers as power_rules
    return (hand_should_play_allows(state, card, auto=auto)
            and power_rules.should_play_allows(state, card, auto))


def card_target(state: CombatState, card: CardInstance) -> str:
    """源码 ``Shiv.TargetType/HasFanOfKnives``：刀扇把拥有者的小刀变为群攻。

    目标同时影响合法动作、自动出牌随机取样与伤害落点；只改其中一处会产生
    “界面要选人却实际群攻”或多消费目标随机数。不能改共享的 CardDef。
    """
    if card.cid == "shiv" and state.player.power("fan_of_knives") > 0:
        return "all_enemies"
    # ⭐ ``SovereignBlade.TargetType`` 的覆写（``SovereignBlade.cs``）：
    # 拥有者身上有 ``SeekingEdgePower`` 时，**单体**改成**全体**。
    # 与刀扇同理：目标只在这一处定义，合法动作 / 自动出牌取样 / 伤害落点一起跟着变。
    if card.cid == "sovereign_blade" and state.player.power("seeking_edge") > 0:
        return "all_enemies"
    return card.definition().target


def card_playable(state: CombatState, card: CardInstance) -> bool:
    """这张牌**此刻**能不能打出 —— **唯一**的可打性判据。

    出处：``CardModel.CanPlay(out reason, out preventer)``（CardModel.cs:1725）。
    真机把四件事收在**一个**函数里，任何显示可打性的地方（UI、AI、合法动作）
    都读它：

    ============================ ==================================================
    真机                         本引擎
    ============================ ==================================================
    ``Keywords.Contains(Unplayable)`` / ``IsPlayable``  ``card.playable()``
    ``HasEnoughResourcesFor(this)``（能量 **与** 星）   ``play_cost`` / ``star_cost``
    ``Hook.ShouldPlay(...)``（苦痛、腐败等封锁）        ``affliction_blocks_play``
    ``CanPlayTargeting(target)``（要有合法目标）        有活着的可指定敌人
    ============================ ==================================================

    ⚠️ **为什么必须收成一个函数**：观察层原先自己写了一份
    ``card.playable() and card.cost() <= state.energy``，于是
    "腐败生效 + 能量 0 + 手里有防御"时出现**自相矛盾的观测** ——
    合法动作里有"打出这张防御"，而观测写着 ``cost=1, playable=False``
    （外部复核 R4 的实测）。模型看到的是两个互相否认的信号，而 mask 补救不了
    矛盾的输入：它会学到"playable 这一列不可信"。
    """
    from . import keywords as keyword_rules
    # ⚠️ ``Unplayable`` 关键字**无例外**：`normality` 也覆写了 ``ShouldPlay``
    # （它允许别的牌、只锁"打出 ≥3 张之后"），但它自己是不可打出的诅咒。
    # 这一条必须在下面的"空卡例外"**之前**判，顺序反了 `normality` 就能被打出。
    if keyword_rules.is_unplayable(card.definition()):
        return False
    if not card.playable() and card.cid not in CARD_SHOULD_PLAY:
        # ⚠️ **窄例外**：覆写了 ``ShouldPlay`` 的牌不适用"空卡不可打出"这条
        # **模拟器约定**。``CardModel.CanPlay``（CardModel.cs:1725）**不要求"有效果"**；
        # ``Enthralled.ShouldPlay``（Enthralled.cs:32-35）专门写了
        # ``if (card is Enthralled) return true;`` —— 而它没有任何 ``OnPlay``。
        # 没有这个例外，``Enthralled`` 在引擎里**永远打不出**，
        # 比真机更狠（真机花 2 费就能把它请出手，代价换回出牌自由）。
        return False
    if play_cost(state, card) > state.energy:
        return False
    # 星费（``UnplayableReason.StarCostTooHigh``）
    if star_cost(state, card) > state.stars:
        return False
    from . import powers as affliction_rules
    if affliction_rules.affliction_blocks_play(state, card):
        return False
    # ⭐ **``ShouldPlay`` 一族**（``Normality`` / ``Enthralled`` 的卡牌级、
    # ``SlothPower`` 的能力级）：与自动打出路径共用同一个组合点
    # （:func:`should_play_allows`）。只封锁动作掩码是不够的 ——
    # ``legal_actions`` / 直接 ``step`` / 观察层都读本函数，漏一处就出现
    # "界面说不能打、直接执行却成功了"这种自相矛盾。
    if not should_play_allows(state, card, auto=False):
        return False
    # 目标：``TargetType.AnyEnemy`` 至少要有一个可指定的活敌人
    # （真机 ``CanPlayTargeting``）。没有目标时这张牌打不出，
    # 观测与合法动作必须**都**说打不出。
    if card_target(state, card) == "enemy":
        return any(enemy.alive() and (not hasattr(enemy, "targetable")
                                     or enemy.targetable())
                   for enemy in state.enemies)
    return True


def legal_actions(state: CombatState) -> list[Action]:
    """合法动作列表。**mask 是硬性要求**（docs/03 §3.5）：无效动作占比高，不 mask 会浪费样本。"""
    if state.finished():
        return []
    # 正在等选牌：此时**只能**选牌 —— 其它动作全部非法。
    # ⚠️ 候选牌每次从牌堆现算，不用下标快照：选一张之后下标会移位，
    # 快照会让"引擎认为的候选"与"观察层显示的候选"错位，选错牌。
    if state.pending is not None:
        return [Action("select_card", index)
                for index in range(len(state.pending.candidates(state)))]
    if state.phase != "player":
        return []

    actions: list[Action] = []
    for index, card in enumerate(state.hand):
        # ⭐ 可打性走**同一个**查询（`card_playable`），与观察层共用 ——
        # 两处各写一份必然漂移，而漂移的表现是"观测说打不出、合法动作说能打"。
        if not card_playable(state, card):
            continue
        target_kind = card_target(state, card)
        if target_kind == "enemy":
            for ei in range(len(state.enemies)):
                enemy = state.enemies[ei]
                # ShouldAllowHitting → false：复活中的单位不能成为目标
                if enemy.alive() and (not hasattr(enemy, "targetable") or enemy.targetable()):
                    actions.append(Action("play_card", index, ei))
        else:
            actions.append(Action("play_card", index, -1))
    # ⭐ 药水与出牌并列：它们都是"玩家回合内的主动动作"（docs/13 §8.2）。
    actions.extend(potion_actions(state))
    actions.append(Action("end_turn"))
    return actions


def play_cost(state: CombatState, card: CardInstance) -> int:
    """打出这张牌的实际费用：卡牌费用 + **苦痛加价** + 减费 + **Late 段归零**。

    两个阶段逐条对应真机的两个钩子（顺序不能换）：

    * **非 Late**（``Hook.TryModifyEnergyCostInCombat``）：``TangledPower`` 给贴了
      ``Entangled`` 的攻击牌 ``+Amount``；``CuriousPower`` 给能力牌 ``-Amount``（下限 0）；
      ``BorrowedTimePower`` 给拥有者的**所有牌** ``+Amount``（`docs/12` §2.30）。
    * **Late**（``Hook.TryModifyEnergyCostInCombatLate``）：``CorruptionPower``（技能）、
      ``FreeAttackPower`` / ``FreeSkillPower`` / ``FreePowerPower``（按牌型）、
      ``VeilpiercerPower``（虚无牌）、``VoidFormPower``（本回合前 Amount 张）→ 费用 **0**。

    真机的阶段名就是 ``…Late``：腐败跑在苦痛**之后**，所以它一旦成立，
    苦痛加的那一笔也一并归零（先加价再清零 = 免费）。顺序反了会得到
    "腐败下打纠缠的技能牌还要付 1 费"这种错误。

    ⚠️ **X 费卡两个阶段都不参与**（`docs/12` §2.30 修正）。
    真机 ``CardEnergyCost.GetWithModifiers``（``CardEnergyCost.cs``）在两处提前返回：

    ::

        if (_base < 0) return num;      // 负费的非 X 卡；X 卡构造时 _base 归零
        if (CostsX) return num;         // X 费卡直接跳出
        if (modifiers.HasFlag(CostModifiers.Global) && _card.CombatState != null)
            num = (int)Hook.ModifyEnergyCostInCombat(_card.CombatState, _card, num);

    所以 X 费卡**既**不吃腐败归零、**也**不吃战斗内加价/减费。旧实现写成
    "X 费卡…**参与**腐败归零——源码里根本没判 ``CostsX``"，那是错的：
    真正的闸门是上面那两行提前返回，只是没有以 ``CostsX`` 为名的判断出现在钩子里。
    这个错在实现 `BorrowedTimePower`（所有牌 +Amount）时会**立刻变成可见故障**：
    X 费卡的引擎费用是"当前全部能量"，再加 1 就等于**永远打不出来**。
    """
    if card.definition().is_x_cost:
        return card.effective_cost(state.energy)
    from . import powers as power_rules
    base = (card.effective_cost(state.energy)
            + power_rules.affliction_cost_bonus(state, card))
    # 非 Late 段：`CuriousPower` 减费 / `BorrowedTimePower` 加价（同一阶段）。
    base = power_rules.energy_cost_reduction(state, card, base)
    # Late 段（`TryModifyEnergyCostInCombatLate`）：免费族 + 腐败把费用清零。
    # ⚠️ 顺序不能反：Late 段在后面，所以"先加价再清零 = 免费"。
    if power_rules.energy_cost_zeroed(state, card):
        return 0
    return max(0, base)


def action_mask(state: CombatState) -> dict[tuple[int, int], bool]:
    """动作 -> 是否合法。给神经网络的 mask 用（骨架里用字典表示）。"""
    return {(a.hand_index, a.target): True for a in legal_actions(state)}


# ==========================================================================
# 数值结算
# ==========================================================================
#: 真机对伤害/格挡/生命的统一钳制上限（``Creature.cs`` 里到处是 ``999999999m``）
VALUE_CAP = 999_999_999


def gain_stars(state: CombatState, amount: int, events: list[str]) -> None:
    """``PlayerCmd.GainStars``：**唯一**的加星入口，并分发 ``Hook.AfterStarsGained``。

    源码::

        public static async Task GainStars(decimal amount, Player player) {
            player.PlayerCombatState.GainStars(amount);
            await Hook.AfterStarsGained(player.Creature.CombatState, (int)amount, player);
        }

    ⚠️ 直接写 ``state.stars += n`` 会**漏掉那个钩子** —— `BlackHolePower` 的
    一半行为就是"获得星时对全体造成伤害"，漏了它这个能力等于只做了一半。
    """
    if amount <= 0:
        return
    state.stars += amount
    events.append(f"星 +{amount}")
    from . import powers as power_rules
    power_rules.on_stars_gained(state, amount, events)


def star_cost(state: CombatState, card: CardInstance) -> int:
    """这张牌的**星费**（``CardModel.GetStarCostWithModifiers()``）。

    真机两条分支（``CardModel.cs``）::

        if (HasStarCostX) return Owner.PlayerCombatState?.Stars ?? 0;   // X 星 = 花光
        return (int)Hook.ModifyStarCost(CombatState, this, CurrentStarCost);

    修正是 ``Hook.ModifyStarCost``：`VoidFormPower` 把符合条件的牌的费用
    （**能量与星一起**）归零、`BrilliantScarf`（遗物）另算 —— 能力侧那一半走
    :func:`powers.star_cost_zeroed`。

    ⚠️ 以前引擎**根本没有**这个量：23 张带星费的卡（`comet` 要 5 星）可以白嫖，
    而门禁只看"效果能不能落地"，谁也发现不了。
    """
    definition = card.definition()
    if getattr(definition, "is_x_star_cost", False):
        return max(0, state.stars)
    base = int(getattr(definition, "star_cost", 0) or 0)
    if base <= 0:
        return 0
    from . import powers as power_rules
    if power_rules.star_cost_zeroed(state, card):
        return 0
    return base


def compute_damage(base: int, attacker: Combatant | None, defender: Combatant,
                   card=None, state: CombatState | None = None,
                   powered: bool = True) -> int:
    """伤害结算——**严格按 ``Hook.ModifyDamageInternal`` 的管线**。

    真机的顺序是：

    1. **加法**：``num += ModifyDamageAdditive(...)``（力量走这条）
    2. **乘法**：``num *= ModifyDamageMultiplicative(...)``（易伤 ×1.5、虚弱 ×0.75，**累乘**）
    3. 钳制：``num = min(num, ModifyDamageCap(...))``
    4. 落地：``Creature.cs`` 里 ``(int)Math.Clamp(amount, 0m, 999999999m)``

    全程 ``decimal``，**中间不取整，只在最后截断一次**。

    ⚠️ 早期版本写成"每步 ``int()`` 截断一次"，那是错的两处：
    ``base 9 + 虚弱 + 易伤`` 真机是 ``floor(9×1.5×0.75) = floor(10.125) = 10``，
    逐次截断会得到 ``int(int(9×0.75)×1.5) = 9``——**差 1 点**，
    而且只在特定数值组合下才暴露。

    ⚠️ 审计 F06：``tainted`` 也曾被放在乘法之后，得到 ``10×1.5+3 = 18``
    而不是源码的 ``int((10+3)×1.5) = 19``。**阶段顺序是这条管线唯一的正确性来源**。
    """
    from . import powers as power_rules

    value = Decimal(base)
    if attacker is not None:
        # 加法阶段：**所有**"每层 +1 伤害"的能力（力量、活力…）。
        # 写成 `attacker.power("strength")` 只覆盖一个，加了新能力就会静默漏算 ——
        # 查表让"哪些能力影响伤害"只有一处定义（`powers.DAMAGE_ADDITIVE`）。
        from .powers import DAMAGE_ADDITIVE
        for pid in DAMAGE_ADDITIVE:
            value += attacker.power(pid)
        # ⭐ **带条件**的加法修正（`AccuracyPower` 只认飞刀、`LeadershipPower` 只认同阵营
        # 的别人）。真机这一步遍历**所有模型**，所以这里也按全部单位查一遍。
        for _pid, extra in power_rules.damage_additive_modifiers(
                attacker, defender, state, card, powered):
            value += extra
    # ⭐ `TaintedPower.ModifyDamageAdditive`：沾染后受到的**有效攻击**伤害 +Amount。
    # **必须留在加法阶段**（与力量同层、在乘法之前）—— 审计 F06 实测：
    # 放在乘法之后会得到 `10×1.5+3 = 18`，而源码是 `int((10+3)×1.5) = 19`。
    # 这个差别只在同时有易伤/虚弱时暴露，所以"看起来对"的旧实现一直没被发现。
    # 出处：`Hook.ModifyDamageInternal`（先跑完 `ModifyDamageAdditive` 再跑
    # `ModifyDamageMultiplicative`，`Hook.cs:2520`）+ `TaintedPower.cs:19`。
    value += power_rules.tainted_damage_bonus(defender)
    # ⭐ 乘法阶段**表驱动**：易伤 ×1.5、虚弱 ×0.75、减伤族（滑翔 / 高飞 / 缩小）
    # 与包围都在 `powers.DAMAGE_MULTIPLIERS` 里。
    # ⚠️ 不能既写死易伤/虚弱又跑表 —— 那会把它们**乘两次**（×2.25）。
    for _source, factor in power_rules.damage_multipliers(attacker, defender,
                                                          card=card, state=state):
        value *= Decimal(factor)
    if value < 0:
        return 0
    return int(min(value, Decimal(VALUE_CAP)))                  # 单次截断 + 钳制


def compute_block(base: int, owner: Combatant,
                  extra_multiplier: Decimal = Decimal(1),
                  card=None, state: "CombatState | None" = None) -> int:
    """格挡获得——``ModifyBlockAdditive`` 后走 ``ModifyBlockMultiplicative``。

    加法阶段：敏捷等"每层 +1 格挡"的能力（``DexterityPower``），以及**带条件**的
    ``FastenPower``（只认 ``Defend`` 标签的牌与怪物招式，见
    :func:`powers.block_additive_modifiers`）。
    乘法阶段：脆弱时 ×0.75（``FrailPower.cs``），以及
    ``UnmovablePower.ModifyBlockMultiplicative``（每回合前 N 次由卡牌获得的格挡 ×2）。

    ⚠️ ``extra_multiplier`` **必须传进来一起乘、一起截断**，不能在算完之后再乘：
    真机全程 ``decimal``、只在最后 ``Math.Max(…, 0m)`` 一次；先 ``int()`` 再乘会得到
    ``int(5×0.75)×2 = 6``，而真机是 ``int(5×0.75×2) = 7``。

    ``card`` 是这次格挡的**来源卡牌**（``ModifyBlockAdditive`` 的条件要用它的标签）。
    """
    from . import powers as power_rules
    from .powers import BLOCK_ADDITIVE
    value = Decimal(base)
    for pid in BLOCK_ADDITIVE:
        value += owner.power(pid)
    for _pid, extra in power_rules.block_additive_modifiers(owner, state, card):
        value += extra
    if owner.power("frail") > 0:
        value *= Decimal("0.75")
    value *= extra_multiplier
    return int(min(value, Decimal(VALUE_CAP)))


def gain_block(state: CombatState, owner: Combatant, amount: int,
               events: list[str], unpowered: bool = False,
               card: "CardInstance | None" = None, label: str = "") -> int:
    """``CreatureCmd.GainBlock`` 的**唯一入口**：获得格挡并通知 ``AfterBlockGained``。

    ⚠️ 为什么要集中：``AfterBlockGained``（``CreatureCmd.cs:699``）是"获得格挡之后"
    的唯一分发点，而引擎里加格挡的地方散在六处（卡牌效果 / 充能球 / 镀层 /
    下回合格挡 / 胆怯 / 壁垒 / 暴怒）。任何一处绕开它，
    ``JuggernautPower``（板甲）就会**静默少打**——而日志一切正常。

    ``unpowered`` 对应 ``ValueProp.Unpowered``：药水 / 能力给的格挡不吃敏捷、
    不被脆弱打折。``card`` 是来源卡牌（``ModifyBlockMultiplicative`` 那些
    "每回合前 N 次由卡牌获得的格挡"要靠它区分来源），非卡牌来源传 ``None``。
    """
    from . import powers as power_rules
    # ⭐ `ShadowmeldPower.ModifyBlockMultiplicative`：拥有者获得的格挡 ×2^Amount，
    # 而且源码**不看 ValueProp**（只判 "被打的是不是自己"）—— 所以
    # Unpowered 的格挡（药水 / 充能球）**同样**要乘。只放进 `compute_block`
    # （只有 powered 才走）会漏掉那一条，引擎就比真机**弱**。
    shadowmeld = power_rules.shadowmeld_multiplier(owner)
    if unpowered:
        gained = int(Decimal(amount) * shadowmeld)
    else:
        multiplier = power_rules.block_multiplier(state, owner, card) * shadowmeld
        gained = compute_block(amount, owner, multiplier, card=card, state=state)
    return _apply_block_gain(state, owner, gained, events, card=card, label=label)


def _apply_block_gain(state: CombatState, owner: Combatant, gained: int,
                      events: list[str], card: "CardInstance | None" = None,
                      label: str = "") -> int:
    """格挡量已经算好（全部修正完毕）之后落地，并分发 ``AfterBlockGained``。"""
    if gained > 0:
        owner.block = min(owner.block + gained, VALUE_CAP)
    # ⭐ 记下"最近一次**实际**获得的格挡"：真机 ``CreatureCmd.GainBlock`` 的返回值
    # 被 ``ToricToughness.OnPlay`` 拿去当 ``SetBlock(blockAmount)`` 的参数 ——
    # 是**过了敏捷/脆弱/倍率之后**的数，不是卡面的那个数。
    state.last_block_gain = gained
    suffix = f"（{label}）" if label else ""
    events.append(f"{owner.name} 获得 {gained} 点格挡{suffix}")
    from . import powers as power_rules
    power_rules.on_block_gained(owner, state, events, gained=gained, card=card)
    # `CombatManager.History.BlockGained`：把"这张牌本回合已经拿过格挡"记下来 ——
    # `UnmovablePower.ModifyBlockMultiplicative` 数的是这份记录
    # （``e.CardPlay != cardPlay``：同一张牌自己的多次拿格挡只算一次）。
    power_rules.block_gained(state, owner, card)
    return gained


def mark_dead(state: CombatState, combatant: Combatant,
              events: list[str], cause: str = "") -> None:
    """**唯一**的死亡入口 —— 对应真机的 ``CreatureCmd.Kill``。

    ⚠️ 这是架构上必须集中的一处。真机所有死亡都走 ``CreatureCmd``，
    ``Hook.AfterDeath`` 在那里调一次；而引擎早期是在各处**硬编码** ``hp = 0``，
    于是有一条路径漏掉死亡钩子就没人发现：

    * 末日处决（``dies_to_doom``）
    * 招式自爆（``kills_self``）
    * 中毒 / 掉血（``deal_raw_damage``）

    漏掉的后果是**静默**的：``crab_rage``（队友阵亡才暴怒）、
    ``stock`` / ``surprise`` / ``infested``（自己死了才召唤）、
    ``minion``（主敌人倒下时小怪跟着死）全都不会触发 ——
    怪看起来还在按部就班地打，只是"该发生的事没发生"。

    ``cause`` 只用于日志。重复调用是安全的（已经死透就不再触发一次）。

    ⚠️ 判"是否已经处理过"**不能**看 ``hp > 0``：伤害路径是**先扣血再调用它**，
    进来时 hp 已经是 0 了 —— 用血量判断会让**所有**由伤害造成的死亡都不触发钩子
    （实测症状：中毒打死的怪不补库存、不连锁倒下，而末日处决那条路径却正常，
    这种"一半对一半错"最难猜）。必须用显式标志。
    """
    if combatant.death_processed:
        return                          # 已经处理过，别重复触发钩子
    # ⭐ ``Hook.BeforeDeath``：**死之前**（此时血量可能已经是 0，但"死亡还没处理"）。
    # `HeistPower` / `SwipePower`（盗贼类）在这里把偷来的东西还回去
    # —— 它们挂在"拥有者自己死了"上（源码 ``if (base.Owner != target) return;``）。
    from . import powers as power_rules
    power_rules.on_before_death(state, combatant, events)
    combatant.death_processed = True
    if combatant.hp > 0:
        combatant.hp = 0
    suffix = f"（{cause}）" if cause else ""
    if isinstance(combatant, EnemyState):
        events.append(f"{combatant.name} 死亡{suffix}")
    power_rules.on_any_death(state, combatant, events)


def deal_raw_damage(state: CombatState, target: Combatant, amount: int,
                    events: list[str], label: str = "",
                    attacker: Combatant | None = None,
                    card: "CardInstance | None" = None) -> int:
    """**不经格挡**的掉血（中毒 / 末日 / 药水直伤 / 自伤）。

    真机走 ``CreatureCmd.Damage(..., ValueProp.Unblockable)``，与普通伤害
    **共用同一条钩子链**：``AfterDamageReceived`` 照样触发（``SlipperyPower``
    就靠它递减），只是 ``IsPoweredAttack()`` 为假、所以 ``ThornsPower`` /
    ``PainfulStabsPower`` 这类"只认有效攻击"的能力不触发。

    ⚠️ ``attacker`` 为 ``None`` 时**就传 None**（真机 ``dealer`` 本来就可空，
    ``CreatureCmd.cs:180``）。以前这里传的是"目标自己"，于是带荆棘的单位中毒
    就会**反伤自己** —— ``ThornsPower`` 的 ``dealer != null`` 被一个假 dealer 骗过。

    ``card`` 对应真机的 ``cardSource``（"这次伤害是哪张牌造成的"）——
    ``RupturePower`` 靠它区分"卡牌自伤（要等牌结算完再发力量）"与"其它掉血（立刻发）"。
    """
    if amount <= 0 or not target.alive():
        return 0
    if hasattr(target, "targetable") and not target.targetable():
        return 0
    dealt = min(amount, target.hp)
    from . import powers as power_rules
    # ⭐ 走 ``apply_hp_loss_cap``（而不是只查 ``hp_loss_cap``）：`BufferPower`
    # 每抵消一次掉血就要减 1 层，递减点必须与"真的压住了"绑在同一处。
    dealt = power_rules.apply_hp_loss_cap(target, dealt, events)
    target.hp -= dealt
    suffix = f"（{label}）" if label else ""
    events.append(f"{target.name} 受到 {dealt} 点伤害{suffix}")
    # `AfterDamageReceived`：掉血也算"受到伤害"，滑溜靠它递减。
    # `from_card=False` + `powered=False`：中毒不是有效攻击。
    # ⚠️ `powered=False` 必须**真的传下去**：本函数的 docstring 一直写着
    # "``IsPoweredAttack()`` 为假"，但早先漏了这个实参，默认值把它变成了 True ——
    # 于是"只认有效攻击"的能力（`ThornsPower` / `PainfulStabsPower` / `FlameBarrierPower`）
    # 会被中毒/自伤触发，`TheGambitPower`（掉血即处决拥有者）更是会被中毒直接打死。
    # `docs/12` §2.30 记了这条（在实现 `the_gambit` 时暴露）。
    power_rules.on_attacked(target, attacker, state, events,
                            unblocked=dealt, from_card=False, powered=False,
                            card=card)
    if isinstance(target, EnemyState) and not target.alive():
        mark_dead(state, target, events, cause=label)
    return dealt


def deal_damage(state: CombatState, attacker: Combatant | None,
                defender: Combatant, base: int, events: list[str],
                unpowered: bool = False, unblockable: bool = False,
                card: "CardInstance | None" = None) -> int:
    """返回实际掉血量。

    ``unpowered``（``ValueProp.Unpowered``）跳过力量/虚弱/易伤修正但**仍可格挡**；
    ``unblockable``（``ValueProp.Unblockable``）连格挡都不吃。
    药水、充能球、中毒/荆棘都走这两个开关 —— 忽略它们会让数值静默偏掉。

    ``card`` 就是真机的 ``cardSource``：这次伤害是**哪张牌**造成的。
    ``RupturePower`` 靠它决定"攒到牌结算完再发力量"还是"立刻发"。
    """
    # `ShouldAllowHitting` → false：**复活中**的单位打不到（幻影 / 再附着 / 试验体）。
    # 少这一条，"死了又活"的怪会在倒地期间被继续打，等于没有复活机制。
    if hasattr(defender, "targetable") and not defender.targetable():
        return 0
    from . import powers as power_rules
    if unpowered:
        dmg = max(0, base)
    else:
        # ⭐ `card` / `state` 必须传下去：带条件的倍率（`DoubleDamagePower` 要
        # `cardSource != null`、`ColossusPower` 要看攻击者身上的易伤）判不了就只能
        # 无条件生效 —— 那是**静默变强**。
        dmg = compute_damage(base, attacker, defender, card=card, state=state,
                             powered=not unpowered)
    # ⭐ `Hook.ModifyDamageCap`（伤害管线的**第 3 步**：加法 → 乘法 → 钳制 → 落地）。
    # `HardToKillPower` 把"每次最多受 N 点"压在这里 —— 少了它，这条能力整条无效，
    # 那只怪会被一刀秒（引擎比真机**强**）。它不判 props，所以 Unpowered 的伤害同样受限。
    damage_cap = power_rules.damage_cap(defender)
    if damage_cap is not None:
        dmg = min(dmg, damage_cap)
    blocked = 0 if unblockable else min(defender.block, dmg)
    defender.block -= blocked
    hp_loss = dmg - blocked
    # `IntangiblePower.ModifyHpLostAfterOsty`：掉血压到最多 1 点。
    # ⚠️ 它管的是**掉血**（格挡之后那一段），不是原始伤害 —— 弄反了会让
    # 无形完全免疫伤害（连格挡都不消耗）。
    # ⭐ `BufferPower` 在**晚段**（`…AfterOstyLate`）压到 0，所以它赢过无形；
    # 递减也在这里发生（每抵消一次掉血减 1 层）。
    hp_loss = power_rules.apply_hp_loss_cap(defender, hp_loss, events)
    defender.hp = max(0, defender.hp - hp_loss)
    events.append(f"{defender.name} 受到 {hp_loss} 点伤害（格挡 {blocked}，总 {dmg}）")
    # 被打之后触发防御方能力（`ThornsPower` 反伤 / `FlameBarrierPower.AfterDamageReceived`）。
    # ⚠️ 时机在**扣血之后**，且反伤走"不可格挡"通道 —— 用 deal_damage
    # 会套上攻击者的力量与防御方的易伤，数值会凭空变化。
    #
    # ⚠️ 真机 `Hook.AfterDamageReceived`（`CreatureCmd.cs:416`）对**每一条**伤害结果
    # 都分发，**不看是谁打的**：自伤（`dealer` 是目标自己、或者根本没有 dealer）
    # 同样会触发。旧实现只在"有第三方攻击者"时才分发，于是
    # `RupturePower` / `InfernoPower`（"自己回合掉血时…"）对自伤牌**完全不响应** ——
    # 而自伤正是这两张牌的设计核心。这里改成无条件分发，`attacker=None` 表示无来源。
    power_rules.on_attacked(defender, attacker, state, events,
                            unblocked=hp_loss,
                            from_card=isinstance(attacker, Combatant)
                            and attacker is state.player,
                            powered=not unpowered, card=card,
                            # `DamageResult.BlockedDamage`：被格挡掉的量
                            # （`ReflectPower` 反弹的就是它）。
                            blocked=blocked)
    if attacker is not None and attacker is not defender:
        # `AfterAttack` / `AfterDamageGiven`：**攻击方**的触发（`PainfulStabsPower`
        # 往玩家弃牌堆塞伤口）。放在这里而不是出牌处，是因为怪物出招也要触发。
        # 自伤没有"攻击方"，所以这一侧不分发。
        power_rules.on_damage_given(attacker, state, events, hp_loss,
                                    powered=not unpowered,
                                    was_fully_blocked=(hp_loss == 0 and blocked > 0),
                                    target=defender,
                                    # `DamageResult.TotalDamage`：**含被格挡的部分**
                                    # （`ReaperFormPower` 按它算末日层数）。
                                    total=dmg)
    # ⭐ `Hook.AfterDamageGiven` 是**全体模型**分发（`dealer` 可以为 null），
    # 而 `on_damage_given` 那一条按 `scope=owner` 只通知经手者自己。所以
    # "**同伴**打的伤害我也要响应"这类能力（`UnderworldPower`）走这条独立调用点。
    power_rules.on_ally_damage_given(state, attacker, defender, dmg,
                                     powered=not unpowered, events=events)
    if isinstance(defender, EnemyState) and not defender.alive():
        mark_dead(state, defender, events)
    return hp_loss


def _opponents_of(state: CombatState, source: Combatant) -> list[Combatant]:
    """效果以**施法者视角**解释目标。

    ``target="enemy"`` 对玩家而言是敌人，对敌人而言是玩家。**这条视角规则必须写对**：
    早期版本把目标永远当成"敌人列表里的某一个"，于是怪物攻击的是它自己——
    玩家永不掉血，随机策略也能 100% 通关，而全部单元测试都发现不了。
    是 ``tests/test_run.py`` 的"区分度测试"（随机策略不该和规则 bot 打平）抓出来的。
    """
    if isinstance(source, EnemyState):
        return [state.player]
    return list(state.living_enemies())


def _primary_target(state: CombatState, source: Combatant,
                    target_index: int) -> Combatant | None:
    if isinstance(source, EnemyState):
        return state.player
    if 0 <= target_index < len(state.enemies):
        return state.enemies[target_index]
    return None


def _orb_raw_damage(state: CombatState, target: Combatant, amount: int,
                    events: list[str], label: str) -> int:
    """充能球的伤害：``ValueProp.Unpowered`` —— **不受力量/虚弱/易伤影响，但可格挡**。

    ⚠️ 不能走 ``deal_damage``：那条路会套上力量与易伤（还会因虚弱打折）。
    Defect 的伤害大半来自球，误差会直接反映成胜率差。
    也不能走 ``deal_raw_damage``：那是"不可格挡"的通道（中毒/荆棘），
    而球的伤害**是要被格挡吃掉的**。
    """
    if amount <= 0 or not target.alive():
        return 0
    # ShouldAllowHitting → false：复活中的单位**打不到**。
    if hasattr(target, "targetable") and not target.targetable():
        return 0
    blocked = min(target.block, amount)
    target.block -= blocked
    hp_loss = amount - blocked
    from . import powers as power_rules
    hp_loss = power_rules.apply_hp_loss_cap(target, hp_loss, events)
    target.hp = max(0, target.hp - hp_loss)
    events.append(f"{target.name} 受到 {hp_loss} 点伤害（{label}，格挡 {blocked}）")
    if isinstance(target, EnemyState) and not target.alive():
        events.append(f"{target.name} 死亡")
    return hp_loss


def _orb_gain_block(state: CombatState, amount: int, events: list[str],
                    label: str) -> None:
    """球的格挡：同样 ``Unpowered`` —— **不吃敏捷，也不被脆弱打折**。

    ⚠️ 走 :func:`_apply_block_gain`（而不是直接 ``block +=``）：真机所有格挡
    都经过 ``CreatureCmd.GainBlock``，冰球的格挡同样会触发 ``AfterBlockGained``
    （板甲在充能球流里是真实存在的联动）。
    """
    if amount <= 0:
        return
    _apply_block_gain(state, state.player, amount, events, label=label)


def _orb_gain_energy(state: CombatState, amount: int, events: list[str],
                     label: str) -> None:
    state.energy += amount
    events.append(f"能量 +{amount}（{label}）")


def _card_matches(card: CardInstance, conditions) -> bool:
    """按 ``move_all_matching`` 的过滤条件判一张牌是否符合。

    条件词汇表与抽取器一致（``FILTER_PATTERNS``）；**遇到不认识的条件返回 False**
    并让效果整体不生效 —— 宁可少动一张牌，也不能用猜出来的条件乱移牌堆。
    """
    definition = card.definition()
    for key, value in conditions:
        if key == "cost":
            if definition.cost != value:
                return False
        elif key == "not_cost":
            if definition.cost == value:
                return False
        elif key == "rarity":
            if definition.rarity.lower() != str(value).lower():
                return False
        elif key == "not_rarity":
            if definition.rarity.lower() == str(value).lower():
                return False
        elif key == "type":
            if definition.card_type.lower() != str(value).lower():
                return False
        elif key == "not_type":
            if definition.card_type.lower() == str(value).lower():
                return False
        elif key == "has_keyword":
            if value not in definition.keywords:
                return False
        elif key == "not_keyword":
            if value in definition.keywords:
                return False
        elif key == "upgradable":
            if card.upgraded:
                return False
        elif key == "not_x_cost":
            if definition.cost < 0:
                return False
        else:
            return False               # 不认识的过滤条件 → 整体不生效
    return True


def _move_matching_cards(state: CombatState, eff: Effect) -> int:
    """``CardPileCmd.Add(<集合>, PileType.Y)``：把符合条件的一批牌整体移动。"""
    source = {"hand": state.hand, "draw": state.draw_pile,
              "discard": state.discard, "exhaust": state.exhaust}.get(
                  eff.select_from)
    if source is None:
        return 0
    matching = [c for c in source if _card_matches(c, eff.card_filter)]
    if eff.take_random:
        # `TakeRandom(n, Rng.CombatCardSelection)`：走**独立随机流**取 n 张
        count = _resolve_take_random(state, eff.take_random, len(matching))
        if count < len(matching):
            indices = sorted(
                state.hidden.rng.next_index("combat_card_selection", len(matching))
                for _ in range(count))
            matching = [matching[i] for i in dict.fromkeys(indices)]
    for card in matching:
        source.remove(card)
        _add_generated_card(state, card, eff.pile, eff.position)
        state.added_cards -= 1          # 移动**已有**的牌，不算"新生成"
    return len(matching)


def _resolve_take_random(state: CombatState, expr: str, available: int) -> int:
    """``TakeRandom(count, …)`` 的 count。

    只认两种**能确认**的写法，其余在**加载期**就被拒绝了
    （见 ``content._source_reject_reason``）—— 运行时才"大概猜一个"会多拿牌，
    而多拿牌是明显的强度提升，属于静默变强。
    """
    text = expr.strip()
    if re.fullmatch(r"\d+", text):
        return min(int(text), available)
    if "MaxCardsInHand" in text:
        # `CardPile.MaxCardsInHand - 手牌数`：补满手牌
        return min(available, max(0, MAX_HAND_SIZE - len(state.hand)))
    return 0                     # 认不出 → 什么都不拿（加载期已拦截）


def add_to_hand(state: CombatState, card: CardInstance,
                events: list[str] | None = None) -> str:
    """把一张牌加进手牌，**手牌满则改放弃牌堆**，返回实际落点。

    出处：``CardPileCmd.Add``（``CardPileCmd.cs``）::

        bool isFullHandAdd = cardPile.Type == PileType.Hand
                             && cardPile.Cards.Count >= CardPile.MaxCardsInHand;
        if (isFullHandAdd) cardPile = CardPile.Get(PileType.Discard, card.Owner);

    真机的手牌上限是 ``CardPile.MaxCardsInHand = 10``，而且**效果送的牌**
    也走同一条 `Add` 路径 —— 所以"手牌满了还硬塞进手牌"会让手牌超过 10 张，
    而那个状态在真机上不存在（后果是观测编码越界、以及模型看到一个假局面）。
    """
    if len(state.hand) >= MAX_HAND_SIZE:
        state.discard.append(card)
        if events is not None:
            events.append(f"手牌已满（{MAX_HAND_SIZE}），{card.definition().name} 进弃牌堆")
        return "discard"
    state.hand.append(card)
    return "hand"


def _add_generated_card(state: CombatState, card: CardInstance, pile: str,
                        position: str = "") -> None:
    """把生成的卡放进**指定**牌堆（``CardPileCmd.AddGeneratedCardToCombat``）。

    ⚠️ 牌堆不能一律当弃牌堆：真机有 ``PileType.Hand`` / ``Draw`` / ``Discard``
    三种，且 ``CardPilePosition.Top`` 表示放到抽牌堆**顶**（下一张就抽到）。
    全部塞弃牌堆会让"加一张 Soul 进抽牌堆"这类卡完全失效 —— 而它看起来正常。

    ``added_cards`` 只统计**战斗中新生成**的卡，用于守恒校验
    （``all_piles`` 总数 = 初始牌组 + added_cards）。
    """
    if pile == "hand":
        add_to_hand(state, card)
    elif pile == "draw":
        if position == "top":
            state.draw_pile.append(card)          # 抽牌用 pop()，尾部就是顶
        elif position == "random":
            # ``CardPilePosition.Random``：真机用 ``Rng.Shuffle.NextInt(count + 1)``
            # 选插入位置（``CardPileCmd.cs``）——`SoulboundPower` 往抽牌堆随机塞牌就走这条。
            index = state.hidden.rng.next_index("shuffle", len(state.draw_pile) + 1)
            state.draw_pile.insert(index, card)
        else:
            state.draw_pile.insert(0, card)
    else:
        state.discard.append(card)
    state.added_cards += 1
    # ⭐ ``Hook.AfterCardGeneratedForCombat``：**唯一**的"战斗内生成了一张牌"出口，
    # `ArsenalPower`（生成牌 → +力量）/ `SmokestackPower`（生成**状态牌** → 群伤）
    # / `TrashToTreasurePower`（生成状态牌 → 引导充能球）都挂在这里。
    # 放在 `added_cards += 1` 之后：能力里再生成牌时计数是对的（`SoulboundPower`
    # 靠重入标志防自激，见源码 `IsAddingSoul`）。
    from . import powers as power_rules
    power_rules.on_card_generated_for_combat(state, [], card,
                                             creator=state.player)
    # ⭐ ``Hook.AfterCardEnteredCombat``（``CardPileCmd.cs:515``）：一张牌**刚进入战斗**。
    # 判据是 ``oldPile == null``（新造出来的牌）—— 生成就是引擎里唯一的"新牌"入口，
    # 所以两个钩子在同一处、`AfterCardGeneratedForCombat` 之后再分发。
    # 苦痛族（tangled / smoggy / ringing / hex / vital_spark / galvanic）靠它给
    # 战斗中途生成的牌补苦痛；`PhantomBladesPower` 靠它给飞刀贴保留。
    power_rules.on_card_entered_combat(state, [], card)


def _apply_effects(state: CombatState, effects: Iterable[Effect],
                   source: Combatant, target_index: int, events: list[str],
                   pending_played: CardInstance | None = None,
                   prefix_events: list[str] | None = None,
                   x_value: int = 0) -> bool:
    """执行效果；遇到 ``select_card`` 就**挂起**。

    返回 ``True`` 表示全部执行完，``False`` 表示正在等玩家选牌
    （此时 ``state.pending`` 里存着续跑所需的剩余效果）。

    ``x_value`` 是 X 费卡的 X —— 真机 ``ResolveEnergyXValue()`` 取的就是
    **打出时消耗的能量**（``CardModel.SpendEnergy`` 里把它存进
    ``EnergyCost.CapturedXValue``）。带 ``times_x`` 的效果执行 X 次，
    带 ``amount_x`` 的效果把量取成 X。
    """
    effects = tuple(effects)
    for index, eff in enumerate(effects):
        if eff.op == "choose_one":
            # ⭐ **S02 的第二种挂起**：候选是**凭空造的临时卡**（不在任何牌堆），
            # 选中只执行它自己的 `OnChosen`。真机出处
            # `KnowledgeDemon.CurseOfKnowledgeMove`（``KnowledgeDemon.cs:157-176``）::
            #
            #     List<CardModel> cards = set[counter].Select(c => CreateCard(...)).ToList();
            #     CardModel chosen = await CardSelectCmd.FromChooseACardScreen(ctx, cards, target.Player);
            #     if (chosen != null) await ((IChoosable)chosen).OnChosen();
            #
            # ⚠️ 与 `select_card` 的三点不同，混用会静默错：
            #  1. 候选**不在牌堆里** —— 选完既不从牌堆取走、也不落堆；
            #  2. 每个候选自带一组效果（同一个 cid 在不同轮次载荷不同：
            #     `Disintegration` 的 6/7/8），不能靠 cid 回查；
            #  3. 玩家可以什么都不选（`chosen == null`）—— 引擎在**候选为空**时
            #     等价处理（真机此时也不会提问）；有候选就必须选一张。
            offered = tuple(CardInstance(str(label)) for label, _ in eff.choices)
            if not offered:
                events.append("跳过二选一（没有候选）")
                continue
            state.pending = PendingSelection(
                purpose="choose", source_pile="offered", remaining=1,
                rest=effects[index + 1:],
                target_index=target_index, played_card=pending_played,
                prefix_events=list(prefix_events if prefix_events is not None
                                   else events),
                card_consumed=pending_played is not None,
                x_value=x_value,
                # ⭐ `source` 必须带过去：怪物出招里的量按**怪物视角**解释。
                actor=source,
                offered=offered,
                choice_effects=tuple(tuple(group) for _, group in eff.choices))
            events.append(f"等待二选一（{len(offered)} 个候选）")
            return False
        if eff.op == "select_card":
            pending = PendingSelection(
                purpose=eff.purpose, source_pile=eff.select_from,
                remaining=max(1, eff.amount), position=eff.position,
                rest=effects[index + 1:],
                target_index=target_index, played_card=pending_played,
                prefix_events=list(prefix_events if prefix_events is not None
                                   else events),
                card_consumed=pending_played is not None,
                x_value=x_value)
            # ⭐ **空候选必须当场跳过，不能挂起**。
            #
            # 真机的选择命令在没有合法候选时直接返回空集合，不会向玩家提问；
            # 而挂起一个"没有候选"的选择会让 `legal_actions()` 返回**空列表** ——
            # 环境既没有动作可发、也不会自动结束，直接在训练里炸成
            # "存在没有任何合法动作的样本"。
            #
            # 实测复现：**回合 1 的 `headbutt`**（"从弃牌堆放一张到抽牌堆顶"），
            # 此时弃牌堆本来就是空的。
            if not pending.candidates(state):
                events.append(f"跳过选择（{eff.purpose}：没有候选）")
                continue
            state.pending = pending
            events.append(f"等待选择 {eff.amount} 张牌（{eff.purpose}）")
            return False
        for _ in range(x_value if eff.times_x else 1):
            _apply_one(state, eff, source, target_index, events, x_value,
                       pending_played)
        if state.pending is not None:
            # ⭐ 这条效果**中途**开了玩家选择（典型：`draw` 抽到空堆 → 洗牌 →
            # `StratagemPower` 要从抽牌堆挑牌）。真机这里是同一个 `await`，
            # 效果链必须停住，**剩余效果等选完再跑**。
            # 不处理的话：候选还没选，后面的效果就已经结算完了 ——
            # 手牌张数、伤害顺序全错，而日志一切正常。
            state.pending.rest = (tuple(state.pending.rest)
                                  + tuple(effects[index + 1:]))
            return False
    return True


def _procure_random_potions(state: CombatState, amount: int,
                            events: list[str]) -> int:
    """``PotionCmd.TryToProcure(CreateRandomPotionOutOfCombat(…))``：随机获得药水。

    真机（``EntropicBrew.cs:24-31``）::

        while (player.HasOpenPotionSlots) {
            potion = PotionFactory.CreateRandomPotionOutOfCombat(player, Rng.CombatPotionGeneration);
            if (!(await PotionCmd.TryToProcure(potion, player)).success) break;
        }

    ``amount <= 0`` 对应那个 ``while``（**填满所有空槽**）；``amount = 1``
    是单次（``Alchemize`` 那类）。

    药水池按 ``PotionFactory.GetPotionOptions``：**角色池 ∪ 共享池**。
    ⚠️ 这里只实现 ``OutOfCombat`` 版（它不过滤 ``CanBeGeneratedInCombat``）；
    ``InCombat`` 版（``Alchemize``）**不在这里近似** —— 抽取器会照旧报缺口。

    ⚠️ 另外要过**准入门禁**：引擎算不对的药水不能发到玩家手里
    （与"生成的牌过准入门禁"同一口径，见 ``docs/13`` §7 R3）。
    """
    from .content import POTIONS

    character = str(getattr(state, "character", "") or "")
    #: 药水的 `pool` 字段值是 ``shared`` 或角色名（``ironclad`` …），
    #: 与卡池名（``IroncladCardPool``）**不是**同一套命名，别混。
    allowed = {"shared"} | ({character} if character else set())
    candidates = sorted(pid for pid, definition in POTIONS.items()
                        if definition.pool in allowed
                        and not definition.effects_incomplete)
    if not candidates:
        events.append("⚠️ 没有可用的随机药水候选，跳过")
        return 0
    limit = amount if amount > 0 else len(state.potions)
    obtained = 0
    for _ in range(max(0, limit)):
        if None not in state.potions:
            break              # `HasOpenPotionSlots` 为假 → 退出 while
        index = state.hidden.rng.next_index("combat_potion_generation",
                                            len(candidates))
        state.potions[state.potions.index(None)] = candidates[index]
        obtained += 1
    return obtained


def _apply_one(state: CombatState, eff: Effect, source: Combatant,
               target_index: int, events: list[str], x_value: int = 0,
               played_card: "CardInstance | None" = None) -> None:
    # X 依赖的量在打出时才知道：`amount_x` 的效果用 X 当量。
    if eff.amount_x:
        eff = replace(eff, amount=x_value)
    # ⭐ **运行期公式**（`docs/12` §2.33）：``CalculatedVar`` 的量在**结算这一刻**
    # 才算（格挡 / 牌堆张数 / 能力层数 / 本场出牌数）。抽取器只写 ``calc_*``
    # 字段、``amount`` 恒为 0 —— 直接用 ``eff.amount`` 会得到"打出去没伤害"。
    #
    # 公式里的 ``target`` 是**这张牌选中的目标**（不是效果的作用对象）：
    # `Mimic` 是"自己获得格挡、量等于被打者的格挡"，效果 target 是 self、
    # 公式读的却是被打者。所以这里统一取"选中的目标"。
    formula_target = _primary_target(state, source, target_index)

    def _amount() -> int:
        if not getattr(eff, "calc_kind", ""):
            return eff.amount
        from . import powers as calc_rules
        return calc_rules.calc_amount(state, source, formula_target, eff)

    if eff.op == "damage":
        damage = _amount()          # ⭐ 运行期公式在这里求值（否则就是卡面的 0）
        # ⚠️ **必须尊重 `eff.target`**：卡牌伤害一般是打敌人，但
        # `target="self"` 的效果（手牌触发器里的燃烧 / 腐朽 / 感染…）打的是**自己**。
        # 一律走 `_primary_target` 会把"在手里烧你"变成"在手里烧敌人" ——
        # 状态牌的代价直接反号，而且不报错。
        if eff.target == "self":
            deal_damage(state, None, source, damage, events,
                        unpowered=eff.unpowered, unblockable=eff.unblockable,
                        card=played_card)
        elif eff.target == "all_enemies":
            # `all_enemies` 是**显式**的全体目标（遗物给全体敌人上力量就写这个），
            # 与 `damage_all` 算子同义，不能退化成"只打第一个"。
            for target in _opponents_of(state, source):
                if target.alive():
                    deal_damage(state, source, target, damage, events,
                                unpowered=eff.unpowered, unblockable=eff.unblockable,
                                card=played_card)
        else:
            target = _primary_target(state, source, target_index)
            if target is not None and target.alive():
                deal_damage(state, source, target, damage, events,
                            unpowered=eff.unpowered, unblockable=eff.unblockable,
                            card=played_card)
    elif eff.op == "damage_all":
        damage = _amount()
        for target in _opponents_of(state, source):
            if target.alive():
                deal_damage(state, source, target, damage, events,
                            unpowered=eff.unpowered, unblockable=eff.unblockable,
                            card=played_card)
    elif eff.op == "gain_max_hp":
        # `CreatureCmd.GainMaxHp`：加 N 上限 **+ 回 N 血**（源码方法**最后一行**
        # 是 `await Heal(creature, num)`，不能只看前十行）。战斗内也会用到 ——
        # 果汁药水就是它，而且加的上限是**永久的**（打完这场也留着）。
        target = source if eff.target == "self" else _primary_target(
            state, source, target_index)
        if target is not None:
            target.max_hp += eff.amount
            healed = min(eff.amount, target.max_hp - target.hp)
            target.hp += healed
            events.append(f"{target.name} 最大生命 +{eff.amount}"
                          f"（现 {target.max_hp}），回复 {healed} 点生命")
    elif eff.op == "block":
        # 同样尊重 `Unpowered`：药水的格挡不吃敏捷、也不被脆弱打折。
        # ⚠️ 走 `gain_block`（`CreatureCmd.GainBlock` 的唯一入口）而不是直接
        # `block +=`：绕开它就等于绕开 `AfterBlockGained`（板甲）。
        gain_block(state, source, _amount(), events, unpowered=eff.unpowered,
                   card=played_card)
    elif eff.op == "apply_power":
        assert eff.power is not None
        from . import powers as power_rules
        # ⭐ 运行期公式也能给"上多少层"用（`Hang` 的 ``max(2, 目标身上的层数)``、
        # `Dominate` 的"目标当前的易伤层数"、`Synchronize` 的"不同球种数"）——
        # `docs/12` §2.34。少了这一步，这三张卡会按 0 层结算（静默无效）。
        power_amount = _amount()

        def apply_to(target: Combatant) -> None:
            if not target.alive():
                return
            # `ArtifactPower`：被抵消的负面能力**完全不上**，且神器减 1 层。
            # 少了这一步，带神器的 8 只怪会被轻易上满 debuff ——
            # 那是"看起来正常"的强度差。
            if power_rules.negates_debuff(target, eff.power, power_amount):
                target.add_power("artifact", -1)
                events.append(f"{target.name} 的神器抵消了 {eff.power}"
                              f"（剩 {target.power('artifact')}）")
                return
            # ``fresh`` = 这条能力**原本不存在**（真机 ``PowerCmd.Apply`` 的
            # "新实例"路径）。叠加走 ``PowerCmd.ModifyAmount``，那条路径**不**调
            # ``BeforeApplied``/``AfterApplied``，只发 ``AfterPowerAmountChanged``
            # （``PowerCmd.cs:119/136/159/249``）。临时属性族就靠这个区分
            # "首次施加"与"叠加"，两边都做会把力量加两遍。
            fresh = target.power(eff.power) == 0
            # ⭐ **施加者必须记下来**（真机 ``PowerModel.Applier``）。
            # 卡牌路径早先只把 ``source`` 透传给 ``on_power_applied`` 钩子，
            # **没有落进** ``powers_applier`` —— 于是"读施加者"的能力
            # （`soulbound` 的 `creator == Applier`、`strangle`/`oblivion` 的
            # `cardPlay.Card.Owner == Applier.Player`，以及
            # `constrict`/`shrink`/`hex` 的 "施加者死了才结束"）经由**卡牌**施加时
            # 全部静默失效（`docs/12` §2.31 记了这条）。怪物的招式走
            # `apply_power_to`，那条路一直是记的，所以问题只出现在卡牌侧。
            target.add_power(eff.power, power_amount, applier=source)
            # ⭐ 记下"这个能力是第几回合上的"。`RitualPower.AfterApplied` 正是
            # 靠这个标记跳过"刚获得的那一次回合结束"；没有它，上仪式的那一回合
            # 会白赚一层力量（觉醒邪教徒的第一个回合多 1 点）。
            if power_amount > 0:
                target.power_applied_turn[eff.power] = state.turn
                # `AfterApplied`：能力刚被施加时触发一次
                # （`TangledPower` 就是在这时候给所有攻击牌贴苦痛的）。
                # ``applied=eff.amount`` 是**这一次**施加的层数（真机 `BeforeApplied`
                # 的 `amount`）—— `TemporaryStrengthPower` 加的是这个量，不是总层数。
                power_rules.on_applied(target, state, events, eff.power,
                                       power_amount, fresh=fresh)
                # ⭐ `Hook.AfterPowerAmountChanged`：**通知双方所有模型**这次施加
                # （带施加者与这一次的量）。`ViciousPower`（凶恶）的机制就是
                # "**我**给别人上易伤 → 抽牌"，所以它必须知道施加者是谁。
                power_rules.on_power_applied(state, events, eff.power,
                                             source, power_amount)

        if eff.target == "self":
            apply_to(source)
        elif eff.target == "all_enemies":
            for target in _opponents_of(state, source):
                apply_to(target)
        else:
            target = _primary_target(state, source, target_index)
            if target is not None:
                apply_to(target)
        events.append(f"{source.name} 施加 {eff.power} {eff.amount:+d}（{eff.target}）")
    elif eff.op == "draw":
        # `CompileDriver` 的"抽 = **不同球种数**"也是运行期公式（`docs/12` §2.34）。
        draw_cards(state, _amount(), events)
    elif eff.op == "lose_hp":
        # ⚠️ 真机的"自伤"是 ``CreatureCmd.Damage(..., Unblockable | Unpowered | Move,
        # dealer, cardSource, cardPlay)``（见 ``Hemokinesis.cs`` / ``Bloodletting.cs``）——
        # 它**照样走整条伤害管线**：``AfterDamageReceived`` 会分发、掉血上限
        # （无形）会生效。旧实现只是 ``hp -= amount``，于是
        # `RupturePower` / `InfernoPower`（"自己回合掉血时…"）对自伤牌完全不响应。
        # 这里改用 :func:`deal_raw_damage`（同一条"不经格挡"的通道），
        # ``attacker=None`` 对应真机"没有外部攻击者"；``card=played_card``
        # 则是真机的 ``cardSource`` —— `RupturePower` 要靠它把这份力量
        # **推迟到这张牌结算完**再发（否则"先自伤、后结算伤害"的牌会被凭空加强）。
        deal_raw_damage(state, source, eff.amount, events, label="失去生命",
                        card=played_card)
    elif eff.op == "gain_energy":
        state.energy += eff.amount
        events.append(f"能量 +{eff.amount}")
    elif eff.op == "set_power_var":
        # ⭐ ``PowerCmd.Apply<X>(…)?.SetDamage(值)``：把量写进**能力实例**。
        # ``TheBombPower`` / ``ToricToughnessPower`` 靠它知道自己该打多少 / 给多少格挡。
        # ``from="last_block_gain"`` 时取的是最近一次 ``GainBlock`` 的**返回值**
        # （实际获得量，不是卡面值）。目标解析与 `apply_power` 走同一套规则。
        # ``from="last_block_gain"`` 时取的是最近一次 ``GainBlock`` 的**返回值**
        # （实际获得量，不是卡面值）。目标解析与 `apply_power` 走同一套规则。
        if eff.set_from == "remembered_card":
            # ⭐ `Apply(...).SetSelectedCard(<选牌结果>)`：写进去的是**那张卡的克隆**
            # （`NightmarePower.SetSelectedCard` → `CreateClone()` + `ClearAffliction`），
            # 不是数值。再克隆一次是为了让"能力实例里的快照"与"手牌里那张牌"
            # 彻底脱钩：原卡之后被升级 / 转化都不该改到这份载荷。
            remembered = state.remembered_card
            value = remembered.clone() if remembered is not None else None
        elif eff.set_from == "last_block_gain":
            value = state.last_block_gain
        else:
            value = eff.amount

        def _set_var(target: Combatant) -> None:
            target.power_vars.setdefault(eff.power or "", {})[eff.var] = value

        if eff.target == "self":
            _set_var(source)
        elif eff.target == "all_enemies":
            for target in _opponents_of(state, source):
                _set_var(target)
        else:
            target = _primary_target(state, source, target_index)
            if target is not None:
                _set_var(target)
        events.append(f"{source.name} 设置 {eff.power}.{eff.var} = {value}")
    elif eff.op == "gain_stars":
        # ⭐ 走**唯一**的加星入口：`PlayerCmd.GainStars` 会分发
        # ``Hook.AfterStarsGained``（``PlayerCmd.cs:95``），`BlackHolePower` 挂在那里。
        gain_stars(state, eff.amount, events)
    elif eff.op == "move_all_matching":
        moved = _move_matching_cards(state, eff)
        events.append(f"移动 {moved} 张牌（{eff.select_from} → {eff.pile}）")
    elif eff.op == "channel":
        from . import orbs as orb_rules
        orb_rules.channel(state, eff.orb, events)
    elif eff.op == "evoke_next":
        from . import orbs as orb_rules
        for _ in range(max(1, eff.amount)):
            # `dequeue: false` = 激发但**不移除**那个球（`DualCast` 先用它激发一次）
            orb_rules.evoke(state, 0, events, dequeue=eff.dequeue)
    elif eff.op == "add_orb_slots":
        from . import orbs as orb_rules
        orb_rules.add_slots(state, eff.amount, events)
    elif eff.op == "heal":
        healed = min(eff.amount, source.max_hp - source.hp)
        if healed > 0:
            source.hp += healed
        events.append(f"{source.name} 回复 {healed} 点生命")
    elif eff.op == "upgrade_hand":
        # ⭐ ``CardCmd.Upgrade(PileType.Hand.GetPile(owner).Cards, style)``
        # （``Bellows.AfterPlayerTurnStart``，Bellows.cs）：把**手牌全部**升级。
        # 真机对**已经升级过**的牌是空操作（``CardCmd.Upgrade`` 只升基础形态），
        # 所以这里跳过它们 —— 顺手数一下升了几张，好在日志里看得出来。
        upgraded_cards = 0
        for hand_card in state.hand:
            if not hand_card.upgraded:
                hand_card.upgraded = True
                upgraded_cards += 1
        events.append(f"手牌全部升级（{upgraded_cards} 张）")
    elif eff.op == "random_draw_cards":
        # ⭐ ``StoneCracker.AfterRoomEntered``：**战斗开始时**从抽牌堆随机取
        # ``Cards`` 张（不重复）升到升级版::
        #
        #     PileType.Draw.GetPile(owner).Cards.Where(c => c.IsUpgradable)
        #         .ToList().StableShuffle(RunState.Rng.CombatCardSelection).Take(Cards);
        #     CardCmd.Upgrade(cards, …);
        #
        # ⚠️ 与 `random_deck_cards`（Run 层的**牌组**）是两回事：这里动的是
        # **战斗内的抽牌堆**，用的是战斗侧的 `combat_card_selection` 流。
        from . import powers as _powers  # noqa: F401  （保持与相邻分支一致的导入风格）
        spec = dict(eff.card_filter)
        if spec.get("upgradable"):
            candidates = [card for card in state.draw_pile if not card.upgraded]
        else:
            candidates = list(state.draw_pile)
        count = max(0, int(eff.amount or 0))
        if eff.pick == "shuffle":
            state.hidden.rng.shuffle(candidates, eff.rng)
            chosen = candidates[:count]
        else:                                   # pragma: no cover - 暂无内容用
            chosen = candidates[:count]
        for card in chosen:
            card.upgraded = True
            events.append(f"抽牌堆随机升级 {card.definition().name}")
        if not chosen:
            events.append("⚠️ 抽牌堆里没有符合条件的牌，跳过")
    elif eff.op == "add_card":
        assert eff.card is not None
        for _ in range(eff.amount):
            # ``CardCmd.Upgrade(刚生成的牌)`` → 直接造成**升级版实例**
            # （`cunning_potion` 的 Shiv / `cosmic_concoction` 的无色牌）。
            _add_generated_card(state, CardInstance(eff.card, eff.upgrade),
                                eff.pile, eff.position)
    elif eff.op == "generate_card":
        # ⭐ ``CardFactory.GetDistinctForCombat(卡池过滤, N, Rng.CombatCardGeneration)``
        # 之后紧跟 ``CardPileCmd.AddGeneratedCard(s)ToCombat(…, PileType.Hand, owner)``：
        # **随机生成 N 张互不重复的牌**（`white_noise` / `infernal_blade` /
        # `jack_of_all_trades` / `toolbox` / 四瓶"选一张"药水的候选来源…）。
        #
        # 引擎这边**不用新写一遍**：`powers.generate_cards_to_hand` 就是照
        # `GetDistinctForCombat` 复刻的（互不重复 / 走 combat_card_generation 流 /
        # 过准入门禁），能力侧（`CreativeAiPower` / `HelloWorldPower`）一直在用它。
        # 这里只是把**卡牌与药水**那条入口接上去。
        from . import powers as power_rules
        spec = dict(eff.card_filter)
        pool = str(spec.get("pool") or "")
        # ``pool="colorless"`` → 无色池；其余（含空）交给 generate_cards_to_hand
        # 按**当前角色**取池（真机是 `player.Character.CardPool`）。
        pool_name = "ColorlessCardPool" if pool == "colorless" else None
        generated = power_rules.generate_cards_to_hand(
            state, eff.amount, events, pool=pool_name,
            card_type=str(spec.get("card_type") or "") or None,
            rarity=str(spec.get("rarity") or "") or None,
            exclude=eff.exclude,
            free=eff.free, upgrade=eff.upgrade)
        events.append(f"生成 {len(generated)} 张牌（{eff.card_filter}）")
    elif eff.op == "shuffle":
        # ⭐ ``CardPileCmd.Shuffle``（``CardPileCmd.cs:1073-1131``）：把**弃牌堆**
        # 全部取出来、接到抽牌堆后面，整体 ``StableShuffle(Rng.Shuffle)``，
        # 再放回抽牌堆，最后 ``await Hook.AfterShuffle``。顺序照抄源码。
        # ⚠️ 挂起由 `_apply_effects` 循环末尾那一段统一处理（剩余效果等选完再跑），
        # 这里**不**自己判断返回值。
        shuffle_piles(state, events)
    elif eff.op == "autoplay_from_draw":
        # ⭐ ``CardPileCmd.AutoPlayFromDrawPile``（``CardPileCmd.cs:1145-1180``）：
        # 循环 N 次 { 抽牌堆空则把弃牌堆洗进来 → 取一张（``Top`` = 列表第一个）
        # → 移到出牌区 }，**然后**才逐张 ``CardCmd.AutoPlay``。
        # 取牌与打出分成两轮是关键：先取够 N 张再打，所以"打出过程中抽上来的牌"
        # 不会被算进这 N 张里。
        taken: list[CardInstance] = []
        for _ in range(max(0, eff.amount)):
            if not state.draw_pile:
                if not state.discard:
                    break
                state.draw_pile = state.discard
                state.discard = []
                state.hidden.rng.shuffle(state.draw_pile, "shuffle")
                # ⚠️ **已知缺口**：这条路径（`autoplay_from_draw`）的洗牌**没有**
                # 派发 `Hook.AfterShuffle`，所以 `StratagemPower` 不会在这里触发。
                # 原因是"取 N 张再逐张自动打出"的循环中途挂起，需要把已经取出的
                # 牌一起存进帧 —— 属于另一批工作。这里如实记账，
                # 而不是"假装洗了就算"（`docs/12` §2.40.5）。
                events.append("抽牌堆已空，弃牌堆洗回")
            if not state.draw_pile:
                break
            # 引擎里抽牌堆**尾部**是顶（`draw_cards` 用 `pop()`）。
            card = (state.draw_pile.pop() if eff.position == "top"
                    else state.draw_pile.pop(0))
            taken.append(card)
        for card in taken:
            if not state.player.alive():
                break
            if eff.force_exhaust:
                # ``item.ExhaustOnNextPlay = forceExhaust``（`Havoc` 的那一半）
                card.exhaust_on_next_play = True
            if not autoplay_card(state, card, events):
                # 挂在选牌上：剩下的牌**不再继续**（真机是顺序 await，
                # 选牌没完成就走不到下一张）。续跑由 `_resolve_selection` 负责。
                break
    elif eff.op == "procure_random_potion":
        obtained = _procure_random_potions(state, eff.amount, events)
        events.append(f"获得药水 × {obtained}")
    else:  # pragma: no cover - content.lint() 已拦截
        raise ValueError(f"未知算子 {eff.op!r}")


# ==========================================================================
# 抽牌 / 洗牌（隐藏信息的核心来源）
# ==========================================================================
def shuffle_piles(state: CombatState, events: list[str]) -> bool:
    """``CardPileCmd.Shuffle``：弃牌堆并入抽牌堆后整体洗牌，再派发 ``AfterShuffle``。

    顺序照抄源码（``CardPileCmd.cs:1073-1131``）：先把**弃牌堆**全部取出、
    接上**抽牌堆**，整体 ``StableShuffle(Rng.Shuffle)``（引擎的 ``"shuffle"`` 流），
    放回抽牌堆，**最后**才 ``await Hook.AfterShuffle``。

    返回 ``True`` = 洗完牌后有处理者挂起了玩家选择（``StratagemPower``），
    调用方必须停下来等 —— 抽牌循环与效果链各有一套续跑方式。
    """
    merged = list(state.discard) + list(state.draw_pile)
    state.discard = []
    state.hidden.rng.shuffle(merged, "shuffle")
    state.draw_pile = merged
    events.append(f"洗牌（弃牌堆并入，共 {len(merged)} 张）")
    from . import powers as power_rules
    return power_rules.on_after_shuffle(state, events)


def draw_cards(state: CombatState, count: int, events: list[str],
               from_hand_draw: bool = False,
               resume_turn_start: bool = False) -> None:
    """抽牌。**手牌满就停止抽牌**（``CardPileCmd.DrawInternal``）。

    出处：``CardPileCmd.DrawInternal``::

        int num = Math.Max(0, CardPile.MaxCardsInHand - hand.Cards.Count);
        if (num == 0) { CheckIfDrawIsPossibleAndShowThoughtBubbleIfNot(player); return result; }
        for (int i = 0; i < drawsRequested; i++) {
            if (num <= 0) break;
            ...
            if (card == null || hand.Cards.Count >= CardPile.MaxCardsInHand) break;
            ...
            num = Math.Max(0, CardPile.MaxCardsInHand - hand.Cards.Count);
        }

    ⚠️ 少了这道闸门，手牌会超过 `MaxCardsInHand` —— 那个状态在真机上**不存在**，
    而后果是"观测编码越界"（实测训练跑到某一步抛"手牌 11 超过容量 10"）。
    牌留在抽牌堆里，不会因为"抽不下"而消失。
    """
    if count <= 0:
        return
    if len(state.hand) >= MAX_HAND_SIZE:
        events.append(f"手牌已满（{MAX_HAND_SIZE}），不抽牌")
        return
    drawn_count = 0
    for _ in range(count):
        if len(state.hand) >= MAX_HAND_SIZE:
            events.append(f"手牌已满（{MAX_HAND_SIZE}），停止抽牌（还差 {count - drawn_count} 张留堆）")
            break
        if not state.draw_pile:
            if not state.discard:
                break
            # ⭐ ``CardPileCmd.Shuffle`` + ``Hook.AfterShuffle``（`CardPileCmd.cs:1047/1131`）：
            # 真机在抽牌循环里 `await ShuffleIfNecessary(...)`，洗完牌就分发 AfterShuffle。
            if shuffle_piles(state, events):
                # ⭐ **S02 抽牌挂起**：`AfterShuffle` 的处理者（`StratagemPower`）
                # 要玩家从抽牌堆挑牌入手。真机这里是一个 `await`，整个抽牌
                # ——以及调用它的 `SetupPlayerTurn`——都停在原地。
                # 引擎把"还要抽几张"存成帧，由 `_resume_after_selection` 接着抽。
                state.draw_frame = DrawFrame(
                    remaining=count - drawn_count,
                    from_hand_draw=from_hand_draw,
                    resume_turn_start=resume_turn_start)
                return
        if not state.draw_pile:
            break
        drawn = state.draw_pile.pop()
        state.hand.append(drawn)
        drawn_count += 1
        # `AfterCardDrawn`：抽到牌时触发（`ChainsOfBindingPower` 给牌贴 `Bound`）。
        # ⚠️ 真机是**两遍遍历**（``Hook.cs:202-213``）：先 `AfterCardDrawnEarly`、
        # 再 `AfterCardDrawn`。`HellraiserPower`（地狱使者）挂在第一遍，
        # 所以它必须在"抽牌时"的其它能力之前把这张 Strike 自动打出去。
        from . import powers as power_rules
        power_rules.on_draw_card_early(state, drawn, events,
                                       from_hand_draw=from_hand_draw)
        power_rules.on_draw_card(state, drawn, events,
                                 from_hand_draw=from_hand_draw)


# ==========================================================================
# 敌人 AI
# ==========================================================================
def _effective_effects_on(effects, value: int):
    """把一组效果里的伤害换成**修正后**的数值（进阶 / 源码提取值）。"""
    if not any(e.op in ("damage", "damage_all") for e in effects):
        return effects
    return tuple(
        replace(e, amount=value) if e.op in ("damage", "damage_all") else e
        for e in effects
    )


def _effective_effects(move, value: int):
    """把招式里的伤害效果换成**修正后**的数值（进阶 / 源码提取值）。

    真机的一招里伤害只来自意图，所以"意图上的数字"就是结算数字。
    我们的数据把两者分开存放（意图数值 + 效果列表），必须在这里对齐，
    否则会出现"显示 16 实际打 14"这种对拍必然失败的情况。
    """
    return _effective_effects_on(move.effects, value)


def _effects_for_move(state: CombatState, enemy: "EnemyState", move, value: int):
    """这一招**这一次**要跑的效果 —— ``effects_by_counter`` 优先。

    真机 ``KnowledgeDemon.CurseOfKnowledgeMove`` 读
    ``_curseOfKnowledgeSets[CurseOfKnowledgeCounter]``：同一个招式状态跑三次、
    每次的候选与数值都不同。

    ⚠️ 找不到匹配组时返回**空**，**不是**退回 ``move.effects`` ——
    那会把"第 4 轮"当成"第 1 轮"再打一遍，而且不会有任何报错。
    """
    if move.effects_by_counter:
        name, groups = move.effects_by_counter
        current = int((getattr(enemy, "ai_counters", {}) or {}).get(name, 0))
        for key, effects in groups:
            if int(key) == current:
                return _effective_effects_on(effects, value)
        return ()
    return _effective_effects(move, value)


def _finish_enemy_move(state: CombatState, enemy: "EnemyState", move,
                       events: list[str]) -> None:
    """一招跑完的收尾：自爆 + 实例计数器递增。

    真机每个招式函数**自己**写收尾::

        // ExplodeMove 结尾
        await CreatureCmd.Kill(base.Creature);
        // CurseOfKnowledgeMove 结尾
        if (base.CombatState.IsLiveCombat()) CurseOfKnowledgeCounter++;

    引擎把它集中成一处，因为"正常跑完"与"S02 挂起后续跑"是**两条路径**
    （:func:`_run_enemy_actions` / :func:`_resume_enemy_turn`）—— 各写一份必然
    漏掉一处，而漏掉递增的表现是"这只怪永远走同一条分支"，完全不会报错。
    """
    if move is not None and move.kills_self:
        mark_dead(state, enemy, events, cause="自爆")
    name = getattr(move, "counter_increment", "") if move is not None else ""
    # `IsLiveCombat()`：只有**带守卫**的那种才在胜负已分后停手
    # （`KnowledgeDemon.cs:172` 有守卫、`TestSubject.cs:256` 是裸的 `Respawns++`）。
    live_only = bool(getattr(move, "counter_increment_live_only", False))
    if name and (not live_only or not state.finished()):
        enemy.ai_counters[name] = int(enemy.ai_counters.get(name, 0)) + 1


def _ascension_value(ai_record: dict, handler: str, fallback: int, ascension: int) -> int:
    """招式数值的进阶修正。

    真机写 `GetValueIfAscension(DeadlyEnemies, 16, 14)`——**A9 起怪物加伤**。
    这里用提取出来的精确值，而不是按比例缩放。
    """
    entry = ((ai_record or {}).get("move_values") or {}).get(handler)
    if not entry:
        return fallback
    if ascension >= entry.get("level", 99):
        return int(entry.get("ascension", fallback))
    return int(entry.get("normal", fallback))


def init_enemy_ai(enemy: EnemyState, position: int,
                  state: CombatState | None = None) -> None:
    """给一只怪初始化**专属运行状态**（槽位 + 实例标志）。

    必须显式初始化而不是留给"查不到就当 False"：
    ``docs/13`` §5 要求缺失必需值能被发现。出厂值来自
    ``monster_ai.INSTANCE_FLAG_DEFAULTS``（对应 C# 字段默认 ``false``）。

    同时校验这只怪的条件里引用到的标志**都在已知表内** —— 拼错一个标志名
    在真机里不可能发生（编译期检查），在我们这里必须由这里拦住。
    """
    from . import monster_ai

    if not enemy.slot_name:
        enemy.slot_name = monster_ai.slot_name_for(position)
    for name, value in monster_ai.INSTANCE_FLAG_DEFAULTS.items():
        enemy.ai_flags.setdefault(name, value)
    for name, value in monster_ai.COUNTER_DEFAULTS.items():
        enemy.ai_counters.setdefault(name, value)
    record = getattr(enemy.definition(), "ai_record", None) or {}
    unknown_flags = monster_ai.flags_referenced(record) - set(monster_ai.FLAG_RULES)
    unknown_counters = (monster_ai.counters_referenced(record)
                        - set(monster_ai.COUNTER_SOURCES))
    if unknown_flags or unknown_counters:
        raise monster_ai.UnsupportedMechanic(
            f"{enemy.eid}: 条件引用了未登记的状态 "
            f"flags={sorted(unknown_flags)} counters={sorted(unknown_counters)}")


def next_open_slot(state: CombatState) -> str:
    """``EncounterModel`` 取下一个**未被占用**的槽位（召唤用）。

    真机 ``EncounterModel``：``Slots.FirstOrDefault(s => combatState.Enemies.All(
    c => c.SlotName != s))`` —— 第一个"没有敌人占着"的槽位名。
    """
    from . import monster_ai

    used = {enemy.slot_name for enemy in state.enemies if enemy.alive()}
    for name in monster_ai.SLOT_ORDINALS:
        if name not in used:
            return name
    raise monster_ai.UnsupportedMechanic("没有空闲槽位（真机最多 5 个敌人槽位）")


def build_machine(ai_record) -> object | None:
    """从提取记录构造出招状态机；没有记录就返回 None（退回近似逻辑）。"""
    if not ai_record:
        return None
    from .monster_ai import build
    return build(ai_record)


def build_monster_context(state: CombatState, enemy: EnemyState):
    """给一只怪构造**真实的**条件求值上下文（审计 F05）。

    审计实测：``_choose_intent`` 构造 ``MonsterContext`` 时既没传专属 flags/counters，
    槽位又被简化成 first/second，而访问器把缺失值默默当成 False/0 —— 于是
    ``Fabricator.CanFabricate``（存活队友 < 4）在单独生成时被判成 False，
    初始意图变成 ``DISINTEGRATE_MOVE``（正确分支是**随机召唤分支**）。

    这里把三件事都补上：

    1. **槽位**来自位置序数（:func:`monster_ai.slot_name_for`），不再是 first/second；
    2. **队友数量**按真机 ``GetTeammatesOf`` 的语义算（**含自己**）；
    3. **专属标志**由 :func:`monster_ai.resolve_flags` 现算 + 实例标志合并，
       缺失值抛 ``UnsupportedMechanic`` 而不是默认 False。
    """
    from . import monster_ai

    position = state.enemies.index(enemy)
    living = state.living_enemies()
    return monster_ai.MonsterContext(
        slot_name=enemy.slot_name or monster_ai.slot_name_for(position),
        is_front=position == 0,
        is_alone=len(living) == 1,
        ally_count=max(0, len(living) - 1),
        hp=enemy.hp, max_hp=enemy.max_hp,
        powers=frozenset(enemy.powers),
        flags=monster_ai.resolve_flags(living, enemy.ai_flags),
        counters=dict(enemy.ai_counters),
    )


def _run_stun_action(state: CombatState, enemy: "EnemyState",
                     events: list[str]) -> None:
    """``CreatureCmd.Stun(creature, stunMove, …)`` 里 ``stunMove`` 的**数值部分**。

    真机的 ``WakeUpMove`` 几乎全是动画/音效，唯一的数值动作是
    ``SlumberingBeetle.WakeUpMove`` 里的"有 Plating 就移除它"
    （``AsleepPower.AfterDamageReceived`` 那一条则是在**挨打当下**就移除）。
    这里只按名字执行这些动作 —— 不塞可调用对象是为了让 ``EnemyState``
    仍然可以直接深拷贝（快照/SL 的基础）。
    """
    if enemy.stun_action == "remove_plating" and enemy.power("plating") > 0:
        enemy.add_power("plating", -enemy.power("plating"))
        events.append(f"{enemy.name} 醒来，失去镀层")


def _choose_intent(state: CombatState, enemy: EnemyState,
                  events: list[str] | None = None) -> None:
    definition = enemy.definition()

    # ⭐ **眩晕**（``CreatureCmd.Stun``）：注入一招"什么都不做"，只生效一次。
    # 真机是给怪物塞一个临时 ``MoveState``（``StunnedMove``），下一回合走完就恢复。
    # 这里用同一个表达：把意图设成一个**招式表里不存在**的 id ——
    # `_run_enemy_turn` 找不到对应招式就不会施加任何效果，正是"被晕住"。
    # ⚠️ 不能靠"移除意图"来表达：那样敌人会一直不行动（永久眩晕）。
    #
    # ⭐ 但 ``Stun(creature, stunMove, nextMoveId)`` 还有两件事（`asleep` / `slumber` 用）：
    #   * ``stunMove`` —— 这一回合要执行的**唤醒动作**（真机是 ``WakeUpMove``，
    #     里面唯一的数值部分是"移除 Plating"）；
    #   * ``nextMoveId`` —— **下一回合**固定走哪一招（拉瓦金的 ``SLASH_MOVE``、
    #     睡甲虫的 ``ROLL_OUT_MOVE``）。
    if enemy.power_flags.pop("stunned", 0):
        enemy.intent = Intent("stun", "STUNNED_MOVE", 0, 1)
        state.hidden.move_queue = list(enemy.machine.state_log[-3:]) \
            if enemy.machine is not None else []
        if enemy.stun_action:
            _run_stun_action(state, enemy, events or [])
        if enemy.stun_follow_up:
            # 真机把 ``FollowUpStateId`` 记在那个临时状态上，走完就落到它。
            # 引擎在**这一步**把状态机挪过去（`SetMoveImmediate` 的等价物），
            # 并把"下一回合就用它"记进 ``forced_move`` —— 因为状态机在
            # "已经出过第一招"之后会先转移再落招，只 set_immediate 会被跳过。
            enemy.forced_move = enemy.stun_follow_up
            if enemy.machine is not None and enemy.stun_follow_up in enemy.machine.states:
                enemy.machine.set_immediate(enemy.stun_follow_up)
        enemy.stun_action = ""
        enemy.stun_follow_up = ""
        return

    # ⭐ ``nextMoveId`` 的兑现：**下一回合固定走这一招**，之后恢复状态机。
    if enemy.forced_move:
        forced = enemy.forced_move
        enemy.forced_move = ""
        move = next((m for m in definition.moves if m.mid == forced), None)
        if move is None:
            raise KeyError(f"{enemy.eid}: 眩晕的后续招式 {forced!r} 不在招式表里")
        enemy.intent = Intent(move.intent, move.mid, move.value, move.times)
        state.hidden.move_queue = list(enemy.machine.state_log[-3:]) \
            if enemy.machine is not None else []
        return

    # ⭐ 首选：真机出招状态机（从反编译源码提取，见 docs/09 L1）。
    # 精确复刻初始招、转移链、随机分支的权重与重复约束、条件分支。
    if enemy.machine is not None:
        from . import monster_ai
        move_state = enemy.machine.roll_move(
            build_monster_context(state, enemy),
            state.hidden.rng["monster_ai"])
        ai_record = getattr(definition, "ai_record", None) or {}
        # ⚠️ 状态机里的 `move` 是**处理函数名**（`ClawMove`），而怪物表的招式 id 是
        # `CLAW`。把状态 id 剥掉 `_MOVE` 也作为候选，两种命名都能对上。
        state_key = move_state.state_id.lower()
        candidates = {state_key, state_key.removesuffix("_move"), move_state.move.lower()}
        move = next((m for m in definition.moves if m.mid.lower() in candidates), None)
        if move is None:
            # 状态机指向了怪物表里没有的招式 → 大声报错，不静默退化
            raise KeyError(f"{enemy.eid}: 状态机指向未知招式 {move_state.state_id!r}"
                           f"（候选 {sorted(candidates)}）")
        value = _ascension_value(ai_record, move_state.move, move.value, state.ascension)
        enemy.intent = Intent(move.intent, move.mid, value, move.times)
        state.hidden.move_queue = list(enemy.machine.state_log[-3:])
        return

    # ⭐ 真机的 attack_pattern.type == "cycle" 可以**忠实还原**：按固定顺序循环。
    # 用已执行招式的数量取模，因此完全确定、可复现。
    if definition.cycle:
        move_id = definition.cycle[len(enemy.move_history) % len(definition.cycle)]
        move = next((m for m in definition.moves if m.mid == move_id), None)
        if move is not None:
            enemy.intent = Intent(move.intent, move.mid, move.value, move.times)
            state.hidden.move_queue = list(definition.cycle)
            return

    # 其余模式（random / conditional / 未知）退回权重随机——**这是近似，不是还原**。
    # 哪些怪属于这一类可以从 EnemyDef.pattern_type 看出来（见 tools/healthcheck.py）。
    pool: list[tuple[MoveDef, float]] = []
    for move in definition.moves:
        if move.first_turn_only and state.turn != 1:
            continue
        if not move.after_first_turn and state.turn == 1:
            continue
        pool.append((move, move.weight))
    if not pool:
        pool = [(definition.moves[-1], 1.0)]
    move = state.hidden.rng.weighted_choice(pool, "monster_ai")
    enemy.intent = Intent(move.intent, move.mid, move.value, move.times)
    # 未来队列（L3）——骨架里只做展示，证明"有隐藏信息但进不了观测"
    state.hidden.move_queue = [
        state.hidden.rng.choice(definition.moves, "monster_ai").mid for _ in range(3)
    ]


def _run_enemy_turn(state: CombatState, events: list[str]) -> bool:
    """敌方阵营回合。

    返回 ``True`` = 整轮跑完；``False`` = **挂在玩家的一次选择上**
    （`state.pending` 与 `state.enemy_action` 都已设置，由
    :func:`_resume_enemy_turn` 续跑）—— 见 ``CombatState.enemy_action`` 的说明。
    """
    from . import powers as power_rules
    state.phase = "enemy"
    # ⭐ `Hook.BeforeSideTurnStart` 的**敌方**那一次（``side="enemy"``）：
    # 与玩家侧同一个钩子、同一个时机（清格挡之前）。目前没有敌人能力用它，
    # 但声明了就必须分发 —— 少一次分发是静默的。
    power_rules.on_before_side_turn_start(state, events, "enemy")
    # ⚠️ **不要**在这里清空玩家格挡：真机是在**玩家回合开始**清空
    # （`start_player_turn` 已经做了）。这里清一次会让敌人打的是无格挡的玩家，
    # 玩家成片暴死 —— 表现为"规则 bot 与随机策略打平"（区分度测试抓到的）。
    #
    # ⭐ 但**敌方格挡必须在这里清**：`Creature.AfterTurnStart(side)` 双方阵营都清，
    # 敌方那次就发生在敌方回合开始。少了它，会加格挡的怪（`guardbot` 每回合 15）
    # 格挡会**无限累积** —— 实测 4 回合后 15→60，那只怪变成打不死的。
    # 例外：`BurrowedPower`（潜伏）声明 `ShouldClearBlock → false`。
    for enemy in list(state.enemies):
        if enemy.alive() and not power_rules.blocks_block_clear(enemy):
            if enemy.block:
                events.append(f"{enemy.name} 的格挡被清空（{enemy.block}）")
            enemy.block = 0
        if enemy.alive():
            # ⭐ 与玩家侧同一个钩子（``CombatManager.cs:768`` 对**每个**开始回合的单位
            # 都分发），而且**不看有没有格挡**。
            power_rules.on_block_cleared(enemy, state, events)
    # ⭐ 敌方阵营回合开始：**这里是中毒等能力触发的地方**。
    # 少了这一步，敌人身上的中毒就永远不掉血 —— 而中毒正是减速/消耗战术的核心。
    for enemy in list(state.enemies):
        if not enemy.alive():
            continue
        power_rules.on_turn_start(enemy, state, events)
        if power_rules.dies_to_doom(enemy):
            # ⚠️ 必须走统一死亡入口：直接 hp = 0 会绕过 AfterDeath，
            # 于是 crab_rage / stock / surprise / minion 这些死亡触发的机制
            # 在末日处决这条路径上**静默不生效**（实测漏了 4 条路径）。
            mark_dead(state, enemy, events, cause="末日处决")
    if not state.player.alive():
        state.phase = "lost"
        return True
    return _run_enemy_actions(state, events, 0)


def _run_enemy_actions(state: CombatState, events: list[str],
                       start_index: int) -> bool:
    """从第 ``start_index`` 只怪开始出招。返回 ``False`` = 挂起等选择。

    ⚠️ ``start_index`` 只在**续跑**时非 0（:func:`_resume_enemy_turn` 传
    ``frame.enemy_index + 1``）：把整轮从头重跑会让已经出过手的怪**再打一次**，
    而日志里看起来只是"这一轮多打了一下"，不会报任何错。
    """
    from . import powers as power_rules
    for index in range(start_index, len(state.enemies)):
        enemy = state.enemies[index]
        if not enemy.alive():
            continue
        # ⭐ 复活中的单位这一回合**用来复活**，不出招。
        # 真机是 SetMoveImmediate(REVIVE_MOVE) —— 那一招就是回血。
        if getattr(enemy, "reviving", False):
            power_rules.perform_revive(enemy, state, events)
            _choose_intent(state, enemy, events)
            continue
        intent = enemy.intent
        if intent is not None:
            move = next((m for m in enemy.definition().moves if m.mid == intent.mid), None)
            if move is not None:
                enemy.move_history.append(move.mid)
                if enemy.machine is not None:
                    enemy.machine.on_move_performed()
                events.append(f"{enemy.name} 使用 {move.mid}")
                # ⚠️ **必须用意图上已修正的数值**，不能直接套 JSON 的效果数值：
                # 否则进阶下"显示的伤害"变了、**实际结算的伤害没变**——
                # 是一个看起来正常、但对拍一定失败的静默错误。
                # ⭐ `_effects_for_move`：带 `effects_by_counter` 的招式按**实例计数器**
                # 现算这一次的载荷（KnowledgeDemon 的三轮二选一）。
                done = _apply_effects(state,
                                      _effects_for_move(state, enemy, move, intent.value),
                                      enemy, 0, events)
                if not done:
                    # ⭐ **S02 挂起**：把"哪只怪、哪一招、要不要自爆"记成**数据**。
                    # 该招剩余的效果已经在 `state.pending.rest` 里，这里不重复存。
                    state.enemy_action = EnemyActionFrame(
                        enemy_index=index, move_mid=move.mid,
                        kills_self=bool(move.kills_self))
                    return False
                _finish_enemy_move(state, enemy, move, events)
        if not state.player.alive():
            state.phase = "lost"
            return True
        if not enemy.alive():
            continue
        _choose_intent(state, enemy, events)
    _end_enemy_side_turn(state, events)
    return True


def _resume_enemy_turn(state: CombatState, events: list[str]) -> bool:
    """选牌结束、且这次选择挂在**敌方回合**上 → 把剩下的出招跑完。

    返回 ``True`` = 敌方回合真的跑完了（调用方接着推进玩家回合）；
    ``False`` = 又挂在下一只怪的选择上。
    """
    frame = state.enemy_action
    if frame is None:
        return True
    if state.pending is not None:
        # 还在选（`choose_one` 的候选不止一张，或者选项效果自己又开了选择）。
        return False
    state.enemy_action = None
    enemy = (state.enemies[frame.enemy_index]
             if 0 <= frame.enemy_index < len(state.enemies) else None)
    move = (next((m for m in enemy.definition().moves if m.mid == frame.move_mid), None)
            if enemy is not None else None)
    if enemy is not None and enemy.alive():
        if move is not None:
            # ⭐ 招式收尾（自爆 + 实例计数器递增）必须在**这里**也走一遍 ——
            # 挂起前只跑完了效果，收尾还没做。漏掉它，`CurseOfKnowledgeCounter`
            # 就永远停在 0（真机是在招式函数末尾递增的）。
            _finish_enemy_move(state, enemy, move, events)
        elif frame.kills_self:
            # 招式表里找不到这一招（数据变了）：至少把帧里记下的自爆补上。
            mark_dead(state, enemy, events, cause="自爆")
        if enemy.alive():
            _choose_intent(state, enemy, events)
    if not state.player.alive():
        state.phase = "lost"
        return True
    return _run_enemy_actions(state, events, frame.enemy_index + 1)


def _end_enemy_phase(state: CombatState, events: list[str]) -> None:
    """敌方回合真的结束 → 回合号 +1、进入玩家回合。

    ⚠️ 抽出来是因为它现在有**两个**调用点：``step`` 的 ``end_turn``（没挂起、
    一路跑完）与 ``select_card``（挂在怪物出招上，续跑完才轮到推进）。
    两份各写一遍迟早漂移，而漂移的表现是"回合号少加一次"这种全局错位。
    """
    if not state.player.alive():
        state.phase = "lost"
        return
    state.turn += 1
    start_player_turn(state, events)


def _resume_after_selection(state: CombatState, events: list[str]) -> None:
    """选牌结束之后，按**帧**决定接着跑什么（S02 的续跑调度）。

    三种情形，必须分开：

    * ``state.enemy_action`` 非空 —— 怪物出招中途的选择（`KnowledgeDemon`）：
      收尾这一招、把敌方回合剩下的出招跑完，跑完才推进玩家回合；
    * ``state.draw_frame`` 非空 —— 抽牌中途洗牌触发的选择（`Stratagem`）：
      接着把该抽的牌抽完；如果这一轮是**回合开始的手牌抽取**，
      还要把 `start_player_turn` 的剩余步骤跑完；
    * 两者都空 —— 这就是卡牌效果链自己的选择，`_resolve_selection` 已经跑完了。

    ⚠️ 少了这个调度，"挂在半路"的帧永远不会被消费：抽牌停在洗牌那一刻、
      回合开始的剩余步骤整段缺失，而 `state.pending` 已经清空 —— 不报错。
    """
    if state.pending is not None:
        return                       # 还在选（多张候选、或选项效果又开了选择）
    if state.enemy_action is not None:
        if _resume_enemy_turn(state, events):
            _end_enemy_phase(state, events)
        return
    frame = state.draw_frame
    if frame is not None and frame.remaining < 0:
        # ⭐ 挂起发生在 `BeforeHandDraw`（`ForegoneConclusion`）。这一段的顺序是：
        #   BeforeHandDraw 里**所有**待办（含"洗牌→挑牌→移除"的嵌套链）
        #   → 才轮到 `ModifyHandDraw` + 抽手牌。
        # ⚠️ 反过来（先抽手牌、再跑待办）会让"挑进手里的牌"晚一步到场，
        #    手牌张数在那一刻与真机不同 —— 而且不报错。
        _run_pending_tasks(state, events)
        if state.pending is not None:
            return                   # 待办自己又开了选择：帧留着，下次接着跑
        state.draw_frame = None
        if _draw_hand(state, events):
            return
        if frame.resume_turn_start:
            _finish_player_turn_start(state, events)
        return
    if frame is not None:
        state.draw_frame = None
        draw_cards(state, frame.remaining, events,
                   from_hand_draw=frame.from_hand_draw,
                   resume_turn_start=frame.resume_turn_start)
        if state.pending is not None:
            return                   # 又洗了一次牌、又挂了选择
        if frame.resume_turn_start:
            _finish_player_turn_start(state, events)
            if state.pending is not None:
                return
    elif state.turn_start_step:
        # ⭐ **回合开始后半段挂起**（`EntropyPower` 在 `AfterPlayerTurnStart` 挑牌转化）：
        # 从记下的那一步接着跑。
        step = state.turn_start_step
        state.turn_start_step = 0
        _finish_player_turn_start(state, events, step)
        if state.pending is not None:
            return
    _run_pending_tasks(state, events)


def _run_pending_tasks(state: CombatState, events: list[str]) -> None:
    """把**能力动作的待办**跑完（`ForegoneConclusionPower` 那类"选牌 → 移除自己"）。

    ⚠️ 真机的移除是**无条件**的：`FromCombatPile` 返回空集合时 `Add` 是空操作，
    但后面的 `PowerCmd.Remove(this)` 照样执行。所以"没有候选"这条分支也要移除能力，
    否则那张牌打出去之后能力永远留着（每回合都空问一次）。
    """
    while state.pending is None and state.pending_tasks:
        task = state.pending_tasks.pop(0)
        if task.kind == "select_to_hand":
            if open_pile_selection(state, events, purpose="to_hand",
                                   source_pile="draw", amount=task.amount,
                                   actor=task.actor):
                state.pending.remove_power_after = task.pid if task.remove else ""
                return
            if task.remove:
                _remove_power_whole(state, task.actor or state.player,
                                    task.pid, events)


def _remove_power_whole(state: CombatState, holder: Combatant, pid: str,
                        events: list[str]) -> None:
    """把某个能力**整个**移除（真机 ``PowerCmd.Remove(this)``，不是减层）。

    ⚠️ 用 ``add_power(pid, -层数)`` 也能清空，但那走的是"层数变化"那条路径，
    会派发 ``Hook.AfterPowerAmountChanged`` —— 真机的 ``Remove`` 是另一条路，
    **不**派发它。用错会让"层数变化时触发"的能力多响应一次（静默）。
    """
    if holder.power(pid) != 0:
        holder.powers.pop(pid, None)
        holder.powers_applier.pop(pid, None)
        events.append(f"{pid} 被移除（整个能力）")


# ==========================================================================
# 回合流转
# ==========================================================================
def tick_powers(combatant: Combatant, state: "CombatState | None" = None,
                events: list[str] | None = None) -> None:
    """拥有者所在阵营回合结束：跑能力钩子（镀层 / 再生）。

    ⚠️ **不在这里递减易伤 / 虚弱 / 脆弱**：真机那三个的递减条件是
    ``side == CombatSide.Enemy``（见 ``powers.DECREMENTS_AT_ENEMY_SIDE_TURN_END``），
    在玩家回合结束递减会让"敌人先上易伤再打你"少算一整个回合。
    递减由 ``_end_enemy_side_turn`` 统一处理。
    """
    from . import powers as power_rules
    if state is not None:
        power_rules.on_turn_end(combatant, state, events if events is not None else [])


def _end_enemy_side_turn(state: CombatState, events: list[str]) -> None:
    """敌方阵营回合结束：跑每个敌人的钩子，再对**场上所有单位**递减 duration。

    ⚠️ 在递减之前还要分发一次 ``AfterSideTurnEnd`` 的**全体**视角（``side="enemy"``）：
    玩家身上的能力也要知道"敌方回合结束了"（``FlameBarrierPower`` 就是在这时移除的，
    见 ``powers._flame_barrier_expire``）。
    """
    from . import powers as power_rules
    # ⭐ ``Hook.BeforeSideTurnEnd``：能力侧的"回合结束**之前**"（``TheBombPower`` /
    # ``HailstormPower`` 靠它结算 —— 它们要在回合真的结束前造成伤害）。
    power_rules.on_before_side_turn_end(state, events, "enemy")
    for enemy in state.enemies:
        if enemy.alive():
            power_rules.on_turn_end(enemy, state, events)
    power_rules.on_side_turn_end(state, events, "enemy")
    # ``AfterSideTurnEndLate``（第一遍之后、别的事情之前）—— 见玩家侧的同一处注释。
    power_rules.on_side_turn_end_late(state, events, "enemy")
    power_rules.tick_durations(state, events)


def start_player_turn(state: CombatState, events: list[str]) -> None:
    """玩家回合开始。**顺序照抄真机 ``CombatManager.StartTurn`` + ``SetupPlayerTurn``**。

    真机的顺序（``CombatManager.cs:688-927``，逐一对应过）::

        Creature.BeforeTurnStart            能力快照 AmountOnTurnStart
        Hook.BeforeSideTurnStart            ← 弹珠袋 / 破碎核心 / 力量电池 / 红面具
        Creature.AfterTurnStart → ClearBlock  ⚠️ **玩家第 1 回合不清格挡**
        Hook.AfterBlockCleared
        ── SetupPlayerTurn ──
        ResetEnergy                          能量重置
        Hook.AfterEnergyReset               ← 战争艺术 / 古茶具
        Hook.BeforeHandDraw
        ModifyHandDraw(…, 5m)                基础抽 5 张 + 遗物修正
        CardPileCmd.Draw
        Hook.AfterPlayerTurnStart           ← 皇家毒药 / 赌徒筹码
        ── 回到 StartTurn ──
        Hook.AfterSideTurnStart             ← 提灯 / 赤備 / 力量电池
        OrbQueue.AfterTurnStart              充能球的回合开始

    ⚠️ 两个曾经的错处：
    1. 格挡**无条件**清零。真机在 ``Creature.AfterTurnStart`` 里对**玩家第 1 回合
       直接 return**（``Creature.cs:686-697``）—— 这正是"锚"给的 10 点格挡
       能留到第 1 回合的原因。无条件清零会让所有开局格挡遗物静默失效。
    2. ``AfterSideTurnStart``（能力规则的 ``on_turn_start``）跑在**抽牌之前**。
       真机是**之后**：中毒掉血、下回合格挡都在手牌到手之后才结算。
    """
    from . import hooks
    from . import powers as power_rules
    from . import relics as relic_rules
    state.phase = "player"
    relics = tuple(getattr(state, "relics", ()) or ())
    # ⭐ `Hook.BeforeSideTurnStart`：**清格挡之前**的能力时机。
    # `AggressionPower`（侵略）在这里从弃牌堆捞牌进手牌 —— 放在抽牌之后
    # （`on_owner_turn_start`）会让"捞进来的牌"与"这一抽的牌"顺序与真机不同。
    power_rules.on_before_side_turn_start(state, events, "player")
    hooks.fire("relic_before_side_turn_start", state, events, relic_rules.dispatch)
    # `Creature.AfterTurnStart`：清格挡。两个例外——
    # 玩家第 1 回合豁免（`Creature.cs:686-697`），以及 `ShouldClearBlock` 为 false
    # 的能力（壁垒 / 潜伏）。锚的 10 点开局格挡能留到第 1 回合就靠前者。
    if state.turn != 1 and not power_rules.blocks_block_clear(state.player):
        state.player.block = 0
    # ⭐ ``Hook.AfterBlockCleared``（``CombatManager.cs:768``）：对**每个开始回合的单位**
    # 都分发一次，且**不看有没有格挡**（玩家第 1 回合不清格挡也照样分发）。
    # `ToricToughnessPower` / `SelfFormingClayPower` 挂在这里。
    power_rules.on_block_cleared(state.player, state, events)
    # ⭐ 真机 `CombatManager.StartTurn` 的**第 1 步**就是 `Creature.BeforeTurnStart`：
    # 给每个能力存一份 `AmountOnTurnStart = Amount`（`Creature.cs:682`）。
    for combatant in (state.player, *state.enemies):
        combatant.power_amount_on_turn_start = dict(combatant.powers)
    # 本回合的攻击牌计数清零（`JugglingPower` 的"每回合第 3 张"要靠它）。
    state.attacks_played_this_turn = 0
    state.skills_played_this_turn = 0
    state.tag_plays_finished_this_turn = {}
    # 本回合已开始结算的牌数清零（`EchoFormPower` 的"前 Amount 张打两次"要靠它）。
    state.cards_played_started_this_turn = 0
    state.attack_skill_plays_started_this_turn = 0
    # ⚠️ 这里**没有** ``power("energy")``：真机不存在叫 ``energy`` 的能力，
    # 每回合能量上限的加法修正一律走 ``Hook.ModifyMaxEnergy``
    # （`powers.max_energy_bonus`）。旧实现读了一个永远取到 0 的假 id ——
    # 它不会报错，只会让人以为"存在一个叫 energy 的能力"。
    # `tools/audit_power_refs.py` 就是查这一类（读到不存在的 id / 漏登记）。
    state.energy = (state.max_energy
                    + power_rules.max_energy_bonus(state)
                    + relic_rules.energy_bonus(relics, state.turn))
    # ⚠️ 必须**在能量重置之后**触发：真机的 `AfterEnergyReset` 语义如此，
    # 放在前面会被重置覆盖掉（`energy_next_turn` 会静默失效）。
    power_rules.on_energy_reset(state.player, state, events)
    hooks.fire("relic_energy_reset", state, events, relic_rules.dispatch)
    # ⭐ ``Hook.BeforeHandDraw``（``CallOfTheVoidPower`` / ``CreativeAiPower`` /
    # ``HelloWorldPower`` / ``SentryModePower`` 等在这时把生成的牌塞进手牌）。
    # 必须在**抽牌之前**：它们生成的是"这回合手牌的一部分"，
    # 放到抽牌之后会绕过手牌上限的判定顺序。
    power_rules.on_before_hand_draw(state, events)
    if state.pending is not None:
        # ⭐ **`BeforeHandDraw` 挂起**（`ForegoneConclusionPower` 要玩家先挑牌）。
        # 此刻连"抽几张"都还没算（真机的 `ModifyHandDraw` 排在它之后），
        # 所以帧用 `remaining = -1` 表示"整段手牌抽取都还没开始"。
        state.draw_frame = DrawFrame(remaining=-1, from_hand_draw=True,
                                     resume_turn_start=True)
        return
    if _draw_hand(state, events):
        return
    _finish_player_turn_start(state, events)


def _draw_hand(state: CombatState, events: list[str]) -> bool:
    """算抽牌数并抽手牌。返回 ``True`` = 挂在选牌上（调用方必须停下）。

    顺序照抄 ``CombatManager.SetupPlayerTurn``：``ModifyHandDraw`` →
    ``Innate`` 提升 → ``ShouldDraw`` 闸门 → ``CardPileCmd.Draw``。

    ⚠️ 抽出来是因为它现在有**两个**调用点：正常回合开始，以及
    "`BeforeHandDraw` 挂了选择、选完回来接着抽"（那时 `ModifyHandDraw` 还没算过）。
    """
    from . import powers as power_rules
    from . import relics as relic_rules
    relics = tuple(getattr(state, "relics", ()) or ())
    # `ModifyHandDraw`：基础 5 张，能力按**链式**修正（`ClarityPower` 恒 +1、
    # `DemesnePower` 等 +层数、`MindRotPower` 是 `Math.Max(0, count - 层数)`），
    # 之后才是遗物按回合守卫的加减（准备背包 +2、大蘑菇 -2）。
    # ⚠️ 三件事都不能改：
    #  1. 能力链必须走 `modify_hand_draw`（**链式**）而不是 `hand_draw_bonus`（净和）——
    #     `mind_rot` 带截零，`5 + 1 - 10` 与 `max(0, 5 - 10) + 1` 不是同一个数；
    #  2. 顺序是"能力 → 遗物"：真机 `CombatState.IterateHookListeners`
    #     先遍历各 creature 的能力、再遍历该玩家的遗物（`CombatState.cs:411`）；
    #  3. 两边都要算：只算遗物会"遗物给了、能力没给"，而且症状只是少抽一张。
    draw_count = (power_rules.modify_hand_draw(state, CARDS_PER_TURN)
                  + relic_rules.hand_draw_bonus(relics, state.turn))
    # ⭐ ``Innate``：真机 ``CombatManager.SetupPlayerTurn`` 只在**第 1 回合**处理
    # （``CombatManager.cs:908-923``）：先把 ``ShouldStartAtBottomOfDrawPile`` 的附魔牌
    # 压到底，再把带 Innate 的牌 ``MoveToTopInternal``，最后
    # ``handDraw = max(handDraw, 固有张数)``（**保证固有点都能上手**）。
    # 不做这一步，9 张固有牌（``backstab`` / ``apotheosis`` 这类核心牌）
    # 会和普通牌一样随机沉底 —— 完全不是它们的设计意图。
    if state.turn == 1:
        draw_count = _hoist_innate(state, draw_count, events)
    # `NoDrawPower.ShouldDraw` → false：这一回合不抽牌。
    if state.player.alive() and not power_rules.blocks_draw(state.player):
        # ⭐ `fromHandDraw=True`：真机 `Hook.AfterCardDrawn(..., fromHandDraw)`
        # 用这个标志区分「回合开始的手牌抽取」与「效果抽牌」——
        # `SpeedsterPower` 只在**非**手牌抽取时触发。
        draw_cards(state, max(0, draw_count), events, from_hand_draw=True,
                   resume_turn_start=True)
    return state.pending is not None


def _finish_player_turn_start(state: CombatState, events: list[str],
                              start_step: int = 0) -> None:
    """``start_player_turn`` 里**抽牌之后**的剩余步骤。

    抽出来是因为它现在有**三个**调用点：正常回合开始（一路跑完）、
    "抽牌中途洗牌挂了选择、选完再回来"、以及"某一步的能力挂了选择、选完再回来"
    （:func:`_resume_after_selection`）。各写一份必然漂移，
    而漂移的表现是"那一回合少跑了一段回合开始流程"。

    ``start_step`` 是**挂起后续跑**用的下标：真机这里是一串 ``await``，
    任何一步里的能力开了玩家选择（`EntropyPower` 在 `AfterPlayerTurnStart`
    要挑牌转化），整个 ``SetupPlayerTurn`` 都停住，剩下的步骤等选完再跑。
    挂起时把"下一步是第几"写进 ``state.turn_start_step``。
    """
    from . import hooks
    from . import orbs as orb_rules
    from . import powers as power_rules
    from . import relics as relic_rules
    steps: tuple[tuple[str, object], ...] = (
        ("遗物：抽牌后回合开始",
         lambda: hooks.fire("relic_after_player_turn_start", state, events,
                            relic_rules.dispatch)),
        # ⭐ `Hook.AfterPlayerTurnStart`（``CombatManager.cs:910``）：抽牌之后、
        # `AfterSideTurnStart` **之前**的能力时机。`InfernoPower` / `CrimsonMantlePower`
        # 的"每回合开始掉血"在这里 —— 放到 `on_turn_start` 之后会让中毒之类的
        # 回合开始效果先于自伤结算，顺序就与真机不同。
        # `EntropyPower`（每回合挑 `Amount` 张手牌随机转化）也挂在这里。
        ("能力：AfterPlayerTurnStart",
         lambda: power_rules.on_after_player_turn_start(state, events)),
        ("能力：AfterSideTurnStart（拥有者视角）",
         lambda: power_rules.on_turn_start(state.player, state, events)),
        ("遗物：AfterSideTurnStart",
         lambda: hooks.fire("relic_side_turn_start", state, events,
                            relic_rules.dispatch)),
        # `AfterSideTurnStart` 的**敌方订阅者**：拥有者是敌人、触发条件却是
        # "玩家阵营回合开始"（`RampartPower`：炮台在玩家回合开始时拿格挡）。
        ("能力：AfterSideTurnStart（敌方订阅者）",
         lambda: power_rules.on_player_turn_start(state, events)),
        # `AfterTurnStartOrbTrigger`：等离子在**回合开始**给能量（不是回合结束）。
        ("充能球：回合开始", lambda: orb_rules.on_turn_start(state, events)),
        # ⭐ ``Hook.AfterAutoPrePlayPhaseEntered``（``CombatManager.cs:867``）：真机的阶段是
        # ``Start → AutoPrePlay → Play → AutoPostPlay``，这个时机在**抽完牌、
        # 即将进入出牌阶段**那一刻（`MayhemPower` 在这里自动打出抽牌堆顶的牌）。
        # ⚠️ 与回合结束时那个 `on_auto_post_play` 差着整个出牌阶段，不能互相顶替。
        ("能力：AutoPrePlay", lambda: power_rules.on_auto_pre_play(state, events)),
    )
    for index in range(start_step, len(steps)):
        steps[index][1]()
        if state.pending is not None:
            # 这一步里的能力开了选择：记下"下一步是第几"，由
            # `_resume_after_selection` 在选完之后接着跑。
            state.turn_start_step = index + 1
            return
    state.turn_start_step = 0


def _hoist_innate(state: CombatState, draw_count: int,
                  events: list[str]) -> int:
    """第 1 回合把 ``Innate`` 牌移到抽牌堆顶，并保证起手抽牌数不少于其张数。

    出处：``CombatManager.SetupPlayerTurn``（``CombatManager.cs:908-923``）::

        if (player.PlayerCombatState.TurnNumber == 1) {
            // 附魔 `ShouldStartAtBottomOfDrawPile` 的牌先压到底（引擎未实现附魔，跳过）
            list2 = pile.Cards.Where(c => c.Keywords.Contains(Innate)).Except(list);
            foreach (item in list2) pile.MoveToTopInternal(item);
            handDraw = Math.Max(handDraw, list2.Count);
            handDraw = Math.Min(handDraw, CardPile.MaxCardsInHand);
        }

⚠️ 方向：本引擎的抽牌堆**末尾就是堆顶**（抽牌用 ``pop()``），所以"移到堆顶"
是 ``append``。多张固有牌之间的相对顺序真机不保证（``MoveToTop`` 逐个调用），
这里按原顺序 append，属于**不改数值的实现自由度**。
    """
    from . import keywords as keyword_rules
    innate = [c for c in state.draw_pile if keyword_rules.is_innate(c.definition())]
    if not innate:
        return draw_count
    for card in innate:
        state.draw_pile.remove(card)
        state.draw_pile.append(card)
    events.append(f"固有点：{len(innate)} 张移到抽牌堆顶")
    return max(draw_count, len(innate))


def combat_deck(deck: Sequence[object]) -> list[CardInstance]:
    """把 Run 的**永久牌组**变成战斗副本，并保留 :attr:`CardInstance.link_uid`。

    ``docs/13`` §3.2「入场：克隆完整实例并保留关联」。审计 F01 的根因就在这一步
    缺失：``_enter_node()`` 只传了 ``card.cid``，于是升级状态被丢掉、
    ``start_combat`` 又从字符串重建了默认未升级实例。

    ``str`` 条目仍然接受（测试与旧调用点），建成未升级实例且无关联。
    """
    out: list[CardInstance] = []
    for item in deck:
        if isinstance(item, CardInstance):
            out.append(CardInstance(cid=item.cid, upgraded=item.upgraded,
                                    link_uid=item.uid))
        else:
            out.append(CardInstance(cid=str(item), link_uid=None))
    return out


def mark_permanent_upgrade(state: CombatState, card: CardInstance) -> bool:
    """在战斗内把 ``card`` 标成**永久**升级（回写到 Run 的永久牌组）。

    只对**克隆自永久牌组**的牌有效（``link_uid is not None``）；战斗内生成的牌
    （``add_card``）没有永久本体，返回 ``False`` —— 这不是失败，而是语义：
    生成的牌随战斗消失，没有东西可回写。

    出处：``Player.PopulateCombatState`` 克隆实际卡牌并保留 ``DeckVersion`` 关联，
    战斗结束由 ``RunManager`` 按关联回写（``docs/12`` F01 的源码依据）。
    """
    if card.link_uid is None:
        return False
    card.upgraded = True
    state.permanent_card_changes[card.link_uid] = {"op": "upgrade"}
    return True


def apply_permanent_changes(state: CombatState,
                            deck: Sequence[CardInstance]) -> list[int]:
    """把战斗内记录的**永久改牌**回写到 Run 的永久牌组，返回受影响的 uid。

    ``docs/13`` §3.2 的「仅明确的永久改牌回写」：这里就是"明确"的那一份清单。
    """
    touched: list[int] = []
    by_uid = {card.uid: card for card in deck}
    for uid, change in state.permanent_card_changes.items():
        card = by_uid.get(uid)
        if card is None:
            continue                      # 牌已不在牌组（被事件删掉了）→ 忽略
        if change.get("op") == "upgrade" and not card.upgraded:
            card.upgraded = True
            touched.append(uid)
    return touched


@dataclass(frozen=True)
class CombatInit:
    """一场战斗的**显式上下文**。

    为什么要有这个类型：Run 与训练（``SpireEnv``）**各自**拼一遍
    ``start_combat`` 的参数表，结果就是"声明了角色、却没传下去"这类静默错误 ——
    实测 ``CombatPool(character="defect")`` 里跑出来的 ``CombatState.character``
    是 ``ironclad``，于是"生成一张能力牌"按**铁甲战士**的卡池取牌
    （``CreativeAiPower`` 走 ``Character.CardPool``），而报告里一切正常。

    所以把上下文收成**一份定义**：两条入口都构造 :class:`CombatInit`，
    再用 :meth:`as_kwargs` 喂给 :func:`start_combat`。新增一个"战斗有关的上下文"
    时只有这里一处要改，漏传就会表现为**这个 dataclass 里没有那个字段**，
    而不是某个入口悄悄用了默认值。

    ⚠️ 只放"决定战斗怎么开始"的字段。**课程覆盖值**（``player_hp`` 起始血量、
    随机牌组这类为训练而降级的东西）由调用方单独持有并写进训练清单 ——
    它们是**声明过的课程变量**，不是"忘了传所以退化成默认值"。
    """

    #: 角色（决定生成牌时的卡池，`Character.CardPool`）。
    character: str = "ironclad"
    #: 进阶（怪物血量/伤害的进阶修正；不传 = 打 A0 的怪，见审计 F01）。
    ascension: int = 0
    #: 房间种类（`sling_of_courage` 只在精英战给力量、`pantograph` 只在 Boss 战回血）。
    room: str = "monster"
    #: 永久遗物（战斗内钩子按它分发）。
    relics: tuple[str, ...] = ()
    #: 药水槽（Run 侧列表的副本；战斗里用掉的由 `_sync_player_from_combat` 读回）。
    potions: tuple[str | None, ...] = ()
    #: 每回合能量上限。
    max_energy: int = BASE_ENERGY
    #: 永久最大生命（``None`` = 用 :data:`PLAYER_MAX_HP`）。
    player_max_hp: int | None = None
    #: 进阶的怪物血量倍率。
    ascension_hp_mult: float = 1.0

    def as_kwargs(self) -> dict:
        """:func:`start_combat` 的关键字实参（``player_hp`` 与 ``deck`` 除外）。"""
        return {
            "character": self.character,
            "ascension": self.ascension,
            "room": self.room,
            "relics": self.relics,
            "potions": self.potions,
            "max_energy": self.max_energy,
            "player_max_hp": self.player_max_hp,
            "ascension_hp_mult": self.ascension_hp_mult,
        }


def start_combat(deck: Sequence[object], enemy_ids: Sequence[str], seed: int,
                 player_hp: int = PLAYER_MAX_HP,
                 relics: Sequence[str] = (),
                 ascension_hp_mult: float = 1.0,
                 ascension: int = 0,
                 room: str = "monster",
                 player_max_hp: int | None = None,
                 potions: Sequence[str | None] = (),
                 max_energy: int = BASE_ENERGY,
                 character: str = "ironclad") -> CombatState:
    """开一场战斗。

    ``deck`` 可以是卡 id 字符串，也可以是 Run 层**永久牌组的完整实例**
    （推荐）—— 后者会保留升级状态与永久关联（审计 F01）。

    ``player_max_hp`` 必须由调用方给出永久上限。审计 F01 的实测症状是
    "战斗当前生命 105 > 战斗最大生命 80"：上限被写死成 ``PLAYER_MAX_HP``，
    而当前生命是按 Run 传进来的。
    """
    rng = RngSet(seed)
    cards = combat_deck(deck)
    rng.shuffle(cards, "shuffle")

    enemies: list[EnemyState] = []
    for index, eid in enumerate(enemy_ids):
        definition = ENEMY_DB[eid]
        lo, hi = definition.hp
        hp = int(rng.randint(lo, hi, "up_front") * ascension_hp_mult)
        # ⭐ 进阶加血：真机写 `GetValueIfAscension(ToughEnemies, 76, 72)`，
        # 即 **A8 起怪物血量提高**。用提取出来的精确值，不用百分比估算。
        ai_record = getattr(definition, "ai_record", None) or {}
        hp_range = ai_record.get("hp_range")
        if hp_range and ascension >= hp_range.get("level", 99):
            lo = hi = hp_range["ascension"]
            hp = lo
        machine = build_machine(ai_record or None)
        enemy = EnemyState(name=f"{definition.name}#{index}", hp=hp, max_hp=hp,
                           eid=eid, machine=machine)
        init_enemy_ai(enemy, index)
        # ⭐ 固有属性：真机在这只怪入场时施加（``AfterAddedToRoom``）。
        # 少了这一步，46 只怪（40%）开局就拿不到自己的固有属性 ——
        # `aeonglass` 的 `artifact 3` 是它整场战斗的核心，不给等于换了一只怪。
        for pid, amount in definition.innate_powers:
            if amount:
                enemy.add_power(pid, amount)
        enemies.append(enemy)

    max_hp = PLAYER_MAX_HP if player_max_hp is None else int(player_max_hp)
    state = CombatState(
        player=Combatant("player", player_hp, max_hp),
        enemies=enemies,
        hand=[], draw_pile=cards, discard=[], exhaust=[], play_area=[],
        hidden=Hidden(rng),
        # ⭐ 角色：能力生成牌时要按它选卡池（`Character.CardPool`）。
        character=str(character),
        ascension=ascension,
        relics=tuple(str(r).lower() for r in relics),
        room=str(room).lower(),
        # ⭐ 药水槽是**共用库存**：传进来的是 Run 侧列表的副本，
        # 战斗里用掉之后由 Run 侧通过 `_sync_player_from_combat` 读回。
        potions=list(potions),
        max_energy=int(max_energy),
        energy=int(max_energy),
    )
    # ⭐ 反向引用（真机 `Creature.CombatState`）：`add_power` 靠它发
    # `AfterPowerAmountChanged`。必须在**任何施加上能力之前**设好。
    state.player.state = state
    for enemy in state.enemies:
        enemy.state = state
    events: list[str] = []
    for enemy in state.enemies:
        _choose_intent(state, enemy, events)
    # ⚠️ 开局遗物必须在**第 1 回合开始之前**施加（真机 `Hook.BeforeCombatStart`
    # 在 `StartTurn` 之前，`CombatManager.cs:594`）。格挡能留到第 1 回合，
    # 靠的是 `Creature.AfterTurnStart` 对玩家第 1 回合**不清格挡**，不是靠延后施加。
    from . import hooks
    from . import relics as relic_rules
    hooks.fire("relic_combat_start", state, events, relic_rules.dispatch)
    start_player_turn(state, events)
    state.log.extend(events)
    return state


def _apply_start_of_combat_relics(state: CombatState, relics: Sequence[str],
                                  events: list[str]) -> None:
    """兼容旧接口：等价于在"战斗开始"时机施加遗物。

    ⚠️ 实现已搬到 :mod:`sts2_sim.relics`，**数据驱动**（``relics_source.json``
    里 299 个遗物的钩子）。旧版在这里硬编码了 ``vajra`` / ``anchor`` /
    ``bag_of_preparation`` 三个 id —— 增删一个遗物就得改引擎代码，而且漏掉的
    遗物不会报错，只会静默不生效。
    """
    from . import hooks
    from . import relics as relic_rules
    hooks.fire("relic_combat_start", state, events, relic_rules.dispatch)


# ==========================================================================
# step
# ==========================================================================
@dataclass
class StepResult:
    events: list[str]
    reward: float
    done: bool
    won: bool


def step(state: CombatState, action: Action) -> StepResult:
    """执行一个动作。纯函数：同样的 (state, action) 必然得到同样的结果。"""
    if state.finished():
        raise RuntimeError("战斗已结束")

    events: list[str] = []

    if action.kind == "end_turn":
        from . import powers as power_rules
        # ⭐ `Hook.AfterAutoPostPlayPhaseEntered`（``CombatManager.cs:1545``）：
        # 真机在 `EndPlayerTurnPhaseOneInternal` 的**最开头**分发它 ——
        # 那时手牌还完整（清手牌在后面）。`StampedePower` 就在这里随机自动打出
        # 一张攻击牌。放到清手牌之后就永远打不出东西了。
        power_rules.on_auto_post_play(state, events)
        # ⭐ **顺序照抄真机**：``CombatManager.DoTurnEnd``（CombatManager.cs:1602-1628）
        #
        #   1. ``OrbQueue.BeforeTurnEnd``            ← 球的被动**先**结算
        #   2. 手里带 ``OnTurnEndInHand`` 的牌逐张结算（真机走 ``DoTurnEndCards``）
        #   3. 这些牌结算完 → 带 Ethereal 的进消耗堆，其余进弃牌堆
        #   4. 没被 3 处理掉的 Ethereal 牌（无回合结束效果的）→ ``CardCmd.Exhaust``
        #   5. 之后才是 ``EndPlayerTurnPhaseTwo``：``FlushPlayerHand`` 弃掉**没保留**的手牌
        #
        # 早先这里把球和手牌触发器的顺序写反了（先手牌后球），
        # 且"回合结束在手里"的牌被无条件弃掉 —— 后者会让 ``burn`` 这类牌
        # 结算完还留在手里，与真机不符。
        from . import orbs as orb_rules
        orb_rules.on_turn_end(state, events)
        # ⭐ **手牌触发器要在弃手牌之前跑**：真机的 `OnTurnEndInHand` 语义就是
        # "回合结束时若这张牌还在手里"。`burn` / `decay` / `infection` 这些
        # 敌人塞给你的状态牌全靠它 —— 不跑的话它们毫无代价，
        # 模型会学到"囤着不管"，而真机里是要掉血的。
        _run_hand_triggers(state, events)
        # `RetainHandPower.ShouldFlush` → false：回合结束**不清空**手牌。
        # 判断落到**每张牌**上（``FlushPlayerHand``：``!flag || card.ShouldRetainThisTurn``），
        # 因为 ``Retain`` 关键字是逐张的（实测 12 张牌带它）。
        retains_all = power_rules.retains_hand(state.player)
        from . import keywords as keyword_rules
        flushed: list[CardInstance] = []
        retained: list[CardInstance] = []
        for card in state.hand:
            if retains_all or keyword_rules.retains(card):
                retained.append(card)
            else:
                flushed.append(card)
        # ``CardPileCmd.Add(cardsToFlush, PileType.Discard)`` —— **不走 CardCmd.Discard**，
        # 所以这里**不触发 Sly**（真机 Sly 只在 `CardCmd.DiscardAndDraw` 里自动打出）。
        state.hand = retained
        if flushed:
            events.append(f"回合结束弃掉 {len(flushed)} 张手牌")
        state.discard.extend(flushed)
        # ⭐ ``PlayerCombatState.EndOfTurnCleanup``（``CombatManager.cs:1813``，
        # 紧跟 ``FlushPlayerHand`` 之后；敌人侧还有一处 ``:1697``）：
        # 遍历玩家的**所有**牌（手牌 / 抽牌堆 / 弃牌堆 / 消耗堆）清掉
        # "本回合有效"的临时费用修饰。只清手牌是不够的 —— ``SetToFreeThisTurn``
        # 可以作用在抽牌堆里的牌上（`Eidolon` 那类"直到打出"），漏掉就让
        # 免费跨回合留着，也就是**静默变强**。
        for pile in (state.hand, state.draw_pile, state.discard, state.exhaust):
            for pile_card in pile:
                pile_card.free_this_turn = False
                # ``CardModel.EndOfTurnCleanup``（CardModel.cs:1611-1624）同处还清
                # ``ExhaustOnNextPlay`` —— 它是**回合末**清，**不是**打出后清。
                # ⚠️ 第一版写在"打出后清"，于是 `Havoc` 的强制消耗永远不生效
                # （标志在判断之前就被抹掉了），而日志一切正常。
                pile_card.exhaust_on_next_play = False
        if power_rules.dies_to_doom(state.player):
            state.player.hp = 0
            events.append(f"{state.player.name} 被末日处决")
        tick_powers(state.player, state, events)
        # ⭐ ``Hook.BeforeSideTurnEnd`` 的玩家侧（``HailstormPower`` 在这时结算）。
        power_rules.on_before_side_turn_end(state, events, "player")
        # ⭐ `Hook.AfterSideTurnEnd` 的**全体**视角（``side="player"``）：玩家自己这边
        # 回合结束时，双方模型都要收到通知（`FlameBarrierPower` 靠它区分
        # "是自己这边结束（保留）还是对方那边结束（移除）"）。
        power_rules.on_side_turn_end(state, events, "player")
        # ⭐ ``AfterSideTurnEndLate``：真机在**同一个** ``Hook.AfterSideTurnEnd`` 里
        # 跑完第一遍再跑这一遍（`Hook.cs`），所以必须紧跟第一遍、且在别的事情之前。
        # `DisintegrationPower` 的"回合末自伤"挂在这里。
        power_rules.on_side_turn_end_late(state, events, "player")
        if not state.player.alive():
            state.phase = "lost"
            state.log.extend(events)
            return StepResult(events, -1.0, True, False)
        finished = _run_enemy_turn(state, events)
        if not finished:
            # ⭐ **挂在怪物出招的玩家选择上**（S02）：这一步到此为止，等 `select_card`。
            # ⚠️ 这里**不能**推进回合、也不能让 `state.turn` 加一 ——
            # 推进只在 `_end_enemy_phase` 里发生（续跑完才会走到）。
            state.log.extend(events)
            return StepResult(events, 0.0, False, False)
        if state.phase == "lost":
            state.log.extend(events)
            return StepResult(events, -1.0, True, False)
        _end_enemy_phase(state, events)
    elif action.kind == "play_card":
        card = state.hand[action.hand_index]
        definition = card.definition()
        # ⭐ 非法动作的判据与 `legal_actions` **同一个查询**：两处各写一份必然漂移，
        # 而漂移的表现是"合法动作列表里有它、真去执行却抛错"（或反过来）。
        if not card_playable(state, card):
            raise ValueError(f"非法动作：{definition.name} 不可打出"
                             f"（费用 {play_cost(state, card)}、能量 {state.energy}、"
                             f"星 {star_cost(state, card)}/{state.stars}）")
        if card_target(state, card) == "enemy" and not (
            0 <= action.target < len(state.enemies) and state.enemies[action.target].alive()
        ):
            raise ValueError("非法动作：目标无效")
        cost = play_cost(state, card)
        # ⭐ X 费卡的 X = **打出时消耗的能量**（真机 `SpendEnergy` 把它存进
        # `EnergyCost.CapturedXValue`，效果再通过 `ResolveEnergyXValue()` 读）。
        # 必须在**扣费之前**取 —— 扣费后 energy 已经是剩余值了。
        x_value = state.energy if card.definition().is_x_cost else 0
        state.energy -= cost
        # ⭐ ``CardPlay.Resources.EnergyValue``：这一张实际花了多少能量
        # （`FeralPower` / `OneForAllPower` 的唯一判据）。放在出牌流程最前面，
        # `_finish_played_card` 里再置回 -1（"不在出牌中"）。
        state.current_play_energy_spent = cost
        if cost == 0 and definition.card_type == "attack":
            # `FeralPower.AfterApplied` 初始化计数要用它（见字段说明）。
            state.zero_cost_attacks_played_this_turn += 1
        # ⭐ ``Hook.AfterEnergySpent``（``Hook.cs`` 的 ``AfterEnergySpent(CardModel, int)``）：
        # **卡牌**的能量费用刚扣掉就分发。`OrbitPower` 靠它累计"一共花了多少能量"。
        # ⚠️ 时机是"已经扣完"（源码在 `SpendEnergy` 之后），X 费卡这里 amount = 全部能量。
        from . import powers as power_rules
        power_rules.on_energy_spent(state, card, cost, events)
        # ⭐ **星费**（``CardModel.SpendStars``）：真机在能量**之后**扣，
        # 而且**只有真的花了星才分发** ``Hook.AfterStarsSpent``
        # （``if (amount > 0) { LoseStars(amount); await Hook.AfterStarsSpent(...); }``）
        # —— `ChildOfTheStarsPower` 就挂在这个钩子上（"花星就换格挡"）。
        stars = star_cost(state, card)
        state.current_play_stars_spent = stars
        if stars > 0:
            state.stars = max(0, state.stars - stars)
            state.x_stars_value = stars
            power_rules.on_stars_spent(state, stars, events)
        state.x_value = x_value
        events.append(f"打出 {definition.name}"
                      + (f" → {state.enemies[action.target].name}" if action.target >= 0 else ""))
        # ⭐ ①–④ 全部交给**统一执行器**（`play_card`）：落点 → 打出次数 →
        # 登记开始结算 → 每一次打出分发 Before/AfterCardPlayed → 按落点入堆。
        # 手动 / 自动 / 狡诈三条入口共用它，所以"某条路漏了哪一步"在结构上不可能
        # （外部复核 R1：自动打出原先漏了打出次数与挂起处理）。
        # ⚠️ 返回 False = 效果链挂在**选牌**上：剩下的事由 `_resolve_selection`
        # 续跑，此刻**不许**做任何"这张牌打完了"的动作（旧实现在这里立刻触发
        # `AfterCardPlayed` 并收尾入堆，导致同一 uid 出现在两个牌堆里）。
        play_card(state, card, events, target_index=action.target,
                  x_value=x_value, energy_spent=cost, stars_spent=stars,
                  from_hand=True)
    elif action.kind == "select_card":
        _resolve_selection(state, action.hand_index, events)
        # ⭐ **S02 续跑调度**：这次选择可能挂在敌方回合（怪物出招）或抽牌中途
        # （洗牌触发 `Stratagem`）上 —— 由 `_resume_after_selection` 按帧分派。
        _resume_after_selection(state, events)
    elif action.kind == "use_potion":
        use_potion_at(state, action.slot, action.target, events)
    else:  # pragma: no cover
        raise ValueError(f"未知动作 {action.kind!r}")

    if not state.living_enemies():
        state.phase = "won"
    elif not state.player.alive():
        state.phase = "lost"

    state.log.extend(events)
    if state.phase == "won":
        return StepResult(events, 1.0, True, True)
    if state.phase == "lost":
        return StepResult(events, -1.0, True, False)
    return StepResult(events, 0.0, False, False)


def use_potion(state: CombatState, pid: str, target_index: int = -1,
               events: list[str] | None = None) -> bool:
    """使用一瓶药水（``PotionModel.OnUse``）。

    药水与卡牌共用同一套效果 DSL 与变量解析，所以这里把它们按卡牌的方式结算。
    目标由**声明的** ``target_type`` 决定，不看 ``target`` 形参：
    单人游戏里 ``any_player`` 就是自己、``any_enemy`` 就是敌人。

    返回是否结算成功。**效果残缺的药水直接拒绝**（`effects_incomplete`）——
    用一瓶"少了一半效果"的药水等于给模型一个错误的强度，比不让用危险得多。
    """
    from .content import POTIONS
    events = [] if events is None else events
    potion = POTIONS.get(pid)
    if potion is None:
        raise KeyError(f"未知药水 {pid!r}")
    if potion.effects_incomplete:
        raise ValueError(f"药水 {pid!r} 的效果残缺，不能使用（会静默变强/变弱）")
    if potion.usage == "automatic":
        raise ValueError(f"药水 {pid!r} 是自动触发的，不能主动使用")
    events.append(f"使用药水 {potion.name}")
    target = target_index if "enemy" in potion.target_type else -1
    _apply_effects(state, potion.effects, state.player, target, events)
    return True


def use_potion_at(state: CombatState, slot: int, target_index: int = -1,
                  events: list[str] | None = None) -> str | None:
    """使用第 ``slot`` 号槽位的药水，并**清空该槽位**（``PotionCmd.Discard``）。

    与 :func:`use_potion` 的区别：那个按 ``pid`` 结算、不清库存（供测试与
    效果复用）；这个才是**动作入口**，所以它负责库存生命周期 ——
    审计 F09 的一条就是"药水有可调用效果函数，但没有走进动作空间"。
    """
    if not 0 <= slot < len(state.potions):
        raise ValueError(f"非法动作：药水槽位 {slot} 越界")
    pid = state.potions[slot]
    if not pid:
        raise ValueError(f"非法动作：槽位 {slot} 没有药水")
    log: list[str] = [] if events is None else events
    use_potion(state, pid, target_index, log)
    state.potions[slot] = None
    return pid


def _exhaust_card(state: CombatState, card: CardInstance, events: list[str],
                  caused_by_ethereal: bool = False) -> None:
    """``CardCmd.Exhaust`` 的**唯一入口**：把牌放进消耗堆并分发 ``AfterCardExhausted``。

    ⚠️ 为什么要集中：真机 ``Hook.AfterCardExhausted`` 只有**一个**调用者
    （``CardCmd.Exhaust``，``CardCmd.cs:246``），所以"什么算被消耗"是明确的；
    引擎里进消耗堆的地方有四条（打出的牌带 Exhaust / ``Corruption`` 让技能牌消耗 /
    回合结束的虚无牌 / 选牌效果消耗）。任何一处漏掉通知，
    ``FeelNoPainPower``（无痛）与 ``DarkEmbracePower``（黑暗拥抱）就**静默少算**。

    ``caused_by_ethereal`` 对应真机的同名参数（``CardCmd.cs:232-235``）：
    **只有** ``CombatManager`` 结算回合结束牌的那两处会传 ``true``，
    其余一律 ``false``。``DarkEmbracePower`` 靠它区分"立刻抽"与"攒到回合末抽"。

    ⚠️ 只有 ``CardCmd.Exhaust`` 会触发这个钩子；``CardPileCmd.Add(cards, Exhaust)``
    （把一批牌**移动**进消耗堆）**不会** —— 所以 ``move_all_matching`` 那条路径
    不在这里，引擎里也没有走到消耗堆的 ``move_all_matching``。
    """
    if card not in state.exhaust:
        state.exhaust.append(card)
    from . import powers as power_rules
    power_rules.on_card_exhausted(state.player, state, events, card,
                                  caused_by_ethereal)


def _run_hand_triggers(state: CombatState, events: list[str]) -> None:
    """跑手里每张牌的 ``OnTurnEndInHand`` 触发器，并按真机落牌堆。

    按**手牌顺序**逐张结算（真机也是逐张）。触发器里的 ``self`` 指**玩家**，
    因为"在手里烧你"的受害者是持牌者。

    ⚠️ 真机在 ``CombatManager.ResolveTurnEndCardEffects``（CombatManager.cs:1671-1676）
    里明确写了两件事：

    * 触发器的结算发生在牌被 ``CardPileCmd.Add(card, PileType.Play)``
      移进**出牌区**之后（所以触发器里的"这张牌"已经在出牌区）；
    * 结算完立刻决定去向：**带 Ethereal → ``CardCmd.Exhaust``**，否则进弃牌堆。

    早先引擎只跑触发器、不管去向，于是结算完的牌继续留在手里，
    下一回合又烧一次 —— ``burn`` / ``decay`` 的代价被放大成"每回合都触发"。
    """
    from . import keywords as keyword_rules
    for card in list(state.hand):
        if not state.player.alive():
            return
        triggers = [(hook, effects) for hook, effects in card.definition().triggers
                    if hook == "on_turn_end_in_hand"]
        # ⚠️ 真机的分类是 **if / else if**（``CombatManager.cs:1612-1622``）：
        # 有回合结束效果的牌进 turnEndCards 分支，**没有**回合结束效果但带
        # Ethereal 的牌才进"直接消耗"分支。两者都不沾的牌**什么都不做**，
        # 留在手里等后面的 ``FlushPlayerHand`` 按 Retain 决定去留 ——
        # 早先这里对每张牌都做落堆处理，等于把所有手牌提前弃掉，
        # `Retain`（含 `RetainHandPower`）就完全失效了。
        if triggers:
            for _hook, effects in triggers:
                events.append(f"{card.definition().name} 在手牌中触发")
                _apply_effects(state, effects, state.player, -1, events)
            state.hand.remove(card)
            # `ResolveTurnEndCardEffects`：带 Ethereal → `CardCmd.Exhaust(causedByEthereal: true)`，
            # 否则进弃牌堆。
            if keyword_rules.is_ethereal(card.definition()):
                _exhaust_card(state, card, events, caused_by_ethereal=True)
                events.append(f"{card.definition().name} 结算后消耗（虚无）")
            else:
                state.discard.append(card)
        elif keyword_rules.is_ethereal(card.definition()):
            state.hand.remove(card)
            # `DoTurnEnd` 里"没有回合结束效果、但带 Ethereal"的牌：
            # `CardCmd.Exhaust(..., causedByEthereal: true)`（``CombatManager.cs:1625``）。
            _exhaust_card(state, card, events, caused_by_ethereal=True)
            events.append(f"{card.definition().name} 虚无 → 消耗")
        # 其余牌留在手里，交给 FlushPlayerHand。


def _enter_play_area(state: CombatState, card: CardInstance, *,
                     from_hand: bool) -> None:
    """把打出的牌放进**出牌区**（真机 ``PileType.Play``）。

    三条入口的差别只有"牌从哪来"：手动与通用自动打出从**手牌**取；
    狡诈（``SlyDiscard``）的牌已经被 ``CardCmd.DiscardAndDraw`` 放进**弃牌堆**，
    再被自动打出 —— 那时要先把它从弃牌堆拿出来，否则它会**同时**待在
    弃牌堆与出牌区（一份牌实例出现在两个牌堆里）。
    """
    if from_hand:
        if card in state.hand:
            state.hand.remove(card)
    elif card in state.discard:
        state.discard.remove(card)
    if card not in state.play_area:
        state.play_area.append(card)


def play_card(state: CombatState, card: CardInstance, events: list[str], *,
              target_index: int, x_value: int, energy_spent: int,
              stars_spent: int = 0, from_hand: bool = True) -> bool:
    """**统一**的出牌执行器：手动 / 通用自动打出 / 狡诈自动打出都在这里汇合。

    真机只有一条路径 —— ``CardCmd.AutoPlay``（``CardCmd.cs:51-131``）与玩家出牌
    最终都调 ``CardModel.OnPlayWrapper``（``CardModel.cs:1882-1997``）::

        resultLocation = GetResultLocationForCardPlay()      // ① 先算落点
        playCount      = GeneratePlayCount()                 // ② 再算打出次数
        History.CardPlayStarted(cardPlay)                    // ③ 才登记"开始结算"
        for (i < playCount) {                                // ④ 每一次打出：
            Hook.BeforeCardPlayed(cardPlay);                 //    分发前置钩子
            await OnPlay(...);                               //    结算效果
            Hook.AfterCardPlayed(cardPlay);                  //    分发后置钩子
        }
        …按落点入堆（Exhaust / Discard / Draw / Hand）

    引擎里这三条入口原先**各写了一份**，于是各自漏掉不同的东西 ——
    实测（外部复核 R1）通用自动打出漏了 ② 与"挂起时不许收尾"：

    * ``OneTwoPunchPower``（"攻击牌多打一次"）对手动出牌生效、对自动打出**不生效**
      （基础打击手动 12 点、自动只有 6 点）；
    * 自动打出遇到**选牌**时会立刻 ``AfterCardPlayed`` + 收尾入堆，
      而效果链其实还挂在选牌上 —— 结果是"选牌还没选完，格挡已经拿到、
      牌已经进了弃牌堆，而且它自己出现在自己的选牌候选里"，
      续跑完成后同一个 ``uid`` 在牌堆里出现**两次**。

    所以这里把 ①–④ 收成一处，参数只剩"从哪来、打谁、花多少"：

    * ``energy_spent`` —— 手动出牌是实际费用；**自动打出恒 0**
      （真机 ``AutoPlay`` 不经 ``SpendEnergy``）。它同时是 ``CardPlay.Resources.
      EnergyValue``，`FeralPower` / `OneForAllPower` 按它判"这张是不是免费打的"。
    * ``stars_spent`` —— 手动出牌扣掉的星；自动打出恒 0。
    * ``from_hand`` —— 见 :func:`_enter_play_area`。

    返回 ``True`` = 这一次出牌**完整跑完**（牌已按落点入堆）；
    ``False`` = 效果链挂在**选牌**上，剩下的事情由 :func:`_resolve_selection`
    续跑（它会接着跑剩余的打出次数并负责收尾）。调用方**必须**在 ``False`` 时
    停下来，不能再做任何"这张牌打完了"的动作。
    """
    from . import powers as power_rules

    definition = card.definition()
    _enter_play_area(state, card, from_hand=from_hand)
    state.current_play_energy_spent = energy_spent
    state.current_play_stars_spent = stars_spent
    state.x_value = x_value
    if stars_spent > 0:
        state.x_stars_value = stars_spent
    if energy_spent == 0 and definition.card_type == "attack":
        # `FeralPower.AfterApplied` 初始化计数要用它（见字段说明）。
        state.zero_cost_attacks_played_this_turn += 1
    # ① 落点快照：必须在 `GeneratePlayCount` 与 `History.CardPlayStarted` **之前**
    #    （`NostalgiaPower` 判据是"本回合已开始的攻击/技能牌数 < 层数"，
    #    把自己算进去会让第 Amount 张牌少改写一次）。
    result_location = _play_result_location(state, card)
    # ② 打出次数（`OneTwoPunchPower`）。③ 登记"开始结算"。
    plays = power_rules.card_play_count(state, card)
    power_rules.on_card_play_count_modified(state, card)
    _note_card_play_started(state, card)
    # ④ 每一次打出都分发 Before/AfterCardPlayed（可能挂起）
    done = _run_card_plays(state, card, definition, target_index, events,
                           x_value, plays)
    if not done:
        if state.pending is not None:
            # 续跑时**不能重算**落点（那时计数已经含当前这张牌）。
            state.pending.result_location = result_location
        return False
    _finish_played_card(state, card, definition, events, result_location)
    return True


def _autoplay_sly(state: CombatState, card: CardInstance,
                  events: list[str]) -> bool:
    """被弃掉的 ``Sly`` 牌自动打出一次（``AutoPlayType.SlyDiscard``）。

    出处：``CardCmd.DiscardAndDraw``（``CardCmd.cs:184-204``）::

        foreach (card in discardCards) { add to discard; Hook.AfterCardDiscarded }
        if (cardsToDraw > 0) await CardPileCmd.Draw(…);
        foreach (item in slyCards) await AutoPlay(choiceContext, item, null, AutoPlayType.SlyDiscard);

    以及 ``CardCmd.AutoPlay``（``CardCmd.cs:51-84``）的三道门：

    1. 带 ``Unplayable`` → ``MoveToResultPileWithoutPlaying``（**不结算效果**）；
    2. ``Hook.ShouldPlay`` 否决 → 同上（引擎未实现该钩子的否决方，如实记为未实现）；
    3. ``TargetType.AnyEnemy`` 且没有活着的敌人 → 同样不结算。

    目标选择：真机 ``target = Rng.CombatTargets.NextItem(HittableEnemies)`` ——
    走 ``combat_targets`` 这条**独立随机流**（不是主随机）。这里照抄，
    否则同一种子下的随机序列会与真机分叉。
    """
    from . import keywords as keyword_rules
    definition = card.definition()
    # ⚠️ 看**实例**关键字：`MasterPlannerPower` 给打出的技能牌加 `Sly`，
    # 只看卡牌定义的话那些牌被弃掉时不会自动打出（静默失效）。
    if not keyword_rules.is_sly_card(card):
        return True
    events.append(f"{definition.name} 因狡诈被自动打出")
    if keyword_rules.is_unplayable(definition):
        events.append(f"{definition.name} 不可打出 → 只落牌堆，不结算")
        return True
    if not should_play_allows(state, card, auto=True):
        # 真机 ``CardCmd.AutoPlay`` 的第二道门：``Hook.ShouldPlay`` 否决 →
        # ``MoveToResultPileWithoutPlaying``（**只落牌堆、不结算**）。
        events.append(f"{definition.name} 被 ShouldPlay 否决 → 只落牌堆，不结算")
        return True
    if not definition.effects:
        return True
    target_index = -1
    if card_target(state, card) == "enemy":
        alive = [i for i, enemy in enumerate(state.enemies) if enemy.alive()]
        if not alive:
            events.append(f"{definition.name} 没有合法目标 → 不结算")
            return True
        # 目标走 `combat_targets` 这条**独立随机流**（不是主随机）。
        target_index = alive[state.hidden.rng.next_index("combat_targets", len(alive))]
    if not should_play_allows(state, card, auto=True):
        # 真机的第二道门（`CardCmd.AutoPlay`）：`Hook.ShouldPlay` 否决 →
        # 只落牌堆、不结算。`Normality` 不区分是否自动打出，所以这里必须判。
        events.append(f"{definition.name} 被 ShouldPlay 否决 → 只落牌堆，不结算")
        return True
    # ⭐ 与手动出牌**同一条**执行器：落点、次数、`Before/AfterCardPlayed`、
    # 挂起与收尾都在 `play_card` 里，这里只说明"从弃牌堆来、免费"。
    return play_card(
        state, card, events, target_index=target_index,
        x_value=state.energy if definition.is_x_cost else 0,
        energy_spent=0, stars_spent=0,
        # 狡诈牌先被 `CardCmd.DiscardAndDraw` 放进弃牌堆，再从那里被打出。
        from_hand=False)


def autoplay_card(state: CombatState, card: CardInstance,
                  events: list[str]) -> bool:
    """``CardCmd.AutoPlay``：**免费**自动打出一张牌（``CardCmd.cs:51-131``）。

    与 :func:`_autoplay_sly` 的区别：那个是"被弃掉的狡诈牌"这条特定路径
    （会检查 ``IsSlyThisTurn``），这个是给能力调用的通用入口
    （``StampedePower`` / ``HellraiserPower``）。

    真机的三道门照抄：

    1. 带 ``Unplayable`` → ``MoveToResultPileWithoutPlaying``（**不结算效果**，只落堆）；
    2. ``Hook.ShouldPlay`` 否决 → 同上（卡牌级 ``Normality`` / ``Enthralled``、
       能力级 ``SlothPower``，都走 :func:`should_play_allows`）；
    3. ``TargetType.AnyEnemy`` → 目标从 ``combat_targets`` 流随机；没有活着的敌人
       就同样只落堆。

    ⚠️ 自动打出**不付费用**（真机 ``AutoPlay`` 不经过 ``SpendEnergy``），
    也不检查"当前能量够不够"。X 费卡在这里取"当前能量"当 X
    （``CardModel.cs:99-102``：``CapturedXValue = playerCombatState.Energy``）。

    ⭐ 除了上面三道门，剩下的**全部**交给 :func:`play_card` —— 其中就包括
    ``GeneratePlayCount``（``OneTwoPunchPower`` 的"多打一次"）与"挂起时不许收尾"。
    这两条以前在自动打出这条路上是**整个漏掉**的（外部复核 R1 的两个反例）。
    返回 ``True`` = 跑完；``False`` = 挂在选牌上（调用方必须立刻停手）。
    """
    from . import keywords as keyword_rules
    definition = card.definition()
    if keyword_rules.is_unplayable(definition):
        events.append(f"{definition.name} 不可打出 → 只落牌堆，不结算")
        return True
    if not should_play_allows(state, card, auto=True):
        # 见 `_autoplay_sly` 里的同一条门（`CardCmd.AutoPlay` 的第二道门）。
        events.append(f"{definition.name} 被 ShouldPlay 否决 → 只落牌堆，不结算")
        return True
    if not definition.effects and not definition.upgrade:
        return True
    target_index = -1
    if card_target(state, card) == "enemy":
        alive = [i for i, enemy in enumerate(state.enemies) if enemy.alive()]
        if not alive:
            events.append(f"{definition.name} 没有合法目标 → 不结算")
            return True
        target_index = alive[state.hidden.rng.next_index("combat_targets", len(alive))]
    events.append(f"{definition.name} 被自动打出")
    return play_card(
        state, card, events, target_index=target_index,
        x_value=state.energy if definition.is_x_cost else 0,
        # 自动打出不花能量 → `Resources.EnergyValue == 0`
        # （`FeralPower` 据此把 0 费攻击牌送回手牌，与真机一致）。
        energy_spent=0, stars_spent=0, from_hand=True)


def _play_result_location(state: CombatState, card: CardInstance):
    """算这张牌打出后的**落点**（真机 ``CardModel.cs:1882``）。

    ⚠️ 调用时机是语义的一部分：真机在 ``GeneratePlayCount`` 与
    ``History.CardPlayStarted`` **之前**就算好落点，所以算落点时
    ``History.CardPlaysStarted`` 里**还没有当前这张牌** ——
    `NostalgiaPower` 的判据是"本回合已开始的攻击/技能牌数 < 层数"，
    把自己算进去会让第 Amount 张牌少改写一次。
    """
    from . import powers as power_rules
    return power_rules.card_play_result_location(state, card)


def _note_card_play_started(state: CombatState, card=None) -> None:
    """登记"这张牌开始结算了"（真机 ``History.CardPlayStarted``）。

    ⚠️ 调用时机是语义的一部分：必须在 `card_play_count` **之后**，
    因为真机是 playCount 算完、进循环时才把 ``CardPlay`` 登记进历史
    （``CardModel.cs``），``EchoFormPower`` 读到的计数因此**不含当前这张牌**。
    """
    state.cards_played_started_this_turn += 1
    if card is not None:
        card_type = getattr(card.definition(), "card_type", "")
        if card_type in ("attack", "skill"):
            state.attack_skill_plays_started_this_turn += 1


def _run_card_plays(state: CombatState, card: CardInstance, definition,
                    target_index: int, events: list[str], x_value: int,
                    plays: int) -> bool:
    """把这张牌结算 ``plays`` 次，返回是否**全部跑完**（``False`` = 挂在选牌上）。

    真机 ``CardModel.OnPlayWrapper`` 的结构就是一个 ``for (i < playCount)`` 循环，
    循环体依次做三件事（``CardModel.cs:1926-1970``）::

        Hook.BeforeCardPlayed(cardPlay)   ← 每一次打出都分发
        await OnPlay(...)                 ← 效果结算
        Hook.AfterCardPlayed(cardPlay)    ← 每一次打出都分发

    ``playCount`` 由 ``GeneratePlayCount`` 算出（``OneTwoPunchPower`` 给攻击牌 +1）。
    ⚠️ 两个钩子都是**每一次打出**都分发，不是"整张牌一次" ——
    `EnragePower`（愤怒）与 `RagePower`（暴怒）因此会对"多打一次"的攻击牌
    响应两次，这是真机行为。

    挂起时把"还剩几次"记进 ``state.pending.plays_remaining``，
    由 :func:`_resolve_selection` 续跑（否则"多打一次 + 选牌"会少打一次）。
    """
    effects = (definition.upgrade if card.upgraded and definition.upgrade
               else definition.effects)
    from . import powers as power_rules
    for index in range(max(1, plays)):
        _fire_before_card_played(state, definition, events, card)
        # ⭐ `card_target` 说"全体"时，把**伤害**效果的目标改成全体 ——
        # 出处：`Shiv.OnPlay` 读 `HasFanOfKnives`、`SovereignBlade.OnPlay` 读
        # `HasSeekingEdge`，两者都是"同一条 `DamageCmd.Attack` 换成
        # `TargetingAllOpponents`"。只改**这一次调用**的效果，
        # 不污染基础/升级定义，也不影响别的牌。
        play_effects = effects
        if card_target(state, card) == "all_enemies":
            play_effects = tuple(replace(e, target="all_enemies")
                                 if e.op == "damage" else e for e in effects)
        done = _apply_effects(state, play_effects, state.player, target_index, events,
                              pending_played=card, x_value=x_value)
        if not done:
            if state.pending is not None:
                state.pending.plays_remaining = max(1, plays) - index - 1
            return False
        # ⭐ 真机每个 playIndex 建**一条**攻击指令：`AfterAttack` 在这里结算
        # （`GigantificationPower` 递减层数）—— 必须逐条调用，见该函数文档。
        power_rules.on_attack_command_finished(state, card)
        _fire_after_card_played(state, definition, events, card,
                                is_last_in_series=(index == max(1, plays) - 1))
    return True


def _fire_before_card_played(state: CombatState, definition,
                             events: list[str], card=None) -> None:
    """``Hook.BeforeCardPlayed``（``CardModel.cs:1926``）的**唯一**触发入口。

    真机的调用点在 ``CardModel.OnPlayWrapper`` 的每一段出牌循环里 ——
    手动出牌与 ``AutoPlay`` 都要经过它，所以引擎里两个调用点分别落在
    ``step`` 与 :func:`_autoplay_sly` / :func:`autoplay_card`。

    ⚠️ 顺序：**先**把"本回合第几张攻击牌"记上（真机的 ``History.CardPlayStarted``
    紧跟在 ``BeforeCardPlayed`` 之后、效果之前），再分发钩子 —— 这样
    ``JugglingPower``（戏法）在钩子里读到的计数**包含当前这张牌**，
    与它源码里"先自增再比较 ``== 3``"完全一致。
    """
    from . import powers as power_rules
    if getattr(definition, "card_type", "") == "attack":
        state.attacks_played_this_turn += 1
    elif getattr(definition, "card_type", "") == "skill":
        # ⭐ 「本回合已打出的**技能**牌数」：`SmoggyPower.AfterCardEnteredCombat`
        # 的判据（``CardPlaysStarted`` 过滤 ``CardType.Skill``）。与攻击计数分开，
        # 因为"打过技能"和"打过攻击"在这一条上不能互换。
        state.skills_played_this_turn += 1
    power_rules.on_before_card_played(state, events, card)


def _relic_played_counts(state: CombatState) -> dict[str, int]:
    """本回合已打出的张数（``relics._applies`` 的 ``every_n_turn`` 守卫要用）。

    出处是**真机自己的计数器名**（见 ``tools.extract_relics.PER_TURN_COUNTERS``）：

    ========================== ==========================================
    源码里的私有计数器          引擎里同义的量
    ========================== ==========================================
    ``AttacksPlayedThisTurn``   ``state.attacks_played_this_turn``
    ``SkillsPlayedThisTurn``    ``state.skills_played_this_turn``
    ``CardsPlayedThisTurn``     ``state.cards_played_started_this_turn``
    ========================== ==========================================

    ⚠️ 三者都必须**包含当前这张牌**：真机是"先 ``Counter++`` 再判
    ``% N == 0``"，而引擎的三个计数器分别在 ``_fire_before_card_played`` 与
    ``_note_card_play_started`` 里、**早于** ``AfterCardPlayed`` 加过 ——
    所以这里直接读即可，不必再 +1（再加一次会让"第 3 张"变成"第 4 张"）。
    """
    return {
        "attack": state.attacks_played_this_turn,
        "skill": state.skills_played_this_turn,
        "card": state.cards_played_started_this_turn,
    }


def _fire_after_card_played(state: CombatState, definition,
                            events: list[str], card=None,
                            is_last_in_series: bool = True) -> None:
    """``Hook.AfterCardPlayed``（``Hook.cs:278``）的**唯一**触发入口。

    一个事件型钩子散在多个调用点时，"某条路径忘了通知"是静默的（``docs/11``
    §11.3 的结构性风险）。所以这里集中成一处，调用点只有：

    * ``step`` 的 ``play_card`` 分支（效果链一次跑完）
    * ``_resolve_selection``（挂起后续跑完成）
    * ``_autoplay_sly``（自动打出的牌**同样**触发 —— 旧实现整个漏了这条）
    * :func:`autoplay_card`（能力引起的自动打出）

    ``card`` 是打出的那张牌本体：``RupturePower.AfterCardPlayed`` 要按照
    "哪张牌"把它欠下的力量发下去（真机是 ``playedCards.Remove(cardPlay.Card, out value)``）。
    """
    from . import powers as power_rules
    # ⭐ ``History.CardPlaysFinished`` 按**标签**计数：`PhantomBladesPower` 数的是
    # "本回合**已经打完**的飞刀"（正在打的这一张还没登记，所以第一张吃加成、
    # 第二张不吃）。计数放在这里 —— 这是 `AfterCardPlayed` 的唯一入口，
    # 四个调用点都会经过，漏不掉。
    tags = tuple(getattr(definition, "tags", ()) or ())
    if tags:
        counter = state.tag_plays_finished_this_turn
        for tag in tags:
            counter[tag] = counter.get(tag, 0) + 1
    power_rules.on_card_played(state, events, definition.card_type, card,
                               is_last_in_series=is_last_in_series)
    # ⭐ 遗物走**同一条时机**（`relic_after_card_played`），并把**牌型**传下去：
    # `GamePiece`（仅能力牌 → 抽牌）/ `LostWisp` / `Permafrost` / `RainbowRing`
    # 的整个行为就是"只在某类牌上触发"；不传牌型，`relics._applies` 的
    # `card_type` 守卫会算不出而**跳过**（安全方向），那样这些遗物等于没接。
    #
    # ⭐ 还要传**本回合已打出的张数**（`Kunai` / `Shuriken` / `LetterOpener` /
    # `OrnamentalFan` = "每 N 张某类牌触发一次"）。两者都**包含当前这张牌**：
    # 真机是"先 `Counter++` 再判 `% N == 0`"，而三个计数器都在
    # `_fire_before_card_played` / `_note_card_play_started` 里已经加过。
    from . import hooks as hook_bus
    from . import relics as relic_rules
    hook_bus.fire("relic_after_card_played", state, events, relic_rules.dispatch,
                  card_type=str(getattr(definition, "card_type", "") or ""),
                  played_counts=_relic_played_counts(state))


def _finish_played_card(state: CombatState, card: CardInstance,
                        definition, events: list[str],
                        result_location=None) -> None:
    """效果全部跑完，把打出的牌落进正确的牌堆。

    真机是先算 ``GetResultLocationForCardPlay``（带 ``Exhaust`` 关键字 → 消耗堆），
    再让 ``Hook.ModifyCardPlayResultLocation`` 覆写（``CorruptionPower``：技能牌 → 消耗堆），
    最后按落点决定走 ``CardCmd.Exhaust`` 还是进弃牌堆（``CardModel.cs:1997``）。
    所以腐败生效时技能牌**走的是消耗**，``AfterCardExhausted`` 照样分发。

    ``result_location`` 是**打出之前**算好的落点快照（``_play_result_location``）；
    调用方必须传（`NostalgiaPower` 依赖"算落点时还没登记当前这张牌"的时机）。
    """
    from . import powers as power_rules
    # ⭐ ``CardModel.AfterCardPlayedCleanup``（``CardModel.cs:2007``）：这张牌打完之后
    # 清掉"打出前有效"的临时费用修饰（``SetToFreeThisTurn`` 的 WhenPlayed 那半）。
    # 放在函数**最前面**是为了覆盖下面"消耗牌提前 return"的分支 —— 真机的清理在
    # ``OnPlayWrapper`` 末尾，与落点无关；漏掉它，一张免费牌被多次回手
    # （``FeralPower`` 回手 / ``NostalgiaPower`` 回抽牌堆）后会**一直免费**。
    card.free_this_turn = False
    # ⭐ ``History.CardPlaysFinished`` 的**整场战斗**计数：运行期公式
    # （`GoldAxe` 的"伤害 = 本场打出的牌数"，`docs/12` §2.33）读的就是它。
    # 与 ``cards_played_started_this_turn`` 的区别：那个按**回合**清零，
    # 这个整场累加。少一个计数器，`GoldAxe` 只能算成本回合出牌数（静默算错）。
    state.card_plays_finished_this_combat = getattr(
        state, "card_plays_finished_this_combat", 0) + 1
    # ⭐ 出牌到此结束：`Resources.EnergyValue` 只在"这一次出牌里"有意义，
    # 立刻置回 -1（**在分支之前**，否则消耗牌那条提前 return 会漏掉它，
    # 让后面别的伤害读到上一次的费用）。
    state.current_play_energy_spent = -1
    if card in state.play_area:
        state.play_area.remove(card)
    if definition.exhaust or card.exhaust_on_next_play \
            or power_rules.corruption_applies_to(state, card):
        _exhaust_card(state, card, events)
        return
    # ⭐ ``Hook.ModifyCardPlayResultLocation``：默认落点是**弃牌堆**，由能力改写
    # （`ReboundPower` / `NostalgiaPower` 把它改成**抽牌堆顶**；`CorruptionPower`
    # 的技能 → 消耗堆已经在上面那个分支处理）。改完还要通知
    # ``Hook.AfterModifyingCardPlayResultLocation``（`ReboundPower` 靠它递减层数）。
    if result_location is None:
        # 兜底：直接调用本函数的用例/工具没有落点快照（正常出牌路径一定传）。
        result_location = power_rules.card_play_result_location(state, card)
    pile, modifiers = result_location
    if pile == "draw":
        state.draw_pile.append(card)              # 抽牌用 pop()，尾部就是顶
    elif pile == "hand":
        # ⭐ `FeralPower`：**0 费攻击牌回到手牌**（`location.pileType = Hand`）。
        # 走 `add_to_hand` 是为了照抄"手牌满时的落点规则"（`CardPileCmd.Add`）。
        add_to_hand(state, card, events)
    else:
        state.discard.append(card)
    power_rules.on_card_play_result_location(state, card, pile, modifiers)


def _resolve_selection(state: CombatState, index: int,
                       events: list[str]) -> None:
    """处理一次选牌，必要时续跑剩余效果。"""
    pending = state.pending
    if pending is None:
        raise ValueError("非法动作：当前没有等待选择")
    candidates = pending.candidates(state)
    if not (0 <= index < len(candidates)):
        raise ValueError(f"非法动作：候选下标 {index} 越界（共 {len(candidates)} 张）")
    chosen = candidates[index]

    if pending.purpose == "transform":
        # ⭐ `CardCmd.Transform`：**原位**换成随机的另一张。
        # 绝不能走下面那段"从来源牌堆取走" —— 那会先把牌移出手牌，位置就丢了
        # （真机是先 `RemoveFromCurrentPile()`、记下下标，替换完再插回原位）。
        transform_card_random(state, chosen, events)
    elif pending.purpose == "remember":
        # ⭐ **记住这张卡**（`Nightmare`）：真机只做 `CreateClone()` ——
        # 牌**留在手牌里**（既不移动也不消耗），所以绝不能走下面那段"从牌堆取走"。
        state.remembered_card = chosen.clone()
        events.append(f"记住 {chosen.definition().name}")
    elif pending.purpose == "choose":
        # ⭐ **显式候选**（`choose_one`，S02）：那张卡是 `CombatState.CreateCard`
        # 出来的临时实例，**不在任何牌堆里** —— 真机选完只 `await OnChosen()`，
        # 既不从牌堆取走、也不落进结果堆。移动它会让同一张卡凭空多出来。
        events.append(f"选择 {chosen.definition().name}")
    else:
        # 从来源牌堆取走
        for pile in (state.hand, state.discard, state.draw_pile):
            if chosen in pile:
                pile.remove(chosen)
                break
        if pending.purpose == "exhaust":
            # 选牌消耗走真机的 `CardCmd.Exhaust` → ``AfterCardExhausted`` 分发。
            _exhaust_card(state, chosen, events)
        elif pending.purpose == "discard":
            state.discard.append(chosen)
        elif pending.purpose == "to_hand":
            # ⚠️ 走 `CardPileCmd.Add` 的同一条规则：手牌满时改放弃牌堆。
            add_to_hand(state, chosen, events)
        elif pending.purpose == "to_draw":
            # `CardPilePosition.Top` → 放到抽牌堆**顶**（抽牌用 pop()，尾部就是顶）
            if pending.position == "top":
                state.draw_pile.append(chosen)
            else:
                state.draw_pile.insert(0, chosen)
        else:                                   # pragma: no cover - lint 已拦截
            raise ValueError(f"未实现的选牌用途 {pending.purpose!r}")
        events.append(f"选择 {chosen.definition().name}（{pending.purpose}）")
        if pending.purpose == "discard":
            # ⭐ ``CardKeyword.Sly``：真机 ``CardCmd.DiscardAndDraw``（CardCmd.cs:184-204）
            # 把牌**先**放进弃牌堆、触发 ``AfterCardDiscarded``，**然后**才对每张
            # ``IsSlyThisTurn`` 的牌执行 ``AutoPlay(…, AutoPlayType.SlyDiscard)``。
            # 顺序不能反：自动打出时这张牌已经在弃牌堆里了，所以
            # "不可打出 → ``MoveToResultPileWithoutPlaying``" 那一步对它是空操作
            # （那张方法只在牌位于出牌区时才移动，见 CardModel.cs:2103-2121）。
            _autoplay_sly(state, chosen, events)

    pending.remaining -= 1
    if pending.remaining > 0:
        # ⭐ 候选**中途抽干**（"弃 2 张"但手里只有 1 张）时不能继续挂起：
        # 真机的选择命令拿不到足够候选就按现有数量结束，不会卡住。
        # 继续挂起会让 `legal_actions()` 返回**空列表** ——
        # 环境既没有合法动作也不会自己往前走，训练直接报"没有合法动作的样本"。
        if pending.candidates(state):
            return
        events.append(f"候选不足，提前结束选择（还差 {pending.remaining} 张）")
    # 选够了（或候选已抽干）→ 续跑剩余效果（可能再次挂起）
    rest, target_index = pending.rest, pending.target_index
    played = pending.played_card
    x_value = pending.x_value
    plays_remaining = pending.plays_remaining
    # ⭐ 落点快照跟着走（不能在续跑时重算，见 `PendingSelection.result_location`）。
    result_location = pending.result_location
    # ⭐ `choose_one`：选中那张卡自己的效果（真机 `IChoosable.OnChosen`）排在
    # **招式剩余效果之前** —— 真机是同一次 await 之后立刻 OnChosen，再往下走。
    chosen_effects = (tuple(pending.choice_effects[index])
                      if pending.purpose == "choose"
                      and index < len(pending.choice_effects) else ())
    # ⭐ `source` 用挂起时记下的 `actor`：怪物出招的效果按**怪物视角**解释
    # （`target="enemy"` 对怪物而言是玩家）。用 `state.player` 会把效果打反，
    # 而且不会有任何报错。
    actor = pending.actor if pending.actor is not None else state.player
    remove_power = pending.remove_power_after
    state.pending = None
    # ⭐ **选完了才移除能力**：真机 `ForegoneConclusionPower.BeforeHandDraw` 的最后
    # 一句是 `await PowerCmd.Remove(this)` —— 在选牌之后。提前移除会让
    # "选牌界面还开着、能力已经没了"。
    if remove_power:
        _remove_power_whole(state, actor, remove_power, events)
    if chosen_effects or rest:
        # ⚠️ 续跑时**必须把打出的那张牌带上**（`pending_played`）：剩余效果里的
        # `block` 要靠它认来源（`ModifyBlockMultiplicative` 那一族按"来源卡牌"计数）。
        done = _apply_effects(state, tuple(chosen_effects) + tuple(rest), actor,
                              target_index, events,
                              pending_played=played, x_value=x_value)
        if not done:
            # 剩余效果里还有选牌：保留"打出的牌待落堆"的信息
            if state.pending is not None and played is not None:
                state.pending.played_card = played
                state.pending.card_consumed = True
                state.pending.plays_remaining = plays_remaining
                state.pending.result_location = result_location
            return
    if played is not None:
        # ⭐ 与 `_run_card_plays` 同序：效果跑完 → ``AfterAttack``（攻击指令结算）→
        # ``AfterCardPlayed``。挂起过的这一遍同样要结账，少一次
        # `GigantificationPower` 就少递减一层（静默变强）。
        from . import powers as power_rules
        power_rules.on_attack_command_finished(state, played)
        # ⭐ 挂起结束了：现在才算"这张牌打完了"，`AfterCardPlayed` 在这里触发
        # （与 `step` 的直通路径共用一个入口，保证只触发一次）。
        _fire_after_card_played(state, played.definition(), events, played,
                                is_last_in_series=(plays_remaining <= 0))
        # ⭐ **还没打完的次数**（`OneTwoPunchPower` 的"多打一次"）：接着打完。
        # 少了这一段，"多打一次 + 带选牌的攻击牌"就只结算一次，而日志一切正常。
        if plays_remaining > 0:
            done = _run_card_plays(state, played, played.definition(), target_index,
                                   events, x_value, plays_remaining)
            if not done:
                if state.pending is not None:
                    state.pending.played_card = played
                    state.pending.card_consumed = True
                    state.pending.result_location = result_location
                return
        _finish_played_card(state, played, played.definition(), events,
                            result_location)


# ==========================================================================
# 文本渲染（排查机制错误与人工审查 agent 行为的唯一实用工具）
# ==========================================================================
def render_text(state: CombatState) -> str:
    lines = [
        f"--- 回合 {state.turn} | 能量 {state.energy} | 阶段 {state.phase} ---",
        f"我方 HP {state.player.hp}/{state.player.max_hp} 格挡 {state.player.block} "
        f"{_fmt_powers(state.player)}",
    ]
    for enemy in state.enemies:
        if not enemy.alive():
            lines.append(f"  [死亡] {enemy.name}")
            continue
        intent = enemy.intent
        intent_text = (f"{intent.kind} {intent.value}"
                       + (f"×{intent.times}" if intent.times > 1 else "")) if intent else "?"
        lines.append(f"  {enemy.name} HP {enemy.hp}/{enemy.max_hp} 格挡 {enemy.block} "
                     f"{_fmt_powers(enemy)} 意图 {intent_text}")
    hand = ", ".join(f"{c.definition().name}{'*' if c.upgraded else ''}({c.cost()})"
                     for c in state.hand) or "（空）"
    lines.append(f"手牌[{len(state.hand)}]: {hand}")
    lines.append(f"抽牌堆 {len(state.draw_pile)} | 弃牌堆 {len(state.discard)} "
                 f"| 消耗 {len(state.exhaust)}")
    return "\n".join(lines)


def _fmt_powers(combatant: Combatant) -> str:
    if not combatant.powers:
        return ""
    return "{" + ", ".join(f"{k}:{v}" for k, v in sorted(combatant.powers.items())) + "}"
