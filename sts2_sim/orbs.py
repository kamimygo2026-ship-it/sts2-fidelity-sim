"""充能球（orb）机制 —— ``docs/09`` L4 角色机制。

数值与时机全部抄自反编译源码（``MegaCrit.Sts2.Core.Models.Orbs.*``）：

| 球 | Passive | Evoke | 触发时机 | 目标 |
|---|---|---|---|---|
| Lightning | ``ModifyOrbValue(3)`` | ``ModifyOrbValue(8)`` | 回合结束 | 随机敌人 |
| Frost | ``ModifyOrbValue(2)`` | ``ModifyOrbValue(5)`` | 回合结束 | 自己（格挡） |
| Dark | ``ModifyOrbValue(6)`` | 累计值（初值 6） | 回合结束 | 最低血敌人 |
| Glass | ``ModifyOrbValue(当前值)``（初值 4） | ``PassiveVal * 2`` | 回合结束 | 全体敌人 |
| Plasma | ``1``（**不吃 Focus**） | ``2`` | **回合开始** | 自己（能量） |

三条容易搞错、但错了就会静默偏掉的地方：

1. **伤害与格挡都带 ``ValueProp.Unpowered``** —— 不受力量 / 虚弱 / **易伤**影响，
   但**可以被格挡**。按普通伤害走会让充能球的输出凭空变化
   （Defect 的伤害来源大半是球，误差会直接反映成胜率差）。
2. **``ModifyOrbValue`` 才是 Focus 的作用点**，而 Plasma 用的是裸 ``1m``/``2m`` ——
   所以"Plasma 不受 Focus 影响"是**源码层面的**事实，不是平衡说明。
3. **Dark 与 Glass 有跨回合状态**：Dark 每次回合结束把 ``PassiveVal`` 累加进
   ``_evokeVal``；Glass 每次触发先 ``max(0, value - 1)`` 再造成伤害。
   当成无状态会让 Dark 永远只能 evoke 6、Glass 永远不减。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # pragma: no cover
    from .core import CombatState


@dataclass(frozen=True)
class OrbDef:
    """一种球的静态定义。``source`` 注明源码依据。"""

    oid: str
    name: str
    passive: int
    evoke: int
    #: 触发钩子：``turn_end``（BeforeTurnEndOrbTrigger）或 ``turn_start``。
    hook: str
    #: 作用形态：``damage_random`` / ``damage_all`` / ``damage_weakest`` /
    #: ``block`` / ``energy``。
    kind: str
    #: 是否吃 Focus（``ModifyOrbValue``）。Plasma 是唯一 False 的。
    focus_scaled: bool
    #: 跨回合状态的初值（Dark 的累计伤害 / Glass 的当前 passive 值）。
    initial_value: int = 0
    #: 每次触发后对 ``value`` 的增量（Dark 用 passive 累加，Glass 是 -1）。
    decays: bool = False
    source: str = ""


#: 球注册表。数值与源码一一对应，改动必须能指到具体那一行。
ORB_DEFS: dict[str, OrbDef] = {
    "lightning": OrbDef("lightning", "Lightning", 3, 8, "turn_end", "damage_random",
                        True, source="LightningOrb.PassiveVal/EvokeVal"),
    "frost": OrbDef("frost", "Frost", 2, 5, "turn_end", "block",
                    True, source="FrostOrb.PassiveVal/EvokeVal"),
    "dark": OrbDef("dark", "Dark", 6, 6, "turn_end", "damage_weakest",
                   True, initial_value=6, source="DarkOrb._evokeVal 初值 6"),
    "glass": OrbDef("glass", "Glass", 4, 8, "turn_end", "damage_all",
                    True, initial_value=4, decays=True,
                    source="GlassOrb._passiveVal 初值 4，EvokeVal = Passive*2"),
    "plasma": OrbDef("plasma", "Plasma", 1, 2, "turn_start", "energy",
                     False, source="PlasmaOrb：裸 1m/2m，不走 ModifyOrbValue"),
}

#: 缺陷（Defect）的基础充能球槽位。
DEFAULT_ORB_SLOTS = 3


@dataclass
class OrbState:
    """场上的一个球。``value`` 是跨回合状态（Dark 累计 / Glass 当前值）。"""

    oid: str
    value: int = 0

    def definition(self) -> OrbDef:
        return ORB_DEFS[self.oid]


def _focused(state: "CombatState", amount: int, scaled: bool) -> int:
    """``ModifyOrbValue``：Focus 只作用于声明了 ``focus_scaled`` 的球。

    源码（``FocusPower.cs:15-21``）::

        if (base.Owner.Player != orb.Owner) return value;
        return Math.Max(value + (decimal)base.Amount, 0m);

    ⚠️ **必须钳到 0**：``FocusPower.AllowNegative => true``，而
    ``BiasedCognitionPower.cs:26`` / ``TemporaryFocusPower.cs:149`` 会给**负**专注
    （``Apply<FocusPower>(-Amount)``）。少这一钳，闪电被动 3 点配 −5 专注会算出
    **−2**，然后一路带着负伤害往下走 —— 数值静默偏掉，而且偏在"负伤害"这种
    谁都不会去查的方向上。

    ``base.Owner.Player != orb.Owner``（别人的球不吃我的专注）由"单人局里
    球都属于玩家、专注也挂在玩家身上"保证。
    """
    if not scaled:
        return amount
    return max(amount + state.player.power("focus"), 0)


def passive_value(state: "CombatState", orb: OrbState) -> int:
    definition = orb.definition()
    if definition.oid == "glass":
        # Glass 的 passive 用的是**会衰减的当前值**
        return _focused(state, orb.value, definition.focus_scaled)
    return _focused(state, definition.passive, definition.focus_scaled)


def evoke_value(state: "CombatState", orb: OrbState) -> int:
    definition = orb.definition()
    if definition.oid == "dark":
        return orb.value              # Dark 的 evoke 是累计值（不吃 Focus，已累计过）
    if definition.oid == "glass":
        return passive_value(state, orb) * 2      # EvokeVal = PassiveVal * 2
    return _focused(state, definition.evoke, definition.focus_scaled)


def make_orb(oid: str) -> OrbState:
    definition = ORB_DEFS[oid]
    return OrbState(oid=oid, value=definition.initial_value)


# ==========================================================================
# 引擎入口
# ==========================================================================
def channel(state: "CombatState", oid: str, events: list[str]) -> None:
    """``OrbCmd.Channel``：获得一个球；槽位已满时**先 evoke 最左边那个**。

    真机的行为是"满了就把最旧的激发掉"（FIFO），不是"丢弃新的"。
    搞反的话球的节奏完全变样。
    """
    from .core import _orb_raw_damage, _orb_gain_block, _orb_gain_energy
    if oid not in ORB_DEFS:
        raise ValueError(f"未知充能球 {oid!r}")
    if len(state.orbs) >= state.orb_slots:
        evoke(state, 0, events)
    state.orbs.append(make_orb(oid))
    events.append(f"引导 {ORB_DEFS[oid].name}（球位 {len(state.orbs)}/{state.orb_slots}）")


def evoke(state: "CombatState", index: int, events: list[str],
          dequeue: bool = True) -> None:
    """``OrbCmd.EvokeNext``：激发指定球。

    ``dequeue=False`` 表示**激发但保留**那个球（真机第 3 个参数）——
    `DualCast`（激发两次）先用它激发一次，再正常激发一次把球移走。

    ⭐ 激发之后要分发 ``Hook.AfterOrbEvoked``（``OrbCmd.cs:147``）并带上
    **这次激发打到的目标**（源码是 ``targets = await evokedOrb.Evoke(...)`` 的返回值）。
    `ThunderPower` 就是"激发闪电球时，再对同一批目标造成 Amount 点伤害"。
    """
    from .core import _orb_raw_damage, _orb_gain_block, _orb_gain_energy
    if not (0 <= index < len(state.orbs)):
        return
    orb = state.orbs[index]
    if dequeue:
        state.orbs.pop(index)
    definition = orb.definition()
    value = evoke_value(state, orb)
    events.append(f"激发 {definition.name}（{value}）"
                  + ("" if dequeue else "（球保留）"))
    targets = _apply_orb_effect(state, definition, value, events, is_evoke=True)
    from . import powers as power_rules
    power_rules.on_orb_evoked(state, orb, targets, events)


def _apply_orb_effect(state: "CombatState", definition: OrbDef, value: int,
                      events: list[str], is_evoke: bool) -> list:
    """按球的种类结算一次效果，返回**这次影响到的单位**。

    返回值就是真机 ``OrbModel.Evoke(...)`` 的返回值（激发目标），
    ``Hook.AfterOrbEvoked`` 要把它交给订阅者（`ThunderPower`）。
    ``block`` / ``energy`` 这两类源码里返回的是**获得方**（自己），
    但今天没有订阅者会读它们 —— 先如实返回空表，等真有消费者再按源码补。
    """
    from .core import _orb_gain_block, _orb_gain_energy, _orb_raw_damage
    if definition.kind == "damage_random":
        target = _random_enemy(state)
        if target is not None:
            _orb_raw_damage(state, target, value, events, definition.name)
            return [target]
        return []
    if definition.kind == "damage_all":
        hit = list(state.living_enemies())
        for enemy in hit:
            _orb_raw_damage(state, enemy, value, events, definition.name)
        return hit
    if definition.kind == "damage_weakest":
        target = _weakest_enemy(state)
        if target is not None:
            _orb_raw_damage(state, target, value, events, definition.name)
            return [target]
        return []
    if definition.kind == "block":
        _orb_gain_block(state, value, events, definition.name)
        return []
    if definition.kind == "energy":
        _orb_gain_energy(state, value, events, definition.name)
        return []
    return []


def _random_enemy(state: "CombatState"):
    """``Rng.CombatTargets.NextItem``：随机敌人走**独立 RNG 流**，不是主随机。"""
    living = state.living_enemies()
    if not living:
        return None
    if len(living) == 1:
        return living[0]
    index = state.hidden.rng.next_index("combat_targets", len(living))
    return living[index]


def _weakest_enemy(state: "CombatState"):
    living = state.living_enemies()
    if not living:
        return None
    return min(living, key=lambda e: e.hp)


def passive(state: "CombatState", orb: OrbState, events: list[str]) -> None:
    """``OrbCmd.Passive``：**立刻**触发某个球的被动（不等它的时机）。

    ``LoopPower``（循环）就是靠它："玩家回合开始时，按层数把**最左边那个球**
    的被动触发 N 次"。

    ⚠️ 跨回合状态必须一起处理 —— 真机把状态更新写在**球自己的
    ``Passive(…)`` 里**，而不是写在时机钩子里：

    * ``DarkOrb.Passive`` → ``_evokeVal += PassiveVal``（暗球越转越大）
    * ``GlassOrb.Passive`` → 先按**当前值**造成伤害，再 ``_passiveVal -= 1``

    所以"被叫醒"和"到点了自动触发"必须走同一条路径。少这一步，
    循环触发的暗球永远不涨、玻璃球永远不衰减 —— 而且不报错。
    """
    definition = orb.definition()
    value = passive_value(state, orb)
    _apply_orb_effect(state, definition, value, events, is_evoke=False)
    if definition.oid == "dark":
        orb.value += value
    elif definition.decays:
        orb.value = max(0, orb.value - 1)


def on_turn_end(state: "CombatState", events: list[str]) -> None:
    """``BeforeTurnEndOrbTrigger``：回合结束时按**从左到右**的顺序触发被动。"""
    for orb in list(state.orbs):
        if orb.definition().hook != "turn_end":
            continue
        passive(state, orb, events)


def on_turn_start(state: "CombatState", events: list[str]) -> None:
    """``AfterTurnStartOrbTrigger``：等离子在**回合开始**给能量。"""
    for orb in list(state.orbs):
        definition = orb.definition()
        if definition.hook != "turn_start":
            continue
        _apply_orb_effect(state, definition, passive_value(state, orb),
                          events, is_evoke=False)


def add_slots(state: "CombatState", amount: int, events: list[str]) -> None:
    """``OrbCmd.AddSlots``。"""
    state.orb_slots += amount
    events.append(f"球位 +{amount}（现在 {state.orb_slots}）")


def coverage() -> dict:
    return {"orbs_total": len(ORB_DEFS), "implemented": sorted(ORB_DEFS)}
