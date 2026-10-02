"""敌人出招状态机——严格按反编译源码实现（``docs/09`` L1）。

**真相来源**：``MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine.*``。
状态机**算法本身**是那份 C# 的逐行翻译。

⚠️ 但算法翻译正确**不等于**运行时出招正确（审计 F05）：条件求值需要**真实的运行状态**
（队友数量、血量阈值、槽位名、怪物专属 flags/counters）。缺失时本模块抛
``UnsupportedMechanic``，**不再默认 False/0** —— 默认值会让
``Fabricator.CanFabricate`` 在单独生成时被判成假，初始意图从召唤分支
悄悄变成 ``DISINTEGRATE_MOVE``，而且没有任何信号。

关键语义（全部来自源码）：

1. **初始状态本身就是第一招**。``FindNextMoveState`` 开头：
   ``if (!CanTransitionAway || (!_performedFirstMove && IsMove)) return;``
2. **``next`` 为空 → 回到初始状态**（不是"结束"，也不是"随机"）：
   ``SetCurrentState(string.IsNullOrEmpty(next) ? _initialState : States[next]);``
3. **非 move 状态一路走到 move 为止**：``do { … } while (!_currentState.IsMove);``
4. **随机分支的权重算法**见 ``_branch_weight``，与 ``RandomBranchState.GetStateWeight`` 逐行对应。
5. **``MustPerformOnceBeforeTransitioning``**：未执行过就不许转移（4 只怪用）。
6. 出招随机走 **``RunRng.MonsterAi``** 流——与"遭遇选择"是两条独立的流。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

#: 与 MoveRepeatType 一一对应
CANNOT_REPEAT = "CannotRepeat"
CAN_REPEAT_FOREVER = "CanRepeatForever"
CAN_REPEAT_X_TIMES = "CanRepeatXTimes"
USE_ONLY_ONCE = "UseOnlyOnce"


class MonsterAiError(RuntimeError):
    """状态机运行到源码里会抛异常的分支（悬空引用 / 无可用分支）。"""


class UnknownCondition(MonsterAiError):
    """遇到了还没实现的 ``condition`` 表达式——**必须报出来**，不能默默当 false。"""


class UnsupportedMechanic(MonsterAiError):
    """条件需要的**运行状态没有被提供**（``docs/13`` §5）。

    ``MonsterContext.flag`` / ``counter`` 以前对缺失的键返回 False / 0 —— 于是
    ``Fabricator.CanFabricate``（"存活队友 < 4"）在单独生成时被判成 False，
    初始意图变成 ``DISINTEGRATE_MOVE``。正确的分支是**随机分支**（召唤系）。
    这个错误没有任何信号：战斗能跑完，只是怪的出招一直是错的。

    所以缺失值一律报错：要么调用方把真实运行状态算出来，要么显式报"未支持"。
    """


# ==========================================================================
# 数据
# ==========================================================================
@dataclass(frozen=True)
class Branch:
    state_id: str
    repeat: str = CAN_REPEAT_FOREVER
    max_times: int = 0
    weight: float = 1.0
    cooldown: int = 0


@dataclass(frozen=True)
class State:
    state_id: str
    kind: str                       # move | random | conditional
    move: str = ""                  # move 状态绑定的处理函数名
    follow_up: str | None = None
    must_perform_once: bool = False
    branches: tuple[Branch, ...] = ()
    conditions: tuple[tuple[str, str], ...] = ()   # (目标状态, 条件表达式)

    @property
    def is_move(self) -> bool:
        return self.kind == "move"

    @property
    def in_logs(self) -> bool:
        """``ShouldAppearInLogs``：只有 move 状态进日志。"""
        return self.kind == "move"


@dataclass
class MonsterContext:
    """条件求值所需的上下文。

    这是"生物能观察到的状态"，不是全局游戏状态——条件表达式只引用这些东西。

    ⚠️ ``flags`` / ``counters`` 里**必须**包含该怪条件里引用到的每一个名字。
    缺失时 :meth:`flag` / :meth:`counter` 抛 :class:`UnsupportedMechanic`，
    **不再默认 False / 0**（``docs/13`` §5，审计 F05）。
    """

    slot_name: str = ""
    is_front: bool = False
    is_alone: bool = False
    ally_count: int = 0
    hp: int = 0
    max_hp: int = 1
    powers: frozenset[str] = frozenset()
    #: 怪物专属标志与计数器（HasAmalgamDied / Respawns / _curseOfKnowledgeCounter…）
    flags: dict[str, bool] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def flag(self, name: str) -> bool:
        if name not in self.flags:
            raise UnsupportedMechanic(
                f"条件引用了标志 {name!r}，但运行状态里没有它的值。"
                f"缺少必需值时必须报错，不能默认 False（docs/13 §5）")
        return bool(self.flags[name])

    def counter(self, name: str) -> int:
        if name not in self.counters:
            raise UnsupportedMechanic(
                f"条件引用了计数器 {name!r}，但运行状态里没有它的值。")
        return int(self.counters[name])


# ==========================================================================
# 槽位与专属状态（审计 F05）
# ==========================================================================
#: 真机的槽位名。**默认按位置取序数**：
#:
#: * 多数遭遇在源码里就是 ``"first" / "second" / "third" / "fourth"``
#:   （``ExoskeletonEncounter`` / ``MyteEncounter`` / ``KnightEncounter`` …）
#: * ``BowlbugsNormal`` 是唯一用 ``first / middle / last`` 的（``BowlbugsNormal.cs``
#:   的 ``_slotNames``）
#:
#: ⚠️ 把"第几个敌人"直接当成槽位名就是审计 F05 的前半段：三只怪的第 3 只
#: 会被叫成 ``second``，而 ``Exoskeleton`` 的条件是 ``SlotName == "third"`` ——
#: 于是它的出招永远是错的。这里至少把序数补全，并把"非序数槽位"记成已知缺口。
SLOT_ORDINALS: tuple[str, ...] = ("first", "second", "third", "fourth", "fifth")

#: 已知**不使用序数**的遭遇（源码里的自定义 ``_slotNames``）。
#: 这些遭遇的槽位需要专门的遭遇数据，当前只能按位置近似 —— 必须能被报出来。
NON_ORDINAL_SLOT_ENCOUNTERS: frozenset[str] = frozenset({"bowlbugs_normal"})


def slot_name_for(position: int) -> str:
    """第 ``position`` 个敌人的槽位名（0 基）。超出序数表就报错，不静默兜底。"""
    if 0 <= position < len(SLOT_ORDINALS):
        return SLOT_ORDINALS[position]
    raise UnsupportedMechanic(
        f"位置 {position} 超出已知槽位表 {SLOT_ORDINALS}；"
        f"真机最多 5 个敌人槽位（EncounterModel.Slots）")


#: 每个已知标志的求值规则。
#:
#: ``("derived", 函数)`` —— 每次求值**从运行状态现算**（队友数量、血量…）
#: ``("instance", None)``  —— 怪物实例上的持久标志，由招式副作用写入
#:
#: 每条都带源码出处（铁律 R1）。新增条件标志必须同时加进这里，否则
#: :func:`resolve_flags` 会报 UnsupportedMechanic 而不是猜一个值。
FLAG_RULES: dict[str, tuple[str, object]] = {
    # `Fabricator.cs:51`：`GetTeammatesOf(Creature).Count(IsAlive) < 4`
    # （`GetTeammatesOf` **含自己**，见 `CombatState.cs`）
    "CanFabricate": ("derived", lambda allies: len(allies) < 4),
    # `Ovicopter.cs`：`GetTeammatesOf(Creature).Count(IsAlive) <= 3`
    "CanLay": ("derived", lambda allies: len(allies) <= 3),
    # 以下三个是**实例标志**：源码里是私有字段，由招式体写入。
    "HasAmalgamDied": ("instance", None),    # Queen：羊膜体死亡时置 true
    "IsOffBalance": ("instance", None),      # BowlbugRock：失衡/眩晕
    "HasBeetleCharged": ("instance", None),  # FrogKnight：甲虫冲撞后置 true
}

#: 实例标志的出厂值（C# 字段默认 ``false``）。
#: **显式初始化**，而不是查不到就当 False —— 两者的区别是后者查不出拼写错误。
INSTANCE_FLAG_DEFAULTS: dict[str, bool] = {
    name: False for name, (kind, _rule) in FLAG_RULES.items() if kind == "instance"
}

#: 条件里用到的**实例计数器**：``名字 → (源码出处, 出厂值)``。
#: 同样是 C# 私有 ``int`` 字段（默认 0），由招式体自增：
#: ``KnowledgeDemon._curseOfKnowledgeCounter``、``TestSubject._respawns``。
COUNTER_SOURCES: dict[str, tuple[str, int]] = {
    "_curseOfKnowledgeCounter": ("KnowledgeDemon.cs:66", 0),
    "Respawns": ("TestSubject.cs:63", 0),
}

#: 实例计数器的出厂值。
COUNTER_DEFAULTS: dict[str, int] = {
    name: default for name, (_source, default) in COUNTER_SOURCES.items()}


def flags_referenced(record: dict) -> set[str]:
    """一条状态机记录的条件里引用到的**全部标志名**。

    用来检查"这只怪需要的标志是否都被初始化了"，而不是等运行到某个分支才发现。
    """
    names: set[str] = set()
    for branches in (record.get("conditionals") or {}).values():
        for branch in branches:
            for token in re.findall(r"[A-Za-z_]\w*", str(branch.get("condition", ""))):
                if token in FLAG_RULES:
                    names.add(token)
    return names


def counters_referenced(record: dict) -> set[str]:
    """条件里引用到的**计数器名**（与 :func:`flags_referenced` 同一套扫描）。"""
    names: set[str] = set()
    for branches in (record.get("conditionals") or {}).values():
        for branch in branches:
            condition = str(branch.get("condition", ""))
            for token in re.findall(r"[A-Za-z_]\w*", condition):
                if token in COUNTER_SOURCES:
                    names.add(token)
    return names


def resolve_flags(living_allies: Sequence[object],
                  instance_flags: dict[str, bool] | None = None) -> dict[str, bool]:
    """算出一次条件求值要用的**完整**标志表。

    ``living_allies`` 是**同阵营存活单位（含自己）**——真机的
    ``GetTeammatesOf`` 就是这么定义的（``CombatState.cs``："including the
    creature itself"）。审计 F05 的 Fabricator 用例正是靠这一条修正的。
    """
    flags = dict(INSTANCE_FLAG_DEFAULTS)
    if instance_flags:
        flags.update(instance_flags)
    for name, (kind, rule) in FLAG_RULES.items():
        if kind == "derived":
            flags[name] = bool(rule(living_allies))  # type: ignore[operator]
    return flags


# ==========================================================================
# 条件求值（32 个表达式全覆盖）
# ==========================================================================
_CAST = re.compile(r"\(\s*\w+\s*\)")
_SLOT = re.compile(r'SlotName\s*==\s*"([^"]+)"')
_HAS_POWER = re.compile(r"HasPower<(\w+)>\(\)")
_ALLIES = re.compile(r"GetAllyCount\(\)\s*(==|>=|<=|>|<)\s*(\d+)")
#: ⚠️ 真机写的是 `CurrentHp >= base.Creature.MaxHp / 2`——`MaxHp` 前面还有一段路径，
#: 只写 `CurrentHp\s*op\s*MaxHp` 会漏掉，导致 frogknight 的两个条件求值失败。
_HP = re.compile(r"CurrentHp\s*(==|>=|<=|>|<)\s*[\w.]*MaxHp\s*/\s*(\d+)")
_MEMBER = re.compile(r"\.(\w+)$")
_COUNTER = re.compile(r"^(\w+)\s*(==|>=|<=|>|<)\s*(\d+)$")

#: 数据里出现过的全部布尔标志。
#: ⚠️ 白名单是必要的：裸标识符一律当 flag 的话，**拼错的 flag 会静默返回 False**，
#: 出招就会悄悄走错分支而且没有任何信号。新增标志必须显式加进来
#: （``tests/test_monster_ai_runtime.py`` 会扫全量数据，漏了会失败）。
KNOWN_FLAGS: frozenset[str] = frozenset({
    "HasAmalgamDied",       # queen
    "IsOffBalance",         # bowlbugrock
    "CanFabricate",         # fabricator
    "HasBeetleCharged",     # frogknight
    "CanLay",               # ovicopter
})

_OPS: dict[str, Callable[[float, float], bool]] = {
    "==": lambda a, b: a == b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}


def evaluate_condition(expr: str, ctx: MonsterContext) -> bool:
    """求值一个真机的 ``condition`` 表达式。

    未支持的表达式抛 :class:`UnknownCondition`——**不静默返回 False**，
    否则怪的出招会悄悄走错分支，而且没有任何信号。
    """
    return _eval_or(expr.strip(), ctx)


def _split_top(expr: str, operator: str) -> list[str]:
    """按顶层运算符切分（忽略括号内）。"""
    parts, depth, current = [], 0, []
    index = 0
    while index < len(expr):
        char = expr[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth == 0 and expr.startswith(operator, index):
            parts.append("".join(current))
            current = []
            index += len(operator)
            continue
        current.append(char)
        index += 1
    parts.append("".join(current))
    return parts


def _eval_or(expr: str, ctx: MonsterContext) -> bool:
    parts = _split_top(expr, "||")
    return any(_eval_and(p, ctx) for p in parts) if len(parts) > 1 else _eval_and(expr, ctx)


def _eval_and(expr: str, ctx: MonsterContext) -> bool:
    parts = _split_top(expr, "&&")
    return all(_eval_atom(p, ctx) for p in parts) if len(parts) > 1 else _eval_atom(expr, ctx)


def _eval_atom(expr: str, ctx: MonsterContext) -> bool:
    expr = expr.strip()
    if expr.startswith("(") and expr.endswith(")") and _balanced(expr):
        return _eval_or(expr[1:-1], ctx)

    negated = False
    while expr.startswith("!"):
        negated = not negated
        expr = expr[1:].strip()

    value = _eval_positive(expr, ctx)
    return not value if negated else value


def _balanced(expr: str) -> bool:
    depth = 0
    for index, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(expr) - 1:
                return False
    return depth == 0


def _eval_positive(expr: str, ctx: MonsterContext) -> bool:
    normalized = _CAST.sub("", expr).strip()

    match = _SLOT.search(normalized)
    if match:
        return ctx.slot_name == match.group(1)

    match = _HAS_POWER.search(normalized)
    if match:
        return _power_key(match.group(1)) in ctx.powers

    match = _ALLIES.search(normalized)
    if match:
        return _OPS[match.group(1)](ctx.ally_count, int(match.group(2)))

    match = _HP.search(normalized)
    if match:
        return _OPS[match.group(1)](ctx.hp, ctx.max_hp / int(match.group(2)))

    member = _MEMBER.search(normalized.rstrip("()"))
    if member:
        name = member.group(1)
        if name == "IsFront":
            return ctx.is_front
        if name == "IsAlone":
            return ctx.is_alone

    match = _COUNTER.match(normalized)
    if match:
        return _OPS[match.group(2)](ctx.counter(match.group(1)), int(match.group(3)))

    if re.fullmatch(r"\w+", normalized):
        if normalized not in KNOWN_FLAGS:
            raise UnknownCondition(f"未知标志 {normalized!r}（若是新机制，请加进 KNOWN_FLAGS）")
        return ctx.flag(normalized)

    raise UnknownCondition(expr)


def _power_key(class_name: str) -> str:
    """``AsleepPower`` → ``asleep``（与 powers.json 的 id 对齐）。"""
    name = class_name[:-5] if class_name.endswith("Power") else class_name
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


# ==========================================================================
# 状态机
# ==========================================================================
class MoveStateMachine:
    """``MonsterMoveStateMachine`` 的忠实翻译。"""

    def __init__(self, states: dict[str, State], initial: str) -> None:
        missing = [initial] + [s.follow_up for s in states.values() if s.follow_up]
        for state_id in missing:
            if state_id and state_id not in states:
                raise MonsterAiError(f"状态机引用了未定义的状态 {state_id!r}")
        self.states = dict(states)
        self.initial = initial
        self.current = initial
        self.state_log: list[str] = []
        self._performed_first_move = False
        self._performed_once: set[str] = set()
        if states[initial].in_logs:
            self.state_log.append(initial)

    # ---- 供引擎调用 ---------------------------------------------------
    def set_immediate(self, state_id: str) -> None:
        """``MonsterModel.SetMoveImmediate(state)``：**强制跳到某个状态**。

        真机用它做"死后进入下一形态"（``WaterfallGiant.TriggerAboutToBlowState``）、
        "被晕住时换成 StunnedMove" 这类**状态注入**。没有它，
        "死了之后变身再打一轮"的怪只能被近似成"直接死掉"。
        """
        if state_id not in self.states:
            raise MonsterAiError(f"强制跳转到未定义的状态 {state_id!r}")
        self.current = state_id
        if self.states[state_id].in_logs:
            self.state_log.append(state_id)

    def roll_move(self, ctx: MonsterContext, rng) -> State:
        """决定这一回合的招式。对应 ``RollMove``。"""
        current = self.states[self.current]
        if not self._can_transition_away(current) or (
                not self._performed_first_move and current.is_move):
            return current

        first_logged: str | None = None
        for _ in range(64):                     # 防悬空环：真机是 do/while，同样可能死循环
            next_id = self._next_state_id(current, ctx, rng)
            self.current = self.initial if not next_id else next_id
            current = self.states[self.current]
            if first_logged is None and current.in_logs:
                first_logged = current.state_id
            if current.is_move:
                break
        else:
            raise MonsterAiError(f"{self.initial}: 走了 64 步仍没落到 move 状态（疑似成环）")

        if first_logged is not None:
            self.state_log.append(first_logged)
        return current

    def on_move_performed(self) -> None:
        """招式执行完毕。对应 ``OnMovePerformed`` + ``MoveState.PerformMove``。"""
        self._performed_first_move = True
        self._performed_once.add(self.current)

    @property
    def next_move(self) -> State:
        return self.states[self.current]

    # ---- 内部 ---------------------------------------------------------
    def _can_transition_away(self, state: State) -> bool:
        """``MoveState.CanTransitionAway``：must_perform_once 未满足时不许转移。"""
        if state.must_perform_once:
            return state.state_id in self._performed_once
        return True

    def _next_state_id(self, state: State, ctx: MonsterContext, rng) -> str:
        if state.kind == "move":
            if state.follow_up is None:
                # 源码在这里抛异常：MoveState.GetNextState 要求必有后继
                raise MonsterAiError(f"{state.state_id} 没有后继状态")
            # ⚠️ 空字符串与 None **语义不同**：
            # `string.IsNullOrEmpty(next) ? _initialState : States[next]`
            # 空 → 回到初始状态；缺失 → 源码直接抛异常。
            return state.follow_up

        if state.kind == "random":
            weights = [(b, self._branch_weight(b, state)) for b in state.branches]
            total = sum(w for _b, w in weights)
            if total <= 0:
                raise MonsterAiError(f"{state.state_id}: 所有分支权重都是 0")
            roll = rng.uniform(0.0, total)
            for branch, weight in weights:
                roll -= weight
                if roll <= 0:
                    return branch.state_id
            return weights[-1][0].state_id

        if state.kind == "conditional":
            for state_id, condition in state.conditions:
                if evaluate_condition(condition, ctx):
                    return state_id
            raise MonsterAiError(f"{state.state_id}: 没有条件成立（源码此处同样抛异常）")

        raise MonsterAiError(f"未知状态类型 {state.kind!r}")

    def _branch_weight(self, branch: Branch, owner: State) -> float:
        """``RandomBranchState.GetStateWeight`` 的逐行翻译。"""
        weight = 1.0
        if branch.repeat == USE_ONLY_ONCE:
            if branch.state_id in self.state_log:
                weight = 0.0
        elif branch.repeat != CAN_REPEAT_FOREVER:
            limit = 1 if branch.repeat == CANNOT_REPEAT else branch.max_times
            weight = 1.0 if len(self.state_log) < limit else 0.0
            index = 0
            while (len(self.state_log) >= limit and index < limit
                   and len(self.state_log) - index > 0):
                if self.state_log[len(self.state_log) - 1 - index] != branch.state_id:
                    weight = 1.0
                    break
                index += 1

        if branch.cooldown > 0:
            recent = list(reversed(self.state_log))[:branch.cooldown]
            if branch.state_id in recent:
                return 0.0
        return weight * branch.weight


# ==========================================================================
# 从提取数据构造
# ==========================================================================
def build(record: dict) -> MoveStateMachine:
    """把 ``monster_ai.json`` 的一条记录变成状态机。

    提取出来的数据把转移/分支/条件放在**顶层**三个字典里（按状态 id 索引），
    而状态机内部希望它们挂在状态上——这里做一次归拢。
    """
    follow_up = record.get("follow_up") or {}
    branches = record.get("branches") or {}
    conditionals = record.get("conditionals") or {}
    once = set(record.get("must_perform_once") or ())

    states: dict[str, State] = {}
    for state_id, raw in record["states"].items():
        states[state_id] = State(
            state_id=state_id,
            kind=raw.get("kind", "move"),
            move=raw.get("move") or "",
            follow_up=follow_up.get(state_id),
            must_perform_once=state_id in once,
            branches=tuple(
                Branch(state_id=b["state"],
                       repeat=b.get("repeat") or CAN_REPEAT_FOREVER,
                       max_times=int(b.get("max_times") or 0),
                       weight=float(b.get("weight") or 1.0),
                       cooldown=int(b.get("cooldown") or 0))
                for b in branches.get(state_id, ())),
            conditions=tuple((c["state"], c["condition"])
                             for c in conditionals.get(state_id, ())),
        )
    return MoveStateMachine(states, record["initial"])
