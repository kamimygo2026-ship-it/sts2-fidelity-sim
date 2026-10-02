"""从反编译的 C# 源码里提取敌人出招状态机（``docs/09`` L1）。

**为什么不直接用 codex 的 JSON**：实测它在这件事上**不可靠**。以 ``BowlbugRock`` 为例，
C# 写的是链式赋值 ``headbutt.FollowUpState = new ConditionalBranchState("POST_HEADBUTT")``，
而 JSON 把这一条记成了 ``"next": null``——提取器漏掉了链式赋值，于是"Dizzy 之后回到 Headbutt"
的转移整条消失。**控制流必须以源码为准，JSON 只用来取数值。**

    python tools/extract_monsters.py            # 提取并写 data/content/repo/monster_ai.json
    python tools/extract_monsters.py --report   # 只看提取质量报告

产出结构（每只怪）：

    {
      "eid": "mawler",
      "states": {"CLAW_MOVE": {"kind": "move", "move": "CLAW"}, "RAND": {"kind": "random"}},
      "follow_up": {"CLAW_MOVE": "RAND", ...},      # 缺省 → 回到初始状态
      "branches": {"RAND": [{"state": "CLAW_MOVE", "repeat": "CannotRepeat", "weight": 1.0}]},
      "conditionals": {"POST_HEADBUTT": [{"state": "DIZZY_MOVE", "condition": "IsOffBalance"}]},
      "initial": "RIP_AND_TEAR_MOVE" 等,
      "moves": {"CLAW": {"intent": "MultiAttackIntent", "args": ["ClawDamage", "2"]}},
      "constants": {"ClawDamage": [5, 4], ...},     # [进阶值, 普通值]
      "hp": [76, 72],
    }
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

DECOMPILED = pathlib.Path("data/decompiled/sts2/MegaCrit.Sts2.Core.Models.Monsters")
OUT_PATH = pathlib.Path("data/content/repo/monster_ai.json")

STATE_KINDS = ("MoveState", "RandomBranchState", "ConditionalBranchState")
#: MoveRepeatType 枚举 → 我们的表示
REPEAT_TYPES = ("CannotRepeat", "CanRepeatForever", "CanRepeatXTimes", "UseOnlyOnce")


# ==========================================================================
# C# 小工具
# ==========================================================================
def method_body(source: str, name: str) -> str | None:
    """按大括号配平取出方法体。"""
    match = re.search(rf"\b{name}\s*\([^)]*\)\s*\{{", source)
    if not match:
        return None
    start = match.end() - 1
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1:index]
    return None


def statements(body: str) -> list[str]:
    """按分号切语句（足够用：方法体里没有含分号的字符串字面量）。"""
    return [s.strip() for s in body.split(";") if s.strip()]


ASC_CALL = re.compile(
    r"AscensionHelper\.GetValueIfAscension\(\s*AscensionLevel\.(\w+)\s*,\s*"
    r"([^,]+?)\s*,\s*([^)]+?)\s*\)")


def parse_constant(expr: str, consts: dict[str, list[int]]) -> int | None:
    """把 ``ClawDamage`` / ``AscensionHelper.GetValueIfAscension(...)`` / 字面量解析成普通难度数值。"""
    expr = expr.strip()
    call = ASC_CALL.search(expr)
    if call:
        return _int_of(call.group(3), consts)          # 第三个参数 = 普通难度
    return _int_of(expr, consts)


def _resolve_expr(expr: str, consts: dict[str, list[int]]) -> int | None:
    """把数值表达式解析成**普通难度**的值。

    支持三种形态：字面量、``AscensionHelper.GetValueIfAscension(L, asc, normal)``、
    以及对另一个常量的引用（真机常写 ``MaxInitialHp => MinInitialHp``）。
    """
    expr = expr.strip()
    call = ASC_CALL.search(expr)
    if call:
        return _int_of(call.group(3), consts)
    return _int_of(expr, consts)


def _int_of(expr: str, consts: dict[str, dict]) -> int | None:
    # ⚠️ 口径要与 `_ascension_of` 一致：它也剥 lambda。只在一边剥会让
    # "未解析参数"的统计虚高（实测虚报 5 个）。
    expr = _strip_lambda(expr).rstrip("mM")
    if re.fullmatch(r"-?\d+", expr):
        return int(expr)
    if expr in consts:
        return consts[expr]["normal"]
    return None


def _strip_lambda(expr: str) -> str:
    """去掉 lambda 包装：``() => ExplodeDamage`` → ``ExplodeDamage``。

    真机常把意图参数写成 lambda（``new SingleAttackIntent(() => ExplodeDamage)``），
    不剥掉就解析不出数值。
    """
    return re.sub(r"^\(\s*\)\s*=>\s*", "", expr.strip()).strip()


def _ascension_of(expr: str, consts: dict[str, dict]) -> dict | None:
    """取某个表达式的进阶值，用于生成"进阶感知"的招式数值表。

    支持：常量名、字面量、lambda 包装、以及**内联的**
    ``AscensionHelper.GetValueIfAscension(...)``（HP 就是内联写法）。
    """
    expr = _strip_lambda(expr)
    call = ASC_CALL.search(expr)
    if call:
        asc = _literal(call.group(2))
        normal = _literal(call.group(3))
        if asc is not None and normal is not None:
            return {"normal": normal, "ascension": asc,
                    "level": ASCENSION_LEVELS.get(call.group(1), 0)}
        return None
    if expr in consts:
        return dict(consts[expr])
    if re.fullmatch(r"-?\d+", expr):
        return {"normal": int(expr), "ascension": int(expr), "level": 0}
    return None


def collect_constants(source: str, inherited: dict[str, dict] | None = None) -> dict[str, dict]:
    """收集 ``X => AscensionHelper.GetValueIfAscension(Level, asc, normal)`` 形式的常量。

    返回 ``{名字: {"normal": n, "ascension": a, "level": 门槛}}``。
    **``AscensionLevel`` 的枚举值就是进阶等级**（``AscensionLevel.cs``）：
    ``ToughEnemies = 8``（怪物加血）、``DeadlyEnemies = 9``（怪物加伤）……

    ``inherited`` 是基类常量兜底：``DecimillipedeSegment*`` 与 ``MysteriousKnight``
    的伤害常量定义在基类里，只看自己的文件会取不到（实测少 12 处未解析参数）。
    **派生类优先**，不被基类覆盖。
    """
    consts: dict[str, dict] = dict(inherited or {})
    pattern = re.compile(
        r"\b(?:public|private|protected|internal)\s+(?:static\s+)?(?:override\s+)?"
        r"(?:int|decimal)\s+(\w+)\s*=>\s*([^;]+);")
    aliases: dict[str, str] = {}
    for name, expr in pattern.findall(source):
        call = ASC_CALL.search(expr)
        if call:
            asc = _literal(call.group(2))
            normal = _literal(call.group(3))
            if asc is not None and normal is not None:
                consts[name] = {"normal": normal, "ascension": asc,
                                "level": ASCENSION_LEVELS.get(call.group(1), 0)}
            continue
        value = _literal(expr)
        if value is not None:
            consts[name] = {"normal": value, "ascension": value, "level": 0}
            continue
        # 别名属性：真机大量写 `MaxInitialHp => MinInitialHp`，而别名会**成链**
        # （`MaxInitialHp` → `MinInitialHp` → `FirstFormHp`）。
        # 只认字面量会让整条链一个都记不下来 —— 实测 `TestSubject` 的
        # `MaxInitialHp` 因此变成 None，写进表里就是"最大 HP 未知"。
        token = _strip_lambda(expr)
        if re.fullmatch(r"[A-Za-z_]\w*", token):
            aliases[name] = token

    # 别名要**迭代到不动点**：声明顺序不保证（`MinInitialHp` 在 `FirstFormHp`
    # 之前声明），单趟扫描解析不出来。
    while True:
        progressed = False
        for name, target in aliases.items():
            if name in consts or target not in consts:
                continue
            consts[name] = dict(consts[target])       # 别名继承目标的进阶信息
            progressed = True
        if not progressed:
            break
    return consts


def base_class_of(source: str) -> str | None:
    match = re.search(r"\bclass\s+\w+\s*:\s*(\w+)", source)
    return match.group(1) if match else None


def inherited_constants(class_name: str, sources: dict[str, str],
                        cache: dict[str, dict] | None = None,
                        depth: int = 0) -> dict[str, dict]:
    """沿继承链收集常量（派生类覆盖基类）。"""
    cache = {} if cache is None else cache
    if depth > 6 or class_name not in sources:
        return {}
    if class_name in cache:
        return cache[class_name]
    source = sources[class_name]
    merged: dict[str, dict] = {}
    parent = base_class_of(source)
    if parent:
        merged.update(inherited_constants(parent, sources, cache, depth + 1))
    merged.update(collect_constants(source))
    cache[class_name] = merged
    return merged


#: ``AscensionLevel`` 枚举序号 —— 枚举值本身就是进阶等级
ASCENSION_LEVELS: dict[str, int] = {
    "None": 0, "SwarmingElites": 1, "WearyTraveler": 2, "Poverty": 3,
    "TightBelt": 4, "AscendersBane": 5, "Inflation": 6, "Scarcity": 7,
    "ToughEnemies": 8, "DeadlyEnemies": 9, "DoubleBoss": 10,
}


def _literal(expr: str) -> int | None:
    expr = expr.strip().rstrip("mM")
    if re.fullmatch(r"-?\d+", expr):
        return int(expr)
    return None


def unconsumed_terms(expr: str) -> str:
    """返回表达式里**没能进入数值**的残余部分。

    真机写 ``GetValueIfAscension(ToughEnemies, 76, 70) + RespawnMaxHpBonus``，
    正则只吃掉了前半段，``+ RespawnMaxHpBonus`` 被静默丢弃 —— 算出来的数
    看起来完全正常，但是错的（``Axebot`` 的加成恰好是 0 才蒙对）。

    宁可显式标注也不留暗伤：残余非空就记进报告（``docs/02`` §2.5）。
    """
    expr = _strip_lambda(expr).strip().rstrip("mM").strip()
    call = ASC_CALL.search(expr)
    if call:
        expr = (expr[:call.start()] + expr[call.end():]).strip()
    elif re.fullmatch(r"[A-Za-z_]\w*|-?\d+", expr):
        return ""
    else:
        return expr
    return expr.strip(" +-*/").strip()


def prop_expr(prop: str, class_name: str, sources: dict[str, str],
              depth: int = 0) -> tuple[str, str] | None:
    """沿**继承链**找属性的表达式，返回 ``(定义所在类, 表达式)``。

    只搜自己的文件会漏掉整类怪物：``MysteriousKnight : FlailKnight``、
    ``DecimillipedeSegment{Front,Middle,Back} : DecimillipedeSegment``
    的 ``MinInitialHp`` 只写在基类里，派生类文件里一个字都没有 ——
    实测这 4 只怪的 HP 全成了 ``None``。
    """
    if depth > 6 or class_name not in sources:
        return None
    source = sources[class_name]
    match = re.search(rf"\bint\s+{prop}\s*=>\s*([^;]+);", source)
    if match:
        return class_name, match.group(1).strip()
    base = base_class_of(source)
    if not base:
        return None
    return prop_expr(prop, base, sources, depth + 1)


# ==========================================================================
# 状态机提取
# ==========================================================================
DECL = re.compile(
    r"(?:(MoveState|RandomBranchState|ConditionalBranchState)\s+(\w+)\s*=\s*)?"
    r"new\s+(MoveState|RandomBranchState|ConditionalBranchState)\s*\(\s*\"([^\"]+)\"")
#: 语句开头的声明（允许 `= (MoveState)(...)` 这种带转型的初始化）
DECL_VAR = re.compile(
    r"^\s*(MoveState|RandomBranchState|ConditionalBranchState)\s+(\w+)\s*=")
#: 局部变量形式的状态别名，例如 `MoveState initialState = (_flag ? a : b);`
VAR_ALIAS = re.compile(
    r"^\s*(?:MoveState|MonsterState|RandomBranchState|ConditionalBranchState)\s+"
    r"(\w+)\s*=\s*(.+)$")
#: **属性/字段**形式的赋值：``DeadState = new MoveState("RESPAWN_MOVE", …)``
#:
#: ⚠️ 真机会把状态存进**带 setter 的属性**里
#: （``private MoveState DeadState { get => _deadState; set {…} }``），
#: 于是赋值语句**没有类型前缀**，``DECL_VAR`` 匹配不到，变量名绑不上状态 id。
#: 后果不是少一个状态，而是后续 ``DeadState.FollowUpState = …`` 整条转移被丢掉 ——
#: 运行时报"没有后继状态"直接炸（实测 ``TestSubject`` 就死在这里）。
DECL_PROP = re.compile(
    r"^\s*(\w+)\s*=\s*(?:\([^)]*\)\s*)?"
    r"new\s+(MoveState|RandomBranchState|ConditionalBranchState)\b")
RETURN_MACHINE = re.compile(r"new\s+MonsterMoveStateMachine\s*\(")


def extract_machine(source: str) -> dict | None:
    body = method_body(source, "GenerateMoveStateMachine")
    if body is None:
        return None

    states: dict[str, dict] = {}          # 状态 id → {kind, move}
    var_to_state: dict[str, str] = {}     # 变量名 → 状态 id
    aliases: dict[str, str] = {}          # 局部变量 → 原始表达式（条件初始状态）
    follow_up: dict[str, str] = {}
    branches: dict[str, list] = {}
    conditionals: dict[str, list] = {}
    moves: dict[str, dict] = {}
    initial: str | None = None
    initial_expr: str | None = None
    extra_returns: list[dict] = []
    must_once: set[str] = set()

    stmt_list = statements(body)
    for index, stmt in enumerate(stmt_list):
        # --- 状态声明 -------------------------------------------------
        # ⚠️ 真机会写 `MoveState x = (MoveState)(y.FollowUpState = new MoveState("ID",…))`，
        # `=` 后面是转型而不是 `new`。变量名必须与 `new` 分开匹配（漏了会少算 4 只怪）。
        declared = DECL_VAR.match(stmt)
        # 属性形式：`DeadState = new MoveState("RESPAWN_MOVE", …)`，没有类型前缀。
        # 只在**整条语句只有一个 new** 时绑定，避免一句里建多个状态时张冠李戴。
        prop_assigned = DECL_PROP.match(stmt)
        # ⚠️ `MustPerformOnceBeforeTransitioning = true` 写在**对象初始化器**里
        # （`new MoveState(...) { MustPerformOnceBeforeTransitioning = true }`），
        # 没有变量前缀，所以不能靠 `x.MustPerform...` 正则去找。
        statement_must_once = "MustPerformOnceBeforeTransitioning = true" in stmt
        for _kind, _var, new_kind, state_id in DECL.findall(stmt):
            states[state_id] = {"kind": _kind_of(new_kind), "move": None}
            if statement_must_once:
                must_once.add(state_id)
            if declared:
                var_to_state.setdefault(declared.group(2), state_id)
            elif prop_assigned and len(DECL.findall(stmt)) == 1:
                var_to_state.setdefault(prop_assigned.group(1), state_id)
            if new_kind == "MoveState":
                args = _ctor_args(stmt, state_id)
                if args:
                    move_name = args[0]
                    states[state_id]["move"] = move_name
                    moves.setdefault(move_name, {})
                    if len(args) >= 2:
                        intent_match = re.match(r"new\s+(\w+Intent)\s*\((.*)\)\s*$",
                                                args[1], re.S)
                        if intent_match:
                            moves[move_name]["intent"] = intent_match.group(1)
                            moves[move_name]["intent_args"] = _split_args(
                                intent_match.group(2))

        # --- 局部变量别名：`MoveState initialState = (_flag ? a : b);` ---
        alias = VAR_ALIAS.match(stmt)
        if alias and "new " not in alias.group(2):
            aliases[alias.group(1)] = alias.group(2)

        # --- 初始状态 -------------------------------------------------
        # ⚠️ 不能用 `([^)]+)` 抓参数：真机会写成 `(MoveState)moveState3`，
        # 正则会在转型的右括号处截断，于是初始状态解析失败（实测少算了 19 只怪）。
        init_match = RETURN_MACHINE.search(stmt)
        if init_match:
            init_args = _split_args(_balanced_args(stmt, init_match.end() - 1))
            if len(init_args) >= 2:
                raw = init_args[1].strip()
                resolved = _resolve_state(raw, var_to_state, aliases)
                condition = _enclosing_condition(stmt_list[:index], stmt[:init_match.start()])
                if condition and resolved:
                    extra_returns.append({"if": condition, "initial": resolved})
                elif resolved and initial is None:
                    initial, initial_expr = resolved, raw
                elif resolved is None and initial_expr is None:
                    initial_expr = raw

        # --- FollowUpState 赋值 ---------------------------------------
        if ".FollowUpState" in stmt:
            target = _assign_target(stmt, var_to_state)
            if target:
                for lhs in re.findall(r"(\w+)\.FollowUpState\s*=", stmt):
                    if lhs in var_to_state:
                        # ⚠️ **不要**加 `!= target` 的去重判断：
                        # 单招怪写的是 `moveState.FollowUpState = moveState;`（自环），
                        # 去重会把自环整条丢掉，运行时变成"没有后继状态"直接抛异常。
                        follow_up[var_to_state[lhs]] = target

        # --- 分支 -----------------------------------------------------
        # 只处理 AddBranch；AddState（条件分支）统一交给 extract_conditionals，
        # 否则同一条件会被记两遍（一遍带 condition、一遍不带）。
        for match in re.finditer(r"(\w+)\.(AddBranch)\s*\(", stmt):
            receiver = match.group(1)
            if receiver not in var_to_state:
                continue                      # 动画器的 AddBranch，跳过
            args = _split_args(_balanced_args(stmt, match.end() - 1))
            if not args:
                continue
            state_id = var_to_state[receiver]
            entry = _branch_entry(args, var_to_state)
            if entry is None:
                continue
            if match.group(2) == "AddBranch":
                branches.setdefault(state_id, []).append(entry)
            else:
                conditionals.setdefault(state_id, []).append(entry)

    if not states:
        return None

    # --- 自爆招式 -------------------------------------------------------
    # 真机在招式处理函数结尾写 `await CreatureCmd.Kill(base.Creature)`：打完
    # 自己就死，所以**永远不会请求后继状态**。全库只有 `gas_bomb` 与
    # `waterfall_giant` 的 `EXPLODE_MOVE` 是这样，但不标记的话模拟器下一回合
    # 会去 roll 后继并抛"没有后继状态"，直接炸掉 Run 层（实测就是如此）。
    for handler in moves:
        body = method_body(source, handler)
        if body and "CreatureCmd.Kill(base.Creature)" in body:
            moves[handler]["kills_self"] = True
    if initial is None and extra_returns:
        initial = extra_returns[0]["initial"]
        extra_returns = extra_returns[1:]
    if initial is None and initial_expr:
        # 条件初始状态：取表达式里出现的第一个状态
        initial = next((s for s in re.findall(r"\w+", initial_expr) if s in states), None)
    if initial is None:
        return None
    return {"states": states, "follow_up": follow_up, "branches": branches,
            "conditionals": conditionals, "moves": moves, "initial": initial,
            "initial_expr": initial_expr, "conditional_initials": extra_returns,
            "must_perform_once": sorted(must_once), "_vars": var_to_state}


def _snake(name: str) -> str:
    """``BowlbugRock`` → ``bowlbug_rock``（与怪物表的 id 对齐）。"""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _kind_of(kind: str) -> str:
    return {"MoveState": "move", "RandomBranchState": "random",
            "ConditionalBranchState": "conditional"}[kind]


def _ctor_args(stmt: str, state_id: str) -> list[str]:
    """取 ``new MoveState("ID", handler, intent…)`` 里 "ID" 之后的参数。

    ⚠️ 不能"从 state id 往后找第一个 ``(``"——那个括号属于**意图**而不是构造函数，
    截出来的会是意图的实参（实测所有 MoveState 的意图都因此丢失）。
    正确做法是从 state id 往后扫，砍在深度首次变负的位置（即构造函数的右括号）。
    """
    marker = f'"{state_id}"'
    index = stmt.find(marker)
    if index < 0:
        return []
    rest = stmt[index + len(marker):].lstrip()
    if rest.startswith(","):
        rest = rest[1:]
    depth, in_str, cut = 0, False, len(rest)
    for position, char in enumerate(rest):
        if char == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                cut = position
                break
            depth -= 1
    return _split_args(rest[:cut])


def _balanced_args(text: str, open_index: int) -> str:
    """从 ``(`` 开始取配平的括号内容。"""
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:index]
    return ""


def _split_args(text: str) -> list[str]:
    """按顶层逗号切参数（忽略括号/引号内的逗号）。"""
    out, depth, current, in_str = [], 0, [], False
    for char in text:
        if char == '"':
            in_str = not in_str
        if not in_str:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                out.append("".join(current).strip())
                current = []
                continue
        current.append(char)
    if current:
        out.append("".join(current).strip())
    return [a for a in out if a]


def _enclosing_condition(previous: list[str], current_prefix: str = "") -> str | None:
    """判断某个 ``return`` 是否包在 ``if (...)`` 里（Axebot 这类多返回的怪）。

    按分号切语句会丢掉块结构，所以往回扫未配平的 ``{``，从那里取 ``if`` 条件。
    """
    text = current_prefix
    candidates = [text] + list(reversed(previous))
    depth = text.count("{") - text.count("}")
    for chunk in candidates:
        match = re.search(r"if\s*\(([^)]*)\)\s*\{[^{}]*$", chunk)
        if match and depth > 0:
            return match.group(1).strip()
        depth += chunk.count("{") - chunk.count("}")
        if depth <= 0:
            break
    return None


def _resolve_state(expr: str, var_to_state: dict[str, str],
                   aliases: dict[str, str] | None = None, depth: int = 0) -> str | None:
    """把表达式解析成状态 id。支持带转型、以及条件表达式里的**第一个**候选状态。

    条件初始状态（``_flag ? a : b``）无法在静态解析时定夺——真机的标志位由
    ``AfterAddedToRoom`` 之类的逻辑设置。这里取第一个候选，并把原始表达式一并留存，
    交给引擎按怪物特例处理。
    """
    if depth > 4:
        return None
    expr = expr.strip()
    for token in re.findall(r"\w+", expr):
        if token in var_to_state:
            return var_to_state[token]
    if aliases:
        for token in re.findall(r"\w+", expr):
            if token in aliases:
                return _resolve_state(aliases[token], var_to_state, aliases, depth + 1)
    return None


def _assign_target(stmt: str, var_to_state: dict[str, str]) -> str | None:
    """链式赋值 ``a.F = (T)(b.F = (c.F = new XState("ID")))`` 的目标状态。

    关键：整条语句里所有 ``.FollowUpState =`` 的左值都指向**同一个**新建状态。
    """
    new_match = re.search(
        r"new\s+(MoveState|RandomBranchState|ConditionalBranchState)\s*\(\s*\"([^\"]+)\"", stmt)
    if new_match:
        return new_match.group(2)
    rhs = stmt.split("=", 1)[-1]
    return _resolve_state(rhs, var_to_state)


def _branch_entry(args: list[str], var_to_state: dict[str, str]) -> dict | None:
    target = _resolve_state(args[0], var_to_state)
    if target is None:
        return None
    entry: dict = {"state": target}
    if len(args) >= 2:
        second = args[1].strip()
        repeat_match = re.search(r"MoveRepeatType\.(\w+)", second)
        if repeat_match:
            entry["repeat"] = repeat_match.group(1)
        elif re.fullmatch(r"\d+", second):
            # AddBranch(state, maxTimes) 重载 → 可重复 N 次
            entry["repeat"] = "CanRepeatXTimes"
            entry["max_times"] = int(second)
            return entry
    if len(args) >= 3:
        weight = re.fullmatch(r"([\d.]+)f?", args[2].strip())
        if weight:
            entry["weight"] = float(weight.group(1))
    if len(args) >= 4 and re.fullmatch(r"\d+", args[3].strip()):
        entry["max_times"] = int(args[3])
    for arg in args[1:]:
        named = re.match(r"(\w+)\s*:\s*(.+)", arg)
        if named and named.group(1) in ("weight", "maxTimes", "repeat"):
            key = {"maxTimes": "max_times"}.get(named.group(1), named.group(1))
            value = named.group(2).strip()
            entry[key] = (re.search(r"MoveRepeatType\.(\w+)", value).group(1)
                          if key == "repeat" and "MoveRepeatType" in value
                          else (float(value.rstrip("f")) if re.fullmatch(r"[\d.]+f?", value)
                                else value))
    if "condition" not in entry:
        cond = re.search(r"condition\s*:\s*(.+)", " ".join(args[1:]))
        if cond:
            entry["condition"] = cond.group(1).strip()
    return entry


def extract_conditionals(source: str, machine: dict) -> None:
    """``AddState(state, () => Expr)`` 的条件在 lambda 里，单独扫一遍。"""
    body = method_body(source, "GenerateMoveStateMachine") or ""
    var_to_state = machine.get("_vars") or {}
    for match in re.finditer(r"(\w+)\.AddState\s*\(", body):
        if match.group(1) not in var_to_state:
            continue
        args = _split_args(_balanced_args(body, match.end() - 1))
        if len(args) < 2:
            continue
        target = _resolve_state(args[0], var_to_state)
        if not target:
            continue
        condition = re.sub(r"^\(\)\s*=>\s*", "", args[1]).strip()
        machine["conditionals"].setdefault(var_to_state[match.group(1)], []).append(
            {"state": target, "condition": condition})


# ==========================================================================
def extract_monster(path: pathlib.Path, report: Counter,
                    inherited: dict[str, dict] | None = None,
                    class_sources: dict[str, str] | None = None) -> dict | None:
    source = path.read_text(encoding="utf-8", errors="replace")
    machine = extract_machine(source)
    if machine is None:
        # 有些怪（DecimillipedeSegment*、MysteriousKnight）不重写状态机，直接继承基类
        base = re.search(r"class\s+\w+\s*:\s*(\w+)", source)
        if base and inherited and base.group(1).lower() in inherited:
            report["inherited_state_machine"] += 1
            machine = dict(inherited[base.group(1).lower()])
        else:
            report["no_state_machine"] += 1
            return None
    else:
        extract_conditionals(source, machine)

    if class_sources:
        consts = inherited_constants(path.stem, class_sources)
    else:
        consts = collect_constants(source)
        class_sources = {path.stem: source}

    # HP 必须沿继承链找，并且**报告残余项**（见 unconsumed_terms 的说明）。
    hp = hp_max = None
    hp_match = prop_expr("MinInitialHp", path.stem, class_sources)
    hp_max_match = prop_expr("MaxInitialHp", path.stem, class_sources)
    if hp_match:
        hp = _resolve_expr(hp_match[1], consts)
        leftover = unconsumed_terms(hp_match[1])
        if leftover:
            report[f"hp_expr_with_extra_terms::{leftover[:24]}"] += 1
    if hp_max_match:
        hp_max = _resolve_expr(hp_max_match[1], consts)
        leftover = unconsumed_terms(hp_max_match[1])
        if leftover:
            report[f"hp_max_expr_with_extra_terms::{leftover[:24]}"] += 1
    if hp is None:
        report["hp_unresolved"] += 1
    elif hp_max is None:
        report["hp_max_unresolved"] += 1

    unresolved = [name for name, value in machine["moves"].items()
                  for arg in value.get("intent_args", []) if _int_of(arg, consts) is None]
    if unresolved:
        report["moves_with_unresolved_intent_arg"] += len(unresolved)
    for state in machine["states"].values():
        if state["kind"] == "move" and not state["move"]:
            report["move_state_without_handler"] += 1
    if machine["initial"] not in machine["states"]:
        report["initial_state_missing"] += 1
    for src_state, dst in machine["follow_up"].items():
        if dst not in machine["states"]:
            report["dangling_follow_up"] += 1
    for entries in list(machine["branches"].values()) + list(machine["conditionals"].values()):
        for entry in entries:
            if entry["state"] not in machine["states"]:
                report["dangling_branch"] += 1

    # --- 进阶感知的数值表 ---------------------------------------------
    # 真机写 `AscensionHelper.GetValueIfAscension(DeadlyEnemies, 16, 14)`：
    # 进阶 ≥9 时伤害是 16，否则 14。只存普通值会让 A9+ 的数值全错。
    move_values: dict[str, dict] = {}
    for handler, info in machine["moves"].items():
        args = info.get("intent_args") or []
        entry: dict = {"intent": info.get("intent")}
        if args:
            resolved = _ascension_of(args[0], consts)
            if resolved:
                entry.update(resolved)
            if len(args) >= 2 and re.fullmatch(r"\d+", args[1].strip()):
                entry["times"] = int(args[1])
        if len(entry) > 1:
            move_values[handler] = entry

    hp_range = None
    if hp_match:
        resolved = _ascension_of(hp_match[1], consts)
        if resolved:
            hp_range = resolved

    return {
        # ⚠️ eid 必须与怪物表（codex JSON 的 id）对齐：类名是驼峰，id 是下划线。
        # 直接用 `path.stem.lower()` 会让 `BowlbugRock` → `bowlbugrock` 对不上
        # `bowlbug_rock`，实测 114 只怪只能匹配上 30 只，而且**不报错**。
        "eid": _snake(path.stem),
        "class": path.stem,
        "hp": [hp, hp_max] if hp is not None else None,
        "initial": machine["initial"],
        "initial_expr": machine.get("initial_expr"),
        "conditional_initials": machine.get("conditional_initials") or [],
        "states": machine["states"],
        "follow_up": machine["follow_up"],
        "branches": machine["branches"],
        "conditionals": machine["conditionals"],
        "moves": machine["moves"],
        "move_values": move_values,
        "hp_range": hp_range,
        "constants": consts,
        "must_perform_once": machine.get("must_perform_once") or [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从反编译源码提取敌人出招状态机")
    parser.add_argument("--report", action="store_true", help="只打印报告，不写文件")
    args = parser.parse_args(argv)

    if not DECOMPILED.is_dir():
        print(f"找不到反编译源码：{DECOMPILED}")
        print("先运行反编译（见 docs/09 的工具链说明）")
        return 3

    report: Counter = Counter()
    kkinds: Counter = Counter()
    extracted = []
    # 先按类名顺序扫一遍，把已提取的状态机留作**继承兜底**：
    # DecimillipedeSegment* 与 MysteriousKnight 不重写 GenerateMoveStateMachine，
    # 直接用基类的（这正是 codex JSON 里那 4 个"无 attack_pattern"的怪）。
    inherited: dict[str, dict] = {}
    class_sources = {p.stem: p.read_text(encoding="utf-8", errors="replace")
                     for p in DECOMPILED.glob("*.cs")}
    for path in sorted(DECOMPILED.glob("*.cs")):
        if path.stem == "MonsterModel":
            continue
        report["files"] += 1
        data = extract_monster(path, report, inherited, class_sources)
        if data:
            extracted.append(data)
            report["extracted"] += 1
            for state in data["states"].values():
                kkinds[state["kind"]] += 1
            inherited[path.stem.lower()] = {
                "states": data["states"], "follow_up": data["follow_up"],
                "branches": data["branches"], "conditionals": data["conditionals"],
                "moves": data["moves"], "initial": data["initial"],
                "initial_expr": data.get("initial_expr"),
                "conditional_initials": data.get("conditional_initials") or [],
            }

    print(f"扫描 {report['files']} 个怪物类，提取出状态机 {report['extracted']} 个")
    print(f"状态类型分布：{dict(kkinds)}")
    problems = {k: v for k, v in report.items()
                if k not in ("files", "extracted") and v}
    if problems:
        print("提取质量问题：")
        for key, count in sorted(problems.items()):
            print(f"  {count:4d}  {key}")
    else:
        print("无提取质量问题 ✅")

    if not args.report:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(extracted, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"\n写入 {OUT_PATH}（{len(extracted)} 只怪）")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
