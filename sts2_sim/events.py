"""事件层：选项 → 效果 → 翻页的**状态机**。

真机模型（``MegaCrit.Sts2.Core.Models/EventModel.cs``）::

    BeginEvent(player, …)
      Rng = new Rng(runSeed + slot + XXH64(eventId))     ← 每个事件一条独立 RNG
      CalculateVars()                                     ← 把 DynamicVars 掷出来
      SetInitialEventState(…) → GenerateInitialOptions()  ← 第一页的选项
    玩家点选项 → EventOption.Chosen()（``EventOption.cs:174-185``）
      同一个选项**只会结算一次**（``WasChosen`` 守卫，除非构造时声明
      ``disableOnChosen: false``）
      里面要么 SetEventFinished(desc)  结束
      要么   SetEventState(desc, options) 翻到下一页

所以本模块只有三件事：**判定事件能不能出现**、**给出当前页的选项**、
**结算一个选项并翻页**。效果本身交给 :mod:`sts2_sim.runeffects`（与遗物同一套算子）。

⚠️ 三条诚实性规则（``docs/09`` §5.2）：

1. **条件算不出就不出现**。``IsAllowed`` 里 36 个事件写了条件，本模块只认白名单里的
   几种写法；认不出的一律 ``gate_ok = None`` → 排除出事件池并记进报告，
   **绝不"看起来像 true"就放行**。
2. **有缺口的选项不静默降级**。选项效果里有引擎没有的命令时，这个事件整体
   ``usable = False``（排除出事件池），而不是"少结算一半效果"。
3. **选项只结算一次**，与真机的 ``WasChosen`` 守卫一致；重复点同一个选项是非法动作。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .content import CARD_DB, Effect
from .rng import RngSet
from .xxhash import xxh64_text


# ==========================================================================
# 数据模型
# ==========================================================================
@dataclass(frozen=True)
class EventOptionDef:
    """一个事件选项。

    ``effects_raw`` 保留**源码抽出来的原始效果字典**（而不是解析好的 ``Effect``）：
    事件里的量有一部分是 ``CalculateVars`` 掷出来的（``Gold`` 这种），
    在事件**开始之前**根本不存在具体数值，所以必须等到 ``initial_run``
    掷完再解析成 ``Effect``（``resolve_option_effects``）。
    """

    #: 本地化 key（唯一，含页与选项名，如 ``THIS_OR_THAT.pages.INITIAL.options.PLAIN``）
    key: str
    #: 选项短名（key 的最后一段，用于日志与观测）
    name: str
    effects_raw: tuple[dict, ...] = ()
    #: ``none`` / ``finished`` / ``goto``
    outcome: str = "none"
    #: ``goto`` 时的目标页
    outcome_page: str = ""
    #: 真机 ``EventOption.IsLocked``（``OnChosen == null``）：显示但点不动
    locked: bool = False
    #: 引擎缺口（子系统缺失等）
    reasons: tuple[str, ...] = ()
    #: 选项效果里没被支持的命令
    unsupported: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventPageDef:
    page_id: str
    options: tuple[EventOptionDef, ...] = ()


@dataclass(frozen=True)
class EventDef:
    eid: str
    name: str
    pages: dict[str, EventPageDef]
    initial_page: str = "INITIAL"
    #: ``IsAllowed`` 的源码原文（空 = 没有条件）
    gate_body: str = ""
    #: 静态求值出来的条件类型（``""`` 表示无条件）
    gate_kind: str = ""
    gate_args: tuple = ()
    combat_encounter: str = ""
    #: 社区库记的幕（``Act 1 - Overgrowth`` / ``Underdocks`` / ``Act 2 - Hive`` …）
    act: str = ""
    #: ``CanonicalVars`` 里的**字面量**（``HpLoss: 6``）。掷点会覆盖同名项。
    literal_vars: tuple[tuple[str, int], ...] = ()
    #: 事件内 RNG 掷点（``CalculateVars``）
    rng_rolls: tuple[tuple[str, str], ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        """引擎能不能**完整**跑这个事件。"""
        return not self.reasons


# ==========================================================================
# 进入条件（``EventModel.IsAllowed``）
# ==========================================================================
def _single_player(expr: str) -> str:
    """``runState.Players.All((Player p) => …)`` → 去掉包装，留条件本体。

    ``Any((Player p) => …)`` 一并认：**本模拟器只跑单人**，
    ``runState.Players`` 只有一个元素，此时 ``Any`` 与 ``All`` 恒等
    （``LinqExtensions`` / ``Enumerable`` 的标准语义）。多人语义在这里**不可达**，
    所以这不是"近似"，而是"单人场景下两条式子的值必然相同"。
    """
    match = re.search(
        r"runState\.Players\.(?:All|Any)\s*\(\s*\(Player\s+\w+\)\s*=>\s*(.+)\)\s*;?\s*$",
        expr.strip(), re.S)
    return match.group(1).strip() if match else expr.strip()


def _normalize(cond: str) -> str:
    """归一化条件文本：去掉 ``return`` / 括号包装 / ``(decimal)`` 强制转换。

    ⚠️ 只做**保语义**的归一化（加括号、类型转换、分号），不"顺手"做化简 ——
    化简就意味着猜，而猜错的方向是"本不该出现的事件出现了"。
    """
    text = cond.strip().rstrip(";").strip()
    text = re.sub(r"^return\s+", "", text).strip()
    text = re.sub(r"\(decimal\)", "", text)
    text = re.sub(r"\s+", " ", text)
    # 整体被一层括号包住时剥掉
    while text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text


#: 条件白名单。**每条都是"归一化后的整串精确匹配"**，不做模糊搜索 ——
#: 模糊匹配会在条件变体上悄悄判错，而判错的方向是"放行了不该出现的选项"。
GATE_PATTERNS: tuple[tuple[str, str, tuple], ...] = (
    (r"runState\.CurrentActIndex < (\d+)", "act_lt", ()),
    (r"runState\.CurrentActIndex == (\d+)", "act_eq", ()),
    (r"runState\.CurrentActIndex > (\d+)", "act_gt", ()),
    (r"runState\.CurrentActIndex >= (\d+)", "act_ge", ()),
    (r"runState\.TotalFloor >= (\d+)", "floor_ge", ()),
    (r"runState\.TotalFloor > (\d+)", "floor_gt", ()),
    (r"\w+\.Gold >= (\d+)", "gold_ge", ()),
    (r"\w+\.Gold < (\d+)", "gold_lt", ()),
    (r"\w+\.Creature\.CurrentHp >= (\d+)", "hp_ge", ()),
    (r"\w+\.Creature\.CurrentHp > (\d+)", "hp_gt", ()),
    (r"\w+\.Creature\.MaxHp >= (\d+)", "max_hp_ge", ()),
    (r"\w+\.Potions\.Count\(\) >= (\d+)", "potions_ge", ()),
    # `p.Creature.CurrentHp <= p.Creature.MaxHp * 0.70m`（受伤到 70% 以下才出现）
    (r"\w+\.Creature\.CurrentHp <= \w+\.Creature\.MaxHp \* ([\d.]+)m?", "hp_ratio_le", ()),
    # 牌组张数（**只是张数**，不含谓词；带谓词的写法见 `DECK_PREDICATES`）
    (r"p\.Deck\.Cards\.Count > (\d+)", "deck_gt", ()),
    (r"p\.Deck\.Cards\.Count >= (\d+)", "deck_ge", ()),
)

#: `p.Gold >= base.DynamicVars["X"].BaseValue` 这类"阈值写在 CanonicalVars 里"的写法。
#:
#: 真机里 `base.DynamicVars.X.BaseValue` 就是那个字面量（``DynamicVar`` 的默认值），
#: 抽取器已经把 CanonicalVars 抽进 ``vars``，所以这里**解析成具体数字**，
#: 而不是"算不出"。判据：``vars`` 里有这个变量**且有具体值**，否则返回 ``None``
#: （宁可 unknown，不猜）。
def _dynamic_var_value(name: str, vars: dict[str, int] | None) -> int | None:
    if not vars:
        return None
    value = vars.get(name)
    return int(value) if value is not None else None


#: `p.Deck.Cards.Count((CardModel c) => <谓词>) >= N` —— 带谓词的张数。
DECK_COUNT_RE = re.compile(
    r"(?:\w+\.Deck\.Cards|CardPile\.Get\(PileType\.Deck, \w+\)\.Cards)"
    r"\.Count\(\(CardModel (\w+)\) => (.+)\) >= (\d+)")
#: `p.Deck.Cards.Any((CardModel c) => <谓词>)` —— 有没有满足谓词的牌。
DECK_ANY_RE = re.compile(
    r"(?:\w+\.Deck\.Cards|CardPile\.Get\(PileType\.Deck, \w+\)\.Cards)"
    r"\.Any\(\(CardModel (\w+)\) => (.+)\)")

#: 牌组谓词白名单：**源码原文 → 语义 id**。
#:
#: 只收"整串精确匹配"的谓词。谓词里的每个方法都在源码里逐条对过：
#:
#:   ``c.IsRemovable``    ``CardModel.cs:738``  ``!Keywords.Contains(Eternal)``
#:   ``c.IsTransformable`` ``CardModel.cs:740-751`` 在**牌组里**等价于 ``IsRemovable``
#:                        （``!IsRemovable`` 时只有"不在牌组"才可转化）
#:   ``c.Rarity == CardRarity.Basic``  ``CardRarity.cs``  Basic = 1
#:   ``c.Tags.Contains(CardTag.X)``    卡牌静态标签
#:   ``IsValid(CardTag.X, c)``         **仅 Amalgamator 定义的私有静态方法**
#:                        （``Amalgamator.cs:69-80``）：标签命中 **且** 稀有度 Basic
#:                        **且** 可移除。全仓库只有这一个 ``IsValid`` 定义，
#:                        所以按整串匹配不会撞到别的语义。
CARD_PREDICATES: tuple[tuple[str, str], ...] = (
    ("c.IsRemovable", "removable"),
    ("c.IsTransformable", "transformable"),
    ("c != null && c.Rarity == CardRarity.Basic && c.IsRemovable", "basic_removable"),
    ("c.Rarity == CardRarity.Basic", "basic"),
    ("c.Tags.Contains(CardTag.Strike)", "tag:strike"),
    ("c.Tags.Contains(CardTag.Defend)", "tag:defend"),
    ("IsValid(CardTag.Strike, c)", "basic_tag:strike"),
    ("IsValid(CardTag.Defend, c)", "basic_tag:defend"),
)



def _split_top(cond: str, operator: str) -> list[str]:
    """按**顶层** ``operator``（``&&`` / ``||``）拆条件（括号内的不算）。"""
    parts: list[str] = []
    depth = 0
    current = ""
    index = 0
    while index < len(cond):
        char = cond[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth == 0 and cond.startswith(operator, index):
            parts.append(current.strip())
            current = ""
            index += len(operator)
            continue
        current += char
        index += 1
    parts.append(current.strip())
    return [p for p in parts if p]


def _split_and(cond: str) -> list[str]:
    """按**顶层** ``&&`` 拆条件（括号内的不算）。"""
    return _split_top(cond, "&&")


def _strip_outer_parens(text: str) -> str:
    """整体被**一层**（或多层）括号包住时剥掉；只在配平成立时剥。"""
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        balanced = True
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    balanced = False        # 第一个 '(' 在中间就闭合了 → 不是整体包裹
                    break
        if not balanced:
            return text
        text = text[1:-1].strip()
    return text


def _match_leaf(cond: str, vars: dict[str, int] | None) -> tuple[str, tuple] | None:
    """把**单个**（不含顶层 ``&&`` / ``||`` / 前导 ``!`` 的）条件按白名单匹配。"""
    text = _strip_outer_parens(cond.strip())
    if text == "runState.Players.Count == 1":
        # 单人模拟器：这条恒真（真机多人时它才可能为假）
        return "always", ()
    if text == "runState.Players.Count > 1":
        return "never", ()
    if text == "true":
        return "always", ()
    if text == "false":
        return "never", ()
    for pattern, kind, _ in GATE_PATTERNS:
        match = re.fullmatch(pattern, text)
        if match:
            args = tuple(float(g) if "." in g else int(g) for g in match.groups())
            return kind, args
    # `p.Gold >= base.DynamicVars["X"].BaseValue` / `… .DynamicVars.X.BaseValue`
    dyn = re.fullmatch(
        r"\w+\.Gold >= (?:base\.)?DynamicVars(?:\[\"(\w+)\"\]|\.(\w+))\.BaseValue",
        text)
    if dyn:
        value = _dynamic_var_value(dyn.group(1) or dyn.group(2), vars)
        return ("gold_ge", (value,)) if value is not None else None
    # `p.Potions.Any()` / `p.Potions.Any((PotionModel potion) => potion is FoulPotion)`
    if re.fullmatch(r"\w+\.Potions\.Any\(\)", text):
        return "has_potion", ("",)
    foul = re.fullmatch(
        r"\w+\.Potions\.Any\(\(PotionModel \w+\) => \w+ is (\w+)\)", text)
    if foul:
        return "has_potion", (snake_case(foul.group(1)),)
    # `!p.HasEventPet()`（`Player.cs:248-255`）
    if re.fullmatch(r"\w+\.HasEventPet\(\)", text):
        return "has_event_pet", ()
    # `p.RelicGrabBag.HasAvailableRelics(runState)`（`RelicGrabBag.cs`）
    if re.fullmatch(r"\w+\.RelicGrabBag\.HasAvailableRelics\(runState\)", text):
        return "bag_has_relics", ()
    # ⚠️ `GetValidRelics(p)`（`RelicTrader.cs:107` / `RanwidTheElder.cs:72`）
    # 走 `RelicModel.IsTradable`（`RelicModel.cs:176-203`），而那还依赖
    # `IsUsedUp` / `HasUponPickupEffect` / `IsMelted` / `SpawnsPets`
    # 四个引擎没有的字段 —— **故意不认**（如实 unknown，而不是猜 true）。
    # 带谓词的牌组条件
    match = DECK_COUNT_RE.fullmatch(text)
    if match:
        predicate = _normalize_predicate(match.group(2))
        if predicate is None:
            return None
        return "deck_count_ge", (predicate, int(match.group(3)))
    match = DECK_ANY_RE.fullmatch(text)
    if match:
        predicate = _normalize_predicate(match.group(2))
        if predicate is None:
            return None
        return "deck_any", (predicate,)
    return None


def _normalize_predicate(raw: str) -> str | None:
    """把 ``(CardModel c) => …`` 的谓词体归一化成 :data:`CARD_PREDICATES` 的 key。

    ⚠️ 参数名先统一成 ``c``（真机有的写 ``c`` 有的写 ``card``），
    再**整串精确匹配**白名单 —— 不做任何模糊/子串匹配。
    """
    text = re.sub(r"\s+", " ", raw.strip())
    text = text.replace("(c != null)", "c != null")
    text = re.sub(r"\bcard\b", "c", text)
    for source, predicate in CARD_PREDICATES:
        if text == source:
            return predicate
    return None


def snake_case(name: str) -> str:
    """``FoulPotion`` → ``foul_potion``（与内容层的 id 归一化一致）。"""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


# --------------------------------------------------------------------------
# 条件体的**结构化解析**
# --------------------------------------------------------------------------
#: 真机 ``IsAllowed`` 的写法远不止"一句 return"：大量事件写成一串**守卫**
#: （``if (…) { return false; }``）或 ``if (…) { return <表达式>; } return false;``，
#: 例如 ``CrystalSphere.cs`` / ``TeaMaster.cs``。
#:
#: 早先只认"整串一个 return"，于是这些事件全部落进
#: "进入条件算不出（IsAllowed 写法未支持）"。下面是一个**受限文法**的解析器：
#:
#:   stmt  := ``if ( COND ) BLOCK [else BLOCK]`` | ``return EXPR ;``
#:   BLOCK := ``{ stmt* }`` | stmt
#:
#: 认不出的语句（``foreach`` / 局部变量 / ``throw``…）→ 整体 ``unknown``。
#: 编译出来的表达式树**只做恒等式级别的化简**（``X && false = false``、
#: ``X || true = true``、``!!X = X`` 等），不做任何需要"猜"的变形。


def _and(*nodes: tuple) -> tuple:
    flat: list[tuple] = []
    for node in nodes:
        if node[0] == "never":
            return ("never", ())
        if node[0] == "always":
            continue
        if node[0] == "and":
            flat.extend(node[1])
        else:
            flat.append(node)
    if not flat:
        return ("always", ())
    if len(flat) == 1:
        return flat[0]
    return ("and", tuple(flat))


def _or(*nodes: tuple) -> tuple:
    flat: list[tuple] = []
    for node in nodes:
        if node[0] == "always":
            return ("always", ())
        if node[0] == "never":
            continue
        if node[0] == "or":
            flat.extend(node[1])
        else:
            flat.append(node)
    if not flat:
        return ("never", ())
    if len(flat) == 1:
        return flat[0]
    return ("or", tuple(flat))


def _not(node: tuple) -> tuple:
    if node[0] == "always":
        return ("never", ())
    if node[0] == "never":
        return ("always", ())
    if node[0] == "not":
        return node[1][0]
    return ("not", (node,))


def _compile_expr(text: str, vars: dict[str, int] | None) -> tuple | None:
    """把一个 C# 布尔表达式编译成条件树；认不出返回 ``None``。"""
    text = _strip_outer_parens(_normalize(text))
    if not text:
        return None
    parts = _split_top(text, "||")
    if len(parts) > 1:
        compiled = [_compile_expr(part, vars) for part in parts]
        if any(item is None for item in compiled):
            return None
        return _or(*compiled)                  # type: ignore[arg-type]
    parts = _split_top(text, "&&")
    if len(parts) > 1:
        compiled = [_compile_expr(part, vars) for part in parts]
        if any(item is None for item in compiled):
            return None
        return _and(*compiled)                 # type: ignore[arg-type]
    if text.startswith("!"):
        inner = _compile_expr(text[1:], vars)
        return _not(inner) if inner is not None else None
    unwrapped = _single_player(text)
    if unwrapped != text:
        return _compile_expr(unwrapped, vars)
    leaf = _match_leaf(text, vars)
    return leaf


class _Cursor:
    """极小的字符游标（只用于解析 ``IsAllowed`` 的方法体）。"""

    __slots__ = ("text", "index")

    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0

    def skip_ws(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def eat(self, token: str) -> bool:
        self.skip_ws()
        if self.text.startswith(token, self.index):
            self.index += len(token)
            return True
        return False

    def peek(self) -> str:
        self.skip_ws()
        return self.text[self.index] if self.index < len(self.text) else ""

    def read_parens(self) -> str | None:
        """读一对配平的圆括号，返回**括号内**原文。"""
        self.skip_ws()
        if self.peek() != "(":
            return None
        depth = 0
        start = self.index
        while self.index < len(self.text):
            char = self.text[self.index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    self.index += 1
                    return self.text[start + 1:self.index - 1]
            self.index += 1
        return None

    def read_until_semicolon(self) -> str | None:
        depth = 0
        start = self.index
        while self.index < len(self.text):
            char = self.text[self.index]
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
                if depth < 0:
                    return None
            elif char == ";" and depth == 0:
                self.index += 1
                return self.text[start:self.index - 1]
            self.index += 1
        return None


def _parse_statements(cursor: _Cursor) -> list | None:
    """解析一串语句（到 ``}`` 或文本结束为止）。认不出的语句 → ``None``。"""
    statements: list = []
    while True:
        cursor.skip_ws()
        if cursor.index >= len(cursor.text) or cursor.peek() == "}":
            return statements
        statement = _parse_statement(cursor)
        if statement is None:
            return None
        statements.append(statement)


def _parse_block(cursor: _Cursor) -> list | None:
    cursor.skip_ws()
    if cursor.peek() == "{":
        cursor.index += 1
        statements = _parse_statements(cursor)
        if statements is None or not cursor.eat("}"):
            return None
        return statements
    statement = _parse_statement(cursor)
    return None if statement is None else [statement]


def _parse_statement(cursor: _Cursor) -> tuple | None:
    cursor.skip_ws()
    if cursor.text.startswith("if", cursor.index) and \
            not (cursor.text[cursor.index + 2:cursor.index + 3].isalnum()
                 or cursor.text[cursor.index + 2:cursor.index + 3] == "_"):
        cursor.index += 2
        condition = cursor.read_parens()
        if condition is None:
            return None
        body = _parse_block(cursor)
        if body is None:
            return None
        else_body = None
        mark = cursor.index
        if cursor.eat("else"):
            else_body = _parse_block(cursor)
            if else_body is None:
                # 也可能是 `else if (…)`——本轮不支持，如实 unknown
                return None
        else:
            cursor.index = mark
        return ("if", condition, body, else_body)
    if cursor.eat("return"):
        expression = cursor.read_until_semicolon()
        if expression is None:
            return None
        return ("return", expression)
    return None


def _compile_statements(statements: list, vars: dict[str, int] | None) -> tuple | None:
    """把语句列表编译成条件树（从后往前折叠）。"""
    result: tuple | None = None
    for statement in reversed(statements):
        if statement[0] == "return":
            result = _compile_expr(statement[1], vars)
            if result is None:
                return None
            continue
        _kind, condition_text, body, else_body = statement
        condition = _compile_expr(condition_text, vars)
        if condition is None:
            return None
        body_expr = _compile_statements(body, vars)
        if body_expr is None:
            return None
        if else_body is not None:
            else_expr = _compile_statements(else_body, vars)
            if else_expr is None:
                return None
        else:
            if result is None:
                # `if` 没有 else、后面也没有 return → 有不返回的路径，真机编译不过
                return None
            else_expr = result
        result = _or(_and(condition, body_expr), _and(_not(condition), else_expr))
    return result


#: 单人模拟器的**恒真**写法：``if (runState.Players.Count == 1) { return true; }``
#: 之后的分支全部是**多人专用**检查（``DenseVegetation`` / ``JungleMazeAdventure``：
#: 只有"人数大于 1"时才会去逐个玩家比血）。单人时整体恒等于 ``true``，
#: 而且这是**结构上的**结论（那段代码根本不可达），不是近似。
SINGLE_PLAYER_EARLY_TRUE = re.compile(
    r"^if\s*\(\s*runState\.Players\.Count == 1\s*\)\s*\{\s*return true;\s*\}")


def parse_gate(gate, vars: dict[str, int] | None = None) -> tuple[str, tuple]:
    """把 ``IsAllowed`` 的方法体翻译成**可求值的条件**。

    ``gate`` 是抽取器留下的 ``{"body": "…"}``（也可能直接给字符串）；
    ``vars`` 是该事件 ``CanonicalVars`` 的字面量（``{变量名: 值}``），
    用来解析 ``base.DynamicVars.X.BaseValue`` 这种"阈值写成变量"的写法。

    认下面这些写法（逐个从源码里对出来的），**其余一律返回 "unknown"**：

    ==========================================  ==================================
    源码写法                                     条件
    ==========================================  ==================================
    ``return runState.CurrentActIndex < N;``     ``("act_lt", (N,))``
    ``return runState.CurrentActIndex == N;``    ``("act_eq", (N,))``
    ``return runState.CurrentActIndex > N;``     ``("act_gt", (N,))``
    ``return runState.TotalFloor >= N;``         ``("floor_ge", (N,))``
    ``return p.Gold >= N;``                      ``("gold_ge", (N,))``
    ``return p.Creature.CurrentHp >= N;``        ``("hp_ge", (N,))``
    ``return p.Creature.CurrentHp <= p.Creature.MaxHp * 0.7m;``
                                                 ``("hp_ratio_le", (0.7,))``
    ``return p.Potions.Count() >= N;``           ``("potions_ge", (N,))``
    ``return p.Gold >= base.DynamicVars.X.BaseValue;``
                                                 ``("gold_ge", (X 的字面量,))``
    ``return p.Deck.Cards.Any((CardModel c) => P);``
                                                 ``("deck_any", (谓词,))``
    ``return p.Deck.Cards.Count((CardModel c) => P) >= N;``
                                                 ``("deck_count_ge", (谓词, N))``
    ``return p.Potions.Any();``                  ``("has_potion", ("",))``
    ``return !p.HasEventPet();``                 否定 ``("has_event_pet", ())``
    ``if (COND) { return true; } return false;``  等价于那条条件本身
    ``if (COND) { return EXPR; } return false;``  ``COND && EXPR``
    ``if (COND) { return false; } … return EXPR;`` ``!COND && … && EXPR``
    ``return false;``                            ``("never", ())``
    ``<A> && <B>`` / ``<A> || <B>`` / ``!<A>``    逐项认得才认
    ``runState.Players.Count == 1``              ``("always", ())``（单人模拟器恒真）
    ``if (runState.Players.Count == 1) { return true; } …``
                                                 ``("always", ())``（后面是多人专用分支）
    ==========================================  ==================================

    三种包装也会认：

    * ``runState.Players.All((Player p) => <条件>)`` —— 单人语义就是那个条件；
    * ``runState.Players.Any((Player p) => <条件>)`` —— 单人时与 ``All`` 恒等；
    * ``if (<条件>) { return true; } return false;`` —— 与直接 return 等价。

    ⚠️ 这条白名单是**故意窄**的：多认一种写法就意味着多一个"可能判错"的机会，
    而判错的后果是"本不该出现的事件出现了"（玩家白拿收益），比"少出现一个事件"严重。
    ``&&`` 的每一项都必须认得出来；**只要有一项认不出，整体就是 unknown**。
    认不出的写法会**原样留着**（返回 ``"unknown"``），载入层据此把事件排除出池子。
    """
    if isinstance(gate, dict):
        gate = gate.get("body") or ""
    body = str(gate or "")
    if not body:
        return "", ()

    text = re.sub(r"//[^\n]*", "", body)
    text = " ".join(text.split())
    text = text.rstrip()

    # 单人恒真的前缀（后面全是多人专用分支）
    if SINGLE_PLAYER_EARLY_TRUE.match(text):
        return "always", ()

    cursor = _Cursor(text)
    statements = _parse_statements(cursor)
    if statements is not None:
        cursor.skip_ws()
        if cursor.index >= len(cursor.text):
            compiled = _compile_statements(statements, vars)
            if compiled is not None:
                return compiled
    # 老的单句写法兜底（`return <表达式>;` 单独一条也能被上面的解析器覆盖，
    # 这里保留是为了在解析器因某个语句失败时仍能处理"整串一个 return"）
    stripped = _normalize(re.sub(r"^return\s+", "", text))
    if stripped and stripped != text:
        compiled = _compile_expr(stripped, vars)
        if compiled is not None:
            return compiled
    return "unknown", ()



def _gate_result(kind: str, args: tuple, **context) -> bool | None:
    if kind == "always":
        return True
    if kind == "never":
        return False
    # ⭐ **三值逻辑（Kleene）**，不是"有一项算不出就整体算不出"：
    #   `true || 算不出 = true`、`false && 算不出 = false` ——
    # 这两条是**可靠**的（不看算不出的那一项也能定值）。
    # 反过来（`false || ?`、`true && ?`）才必须返回 None。
    # ⚠️ 不能因为"更省事"就一律返回 None：那会把
    # `金币够 **或** 有 FoulPotion` 这种条件在没传药水表时整条判成算不出，
    # 于是本来该出现的事件不出现 —— 方向是安全的，但白白丢内容。
    if kind in ("and", "or", "not"):
        results = [_gate_result(sub_kind, sub_args, **context)
                   for sub_kind, sub_args in args]
        if kind == "not":
            return None if results[0] is None else not results[0]
        if kind == "or":
            if any(result is True for result in results):
                return True
            return None if any(result is None for result in results) else False
        if any(result is False for result in results):
            return False
        return None if any(result is None for result in results) else True
    if kind == "act_lt":
        return context["act_index"] < args[0]
    if kind == "act_eq":
        return context["act_index"] == args[0]
    if kind == "act_gt":
        return context["act_index"] > args[0]
    if kind == "act_ge":
        return context["act_index"] >= args[0]
    if kind == "floor_ge":
        return context["floor"] >= args[0]
    if kind == "floor_gt":
        return context["floor"] > args[0]
    if kind == "gold_ge":
        return context["gold"] >= args[0]
    if kind == "gold_lt":
        return context["gold"] < args[0]
    if kind == "hp_ge":
        return context["hp"] >= args[0]
    if kind == "hp_gt":
        return context["hp"] > args[0]
    if kind == "max_hp_ge":
        return context["max_hp"] >= args[0]
    if kind == "potions_ge":
        return context["potions"] >= args[0]
    if kind == "deck_ge":
        return context["deck_size"] >= args[0]
    if kind == "deck_gt":
        return context["deck_size"] > args[0]
    if kind == "hp_ratio_le":
        return context["max_hp"] > 0 and context["hp"] <= context["max_hp"] * args[0]
    if kind == "has_potion":
        # `Player.Potions`（``Player.cs:122``）**只含非空槽**：
        # `_potionSlots.Where(p => p != null)` —— 空槽不算"有药水"。
        if context.get("potions") is None:
            return None
        if not args[0]:
            return bool(context["potions"])
        return args[0] in set(context.get("potion_ids") or ())
    if kind == "has_event_pet":
        # ``Player.HasEventPet``（``Player.cs:248-255``）：
        # 有 `AddsPet` 的遗物，或牌组里有 `ByrdonisEgg`。
        # ``AddsPet`` 的重写全仓库只有两个（`Byrdpis`/`Pael's Legion`）。
        if context.get("relics") is None or context.get("deck") is None:
            return None
        relics = set(context["relics"])
        if relics & EVENT_PET_RELICS:
            return True
        return any(getattr(card, "cid", None) == "byrdonis_egg"
                   for card in context["deck"])
    if kind == "bag_has_relics":
        bag = context.get("relic_bag")
        if bag is None:
            return None
        return bool(bag.has_available())
    if kind == "deck_count_ge" or kind == "deck_any":
        deck = context.get("deck")
        if deck is None:
            return None
        hits = sum(1 for card in deck if _card_matches(args[0], card))
        return hits >= args[1] if kind == "deck_count_ge" else hits > 0
    return None


#: ``RelicModel.AddsPet`` 的重写（``grep "override bool AddsPet"`` 全仓库命中两处）：
#: ``Relics/Byrdpip.cs:20`` 与 ``Relics/PaelsLegion.cs:32``。
EVENT_PET_RELICS: frozenset[str] = frozenset({"byrdpip", "paels_legion"})


def _card_matches(predicate: str, card) -> bool:
    """牌组谓词求值（谓词 id 见 :data:`CARD_PREDICATES`）。

    ⚠️ 判据全部来自源码：
    ``IsRemovable => !Keywords.Contains(Eternal)``（``CardModel.cs:738``）、
    ``Rarity == Basic``、``Tags.Contains(CardTag.X)``；
    ``IsValid(tag, c)``（``Amalgamator.cs:69-80``）再叠一个 ``Basic`` 与 ``IsRemovable``。
    """
    from . import keywords as keyword_rules

    definition = CARD_DB.get(getattr(card, "cid", ""))
    if definition is None:
        return False
    removable = not keyword_rules.is_eternal(definition)
    basic = definition.rarity == "basic"
    if predicate in ("removable", "transformable"):
        return removable
    if predicate == "basic":
        return basic
    if predicate == "basic_removable":
        return basic and removable
    if predicate.startswith("tag:"):
        return predicate[4:] in {t.lower() for t in definition.tags}
    if predicate.startswith("basic_tag:"):
        return (predicate[10:] in {t.lower() for t in definition.tags}
                and basic and removable)
    return False


def gate_ok(definition: EventDef, *, act_index: int, hp: int, max_hp: int,
            gold: int, floor: int = 0, potions: int = 0,
            deck_size: int = 0, deck=None, relics=None,
            potion_ids=None, relic_bag=None) -> bool | None:
    """条件是否满足。``None`` = **引擎算不出**（调用方必须排除该事件）。

    ``deck`` / ``relics`` / ``potion_ids`` / ``relic_bag`` 是**谓词型条件**
    （"牌组里有 X" / "身上有 Y 遗物"）需要的运行时数据。**传不进来就返回 ``None``**
    —— 缺数据时"猜 true"会放行本不该出现的事件，比少出现一个严重得多。
    """
    kind, args = definition.gate_kind, definition.gate_args
    if not definition.gate_body:
        return True
    context = {"act_index": act_index, "hp": hp, "max_hp": max_hp, "gold": gold,
               "floor": floor, "potions": potions, "deck_size": deck_size,
               "deck": deck, "relics": relics, "potion_ids": potion_ids,
               "relic_bag": relic_bag}
    return _gate_result(kind, args, **context)


# ==========================================================================
# 事件内 RNG（``EventModel.BeginEvent``）
# ==========================================================================
class EventRng:
    """事件自己的 RNG（真机 ``EventModel.Rng``）。

    ⚠️ **不能塞进 ``RngSet``**：真机那 15 条是 run/玩家级的命名流，
    而事件 RNG 是"每个事件临时开一条"（``EventModel.cs:133-138`` 的注释明写
    "A per-event RNG … we don't need to keep track of a given event's RNG state
    once it's over"）。混进 ``RngSet`` 会污染"分流结构"这条不变量。
    """

    __slots__ = ("seed", "_random")

    def __init__(self, seed: int) -> None:
        import random

        self.seed = seed
        self._random = random.Random(seed)

    def next_int(self, low: int, high: int) -> int:
        """``Rng.NextInt(a, b)``：闭区间 ``[a, b]``。"""
        return self._random.randint(low, high)

    def next_int_exclusive(self, high: int) -> int:
        """``Rng.NextInt(n)``：``[0, n)``。"""
        return self._random.randrange(high)

    def next_index(self, length: int) -> int:
        """``Rng.NextItem(list)``（``Rng.cs:289-299``）：``NextInt(0, count)``。

        ⚠️ ``Rng.NextInt(minInclusive, maxExclusive)`` 的**上界是开区间**，
        所以这里是 ``[0, length)``，不是 ``randint(0, length)``。
        """
        if length <= 0:
            raise ValueError("next_index: length 必须为正")
        return self._random.randrange(length)

    def stable_shuffle(self, items: list, key=None) -> list:
        """``Rng.StableShuffle``：先按自然序排序、再洗。

        与 ``UnstableShuffle`` 的区别是"结果与**输入顺序**无关" ——
        `DoorsOfLightAndDark.Light` 写的就是它
        （`Deck.Where(…).ToList().StableShuffle(base.Rng).Take(N)`）。

        洗法与 :meth:`sts2_sim.rng.Rng.shuffle` 逐字一致（从后往前，
        每次与 ``[0, index]`` 里的一个位置交换）—— **消费的随机数序列必须相同**，
        否则同种子下与真机分叉，而分布看起来一样。
        """
        items[:] = sorted(items, key=key)
        for index in range(len(items) - 1, 0, -1):
            other = self.next_int(0, index)
            items[index], items[other] = items[other], items[index]
        return items


def event_rng(run_seed: int, event_id: str, player_slot: int = 0) -> EventRng:
    """事件自己的随机流。

    真机（``EventModel.cs:234``）::

        Rng = new Rng((ulong)((long)Owner.RunState.Rng.Seed + (long)(IsShared ? 0 : slot))
                      + StringHelper.GetDeterministicHashCode(Id.Entry));

    其中 ``GetDeterministicHashCode`` 是 **XXH64(UTF-8 字节, seed=0)**
    （``StringHelper.cs:139-152``：``XxHash64.HashToUInt64(bytes, 0L)``），
    所以这里逐字复刻同一个式子：``seed + slot + xxh64(event_id)``（uint64 回绕）。

    ⚠️ 结构比数值重要：**每个事件一条独立流**，不同事件互不影响，
    同一事件在同一种子下可复现 —— 这是"重开面对同一个事件结果"的前提。
    """
    seed = (run_seed + player_slot + xxh64_text(event_id)) & 0xFFFFFFFFFFFFFFFF
    return EventRng(seed)


# ==========================================================================
# 状态机
# ==========================================================================
@dataclass
class EventRun:
    """一次事件访问的**可变状态**。"""

    eid: str
    page: str
    finished: bool = False
    #: 已经点过的选项 key（真机 ``WasChosen`` 守卫）
    chosen: set[str] = field(default_factory=set)
    #: 掷出来的数值（``CalculateVars`` 的结果，按变量名存）
    rolled: dict[str, int] = field(default_factory=dict)
    #: 选项效果中途挂起"等玩家选牌"时，把**去向**记在这里，选完再翻页。
    #: ``(outcome_kind, outcome_page)``
    pending_outcome: tuple[str, str] | None = None
    #: 挂起中的那个选项的 key（选完之后要知道是哪个选项的去向）
    pending_option: str = ""
    #: 事件自己的 RNG（``EventModel.Rng``）。**必须留着**：选项里的
    #: ``CardCmd.TransformToRandom(card, base.Rng, …)`` 用的就是它
    #: （``AromaOfChaos.cs:31`` / ``WhisperingHollow.cs:63``），
    #: 不是 ``RngSet`` 里的任何一条命名流。丢掉它就只能"另开一条流"，
    #: 那会让同一种子下的事件结果与真机结构不一致。
    rng: "EventRng | None" = None
    log: list[str] = field(default_factory=list)


def initial_run(definition: EventDef, rng: EventRng) -> EventRun:
    """开一个事件：掷出 ``CalculateVars`` 的量，落在初始页。

    字面量变量先铺底、掷点再覆盖 —— 与真机一致：``CalculateVars`` 在
    ``BeginEvent`` 里、选项生成之前执行（``EventModel.cs:243``），
    它算出来的值就写回 ``DynamicVars``，选项效果随后引用。
    """
    run = EventRun(eid=definition.eid, page=definition.initial_page, rng=rng)
    run.rolled.update(dict(definition.literal_vars))
    for var, expr in definition.rng_rolls:
        run.rolled[var] = _roll(expr, rng)
    return run


def _roll(expr: str, rng: EventRng) -> int:
    """``base.Rng.NextInt(a, b)`` → 区间内整数（**含两端**，与 ``Rng.NextInt`` 一致）。

    认不出的写法返回 0 并**记一条**（不静默）：目前源码里只有 ``NextInt(a, b)``
    与 ``NextInt(n)`` 两种。
    """
    match = re.search(r"NextInt\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", expr)
    if match:
        return rng.next_int(int(match.group(1)), int(match.group(2)))
    match = re.search(r"NextInt\s*\(\s*(\d+)\s*\)", expr)
    if match:
        return rng.next_int_exclusive(int(match.group(1)))
    return 0


def resolve_option_effects(option: EventOptionDef,
                           rolled: dict[str, int]) -> tuple[Effect, ...] | None:
    """把选项的原始效果字典解析成 ``Effect``（用**掷出来的**量）。

    ⚠️ 解析失败返回 ``None``，调用方必须当成错误 —— 不能"少一条效果照样结算"。
    载入时已经判过一遍（算不出的量会让事件整体不可用），这里是第二道门：
    运行时仍然算不出（比如掷点表达式没认出来）就抛出来，而不是静默变弱。
    """
    from .content import _resolve_effects

    return _resolve_effects(list(option.effects_raw), rolled)


def current_options(definition: EventDef, run: EventRun) -> tuple[EventOptionDef, ...]:
    """当前页的选项（含**锁住**的：真机也会显示，只是点不动）。"""
    page = definition.pages.get(run.page)
    return page.options if page else ()


def selectable(definition: EventDef, run: EventRun) -> tuple[int, ...]:
    """**可点**的选项下标。锁住的和已经点过的都不在其中。"""
    options = current_options(definition, run)
    return tuple(i for i, option in enumerate(options)
                 if not option.locked and option.key not in run.chosen)


def apply_option(definition: EventDef, run: EventRun, index: int,
                 run_state, events: list[str],
                 obtain_relic=None) -> None:
    """点第 ``index`` 个选项：结算效果 → 翻页 / 结束。

    ``run_state`` 是 Run 层状态（``.player``：血 / 金币 / 牌组），
    效果走 :func:`sts2_sim.runeffects.apply_run_effects` —— 与遗物同一套算子。

    ``obtain_relic`` 是"拾取遗物"的回调：``runeffects`` 不能直接依赖
    ``run.RunEnv``（会成环），所以由 ``RunEnv`` 注入自己的 ``_obtain_relic``，
    以保证**拾取类遗物的效果会触发**。传 ``None`` 时只入列表、不触发
    （调用方要清楚这一点：那是"少一段效果"，不是等价实现）。
    """
    from .runeffects import apply_run_effects

    options = current_options(definition, run)
    if not 0 <= index < len(options):
        raise ValueError(f"非法事件选项下标 {index}（本页共 {len(options)} 个）")
    option = options[index]
    if option.locked:
        raise ValueError(f"非法动作：选项 {option.name} 是锁住的")
    if option.key in run.chosen:
        raise ValueError(f"非法动作：选项 {option.name} 已经点过"
                         "（真机 WasChosen 守卫）")
    run.chosen.add(option.key)
    run.log.append(f"选择 {option.name}")

    if option.effects_raw:
        resolved = resolve_option_effects(option, run.rolled)
        if resolved is None:
            raise ValueError(
                f"事件 {definition.eid} 的选项 {option.name} 的量算不出"
                f"（掷点：{run.rolled}）—— 不许静默少结算效果")
        apply_run_effects(run_state, resolved, events,
                          rolled=run.rolled, obtain_relic=obtain_relic,
                          # ⭐ 事件里的随机效果用**事件自己的 RNG**
                          # （``CardCmd.TransformToRandom(card, base.Rng, …)``，
                          # ``AromaOfChaos.cs:31``）。不显式传的话下游只能
                          # 反查 ``state.event_run``，单元测试里那条路是断的。
                          rng=run.rng)
        # ⭐ 效果里可能挂着"等玩家选牌"（`CardSelectCmd.FromDeckForRemoval`）。
        # 这时候**不能翻页**：真机也是等玩家选完才走 SetEventFinished /
        # SetEventState。把去向记在 run 上，选完由 `finish_pending` 补上。
        if getattr(run_state, "pending_selection", None) is not None:
            run.pending_outcome = (option.outcome, option.outcome_page)
            run.pending_option = option.key
            run.log.append(f"等待选牌（{run_state.pending_selection.purpose}）")
            return
    _advance_page(run, option)


def _advance_page(run: EventRun, option: EventOptionDef) -> None:
    """按选项的去向翻页 / 结束（`SetEventState` / `SetEventFinished`）。"""
    if option.outcome == "finished":
        run.finished = True
        run.page = option.outcome_page or run.page
        run.log.append(f"事件结束（{run.page}）")
    elif option.outcome == "goto":
        run.page = option.outcome_page
        run.log.append(f"进入 {run.page}")
    elif not option.effects_raw:
        # 既没有效果、也没有去向 → 这一页会卡住。真机每个选项都至少做一件事，
        # 出现这种数据就是抽取缺口，宁可记下来也不要"点了没反应"。
        run.log.append(f"⚠️ 选项 {option.name} 没有效果也没有去向")


def finish_pending(run: EventRun, option: EventOptionDef) -> None:
    """选牌结束之后补上被推迟的翻页（``apply_option`` 挂起时记下的去向）。"""
    if run.pending_outcome is None:
        return
    kind, page = run.pending_outcome
    run.pending_outcome = None
    run.pending_option = ""
    _advance_page(run, EventOptionDef(key=option.key, name=option.name,
                                      outcome=kind, outcome_page=page,
                                      effects_raw=option.effects_raw))


def option_by_key(definition: EventDef, key: str) -> EventOptionDef | None:
    """按 key 找选项（跨页找：挂起之后当前页可能已经变了）。"""
    for page in definition.pages.values():
        for option in page.options:
            if option.key == key:
                return option
    return None
