"""从反编译源码抽取遗物（``docs/09`` L5）。

300 个遗物不可能一口气全建出来，所以抽取器**分层**产出，并如实报告每层覆盖：

1. **身份层**（全量）：id / 稀有度 / 变量 / 重写了哪些钩子
2. **数值层**：``ModifyMaxEnergy`` / ``ModifyHandDraw`` 的增量与触发条件 ——
   这两组直接改变**每回合的能量与手牌**，是全套遗物里性价比最高的
3. **效果层**：``AfterObtained`` 等钩子里的效果，能落进效果 DSL 的照实抽出
4. 其余钩子**照实记录但不实现** —— 报告里能看到还差多少，而不是"看起来已经一致"

用法
----
    python tools/extract_relics.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.extract_cards import (  # noqa: E402
    apply_value_props, balanced, balanced_braces, build_key_map, enum_tail,
    extract_effects, extract_rewards, method_body, parse_vars, property_expr, snake,
)

SRC_ROOT = Path("data/decompiled/sts2/MegaCrit.Sts2.Core.Models.Relics")
OUT_DEFAULT = Path("data/content/repo/relics_source.json")
CODEX_DEFAULT = Path("data/content/repo/relics.json")

#: 这些属性不是"效果"，抽出来只是噪声。
IGNORED_PROPS = frozenset({
    "Rarity", "CanonicalVars", "ExtraHoverTips", "HasUponPickupEffect",
    "ShowCounter", "DisplayAmount", "IsUsedUp", "MerchantCost", "FlashSfx",
    "IsAllowedInShops", "SpawnsPets", "AddsPet", "IsStackable",
    "ShouldFlashOnPlayer", "IsAllowed", "PackOdds", "SpawnWeight",
})

#: 钩子体里"真的改变了游戏"的判据：调用了命令（``PowerCmd.Apply`` /
#: ``CreatureCmd.Damage`` / ``PlayerCmd.GainEnergy`` …）。
COMMAND_CALL = re.compile(r"\b\w+Cmd\s*\.")

#: 靠**返回值**改变行为的钩子前缀。这类钩子体里通常一条命令都没有，
#: 但它返回的数值/布尔就是行为本身（``TryModifyEnergyCostInCombatLate`` 把费用改成 0）。
QUERY_HOOK_PREFIXES: tuple[str, ...] = (
    "TryModify", "Should", "Modify", "Can", "Is", "Get",
)


def inline_zero_arg_helpers(source: str, body: str, depth: int = 2) -> str:
    """把体里调用的**无参私有辅助方法**展开 —— **只用于分类**。

    ⚠️ 展开物**绝不进效果抽取**：辅助方法里可能带条件
    （``BeltBuckle.ApplyDexterity`` 只在"没有药水"时才被调用），
    无条件摊平会让遗物"永远生效" —— 静默的强度错误，比不实现更糟。
    这里只回答一个问题："这个钩子到底有没有改变游戏"。
    """
    if depth <= 0:
        return body
    names = {match.group(1)
             for match in re.finditer(r"(?:await\s+)?([A-Z]\w*)\s*\(\s*\)\s*;", body)}
    parts = [body]
    for name in names:
        helper = method_body(source, name)
        if helper:
            parts.append(inline_zero_arg_helpers(source, helper, depth - 1))
    return "\n".join(parts)


def classify_hook(source: str, hook: str, body: str) -> str:
    """这个钩子属于哪一类 —— 决定它算不算"缺失的遗物行为"。

    * ``commands`` —— 体里（含展开一层的无参私有辅助方法）调用了命令，
      这是**真的缺失行为**；
    * ``query`` —— 形如 ``TryModify…`` / ``Should…``，靠**返回值**改变行为：
      一条命令都没有，但漏掉它照样是行为错误；
    * ``bookkeeping`` —— 只写自己的私有字段 / 状态显示
      （``CardsPlayedThisTurn = 0``、``base.Status = RelicStatus.Normal``），
      **本身不改变任何游戏状态**，只是给同类里别的钩子记数。

    为什么必须分开：把 ``bookkeeping`` 也算成"遗物缺行为"会**高估缺口**
    （审计 F11 的同一类问题）—— 实测 109 处"体是空的"钩子里绝大多数是这一类，
    而它们单独实现出来**不会改变任何数值**。
    """
    if hook.startswith(QUERY_HOOK_PREFIXES):
        return "query"
    if COMMAND_CALL.search(inline_zero_arg_helpers(source, body)):
        return "commands"
    return "bookkeeping"

#: 数值钩子里认得的"条件"写法 → 声明式条件。
#: ⚠️ 认不出就**整条不采纳**并记进报告，绝不默认成"无条件生效" ——
#: 无条件生效会让"第三回合起才加能量"变成"每回合都加"，强度差很多。
CONDITION_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (r"TurnNumber\s*>\s*(\d+)", "turn_gt", "int"),
    (r"TurnNumber\s*>=\s*(\d+)", "turn_gte", "int"),
    (r"TurnNumber\s*<\s*(\d+)", "turn_lt", "int"),
    # ⚠️ `turn_lte` 是**最主流的写法**（真机统一写 `TurnNumber <= 1` 表示"只在
    # 第 1 回合"）。少了它，`BagOfMarbles` / `CrackedCore` / `Lantern` / `Akabeko`
    # 这一整批"开局型"遗物的守卫全都抽不出来。
    (r"TurnNumber\s*<=\s*(\d+)", "turn_lte", "int"),
    (r"TurnNumber\s*==\s*(\d+)", "turn_eq", "int"),
    (r"TurnNumber\s*!=\s*(\d+)", "turn_ne", "int"),
    # ⭐ **牌型条件**：`cardPlay.Card.Type == CardType.Attack`。
    # `GamePiece`（仅能力牌 → 抽牌）/ `LostWisp` / `Permafrost` 这些遗物
    # 整个行为就是"只在打出某类牌时触发" —— 丢掉它等于**每张牌都触发**。
    # 引擎侧由 `relics._applies` 的 `card_type` 守卫求值。
    (r"CardType\.(\w+)", "card_type", "card_type"),
    (r"TurnsSeen\s*<\s*([\w\[\]\"\.]+)", "turns_seen_lt", "expr"),
    (r"TurnsSeen\s*>\s*([\w\[\]\"\.]+)", "turns_seen_gt", "expr"),
    (r"TurnsSeen\s*!=\s*(\d+)", "turns_seen_ne", "int"),
    (r"KindleCount\s*<=\s*(\d+)", "kindle_lte", "int"),
    (r"_cardsPlayedLastTurn\s*<=\s*([\w\.\[\]\" ]+)", "cards_last_turn_lte", "expr"),
)


#: "是不是我"的守卫 —— **不是**触发条件，只是把影响限制在拥有者身上。
#: ⚠️ 这里匹配的是**括号里的条件表达式**（`player != base.Owner`），
#: 不是整条 `if (...) { return amount; }` 语句 —— 拿整句正则去匹配表达式
#: 会一条都对不上，于是连 `Sozu` 这种只有守卫的遗物都被判成"条件认不出"
#: （实测把 12 个无条件能量遗物全误报了）。
OWNERSHIP_CONDITIONS: tuple[str, ...] = (
    r"^player\s*!=\s*base\.Owner$",
    r"^target\s*!=\s*base\.Owner$",
    r"^player\s*==\s*base\.Owner$",
    r"^base\.Owner\s*==\s*null$",
    r"^base\.Owner\s*!=\s*null$",
    r"^.*\s*==\s*null$",                 # 变量取值判断，不是触发条件
    r"^.*\s*!=\s*null$",
    # ⭐ 下面这些是"**不是触发条件**"的上下文判断。它们**恒真**（在本模拟器的
    # 单人战斗里），所以允许从条件里剥掉；剥不掉的那些（私有计数器、牌型标签、
    # 能量阈值…）必须让整个钩子被拒 —— 见 :func:`unrecognized_guard`。
    #
    # * 归属：是不是我打出的牌（`RazorTooth` / `GamePiece` 都有这一段）
    r"^cardPlay\.Card\.Owner\s*==\s*base\.Owner$",
    r"^cardPlay\.Card\.Owner\s*!=\s*base\.Owner$",
    r"^base\.Owner\s*==\s*cardPlay\.Card\.Owner$",
    # * 战斗进行中：战斗内的钩子必然成立
    r"^CombatManager\.Instance\.IsInProgress$",
    r"^!CombatManager\.Instance\.IsInProgress$",
    # * 单人恒真：玩家永远是自己阵营的 participants 成员（docs/13 §5 第一交付范围）
    r"^participants\.Contains\(base\.Owner\.Creature\)$",
    r"^participants\.Contains\(.*base\.Owner.*\)$",
    # * **死亡安全网**：`if (!Owner.Creature.IsDead)` 是"人死了就别触发"，
    #   不是触发条件（`hook_scope` 也是这么归类的）。
    #   ⚠️ 少了这一条会**误伤**：`BurningBlood` / `BlackBlood`（战斗胜利回血）
    #   的整个条件就只是这一个 —— 判成"认不出"会把核心遗物整批变成缺口。
    r"^!?base\.Owner\.Creature\.IsDead$",
    r"^!?[\w\.]*\.IsDead$",
)

#: 兼容旧名（别处可能引用）
OWNERSHIP_GUARDS = OWNERSHIP_CONDITIONS

#: 钩子体里的**房间条件**。
#:
#: ⚠️ `AfterRoomEntered` 在**所有**房间类型都会触发 —— 源码
#: `AbstractModel.cs:64` 写得很清楚：*"AfterRoomEntered is relevant outside of
#: combat, and should be called on all models"*，且 CombatRoom / EventRoom /
#: MerchantRoom / RestSiteRoom / TreasureRoom 五处各自 `Hook.AfterRoomEntered(...)`。
#: 所以"这个遗物什么时候生效"**完全**取决于函数体里的这个判断：
#: `Vajra` 判 `room is CombatRoom`（每场战斗 +1 力量），`MealTicket` 判
#: `room is MerchantRoom`（进商店回 15 血）。
#:
#: 不抽这个判断、把 `AfterRoomEntered` 一律当成"战斗开始"，会把
#: `MealTicket` 变成"**每场战斗**回 15 血" —— 数值差一个数量级，而且不报错。
ROOM_CONDITIONS: tuple[tuple[str, str], ...] = (
    # ⚠️ 匹配必须**与接收者无关**：`Pantograph` 写的是
    # `base.Owner.RunState.CurrentRoom.RoomType == RoomType.Boss`，
    # 要求变量名字面叫 `room` 会漏掉它，于是 Boss 战回血被当成"每场战斗回血"。
    (r"\broom\s+is\s+(\w+Room)\b", "class"),
    (r"\bRoomType\.(\w+)", "enum"),
)

#: 第三个判断维度：**地图点类型**（`Planisphere` 判 `MapPointType.Unknown`）。
MAP_POINT_CONDITIONS: tuple[str, ...] = (r"\bMapPointType\.(\w+)",)

#: 与房间无关、但同样决定**触发时机**的状态条件（跨房间计数 / 用尽标志）。
#: 这些活在 Run 层，战斗内复刻不出来 —— 照实记录，不猜。
STATEFUL_CONDITIONS: tuple[tuple[str, str], ...] = (
    (r"\bTimesLifted\b", "times_lifted"),
    (r"\bCombatsLeft\b", "combats_left"),
    (r"\bIsUsedUp\b", "is_used_up"),
    (r"\bTimesUsed\b", "times_used"),
    # ⭐ 下面这五条是**审计中实测出来的静默变强**（见 docs/12 §2.6「遗物口径」）。
    # 抽取器原本只认"回合数守卫"和四个跨回合计数器，于是下面这些条件**全部丢失**，
    # 钩子被当成无条件生效：
    #
    #   HappyFlower   `(TurnsSeen + 1) % Turns == 0` → 每 3 回合 +1 能量
    #                 丢了条件就变成**每回合** +1 能量
    #   WarPaint      `.Where(c => c.Type == CardType.Skill).Take(N)` → 升级 N 张**技能**牌
    #                 丢了条件会升级到任意牌
    #   Whetstone     同上（Attack）
    #   IronClub      `CardsPlayed % N == 0` → 每 N 张牌抽 1 张
    #   LetterOpener  `SkillsPlayedThisTurn % N == 0` → 每 N 张技能牌造成伤害
    #
    # ⚠️ 认不出就**标 stateful 拒绝采纳**（变成缺口，如实报出来），
    # 绝不默认"无条件生效" —— 后者是静默的强度偏差，比不实现危险得多。
    #
    # 说明：`participants.Contains(base.Owner.Creature)` **不在**这里。
    # 本模拟器是单人环境（docs/13 §5 的第一交付范围），玩家永远是自己阵营的
    # participants 成员，所以这个前缀**恒真**，丢掉它不会改变任何数值。
    (r"%\s*[\w\[]", "modulo_counter"),
    # ⭐ **私有计数器**。这一类比"取模"更宽：只要钩子体读了/写了自己的计数器，
    # 抽取器就建不出它的触发条件 —— 必须拒绝，不能默认无条件生效。
    #
    # 实测漏网的一例（`RainbowRing`，彩虹环）：
    #   if (… && ActivationCountThisTurn < 1)
    #       AttacksPlayedThisTurn += (Type == Attack ? 1 : 0);
    #       SkillsPlayedThisTurn  += (…);
    #       PowersPlayedThisTurn  += (…);
    #       if (AttacksPlayedThisTurn > 0 && SkillsPlayedThisTurn > 0
    #           && PowersPlayedThisTurn > 0) { +1 力量 +1 敏捷 }
    # 即"一回合内打出攻击+技能+能力牌各至少一张"才触发。
    # 原来的模式只看 `%`，于是它被当成**每张牌都 +1 力量 +1 敏捷**。
    # ⚠️ **只排除 `SetToFreeThisTurn`**（它的语义已经实现，见 `docs/12` §2.14；
    # 生成牌之外那些用法由 `extract_cards.SILENT_GAP_PATTERNS` 单独报缺口）。
    #
    # 曾经写成"排除所有后面跟 `(` 的标识符"，那会**误伤历史查询**：
    # `RippleBasin` 的 `!History.CardPlaysFinished.Any(e => e.HappenedThisTurn(…)
    # && e.CardPlay.Card.Type == Attack)` 是"本回合**没有**打出过攻击牌"，
    # 引擎没有战斗历史 → 必须算缺口。放宽之后它会被"采纳"，
    # 却因为 `card_type` 守卫在 `BeforeSideTurnEnd` 时机上算不出而**永不触发** ——
    # 正是"报告说完全复刻、实际什么都不做"那一类。
    # 同一个坑还适用于 `LostHpThisTurn` / `AnyCardsPlayedThisTurn` /
    # `AddThisTurn` / `SetThisCombat`（改费用，引擎没实现）。
    (r"\b(?!SetToFreeThisTurn\b)\w+(?:ThisTurn|ThisCombat|ThisBattle)\b",
     "per_turn_counter"),
    (r"\b(?:TurnsSeen|TimesUsed|TimesLifted|CombatsLeft|IsUsedUp|ActivationCount)"
     r"\b", "persistent_counter"),
    # ⚠️ `CardType\.` **不在**这里：牌型条件现在已经能建模（见 CONDITION_PATTERNS
    # 的 `card_type`），由 `relics._applies` 求值。
    (r"\.Take\s*\(", "collection_take"),
    (r"\bStableShuffle\b", "random_sample"),
    (r"\.Where\s*\(", "collection_predicate"),
    (r"\bNextItem\s*\(", "random_target"),
)


def strip_ownership_guards(body: str) -> str:
    return body


#: 「每打出 N 张某类牌触发一次」的私有**本回合**计数器 → 引擎里同义的计数器名。
#:
#: 真机这一族的写法**高度统一**（`Kunai` / `Shuriken` / `LetterOpener` /
#: `OrnamentalFan` 四个**逐字相同**，只有最后那一条效果不同）::
#:
#:     if (cardPlay.Card.Owner == base.Owner && CombatManager.Instance.IsInProgress
#:         && cardPlay.Card.Type == CardType.Attack)
#:     {
#:         AttacksPlayedThisTurn++;                                  // 先自增
#:         int intValue = base.DynamicVars.Cards.IntValue;
#:         if (AttacksPlayedThisTurn % intValue == 0) { …效果… }     // 再判取模
#:     }
#:
#: ⚠️ **只认本回合计数器**（名字以 ``ThisTurn`` 结尾）。引擎里
#: ``CombatState.attacks_played_this_turn`` / ``skills_played_this_turn`` /
#: ``cards_played_started_this_turn`` 与它们**同名同义**，所以取模能精确复刻。
#:
#: 跨回合的计数器**不在这里**：``Nunchaku.AttacksPlayed``（带 ``[SavedProperty]``）
#: 与 ``IronClub.CardsPlayed`` 是"整场累计、不随回合清零"，需要"每遗物一份
#: 跨回合计数"，本批次不做 —— 认不出就照旧算缺口，
#: 绝不把"每 10 次攻击"近似成"每回合重置"。
PER_TURN_COUNTERS: dict[str, str] = {
    "AttacksPlayedThisTurn": "attack",
    "SkillsPlayedThisTurn": "skill",
    "CardsPlayedThisTurn": "card",
}

#: 计数器自增：``AttacksPlayedThisTurn++;``
MODULO_COUNTER = re.compile(r"\b(\w+ThisTurn)\s*(?:\+\+|\+=\s*1)\s*;")
#: 取模判据：``AttacksPlayedThisTurn % intValue == 0``
MODULO_TEST = re.compile(r"\b(\w+)\s*%\s*([\w\.\[\]\"]+)\s*==\s*0")


def modulo_guard(body: str, variables: list[dict]) -> tuple[str, dict | None]:
    """把"每 N 张某类牌"抽成**可求值**的守卫，返回 ``(剥干净的体, 守卫)``。

    返回的第二项是 ``None`` 时表示**没认出来** —— 调用方照原样去判
    ``stateful`` / ``unrecognized_guard``，于是这个钩子照旧算缺口（安全方向）。

    为什么要"剥干净"：``hook_scope`` 的 ``STATEFUL_CONDITIONS`` 靠 ``%`` 与
    ``…ThisTurn`` 这两个**文本特征**判"建不出条件"。这两个特征确实出现过
    （见 ``docs/12`` 的静默变强清单），但现在它们的语义已经建出来了 ——
    不剥掉的话，能精确复刻的遗物会被自己的守卫模式误判成缺口。
    """
    text = re.sub(r"//[^\n]*", "", body)
    counter = MODULO_COUNTER.search(text)
    if not counter or counter.group(1) not in PER_TURN_COUNTERS:
        return body, None
    name = counter.group(1)
    test = MODULO_TEST.search(text, counter.end())
    if test is None or test.group(1) != name:
        return body, None
    # 取模的右操作数常是"先取到局部变量的动态变量"（``int intValue =
    # base.DynamicVars.Cards.IntValue;``），所以要跟着局部变量回到声明。
    operand = test.group(2)
    definition = re.search(rf"\b{re.escape(operand)}\s*=\s*([^;]+);", text)
    resolved = (definition.group(1) if definition else operand)
    var = re.search(r'DynamicVars(?:\[\s*"(\w+)"\s*\]|\.(\w+))', resolved)
    if var is None:
        return body, None
    var_name = var.group(1) or var.group(2)
    base = next((v.get("base") for v in variables if v.get("name") == var_name),
                None)
    if not isinstance(base, int) or base <= 0:
        return body, None               # 数值抽不出 → 不猜
    guard = {"kind": "every_n_turn",
             "value": {"counter": PER_TURN_COUNTERS[name], "n": base,
                       "var": var_name},
             "polarity": "require"}
    # ⚠️ 剥的时候要连**整个 ``if (...)`` 头一起删掉**（留一个裸块 `{…}`），
    # 不能只把条件换成 ``true``：`unrecognized_guard` 是逐条解释 ``if`` 条件的，
    # 换出来的 ``if (true)`` 反而会被判成"认不出的条件" —— 实测这样改完之后
    # `Kunai` / `Shuriken` / `LetterOpener` / `OrnamentalFan` 四个仍然报缺口。
    #
    # 下标从后往前删（先删位置靠后的 `if`，再删前面的自增语句），免得错位。
    head = text.rfind("if", 0, test.start())
    if head == -1:
        return body, None
    open_paren = text.find("(", head)
    if open_paren == -1:
        return body, None
    _condition, close = balanced(text, open_paren)
    stripped = text[:head] + text[close + 1:]
    stripped = stripped[:counter.start()] + stripped[counter.end():]
    return stripped, guard


def conditional_blocks(body: str) -> list[tuple[str, str]]:
    """取出 ``if (…)`` 的 ``(条件表达式, 极性)``。

    ⚠️ **极性是这里的核心**，搞反了整批遗物会反着生效：

    * **排除式守卫**（数值钩子都是这个形态）—— 条件成立就 ``return``，即
      "满足则**不**生效"：:

          if (TurnNumber > 1) return cardsToDraw;      // 只在第 1 回合多抽

    * **包含式条件**（效果钩子几乎都是这个形态）—— 条件成立才**执行**效果：:

          if (participants.Contains(...) && TurnNumber <= 1)
          {
              await PlayerCmd.GainEnergy(...);          // 只在第 1 回合给能量
          }

    只看条件表达式、不看后面跟的是 ``return`` 还是效果体，会把提灯变成
    "第 2 回合起每回合 +1 能量"、把弹珠袋变成"永远不上易伤" ——
    两者都不报错，只是数值全错。
    """
    blocks: list[tuple[str, str]] = []
    for match in re.finditer(r"\bif\s*\(", body):
        args, close = balanced(body, match.end() - 1)
        # ⚠️ `balanced` 返回的是**右括号的下标**，不是它后面一位。
        # 从 `close` 开始切会切到 `)` 自己，于是后面永远判不出 `return`，
        # 全部条件都被当成包含式 —— 实测就是这样把 `TwistedFunnel`
        # 的"跳过"判成了"生效"。
        tail = body[close + 1:close + 61].lstrip()
        if tail.startswith("{"):
            tail = tail[1:].lstrip()
        polarity = ("skip_when" if tail.startswith(("return", "continue"))
                    else "require")
        blocks.append((" ".join(args.split()), polarity))
    return blocks


def hook_scope(body: str) -> dict:
    """抽出一个钩子**什么时候**生效：房间 / 地图点范围 + 是否有状态条件。

    ``room``：``combat`` / ``elite`` / ``boss`` / ``merchant`` / ``rest_site`` /
    ``event`` / ``treasure_room``，或 ``any``（函数体里没有任何房间判断）、
    ``mixed``（判了多种）。
    """
    rooms: list[str] = []
    for pattern, kind in ROOM_CONDITIONS:
        for found in re.finditer(pattern, body):
            name = found.group(1)
            rooms.append(snake(name[:-4]) if kind == "class" and name.endswith("Room")
                         else snake(name))
    points: list[str] = []
    for pattern in MAP_POINT_CONDITIONS:
        points.extend(snake(m.group(1)) for m in re.finditer(pattern, body))
    stateful = sorted({name for pattern, name in STATEFUL_CONDITIONS
                       if re.search(pattern, body)})
    # ⚠️ **钩子体里的回合守卫也必须抽出来**。`Lantern` 是
    # `AfterSideTurnStart` + `if (participants.Contains(...) && TurnNumber <= 1)`：
    # 丢掉这个守卫会把"**第 1 回合** +1 能量"变成"**每回合** +1 能量" ——
    # 数值翻几倍，而且日志看着完全正常。
    # 极性按 :func:`conditional_blocks` 判定（效果钩子多为**包含式**）。
    turn_guards: list[dict] = []
    for condition, polarity in conditional_blocks(body):
        for pattern, name, kind in CONDITION_PATTERNS:
            for found in re.finditer(pattern, condition):
                value: object = found.group(1).strip()
                if kind == "int":
                    value = int(value)
                elif kind == "card_type":
                    # `CardType.Attack` → `attack`（与 CardDef.card_type 同一套写法）
                    value = snake(str(value))
                turn_guards.append({"kind": name, "value": value,
                                    "polarity": polarity})
    return {
        "room": rooms[0] if len(set(rooms)) == 1 else ("any" if not rooms else "mixed"),
        "all_rooms": sorted(set(rooms)),
        "map_point": sorted(set(points)),
        # 「人死了就别触发」不是触发条件，是安全网
        "dead_check": bool(re.search(r"\bIsDead\b", body)),
        "stateful": stateful,
        "turn_guards": turn_guards,
    }


def if_conditions(body: str) -> list[str]:
    """取出函数体里每个 ``if (...)`` 的条件表达式。

    ⚠️ **``if (...) { break; }`` 不算条件**：那只是 ``while`` 循环里的提前退出
    （``DelicateFrond``：*"有空的药水槽就继续买，买不进去就退出"*）。
    把它当成触发守卫会让整个钩子被判成"触发条件认不出"——
    实测 `delicate_frond` 的 `BeforeCombatStart` 就是这样被整条拒掉的。
    """
    conditions: list[str] = []
    for match in re.finditer(r"\bif\s*\(", body):
        args, end = balanced(body, match.end() - 1)
        brace = body.find("{", end)
        if brace != -1:
            inner, _close = balanced_braces(body, brace)
            if inner.strip() in ("break;", "break"):
                continue
        conditions.append(" ".join(args.split()))
    return conditions


def unrecognized_guard(body: str) -> bool:
    """剥掉认得的守卫后，还有**认不出**的条件吗？

    认得的三类（全部能从**条件表达式**层面判定）：
    1. 归属 / 上下文 / 单人恒真（:data:`OWNERSHIP_CONDITIONS`）
    2. 回合数与牌型（:data:`CONDITION_PATTERNS`）
    3. 房间与地图点范围（:data:`ROOM_CONDITIONS` / :data:`MAP_POINT_CONDITIONS`）

    ⚠️ 两个坑都踩过：
    1. 不能靠"删掉认得的条件再看还剩不剩 ``if``" —— 会留下空壳，
       于是**所有**带守卫的遗物都被判成认不出（误报 11 个）。
    2. 守卫正则必须是**条件表达式**级的（`player != base.Owner`），
       拿整条语句的正则来匹配表达式会一条都对不上（误报 12 个）。

    ⭐ 这是本模块**最可靠的准入判据**：不靠"猜标识符名字"（那种做法实测漏掉了
    `RainbowRing` 的 `ActivationCountThisTurn`、`TuningFork` 的
    `SkillsPlayed >= SkillsThreshold`、`HelicalDart` 的 `Tags.Contains(CardTag.Shiv)`、
    `IvoryTile` 的 `EnergyValue >= …` —— 四个都因此被当成**无条件生效**），
    而是要求"每一个 ``if`` 条件都能被解释"。**解释不了就整条拒绝。**
    """
    for condition in if_conditions(body):
        normalized = " ".join(condition.split())
        if any(re.fullmatch(pattern, normalized) for pattern in OWNERSHIP_CONDITIONS):
            continue
        if any(re.search(pattern, normalized) for pattern, _n, _k in CONDITION_PATTERNS):
            continue
        if any(re.search(pattern, normalized) for pattern, _k in ROOM_CONDITIONS):
            continue
        if any(re.search(pattern, normalized) for pattern in MAP_POINT_CONDITIONS):
            continue
        return True
    return False


def if_blocks(body: str) -> list[tuple[str, str, int, int]]:
    """取出每个 ``if`` 的 ``(条件, 极性, 块起点, 块终点)``。

    块范围用来判断某条 ``return`` 到底**归哪个条件管**。
    """
    blocks: list[tuple[str, str, int, int]] = []
    for match in re.finditer(r"\bif\s*\(", body):
        condition, close = balanced(body, match.end() - 1)
        tail = body[close + 1:close + 61].lstrip()
        braced = tail.startswith("{")
        if braced:
            tail = tail[1:].lstrip()
        polarity = ("skip_when" if tail.startswith(("return", "continue"))
                    else "require")
        if braced:
            start = body.index("{", close + 1)
            depth = 0
            end = len(body)
            for index in range(start, len(body)):
                if body[index] == "{":
                    depth += 1
                elif body[index] == "}":
                    depth -= 1
                    if depth == 0:
                        end = index
                        break
        else:
            semi = body.find(";", close)
            end = semi + 1 if semi >= 0 else len(body)
        blocks.append((" ".join(condition.split()), polarity, close, end))
    return blocks


def numeric_modifier(source: str, hook: str, variables: list[dict]) -> dict | None:
    """抽 ``ModifyMaxEnergy`` / ``ModifyHandDraw`` 这类**数值钩子**。

    形态高度统一 —— **一串 early-return 守卫 + 最后返回修正值**::

        if (player != base.Owner) return amount;          // 是不是我
        if (TurnNumber > 1) return amount;                // 真正的触发条件
        return amount + base.DynamicVars.Cards.IntValue;

    ⚠️ 注意守卫是**兄弟节点**而不是外层块：修正值那行不在任何 ``if`` 里面，
    所以必须收**整个函数体**里的条件，不能只看包住它的块。

    反过来说，如果修正值那行被一个**包含式**块包住（``if (…) { return amount + X; }``），
    当前的条件模型就表达不了它 —— 标记 ``unresolved_guard``，**不猜**。
    """
    body = method_body(source, hook)
    if body is None:
        return None
    delta = re.search(
        r"return\s+\w+\s*([+\-])\s*(?:\(decimal\))?\s*base\.DynamicVars"
        r"(?:\[\s*\"(\w+)\"\s*\]|\.(\w+))", body)
    if not delta:
        return None
    sign = -1 if delta.group(1) == "-" else 1
    var = delta.group(2) or delta.group(3)
    entry: dict = {"var": var, "sign": sign}
    # ⚠️ **把所有守卫都收集起来，不能只取第一个**：`Pocketwatch` 有
    # 「第 1 回合跳过」+「上回合打出 >3 张则跳过」两个守卫，只取第一个会漏掉
    # 真正的触发条件，变成"每回合都多抽 3 张"。
    guards: list[dict] = []
    for condition, polarity, start, end in if_blocks(body):
        if polarity == "require" and start <= delta.start() < end:
            # 修正值被包含式条件管着 → 现有模型表达不了
            entry["unresolved_guard"] = True
            continue
        if polarity != "skip_when":
            continue
        for pattern, name, kind in CONDITION_PATTERNS:
            for found in re.finditer(pattern, condition):
                value: object = found.group(1).strip()
                if kind == "int":
                    value = int(value)
                guards.append({"kind": name, "value": value,
                               "polarity": "skip_when"})
    if guards:
        entry["guards"] = guards
    if unrecognized_guard(body):
        entry["unresolved_guard"] = True
    return entry


def parse_relic(path: Path, report: collections.Counter) -> dict | None:
    source = path.read_text(encoding="utf-8", errors="replace")
    rarity = property_expr(source, "Rarity")
    if rarity is None:
        report["relic_without_rarity"] += 1
        return None

    variables = parse_vars(source)
    key_map = build_key_map(variables)

    hooks: dict[str, list[dict]] = {}
    for hook in ("AfterObtained", "AfterCombatEnd", "AfterRoomEntered",
                 "AfterCombatVictory", "BeforeCombatStart",
                 "BeforeSideTurnStart", "AfterSideTurnStart",
                 "BeforeSideTurnEnd", "AfterSideTurnEnd",
                 "AfterPlayerTurnStart", "AfterEnergyReset",
                 "AfterCardPlayed", "BeforeHandDraw"):
        body = method_body(source, hook)
        if body is None:
            continue
        effects, unsupported, choices = extract_effects(body, key_map,
                                                            allow_bookkeeping=True)
        apply_value_props(effects, variables)
        # ⭐ **"每 N 张某类牌"的取模计数**先抽出来，再拿**剥干净**的函数体去判
        # 条件。理由与做法见 :func:`modulo_guard`：不剥的话，能精确复刻的
        # `Kunai` / `Shuriken` / `LetterOpener` / `OrnamentalFan` 会被
        # `STATEFUL_CONDITIONS` 的文本特征（`%`、`…ThisTurn`）误判成缺口。
        guard_body, modulo = modulo_guard(body, variables)
        # ⭐⭐ **准入判据：每个 ``if`` 条件都必须能被解释**（只对**有副作用**的钩子）。
        #
        # 解释不了就把效果**清空**并记成缺口 —— 绝不让它"无条件生效"。
        # 实测这一条抓出四个静默变强（都是"条件丢掉 = 每张牌都触发"）：
        #   RainbowRing   ActivationCountThisTurn / AttacksPlayedThisTurn>0…
        #   TuningFork    SkillsPlayed >= SkillsThreshold
        #   HelicalDart   Tags.Contains(CardTag.Shiv)
        #   IvoryTile     Resources.EnergyValue >= EnergyThreshold
        #
        # ⚠️ **只对有副作用的钩子判**：`if effects and …` 这个条件是必须的。
        # 纯显示型钩子（`InvokeDisplayAmountChanged` / `RelicStatus` / `Flash`）
        # 一条效果都抽不出来，本来就不会被采纳；若在这里给它加一条"条件认不出"
        # 的缺口，就会**凭空**把该遗物判成"有战斗阶段缺口"，
        # 连带把它的数值半边也挡掉 —— 实测 `paels_flesh`（血肉之镯）就是这样
        # 从"第 3 回合起 +1 能量"变成完全失效的。
        if effects and unrecognized_guard(guard_body):
            report[f"hook_condition_unresolved::{hook}"] += 1
            effects = []
            unsupported = [*unsupported, "触发条件认不出（抽取器建不出）"]
            choices = 0
        # ⭐ `RewardsCmd.OfferCustom`：奖励列表写在**另一个私有方法**里
        # （`GenerateRewards()`），只抽钩子体会一条奖励都看不到。
        # 这 8 个遗物（召唤铃 / 大锅 / 玻璃眼 / 浑天仪…）都是这个形状。
        if any("OfferCustom" in item for item in unsupported):
            rewards = extract_rewards(body, source)
            if rewards:
                unsupported = [item for item in unsupported if "OfferCustom" not in item]
                for group in rewards["groups"]:
                    effects.append({
                        "op": "offer_rewards",
                        "amount": int(group.get("count") or 1),
                        "target": "self",
                        "times": 1,
                        "filter": tuple(sorted(
                            (k, v) for k, v in group.items() if k != "count")),
                    })
            else:
                unsupported.append("RewardsCmd.OfferCustom(奖励列表解析不出)")
        # ⚠️ **另一种"给奖励"的写法**：`new RewardsSet(owner).WithCustomRewards(…).Offer()`。
        # 它**不叫** `OfferCustom`，所以上面那段不会触发 ——
        # 实测 `NeowsBones`（"随机取 N 个遗物奖励 + 加诅咒"）的遗物奖励整段
        # **丢失且没有任何缺口**，遗物会被算成"有行为"，而玩家少了 N 个奖励。
        # 兜底：只要体里出现"奖励"的构造，而效果表里没有 `offer_rewards`，就报缺口。
        if (not any(e.get("op") == "offer_rewards" for e in effects)
                and re.search(r"RewardsSet\b|new\s+\w+Reward\s*\(", body)
                and not any("OfferCustom" in item for item in unsupported)):
            report[f"hook_rewards_unmodeled::{hook}"] += 1
            unsupported.append("奖励列表未实现（RewardsSet / new *Reward）")
        scope = hook_scope(guard_body)
        if any(effect.get("op") in ("random_deck_cards", "add_random_cards",
                                    "random_draw_cards")
               for effect in effects):
            # ⭐ "从牌组随机取 N 张不重复"已经被那条 `random_deck_cards` 完整表达
            # （张数、谓词、**随机流**、取法都在里面）—— 那些"集合谓词 /
            # 取 N 张 / 随机采样"的文本特征不再算状态缺口。
            # ⚠️ 只有**真的产出**了这条算子才剔除：认不出时照样算缺口，
            # 否则 `Whetstone` 这类会从"残缺"变成"干净"却什么都不做。
            scope["stateful"] = [flag for flag in scope.get("stateful") or ()
                                 if flag not in ("collection_predicate",
                                                 "collection_take",
                                                 "random_sample",
                                                 # `NextItem(...)` 的"随机取一个"
                                                 # —— 取哪一个已经由算子的
                                                 # `pick="item"` 表达了。
                                                 "random_target")]
        if modulo is not None:
            # 取模守卫与回合 / 牌型守卫同属 `turn_guards`（都由 `relics._applies`
            # 求值）。放在**最后**：`Every N` 是"计数命中"，其它守卫是"上下文符合"。
            scope["turn_guards"] = [*scope["turn_guards"], modulo]
        hooks[hook] = {"effects": effects,
                       "unsupported": sorted(set(unsupported)),
                       "choices": choices,
                       "kind": classify_hook(source, hook, body),
                       "scope": scope}
        for command in set(unsupported):
            report[f"unsupported_command::{command}"] += 1

    # 重写了但没实现的其他钩子：照实记录，用于覆盖率报告。
    # ⚠️ 同时记录**类别**（`declared_hook_kinds`）：`declared_hooks` 是纯名字列表，
    # 已经被 content.py 当集合用，改结构会破坏调用方 —— 所以另开一个字段。
    declared = sorted({
        match.group(1) for match in
        re.finditer(r"(?:public|protected)\s+override\s+(?:async\s+)?"
                    r"[\w<>?\[\],\s]+?\s+(\w+)\s*[\(\{]", source)
    } - IGNORED_PROPS - set(hooks))
    declared_kinds: dict[str, str] = {}
    for name in declared:
        body = method_body(source, name)
        declared_kinds[name] = (classify_hook(source, name, body)
                                if body is not None else "unknown")

    return {
        "rid": snake(path.stem),
        "class": path.stem,
        "rarity": enum_tail(rarity),
        "vars": variables,
        "max_energy": numeric_modifier(source, "ModifyMaxEnergy", variables),
        "hand_draw": numeric_modifier(source, "ModifyHandDraw", variables),
        "hooks": hooks,
        "declared_hooks": declared,
        "declared_hook_kinds": declared_kinds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从反编译源码抽取遗物")
    parser.add_argument("--src", default=str(SRC_ROOT))
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--codex", default=str(CODEX_DEFAULT))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f"没有源码目录：{src}")
        return 3
    files = sorted(src.glob("*.cs"))
    report: collections.Counter = collections.Counter()
    relics = [r for r in (parse_relic(f, report) for f in files) if r]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(relics, ensure_ascii=False, indent=1), encoding="utf-8")

    energy = [r for r in relics if r["max_energy"]]
    draw = [r for r in relics if r["hand_draw"]]
    with_hooks = [r for r in relics if r["hooks"]]
    print(f"扫描 {len(files)} 个遗物类，抽取 {len(relics)} 个")
    print(f"  有数值钩子（能量）：{len(energy)}")
    print(f"  有数值钩子（抽牌）：{len(draw)}")
    print(f"  有已抽出的效果钩子：{len(with_hooks)}")
    print(f"  稀有度：{collections.Counter(r['rarity'] for r in relics).most_common()}")

    unresolved = [r["rid"] for r in relics
                  if (r["max_energy"] or {}).get("unresolved_guard")
                  or (r["hand_draw"] or {}).get("unresolved_guard")]
    if unresolved:
        print(f"  ⚠️ 条件认不出的数值遗物 {len(unresolved)} 个：{unresolved[:8]}")

    if not args.quiet:
        unsupported = {k.split("::", 1)[1]: v for k, v in report.items()
                       if k.startswith("unsupported_command::")}
        if unsupported:
            print("\n未支持的命令（按出现次数）：")
            for name, count in sorted(unsupported.items(), key=lambda kv: -kv[1])[:14]:
                print(f"    {count:4d}  {name}")

    codex_path = Path(args.codex)
    if codex_path.exists():
        codex = {c["id"].lower() for c in
                 json.loads(codex_path.read_text(encoding="utf-8"))}
        ours = {r["rid"] for r in relics}
        print(f"\n与社区库对账：社区库 {len(codex)} / 源码 {len(ours)}"
              f"  交集 {len(codex & ours)}")
        print(f"  源码有、社区库没有：{sorted(ours - codex)[:6]}")
        print(f"  社区库有、源码没有：{sorted(codex - ours)[:6]}")
    print(f"\n写出 {out}（{len(relics)} 个）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
