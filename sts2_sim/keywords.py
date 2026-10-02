"""卡牌关键字（``CardKeyword``）的真机行为。

铁律（``docs/09`` §5.2）：**机制声明必须带源码出处**，且语义不能"看起来合理"。
本模块每一条都给出真机的**唯一执行点**，实现照着它写。

枚举与官方语义说明在 ``MegaCrit.Sts2.Core.Entities.Cards/CardKeyword.cs``：
"Card Keywords are extra 'automatic behaviors' you can add to a card" ——
也就是说**七个关键字全部都有行为**，不是纯文本标签：

* ``Exhaust``：打出后进消耗堆（枚举 doc comment 明写 "automatically exhausts when played"）
* ``Unplayable``："blocks the card from being played"
* 其余五个的执行点如下表

======================  ==========================================================
关键字                  真机执行点
======================  ==========================================================
``Ethereal``            ``CombatManager.DoTurnEnd``：手里带 Ethereal 且
                        ``Hook.ShouldEtherealTrigger`` 的牌 → ``CardCmd.Exhaust(causedByEthereal: true)``
``Innate``              ``CombatManager.SetupPlayerTurn``（第 1 回合）：抽牌堆里带 Innate 的牌
                        ``MoveToTopInternal``，且 ``handDraw = max(handDraw, 张数)``
``Retain``              ``CombatManager.FlushPlayerHand``：``ShouldRetainThisTurn`` 为真则**不弃**
``Sly``                 ``CardCmd.DiscardAndDraw``：弃完之后对每张 ``IsSlyThisTurn`` 的牌
                        ``CardCmd.AutoPlay(…, AutoPlayType.SlyDiscard)``
``Eternal``             ``CardModel.IsRemovable`` → ``false``；``IsTransformable`` 在**牌组里**也为 false
``Exhaust``             ``CardModel.GetResultLocationForCardPlay`` → ``PileType.Exhaust``
``Unplayable``          ``CardModel.CanPlay`` → ``UnplayableReason.HasUnplayableKeyword``；
                        被 AutoPlay 时走 ``MoveToResultPileWithoutPlaying``（**不结算效果**）
======================  ==========================================================

⚠️ 两个容易搞错的细节（都以源码为准）：

* ``Sly`` 只在 **``CardCmd.Discard`` 这条路**上触发。回合结束的"清空手牌"走的是
  ``CardPileCmd.Add(cardsToFlush, PileType.Discard)``（``FlushPlayerHand``），
  **不触发 Sly** —— 把两者混为一谈会让 Sly 牌每回合白打一次。
* ``Ethereal`` 的消耗发生在 ``DoTurnEnd`` 的**前面一步**，而清空手牌在
  ``EndPlayerTurnPhaseTwo``；顺序反了会让"回合结束时在手里触发"的牌
  （``burn`` 等）不再触发，因为牌已经被弃掉了。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeywordSpec:
    """一个关键字的真机语义与出处。"""

    #: 真机枚举名（``CardKeyword`` 的取值）。
    name: str
    #: 中文名（报告里用）。
    zh: str
    #: 真机执行点，``文件:方法`` 形式（铁律 R1）。
    source: str
    #: 行为说明（中文）。
    behavior: str


#: 七个关键字的权威表。**顺序照抄真机枚举**（``CardKeyword.cs`` 第 20-27 行）。
KEYWORDS: tuple[KeywordSpec, ...] = (
    KeywordSpec(
        "Exhaust", "消耗",
        "MegaCrit.Sts2.Core.Models/CardModel.cs:GetResultLocationForCardPlay",
        "打出后进消耗堆，不进弃牌堆；``ExhaustOnNextPlay`` 也能一次性触发同一行为。",
    ),
    KeywordSpec(
        "Ethereal", "虚无",
        "MegaCrit.Sts2.Core.Combat/CombatManager.cs:DoTurnEnd",
        "回合结束时若还在手里（且 ``Hook.ShouldEtherealTrigger`` 未否决）→ 消耗。",
    ),
    KeywordSpec(
        "Innate", "固有",
        "MegaCrit.Sts2.Core.Combat/CombatManager.cs:SetupPlayerTurn",
        "仅第 1 回合：抽牌堆里带固有关键字的牌移到**堆顶**，"
        "且起手抽牌数 = ``max(5, 固有条件数)``。",
    ),
    KeywordSpec(
        "Unplayable", "不可打出",
        "MegaCrit.Sts2.Core.Models/CardModel.cs:CanPlay",
        "``CanPlay`` 直接给出 ``HasUnplayableKeyword``，不能主动打出；"
        "被自动打出时只落牌堆、不结算效果。",
    ),
    KeywordSpec(
        "Retain", "保留",
        "MegaCrit.Sts2.Core.Combat/CombatManager.cs:FlushPlayerHand",
        "``ShouldRetainThisTurn`` 为真时回合结束**不清掉**这张牌（留在手里）。",
    ),
    KeywordSpec(
        "Sly", "狡诈",
        "MegaCrit.Sts2.Core.Commands/CardCmd.cs:DiscardAndDraw",
        "被 ``CardCmd.Discard`` 弃掉之后，对这张牌执行一次 ``AutoPlay``（SlyDiscard）。",
    ),
    KeywordSpec(
        "Eternal", "永恒",
        "MegaCrit.Sts2.Core.Models/CardModel.cs:IsRemovable",
        "``IsRemovable`` 为 false（不能被移除）；在牌组里也不能被转化（``IsTransformable``）。",
    ),
)

#: 关键字名 → 说明。
KEYWORD_BY_NAME: dict[str, KeywordSpec] = {k.name: k for k in KEYWORDS}

#: 引擎**已实现**的关键字。全部七个都实现在 ``core`` / ``runeffects`` 里。
#:
#: 这张表是"内容侧诚实性"的判据：卡牌数据里出现表外的关键字时，
#: ``content`` 会把这张卡标成 ``keywords_incomplete`` 并排除出训练集 ——
#: 未知关键字静默失效会让模拟器比真机简单。
ENGINE_KEYWORDS: frozenset[str] = frozenset(KEYWORD_BY_NAME)


def keywords_of(definition) -> frozenset[str]:
    """取一张卡的关键字集合（``CardDef.keywords``）。"""
    return frozenset(getattr(definition, "keywords", ()) or ())


def is_unplayable(definition) -> bool:
    """``CardModel.CanPlay``：带 Unplayable 就只能落牌堆。"""
    return "Unplayable" in keywords_of(definition)


def is_ethereal(definition) -> bool:
    """``CombatManager.DoTurnEnd``：回合结束在手里则消耗。"""
    return "Ethereal" in keywords_of(definition)


def should_retain(definition) -> bool:
    """``CardModel.ShouldRetainThisTurn``（静态关键字部分）。

    真机还有 ``HasSingleTurnRetain``（``Expertise`` 这类一次性效果给的），
    引擎侧对应的是"本回合有效"的临时标记；静态关键字这一半是这里的判据。
    """
    return "Retain" in keywords_of(definition)


def retains(card) -> bool:
    """这张**牌实例**回合结束要不要留在手里（``CardModel.ShouldRetainThisTurn``）。

    ⚠️ 必须同时看**两处**：

    * ``CardDef.keywords`` —— 卡牌固有（``Retain`` 标签写在卡上）；
    * ``CardInstance.keywords`` —— **战斗内**用 ``CardCmd.ApplyKeyword`` 加的
      （``PhantomBladesPower`` 给飞刀加 ``Retain``）。

    只看固有那一处，飞刀在"幻影之刃"下回合结束就会被弃掉 —— 而且是静默的
    （牌还在、只是少留了一回合）。
    """
    if "Retain" in tuple(getattr(card, "keywords", ()) or ()):
        return True
    definition = card.definition() if hasattr(card, "definition") else card
    return should_retain(definition)


def is_sly(definition) -> bool:
    """``CardModel.IsSlyThisTurn``（静态关键字部分）。"""
    return "Sly" in keywords_of(definition)


def is_sly_card(card) -> bool:
    """这张**牌实例**是不是狡诈（``CardModel.IsSlyThisTurn``）。

    ⚠️ 与 :func:`retains` 同一个道理：``CardDef.keywords`` 是固有属性，
    而 ``MasterPlannerPower`` 是用 ``CardCmd.ApplyKeyword(card, Sly)`` 在
    **战斗内给单张牌**加的 —— 只看卡牌定义，"策划大师"打出的技能牌
    被弃掉时不会自动打出（静默失效）。
    """
    if "Sly" in tuple(getattr(card, "keywords", ()) or ()):
        return True
    definition = card.definition() if hasattr(card, "definition") else card
    return is_sly(definition)


def is_innate(definition) -> bool:
    """``CombatManager.SetupPlayerTurn``：第 1 回合起手。"""
    return "Innate" in keywords_of(definition)


def is_eternal(definition) -> bool:
    """``CardModel.IsRemovable`` 的取反。"""
    return "Eternal" in keywords_of(definition)


def unknown_keywords(definition) -> list[str]:
    """卡牌带了、但引擎还没实现的关键字（内容侧诚实性的判据）。"""
    return sorted(k for k in keywords_of(definition) if k not in ENGINE_KEYWORDS)
