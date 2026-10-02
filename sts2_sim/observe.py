"""观测投影：**唯一的出口**（``docs/01`` §1.6 第二层）。

规则：
  * 本模块是唯一允许构造 ``Observation`` 的地方。
  * 只读取 ``core.CombatState`` 的**公开字段**；绝不触碰 ``state.hidden``、
    ``state.hidden.rng``、牌堆顺序、敌人行动队列、种子。
  * 所有"多重集"型信息（抽牌堆 / 弃牌堆 / 消耗堆）一律编码成**排序后的 bag**，
    数学上不含顺序信息——这是本项目里少数几个"结构上不可能出错"的设计。

信息分级（``docs/01`` §1.2）：
  L1 固定知识  → 卡牌数值、怪物 AI 模式、敌人意图（屏幕上显示）        ✅ 允许
  L2 已揭示    → 已抽到的牌、已结算的结果、前几次试次的观察            ✅ 允许
  L3 未揭示    → 剩余牌序、未来掉落、敌人行动队列、RNG 状态            ❌ 禁止
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Sequence

from .core import CombatState

#: 观察协议版本（``docs/13`` §8.1 要求观测自带协议版本）。
#:
#: ⚠️ **只能在这里定义一处**：checkpoint 也记录它，用于加载时判断
#: "这份权重的输入语义还是不是同一套"。两处各写一个常量必然漂移，
#: 而漂移的后果是旧权重在字段含义已变的输入上继续推理 —— 不报错，只是全错。
#: 改动规则：只要 ``Observation`` / ``RunObservation`` 的字段**含义**变了就 +1
#: （纯追加字段也要 +1：旧权重的输入维度对不上）。
OBSERVATION_PROTOCOL = "obs-v2"


# ==========================================================================
# 观测视图（全部 frozen + slots：构造后不可变，且字段固定）
# ==========================================================================
@dataclass(frozen=True, slots=True)
class CardView:
    cid: str
    upgraded: bool
    cost: int
    playable: bool


@dataclass(frozen=True, slots=True)
class BagView:
    """多重集视图：``((cid, upgraded, count), ...)``，**已排序，不含顺序信息**。"""

    items: tuple[tuple[str, bool, int], ...]

    @staticmethod
    def from_cards(cards: Sequence[object]) -> "BagView":
        counts: dict[tuple[str, bool], int] = {}
        for card in cards:
            key = (card.cid, card.upgraded)  # type: ignore[attr-defined]
            counts[key] = counts.get(key, 0) + 1
        items = tuple(sorted((cid, up, n) for (cid, up), n in counts.items()))
        return BagView(items)

    def total(self) -> int:
        return sum(n for _, _, n in self.items)


@dataclass(frozen=True, slots=True)
class EnemyView:
    eid: str
    hp: int
    max_hp: int
    block: int
    powers: tuple[tuple[str, int], ...]
    intent_kind: str
    intent_mid: str
    intent_value: int
    intent_times: int


@dataclass(frozen=True, slots=True)
class AttemptView:
    """前几次试次的**已揭示**观察（L2）。

    这是 SL 的知识通道：打一遍 → 看到每回合手牌（手牌顺序 = 抽牌顺序）→ 重开 →
    带着这份知识重新规划。它携带隐藏状态的信息，但**来源是合法观察**。
    """

    index: int
    hand_by_turn: tuple[tuple[str, ...], ...]
    outcome: str


@dataclass(frozen=True, slots=True)
class OrbView:
    """一个充能球（屏幕上看得见图标与数值）。"""

    oid: str
    value: int


@dataclass(frozen=True, slots=True)
class SelectionView:
    """等待选牌时的**候选列表**（审计 F09 / F04）。

    ⚠️ 关键点：候选来自**真实的选择界面**，包含
    * 来源牌堆（手牌 / 弃牌 / 抽牌）—— 旧实现把 `select_card` 的下标
      一律当成手牌下标，于是"从弃牌堆里选一张"的选择**无法正确表示**；
    * 每张候选的可见属性（卡 id 与升级状态）；
    * 还需要选几张。

    候选顺序用**玩家可见的稳定排序**，绝不使用隐藏的抽牌堆顺序。
    """

    purpose: str
    source_pile: str
    remaining: int
    candidates: tuple[CardView, ...]


@dataclass(frozen=True, slots=True)
class Observation:
    """策略能看到的**全部**内容。"""

    #: 协议版本（``docs/13`` §8.1）。决策样本、搜索与真机适配器都用它对齐。
    protocol: str

    # L1 · 局面（公开）
    turn: int
    energy: int
    player_hp: int
    player_max_hp: int
    player_block: int
    player_powers: tuple[tuple[str, int], ...]
    #: 每回合能量上限（角色定义给的，屏幕上显示为 3/3 这种）。
    player_max_energy: int

    # L1 · 手牌（内容 + 顺序；人类按顺序读牌）
    hand: tuple[CardView, ...]

    # L1/L2 · 牌堆（多重集，无顺序）
    draw_bag: BagView
    discard_bag: BagView
    exhaust_bag: BagView

    # L1 · 敌人（意图是公开信息，必须给）
    enemies: tuple[EnemyView, ...]

    # ⭐ L1 · 遗物 / 药水 / 充能球 / 星（审计 F04：这些机制引擎在结算，
    # 而模型完全看不到 —— 环境按中毒/遗物/球算，策略却没有条件选择不同动作）。
    relics: tuple[str, ...] = ()
    potions: tuple[tuple[int, str], ...] = ()
    orbs: tuple[OrbView, ...] = ()
    orb_slots: int = 0
    stars: int = 0

    # L2 · 跨试次已揭示知识（SL）
    prior_attempts: tuple[AttemptView, ...] = ()

    # 预算（环境公开的规则，不是隐藏信息）
    attempts_used: int = 0
    attempts_left: int = 0

    # L1 · 正在等待选牌时的提示（"弃 1 张" / "消耗 2 张"）。
    # ⚠️ 这不是隐藏信息：屏幕上就写着"选择一张牌弃置"。不给的话模型在选牌时
    # **无法区分**"我在选牌"和"我该出牌"，只能靠动作掩码的形状去猜 ——
    # 而掩码形状会随候选数量变化，等于让模型去猜一个可以直说的东西。
    selecting_purpose: str = ""
    selecting_remaining: int = 0
    #: 候选实体列表（含来源牌堆）。审计 F09：只给"用途 + 剩余数量"不够，
    #: 模型必须能看到**这几个候选分别是什么牌**。
    selection: SelectionView | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))


# ==========================================================================
# 投影
# ==========================================================================
def _orb_views(state: CombatState) -> tuple[OrbView, ...]:
    """充能球 → 可观察视图。球的值在屏幕上显示为图标上的数字。"""
    out: list[OrbView] = []
    for orb in getattr(state, "orbs", ()) or ():
        oid = getattr(orb, "oid", None) or getattr(orb, "kind", "")
        out.append(OrbView(oid=str(oid), value=int(getattr(orb, "value", 0))))
    return tuple(out)


def _card_view(state: CombatState, card) -> CardView:
    """一张牌的可见信息。

    ⚠️ **费用与可打性必须与合法动作同源**（外部复核 R4）：

    * ``cost`` —— ``core.play_cost`` 是**实际要付的**费用（苦痛加价 / 减费 /
      腐败归零都在里面）。原先写 ``card.cost()`` 只给卡面基础值，于是
      "腐败 + 0 能量 + 一张防御"会显示成 ``cost=1, playable=False``，
      而同一局面的合法动作里明明有"打出这张防御"。
    * ``playable`` —— ``core.card_playable``（= 真机 ``CardModel.CanPlay``）
      把所有条件收在一处：关键字、能量与星、``ShouldPlay`` 封锁、目标可用性。

    自相矛盾的观测**没有补救办法**：mask 只能告诉策略"这个动作合法"，
    却没法消除"同一张牌写着不可打"这条输入 —— 模型会学到这一列不可信。
    """
    from .core import card_playable, play_cost
    return CardView(cid=card.cid, upgraded=card.upgraded,
                    cost=play_cost(state, card),
                    playable=card_playable(state, card))


def _selection_view(state: CombatState) -> SelectionView | None:
    """候选列表。

    ⚠️ **不要在这里重新排序**：动作下标指的就是
    ``PendingSelection.candidates()`` 的顺序，观测必须与它**逐位一致**。
    顺序的公开性/稳定性由 ``PendingSelection.candidates`` 一处保证
    （手牌用玩家看到的顺序；弃牌与抽牌堆按 ``(cid, upgraded)`` 规范排序，
    因此抽牌堆的隐藏顺序不会泄漏成候选排列）。
    """
    pending = state.pending
    if pending is None:
        return None
    candidates = tuple(_card_view(state, card)
                       for card in pending.candidates(state))
    return SelectionView(purpose=pending.purpose, source_pile=pending.source_pile,
                         remaining=pending.remaining,
                         candidates=candidates)


def observe(state: CombatState, prior_attempts: Sequence[AttemptView] = (),
            attempts_used: int = 0, attempts_left: int = 0) -> Observation:
    """把完整状态投影成人类可见信息。

    ⚠️ 本函数是**唯一**允许读状态构造观测的地方。任何新增字段都要先问：
        "人类的屏幕上有没有这一项？"
    """
    player = state.player

    hand = tuple(_card_view(state, card) for card in state.hand)

    enemies = tuple(
        EnemyView(
            eid=enemy.eid,
            hp=enemy.hp,
            max_hp=enemy.max_hp,
            block=enemy.block,
            powers=tuple(sorted(enemy.powers.items())),
            intent_kind=enemy.intent.kind if enemy.intent else "unknown",
            intent_mid=enemy.intent.mid if enemy.intent else "unknown",
            intent_value=enemy.intent.value if enemy.intent else 0,
            intent_times=enemy.intent.times if enemy.intent else 0,
        )
        for enemy in state.enemies
    )

    return Observation(
        protocol=OBSERVATION_PROTOCOL,
        turn=state.turn,
        energy=state.energy,
        player_hp=player.hp,
        player_max_hp=player.max_hp,
        player_block=player.block,
        player_powers=tuple(sorted(player.powers.items())),
        player_max_energy=getattr(state, "max_energy", 3),
        hand=hand,
        draw_bag=BagView.from_cards(state.draw_pile),
        discard_bag=BagView.from_cards(state.discard),
        exhaust_bag=BagView.from_cards(state.exhaust),
        enemies=enemies,
        relics=tuple(getattr(state, "relics", ())),
        potions=tuple((slot, pid) for slot, pid in enumerate(state.potions)
                      if pid is not None),
        orbs=_orb_views(state),
        orb_slots=int(getattr(state, "orb_slots", 0)),
        stars=int(getattr(state, "stars", 0)),
        prior_attempts=tuple(prior_attempts),
        attempts_used=attempts_used,
        attempts_left=attempts_left,
        selecting_purpose=state.pending.purpose if state.pending else "",
        selecting_remaining=state.pending.remaining if state.pending else 0,
        selection=_selection_view(state),
    )


# ==========================================================================
# Run 层观测
# ==========================================================================
@dataclass(frozen=True, slots=True)
class MapNodeView:
    """地图节点的**结构**信息。

    真机上整幕地图一开始就全部可见（含类型图标），所以这属于 **L1 公开**。
    节点的**内容**（哪只怪 / 哪个事件）是 **L3**，只存在于 ``RunHidden.node_contents``。
    """

    node_id: int
    row: int
    col: int
    kind: str
    reachable: bool


@dataclass(frozen=True, slots=True)
class DeckSlotView:
    """牌组的一个**规范化槽位**。

    为什么要槽位而不是只有 bag：像"营火升级某张牌"这种动作需要一个下标。但内部
    ``deck`` 列表的顺序是**获得顺序**（玩家看不到），直接拿它当下标等于把内部存储
    顺序暴露给策略。

    解决办法：观测里按 ``(cid, upgraded)`` 规范化排序给出槽位，环境侧用**同一套排序**
    把槽位下标映射回内部下标。排序只依赖玩家可见的卡牌身份，因此不泄漏任何隐藏信息。
    完全相同（同 cid 同升级状态）的牌之间顺序无所谓——它们对玩家不可区分。
    """

    cid: str
    upgraded: bool


def deck_slot_order(deck: Sequence[object]) -> list[int]:
    """规范化槽位顺序 → 内部下标的映射。观测与动作两侧共用这一个定义。"""
    return sorted(range(len(deck)),
                  key=lambda i: (deck[i].cid, deck[i].upgraded))  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class RunObservation:
    """Run 层策略能看到的**全部**内容。

    注意：房间专属字段（卡牌候选、商店货架、事件选项、宝箱遗物）直接从 ``room``
    读取——而 ``room`` 只在**进入节点时**才由隐藏内容填充。换句话说，
    "没进去过的房间，观测里就是空的"这件事是**结构上成立**的，不需要额外判断。
    """

    #: 协议版本（与战斗观测同一个常量）。
    protocol: str

    # L1 · 进度与资源
    act: int
    floor: int
    ascension: int
    hp: int
    max_hp: int
    gold: int
    relics: tuple[str, ...]
    potions: tuple[tuple[int, str], ...]

    # L1 · 牌组（多重集，无顺序）
    deck: BagView
    deck_size: int
    deck_slots: tuple[DeckSlotView, ...]

    # L1 · 地图结构（全图可见）
    phase: str
    map_nodes: tuple[MapNodeView, ...]
    reachable: tuple[int, ...]
    #: ⭐ **完整边表**（审计 F09）：只给"当前可达"不足以规划后续路线 ——
    #: 玩家在屏幕上能看到整张图的连线，必须能给出来。
    map_edges: tuple[tuple[int, int], ...]

    # L2 · 当前房间内已展示的内容
    card_reward: tuple[str, ...]
    rest_options: tuple[str, ...]
    shop_stock: tuple[tuple[str, int], ...]
    event_id: str
    event_options: tuple[str, ...]
    relic_offered: str
    gold_offered: int

    #: ⭐ **Run 层挂起选牌**（审计 F09）。事件/奖励里的"选一张牌"原先在观测里
    #: 完全不可见 —— 模型只能从动作掩码的形状去猜"我在选牌"。
    #: 候选用 ``deck_slots`` 的**槽位编号**，与 ``legal_meta_actions`` 的
    #: ``select_deck_card(slot)`` 同一套下标（``observe.deck_slot_order``），
    #: 因此观测与动作不会各用一套编号。
    selection_purpose: str = ""
    selection_source: str = ""
    selection_remaining: int = 0
    selection_candidates: tuple[int, ...] = ()

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))


def observe_run(state) -> RunObservation:
    """把 ``run.RunState`` 投影成人类可见信息。

    绝不读取 ``state.hidden``（含 ``node_contents`` 与 ``rng``）。
    """
    player = state.player
    room = state.room
    position = state.position
    reachable = state.map.start if position is None else state.map.by_id(position).edges
    reachable_set = set(reachable)

    map_nodes = tuple(
        MapNodeView(node_id=node.node_id, row=node.row, col=node.col,
                    kind=node.kind, reachable=node.node_id in reachable_set)
        for node in state.map.nodes
    )
    # 边表：从每个节点指向它的后继。**拓扑是公开信息**（整幕地图一开始就可见），
    # 缺了它策略就无法回答"走这条路后面能到哪"。
    map_edges = tuple(
        (node.node_id, child) for node in state.map.nodes for child in node.edges
    )

    # 挂起选牌：候选转成**槽位编号**（与动作同一套下标）。
    selection = getattr(state, "pending_selection", None)
    if selection is not None:
        order = deck_slot_order(player.deck)
        slot_of = {internal: slot for slot, internal in enumerate(order)}
        candidate_slots = tuple(sorted(
            slot_of[internal] for internal in selection.candidates
            if internal in slot_of))
        selection_purpose = selection.purpose
        selection_source = selection.source
        selection_remaining = selection.count - len(selection.chosen)
    else:
        candidate_slots = ()
        selection_purpose = selection_source = ""
        selection_remaining = 0

    return RunObservation(
        protocol=OBSERVATION_PROTOCOL,
        act=state.act,
        floor=state.floor,
        ascension=state.ascension,
        hp=player.hp,
        max_hp=player.max_hp,
        gold=player.gold,
        relics=tuple(player.relics),
        potions=tuple((slot, potion) for slot, potion in enumerate(player.potions)
                      if potion is not None),
        deck=BagView.from_cards(player.deck),
        deck_size=len(player.deck),
        deck_slots=tuple(DeckSlotView(player.deck[i].cid, player.deck[i].upgraded)
                         for i in deck_slot_order(player.deck)),
        phase=room.kind,
        map_nodes=map_nodes,
        reachable=tuple(reachable),
        map_edges=map_edges,
        card_reward=tuple(room.card_reward),
        rest_options=tuple(room.rest_options),
        shop_stock=tuple(room.shop_stock),
        event_id=room.event_id,
        event_options=tuple(room.event_options),
        relic_offered=room.relic,
        gold_offered=room.gold_gained,
        selection_purpose=selection_purpose,
        selection_source=selection_source,
        selection_remaining=selection_remaining,
        selection_candidates=candidate_slots,
    )


# ==========================================================================
# 禁止泄漏的字段名
# ==========================================================================
#: 观测中**绝对不允许**出现的字段名（CI 里做字节级扫描，见 tests/test_anticheat.py）
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "seed", "rng", "hidden", "master", "getstate", "future", "move_queue",
    "shuffle", "node_contents", "encounter_id",
)
