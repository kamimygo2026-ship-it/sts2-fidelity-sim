"""Run 层：地图推进、战斗、卡牌奖励、营火、商店、事件、宝箱。

对应 ``docs/04`` 的 Run 层（决策频率 ~50–100 次/局，奖励极稀疏）。
战斗节点内部委托给 ``core`` 的 ``CombatState``，本模块只管**房间之间**的流转。

SL 集成：``restart_node()`` 回到节点入口、**保留隐藏随机状态**——与
``env.SpireEnv`` 的战斗级重开语义一致（``docs/01`` §1.3）。

⚠️ 节点内容（哪只怪 / 哪个事件 / 什么掉落）属于 **L3 未揭示**，存放在
``RunHidden.node_contents``，只有 ``_enter_node()`` 能读。观测里绝不允许出现。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Sequence

from .content import (
    ACTS, ASCENSION_TABLE, CARD_DB, ENCOUNTERS, EVENT_DB, RARITY_WEIGHTS,
    RELIC_POOL, STARTING_DECK, event_pool,
)
from .acts import ActDef, random_act_list
from .core import (
    Action, CardInstance, CombatState, apply_permanent_changes,
    legal_actions as combat_legal_actions, render_text, start_combat,
    step as combat_step,
)
from .mapgen import (
    ANCIENT, BOSS, ELITE, MONSTER, REST_SITE, SHOP, TREASURE, UNKNOWN,
    MapGraph, generate_map,
)
from .pointodds import EVENT, UnknownMapPointOdds, build_blacklist
from .rng import RngSet

PLAYER_MAX_HP = 80
STARTING_GOLD = 99
REST_HEAL_RATIO = 0.30
POTION_SLOTS = 3

#: 房间阶段
PHASE_MAP = "map"
PHASE_COMBAT = "combat"
PHASE_CARD_REWARD = "card_reward"
#: **待选奖励**（`RewardsCmd.OfferCustom`）：一组候选让玩家挑一个。
#: 与 `PHASE_CARD_REWARD` 的区别：那里固定是"三选一卡牌"；
#: 这里可以是**遗物 / 药水 / 卡牌**，而且可能连着好几组
#: （召唤铃一次给"普通/罕见/稀有遗物"各一个候选）。
PHASE_REWARD = "reward"
PHASE_REST = "rest"
PHASE_SHOP = "shop"
PHASE_EVENT = "event"
PHASE_TREASURE = "treasure"
PHASE_WON = "won"
PHASE_LOST = "lost"


# ==========================================================================
# 隐藏信息（Run 层）
# ==========================================================================
class RunHidden:
    """Run 层的 L3 容器：**绝不可进入观测**。"""

    __slots__ = ("rng", "node_contents", "point_odds")

    def __init__(self, rng: RngSet) -> None:
        self.rng = rng
        self.node_contents: dict[int, "NodeContents"] = {}
        #: 未知点的房间概率（``RunOddsSet.UnknownMapPoint``）。它跨房间累计、
        #: 还握着一把专用 RNG，属于典型的 L3 状态 —— 放进隐藏区，
        #: 免得哪天顺手写进观测。
        #: ⚠️ 真机也用 ``unknown_map_point`` 这条**专用流**，不是 ``up_front``。
        self.point_odds = UnknownMapPointOdds(rng.named("unknown_map_point"))


@dataclass
class NodeContents:
    encounter: tuple[str, ...] = ()
    card_reward: tuple[str, ...] = ()
    relic: str = ""
    event_id: str = ""
    shop_stock: tuple[tuple[str, int], ...] = ()
    gold: int = 0


# ==========================================================================
# 房间
# ==========================================================================
@dataclass
class Room:
    kind: str
    encounter: tuple[str, ...] = ()
    card_reward: tuple[str, ...] = ()
    rest_options: tuple[str, ...] = ()
    shop_stock: tuple[tuple[str, int], ...] = ()
    event_id: str = ""
    event_options: tuple[str, ...] = ()
    relic: str = ""
    gold_gained: int = 0
    resolved: bool = False


# ==========================================================================
# 玩家 / 运行状态
# ==========================================================================
@dataclass
class RunPlayer:
    hp: int
    max_hp: int
    gold: int
    deck: list[CardInstance]
    relics: list[str] = field(default_factory=list)
    potions: list[str | None] = field(default_factory=lambda: [None] * POTION_SLOTS)
    #: 角色资源：每回合能量上限（``docs/13`` §3.1 列出的是"玩家 HP/最大 HP/金币、
    #: 永久牌组实例、遗物实例与持久计数、药水槽、解锁配置"；能量上限属于角色定义，
    #: 必须显式传递而不是在战斗里写死 3）。
    energy: int = 3
    #: 遗物的**持久计数**（``docs/13`` §3.1）：`(遗物 id, 计数器名) → 值`。
    #: 真机像 ``GeneticAlgorithm``（每场战斗涨格挡）这类遗物要跨战斗累计，
    #: 记在战斗里就会每场归零。
    relic_counters: dict[tuple[str, str], int] = field(default_factory=dict)


@dataclass
class RewardGroup:
    """一组待选奖励：``kind`` + 候选列表（玩家挑一个，或跳过）。"""

    kind: str                       # relic | potion | card
    options: tuple[str, ...] = ()
    #: 候选是怎么来的（写进日志，便于对账）
    note: str = ""


@dataclass
class RunState:
    seed: int
    ascension: int
    act: int
    floor: int
    player: RunPlayer
    map: MapGraph
    hidden: RunHidden
    #: 本局的幕序列（真机 ``RunState.Acts``）：索引 0 是**密林或暗港**二选一，
    #: 索引 1 是蜂巢，索引 2 是荣耀。``act`` 是 **1 起的幕序号**，不是索引。
    acts: tuple[ActDef, ...] = ()
    position: int | None = None
    room: Room = field(default_factory=lambda: Room(kind=PHASE_MAP))
    combat: CombatState | None = None
    visited: list[int] = field(default_factory=list)
    #: 已经走过的**未知点**数量（教学硬规则要用它，``UnknownMapPointOdds``）。
    unknown_visited: int = 0
    #: 上一个进过的房间类型（``BuildRoomTypeBlacklist`` 要看它）。
    last_room_kind: str = ""
    #: 非最终幕 Boss 的奖励领完之后**要进下一幕**（``_after_card_reward``）。
    pending_act_advance: bool = False
    #: 待选奖励队列（`RewardsCmd.OfferCustom`）。非空时房间阶段是 `PHASE_REWARD`。
    pending_rewards: list["RewardGroup"] = field(default_factory=list)
    #: 当前事件的进行态（``sts2_sim.events.EventRun``）。**属于 L2 可见信息**：
    #: 当前在哪一页、哪些选项点过了，玩家在界面上都看得到。
    event_run: object | None = None
    #: 等待玩家选牌（``runeffects.RunSelection``）。非空时事件选项**不可点**，
    #: 合法动作是"从牌组里选一张"。
    pending_selection: object | None = None
    #: 遗物抓包（``relicbag.RelicGrabBag``）：本局能出的遗物按稀有度分桶，
    #: 取走就不再出现（``RelicGrabBag.cs``）。
    relic_bag: object | None = None
    #: 事件过程的文字记录（可读、可审计；不进观测）
    event_log: list[str] = field(default_factory=list)

    @property
    def current_act(self) -> ActDef | None:
        """本幕定义（``RunState.Act``）。没有幕表时返回 ``None``（内置内容）。"""
        if 1 <= self.act <= len(self.acts):
            return self.acts[self.act - 1]
        return None


# ==========================================================================
# 动作
# ==========================================================================
@dataclass(frozen=True, slots=True)
class MetaAction:
    kind: str          # choose_node | pick_card | skip_card | rest | smith | buy | event_option | proceed
    index: int = -1

    def __repr__(self) -> str:
        return f"{self.kind}({self.index})" if self.index >= 0 else self.kind


def legal_meta_actions(state: RunState) -> list[MetaAction]:
    """当前房间的合法动作。**mask 是硬性要求**（``docs/03`` §3.5）。"""
    room = state.room
    # ⭐ **挂起的选牌优先于房间**：真机的选牌界面盖在当前房间之上（事件页、
    # 战斗结束的移除奖励都一样）。以前只有事件房间认这一条，于是"战斗结束的
    # 移除奖励"会让 run 停在卡牌奖励页、**一个合法动作都没有**（堵死）。
    if state.pending_selection is not None and room.kind not in (PHASE_WON, PHASE_LOST):
        from .observe import deck_slot_order

        order = deck_slot_order(state.player.deck)
        candidates = set(state.pending_selection.candidates)
        return [MetaAction("select_deck_card", slot)
                for slot, internal in enumerate(order) if internal in candidates]
    if room.kind == PHASE_MAP:
        return [MetaAction("choose_node", node_id) for node_id in reachable_nodes(state)]
    if room.kind == PHASE_CARD_REWARD:
        actions = [MetaAction("pick_card", i) for i in range(len(room.card_reward))]
        actions.append(MetaAction("skip_card"))
        return actions
    if room.kind == PHASE_REWARD:
        group = state.pending_rewards[0] if state.pending_rewards else None
        if group is None:
            return []
        actions = [MetaAction("take_reward", i) for i in range(len(group.options))]
        actions.append(MetaAction("skip_reward"))
        return actions
    if room.kind == PHASE_REST:
        actions = [MetaAction("rest")]
        from .observe import deck_slot_order
        order = deck_slot_order(state.player.deck)
        upgradable = [slot for slot, internal in enumerate(order)
                      if not state.player.deck[internal].upgraded]
        actions.extend(MetaAction("smith", slot) for slot in upgradable)
        return actions
    if room.kind == PHASE_SHOP:
        affordable = [MetaAction("buy", i)
                      for i, (_, price) in enumerate(room.shop_stock)
                      if price <= state.player.gold]
        affordable.append(MetaAction("proceed"))
        return affordable
    if room.kind == PHASE_EVENT:
        # ⭐ **先看有没有挂起的选牌**：那时选项不可点，能做的只有"选一张牌"。
        # 真机也是这个交互（选牌界面盖住事件页）。
        if state.pending_selection is not None:
            from .observe import deck_slot_order

            order = deck_slot_order(state.player.deck)
            candidates = set(state.pending_selection.candidates)
            return [MetaAction("select_deck_card", slot)
                    for slot, internal in enumerate(order) if internal in candidates]
        # ⚠️ **只列出能点的选项**：锁住的（真机 `IsLocked`）和已经点过的
        # （真机 `WasChosen` 守卫）都不给 —— mask 是硬性要求，越界的动作
        # 必须在动作空间里就不存在，而不是"发出去再报错"。
        from . import events as event_rules

        definition = EVENT_DB.get(room.event_id)
        run = state.event_run
        if definition is None or run is None:
            return []
        return [MetaAction("event_option", i)
                for i in event_rules.selectable(definition, run)]
    if room.kind == PHASE_TREASURE:
        return [MetaAction("proceed")]
    return []


def reachable_nodes(state: RunState) -> tuple[int, ...]:
    if state.position is None:
        return state.map.start
    return state.map.by_id(state.position).edges


# ==========================================================================
# 内容抽取（每条都只消费指定的 RNG 流——可测，见 tests/test_run.py）
# ==========================================================================
#: 卡牌奖励的稀有度概率 —— **抄自** ``MegaCrit.Sts2.Core.Odds.CardRarityOdds``：
#:
#:     regularCommonOdds = 0.6   regularUncommonOdds = 0.37  RegularRareOdds = 0.03
#:     EliteCommonOdds   = 0.5   eliteUncommonOdds   = 0.4   EliteRareOdds   = 0.1
#:     bossRareOdds = 1.0（Boss 奖励**必定**稀有）
#:     ShopCommonOdds    = 0.54  shopUncommonOdds    = 0.37  ShopRareOdds    = 0.09
#:
#: ⚠️ 真机还有一套**保底偏移**（``_baseRarityOffset = -0.05`` / ``RarityGrowth = 0.01`` /
#: ``_maxRarityOffset = 0.4``）：一直不出稀有会把稀有概率逐步抬上去。
#: 引擎**尚未实现**它 —— 这是已知缺口，不是"已经一致"。
CARD_RARITY_ODDS: dict[str, tuple[tuple[str, float], ...]] = {
    "monster": (("common", 0.6), ("uncommon", 0.37), ("rare", 0.03)),
    "elite": (("common", 0.5), ("uncommon", 0.4), ("rare", 0.1)),
    "boss": (("rare", 1.0),),
    "shop": (("common", 0.54), ("uncommon", 0.37), ("rare", 0.09)),
}


def draw_card_reward(rng: RngSet, kind: str = "monster", count: int = 3,
                     character: str = "ironclad",
                     pool_name: str | None = None) -> tuple[str, ...]:
    """卡牌奖励。

    ⚠️ **必须从角色自己的卡池里抽**。早先的实现从 ``CARD_DB`` 全体里按稀有度抽，
    于是铁甲战士会被给到静默 / 缺陷 / 摄政王的牌 —— 而奖励选牌是整局最重要的
    决策，池子错了等于在另一个游戏里训练。

    ``pool_name`` 是**源码明确指定的卡池**（``CardCreationOptions`` 里的
    ``ModelDb.CardPool<ColorlessCardPool>()`` → ``ColorlessCardPool``）：
    事件奖励常常从无色池出牌（``BrainLeech.Rip``），拿角色池代替就是换了个池子。

    三张**互不重复**（真机 ``GetDistinctForCombat`` / 奖励生成都保证不重复），
    只消费 ``rewards`` 流（真机 ``PlayerRngType.Rewards``，见 ``rng.STREAMS``）。

    ⚠️ **池子要过门禁**（外部复核 R3）：原先直接从 ``CARD_POOLS`` 抽，
    于是 ``RngSet(0)`` 会发出 ``demonic_shield`` —— 一张引擎执行不了的牌。
    现在统一用 ``eligibility.reward_pool``（合格 + 可抽），
    与训练采样器**同一个池子**，所以"发出去的牌算不对"在结构上不可能。
    """
    from .content import character_pool
    from .eligibility import reward_pool

    pool = reward_pool(pool_name, character)
    if not pool:
        raise KeyError(f"没有可发放的卡池 {pool_name or character_pool(character)!r}"
                       f"（角色 {character!r}）")
    odds = CARD_RARITY_ODDS.get(kind, CARD_RARITY_ODDS["monster"])
    by_rarity: dict[str, list[str]] = {}
    for cid in pool:
        definition = CARD_DB.get(cid)
        if definition is not None:
            by_rarity.setdefault(definition.rarity, []).append(cid)

    picked: list[str] = []
    for _ in range(count):
        rarity = rng.weighted_choice(list(odds), "rewards")
        candidates = [c for c in by_rarity.get(rarity, []) if c not in picked]
        if not candidates:
            # 该稀有度抽空了 → 退回池里任一没抽过的卡（真机同样会挑别的）
            candidates = [c for c in pool if c not in picked]
        if not candidates:
            break
        picked.append(rng.choice(candidates, "rewards"))
    return tuple(picked)


def draw_encounter(rng: RngSet, kind: str) -> tuple[str, ...]:
    """遭遇选择。**只消费 ``monster`` 流**。"""
    table = ENCOUNTERS["boss" if kind == BOSS else ("elite" if kind == ELITE else "monster")]
    return rng.choice(table, "up_front")


def draw_gold(rng: RngSet) -> int:
    return rng.randint(10, 20, "niche")


def draw_relic(rng: RngSet) -> str:
    return rng.choice(RELIC_POOL, "up_front")


def draw_event(rng: RngSet, act_id: str | None = None) -> str:
    """抽一个事件。池子 = **本幕**可用的事件（``content.event_pool``）。

    ⚠️ 传了 ``act_id`` 就用 ``acts_source.json`` 里那张幕的事件表当**唯一**依据：
    真机的事件池是 ``ActModel.AllEvents + ModelDb.AllSharedEvents``
    （``ActModel.cs:334``），逐幕不同。早先引擎只有"第 1 幕池"一个概念，
    后两幕会抽到不该出现的事件。

    池子为空时分两种情况，**必须区分**：

    * 内置占位内容（``CONTENT_SOURCE == "builtin"``，测试用）本来就**没有事件**，
      返回 ``""`` 表示"这个节点没有事件"，Run 层照常继续；
    * 真实内容下池子为空 = 内容加载坏了，**抛错** —— 那种情况下"照常能跑"
      只会让人以为事件在做。
    """
    pool = event_pool(act_id)
    if not pool:
        from .content import CONTENT_SOURCE

        if CONTENT_SOURCE == "builtin":
            return ""
        raise RuntimeError("事件池为空：events_source.json 没加载或全部不可用")
    return rng.choice(pool, "up_front")


#: 商店卡片基础价 —— 抄自 ``MerchantCardEntry.GetCost``：
#:
#:     CardRarity.Rare => 150, CardRarity.Uncommon => 75, _ => 50
#:     无色池的牌再 × 1.15
#:
#: 实价 = 基础价 × ``Shops.NextFloat(0.95, 1.05)``（打折再 /2）。
#: ⚠️ 旧实现写的是 ``rng.randint(45, 90, "niche")`` —— **凭空编的**，
#: 既不按稀有度、也不按真机的流。
CARD_BASE_PRICE: dict[str, int] = {"rare": 150, "uncommon": 75}
CARD_DEFAULT_PRICE = 50
#: 无色池的牌贵 15%
COLORLESS_PRICE_MULTIPLIER = 1.15


def card_price(rng: RngSet, cid: str, character: str = "ironclad",
               on_sale: bool = False) -> int:
    """一张牌在商店里的价格（``MerchantCardEntry.CalcCost``）。"""
    definition = CARD_DB.get(cid)
    base = CARD_BASE_PRICE.get(definition.rarity if definition else "",
                               CARD_DEFAULT_PRICE)
    from .content import CARD_POOLS, character_pool
    if cid in set(CARD_POOLS.get("ColorlessCardPool", ())):
        base = round(base * COLORLESS_PRICE_MULTIPLIER)
    # ⚠️ 乘法走 `shops` 流（真机 `PlayerRng.Shops`），不是 `niche`
    price = round(base * rng.uniform(0.95, 1.05, "shops"))
    return price // 2 if on_sale else price


#: 商店货架的**类型结构** —— 抄自 ``MerchantInventory._coloredCardTypes``：
#: 5 个格子，类型是「攻击 / 攻击 / 技能 / 技能 / 能力」。
#:
#: ⚠️ 商店不是"随机 3 张"，而是**每种类型各一格**。旧实现按稀有度随机抽 3 张，
#: 结构就不对：真机永远不会出现"五个格子全是能力牌"。
SHOP_SLOT_TYPES: tuple[str, ...] = ("attack", "attack", "skill", "skill", "power")
SHOP_CARD_COUNT = len(SHOP_SLOT_TYPES)
#: 商店不卖基础牌（``options.Where(c => c.Rarity != CardRarity.Basic)``）
SHOP_EXCLUDED_RARITIES = frozenset({"basic", "curse", "status", "quest"})


def draw_shop_stock(rng: RngSet, character: str = "ironclad",
                    ) -> tuple[tuple[str, int], ...]:
    """商店货架的卡牌部分（``MerchantInventory`` + ``CardFactory.CreateForMerchant``）。

    每格的流程与真机一致：

    1. 用**商店概率**摇一个稀有度（0.54 / 0.37 / 0.09）
    2. 在「该稀有度 + 该格类型」的池子里选一张
    3. 该组合空了 → 退到下一个允许的稀有度（真机 ``GetNextAllowedRarity``）
    4. 价格 = 基础价 × ``Shops.NextFloat(0.95, 1.05)``

    ⚠️ 价格旧实现是 ``rng.randint(45, 90, "niche")`` —— 凭空编的。

    ⚠️ 货架同样要**过门禁**（外部复核 R3）：与卡牌奖励共用
    ``eligibility.reward_pool``。两个入口各用一个池子时，
    商店会卖出引擎执行不了的牌，而奖励不会 —— 症状只出现在商店里。
    """
    from .content import CARD_DB as _CARDS
    from .eligibility import reward_pool

    sellable = [c for c in reward_pool(None, character)
                if _CARDS[c].rarity not in SHOP_EXCLUDED_RARITIES]
    if not sellable:
        raise KeyError(f"没有可出售的卡池（角色 {character!r}）")
    odds = CARD_RARITY_ODDS["shop"]
    # 稀有度从低到高的"退让顺序"（真机 GetNextAllowedRarity 也是往上找）
    order = [rarity for rarity, _ in odds]

    stock: list[tuple[str, int]] = []
    for slot_type in SHOP_SLOT_TYPES:
        rarity = rng.weighted_choice(list(odds), "shops")
        candidates: list[str] = []
        for candidate_rarity in order[order.index(rarity):] + order[:order.index(rarity)]:
            candidates = [c for c in sellable
                          if _CARDS[c].rarity == candidate_rarity
                          and _CARDS[c].card_type == slot_type]
            if candidates:
                break
        if not candidates:
            continue
        cid = rng.choice(candidates, "shops")
        stock.append((cid, card_price(rng, cid, character)))
    return tuple(stock)


# ==========================================================================
# 环境
# ==========================================================================
class RunEnv:
    """一个完整的 run：**3 个幕索引 / 4 张幕地图**（密林|暗港 → 蜂巢 → 荣耀）。

    真机一局只走 3 个幕索引，但索引 0 有两张地图（密林 ``Overgrowth`` 与暗港
    ``Underdocks``）由 ``ActModel.GetRandomList`` 抽一张，所以一共 4 张幕地图。
    末幕 Boss 倒下才算通关（``docs/12`` F08）。
    """

    def __init__(self, seed: int = 0, ascension: int = 0,
                 deck: Sequence[str] | None = None,
                 attempt_budget: int = 64,
                 content_mode: str = "predetermined",
                 character: str = "ironclad",
                 unlocked_acts: frozenset[str] | None = None,
                 discovered_acts: frozenset[str] | None = None,
                 first_act: str | None = None) -> None:
        if content_mode not in ("predetermined", "on_entry"):
            raise ValueError("content_mode ∈ {predetermined, on_entry}")
        self._seed = seed
        self._ascension = ascension
        #: ⚠️ **角色决定了卡牌奖励从哪个池子里抽**。早先 Run 层没有这个概念，
        #: 于是奖励从"全部卡牌"里抽 —— 铁甲战士会被给到静默/缺陷的牌。
        self._character = character
        #: ``None`` = **用角色定义里的起始牌组**（审计 F02）。以前默认写死
        #: ``tuple(STARTING_DECK)``，那是"当前加载的默认角色"的牌组，
        #: 于是 `RunEnv(character="silent")` 拿到的是铁甲战士的牌。
        self._deck = tuple(deck) if deck is not None else None
        self._attempt_budget = attempt_budget
        self._content_mode = content_mode
        #: 存档层（引擎没有）：默认当作"暗港已解锁且见过"的普通存档，
        #: 于是索引 0 在密林/暗港之间等概率抽（真机 ``GetRandomList``）。
        #: 传 ``unlocked_acts=frozenset()`` 就是"暗港还没解锁"。
        self._unlocked_acts = (frozenset({"UnderdocksEpoch"}) if unlocked_acts is None
                              else unlocked_acts)
        self._discovered_acts = (frozenset({"underdocks"}) if discovered_acts is None
                                 else discovered_acts)
        #: 强制第一幕（真机大厅里玩家可以选：``StartRunLobby.GetAct``）。
        self._first_act = first_act

        self._state: RunState | None = None
        self._node_snapshot: RunState | None = None
        self._nodes_restarted = 0

    # ---- 随机数的**唯一所有者** ----------------------------------------
    @property
    def _rng(self) -> RngSet:
        """Run 的持久随机状态 —— **只有** ``state.hidden.rng`` 这一份。

        出处与理由：真机的 Run 存档把随机数状态保存在 ``RunState`` 里
        （``RunManager`` 的持久状态），恢复一个节点就是恢复那一份状态；
        **环境外壳不该再留一个可推进的引用**。

        ⚠️ 以前这里是一个真字段，``reset()`` 里让 ``self._rng`` 与
        ``state.hidden.rng`` 指向同一对象，而 ``restart_node()`` 只
        ``deepcopy`` 恢复 ``_state`` —— 于是重开之后两个引用**分裂**：
        ``_begin_act`` 用旧的那份预生成内容、其它逻辑用恢复出来的那份，
        同一个节点快照重开出**不同的第二幕**（实测 seed=4 下 28 个节点全不同）。
        改成只读属性之后，"第二个所有者"在结构上就不存在了 ——
        不是靠记得去同步，而是根本没有第二份可写。
        构造函数与 ``reset()`` 都**不许**再赋值给 ``_rng``（会直接抛
        ``AttributeError``，这正是我们要的：谁再引入第二所有者就立刻炸）。
        """
        if self._state is None:
            raise RuntimeError("Run 还没 reset()：此刻还没有持久随机状态")
        return self._state.hidden.rng

    # ---- 只读 ---------------------------------------------------------
    @property
    def raw_state(self) -> RunState:
        """⚠️ 仅供测试与调试。策略路径不得使用。"""
        assert self._state is not None
        return self._state

    @property
    def character(self) -> str:
        return self._character

    @property
    def nodes_restarted(self) -> int:
        return self._nodes_restarted

    @property
    def attempts_left(self) -> int:
        return max(0, self._attempt_budget - self._nodes_restarted)

    @property
    def phase(self) -> str:
        assert self._state is not None
        return self._state.room.kind

    # ---- 生命周期 -----------------------------------------------------
    def starting_state(self) -> dict:
        """这个角色的**初始状态**（HP / 金币 / 牌组 / 遗物）。

        审计 F02：``RunEnv.reset()`` 一律用 HP 80 + 铁甲战士牌组 + 空遗物，
        角色参数只影响奖励池。这里统一从 ``CHARACTERS`` 取；内置占位内容
        没有角色表，退回占位常量（并且只在 builtin 下发生）。
        """
        from .content import CHARACTERS, CONTENT_SOURCE, DEFAULT_CHARACTER

        definition = CHARACTERS.get(self._character)
        if definition is None and CONTENT_SOURCE != "builtin":
            raise KeyError(
                f"角色 {self._character!r} 不在已加载的角色表里"
                f"（可用：{sorted(CHARACTERS)}）；内容没加载时请先用 "
                f"--content 指定目录")
        deck = self._deck if self._deck is not None else (
            tuple(definition.deck) if definition is not None
            else tuple(STARTING_DECK))
        return {
            "max_hp": definition.hp if definition is not None else PLAYER_MAX_HP,
            "gold": definition.gold if definition is not None else STARTING_GOLD,
            "deck": deck,
            "relics": list(definition.relics) if definition is not None else [],
            "energy": definition.energy if definition is not None else 3,
            "default_character": DEFAULT_CHARACTER,
        }

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._seed = seed
        rng = RngSet(self._seed)
        hidden = RunHidden(rng)
        # ⭐ 幕序列走**专用派生流** ``act_selection``
        # （``StartRunLobby.cs:471``：`new Rng(hash(seed), "act_selection")`），
        # 不是 ``up_front`` —— 挂错流会让"第一幕是密林还是暗港"与开局掷出的
        # 内容产生真机不存在的相关性。
        acts = list(random_act_list(
            rng.named("act_selection"),
            unlocked_epochs=self._unlocked_acts,
            discovered=self._discovered_acts))
        if self._first_act is not None:
            # 真机大厅可以显式指定第一幕（``list[0] = GetAct(Act1) ?? list[0]``）。
            from .acts import act_def as _act_def

            acts[0] = _act_def(self._first_act)
        graph = generate_map(self._seed, acts[0], act_index=0,
                             ascension=self._ascension)
        table = ASCENSION_TABLE[self._ascension]
        start = self.starting_state()
        max_hp = int(start["max_hp"])
        start_damage = int(table["start_damage"])
        player = RunPlayer(
            hp=max_hp - start_damage, max_hp=max_hp, gold=int(start["gold"]),
            deck=[CardInstance(cid, uid=i) for i, cid in enumerate(start["deck"])],
            relics=list(start["relics"]),
        )
        player.energy = int(start["energy"])
        self._state = RunState(seed=self._seed, ascension=self._ascension, act=1,
                               floor=0, player=player, map=graph, hidden=hidden,
                               acts=tuple(acts))
        # ⭐ 遗物抓包：**开局填一次**、用 `up_front` 流洗牌
        # （`RunManager.cs:522-526`：`SharedRelicGrabBag.Populate(…, State.Rng.UpFront)`
        # 之后每个玩家 `PopulateRelicGrabBagIfNecessary(State.Rng.UpFront)`）。
        # 放在预掷**之前**：真机也是先建包、再走开局内容。
        self._populate_relic_bag(rng)
        if self._content_mode == "predetermined":
            self._preroll_all_contents(rng)
        self._node_snapshot = None
        self._nodes_restarted = 0
        return self._step_view()

    def _begin_act(self, act_number: int) -> None:
        """进入下一幕：换图、**重置未知点概率**、清空节点内容。

        ``UnknownMapPointOdds.ResetToBase`` 明确写着"幕与幕之间调用"
        （``UnknownMapPointOdds.cs:179-189``）—— 不重置会让上一幕攒高的
        事件概率漏到下一幕。
        """
        state = self._state
        assert state is not None
        act_index = act_number - 1
        definition = state.acts[act_index]
        state.act = act_number
        # ⚠️ 地图流按**幕索引**派生（``act_{index+1}_map``），所以三幕的地图
        # 互不相同、且各自只看 seed 与幕索引。
        state.map = generate_map(state.seed, definition, act_index=act_index,
                                 ascension=state.ascension)
        state.position = None
        state.floor = 0
        state.visited = []
        state.room = Room(kind=PHASE_MAP)
        state.hidden.point_odds.reset_to_base()
        state.hidden.node_contents = {}
        if self._content_mode == "predetermined":
            self._preroll_all_contents(self._rng)

    def _populate_relic_bag(self, rng: RngSet) -> None:
        """按真机的池子构成填抓包：共享池 ∪ 角色池，只留 4 种稀有度。"""
        from .content import relic_bag_entries
        from .relicbag import RelicGrabBag

        assert self._state is not None
        bag = RelicGrabBag()
        bag.populate(relic_bag_entries(self._character), rng, "up_front")
        self._state.relic_bag = bag

    def replay(self, actions: Sequence[object]) -> RunState:
        """复现一局（对拍与 bug 报告的入口）。"""
        self.reset()
        for action in actions:
            if self._state is not None and self._state.room.kind in (PHASE_WON, PHASE_LOST):
                break
            self.step(action)  # type: ignore[arg-type]
        assert self._state is not None
        return self._state

    # ---- 内容预掷 -----------------------------------------------------
    def _preroll_all_contents(self, rng: RngSet) -> None:
        """一次性把**所有**节点内容掷出来，存进隐藏区。

        这对应"存档保存了随机数状态"的语义：重开同一节点会遇到同样的内容。
        ``content_mode="on_entry"`` 则相反（进节点才掷），用于实测真机属于哪一种
        （``docs/01`` §1.8 问题 3）。

        ⚠️ **未知点（``Unknown``）不预掷**：真机是"进房间那一刻"才
        ``UnknownMapPointOdds.Roll`` 掷房间类型（``RunManager.cs:985``），而且
        概率跨房间累计。提前掷会凭空多出一层信息，也会让概率序列错位。
        远古点同理 —— 它是起点，不是一个"内容节点"。
        """
        assert self._state is not None
        act_id = self._current_act_id()
        for node in self._state.map.nodes:
            if node.kind in (UNKNOWN, ANCIENT):
                continue
            self._state.hidden.node_contents[node.node_id] = self._roll_contents(
                rng, node.kind, act_id)

    def _current_act_id(self) -> str | None:
        assert self._state is not None
        definition = self._state.current_act
        return None if definition is None else definition.act_id

    def _roll_contents(self, rng: RngSet, kind: str,
                       act_id: str | None = None) -> NodeContents:
        if kind in (MONSTER, ELITE, BOSS):
            contents = NodeContents(encounter=draw_encounter(rng, kind))
            # ⚠️ **Boss 也有卡牌奖励**，而且必定稀有（`bossRareOdds = 1f`）。
            # 早先这里写着 `if kind != BOSS`，等于把 Boss 奖励整个抹掉了。
            contents.card_reward = draw_card_reward(
                rng, "boss" if kind == BOSS else kind, character=self._character)
            contents.gold = draw_gold(rng)
            return contents
        if kind == TREASURE:
            return NodeContents(relic=draw_relic(rng), gold=draw_gold(rng))
        if kind == EVENT:
            return NodeContents(event_id=draw_event(rng, act_id))
        if kind == SHOP:
            return NodeContents(shop_stock=draw_shop_stock(rng, self._character))
        return NodeContents()

    # ---- 节点推进 -----------------------------------------------------
    def _enter_node(self, node_id: int) -> None:
        assert self._state is not None
        state = self._state
        node = state.map.by_id(node_id)
        state.position = node_id
        state.floor = node.row
        state.visited.append(node_id)

        kind = node.kind
        if kind == UNKNOWN:
            # ⭐ **进房间这一刻**才掷（真机 ``RunManager.RollRoomTypeFor``）。
            kind = self._roll_unknown_room(node)
        if kind == ANCIENT:
            # 远古点 = 起点的事件房（``MapPointType.Ancient => RoomType.Event``）。
            # 引擎尚未建模远古事件（缺口已登记），这里只保证它**不是**一个普通房间。
            state.room = Room(kind=PHASE_MAP)
            return

        if node_id not in state.hidden.node_contents:
            # 两种情形会走到这里：
            #   * ``on_entry`` 模式：本来就不预掷；
            #   * ``predetermined`` 模式下的**未知点**：房间类型刚刚才掷出来，
            #     所以内容只能在这一刻掷。其余节点在 ``reset`` 时已经掷好了。
            state.hidden.node_contents[node_id] = self._roll_contents(
                state.hidden.rng, kind, self._current_act_id())
        contents = state.hidden.node_contents.get(node_id) or NodeContents()
        state.last_room_kind = kind

        if kind in (MONSTER, ELITE, BOSS):
            table = ASCENSION_TABLE[self._ascension]
            multiplier = table["elite_hp_mult"] if kind == ELITE else table["enemy_hp_mult"]
            state.combat = start_combat(
                # ⭐ 传**永久牌组的完整实例**，不是 `card.cid`：只传 id 会丢掉升级
                # 状态、附魔与永久关联（审计 F01 的实测症状是"Run 升级 10 张，
                # 战斗里 0 张"，而且不报错）。
                state.player.deck, contents.encounter,
                seed=state.seed ^ (node_id * 0x9E3779B1),
                player_hp=state.player.hp,
                # ⭐ 永久上限必须显式传：写死 80 会让"当前生命 105 > 最大生命 80"
                # 这种不可能的状态出现在战斗里（审计 F01）。
                player_max_hp=state.player.max_hp,
                relics=tuple(state.player.relics),
                ascension_hp_mult=multiplier,
                # ⭐ **进阶必须传**：怪物血量/伤害的进阶修正都在战斗里算，
                # 不传等于 A8 的 run 打的是 A0 的怪（审计 F01）。
                ascension=state.ascension,
                # ⚠️ **房间种类必须传**：`sling_of_courage`（赤備）只在精英战给力量、
                # `pantograph`（万用表）只在 Boss 战回血。不传的话两者都会静默失效 ——
                # 遗物在，效果不在，而且不报错。
                room={MONSTER: "monster", ELITE: "elite", BOSS: "boss"}[kind],
                # ⭐ **药水库存共用**（`docs/13` §3.2 入场行）：把 Run 的槽位传进去，
                # 战斗里用掉的药水由 `_sync_player_from_combat` 读回 —— 于是
                # "用掉的药水下一战不会回来"是结构上成立的。
                potions=tuple(state.player.potions),
                max_energy=state.player.energy,
                # ⭐ 角色必须传：战斗内"生成一张牌"的能力按它选卡池
                # （`CallOfTheVoidPower` 等走 `Character.CardPool`）。
                character=self._character,
            )
            state.room = Room(kind=PHASE_COMBAT,
                              encounter=contents.encounter)
        elif kind == TREASURE:
            state.room = Room(kind=PHASE_TREASURE, relic=contents.relic,
                              gold_gained=contents.gold)
        elif kind == EVENT:
            self._enter_event(contents.event_id)
        elif kind == SHOP:
            state.room = Room(kind=PHASE_SHOP, shop_stock=contents.shop_stock)
        elif kind == REST_SITE:
            state.room = Room(kind=PHASE_REST, rest_options=("rest", "smith"))
        else:
            state.room = Room(kind=PHASE_MAP)

    def _roll_unknown_room(self, node) -> str:
        """未知点掷房间类型（``UnknownMapPointOdds.Roll``）。

        黑名单与"已经走过几个未知点"都来自真机的 ``BuildRoomTypeBlacklist`` /
        ``MapPointHistory``，这里是它们的等价物。
        """
        state = self._state
        assert state is not None
        next_kinds = tuple(state.map.by_id(child).kind for child in node.edges)
        blacklist = build_blacklist((state.last_room_kind,), next_kinds)
        kind = state.hidden.point_odds.roll(blacklist, state.unknown_visited)
        state.unknown_visited += 1
        return kind

    def _enter_event(self, event_id: str) -> None:
        """进入事件房间：开一个 ``EventRun``，把当前页的选项摆出来。

        ⚠️ **条件在"进入"这一刻判**（真机的房间选择器就是在选房间时过 ``IsAllowed``）。
        条件不满足时本节点**没有事件**（回到地图并可继续选路），而不是"给一个不该有的
        事件" —— 后者会让玩家白拿收益，比少一个事件严重得多。
        """
        assert self._state is not None
        state = self._state
        from . import events as event_rules

        definition = EVENT_DB.get(event_id)
        if definition is None or not definition.usable:
            state.room = Room(kind=PHASE_MAP)
            return
        allowed = event_rules.gate_ok(definition,
                                      act_index=state.act - 1,
                                      hp=state.player.hp,
                                      max_hp=state.player.max_hp,
                                      gold=state.player.gold,
                                      floor=state.floor,
                                      potions=sum(1 for p in state.player.potions
                                                  if p is not None),
                                      deck_size=len(state.player.deck),
                                      # ⭐ **谓词型条件**（"牌组里有打击牌" /
                                      # "身上有 X 遗物" / "抓包里还有遗物"）需要
                                      # 真正的运行时数据，不能只给张数 ——
                                      # 数据不到位时 `gate_ok` 返回 None（排除事件），
                                      # 而不是"猜 true 放行"。
                                      deck=state.player.deck,
                                      relics=state.player.relics,
                                      potion_ids=tuple(p for p in state.player.potions
                                                       if p is not None),
                                      relic_bag=state.relic_bag)
        if allowed is not True:
            # `None` = 引擎算不出条件（载入时已把这种事件排除出池子），
            # `False` = 条件确实不满足。两种都不该把事件摆出来。
            state.room = Room(kind=PHASE_MAP)
            return
        state.event_run = event_rules.initial_run(
            definition, event_rules.event_rng(state.seed, event_id))
        options = event_rules.current_options(definition, state.event_run)
        state.room = Room(kind=PHASE_EVENT, event_id=event_id,
                          event_options=tuple(option.name for option in options))

    def _sync_reward_phase(self) -> None:
        """有待选奖励时把房间阶段切到 PHASE_REWARD（拾取类遗物会触发它）。"""
        assert self._state is not None
        if self._state.pending_rewards:
            self._state.room = Room(kind=PHASE_REWARD)

    # ---- step ---------------------------------------------------------
    def _fire_relic_run_hook(self, timing: str, relics=None) -> list[str]:
        """在 **Run 层**时机触发遗物效果（``obtained`` / ``combat_end`` / ``combat_victory``）。

        与战斗内时机的区别：效果作用于**牌组 / 金币 / 最大生命**，
        所以走 ``relics.apply_run_hook``，而不是钩子总线上的战斗分发。
        """
        assert self._state is not None
        state = self._state
        events: list[str] = []
        from . import relics as relic_rules
        targets = tuple(state.player.relics) if relics is None else tuple(relics)
        relic_rules.apply_run_hook(state, timing, targets, events)
        # ⚠️ 遗物可能提供一组奖励（calling_bell 给 3 个遗物候选）——
        # 那会把房间阶段切成待选，必须在施加完立刻同步。
        self._sync_reward_phase()
        return events

    def _apply_after_combat_end_powers(self, combat) -> None:
        """``Hook.AfterCombatEnd`` 的**能力**侧：把奖励描述翻译成 Run 层效果。

        能力只回答"谁该给什么"（:func:`sts2_sim.powers.after_combat_end`），
        金币 / 牌组 / 随机流都在 Run 层 —— 与遗物侧 ``combat_end`` 同一个时机。

        ⚠️ 移除奖励走**挂起选牌**（`runeffects` 的 ``select_card``）：真机
        ``CardRemovalReward`` 是"玩家自己选一张"，替玩家随机移除就是换了个机制。
        """
        if combat is None:
            return
        from .content import Effect
        from . import powers as power_rules
        from . import runeffects

        state = self._state
        assert state is not None
        events: list[str] = []
        effects: list[Effect] = []
        for item in power_rules.after_combat_end(combat, events):
            amount = int(item["amount"])
            if item["kind"] == "gold":
                effects.append(Effect(op="gain_gold", amount=amount))
            elif item["kind"] == "remove_card_choices":
                # 每次移除 = 一次"从牌组选一张"；`count` 交给选牌框架逐个挂起
                effects.append(Effect(op="select_card", amount=amount,
                                      purpose="removal", select_from="deck"))
            elif item["kind"] == "upgrade_random_deck_cards":
                self._upgrade_random_deck_cards(amount, events)
        if effects:
            runeffects.apply_run_effects(state, effects, events)
        # 移除奖励会挂起选牌 → 房间阶段可能已经变了，同步一次（与遗物侧同一套）。
        self._sync_last_selection_phase()

    def _upgrade_random_deck_cards(self, count: int, events: list[str]) -> None:
        """``ImprovementPower``：随机升级 ``count`` 张**可升级**的牌组牌。

        源码::

            List<CardModel> list = Deck.GetPile(player).Cards.Where(c => c.IsUpgradable).ToList();
            for (int i = 0; i < Amount; i++) {
                if (list.Count == 0) break;
                CardModel card = player.RunState.Rng.CombatCardSelection.NextItem(list);
                list.Remove(card);                 // ← 挑一张就移出候选，不会重复
                CardCmd.Upgrade(card);
            }

        ``IsUpgradable`` 是 ``CurrentUpgradeLevel < MaxUpgradeLevel`` ——
        引擎的等价物是"还没升级 **且** 这张牌有升级版"。
        随机流用源码指定的 ``combat_card_selection``（不自己挑一条流）。
        """
        state = self._state
        assert state is not None
        from .content import CARD_DB
        candidates = [card for card in state.player.deck
                      if not card.upgraded and CARD_DB[card.cid].upgrade]
        for _ in range(max(0, count)):
            if not candidates:
                events.append("改良：没有可升级的牌了")
                break
            index = state.hidden.rng.next_index("combat_card_selection",
                                                len(candidates))
            card = candidates.pop(index)
            card.upgraded = True
            events.append(f"改良：升级 {CARD_DB[card.cid].name}")

    def _sync_last_selection_phase(self) -> None:
        """挂起选牌之后让房间阶段与"还能做什么"一致。

        ⚠️ 与 :meth:`_sync_reward_phase` 分开：那个管 ``pending_rewards``（待选奖励），
        这个管 ``pending_selection``（等玩家从牌组里选牌）。战斗结束后的移除奖励
        属于后者 —— 少了它，房间会停在"卡牌奖励"而选牌动作给不出来（run 直接堵死）。
        """
        state = self._state
        assert state is not None
        if state.pending_selection is None:
            return
        if state.room.kind not in (PHASE_WON, PHASE_LOST):
            # 选牌界面盖在当前房间之上：房间类型不变，但合法动作只有"选一张"。
            # `legal_meta_actions` 会因为 `pending_selection` 优先返回选牌动作。
            self._sync_reward_phase()

    def _obtain_relic(self, relic_id: str) -> None:
        """拾取遗物：加入列表 → 触发 ``AfterObtained``。

        ⚠️ 顺序不能反：`AfterObtained` 的效果可能引用"自己"
        （`blood_soaked_rose` 拾取时加一张诅咒进牌组），真机也是先入列再触发。
        """
        assert self._state is not None
        state = self._state
        if not relic_id or relic_id in state.player.relics:
            return
        state.player.relics.append(relic_id)
        # 从抓包里移除：真机 `PullNextRelicFromFront` 取走时就已经删了，
        # 但"直接给某个遗物"（`RelicCmd.Obtain<T>`）也要保证它不会再次出现。
        bag = state.relic_bag
        if bag is not None:
            bag.remove(relic_id)
        # ⚠️ **只传这一个**：真机的 `AfterObtained` 是逐个遗物的回调。
        # 传全部会级联 —— 实测领一次召唤铃的奖励后玩家多了 120 个遗物：
        # 每次领取都重新触发铃铛自己的 `AfterObtained`，又生成 3 个候选。
        self._fire_relic_run_hook("obtained", [relic_id])

    def pull_random_relic(self) -> str:
        """``RelicFactory.PullNextRelicFromFront(player)``：摇稀有度 → 从桶前端取。

        真机（``RelicFactory.cs:21-24, 45-50``）::

            PullNextRelicFromFront(player) → PullNextRelicFromFront(player, RollRarity(player))
            RollRarity(player) → RollRarity(player.PlayerRng.Rewards)
            取不到（桶空）→ FallbackRelic（``Circlet``）

        所以稀有度掷点走 **``rewards``** 流（玩家级），不是事件自己的 RNG。
        """
        from .relicbag import FALLBACK_RELIC, roll_rarity

        assert self._state is not None
        bag = self._state.relic_bag
        if bag is None:
            return FALLBACK_RELIC
        rarity = roll_rarity(self._state.hidden.rng, "rewards")
        pulled = bag.pull_from_front(rarity)
        return pulled or FALLBACK_RELIC

    def _resolve_reward(self, action: MetaAction) -> None:
        """结算一组**待选奖励**（`RewardsCmd.OfferCustom`）。

        每次处理队列头部那一组；取完（或跳过）之后如果还有组，
        房间阶段保持 `PHASE_REWARD`，否则回到地图。

        ⚠️ 取遗物要走 ``_obtain_relic``（触发 `AfterObtained`），
        不能只往列表里 append —— 否则"拾取类"遗物的效果全部不生效。
        """
        assert self._state is not None
        state = self._state
        if not state.pending_rewards:
            raise ValueError("非法动作：当前没有待选奖励")
        group = state.pending_rewards[0]
        if action.kind == "take_reward" and action.index >= 0:
            if not 0 <= action.index < len(group.options):
                raise ValueError("非法动作：奖励下标越界")
            chosen = group.options[action.index]
            if group.kind == "relic":
                self._obtain_relic(chosen)
            elif group.kind == "potion":
                self._obtain_potion(chosen)
            elif group.kind == "card":
                state.player.deck.append(CardInstance(chosen, uid=len(state.player.deck)))
        state.pending_rewards.pop(0)
        if not state.pending_rewards:
            state.room = Room(kind=PHASE_MAP)

    def _obtain_potion(self, potion_id: str) -> None:
        """获得药水：放进第一个空槽（满了就丢弃 —— 真机也是这个行为）。"""
        assert self._state is not None
        slots = self._state.player.potions
        for index, slot in enumerate(slots):
            if slot is None:
                slots[index] = potion_id
                return

    def step(self, action):
        assert self._state is not None, "先调用 reset()"
        state = self._state
        if state.room.kind in (PHASE_WON, PHASE_LOST):
            raise RuntimeError("run 已结束")
        if state.room.kind == PHASE_COMBAT:
            return self._step_combat(action)
        return self._step_meta(action)

    def _step_combat(self, action: Action):
        state = self._state
        assert state is not None and state.combat is not None
        if not isinstance(action, Action):
            raise ValueError("战斗阶段需要 combat Action")
        result = combat_step(state.combat, action)
        if not result.done:
            return self._step_view(), 0.0, False, False, self._info()
        if not result.won:
            state.player.hp = 0
            state.room = Room(kind=PHASE_LOST)
            return self._step_view(), -1.0, True, False, self._info()

        # ⭐ **退场结算契约**（``docs/13`` §3.2 的表）：生命、最大生命、
        # 永久改牌、遗物持久计数逐项回写；临时费用 / 战斗生成牌 / 战斗状态
        # 随 `state.combat = None` 一起丢弃。
        #
        # 审计 F01：旧实现只回写 `hp`，上限与永久改牌都没有生命周期定义。
        self._sync_player_from_combat(state.combat)
        # ⭐ AfterCombatVictory / AfterCombatEnd：遗物的战后效果
        # （燃烧之血回血、黑血回更多、…）。以前这里**硬编码了 burning_blood**，
        # 增删一个战后遗物就要改引擎，而且漏掉的不报错、只是不生效。
        self._fire_relic_run_hook("combat_victory")
        self._fire_relic_run_hook("combat_end")
        # ⭐ ``Hook.AfterCombatEnd``（**能力**侧）：`RoyaltiesPower`（金币）/
        # `ForbiddenGrimoirePower`（各选一张移除）/ `ImprovementPower`（随机升级牌组）。
        # ⚠️ 必须在 `state.combat = None` **之前**：能力挂在战斗内的玩家身上。
        self._apply_after_combat_end_powers(state.combat)
        node = state.map.by_id(state.position) if state.position is not None else None
        contents = (state.hidden.node_contents.get(state.position)
                    if state.position is not None else None)
        state.combat = None
        if node is not None and node.kind == BOSS:
            # ⭐ **Boss 倒下不等于通关**：真机一局走 3 个幕索引
            # （密林|暗港 → 蜂巢 → 荣耀），末幕 Boss 才是胜利。
            # 早先这里直接 `PHASE_WON`，等于把后两幕整个抹掉了（审计 F08）。
            second = state.map.second_boss
            if second is not None and state.position == state.map.boss:
                # A10 ``DoubleBoss``：先打完**第二个** Boss 才换幕
                # （``StandardActMap.SecondBossMapPoint``）。
                self._enter_node(second)
                return self._step_view(), 0.0, False, False, self._info()
            if state.act < len(state.acts):
                # ⚠️ **非最终幕的 Boss 照样给奖励**：源码只在
                # ``room.RoomType == Boss && CurrentActIndex >= Acts.Count - 1``
                # 时省略常规奖励（``RewardsSet.cs:88``）。以前这里直接换幕，
                # 等于把 Boss 的三选一与金币整个吞掉。
                state.pending_act_advance = True
                if contents is not None:
                    state.player.gold += contents.gold
                    state.room = Room(kind=PHASE_CARD_REWARD,
                                      card_reward=contents.card_reward,
                                      gold_gained=contents.gold)
                else:
                    self._begin_act(state.act + 1)
                return self._step_view(), 0.0, False, False, self._info()
            state.room = Room(kind=PHASE_WON)
            return self._step_view(), 1.0, True, False, self._info()

        if contents is not None:
            state.player.gold += contents.gold
            state.room = Room(kind=PHASE_CARD_REWARD, card_reward=contents.card_reward,
                              gold_gained=contents.gold)
        else:
            state.room = Room(kind=PHASE_MAP)
        return self._step_view(), 0.0, False, False, self._info()

    def _sync_player_from_combat(self, combat: CombatState) -> None:
        """战斗 → Run 的**唯一**回写入口。

        集中成一处的原因与 ``mark_dead`` 相同：分散回写时"漏掉一项"是静默的。
        每一项都在这里显式列出，新增持久字段时改这一个函数。
        """
        state = self._state
        assert state is not None
        state.player.hp = max(0, combat.player.hp)
        # 永久上限变化（`gain_max_hp` / `lose_max_hp`）必须跨战斗保持。
        state.player.max_hp = max(1, combat.player.max_hp)
        # 永久改牌（升级 / 附魔）只回写清单里的那些；临时改牌不在这里。
        apply_permanent_changes(combat, state.player.deck)
        # 遗物持久计数：战斗内产生的计数写回 Run 侧（临时计数已在战斗里重置）。
        for key, value in getattr(combat, "relic_counters", {}).items():
            state.player.relic_counters[key] = value
        # ⭐ 药水：战斗与 Run **共用库存**，回写是幂等的（用掉的槽位仍是 None）。
        # 少了这一步，"用掉药水不会在下一战回来"就只能靠"反正没人改"来成立。
        if len(combat.potions) == len(state.player.potions):
            state.player.potions[:] = list(combat.potions)

    def _step_meta(self, action: MetaAction):
        state = self._state
        assert state is not None
        if not isinstance(action, MetaAction):
            raise ValueError("非战斗阶段需要 MetaAction")
        allowed = legal_meta_actions(state)
        if not any(a.kind == action.kind and a.index == action.index for a in allowed):
            raise ValueError(f"非法动作 {action!r}（当前房间 {state.room.kind}）")

        room = state.room
        if action.kind == "choose_node":
            if action.index not in reachable_nodes(state):
                raise ValueError("非法动作：目标节点不可达")
            self._enter_node(action.index)
            # ⭐ 快照取在**进入节点之后**：重开回到"这个节点的开头"
            #    （战斗第 1 回合、同一副牌序、同一份奖励），而不是回到地图。
            self._node_snapshot = copy.deepcopy(state)
        elif action.kind == "pick_card":
            if not 0 <= action.index < len(room.card_reward):
                raise ValueError("非法动作：候选下标越界")
            state.player.deck.append(
                CardInstance(room.card_reward[action.index], uid=len(state.player.deck)))
            self._after_card_reward()
        elif action.kind == "skip_card":
            self._after_card_reward()
        elif action.kind in ("take_reward", "skip_reward"):
            self._resolve_reward(action)
        elif action.kind == "rest":
            heal = int(state.player.max_hp * REST_HEAL_RATIO)
            state.player.hp = min(state.player.max_hp, state.player.hp + heal)
            state.room = Room(kind=PHASE_MAP)
        elif action.kind == "smith":
            from .observe import deck_slot_order
            order = deck_slot_order(state.player.deck)
            if not 0 <= action.index < len(order):
                raise ValueError("非法动作：牌组槽位越界")
            internal = order[action.index]
            if state.player.deck[internal].upgraded:
                raise ValueError("非法动作：该牌已升级")
            state.player.deck[internal].upgraded = True
            state.room = Room(kind=PHASE_MAP)
        elif action.kind == "buy":
            item, price = room.shop_stock[action.index]
            if price > state.player.gold:
                raise ValueError("非法动作：金币不足")
            state.player.gold -= price
            state.player.deck.append(CardInstance(item, uid=len(state.player.deck)))
            stock = list(room.shop_stock)
            stock.pop(action.index)
            state.room = Room(kind=PHASE_SHOP, shop_stock=tuple(stock))
        elif action.kind == "proceed":
            if room.kind == PHASE_TREASURE:
                self._obtain_relic(room.relic)
                state.player.gold += room.gold_gained
            state.room = Room(kind=PHASE_MAP)
        elif action.kind == "event_option":
            self._resolve_event(action.index)
        elif action.kind == "select_deck_card":
            self._resolve_deck_selection(action.index)
        else:
            raise ValueError(f"未知动作 {action.kind!r}")
        return self._step_view(), 0.0, False, False, self._info()

    def _after_card_reward(self) -> None:
        """卡牌奖励房间的出口。

        ⚠️ Boss 的三选一之后**不是**回到本幕地图，而是进下一幕 —— 所以这里要看
        :attr:`RunState.pending_act_advance`。少了这一步，打完 Boss 会回到
        **已经打过的**地图上（那张图的 Boss 已经死过一次），run 就卡住了。
        """
        state = self._state
        assert state is not None
        if state.pending_act_advance:
            state.pending_act_advance = False
            self._begin_act(state.act + 1)
        else:
            state.room = Room(kind=PHASE_MAP)

    def _resolve_deck_selection(self, slot: int) -> None:
        """玩家从牌组里选了第 ``slot`` 个**槽位**的牌（选牌用途由挂起项决定）。

        ⚠️ 动作里给的是**槽位**（``observe.deck_slot_order`` 的编号，人类可见），
        要在这里映射回内部下标；映射定义两处共用，避免"观测一套、动作一套"。
        """
        from . import events as event_rules
        from .observe import deck_slot_order
        from .runeffects import resolve_run_selection

        state = self._state
        assert state is not None
        order = deck_slot_order(state.player.deck)
        if not 0 <= slot < len(order):
            raise ValueError("非法动作：牌组槽位越界")
        internal = order[slot]
        log: list[str] = []
        finished = resolve_run_selection(state, internal, log, obtain_relic=self._obtain_relic)
        state.event_log.extend(log)
        if not finished:
            return                          # 还要再选
        definition = EVENT_DB.get(state.room.event_id)
        run = state.event_run
        if definition is not None and run is not None and run.pending_option:
            option = event_rules.option_by_key(definition, run.pending_option)
            if option is not None:
                event_rules.finish_pending(run, option)
        if run is not None and run.finished:
            state.event_run = None
            state.room = Room(kind=PHASE_MAP)
            self._sync_reward_phase()
        elif definition is not None and run is not None:
            options = event_rules.current_options(definition, run)
            state.room = Room(kind=PHASE_EVENT, event_id=definition.eid,
                              event_options=tuple(o.name for o in options))

    def _resolve_event(self, index: int) -> None:
        """结算一个事件选项，并按结果翻页 / 结束。"""
        from . import events as event_rules

        state = self._state
        assert state is not None
        definition = EVENT_DB.get(state.room.event_id)
        run = state.event_run
        if definition is None or run is None:
            raise ValueError("非法动作：当前不在事件里")
        log: list[str] = []
        event_rules.apply_option(definition, run, index, state, log,
                                 obtain_relic=self._obtain_relic)
        state.event_log.extend(log)
        if run.finished:
            state.event_run = None
            state.room = Room(kind=PHASE_MAP)
            # 选项可能给出一组待选奖励（`RewardsCmd.OfferCustom`）——
            # 那会把阶段切成 PHASE_REWARD，必须同步，否则奖励永远拿不到。
            self._sync_reward_phase()
        else:
            options = event_rules.current_options(definition, run)
            state.room = Room(kind=PHASE_EVENT, event_id=definition.eid,
                              event_options=tuple(o.name for o in options))

    # ---- SL：节点级重开 -----------------------------------------------
    def restart_node(self) -> bool:
        """回到**节点入口**，隐藏随机状态不变（同一场遭遇、同一份奖励）。"""
        if self._node_snapshot is None or self.attempts_left <= 0:
            return False
        self._nodes_restarted += 1
        self._state = copy.deepcopy(self._node_snapshot)
        return True

    # ---- 视图 ---------------------------------------------------------
    def _step_view(self):
        from .observe import observe, observe_run
        assert self._state is not None
        state = self._state
        if state.room.kind == PHASE_COMBAT and state.combat is not None:
            return _Step(phase=PHASE_COMBAT,
                         combat_obs=observe(state.combat),
                         run_obs=observe_run(state),
                         legal=tuple(combat_legal_actions(state.combat)))
        return _Step(phase=state.room.kind, combat_obs=None,
                     run_obs=observe_run(state),
                     legal=tuple(legal_meta_actions(state)))

    def _info(self) -> dict:
        assert self._state is not None
        return {
            "phase": self._state.room.kind,
            "floor": self._state.floor,
            "hp": self._state.player.hp,
            "gold": self._state.player.gold,
            "deck_size": len(self._state.player.deck),
            "relics": tuple(self._state.player.relics),
        }

    def render_text(self) -> str:
        assert self._state is not None
        state = self._state
        lines = [
            f"=== 第 {state.floor} 层 | 房间 {state.room.kind} | "
            f"HP {state.player.hp}/{state.player.max_hp} | 金币 {state.player.gold} ===",
            f"牌组 {len(state.player.deck)} 张 | 遗物 {state.player.relics or '无'}",
        ]
        if state.room.kind == PHASE_COMBAT and state.combat is not None:
            lines.append(render_text(state.combat))
        elif state.room.kind == PHASE_MAP:
            reachable = reachable_nodes(state)
            kinds = ", ".join(f"#{n}({state.map.by_id(n).kind})" for n in reachable)
            lines.append(f"可达节点: {kinds or '（无）'}")
        elif state.room.kind == PHASE_CARD_REWARD:
            names = ", ".join(CARD_DB[c].name for c in state.room.card_reward)
            lines.append(f"卡牌奖励: {names}")
        elif state.room.kind == PHASE_SHOP:
            lines.append("商店: " + ", ".join(
                f"{CARD_DB[c].name}({p}金)" for c, p in state.room.shop_stock))
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _Step:
    """环境每一步的返回：**阶段 + 对应观测 + 合法动作**。

    两个策略头（战斗 / Run）用同一个编码器、不同 task token（``docs/03`` §3.1），
    所以这里同时给出两份观测。
    """

    phase: str
    combat_obs: object | None
    run_obs: object | None
    legal: tuple
