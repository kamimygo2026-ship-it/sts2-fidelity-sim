"""Run 层基线 bot（``docs/04`` §4.1）。

战斗节点委托给 ``bot.RuleBot``；其余房间用启发式：
**路线**按"血量/金币/楼层"决定，**选牌**按静态分 + 牌组需求修正。

与 ``bot.RuleBot`` 同样只接受 ``Observation`` / ``RunObservation``——
策略路径不得触及隐藏信息。
"""

from __future__ import annotations

from typing import Iterable

from .bot import RuleBot
from .content import CARD_DB
from .mapgen import BOSS, ELITE, MONSTER, REST_SITE, SHOP, TREASURE, UNKNOWN
from .observe import RunObservation
from .run import MetaAction

#: 静态卡牌评分（启发式，非游戏内容）。
#:
#: ⚠️ **必须从卡牌定义现算，不能写死 id 表。** 旧实现是一张写死的
#: ``{"strike": 0.0, "bash": 3.0, …}``，只在**内置占位内容**上有效；
#: 换成真实内容后基础牌叫 ``strike_ironclad``，于是每张牌的分数都是默认值 ——
#: 规则 bot 退化成"永远选第一张"，而它本该是判断策略有没有进步的那个基线。
#: （``docs/04`` §4.1：没有可解释基线的 RL 是盲人摸象。）
#:
#: 评分只看**人类可见的静态数值**（费用、伤害、格挡、抽牌、加能量、施加的能力）。
#: 这是启发式，不是学出来的；真实版本应从人类 run 数据里学（``docs/04`` §4.2）。
def card_score(cid: str) -> float:
    card = CARD_DB.get(cid)
    if card is None:
        return 0.0
    # 诅咒 / 状态：占手牌、打不出，强烈回避
    if card.rarity in ("curse", "status", "quest"):
        return -100.0

    damage = sum(e.amount for e in card.effects if e.op == "damage")
    damage_all = sum(e.amount for e in card.effects if e.op == "damage_all")
    block = sum(e.amount for e in card.effects if e.op == "block")
    draw = sum(e.amount for e in card.effects if e.op == "draw")
    energy = sum(e.amount for e in card.effects if e.op == "gain_energy")
    heal = sum(e.amount for e in card.effects if e.op == "heal")
    powers = sum(e.amount for e in card.effects if e.op == "apply_power")
    cost = max(0, card.cost)

    # AoE 按"至少两个目标"折算；能力与抽牌按经验权重估值。
    value = (damage + 1.6 * damage_all + 0.9 * block
             + 3.0 * draw + 4.0 * energy + 1.5 * heal + 2.0 * powers)
    if cost:
        value /= cost
    if card.exhaust:
        value -= 1.0                      # 消耗牌只能用一次
    if card.upgrade:
        value += 0.5                      # 有升级空间
    return value


#: 向后兼容的别名：外部若要按 id 覆盖评分，用 ``CARD_SCORE_OVERRIDES``。
CARD_SCORE_OVERRIDES: dict[str, float] = {}


def score_of(cid: str) -> float:
    """卡牌评分：显式覆盖优先，否则现算。"""
    if cid in CARD_SCORE_OVERRIDES:
        return CARD_SCORE_OVERRIDES[cid]
    return card_score(cid)


#: 路线偏好分（越低越不吸引）。
#:
#: ⚠️ 未知点 ``unknown``（真机的问号）按"事件"的期望价打分：进房间之前**没人**
#: 知道它是事件还是怪（``UnknownMapPointOdds``），bot 也只能按期望来。
#: 早先这里把未知点直接当事件房（``EVENT``），等于让 bot 用了它不该知道的信息。
NODE_SCORES: dict[str, float] = {
    TREASURE: 10.0, REST_SITE: 6.0, MONSTER: 5.0, SHOP: 4.0,
    UNKNOWN: 4.0, ELITE: 3.0, BOSS: 100.0,
}


def _has_block(cid: str) -> bool:
    return any(e.op == "block" for e in CARD_DB[cid].effects)


def _is_attack(cid: str) -> bool:
    return any(e.op in ("damage", "damage_all") for e in CARD_DB[cid].effects)


class RunBot:
    def __init__(self, rest_threshold: float = 0.65,
                 elite_hp_threshold: float = 0.70,
                 buy_threshold: float = 6.0,
                 max_deck: int = 30) -> None:
        self.combat_bot = RuleBot()
        self.rest_threshold = rest_threshold
        self.elite_hp_threshold = elite_hp_threshold
        self.buy_threshold = buy_threshold
        self.max_deck = max_deck

    # ---- 主入口 ------------------------------------------------------
    def act(self, step, run_obs: RunObservation | None = None):
        """``step`` 是 ``run._Step``；战斗阶段委托给 RuleBot。"""
        if step.phase == "combat":
            return self.combat_bot.act(step.combat_obs, step.legal)
        obs = run_obs if run_obs is not None else step.run_obs
        handler = getattr(self, f"_act_{obs.phase}", None)
        if handler is None:
            return step.legal[0] if step.legal else None
        return handler(obs, step.legal)

    # ---- 各房间 ------------------------------------------------------
    def _act_map(self, obs: RunObservation, legal: Iterable[MetaAction]):
        hp_ratio = obs.hp / max(1, obs.max_hp)
        best = None
        for action in legal:
            if action.kind != "choose_node":
                continue
            node = next(n for n in obs.map_nodes if n.node_id == action.index)
            score = NODE_SCORES.get(node.kind, 1.0)
            if node.kind == ELITE and hp_ratio < self.elite_hp_threshold:
                score -= 8.0
            if node.kind == REST_SITE and hp_ratio > self.rest_threshold:
                score -= 3.0
            if node.kind == SHOP and obs.gold < 60:
                score -= 3.0
            score += node.row * 0.01                      # 平局时偏好推进
            if best is None or score > best[0]:
                best = (score, action)
        return best[1] if best else (list(legal)[0] if legal else None)

    def _act_card_reward(self, obs: RunObservation, legal: Iterable[MetaAction]):
        attack_count = sum(1 for (cid, _up, n) in obs.deck.items if _is_attack(cid) for _ in range(n))
        block_count = sum(1 for (cid, _up, n) in obs.deck.items if _has_block(cid) for _ in range(n))
        best = None
        for action in legal:
            if action.kind != "pick_card":
                continue
            cid = obs.card_reward[action.index]
            score = score_of(cid)
            if _has_block(cid) and block_count < 3:
                score += 3.0
            if _is_attack(cid) and attack_count > 8:
                score -= 2.0
            if obs.deck_size >= self.max_deck:
                score -= 5.0
            if best is None or score > best[0]:
                best = (score, action)
        if best is None or best[0] <= 0.0:
            return MetaAction("skip_card")
        return best[1]

    def _act_rest(self, obs: RunObservation, legal: Iterable[MetaAction]):
        hp_ratio = obs.hp / max(1, obs.max_hp)
        actions = list(legal)
        if hp_ratio < self.rest_threshold:
            return next(a for a in actions if a.kind == "rest")
        smiths = [a for a in actions if a.kind == "smith"]
        if smiths:
            best = max(smiths, key=lambda a: score_of(
                obs.deck_slots[a.index].cid
                if 0 <= a.index < len(obs.deck_slots) else ""))
            return best
        return next(a for a in actions if a.kind == "rest")

    def _act_shop(self, obs: RunObservation, legal: Iterable[MetaAction]):
        for action in legal:
            if action.kind != "buy":
                continue
            cid, price = obs.shop_stock[action.index]
            if score_of(cid) >= self.buy_threshold and price <= obs.gold - 40:
                return action
        return next((a for a in legal if a.kind == "proceed"), list(legal)[-1])

    def _act_event(self, obs: RunObservation, legal: Iterable[MetaAction]):
        actions = list(legal)
        return actions[0] if actions else None

    def _act_treasure(self, obs: RunObservation, legal: Iterable[MetaAction]):
        return next((a for a in legal if a.kind == "proceed"), list(legal)[-1])
