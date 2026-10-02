"""基线规则 bot（``docs/04`` §4.1）。

**只接受 ``Observation`` 与合法动作列表**——没有 state、没有 env 引用。
这不是靠自觉：``tests/test_anticheat.py`` 用 AST 扫描强制本模块无法触及隐藏信息。

它同时是：模拟器的压力测试、RL 的回归基线、行为克隆的兜底数据源。
"""

from __future__ import annotations

from typing import Sequence

from .content import CARD_DB
from .core import Action
from .observe import Observation

#: 本模块禁止出现的标识符（CI 静态检查，见 tests/test_anticheat.py）
FORBIDDEN_IDENTIFIERS: frozenset[str] = frozenset({
    "Hidden", "RngSet", "CombatState", "Snapshot", "raw_state",
    "snapshot", "restore", "reroll_hidden", "master_seed", "draw_pile",
})


def _power(powers: tuple[tuple[str, int], ...], name: str) -> int:
    for key, value in powers:
        if key == name:
            return value
    return 0


def _card_damage(cid: str, obs: Observation, target_index: int) -> int:
    """估算一张牌对某敌人的伤害（L1 固定知识：卡牌数值允许使用）。"""
    definition = CARD_DB[cid]
    strength = _power(obs.player_powers, "strength")
    weak = _power(obs.player_powers, "weak")
    enemy = obs.enemies[target_index] if 0 <= target_index < len(obs.enemies) else None
    vulnerable = _power(enemy.powers, "vulnerable") if enemy else 0

    total = 0
    for effect in definition.effects:
        if effect.op == "damage" and enemy is not None:
            total += _damage_formula(effect.amount + strength, weak, vulnerable)
        elif effect.op == "damage_all":
            total += _damage_formula(effect.amount + strength, weak, vulnerable)
    return total


def _damage_formula(base: int, weak: int, vulnerable: int) -> int:
    dmg = base
    if weak > 0:
        dmg = int(dmg * 0.75)
    if vulnerable > 0:
        dmg = int(dmg * 1.5)
    return max(dmg, 0)


def _card_block(cid: str) -> int:
    return sum(e.amount for e in CARD_DB[cid].effects if e.op == "block")


def _is_attack(cid: str) -> bool:
    return any(e.op in ("damage", "damage_all") for e in CARD_DB[cid].effects)


class RuleBot:
    """启发式：能斩杀就斩杀；预计挨打就格挡；否则打伤害最高的攻击。"""

    def __init__(self, block_margin: int = 0) -> None:
        self.block_margin = block_margin

    # ---- 主入口 ------------------------------------------------------
    def act(self, obs: Observation, legal: Sequence[Action]) -> Action:
        # ⚠️ 正在等选牌时，`legal` 里**只有** select_card —— 此时返回 `end_turn`
        # 是**非法动作**（引擎会抛异常）。实测就漏过这一条：
        # `survivor` 这类"格挡 + 弃牌"的卡被 bot 打出去之后，下一次调用就会炸。
        selections = [a for a in legal if a.kind == "select_card"]
        if selections:
            return self._pick_selection(obs, selections)

        plays = [a for a in legal if a.kind == "play_card"]
        if not plays:
            return Action("end_turn")

        lethal = self._find_lethal(obs, plays)
        if lethal is not None:
            return lethal

        incoming = self._incoming_damage(obs)
        need_block = incoming > obs.player_block + self.block_margin

        if need_block:
            blocker = self._best_blocker(obs, plays)
            if blocker is not None:
                return blocker

        attacker = self._best_attack(obs, plays)
        if attacker is not None:
            return attacker

        blocker = self._best_blocker(obs, plays)
        if blocker is not None:
            return blocker
        return Action("end_turn")

    def _pick_selection(self, obs: Observation,
                        selections: Sequence[Action]) -> Action:
        """选牌时挑"最不值钱"的那张。

        评分只看**人类能看到的东西**（费用、类型、是否升级），不窥探隐藏信息：
        优先弃掉「费用高但既不输出伤害也不给格挡」的牌（状态牌/诅咒/多余的能力牌）。
        """
        def score(action: Action) -> tuple:
            # ⭐ 选牌的下标是**候选列表**的序号，不是手牌序号（审计 F09）：
            # 候选可能来自弃牌堆 / 抽牌堆，用 `obs.hand[...]` 会张冠李戴。
            view = self._candidate(obs, action)
            if view is None:
                return (0, 0, 0)
            cid = view.cid
            is_attack = _is_attack(cid)
            is_block = _card_block(cid) > 0
            # 越"没用"越该被弃：非攻非防 → 优先；其次费用高 → 优先
            useless = 0 if (is_attack or is_block) else 1
            return (useless, view.cost, 1 if view.upgraded else 0)

        return max(selections, key=score)

    @staticmethod
    def _candidate(obs: Observation, action: Action):
        """``select_card`` 动作 → 候选卡视图（来自观测里的候选列表）。"""
        index = action.hand_index
        if obs.selection is not None:
            if 0 <= index < len(obs.selection.candidates):
                return obs.selection.candidates[index]
            return None
        if 0 <= index < len(obs.hand):
            return obs.hand[index]
        return None

    # ---- 子策略 ------------------------------------------------------
    def _find_lethal(self, obs: Observation, plays: Sequence[Action]) -> Action | None:
        for action in plays:
            if action.target < 0:
                continue
            enemy = obs.enemies[action.target]
            cid = obs.hand[action.hand_index].cid
            if _card_damage(cid, obs, action.target) >= enemy.hp:
                return action
        return None

    def _incoming_damage(self, obs: Observation) -> int:
        total = 0
        for enemy in obs.enemies:
            if enemy.hp <= 0:
                continue
            if enemy.intent_kind in ("attack", "attack_debuff"):
                per_hit = _damage_formula(enemy.intent_value
                                          + _power(enemy.powers, "strength"),
                                          _power(enemy.powers, "weak"), 0)
                total += per_hit * max(1, enemy.intent_times)
        return total

    def _best_attack(self, obs: Observation, plays: Sequence[Action]) -> Action | None:
        best: tuple[float, Action] | None = None
        for action in plays:
            cid = obs.hand[action.hand_index].cid
            if not _is_attack(cid):
                continue
            if action.target < 0:
                # AoE：按活着的敌人总数计
                score = sum(_card_damage(cid, obs, i)
                            for i, e in enumerate(obs.enemies) if e.hp > 0)
            else:
                score = _card_damage(cid, obs, action.target)
                # 优先补刀血量最低的
                score += (obs.enemies[action.target].max_hp
                          - obs.enemies[action.target].hp) * 0.01
            if best is None or score > best[0]:
                best = (score, action)
        return best[1] if best else None

    def _best_blocker(self, obs: Observation, plays: Sequence[Action]) -> Action | None:
        best: tuple[int, Action] | None = None
        for action in plays:
            gain = _card_block(obs.hand[action.hand_index].cid)
            if gain <= 0:
                continue
            if best is None or gain > best[0]:
                best = (gain, action)
        return best[1] if best else None
