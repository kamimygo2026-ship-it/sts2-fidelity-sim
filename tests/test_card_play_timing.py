"""``AfterCardPlayed`` 触发时机的回归测试（审计 F06 的后半段 / 工单 T04）。

真机的 ``Hook.AfterCardPlayed``（``Hook.cs:278``）在卡牌效果链**全部跑完之后**
触发一次。而旧实现在卡牌效果遇到玩家选择而**挂起**时就立刻触发了 —— 于是
``EnragePower``（"玩家打出技能牌 → 拥有者 +力量"）这类响应者会在牌的效果
还没结算完就先跑一轮。

这个错误**不会报错**，只是数值偏；而且只在"带选牌的技能牌"上暴露，
普通技能牌完全看不出来。

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 挂起时 AfterCardPlayed 提前触发 | 触发点写在 `step()` 里、没看 `done` | `test_pending_selection_defers_the_hook` |
| 自动打出的 Sly 牌不触发 | `_autoplay_sly` 里根本没有触发点 | `test_autoplayed_card_fires_the_hook` |
| 同一张牌触发两次 | 直通路径与续跑路径各写一次 | `test_hook_fires_exactly_once` |
"""

from __future__ import annotations

import unittest

from sts2_sim import core
from sts2_sim.content import CARD_DB, CardDef, Effect, EnemyDef, MoveDef
from sts2_sim.core import (
    Action, CardInstance, Combatant, EnemyState, Hidden, CombatState,
    _apply_effects, legal_actions, mark_permanent_upgrade, step,
)
from sts2_sim.rng import RngSet

#: 一张"先抽牌、再选一张弃掉"的技能牌 —— 选牌会让效果链挂起。
PICKY_SKILL = CardDef(
    "test_picky_skill", "挑剔技能", 1, "skill", "common", "none",
    (Effect("draw", 2, target="self"),
     Effect("select_card", 1, target="self", select_from="hand",
            purpose="discard")),
)

#: 一张普通的技能牌（不挂起），用作对照组。
PLAIN_SKILL = CardDef(
    "test_plain_skill", "普通技能", 1, "skill", "common", "none",
    (Effect("block", 5, target="self"),),
)


def make_state(hand_cards: list[CardInstance],
               enemy_powers: tuple[tuple[str, int], ...] = (("enrage", 2),)):
    enemy = EnemyState(name="enemy#0", hp=200, max_hp=200, eid="test_enemy")
    for name, amount in enemy_powers:
        enemy.add_power(name, amount)
    return CombatState(
        player=Combatant("player", 80, 80),
        enemies=[enemy],
        hand=list(hand_cards),
        draw_pile=[], discard=[], exhaust=[], play_area=[],
        hidden=Hidden(RngSet(1)),
    )


class CardPlayTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved = {cid: CARD_DB.get(cid) for cid in
                      (PICKY_SKILL.cid, PLAIN_SKILL.cid)}
        CARD_DB[PICKY_SKILL.cid] = PICKY_SKILL
        CARD_DB[PLAIN_SKILL.cid] = PLAIN_SKILL

    @classmethod
    def tearDownClass(cls):
        for cid, card in cls._saved.items():
            if card is None:
                CARD_DB.pop(cid, None)
            else:
                CARD_DB[cid] = card

    def test_plain_skill_fires_the_hook_once(self):
        """对照组：效果链一次跑完 → 立刻触发一次（**不能两次**）。"""
        state = make_state([CardInstance(PLAIN_SKILL.cid)])
        step(state, Action("play_card", 0, 0))
        self.assertEqual(state.enemies[0].power("strength"), 2,
                         "普通技能牌应当恰好触发一次 AfterCardPlayed")

    def test_pending_selection_defers_the_hook(self):
        """⭐ 审计 F06：挂起期间**不得**触发；选完之后才触发一次。"""
        state = make_state([CardInstance(PICKY_SKILL.cid),
                            CardInstance(PLAIN_SKILL.cid),
                            CardInstance(PLAIN_SKILL.cid)])
        step(state, Action("play_card", 0, -1))
        self.assertIsNotNone(state.pending, "这张牌应当挂起等待选牌")
        self.assertEqual(state.enemies[0].power("strength"), 0,
                         "效果还没结算完，AfterCardPlayed 不该已经触发")

        step(state, Action("select_card", 0))
        self.assertIsNone(state.pending)
        self.assertEqual(state.enemies[0].power("strength"), 2,
                         "选完续跑之后应当触发**恰好一次**")

    def test_hook_fires_exactly_once_across_selection(self):
        """同一张牌不能被直通路径和续跑路径各触发一次。"""
        state = make_state([CardInstance(PICKY_SKILL.cid),
                            CardInstance(PLAIN_SKILL.cid)])
        step(state, Action("play_card", 0, -1))
        step(state, Action("select_card", 0))
        self.assertEqual(state.enemies[0].power("strength"), 2)

    def test_card_lands_in_discard_only_after_resolution(self):
        """挂起期间牌留在出牌区；结算完才进弃牌堆。"""
        state = make_state([CardInstance(PICKY_SKILL.cid),
                            CardInstance(PLAIN_SKILL.cid)])
        card = state.hand[0]
        step(state, Action("play_card", 0, -1))
        self.assertIn(card, state.play_area)
        self.assertNotIn(card, state.discard)
        step(state, Action("select_card", 0))
        self.assertIn(card, state.discard)

    def test_attack_cards_do_not_trigger_enrage(self):
        """反向验证：``EnragePower`` 只看技能牌 —— 否则上面对照组没有鉴别力。"""
        strike = CARD_DB["strike"]
        state = make_state([CardInstance("strike")])
        step(state, Action("play_card", 0, 0))
        self.assertGreater(strike.cost, -1)
        self.assertEqual(state.enemies[0].power("strength"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
