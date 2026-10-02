"""手牌上限的回归测试（真正的 ``CardPile.MaxCardsInHand = 10``）。

这个 bug 是"真实内容 + SL 预算跑 1500 步"时炸出来的：
``ValueError: 手牌 数量 11 超过编码容量 10`` —— 观测编码器报的错，
**根因却在引擎**：``draw_cards`` 从来没有检查手牌上限。

真机的行为（``CardPileCmd.DrawInternal``）：

* ``num = max(0, MaxCardsInHand - hand.Count)``，为 0 就完全不抽（并提示"手牌已满"）；
* 循环里 ``if (hand.Cards.Count >= MaxCardsInHand) break;`` —— 抽不下的牌**留在抽牌堆**；
* ``CardPileCmd.Add`` 往手牌加牌时若已满，**改放弃牌堆**（``isFullHandAdd``）。

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 训练中途报"手牌 11 超过容量 10" | 抽牌不看上限 | `test_draw_never_exceeds_the_hand_limit` |
| 效果送的牌把手牌撑爆 | `Add` 的满手改弃规则没实现 | `test_effects_routed_to_hand_respect_the_limit` |
| 手牌满时抽牌把牌"抽没了" | 抽不下应当**留堆** | `test_cards_stay_in_the_draw_pile_when_the_hand_is_full` |
"""

from __future__ import annotations

import unittest

from sts2_sim.core import (
    MAX_HAND_SIZE, CardInstance, PendingSelection, add_to_hand, draw_cards,
    start_combat,
)

DECK = ["strike"] * 5 + ["defend"] * 4 + ["bash"]


def make_state(hand_size: int):
    state = start_combat(DECK, ("jaw_worm",), seed=1)
    state.hand = []
    state.draw_pile = []
    state.discard = []
    state.hand = [CardInstance("strike") for _ in range(hand_size)]
    state.draw_pile = [CardInstance("defend") for _ in range(20)]
    return state


class TestHandLimit(unittest.TestCase):
    def test_the_limit_matches_the_real_game(self):
        """``CardPile.MaxCardsInHand == 10``（``CardPile.cs``）。"""
        self.assertEqual(MAX_HAND_SIZE, 10)

    def test_draw_never_exceeds_the_hand_limit(self):
        """⭐ 从 9 张抽 5 张 → 只能到 10（旧实现会到 14）。"""
        state = make_state(9)
        events: list[str] = []
        draw_cards(state, 5, events)
        self.assertEqual(len(state.hand), MAX_HAND_SIZE)
        self.assertTrue(any("手牌已满" in line for line in events),
                        f"应当明确记录「手牌已满」，实际 {events}")

    def test_reverse_verification_drawing_without_the_cap_would_overflow(self):
        """反向验证：证明这道闸门真的在起作用（否则上面的测试没有鉴别力）。"""
        state = make_state(9)
        draw_cards(state, 5, [])
        naive = 9 + 5
        self.assertNotEqual(len(state.hand), naive,
                            "抽牌没有上限 → 闸门没生效")
        self.assertLess(len(state.hand), naive)

    def test_cards_stay_in_the_draw_pile_when_the_hand_is_full(self):
        """抽不下的牌必须**留堆**，不能凭空消失（守恒）。"""
        state = make_state(8)
        before = len(state.draw_pile)
        draw_cards(state, 5, [])
        drawn = len(state.hand) - 8
        self.assertEqual(len(state.draw_pile), before - drawn)

    def test_full_hand_draws_nothing(self):
        state = make_state(MAX_HAND_SIZE)
        events: list[str] = []
        before = list(state.draw_pile)
        draw_cards(state, 3, events)
        self.assertEqual([c.uid for c in state.draw_pile], [c.uid for c in before])
        self.assertEqual(len(state.hand), MAX_HAND_SIZE)

    def test_effects_routed_to_hand_respect_the_limit(self):
        """``CardPileCmd.Add`` 的"满手改弃牌堆"规则（生成牌与选牌都走它）。"""
        state = make_state(MAX_HAND_SIZE)
        card = CardInstance("defend")
        landed = add_to_hand(state, card)
        self.assertEqual(landed, "discard")
        self.assertIn(card, state.discard)
        self.assertNotIn(card, state.hand)

        roomy = make_state(1)
        card2 = CardInstance("defend")
        self.assertEqual(add_to_hand(roomy, card2), "hand")
        self.assertIn(card2, roomy.hand)

    def test_selection_to_hand_respects_the_limit(self):
        """选牌"入手牌"也要走同一条规则（否则选一张就把手牌撑到 11）。"""
        from sts2_sim.core import Action, step
        state = make_state(MAX_HAND_SIZE)
        target = CardInstance("defend")
        state.discard = [target]
        state.pending = PendingSelection(purpose="to_hand", source_pile="discard",
                                         remaining=1)
        step(state, Action("select_card", 0))
        self.assertEqual(len(state.hand), MAX_HAND_SIZE)
        self.assertIn(target, state.discard)

    def test_generated_card_to_hand_respects_the_limit(self):
        from sts2_sim.core import _add_generated_card
        state = make_state(MAX_HAND_SIZE)
        card = CardInstance("defend")
        _add_generated_card(state, card, "hand")
        self.assertEqual(len(state.hand), MAX_HAND_SIZE)
        self.assertIn(card, state.discard)


if __name__ == "__main__":
    unittest.main(verbosity=2)
