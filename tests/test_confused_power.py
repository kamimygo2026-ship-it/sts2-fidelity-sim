"""ConfusedPower.AfterCardDrawn 与 CardEnergyCost 的本场费用、随机流回归。"""

import unittest
from unittest.mock import patch

from sts2_sim import core, powers
from sts2_sim.observe import observe
from tests.test_potions import setup_content, restore_builtin


class TestConfusedPower(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self):
        state = core.start_combat(self.deck, ("nibbit",), seed=7)
        state.hand = []
        state.draw_pile = []
        state.discard = []
        state.player.powers.clear()
        state.enemies[0].powers.clear()
        state.enemies[0].hp = state.enemies[0].max_hp = 1000
        powers.apply_power_to(state, state.player, "confused", 1, [],
                              applier=state.player)
        return state

    def draw(self, state, cid="strike_ironclad", from_hand_draw=False):
        card = core.CardInstance(cid)
        state.draw_pile = [card]
        core.draw_cards(state, 1, [], from_hand_draw=from_hand_draw)
        return card

    def test_draw_randomizes_to_zero_through_three_on_dedicated_stream(self):
        """Confused.NextEnergyCost=CombatEnergyCosts.NextInt(4)，不看是否起手抽牌。"""
        for cost in range(4):
            for from_hand_draw in (False, True):
                with self.subTest(cost=cost, from_hand_draw=from_hand_draw):
                    state = self.combat()
                    with patch.object(state.hidden.rng["combat_energy_costs"],
                                      "randrange", return_value=cost) as roll:
                        card = self.draw(state, from_hand_draw=from_hand_draw)
                    roll.assert_called_once_with(4)
                    self.assertEqual(core.play_cost(state, card), cost)

    def test_negative_canonical_cost_skips_random_stream(self):
        """Confused.AfterCardDrawn：Canonical < 0 直接返回。"""
        state = self.combat()
        before = state.hidden.rng.get_state()
        card = self.draw(state, "normality")
        self.assertEqual(state.hidden.rng.get_state(), before)
        self.assertFalse(core.card_playable(state, card))

    def test_x_cost_consumes_randomness_but_spends_all_energy(self):
        """CardEnergyCost 构造把 CostsX 的 Canonical 设为 0；GetAmountToSpend 仍花全部能量。"""
        state = self.combat()
        state.energy = 5
        with patch.object(state.hidden.rng["combat_energy_costs"],
                          "randrange", return_value=2) as roll:
            card = self.draw(state, "cascade")
        roll.assert_called_once_with(4)
        self.assertEqual(core.play_cost(state, card), 5)

    def test_only_card_owner_randomizes(self):
        """Confused.AfterCardDrawn：card.Owner != Owner.Player 时返回。"""
        state = self.combat()
        state.player.powers.clear()
        state.enemies[0].add_power("confused", 1)
        before = state.hidden.rng.get_state()
        card = self.draw(state)
        self.assertEqual(core.play_cost(state, card), 1)
        self.assertEqual(state.hidden.rng.get_state(), before)

    def test_artifact_blocks_confused_without_changing_existing_hand(self):
        """Confused.Type=Debuff；PowerCmd.Apply 经过 Artifact，施加本身不改已有手牌。"""
        state = self.combat()
        state.player.powers.clear()
        card = core.CardInstance("strike_ironclad")
        state.hand = [card]
        state.player.add_power("artifact", 1)
        self.assertFalse(powers.apply_power_to(state, state.player, "confused", 1, [],
                                              applier=state.player))
        self.assertEqual(state.player.power("confused"), 0)
        self.assertEqual(state.player.power("artifact"), 0)
        self.assertTrue(powers.apply_power_to(state, state.player, "confused", 1, [],
                                             applier=state.player))
        self.assertEqual(core.play_cost(state, card), 1)

    def test_full_hand_does_not_consume_cost_randomness(self):
        """CardPileCmd.DrawInternal：手牌满时没有抽牌事件，不应预掷未抽到牌的费用。"""
        state = self.combat()
        state.hand = [core.CardInstance("strike_ironclad") for _ in range(core.MAX_HAND_SIZE)]
        state.draw_pile = [core.CardInstance("defend_ironclad")]
        before = state.hidden.rng.get_state()
        core.draw_cards(state, 1, [])
        self.assertEqual(state.hidden.rng.get_state(), before)
        self.assertEqual(len(state.draw_pile), 1)

    def test_adding_to_hand_is_not_drawing(self):
        """CardPileCmd.Add 不派发 AfterCardDrawn，生成/捞牌不能多掷一次费用。"""
        state = self.combat()
        before = state.hidden.rng.get_state()
        card = core.CardInstance("strike_ironclad")
        core.add_to_hand(state, card, [])
        self.assertEqual(core.play_cost(state, card), 1)
        self.assertEqual(state.hidden.rng.get_state(), before)

    def test_repeat_draw_rerolls_same_card(self):
        """每次 AfterCardDrawn 都追加新的 Absolute 费用，后写覆盖前写。"""
        state = self.combat()
        with patch.object(state.hidden.rng["combat_energy_costs"],
                          "randrange", side_effect=[3, 0]) as roll:
            card = self.draw(state)
            self.assertEqual(core.play_cost(state, card), 3)
            state.hand.remove(card)
            state.draw_pile = [card]
            core.draw_cards(state, 1, [])
            self.assertEqual(core.play_cost(state, card), 0)
        self.assertEqual(roll.call_count, 2)

    def test_combat_cost_survives_play_and_turn_end(self):
        """SetThisCombat 的 EndOfCombat 修饰不被 AfterCardPlayed/EndOfTurnCleanup 清掉。"""
        state = self.combat()
        with patch.object(state.hidden.rng["combat_energy_costs"],
                          "randrange", return_value=2):
            card = self.draw(state)
        core.step(state, core.Action("play_card", 0, 0))
        self.assertEqual(state.energy, 1)
        self.assertIn(card, state.discard)
        state.player.powers.pop("confused")
        state.draw_pile = [core.CardInstance("defend_ironclad") for _ in range(5)]
        core.step(state, core.Action("end_turn"))
        self.assertEqual(core.play_cost(state, card), 2)

    def test_temporary_free_and_confused_follow_application_order(self):
        """CardEnergyCost.GetWithModifiers 按修饰加入顺序执行 Absolute。"""
        state = self.combat()
        card = core.CardInstance("strike_ironclad", free_this_turn=True)
        state.draw_pile = [card]
        with patch.object(state.hidden.rng["combat_energy_costs"],
                          "randrange", return_value=3):
            core.draw_cards(state, 1, [])
        self.assertEqual(core.play_cost(state, card), 3)
        card.free_this_turn = True
        self.assertEqual(core.play_cost(state, card), 0)
        core.step(state, core.Action("play_card", 0, 0))
        self.assertEqual(core.play_cost(state, card), 3)

    def test_local_cost_precedes_global_cost_hooks(self):
        """CardEnergyCost.GetWithModifiers：本地随机费用→全局加价→Late 免费。"""
        state = self.combat()
        state.player.add_power("borrowed_time", 1)
        with patch.object(state.hidden.rng["combat_energy_costs"],
                          "randrange", return_value=2):
            card = self.draw(state, "defend_ironclad")
        self.assertEqual(core.play_cost(state, card), 3)
        state.player.add_power("corruption", 1)
        self.assertEqual(core.play_cost(state, card), 0)

    def test_observation_and_legality_use_revealed_random_cost(self):
        """公开抽牌之后卡面费用、合法动作与实际扣费必须读同一判据。"""
        state = self.combat()
        state.energy = 2
        with patch.object(state.hidden.rng["combat_energy_costs"],
                          "randrange", return_value=3):
            card = self.draw(state)
        self.assertEqual(observe(state).hand[0].cost, 3)
        self.assertFalse(core.card_playable(state, card))
        self.assertNotIn(core.Action("play_card", 0, 0), core.legal_actions(state))

    def test_snapshot_replays_costs_without_changing_other_random_streams(self):
        """Confused 只消费 CombatEnergyCosts；SL 保存费用字段与隐藏随机流。"""
        state = self.combat()
        state.draw_pile = [core.CardInstance("strike_ironclad") for _ in range(6)]
        before = state.hidden.rng.get_state()
        snap = core.snapshot(state)
        core.draw_cards(state, 3, [])
        restored = core.restore(snap)
        core.draw_cards(restored, 3, [])
        self.assertEqual([core.play_cost(state, c) for c in state.hand],
                         [core.play_cost(restored, c) for c in restored.hand])
        self.assertEqual(state.hidden.rng.get_state(), restored.hidden.rng.get_state())
        after = state.hidden.rng.get_state()
        self.assertEqual([name for name in before if before[name] != after[name]],
                         ["combat_energy_costs"])
        restored_after_draw = core.restore(core.snapshot(state))
        self.assertEqual(observe(state).hand, observe(restored_after_draw).hand)

    def test_new_combat_does_not_keep_randomized_cost(self):
        """CardEnergyCost.SetThisCombat：费用属于战斗副本，不能写进永久牌组。"""
        state = self.combat()
        with patch.object(state.hidden.rng["combat_energy_costs"],
                          "randrange", return_value=3):
            card = self.draw(state)
        next_combat = core.start_combat([card], ("nibbit",), seed=7)
        self.assertEqual(core.play_cost(next_combat, next_combat.hand[0]), 1)

    def test_single_stack_does_not_roll_multiple_times(self):
        """Confused.AfterCardDrawn 不读 Amount；PowerCmd.ModifyAmount 增层也只掷一次。"""
        state = self.combat()
        state.player.add_power("confused", 3)
        with patch.object(state.hidden.rng["combat_energy_costs"],
                          "randrange", return_value=2) as roll:
            self.draw(state)
        roll.assert_called_once_with(4)
