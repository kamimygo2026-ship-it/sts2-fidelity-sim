"""DemisePower.AfterSideTurnEnd / PowderedDemise.OnUse 的行为与依赖门禁。"""

import unittest
from unittest.mock import patch

from sts2_sim import content, core, powers
from tests.test_potions import setup_content, restore_builtin


class TestDemisePotion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self):
        state = core.start_combat(self.deck, ("nibbit", "nibbit"), seed=1,
                                  potions=("powdered_demise",))
        for unit in (state.player, *state.enemies):
            unit.hp = unit.max_hp = 100
            unit.block = 30
            unit.powers.clear()
        return state

    def test_potion_targets_one_enemy_and_consumes_slot(self):
        """PowderedDemise：目标获得 9 层；Demise 不在施加时伤害。"""
        state = self.combat()
        core.step(state, core.Action("use_potion", target=1, slot=0))
        self.assertEqual([e.power("demise") for e in state.enemies], [0, 9])
        self.assertEqual([e.hp for e in state.enemies], [100, 100])
        self.assertEqual(state.potions, [None])

    def test_owner_turn_end_bypasses_block_without_decay(self):
        """DemisePower：participants.Contains(Owner)，Unblockable | Unpowered。"""
        state = self.combat()
        enemy = state.enemies[1]
        core.use_potion(state, "powdered_demise", 1)
        state.player.add_power("strength", 20)
        enemy.add_power("vulnerable", 3)
        powers.on_side_turn_end(state, [], "player")
        self.assertEqual(enemy.hp, 100)
        for expected in (91, 82):
            core._end_enemy_side_turn(state, [])
            self.assertEqual(enemy.hp, expected)
            self.assertEqual(enemy.block, 30)
            self.assertEqual(enemy.power("demise"), 9)
        self.assertEqual(state.enemies[0].hp, 100)

    def test_stacking_uses_current_amount(self):
        """DemisePower.StackType=Counter：两次施加合并，回合末造成 18 点伤害。"""
        state = self.combat()
        for _ in range(2):
            core.use_potion(state, "powdered_demise", 0)
        core._end_enemy_side_turn(state, [])
        self.assertEqual(state.enemies[0].hp, 82)

    def test_player_owner_only_ticks_at_player_turn_end(self):
        """DemisePower 以参与者为条件，不能硬编码只处理敌方拥有者。"""
        state = self.combat()
        state.player.add_power("demise", 9)
        core._end_enemy_side_turn(state, [])
        self.assertEqual(state.player.hp, 100)
        powers.on_side_turn_end(state, [], "player")
        self.assertEqual(state.player.hp, 91)
        self.assertEqual(state.player.block, 30)

    def test_artifact_negates_application(self):
        """DemisePower.Type=Debuff：ArtifactPower.TryModifyPowerAmountReceived 抵消。"""
        state = self.combat()
        state.enemies[0].add_power("artifact", 1)
        core.use_potion(state, "powdered_demise", 0)
        core._end_enemy_side_turn(state, [])
        self.assertEqual(state.enemies[0].hp, 100)
        self.assertEqual(state.enemies[0].power("demise"), 0)
        self.assertEqual(state.enemies[0].power("artifact"), 0)

    def test_buffer_prevents_one_tick(self):
        """CreatureCmd.Damage 经过 BufferPower 的生命损失阶段，不能直接改 hp。"""
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("buffer", 1)
        core.use_potion(state, "powdered_demise", 0)
        core._end_enemy_side_turn(state, [])
        self.assertEqual(enemy.hp, 100)
        self.assertEqual(enemy.power("buffer"), 0)
        core._end_enemy_side_turn(state, [])
        self.assertEqual(enemy.hp, 91)

    def test_null_dealer_and_non_attack_do_not_trigger_retaliation(self):
        """DemisePower.Damage 的 dealer=null，不能反伤玩家或触发有效攻击处决。"""
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("thorns", 5)
        enemy.add_power("the_gambit", 1)
        core.use_potion(state, "powdered_demise", 0)
        core._end_enemy_side_turn(state, [])
        self.assertEqual(enemy.hp, 91)
        self.assertEqual(state.player.hp, 100)

    def test_lethal_tick_uses_death_path_once(self):
        """CreatureCmd.Damage：致死进入统一死亡路径，下一次回合末不重复结算。"""
        state = self.combat()
        enemy = state.enemies[0]
        enemy.hp = 8
        core.use_potion(state, "powdered_demise", 0)
        events = []
        core._end_enemy_side_turn(state, events)
        self.assertEqual(enemy.hp, 0)
        self.assertFalse(enemy.alive())
        self.assertTrue(enemy.death_processed)
        later = []
        core._end_enemy_side_turn(state, later)
        self.assertFalse(any("demise" in event or "凋亡" in event for event in later))

    def test_missing_power_is_rejected_before_use(self):
        """撤掉 Demise 实现后重新装载：不能仍标可用、进入动作或消耗药水。"""
        try:
            with patch.object(powers, "IMPLEMENTED", powers.IMPLEMENTED - {"demise"}):
                setup_content()
                self.assertTrue(content.POTIONS["powdered_demise"].effects_incomplete)
                state = self.combat()
                self.assertFalse(core.potion_actions(state))
                with self.assertRaises(ValueError):
                    core.use_potion_at(state, 0, 0)
                self.assertEqual(state.potions, ["powdered_demise"])
        finally:
            setup_content()

    def test_all_usable_potions_have_implemented_power_dependencies(self):
        """门禁不变量：药水的 apply_power 依赖必须在实际实现白名单中。"""
        for potion in content.POTIONS.values():
            if not potion.effects_incomplete:
                for effect in potion.effects:
                    if effect.op == "apply_power":
                        self.assertIn(effect.power, powers.IMPLEMENTED, potion.pid)
