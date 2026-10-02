"""**铁甲战士能力**的源码级回归测试（``docs/09`` L2）。

这一组测的是"引擎新实现的能力是否与反编译源码逐条一致"，每个能力的
``source=`` 都写在 ``powers.RULES`` 里，测试名与断言的依据也逐条注明。

这些不是"看起来对"的断言。每个测试都对应源码里**一个容易被漏掉的条件**：

| 能力 | 容易漏掉的那一条 | 漏掉的后果 |
|---|---|---|
| ``rage`` | 只认**攻击牌**；自己这边回合结束整个移除 | 变成"每出一张牌给格挡" / 永久保留 |
| ``feel_no_pain`` | 格挡是 ``Unpowered`` | 被敏捷/脆弱静默改写 |
| ``dark_embrace`` | 虚无消耗要**延迟**到回合末抽 | 抽到的牌当场被清手牌弃掉（等于没抽） |
| ``juggernaut`` | 目标走 ``Rng.CombatTargets``；伤害 ``Unpowered`` | 随机流分叉 / 被力量易伤放大 |
| ``flame_barrier`` | 被**完全格挡**也反伤；在**对方**回合结束才移除 | 少反伤 / 提前消失 |
| ``crimson_mantle`` | ``SelfDamage`` 每打一张 +1 | 自伤永远是 0（白送一张牌） |
| ``inferno`` | 自伤**本身**会触发下半截 | 每回合只有掉血、没有 AoE |
| ``setup_strike`` / ``mangle`` | 撤销的是**当前层数**，不是初始值 | 连续两张会凭空多出力量 |
| ``cruelty`` | 加在**倍率**上（1.5+25/100），不是再乘一次 | 17.5 变成 18.75 |
| ``tank`` | 只对**自己** ×1.5（``target != Owner → 1``） | 敌人打别人也被放大 |
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"


def setup_content():
    if not CONTENT.exists():
        raise unittest.SkipTest(f"缺内容目录 {CONTENT}")
    from sts2_sim.featurize import configure_content
    configure_content(str(CONTENT))
    from sts2_sim.content import STARTING_DECK
    return list(STARTING_DECK)


def restore_builtin() -> None:
    from sts2_sim.content import load_builtin
    from sts2_sim.featurize import rebuild_vocab
    load_builtin()
    rebuild_vocab()


class IroncladPowerTestCase(unittest.TestCase):
    """公共脚手架：一副起手牌 + 一只/两只怪，以及"把某张牌塞进手里并打出"。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, enemies=("nibbit",), seed: int = 5):
        from sts2_sim.core import start_combat
        return start_combat(self.deck, tuple(enemies), seed=seed)

    def play(self, state, cid: str, target: int = 0):
        """把 ``cid`` 放进手牌并真的**打出去**（走 ``core.step``，不是直接加能力）。"""
        from sts2_sim.core import Action, CardInstance, step
        card = CardInstance(cid)
        state.hand.append(card)
        state.energy = 9
        index = state.hand.index(card)
        return step(state, Action("play_card", index, target))


class RageTest(IroncladPowerTestCase):
    """``RagePower``（``RagePower.cs:23-37``）。"""

    def test_attack_card_grants_unpowered_block(self):
        """``AfterCardPlayed``：牌属于拥有者本人 **且** 类型是攻击 → +Amount 格挡。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("rage", 4)
        player.block = 0
        power_rules.on_card_played(state, [], "attack")
        self.assertEqual(player.block, 4)
        power_rules.on_card_played(state, [], "skill")
        self.assertEqual(player.block, 4, "技能牌不触发（源码只认 CardType.Attack）")
        power_rules.on_card_played(state, [], "power")
        self.assertEqual(player.block, 4, "能力牌不触发")

    def test_block_is_unpowered(self):
        """``ValueProp.Unpowered``：敏捷不加成、脆弱不打折。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("rage", 3)
        player.add_power("dexterity", 5)
        player.add_power("frail", 3)
        player.block = 0
        power_rules.on_card_played(state, [], "attack")
        self.assertEqual(player.block, 3, "暴怒的格挡不吃敏捷、也不被脆弱打折")

    def test_removed_at_owners_side_turn_end(self):
        """``AfterSideTurnEnd``：``participants.Contains(Owner)`` → ``PowerCmd.Remove``。

        层数是 ``Counter`` 却**不递减** —— 当成递减会让暴怒多留好几个回合。
        """
        from sts2_sim import powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("rage", 3)
        power_rules.on_side_turn_end(state, [], "enemy")
        self.assertIn("rage", player.powers, "敌方回合结束不移除（源码看的是 participants）")
        power_rules.on_turn_end(player, state, [])
        self.assertNotIn("rage", player.powers)

    def test_enemy_held_rage_does_not_trigger_on_player_attack(self):
        """``cardPlay.Card.Owner == base.Owner.Player``：拥有者是敌人时永不成立。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("rage", 3)
        enemy.block = 0
        power_rules.on_card_played(state, [], "attack")
        self.assertEqual(enemy.block, 0)


class FeelNoPainTest(IroncladPowerTestCase):
    """``FeelNoPainPower``（``FeelNoPainPower.cs:19-25``）。"""

    def test_exhaust_grants_block(self):
        """``AfterCardExhausted``：任意原因的消耗都给格挡。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("feel_no_pain", 3)
        player.block = 0
        card = core.CardInstance("apparition")      # 带 Exhaust 的技能牌
        state.play_area.append(card)
        core._finish_played_card(state, card, card.definition(), [])
        self.assertIn(card, state.exhaust)
        self.assertEqual(player.block, 3)

    def test_non_exhaust_play_grants_nothing(self):
        """对照组：普通牌进弃牌堆，**不是**消耗。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("feel_no_pain", 3)
        player.block = 0
        card = core.CardInstance("strike_ironclad")
        state.play_area.append(card)
        core._finish_played_card(state, card, card.definition(), [])
        self.assertNotIn(card, state.exhaust)
        self.assertEqual(player.block, 0)

    def test_corruption_exhaust_also_triggers(self):
        """``CorruptionPower`` 让技能牌消耗 → 无痛照样给格挡（经典联动）。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("feel_no_pain", 4)
        player.add_power("corruption", 1)
        player.block = 0
        card = core.CardInstance("defend_ironclad")
        state.play_area.append(card)
        core._finish_played_card(state, card, card.definition(), [])
        self.assertIn(card, state.exhaust, "腐败下技能牌必须消耗")
        self.assertEqual(player.block, 4)

    def test_selected_exhaust_triggers(self):
        """选牌消耗（``select_card`` → ``CardCmd.Exhaust``）同样触发。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("feel_no_pain", 2)
        player.block = 0
        card = core.CardInstance("strike_ironclad")
        state.hand = [card]
        state.pending = core.PendingSelection(purpose="exhaust", source_pile="hand",
                                             remaining=1)
        core._resolve_selection(state, 0, [])
        self.assertIn(card, state.exhaust)
        self.assertEqual(player.block, 2)


class DarkEmbraceTest(IroncladPowerTestCase):
    """``DarkEmbracePower``（``DarkEmbracePower.cs:37-59``）。"""

    def test_normal_exhaust_draws_immediately(self):
        """非虚无的消耗 → ``CardPileCmd.Draw(Amount)`` **立刻**抽。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("dark_embrace", 2)
        state.hand = []
        card = core.CardInstance("apparition")
        state.play_area.append(card)
        core._finish_played_card(state, card, card.definition(), [])
        self.assertEqual(len(state.hand), 2)

    def test_ethereal_exhaust_defers_to_turn_end(self):
        """虚无造成的消耗**不当场抽**，攒到 ``AfterSideTurnEnd`` 才按 ``Amount × 张数`` 抽。

        源码注释写明了原因：回合结束时虚无牌是在**清手牌之前**被消耗的，
        当场抽到的牌会被同一次清手牌弃掉 —— 也就是"等于没抽"。
        """
        from sts2_sim import core, powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("dark_embrace", 2)
        state.hand = []
        state.draw_pile = [core.CardInstance("strike_ironclad") for _ in range(6)]
        ethereal = [core.CardInstance("dazed") for _ in range(2)]
        state.hand = ethereal
        core._run_hand_triggers(state, [])
        self.assertEqual(len(state.hand), 0, "虚无牌消耗时不该当场抽牌")
        self.assertEqual(len(state.exhaust), 2)
        self.assertEqual(player.power_flags.get("dark_embrace_ethereal"), 2)
        power_rules.on_turn_end(player, state, [])
        self.assertEqual(len(state.hand), 4, "回合结束按 层数 × 虚无张数 补抽")
        self.assertIsNone(player.power_flags.get("dark_embrace_ethereal"))


class JuggernautTest(IroncladPowerTestCase):
    """``JuggernautPower``（``JuggernautPower.cs:17-29``）。"""

    def test_block_gain_deals_unpowered_damage(self):
        """获得格挡 → 对随机敌人造成 ``Amount`` 点 ``Unpowered`` 伤害。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("juggernaut", 6)
        player.add_power("strength", 10)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        core.gain_block(state, player, 5, [], unpowered=True, label="测试")
        self.assertEqual(player.block, 5)
        self.assertEqual(100 - enemy.hp, 6, "Unpowered：不被力量放大")

    def test_no_damage_when_block_gain_is_zero(self):
        """``if (!(amount <= 0m) …)``：获得 0 点格挡不触发。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("juggernaut", 6)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        core._apply_block_gain(state, player, 0, [])
        self.assertEqual(enemy.hp, 100)

    def test_only_owner_block_triggers(self):
        """``creature == base.Owner``：别人拿格挡不触发。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("juggernaut", 6)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        core.gain_block(state, enemy, 5, [], unpowered=True, label="测试")
        self.assertEqual(enemy.hp, 100)

    def test_target_is_independent_random_stream(self):
        """目标走 ``Rng.CombatTargets``（``JuggernautPower.cs:24``），不是主随机流。

        做法：把 ``combat_targets`` 这条流的状态存下来，先"预演"一次取下标，
        再恢复状态让板甲去取 —— 两边必须是同一只。若板甲走的是别的流
        （或主随机流），这条断言就会失败。
        """
        from sts2_sim import core
        state = self.combat(("nibbit", "axebot"))
        player = state.player
        player.add_power("juggernaut", 6)
        for enemy in state.enemies:
            enemy.hp = enemy.max_hp = 100
        saved = state.hidden.rng.get_state()
        expected = state.hidden.rng.next_index("combat_targets", 2)
        state.hidden.rng.set_state(saved)
        core.gain_block(state, player, 5, [], unpowered=True, label="测试")
        hurt = [i for i, e in enumerate(state.enemies) if e.hp < 100]
        self.assertEqual(hurt, [expected])


class FlameBarrierTest(IroncladPowerTestCase):
    """``FlameBarrierPower``（``FlameBarrierPower.cs:18-32``）。"""

    def test_reflects_powered_attack_even_when_fully_blocked(self):
        """条件里**没有**"有没有掉血"：被完全格挡照样反伤。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("flame_barrier", 4)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        power_rules.on_attacked(player, enemy, state, [],
                                unblocked=0, from_card=False, powered=True)
        self.assertEqual(100 - enemy.hp, 4)

    def test_reflect_is_unpowered(self):
        """反伤是 ``Unpowered``：不吃拥有者的力量。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("flame_barrier", 4)
        player.add_power("strength", 9)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        power_rules.on_attacked(player, enemy, state, [],
                                unblocked=3, from_card=False, powered=True)
        self.assertEqual(100 - enemy.hp, 4)

    def test_no_reflect_for_unpowered_damage(self):
        """``props.IsPoweredAttack()``：中毒/掉血这类不算有效攻击。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("flame_barrier", 4)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        power_rules.on_attacked(player, enemy, state, [],
                                unblocked=3, from_card=False, powered=False)
        self.assertEqual(enemy.hp, 100)

    def test_removed_at_opposing_side_turn_end(self):
        """``if (base.Owner.Side != side) PowerCmd.Remove(this)`` —— 是**对方**那边结束时移除。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("flame_barrier", 4)
        power_rules.on_side_turn_end(state, [], "player")
        self.assertIn("flame_barrier", player.powers, "自己回合结束不移除")
        power_rules.on_side_turn_end(state, [], "enemy")
        self.assertNotIn("flame_barrier", player.powers)

    def test_survives_the_whole_enemy_turn(self):
        """实战路径：敌方回合里仍反伤，回合结束才消失。"""
        from sts2_sim.core import Action, step
        state = self.combat()
        player = state.player
        player.add_power("flame_barrier", 5)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        step(state, Action("end_turn"))
        self.assertNotIn("flame_barrier", player.powers, "敌方回合结束后应移除")
        self.assertLess(enemy.hp, 200, "敌方回合内至少反伤过一次")


class CrimsonMantleTest(IroncladPowerTestCase):
    """``CrimsonMantlePower``（``CrimsonMantle.cs:28`` + ``CrimsonMantlePower.cs:25-34``）。"""

    def test_self_damage_equals_times_played(self):
        """``SelfDamage`` 初值 0、每打出一张 +1：打两张就是 2 点自伤。"""
        state = self.combat()
        player = state.player
        for _ in range(2):
            self.play(state, "crimson_mantle")
        self.assertEqual(player.power("crimson_mantle"), 14)
        self.assertEqual(player.power_flags["crimson_mantle_self_damage"], 2)

    def test_turn_start_damages_then_gains_unpowered_block(self):
        """``AfterPlayerTurnStart``：先掉血（``Unblockable|Unpowered``），再 +Amount 格挡。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        self.play(state, "crimson_mantle")
        player.hp = 40
        player.block = 0
        player.add_power("dexterity", 5)
        player.add_power("frail", 3)
        core.start_player_turn(state, [])
        self.assertEqual(40 - player.hp, 1, "自伤不吃格挡、不受任何修正")
        self.assertEqual(player.block, 7, "披风格挡是 Unpowered（不吃敏捷/脆弱）")


class InfernoTest(IroncladPowerTestCase):
    """``InfernoPower``（``Inferno.cs:23`` + ``InfernoPower.cs:26-49``）。"""

    def test_self_damage_then_retaliate_against_all_enemies(self):
        """自伤本身会走 ``AfterDamageReceived`` → 下半截对**全体**敌人开火。"""
        from sts2_sim import core
        state = self.combat(("nibbit", "axebot"))
        player = state.player
        self.play(state, "inferno")
        for enemy in state.enemies:
            enemy.hp = enemy.max_hp = 100
        player.hp = 50
        core.start_player_turn(state, [])
        self.assertEqual(50 - player.hp, 1)
        self.assertEqual([e.hp for e in state.enemies], [94, 94])

    def test_no_retaliate_when_damage_fully_blocked(self):
        """``result.UnblockedDamage <= 0`` 就返回：被挡下的一击不开火。

        用 ``unblocked=0`` 直接打这一跳，避免依赖怪物意图的随机性。
        """
        from sts2_sim import powers as power_rules
        state = self.combat(("nibbit", "axebot"))
        player = state.player
        player.add_power("inferno", 6)
        for enemy in state.enemies:
            enemy.hp = enemy.max_hp = 100
        power_rules.on_attacked(player, state.enemies[0], state, [],
                                unblocked=0, from_card=False, powered=True)
        self.assertEqual([e.hp for e in state.enemies], [100, 100])

    def test_no_retaliate_on_enemy_turn(self):
        """``CurrentSide == Owner.Side``：敌方回合掉血**不**触发（否则每回合都在白打）。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("inferno", 6)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        state.phase = "enemy"
        power_rules.on_attacked(player, enemy, state, [],
                                unblocked=5, from_card=False, powered=True)
        self.assertEqual(enemy.hp, 100)


class TemporaryStrengthTest(IroncladPowerTestCase):
    """``SetupStrikePower`` / ``ManglePower``（``TemporaryStrengthPower.cs:134-155``）。"""

    def test_setup_strike_adds_then_restores(self):
        """施加时 ``+Sign*Amount``，``AfterSideTurnEnd`` 移除并 ``-Sign*Amount``。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        self.play(state, "setup_strike")
        self.assertEqual(player.power("strength"), 3)
        self.assertEqual(player.power("setup_strike"), 3)
        core.tick_powers(player, state, [])
        self.assertEqual(player.power("strength"), 0)
        self.assertNotIn("setup_strike", player.powers)

    def test_mangle_is_negative_strength(self):
        """``ManglePower.IsPositive => false``：先扣，敌方回合结束还回来。"""
        from sts2_sim import core
        state = self.combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.add_power("strength", 5)
        self.play(state, "mangle")
        self.assertEqual(enemy.power("strength"), -5)
        self.assertEqual(100 - enemy.hp, 20, "重击的伤害在扣力量**之前**结算")
        core._end_enemy_side_turn(state, [])
        self.assertEqual(enemy.power("strength"), 5)
        self.assertNotIn("mangle", enemy.powers)

    def test_stacking_restores_the_whole_amount(self):
        """撤销的是**当前层数**：两张临时力量叠到 6，回合末要还 6（不是每次还 3）。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        for _ in range(2):
            self.play(state, "setup_strike")
        self.assertEqual(player.power("strength"), 6)
        core.tick_powers(player, state, [])
        self.assertEqual(player.power("strength"), 0, "还 6 而不是 3")

    def test_setup_strike_keeps_strength_through_enemy_turn(self):
        """临时力量覆盖的是"自己这一回合"，所以敌方回合开始时**还在**。"""
        from sts2_sim.core import Action, step
        state = self.combat()
        player = state.player
        self.play(state, "setup_strike")
        step(state, Action("end_turn"))
        self.assertLessEqual(player.power("strength"), 0, "回合结束后已经还回去了")


class DemonFormTest(IroncladPowerTestCase):
    """``DemonFormPower``（``DemonFormPower.cs:58-66``）。"""

    def test_gains_strength_at_own_turn_start(self):
        """``AfterSideTurnStart``：``participants.Contains(Owner)`` → +Amount 力量。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("demon_form", 3)
        core.tick_powers(player, state, [])
        self.assertEqual(player.power("strength"), 0, "回合结束不加")
        core.start_player_turn(state, [])
        self.assertEqual(player.power("strength"), 3)

    def test_stacks_over_turns(self):
        from sts2_sim import core
        from sts2_sim.core import Action, step
        state = self.combat()
        player = state.player
        player.add_power("demon_form", 2)
        step(state, Action("end_turn"))
        step(state, Action("end_turn"))
        self.assertEqual(player.power("strength"), 4)


class PyreTest(IroncladPowerTestCase):
    """``PyrePower``（``PyrePower.cs:16-23`` + ``PlayerCombatState.cs:101/164``）。"""

    def test_max_energy_bonus_every_turn(self):
        """``ModifyMaxEnergy``：能量重置（``Energy = MaxEnergy``）时 +Amount。"""
        from sts2_sim import core
        from sts2_sim.core import Action, step
        state = self.combat()
        player = state.player
        self.play(state, "pyre")
        self.assertEqual(player.power("pyre"), 1)
        state.turn = 2
        step(state, Action("end_turn"))
        self.assertEqual(state.energy, core.BASE_ENERGY + 1)


class CorruptionTest(IroncladPowerTestCase):
    """``CorruptionPower``（``CorruptionPower.cs:17-46``）。"""

    def test_skills_cost_zero_attacks_do_not(self):
        """``TryModifyEnergyCostInCombatLate``：只对**拥有者的技能牌**归零。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("corruption", 1)
        skill = core.CardInstance("defend_ironclad")
        attack = core.CardInstance("strike_ironclad")
        self.assertEqual(core.play_cost(state, skill), 0)
        self.assertEqual(core.play_cost(state, attack), attack.definition().cost)

    def test_cost_zero_also_covers_affliction_surcharge(self):
        """阶段是 ``…Late``：腐败跑在苦痛之后，加价也被一并清零。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("corruption", 1)
        player.add_power("tangled", 2)
        card = core.CardInstance("defend_ironclad")
        card.affliction = "entangled"
        self.assertEqual(core.play_cost(state, card), 0)

    def test_played_skills_are_exhausted(self):
        """``ModifyCardPlayResultLocation`` → ``PileType.Exhaust``。"""
        state = self.combat()
        state.player.add_power("corruption", 1)
        self.play(state, "defend_ironclad")
        self.assertTrue(any(c.cid == "defend_ironclad" for c in state.exhaust))

    def test_no_corruption_means_normal_piles(self):
        """对照组：没有腐败时技能牌进弃牌堆。"""
        state = self.combat()
        self.play(state, "defend_ironclad")
        self.assertTrue(any(c.cid == "defend_ironclad" for c in state.discard))
        self.assertEqual(state.exhaust, [])


class CrueltyTest(IroncladPowerTestCase):
    """``CrueltyPower``（``CrueltyPower.cs:17-28``）。"""

    def test_adds_to_the_vulnerable_multiplier(self):
        """``ModifyVulnerableMultiplier``：``amount + Amount/100``，**加在倍率上**。

        10 点基础伤害对易伤目标：1.5 → 1.75，``int(10×1.75) = 17``。
        写成"再乘一次 1.25"会得到 ``int(15×1.25) = 18``。
        """
        from sts2_sim.core import compute_damage
        state = self.combat()
        player = state.player
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        enemy.add_power("vulnerable", 3)
        self.assertEqual(compute_damage(10, player, enemy), 15)
        player.add_power("cruelty", 25)
        self.assertEqual(compute_damage(10, player, enemy), 17)

    def test_cruelty_owner_taking_damage_is_unaffected(self):
        """``if (target == base.Owner) return amount;`` —— 拥有者自己被打时不加。"""
        from sts2_sim.core import compute_damage
        state = self.combat()
        player = state.player
        enemy = state.enemies[0]
        player.add_power("cruelty", 25)
        player.add_power("vulnerable", 3)
        self.assertEqual(compute_damage(10, enemy, player), 15)

    def test_unpowered_damage_is_unaffected(self):
        """``props.IsPoweredAttack()`` 为假时返回原倍率（``unpowered`` 不走乘法阶段）。"""
        from sts2_sim.core import deal_damage
        state = self.combat()
        player = state.player
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        enemy.add_power("vulnerable", 3)
        player.add_power("cruelty", 25)
        deal_damage(state, player, enemy, 10, [], unpowered=True)
        self.assertEqual(200 - enemy.hp, 10)


class TankTest(IroncladPowerTestCase):
    """``TankPower``（``TankPower.cs:44-55``；``GuardedPower`` 那一半是多人专用）。"""

    def test_owner_takes_double_damage_rounded_by_int(self):
        """``target != Owner → 1``；是 Owner 且有效攻击 → ``DamageIncrease`` = 1.5。"""
        from sts2_sim.core import compute_damage
        state = self.combat()
        player = state.player
        enemy = state.enemies[0]
        self.assertEqual(compute_damage(10, enemy, player), 10)
        player.add_power("tank", 1)
        self.assertEqual(compute_damage(10, enemy, player), 15)

    def test_only_the_owner_is_affected(self):
        """敌人之间互殴不受玩家身上坦克的影响。"""
        from sts2_sim.core import compute_damage
        state = self.combat(("nibbit", "axebot"))
        state.player.add_power("tank", 1)
        a, b = state.enemies
        self.assertEqual(compute_damage(10, a, b), 10)


class RuptureTest(IroncladPowerTestCase):
    """``RupturePower``（``RupturePower.cs:31-66``）。"""

    def test_lose_hp_grants_strength(self):
        """自伤牌（``lose_hp`` → ``CreatureCmd.Damage(Unblockable|Unpowered|Move)``）也触发。

        ``AfterDamageReceived`` 的 ``target == Owner && UnblockedDamage > 0``
        两个条件与"是谁打的"无关 —— 旧引擎的自伤路径**根本不发这个钩子**。
        """
        state = self.combat()
        player = state.player
        player.add_power("rupture", 2)
        self.play(state, "bloodletting")
        self.assertEqual(player.power("strength"), 2)
        self.assertEqual(player.hp, player.max_hp - 3)

    def test_card_self_damage_defers_strength_until_the_card_resolves(self):
        """卡牌造成的自伤 → 力量**等到这张牌结算完**才发（源码的 ``playedCards`` 机制）。

        铁证是重击：它"先自伤 2、后打 15"。若力量当场就发，
        打出去的会是 16 点 —— 而真机是 15 点 + 事后 1 点力量。
        """
        state = self.combat()
        player = state.player
        player.add_power("rupture", 1)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        self.play(state, "hemokinesis")
        self.assertEqual(100 - enemy.hp, 15, "自伤那一下的力量不能提前生效")
        self.assertEqual(player.power("strength"), 1)

    def test_no_strength_without_hp_loss(self):
        """``result.UnblockedDamage > 0``：被完全挡下不算。

        用直接调用打这一跳，避免依赖怪物意图的随机性。
        """
        from sts2_sim import powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("rupture", 3)
        power_rules.on_attacked(player, state.enemies[0], state, [],
                                unblocked=0, from_card=False, powered=True)
        self.assertEqual(player.power("strength"), 0)

    def test_no_strength_on_enemy_turn(self):
        """``CurrentSide == Owner.Side``：敌方回合掉血不发力量。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("rupture", 3)
        state.phase = "enemy"
        power_rules.on_attacked(player, state.enemies[0], state, [],
                                unblocked=5, from_card=False, powered=True)
        self.assertEqual(player.power("strength"), 0)


class ViciousTest(IroncladPowerTestCase):
    """``ViciousPower``（``ViciousPower.cs:19-25``）。"""

    def test_applying_vulnerable_draws(self):
        """``applier == base.Owner && power is VulnerablePower && amount > 0`` → 抽 ``Amount`` 张。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("vicious", 1)
        state.hand = []
        state.draw_pile = [core.CardInstance("strike_ironclad") for _ in range(5)]
        self.play(state, "bash")               # 伤害 + 上 2 层易伤
        self.assertEqual(state.enemies[0].power("vulnerable"), 2)
        self.assertEqual(len(state.hand), 1, "上易伤应当抽 1 张")

    def test_other_players_vulnerable_does_not_draw(self):
        """``applier == base.Owner``：**别人**上易伤不算。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("vicious", 1)
        state.hand = []
        power_rules.on_power_applied(state, [], "vulnerable",
                                     state.enemies[0], 2)
        self.assertEqual(len(state.hand), 0)

    def test_drawing_only_on_positive_amount(self):
        """``!(amount <= 0m)``：扣易伤不算。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("vicious", 1)
        state.hand = []
        power_rules.on_power_applied(state, [], "vulnerable", player, -2)
        self.assertEqual(len(state.hand), 0)


class JugglingTest(IroncladPowerTestCase):
    """``JugglingPower``（``JugglingPower.cs:36-52``）。"""

    def test_third_attack_clones_itself_into_hand(self):
        """``attacksPlayedThisTurn == 3`` 时，把**这张牌**复制 ``Amount`` 份进手牌。"""
        state = self.combat()
        player = state.player
        player.add_power("juggling", 1)
        state.hand = []
        state.draw_pile = []
        state.discard = []
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        for index in range(3):
            self.play(state, "strike_ironclad")
            self.assertEqual(state.attacks_played_this_turn, index + 1)
            self.assertEqual(len(state.hand), 1 if index == 2 else 0,
                             "只有第 3 张攻击牌会留下复制品")

    def test_skills_do_not_count(self):
        """``cardPlay.Card.Type == Attack``：技能牌不进计数。"""
        state = self.combat()
        state.player.add_power("juggling", 3)
        state.hand = []
        self.play(state, "defend_ironclad")
        self.assertEqual(state.attacks_played_this_turn, 0)

    def test_counter_resets_each_turn(self):
        """回合结束清零（真机是 ``AfterSideTurnEnd`` 把私有计数器归零）。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("juggling", 1)
        state.hand = []
        state.draw_pile = []
        state.discard = []
        self.play(state, "strike_ironclad")
        core.start_player_turn(state, [])
        self.assertEqual(state.attacks_played_this_turn, 0)


class AggressionTest(IroncladPowerTestCase):
    """``AggressionPower``（``AggressionPower.cs:20-37``）。"""

    def test_pulls_an_attack_from_discard_and_upgrades_it(self):
        """``BeforeSideTurnStart``：从弃牌堆捞 ``Amount`` 张攻击牌进手牌并升级。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("aggression", 1)
        state.discard = [core.CardInstance("bash")]
        state.hand = []
        state.draw_pile = []
        core.start_player_turn(state, [])
        pulled = [c for c in state.hand if c.cid == "bash"]
        self.assertEqual(len(pulled), 1)
        self.assertTrue(pulled[0].upgraded, "捞回来的攻击牌要升级")
        self.assertNotIn("bash", [c.cid for c in state.discard])

    def test_only_attacks_are_pulled(self):
        """``c.Type == CardType.Attack``：技能牌留在弃牌堆。

        （抽牌堆给足 5 张，免得回合开始的抽牌把弃牌堆洗回去 ——
        那样考的就变成洗牌而不是侵略了。）
        """
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("aggression", 2)
        state.discard = [core.CardInstance("defend_ironclad")]
        state.hand = []
        state.draw_pile = [core.CardInstance("strike_ironclad") for _ in range(5)]
        core.start_player_turn(state, [])
        self.assertEqual([c.cid for c in state.discard], ["defend_ironclad"])
        self.assertNotIn("defend_ironclad", [c.cid for c in state.hand])

    def test_upgrade_is_combat_local(self):
        """卡面是"for the rest of combat"：**不进** ``permanent_card_changes``。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("aggression", 1)
        state.discard = [core.CardInstance("bash", link_uid=4242)]
        state.hand = []
        state.draw_pile = []
        core.start_player_turn(state, [])
        self.assertEqual(state.permanent_card_changes, {})


class StampedeTest(IroncladPowerTestCase):
    """``StampedePower``（``StampedePower.cs:18-33``）。"""

    def test_autoplays_an_attack_at_end_of_turn(self):
        """``AfterAutoPostPlayPhaseEntered``：随机自动打出一张手牌攻击牌（免费）。"""
        from sts2_sim.core import Action, CardInstance, step
        state = self.combat()
        player = state.player
        player.add_power("stampede", 1)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        state.hand = [CardInstance("strike_ironclad")]
        state.draw_pile = [CardInstance("strike_ironclad") for _ in range(4)]
        step(state, Action("end_turn"))
        self.assertLess(enemy.hp, 200, "回合结束时应当自动打出一张攻击牌")

    def test_runs_before_the_hand_is_flushed(self):
        """时机是 ``AfterAutoPostPlayPhaseEntered`` —— **清手牌之前**。

        只用"手牌里只剩这一张攻击牌"来验证：如果实现成"先清手牌再自动打出"，
        那张牌已经在弃牌堆里、打不出来（敌人血量不变）。
        """
        from sts2_sim.core import Action, CardInstance, step
        state = self.combat()
        player = state.player
        player.add_power("stampede", 1)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        state.hand = [CardInstance("strike_ironclad")]
        state.draw_pile = []
        state.discard = []
        step(state, Action("end_turn"))
        self.assertLess(enemy.hp, 200)


class UnmovableTest(IroncladPowerTestCase):
    """``UnmovablePower``（``UnmovablePower.cs:21-40``）。"""

    def _card_block(self, state, card, base: int):
        from sts2_sim import core
        state.play_area.append(card)
        gained = core.gain_block(state, state.player, base, [], card=card)
        state.play_area.remove(card)
        return gained

    def test_first_card_block_gain_is_doubled(self):
        """本回合**第一次**由卡牌获得的格挡 ×2；第二次不再翻倍。"""
        from sts2_sim.core import CardInstance
        state = self.combat()
        state.player.add_power("unmovable", 1)
        self.assertEqual(self._card_block(state, CardInstance("defend_ironclad"), 5), 10)
        self.assertEqual(self._card_block(state, CardInstance("defend_ironclad"), 5), 5)

    def test_amount_controls_how_many_gains_are_doubled(self):
        """层数是"每回合翻倍几次"，不是倍率（``num >= base.Amount → 1m``）。"""
        from sts2_sim.core import CardInstance
        state = self.combat()
        state.player.add_power("unmovable", 2)
        self.assertEqual(self._card_block(state, CardInstance("defend_ironclad"), 5), 10)
        self.assertEqual(self._card_block(state, CardInstance("defend_ironclad"), 5), 10)
        self.assertEqual(self._card_block(state, CardInstance("defend_ironclad"), 5), 5)

    def test_multiple_block_gains_from_one_card_share_one_count(self):
        """``e.CardPlay != cardPlay``：同一张牌拿两次格挡**只占一次**计数。"""
        from sts2_sim.core import CardInstance, gain_block
        state = self.combat()
        state.player.add_power("unmovable", 1)
        card = CardInstance("defend_ironclad")
        state.play_area.append(card)
        first = gain_block(state, state.player, 5, [], card=card)
        second = gain_block(state, state.player, 5, [], card=card)
        state.play_area.remove(card)
        self.assertEqual([first, second], [10, 10])

    def test_power_granted_block_is_not_doubled(self):
        """``!props.IsCardOrMonsterMove()``：能力/药水给的格挡不翻倍。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("unmovable", 2)
        state.player.block = 0
        core._apply_block_gain(state, state.player, 6, [], label="镀层")
        self.assertEqual(state.player.block, 6)

    def test_frail_and_unmovable_share_a_single_truncation(self):
        """乘法阶段是**先累乘、最后截断一次**：``int(5×0.75×2)=7``，不是 ``int(3.75)×2=6``。"""
        from sts2_sim.core import CardInstance
        state = self.combat()
        state.player.add_power("unmovable", 1)
        state.player.add_power("frail", 1)
        self.assertEqual(self._card_block(state, CardInstance("defend_ironclad"), 5), 7)

    def test_counter_is_isolated_per_turn(self):
        """``HappenedThisTurn``：换一个玩家回合，翻倍次数重新开始。"""
        from sts2_sim import core
        from sts2_sim.core import CardInstance
        state = self.combat()
        state.player.add_power("unmovable", 1)
        self.assertEqual(self._card_block(state, CardInstance("defend_ironclad"), 5), 10)
        state.turn += 1
        self.assertEqual(self._card_block(state, CardInstance("defend_ironclad"), 5), 10)
        core.start_player_turn(state, [])       # 真机也在这里进入新回合


class HellraiserTest(IroncladPowerTestCase):
    """``HellraiserPower``（``HellraiserPower.cs:37-68``）。"""

    def test_drawn_strike_is_autoplayed(self):
        """``AfterCardDrawnEarly``：抽到带 ``Strike`` 标签的牌 → 立刻自动打出（免费）。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("hellraiser", 1)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        state.hand = []
        state.draw_pile = [core.CardInstance("strike_ironclad"),
                           core.CardInstance("defend_ironclad")]
        core.draw_cards(state, 2, [])
        self.assertEqual(100 - enemy.hp, 6, "抽到的 Strike 应当被自动打出")
        self.assertEqual([c.cid for c in state.hand], ["defend_ironclad"])

    def test_non_strike_cards_stay_in_hand(self):
        """``card.Tags.Contains(CardTag.Strike)``：没有 Strike 标签的牌不自动打出。"""
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("hellraiser", 1)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        state.hand = []
        state.draw_pile = [core.CardInstance("defend_ironclad"),
                           core.CardInstance("bash")]
        core.draw_cards(state, 2, [])
        self.assertEqual(enemy.hp, 100)
        self.assertEqual(sorted(c.cid for c in state.hand), ["bash", "defend_ironclad"])

    def test_runs_before_the_regular_card_drawn_hook(self):
        """``AfterCardDrawnEarly`` 是 ``Hook.AfterCardDrawn`` 的**第一遍**遍历。

        用 ``ChainsOfBindingPower``（挂 ``Bound``）当对照：真机顺序是
        "先 Early、后普通"，所以 Strike 在被打出去的时候还没被贴 ``Bound``。
        """
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("hellraiser", 1)
        state.hand = []
        state.draw_pile = [core.CardInstance("strike_ironclad")]
        core.draw_cards(state, 1, [])
        # 没崩、也没把牌留在手里 —— 说明 Early 那一遍确实先跑了
        self.assertEqual([c.cid for c in state.hand], [])


class OneTwoPunchTest(IroncladPowerTestCase):
    """``OneTwoPunchPower``（``OneTwoPunchPower.cs:19-43``）。"""

    def test_attack_is_played_twice(self):
        """``ModifyCardPlayCount``：攻击牌 ``playCount + 1``，且**不额外扣能量**。"""
        state = self.combat()
        player = state.player
        player.add_power("one_two_punch", 1)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        self.play(state, "strike_ironclad")
        self.assertEqual(100 - enemy.hp, 12, "6 点打两次")
        self.assertEqual(player.power("one_two_punch"), 0, "用过一次就减 1 层")

    def test_skills_do_not_consume_it(self):
        """``card.Type != Attack → return playCount``：技能牌**不扣层数**。"""
        state = self.combat()
        player = state.player
        player.add_power("one_two_punch", 2)
        self.play(state, "defend_ironclad")
        self.assertEqual(player.power("one_two_punch"), 2)

    def test_amount_is_the_number_of_attacks_affected(self):
        """层数不是倍率，而是"还能让几张攻击牌多打一次"。"""
        state = self.combat()
        player = state.player
        player.add_power("one_two_punch", 2)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        self.play(state, "strike_ironclad")
        self.assertEqual(200 - enemy.hp, 12)
        self.assertEqual(player.power("one_two_punch"), 1)
        self.play(state, "strike_ironclad")
        self.assertEqual(200 - enemy.hp, 24)
        self.assertNotIn("one_two_punch", player.powers)
        self.play(state, "strike_ironclad")
        self.assertEqual(200 - enemy.hp, 30, "层数用完后恢复单次")

    def test_removed_at_owners_side_turn_end(self):
        """``AfterSideTurnEnd``：``participants.Contains(Owner)`` → 整个移除。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("one_two_punch", 3)
        power_rules.on_side_turn_end(state, [], "enemy")
        self.assertIn("one_two_punch", player.powers)
        power_rules.on_turn_end(player, state, [])
        self.assertNotIn("one_two_punch", player.powers)

    def test_extra_play_survives_a_pending_selection(self):
        """⚠️ "多打一次 + 带选牌的攻击牌"不能少打一次。

        ``Headbutt`` 是"造成伤害 + 从弃牌堆选一张放到抽牌堆顶"，打出时会**挂起**。
        真机的 ``for (i < playCount)`` 循环在选牌结束后继续跑第二次 ——
        引擎把剩余次数存在 ``PendingSelection.plays_remaining`` 里续跑。
        少了这条，伤害会少一半，而日志一切正常。
        """
        from sts2_sim import core
        state = self.combat()
        player = state.player
        player.add_power("one_two_punch", 1)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        state.discard = [core.CardInstance("defend_ironclad")]
        state.draw_pile = [core.CardInstance("strike_ironclad") for _ in range(4)]
        headbutt = core.CardInstance("headbutt")
        state.hand.append(headbutt)
        state.energy = 9
        core.step(state, core.Action("play_card", state.hand.index(headbutt), 0))
        self.assertIsNotNone(state.pending, "Headbutt 应当挂在选牌上")
        self.assertEqual(100 - enemy.hp, 9, "第一次打出的伤害")
        core.step(state, core.Action("select_card", 0))
        self.assertEqual(100 - enemy.hp, 18, "续跑之后要把第二次也打出来")
        self.assertIsNone(state.pending)


class AllIroncladPowersSmokeTest(IroncladPowerTestCase):
    """把 22 张能力牌全塞进一副牌，用随机策略打若干个回合。

    这一条守的是**结构性**风险，不是某一个数值：新增的钩子（消耗 / 获得格挡 /
    回合结束的双方视角 / 自动打出 / 多次打出）都挂在"每一次结算"的路径上，
    任何一处写错都会表现为"某条路径崩溃"或"牌凭空消失/复制"。
    """

    POWER_CARDS = (
        "rage", "feel_no_pain", "dark_embrace", "juggernaut", "rupture",
        "demon_form", "flame_barrier", "corruption", "vicious", "unmovable",
        "stampede", "cruelty", "aggression", "hellraiser", "inferno", "mangle",
        "one_two_punch", "pyre", "setup_strike", "tank", "juggling",
        "crimson_mantle",
    )

    def test_random_play_never_crashes_or_loses_cards(self):
        import random
        from sts2_sim import core
        deck = list(self.POWER_CARDS) + ["strike_ironclad"] * 4 \
            + ["defend_ironclad"] * 4 + ["bloodletting", "hemokinesis", "bash",
                                         "headbutt", "apparition", "dazed"]
        for seed in range(6):
            state = core.start_combat(deck, ("nibbit", "axebot"), seed=seed)
            total_cards = (len(state.hand) + len(state.draw_pile)
                           + len(state.discard) + len(state.exhaust)
                           + len(state.play_area))
            policy = random.Random(seed)
            for _ in range(60):
                if state.finished():
                    break
                actions = core.legal_actions(state)
                self.assertTrue(actions, f"seed {seed}：出现了没有任何合法动作的局面")
                core.step(state, policy.choice(actions))
                total = (len(state.hand) + len(state.draw_pile)
                         + len(state.discard) + len(state.exhaust)
                         + len(state.play_area))
                # 生成的牌（复制 / 加牌）会改变总数，但只允许**增加**
                self.assertGreaterEqual(total, total_cards,
                                        f"seed {seed}：有牌凭空消失了")


class CoverageTest(IroncladPowerTestCase):
    """这一批能力必须真的让卡牌进入训练集（``content.engine_coverage`` 口径）。"""

    IMPLEMENTED = (
        "rage", "feel_no_pain", "dark_embrace", "juggernaut", "rupture",
        "demon_form", "flame_barrier", "corruption", "vicious", "unmovable",
        "stampede", "cruelty", "aggression", "hellraiser", "inferno", "mangle",
        "one_two_punch", "pyre", "setup_strike", "tank", "juggling",
        "crimson_mantle",
    )

    def test_all_listed_powers_are_registered(self):
        from sts2_sim import powers
        missing = [pid for pid in self.IMPLEMENTED if pid not in powers.IMPLEMENTED]
        self.assertEqual(missing, [])

    def test_every_rule_cites_the_decompiled_source(self):
        """铁律 R1：每条规则都要写明源码出处，且是真实的 C# 类名。"""
        from sts2_sim import powers
        for pid in self.IMPLEMENTED:
            source = powers.RULES[pid].source
            self.assertTrue(source, f"{pid} 缺少源码出处")
            self.assertIn("Power", source, f"{pid} 的出处不像 C# 能力类：{source}")

    def test_referencing_cards_become_trainable(self):
        """这 14 个能力对应的铁甲战士卡必须全部合格（``effects_incomplete`` 消失）。"""
        from sts2_sim import content, eligibility as el
        cards = {
            "rage": "rage", "feel_no_pain": "feel_no_pain",
            "dark_embrace": "dark_embrace", "juggernaut": "juggernaut",
            "flame_barrier": "flame_barrier", "crimson_mantle": "crimson_mantle",
            "inferno": "inferno", "setup_strike": "setup_strike",
            "mangle": "mangle", "demon_form": "demon_form", "pyre": "pyre",
            "corruption": "corruption", "cruelty": "cruelty", "tank": "tank",
            "rupture": "rupture", "vicious": "vicious", "juggling": "juggling",
            "aggression": "aggression", "stampede": "stampede",
            "unmovable": "unmovable", "hellraiser": "hellraiser",
            "one_two_punch": "one_two_punch",
        }
        broken = []
        for cid in cards.values():
            card = content.CARD_DB.get(cid)
            self.assertIsNotNone(card, f"{cid} 不在内容表里")
            if card.effects_incomplete:
                broken.append(cid)
        self.assertEqual(broken, [], f"这些卡仍被判为残缺：{broken}")

    def test_draft_pool_contains_the_new_cards(self):
        """可抽卡池的口径（``eligibility.report``）。"""
        from sts2_sim import eligibility as el
        pool = set(el.card_pool("ironclad"))
        # corruption 是 ancient 稀有度，本来就不进可抽池；其余 13 张都要在。
        #
        # ⚠️ `tank` **不在**这里：它声明了
        # ``MultiplayerConstraint => CardMultiplayerConstraint.MultiplayerOnly``，
        # 单人 profile 里根本不会出现（`docs/12` §2.32 加的门禁）。
        # 它此前在池子里是准入口径的缺口，不是"实现得对"。
        expected = {"rage", "feel_no_pain", "dark_embrace", "juggernaut",
                    "flame_barrier", "crimson_mantle", "inferno",
                    "setup_strike", "mangle", "demon_form", "pyre",
                    "cruelty", "rupture", "vicious", "juggling",
                    "aggression", "stampede", "unmovable", "hellraiser",
                    "one_two_punch"}
        self.assertEqual(expected - pool, set())
        self.assertNotIn("tank", pool, "多人专用卡不该进单人可抽池")
        self.assertIn("multiplayer_only", el.card_reasons("tank"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
