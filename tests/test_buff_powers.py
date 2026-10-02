"""**Wave A 能力**的源码级回归测试（``docs/09`` L2 / L4）。

与 ``tests/test_ironclad_powers.py`` 同一套路：每个能力一条 ``PowerRules``，
``source=`` 写明反编译出处，测试断言的是源码里**最容易漏掉的那一条条件**。

分节与交付批次一一对应：

* 批次 1 —— 纯标记 + 已经在总线上声明过的钩子（``AfterEnergyReset`` /
  ``AfterSideTurnStart``）
* 批次 2 —— ``AfterCardDrawn`` / ``AfterDamageGiven``
* 批次 3 —— ``BeforeCardPlayed`` / ``AfterCardPlayed``
* 批次 4 —— 新钩子 ``BeforeSideTurnEnd``
* 批次 5 —— 充能球（``OrbCmd``）

⚠️ 测试只用 ``core`` 的**公开流程**（``step`` / 钩子入口）驱动，不直接改数值 ——
绕过流程就等于绕过被验证的那条路径。
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


class BuffPowerTestCase(unittest.TestCase):
    """公共脚手架：一副起手牌 + 一只怪，以及"把某张牌塞进手里再打出去"。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, enemies=("nibbit",), seed: int = 5):
        from sts2_sim.core import start_combat
        return start_combat(self.deck, tuple(enemies), seed=seed)

    def play(self, state, cid: str, target: int = 0):
        from sts2_sim.core import Action, CardInstance, step
        card = CardInstance(cid)
        state.hand.append(card)
        state.energy = 9
        index = state.hand.index(card)
        return step(state, Action("play_card", index, target))


# ==========================================================================
# 批次 1
# ==========================================================================
class AccelerantTest(BuffPowerTestCase):
    """``AccelerantPower``（``AccelerantPower.cs:7-17``）—— 纯标记，无自身行为。"""

    def test_poison_triggers_extra_times(self):
        """``PoisonPower`` 读 ``accelerant`` 来多触发：``min(层数, 1 + accelerant)``。

        源码注释：``the Poison power checks for this power and re-triggers itself``。
        3 层中毒 + 1 层催速，触发 2 次 → 3 + 2 = 5 点伤害，而不是 3。
        """
        from sts2_sim import powers as power_rules
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("poison", 3)
        state.player.add_power("accelerant", 1)
        hp = enemy.hp
        power_rules.on_turn_start(enemy, state, [])
        self.assertEqual(hp - enemy.hp, 5, "2 次触发：3 + 2")
        self.assertEqual(enemy.power("poison"), 1, "掉完 2 层还剩 1 层")

    def test_without_accelerant_poison_triggers_once(self):
        """"反向验证"：没有催速时只触发 1 次（证明上一条断言的鉴别力）。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("poison", 3)
        hp = enemy.hp
        power_rules.on_turn_start(enemy, state, [])
        self.assertEqual(hp - enemy.hp, 3)


class GenesisTest(BuffPowerTestCase):
    """``GenesisPower.AfterEnergyReset``（``GenesisPower.cs:14-21``）。"""

    def test_energy_reset_grants_stars(self):
        """能量重置后 +``Amount`` 星（``PlayerCmd.GainStars``）。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("genesis", 2)
        state.stars = 0
        power_rules.on_energy_reset(state.player, state, [])
        self.assertEqual(state.stars, 2)

    def test_stars_are_not_the_power_amount_scaled(self):
        """加的是**层数本身**（``base.Amount``），3 层就是 3 颗。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("genesis", 3)
        state.stars = 1
        power_rules.on_energy_reset(state.player, state, [])
        self.assertEqual(state.stars, 4)


class RadianceTest(BuffPowerTestCase):
    """``RadiancePower.AfterEnergyReset``（``RadiancePower.cs:21-27``）。"""

    def test_gains_one_energy_per_turn_then_decrements(self):
        """能量 +``EnergyVar(1)``（**恒为 1**，不是层数），然后递减 1 层。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("radiance", 3)
        state.energy = 0
        power_rules.on_energy_reset(state.player, state, [])
        self.assertEqual(state.energy, 1, "3 层光辉也只给 1 点能量")
        self.assertEqual(state.player.power("radiance"), 2)

    def test_last_layer_is_removed(self):
        """递减到 0 就整个移除（``Counter`` + 不允许负值）。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("radiance", 1)
        power_rules.on_energy_reset(state.player, state, [])
        self.assertEqual(state.player.power("radiance"), 0)


class LightningRodTest(BuffPowerTestCase):
    """``LightningRodPower.AfterEnergyReset``（``LightningRodPower.cs:31-37``）。"""

    def test_channels_exactly_one_lightning_then_decrements(self):
        """引导 **1** 个闪电球（不是 ``Amount`` 个），然后递减 1 层。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("lightning_rod", 3)
        state.orbs = []
        power_rules.on_energy_reset(state.player, state, [])
        self.assertEqual([o.oid for o in state.orbs], ["lightning"])
        self.assertEqual(state.player.power("lightning_rod"), 2)


class SpinnerTest(BuffPowerTestCase):
    """``SpinnerPower.AfterEnergyReset``（``SpinnerPower.cs:24-32``）。"""

    def test_channels_amount_glass_orbs(self):
        """引导 ``Amount`` 个玻璃球（与 LightningRod 的"恒 1 个"相反）。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("spinner", 2)
        state.orbs = []
        state.orb_slots = 5
        power_rules.on_energy_reset(state.player, state, [])
        self.assertEqual([o.oid for o in state.orbs], ["glass", "glass"])


class CoolantTest(BuffPowerTestCase):
    """``CoolantPower.AfterSideTurnStart``（``CoolantPower.cs:21-29``）。"""

    def test_block_counts_orb_kinds_not_orb_count(self):
        """``group orb by orb.Id`` 数的是**种类**数：3 个闪电球只算 1 种。

        漏掉 ``group by`` 会变成 3 倍格挡 —— 这是这条能力最容易写错的地方。
        """
        from sts2_sim import orbs as orb_rules
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("coolant", 3)
        state.player.block = 0
        state.orbs = []
        state.orb_slots = 5
        for _ in range(3):
            orb_rules.channel(state, "lightning", [])
        power_rules.on_turn_start(state.player, state, [])
        self.assertEqual(state.player.block, 3, "1 种球 × 3 层 = 3 点格挡")

    def test_two_kinds_multiply(self):
        """两种球 → ``2 × Amount``。"""
        from sts2_sim import orbs as orb_rules
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("coolant", 3)
        state.player.block = 0
        state.orbs = []
        state.orb_slots = 5
        orb_rules.channel(state, "lightning", [])
        orb_rules.channel(state, "frost", [])
        power_rules.on_turn_start(state.player, state, [])
        self.assertEqual(state.player.block, 6)

    def test_no_orbs_gives_no_block(self):
        """没有球 → 0 点格挡（不能退化成"至少 1 点"）。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("coolant", 3)
        state.player.block = 0
        state.orbs = []
        power_rules.on_turn_start(state.player, state, [])
        self.assertEqual(state.player.block, 0)


class CountdownTest(BuffPowerTestCase):
    """``CountdownPower.AfterSideTurnStart``（``CountdownPower.cs:21-31``）。"""

    def test_applies_doom_to_a_random_hittable_enemy(self):
        """给随机一个可打敌人上 ``Amount`` 层末日，走 ``Rng.CombatTargets``。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("countdown", 4)
        power_rules.on_turn_start(state.player, state, [])
        doomed = [e for e in state.enemies if e.power("doom") > 0]
        self.assertEqual(len(doomed), 1, "只该有一个敌人吃到末日")
        self.assertEqual(doomed[0].power("doom"), 4)

    def test_no_enemies_is_safe(self):
        """没有可打敌人时不报错、也不上末日。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("countdown", 4)
        state.enemies = []
        power_rules.on_turn_start(state.player, state, [])


class NoxiousFumesTest(BuffPowerTestCase):
    """``NoxiousFumesPower.AfterSideTurnStart``（``NoxiousFumesPower.cs:26-43``）。"""

    def test_applies_poison_to_all_hittable_enemies(self):
        """``HittableEnemies`` 是**集合**：两只怪都要上毒。"""
        from sts2_sim import powers as power_rules
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("noxious_fumes", 2)
        power_rules.on_turn_start(state.player, state, [])
        self.assertEqual([e.power("poison") for e in state.enemies], [2, 2])

    def test_skips_enemies_that_cannot_be_hit(self):
        """``HittableEnemies => Enemies.Where(IsHittable)``：**复活中**的敌人打不到，
        因此不该被上毒（引擎里 ``living_enemies()`` 会把它们算活着，两者不等价）。"""
        from sts2_sim import powers as power_rules
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("noxious_fumes", 2)
        state.enemies[1].reviving = True
        power_rules.on_turn_start(state.player, state, [])
        self.assertEqual(state.enemies[0].power("poison"), 2)
        self.assertEqual(state.enemies[1].power("poison"), 0, "复活中的敌人不算可打目标")


class PrepTimeTest(BuffPowerTestCase):
    """``PrepTimePower.AfterSideTurnStart``（``PrepTimePower.cs:18-23``）。"""

    def test_grants_vigor(self):
        """拥有者所在阵营回合开始 → ``PowerCmd.Apply<VigorPower>(Amount)``。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("prep_time", 2)
        power_rules.on_turn_start(state.player, state, [])
        self.assertEqual(state.player.power("vigor"), 2)

    def test_artifact_does_not_block_vigor(self):
        """活力是 Buff —— 神器**不**抵消它（``TryModifyPowerAmountReceived`` 只看 Debuff）。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("prep_time", 2)
        state.player.add_power("artifact", 1)
        power_rules.on_turn_start(state.player, state, [])
        self.assertEqual(state.player.power("vigor"), 2)
        self.assertEqual(state.player.power("artifact"), 1)


class FriendshipTest(BuffPowerTestCase):
    """``FriendshipPower.ModifyMaxEnergy``（``FriendshipPower.cs:12-19``）。"""

    def test_raises_max_energy_by_amount(self):
        """``return amount + (decimal)base.Amount``：每层 +1 能量上限。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        base = state.max_energy
        state.player.add_power("friendship", 2)
        self.assertEqual(power_rules.max_energy_bonus(state), 2)
        state.start_player_turn() if hasattr(state, "start_player_turn") else None
        self.assertGreaterEqual(base, 3)

    def test_pyre_still_counted(self):
        """名单是**求和**，不是替换 —— 加了新能力之后 ``pyre`` 不能被漏掉。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("pyre", 1)
        state.player.add_power("friendship", 3)
        self.assertEqual(power_rules.max_energy_bonus(state), 4)


class DanseMacabreTest(BuffPowerTestCase):
    """``DanseMacabrePower.BeforeCardPlayed``（``DanseMacabrePower.cs:14-24``）。"""

    def test_gives_block_for_a_card_costing_two(self):
        """费用 ≥ ``EnergyVar(2)`` → ``GainBlock(Amount, Unpowered)``。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("danse_macabre", 3)
        # `inflame` 费 1 → 不触发；`bash` 费 2 → 触发
        self.play(state, "inflame")
        self.assertEqual(state.player.block, 0, "1 费牌不该触发")
        self.play(state, "bash")
        self.assertEqual(state.player.block, 3)

    def test_cost_is_resolved_not_printed(self):
        """比的是**修正后**费用：腐败把技能牌清零后，2 费技能牌也不再触发。

        用卡面费用会在腐败下**多给**格挡 —— 这是这类能力最容易漏的一条。
        用 ``shockwave``（2 费技能，自身不产生任何格挡）当探针。
        """
        state = self.combat()
        state.player.add_power("danse_macabre", 2)
        self.play(state, "shockwave")             # 对照组：2 费 → 触发
        self.assertEqual(state.player.block, 2)
        state.player.block = 0
        state.player.add_power("corruption", 1)   # 技能牌费用归零
        self.play(state, "shockwave")
        self.assertEqual(state.player.block, 0, "腐败把费用清零后不该再给格挡")


class HailstormTest(BuffPowerTestCase):
    """``HailstormPower.BeforeSideTurnEnd``（``HailstormPower.cs:20-33``）。"""

    def _fire(self, state, side="player"):
        from sts2_sim import powers as power_rules
        power_rules.on_before_side_turn_end(state, [], side)

    def test_frost_orb_makes_it_rain_on_all_enemies(self):
        """闸门是"场上**至少 1 个冰球**"（常量 ``FrostOrbs = 1``），伤害是 Unpowered。"""
        from sts2_sim import orbs
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("hailstorm", 5)
        orbs.channel(state, "frost", [])
        before = [e.hp for e in state.enemies]
        self._fire(state)
        self.assertEqual([e.hp for e in state.enemies], [hp - 5 for hp in before])

    def test_other_orb_kinds_do_not_trigger(self):
        """``o is FrostOrb`` 只看冰球 —— 闪电球再多也不下冰雹。"""
        from sts2_sim import orbs
        state = self.combat()
        state.player.add_power("hailstorm", 5)
        orbs.channel(state, "lightning", [])
        before = state.enemies[0].hp
        self._fire(state)
        self.assertEqual(state.enemies[0].hp, before)

    def test_enemy_side_turn_end_does_not_trigger(self):
        """``participants.Contains(base.Owner)``：只有自家阵营结束才结算。"""
        from sts2_sim import orbs
        state = self.combat()
        state.player.add_power("hailstorm", 5)
        orbs.channel(state, "frost", [])
        before = state.enemies[0].hp
        self._fire(state, side="enemy")
        self.assertEqual(state.enemies[0].hp, before)


class SneakyTest(BuffPowerTestCase):
    """``SneakyPower.AfterCardPlayed``（``SneakyPower.cs:13-22``）。"""

    def test_enemy_gains_block_when_the_player_attacks(self):
        """条件是 ``card.Owner != owner``：这是**敌人**的能力，玩家打攻击牌它涨格挡。"""
        state = self.combat()
        state.enemies[0].add_power("sneaky", 3)
        self.play(state, "strike_ironclad")
        self.assertEqual(state.enemies[0].block, 3)

    def test_skill_does_not_trigger(self):
        state = self.combat()
        state.enemies[0].add_power("sneaky", 3)
        self.play(state, "defend_ironclad")
        self.assertEqual(state.enemies[0].block, 0)

    def test_on_the_player_it_never_triggers(self):
        """反向验证：挂在玩家身上时 ``card.Owner == owner``，永远不触发。"""
        state = self.combat()
        state.player.add_power("sneaky", 3)
        self.play(state, "strike_ironclad")
        self.assertEqual(state.player.block, 0)


class HauntTest(BuffPowerTestCase):
    """``HauntPower.AfterCardPlayed``（``HauntPower.cs:9-21``）。"""

    def test_soul_deals_unblockable_damage(self):
        """``Soul`` 才触发；伤害同时是 ``Unblockable``（不吃格挡）与 ``Unpowered``。"""
        state = self.combat()
        state.player.add_power("haunt", 4)
        enemy = state.enemies[0]
        enemy.block = 99
        before = enemy.hp
        self.play(state, "soul")
        self.assertEqual(enemy.hp, before - 4, "不可格挡伤害必须穿格挡")
        self.assertEqual(enemy.block, 99, "格挡不该被消耗")

    def test_other_cards_do_not_trigger(self):
        state = self.combat()
        state.player.add_power("haunt", 4)
        before = state.enemies[0].hp
        self.play(state, "strike_ironclad")
        self.assertEqual(state.enemies[0].hp, before - 6)   # 只有打击本身的 6 点


class ReaperFormTest(BuffPowerTestCase):
    """``ReaperFormPower.AfterDamageGiven``（``ReaperFormPower.cs:30-39``）。"""

    def test_doom_is_total_damage_times_amount(self):
        """用的是 ``result.TotalDamage``（**含被格挡的部分**）× 层数。

        这里刻意给敌人 5 点格挡：打击 6 点 → 掉血 1、总伤害 6 →
        末日应当是 ``6 × 2 = 12``。用掉血量算会只给 2 层 —— 差距一眼可见。
        """
        state = self.combat()
        state.player.add_power("reaper_form", 2)
        enemy = state.enemies[0]
        enemy.block = 5
        self.play(state, "strike_ironclad")
        self.assertEqual(enemy.power("doom"), 12)

    def test_enemy_damage_does_not_trigger(self):
        """``dealer == base.Owner``：玩家身上挂的收割形态只认**自己**造成的伤害。"""
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("reaper_form", 2)
        self.play(state, "strike_ironclad")
        self.assertEqual(state.player.power("doom"), 0)


class AfterimageTest(BuffPowerTestCase):
    """``AfterimagePower``（``AfterimagePower.cs:26-48``）—— 快照语义。"""

    def test_gives_block_per_card_played(self):
        state = self.combat()
        state.player.add_power("afterimage", 3)
        self.play(state, "strike_ironclad")
        self.assertEqual(state.player.block, 3)

    def test_own_play_uses_the_snapshot_not_the_new_amount(self):
        """⭐ 打出残影本体时用**结算开始时**的层数（2），而不是加完之后的 3。

        源码注释专门写了这一条：``amountsForPlayedCards`` 就是为了
        "avoid gaining extra block on multiple plays of After Image"。
        """
        state = self.combat()
        state.player.add_power("afterimage", 2)
        self.play(state, "afterimage")            # 打出后层数变成 3
        self.assertEqual(state.player.power("afterimage"), 3)
        self.assertEqual(state.player.block, 2, "必须按快照值给格挡")

    def test_cards_played_before_it_existed_do_not_trigger(self):
        """反向验证：先打一张牌（那时没有残影），再把残影挂上 —— 不该补发格挡。"""
        state = self.combat()
        self.play(state, "strike_ironclad")
        self.assertEqual(state.player.block, 0)
        state.player.add_power("afterimage", 3)
        self.assertEqual(state.player.block, 0)


class SerpentFormTest(BuffPowerTestCase):
    """``SerpentFormPower``（``SerpentFormPower.cs:32-52``）—— 同一套快照语义。"""

    def test_deals_snapshot_damage_to_an_enemy_per_card(self):
        state = self.combat()
        state.player.add_power("serpent_form", 4)
        enemy = state.enemies[0]
        before = enemy.hp
        self.play(state, "strike_ironclad")
        # 打击 6 点 + 蛇形 4 点 = 10
        self.assertEqual(enemy.hp, before - 10)

    def test_its_own_play_does_not_deal_damage(self):
        """打出蛇形本体时快照是 0（挂上之前根本没有这个能力）→ 不造成伤害。"""
        state = self.combat()
        enemy = state.enemies[0]
        before = enemy.hp
        self.play(state, "serpent_form")
        self.assertEqual(enemy.hp, before, "本体那一次不该造成伤害")


class ReflectTest(BuffPowerTestCase):
    """``ReflectPower``（``ReflectPower.cs:11-30``）。"""

    def _enemy_hits_player(self, state, amount: int):
        from sts2_sim import core
        return core.deal_damage(state, state.enemies[0], state.player, amount, [])

    def test_reflects_the_blocked_damage(self):
        """⭐ 反弹的是 ``BlockedDamage``：挡下 4 点就反弹 4 点（不是掉血量 0）。"""
        state = self.combat()
        state.player.add_power("reflect", 3)
        state.player.block = 10
        enemy = state.enemies[0]
        before = enemy.hp
        self._enemy_hits_player(state, 4)
        self.assertEqual(state.player.hp, state.player.max_hp, "应当全被格挡")
        self.assertEqual(enemy.hp, before - 4, "被格挡的 4 点要弹回去")

    def test_unblocked_damage_does_not_reflect(self):
        """没有被格挡的量（``BlockedDamage == 0``）→ 不反弹。"""
        state = self.combat()
        state.player.add_power("reflect", 3)
        enemy = state.enemies[0]
        before = enemy.hp
        self._enemy_hits_player(state, 4)
        self.assertEqual(enemy.hp, before)

    def test_decrements_on_own_turn_start(self):
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("reflect", 1)
        power_rules.on_turn_start(state.player, state, [])
        self.assertEqual(state.player.power("reflect"), 0, "1 层应当整个移除")


class BurstTest(BuffPowerTestCase):
    """``BurstPower``（``BurstPower.cs:16-40``）—— 技能牌多打一次。"""

    def test_skill_card_is_played_twice(self):
        state = self.combat()
        state.player.add_power("burst", 1)
        self.play(state, "defend_ironclad")          # 5 格挡 × 2 次
        self.assertEqual(state.player.block, 10)

    def test_attack_does_not_consume_a_layer(self):
        """``card.Type != Skill`` 就不改次数 → ``AfterModifyingCardPlayCount`` 不通知它。

        少了这条"只通知改过次数的模型"，打一张攻击牌就把爆发白耗一层。
        """
        state = self.combat()
        state.player.add_power("burst", 1)
        before = state.enemies[0].hp
        self.play(state, "strike_ironclad")
        self.assertEqual(state.enemies[0].hp, before - 6)
        self.assertEqual(state.player.power("burst"), 1, "没加倍就不该扣层")

    def test_expires_at_side_turn_end(self):
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("burst", 2)
        power_rules.on_side_turn_end(state, [], "player")
        self.assertEqual(state.player.power("burst"), 0, "回合结束整个移除（不递减）")


class SignalBoostTest(BuffPowerTestCase):
    """``SignalBoostPower``（``SignalBoostPower.cs:13-30``）—— 能力牌多打一次。"""

    def test_power_card_is_played_twice(self):
        state = self.combat()
        state.player.add_power("signal_boost", 1)
        self.play(state, "inflame")                  # +2 力量，打两次 → +4
        self.assertEqual(state.player.power("strength"), 4)

    def test_skill_does_not_consume_a_layer(self):
        state = self.combat()
        state.player.add_power("signal_boost", 1)
        self.play(state, "defend_ironclad")
        self.assertEqual(state.player.power("signal_boost"), 1)


class DuplicationTest(BuffPowerTestCase):
    """``DuplicationPower``（``DuplicationPower.cs:11-32``）—— 任意牌多打一次。"""

    def test_any_card_is_played_twice(self):
        state = self.combat()
        state.player.add_power("duplication", 1)
        self.play(state, "defend_ironclad")
        self.assertEqual(state.player.block, 10)

    def test_one_card_per_layer(self):
        state = self.combat()
        state.player.add_power("duplication", 1)
        self.play(state, "defend_ironclad")          # 10
        self.play(state, "defend_ironclad")          # 只算一次 → +5
        self.assertEqual(state.player.block, 15)


class EchoFormTest(BuffPowerTestCase):
    """``EchoFormPower.ModifyCardPlayCount``（``EchoFormPower.cs:55-72``）。"""

    def test_first_amount_cards_are_doubled(self):
        """⭐ 闸门是"本回合**已开始**的牌数 < 层数"。

        层数 2 → 第 1、2 张各打两次（10+10），第 3 张只打一次（+5）= 25。
        如果计数把**当前这张牌**也算进去，就会变成 10+5+5=20 —— 差一张牌的量。
        """
        state = self.combat()
        state.player.add_power("echo_form", 2)
        for _ in range(3):
            self.play(state, "defend_ironclad")
        self.assertEqual(state.player.block, 25)

    def test_is_not_consumed_by_playing(self):
        """形态不递减（与 Burst/Duplication 不同）。"""
        state = self.combat()
        state.player.add_power("echo_form", 2)
        self.play(state, "defend_ironclad")
        self.assertEqual(state.player.power("echo_form"), 2)


class LoopTest(BuffPowerTestCase):
    """``LoopPower.AfterPlayerTurnStart``（``LoopPower.cs:11-21``）。"""

    def _fire(self, state):
        from sts2_sim import powers as power_rules
        power_rules.on_after_player_turn_start(state, [])

    def test_triggers_the_leftmost_orb_amount_times(self):
        from sts2_sim import orbs
        state = self.combat()
        state.player.add_power("loop", 2)
        orbs.channel(state, "lightning", [])          # 被动 3 点随机伤害
        before = state.enemies[0].hp
        self._fire(state)
        self.assertEqual(state.enemies[0].hp, before - 6, "2 层 = 触发 2 次被动")

    def test_only_the_leftmost_orb_triggers(self):
        """⭐ 触发的是 ``Orbs[0]`` —— 第二个球的被动**不该**被带上。"""
        from sts2_sim import orbs
        state = self.combat()
        state.player.add_power("loop", 1)
        orbs.channel(state, "frost", [])              # 左边：冰球（给格挡）
        orbs.channel(state, "lightning", [])          # 右边：闪电球（给伤害）
        before = state.enemies[0].hp
        self._fire(state)
        self.assertEqual(state.enemies[0].hp, before, "闪电球在右边，不该被触发")
        self.assertGreater(state.player.block, 0, "左边那个冰球的被动应当触发")

    def test_no_orbs_does_nothing(self):
        state = self.combat()
        state.player.add_power("loop", 3)
        before = state.enemies[0].hp
        self._fire(state)
        self.assertEqual(state.enemies[0].hp, before)


class HibernateTest(BuffPowerTestCase):
    """``HibernatePower.AfterPlayerTurnStart``（``HibernatePower.cs:17-24``）。"""

    def test_decrements_each_player_turn(self):
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("hibernate", 3)
        power_rules.on_after_player_turn_start(state, [])
        self.assertEqual(state.player.power("hibernate"), 2)

    def test_last_layer_is_removed(self):
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("hibernate", 1)
        power_rules.on_after_player_turn_start(state, [])
        self.assertEqual(state.player.power("hibernate"), 0)


class DrawTriggeredPowerTest(BuffPowerTestCase):
    """抽牌触发的两个能力：``CorrosiveWavePower`` 与 ``SpeedsterPower``。

    这一对的**全部区别**就是 ``fromHandDraw``：

    * ``CorrosiveWavePower`` 不看它 —— 回合开始抽的 5 张牌也上毒；
    * ``SpeedsterPower`` 要求 ``!fromHandDraw`` —— 只有效果抽牌才打伤害。
    """

    def _draw(self, state, count=1, from_hand=False):
        from sts2_sim.core import draw_cards
        draw_cards(state, count, [], from_hand_draw=from_hand)

    def test_corrosive_wave_poisons_on_hand_draw(self):
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("corrosive_wave", 2)
        self._draw(state, 1, from_hand=True)
        self.assertEqual([e.power("poison") for e in state.enemies], [2, 2])

    def test_corrosive_wave_poisons_on_effect_draw(self):
        state = self.combat()
        state.player.add_power("corrosive_wave", 3)
        self._draw(state, 1, from_hand=False)
        self.assertEqual(state.enemies[0].power("poison"), 3)

    def test_speedster_ignores_hand_draw(self):
        """⭐ 反向验证：手牌抽取**不**触发（这正是这条能力与上一条的区别）。"""
        state = self.combat()
        state.player.add_power("speedster", 4)
        before = state.enemies[0].hp
        self._draw(state, 1, from_hand=True)
        self.assertEqual(state.enemies[0].hp, before, "手牌抽取不该触发疾行者")

    def test_speedster_hits_on_effect_draw(self):
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("speedster", 4)
        before = [e.hp for e in state.enemies]
        self._draw(state, 1, from_hand=False)
        self.assertEqual([e.hp for e in state.enemies], [hp - 4 for hp in before])


class MonologueTest(BuffPowerTestCase):
    """``MonologuePower``（``MonologuePower.cs:38-65``）—— 快照 + 回合末撤回。"""

    def test_gains_strength_per_card_and_retracts_at_turn_end(self):
        """每张牌 +1 力量（``DynamicVars.Strength``），回合结束**撤回累计值**。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("monologue", 1)
        self.play(state, "defend_ironclad")
        self.play(state, "defend_ironclad")
        self.assertEqual(state.player.power("strength"), 2, "两张牌 = +2 力量")
        power_rules.on_side_turn_end(state, [], "player")
        self.assertEqual(state.player.power("strength"), 0, "撤回累计的 2 点")
        self.assertEqual(state.player.power("monologue"), 0, "同时移除自己")

    def test_snapshot_is_one_not_the_power_amount(self):
        """⚠️ 给的是 ``DynamicVars.Strength``（写死的 1），**不是**层数。

        3 层独白打一张牌仍然只 +1 —— 用层数会得到 +3。
        """
        state = self.combat()
        state.player.add_power("monologue", 3)
        self.play(state, "defend_ironclad")
        self.assertEqual(state.player.power("strength"), 1)


class PanacheTest(BuffPowerTestCase):
    """``PanachePower.AfterCardPlayed``（``PanachePower.cs:29-50``）。"""

    def test_fires_every_five_cards_after_itself(self):
        """⭐ ``alreadyApplied`` 让**潘趣本体那一次**不计数：打出后还要 5 张才触发。"""
        state = self.combat()
        self.play(state, "panache")                  # 本体：只翻标志
        for _ in range(5):
            self.play(state, "defend_ironclad")
        self.assertEqual(state.enemies[0].hp, state.enemies[0].max_hp - 10,
                         "应当造成 PanacheDamage=10 点全体伤害")

    def test_four_cards_are_not_enough(self):
        state = self.combat()
        self.play(state, "panache")
        for _ in range(4):
            self.play(state, "defend_ironclad")
        self.assertEqual(state.enemies[0].hp, state.enemies[0].max_hp)


class PaleBlueDotTest(BuffPowerTestCase):
    """``PaleBlueDotPower``（``PaleBlueDotPower.cs:32-60``）。"""

    def test_five_cards_grant_next_turn_draw(self):
        state = self.combat()
        state.player.add_power("pale_blue_dot", 1)
        for _ in range(5):
            self.play(state, "defend_ironclad")
        self.assertEqual(state.player.power("draw_cards_next_turn"), 1)

    def test_only_once_per_turn(self):
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("pale_blue_dot", 1)
        for _ in range(7):
            self.play(state, "defend_ironclad")
        self.assertEqual(state.player.power("draw_cards_next_turn"), 1, "每回合只触发一次")
        power_rules.on_side_turn_end(state, [], "player")     # 复位
        state.player.power_flags  # 只是取一下，确认字段还在
        self.assertFalse(state.player.power_flags.get("pale_blue_dot_used"),
                         "回合结束应当复位，下一回合还能再触发")


class StormTest(BuffPowerTestCase):
    """``StormPower``（``StormPower.cs:34-52``）—— 能力牌快照 → 引导闪电球。"""

    def test_power_card_channels_lightning_per_layer(self):
        state = self.combat()
        state.player.add_power("storm", 2)
        self.play(state, "inflame")                  # 能力牌 → 引导 2 个闪电球
        self.assertEqual([orb.oid for orb in state.orbs], ["lightning", "lightning"])

    def test_skill_card_does_not_channel(self):
        """``BeforeCardPlayed`` 里判了 ``CardType.Power`` —— 技能牌不进快照。"""
        state = self.combat()
        state.player.add_power("storm", 2)
        self.play(state, "defend_ironclad")
        self.assertEqual(list(state.orbs), [])

    def test_its_own_play_uses_the_snapshot(self):
        """打出风暴本体那一次：快照是 0（挂上之前没有这个能力）→ 不引导。"""
        state = self.combat()
        self.play(state, "storm")
        self.assertEqual(state.player.power("storm"), 1, "风暴本体给 1 层")
        self.assertEqual(list(state.orbs), [], "本体那一次不该引导")


class SpiritOfAshTest(BuffPowerTestCase):
    """``SpiritOfAshPower.BeforeCardPlayed``（``SpiritOfAshPower.cs:14-21``）。"""

    def _before_play(self, state, cid):
        from sts2_sim import powers as power_rules
        from sts2_sim.core import CardInstance
        power_rules.on_before_card_played(state, [], CardInstance(cid))

    def test_ethereal_card_gives_block(self):
        state = self.combat()
        state.player.add_power("spirit_of_ash", 3)
        self._before_play(state, "apparition")       # 带 Ethereal
        self.assertEqual(state.player.block, 3)

    def test_non_ethereal_card_gives_nothing(self):
        state = self.combat()
        state.player.add_power("spirit_of_ash", 3)
        self._before_play(state, "strike_ironclad")
        self.assertEqual(state.player.block, 0)


class TheSealedThroneTest(BuffPowerTestCase):
    """``TheSealedThronePower.BeforeCardPlayed``（``TheSealedThronePower.cs:10-17``）。"""

    def test_any_card_grants_stars(self):
        state = self.combat()
        state.player.add_power("the_sealed_throne", 2)
        self.play(state, "strike_ironclad")
        self.assertEqual(state.stars, 2)

    def test_stars_accumulate_across_cards(self):
        state = self.combat()
        state.player.add_power("the_sealed_throne", 1)
        self.play(state, "defend_ironclad")
        self.play(state, "defend_ironclad")
        self.assertEqual(state.stars, 2)


class HandDrawBonusTest(BuffPowerTestCase):
    """``ModifyHandDraw`` 一族（``ClarityPower.cs`` / ``DemesnePower.cs`` 等）。"""

    def _draw_turn(self, state, powers):
        """把开局手牌**塞回抽牌堆**再开一个新回合，返回这一回合抽到的张数。

        ⚠️ 不能只 `hand.clear()`：那样等于把那 5 张牌从模拟里抹掉，
        抽牌堆只剩 5 张 → 想抽 6 张也只抽得到 5，测出来的差值全是假的。
        另外手牌上限是 10，留着开局手牌会让多抽的部分被**截断**。
        """
        from sts2_sim import core
        state.draw_pile.extend(state.hand)
        state.hand.clear()
        for pid, amount in powers:
            state.player.add_power(pid, amount)
        core.start_player_turn(state, [])
        return len(state.hand)

    def test_clarity_adds_exactly_one(self):
        """⭐ ``ClarityPower`` 加的是写死的 **1**，不是层数：3 层也只 +1。"""
        state = self.combat()
        self.assertEqual(self._draw_turn(state, [("clarity", 3)]), 6)

    def test_amount_based_power_adds_per_layer(self):
        state = self.combat()
        self.assertEqual(self._draw_turn(state, [("demesne", 2)]), 7)

    def test_bonuses_stack(self):
        state = self.combat()
        hand = self._draw_turn(state, [("clarity", 1), ("machine_learning", 1)])
        self.assertEqual(hand, 7, "5 + 1（清晰）+ 1（机器学习）")

    def test_bonus_function_is_table_driven(self):
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("tools_of_the_trade", 2)
        state.player.add_power("tyranny", 1)
        self.assertEqual(power_rules.hand_draw_bonus(state), 3)


class BeforeHandDrawTest(BuffPowerTestCase):
    """``BeforeHandDraw`` 一族：抽牌**之前**把生成的牌塞进手牌。

    判据都落在"生成的是什么牌"上 —— 池子选错、过滤条件漏掉，
    生成的牌**照样进手牌**，只有内容不对，不会报错。
    """

    def _new_turn(self, state, powers=()):
        from sts2_sim import core
        state.draw_pile.extend(state.hand)
        state.hand.clear()
        for pid, amount in powers:
            state.player.add_power(pid, amount)
        core.start_player_turn(state, [])
        return list(state.hand)

    def test_infinite_blades_adds_shivs(self):
        state = self.combat()
        hand = self._new_turn(state, [("infinite_blades", 2)])
        self.assertEqual([c.cid for c in hand].count("shiv"), 2)

    def test_sentry_mode_adds_sweeping_gaze(self):
        state = self.combat()
        hand = self._new_turn(state, [("sentry_mode", 1)])
        self.assertEqual([c.cid for c in hand].count("sweeping_gaze"), 1)

    def test_call_of_the_void_excludes_basic_and_ancient(self):
        """⭐ 过滤的是**稀有度**：不能塞进打击/防御（basic）或远古牌。"""
        from sts2_sim import content
        state = self.combat()
        before = {id(card) for card in state.hand}
        hand = self._new_turn(state, [("call_of_the_void", 3)])
        added = [card for card in hand if id(card) not in before]
        self.assertEqual(len(added), 3, "应当生成 3 张")
        for card in added:
            definition = content.CARD_DB[card.cid]
            self.assertNotIn(definition.rarity, ("basic", "ancient"),
                             f"{card.cid} 不该被虚空召唤生成")

    def test_creative_ai_only_generates_power_cards(self):
        from sts2_sim import content
        state = self.combat()
        hand = self._new_turn(state, [("creative_ai", 2)])
        powers_in_hand = [c.cid for c in hand
                          if content.CARD_DB[c.cid].card_type == "power"]
        self.assertGreaterEqual(len(powers_in_hand), 1)

    def test_spectrum_shift_draws_from_the_colorless_pool(self):
        from sts2_sim import content
        state = self.combat()
        hand = self._new_turn(state, [("spectrum_shift", 1)])
        colorless = set(content.CARD_POOLS.get("ColorlessCardPool", ()))
        self.assertTrue(any(c.cid in colorless for c in hand),
                        "无色变形应当从 ColorlessCardPool 里生成")

    def test_hello_world_uses_the_turn_start_snapshot(self):
        """⭐ 张数取 **``AmountOnTurnStart``**（回合开始时的层数），不是当前层数。

        快照只在 ``start_player_turn`` 的第 1 步产生，所以这里必须真的开一个回合；
        回合中途把层数加到 3，再触发一次仍应按快照 1 张。
        """
        from sts2_sim import content, core, powers as power_rules
        state = self.combat()
        # 先把开局手牌塞回抽牌堆：否则 hand 会顶到上限 10，后续生成放不进来
        state.draw_pile.extend(state.hand)
        state.hand.clear()
        state.player.add_power("hello_world", 1)
        core.start_player_turn(state, [])            # 快照 = 1 → 生成 1 张
        first = [c.cid for c in state.hand
                 if content.CARD_DB[c.cid].rarity == "common"]
        state.player.add_power("hello_world", 2)     # 回合中途变成 3 层
        power_rules.on_before_hand_draw(state, [])
        second = [c.cid for c in state.hand
                  if content.CARD_DB[c.cid].rarity == "common"]
        self.assertEqual(len(first), 1, "快照 1 层 → 生成 1 张")
        self.assertEqual(len(second) - len(first), 1,
                         "再触发一次仍应按快照 1 张，而不是当前 3 层")


class ConditionalDamageModifierTest(BuffPowerTestCase):
    """带条件的数值修正（``ModifyDamageAdditive`` / ``ModifyDamageMultiplicative``）。

    这一族最容易出错的地方是**条件**：丢掉条件 = 无条件生效 = **能力变强**，
    而且症状只是"伤害高几点"。所以每条测试都配了"条件不满足时不该生效"的对照。
    """

    def _hit(self, state, attacker, defender, amount):
        from sts2_sim import core
        return core.deal_damage(state, attacker, defender, amount, [])

    def test_accuracy_only_boosts_shivs(self):
        """``base.Owner == dealer`` + 牌带 ``Shiv`` 标签 —— 普通攻击不加。"""
        from sts2_sim import content
        state = self.combat()
        state.player.add_power("accuracy", 2)
        shiv = content.CARD_DB["shiv"]
        base_damage = sum(e.amount for e in shiv.effects if e.op == "damage")
        enemy = state.enemies[0]
        before = enemy.hp
        self.play(state, "shiv")
        self.assertEqual(enemy.hp, before - (base_damage + 2), "飞刀应当 +2")
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(enemy.hp, before - 6, "打击不该吃到精准")

    def test_leadership_only_boosts_same_side_others(self):
        """``base.Owner != dealer`` → 挂在敌人身上时，**队友**的攻击 +Amount。"""
        from sts2_sim import core
        state = self.combat(("nibbit", "nibbit"))
        state.enemies[0].add_power("leadership", 2)
        before = state.player.hp
        core.deal_damage(state, state.enemies[1], state.player, 5, [])
        self.assertEqual(state.player.hp, before - 7, "队友攻击 +2")
        before = state.player.hp
        core.deal_damage(state, state.enemies[0], state.player, 5, [])
        self.assertEqual(state.player.hp, before - 5, "自己打的没有加成（!= dealer）")

    def test_covered_blocks_all_powered_attacks(self):
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("covered", 1)
        before = state.player.hp
        core.deal_damage(state, state.enemies[0], state.player, 10, [])
        self.assertEqual(state.player.hp, before, "×0 = 完全免疫")

    def test_guarded_halves_incoming_attacks(self):
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("guarded", 1)
        before = state.player.hp
        core.deal_damage(state, state.enemies[0], state.player, 10, [])
        self.assertEqual(state.player.hp, before - 5, "×0.5")

    def test_colossus_needs_vulnerable_on_the_attacker(self):
        """⭐ 条件里含"**攻击者**身上有易伤" —— 漏掉它就变成无条件减半。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("colossus", 1)
        before = state.player.hp
        core.deal_damage(state, state.enemies[0], state.player, 10, [])
        self.assertEqual(state.player.hp, before - 10, "攻击者没有易伤 → 不减半")
        state.enemies[0].add_power("vulnerable", 1)
        before = state.player.hp
        core.deal_damage(state, state.enemies[0], state.player, 10, [])
        self.assertEqual(state.player.hp, before - 5, "攻击者有易伤 → ×0.5")

    def test_double_damage_needs_a_card_source(self):
        """``cardSource != null``：怪物招式（没有牌来源）**不**翻倍。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("double_damage", 1)
        before = state.enemies[0].hp
        self.play(state, "strike_ironclad")          # 卡牌伤害 → ×2
        self.assertEqual(state.enemies[0].hp, before - 12)
        # 敌人持有时：无牌来源的攻击不翻倍
        state.enemies[0].add_power("double_damage", 1)
        before = state.player.hp
        core.deal_damage(state, state.enemies[0], state.player, 5, [])
        self.assertEqual(state.player.hp, before - 5, "怪物招式没有 cardSource → 不翻倍")

    def test_tracking_needs_weak_on_the_target(self):
        """``target.HasPower<WeakPower>()``：目标身上没有虚弱就不加成。"""
        state = self.combat()
        state.player.add_power("tracking", 50)
        enemy = state.enemies[0]
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(enemy.hp, before - 6, "没有虚弱 → 原伤害")
        enemy.add_power("weak", 1)
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(enemy.hp, before - 9, "有虚弱 → ×1.5")


class CardGeneratedForCombatTest(BuffPowerTestCase):
    """``AfterCardGeneratedForCombat`` 一族（``ArsenalPower`` / ``SmokestackPower`` 等）。"""

    def _generate(self, state, cid):
        from sts2_sim import powers as power_rules
        from sts2_sim.core import CardInstance
        power_rules.on_card_generated_for_combat(
            state, [], CardInstance(cid), creator=state.player)

    def test_arsenal_gains_strength_on_generated_card(self):
        state = self.combat()
        state.player.add_power("arsenal", 2)
        self._generate(state, "shiv")
        self.assertEqual(state.player.power("strength"), 2)

    def test_pillar_of_creation_gains_block(self):
        state = self.combat()
        state.player.add_power("pillar_of_creation", 3)
        self._generate(state, "shiv")
        self.assertEqual(state.player.block, 3)

    def test_smokestack_only_fires_on_status_cards(self):
        """⭐ 两道条件：**牌型是 Status** + 是自己生成的。少第一条强度差一个量级。"""
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("smokestack", 4)
        before = [e.hp for e in state.enemies]
        self._generate(state, "shiv")                     # 攻击牌 → 不触发
        self.assertEqual([e.hp for e in state.enemies], before)
        self._generate(state, "dazed")                    # 状态牌 → 群伤
        self.assertEqual([e.hp for e in state.enemies], [hp - 4 for hp in before])

    def test_trash_to_treasure_channels_orbs_on_status_only(self):
        state = self.combat()
        state.player.add_power("trash_to_treasure", 2)
        self._generate(state, "shiv")
        self.assertEqual(list(state.orbs), [], "非状态牌不该引导充能球")
        self._generate(state, "dazed")
        self.assertEqual(len(state.orbs), 2, "状态牌 → 引导 2 个随机球")

    def test_power_generated_cards_also_fire_the_hook(self):
        """⭐ 能力自己生成的牌也必须走同一个出口，否则军械库对"无限之刃"无效。"""
        from sts2_sim import core
        state = self.combat()
        state.draw_pile.extend(state.hand)
        state.hand.clear()
        state.player.add_power("infinite_blades", 1)
        state.player.add_power("arsenal", 2)
        core.start_player_turn(state, [])                 # 抽牌前生成飞刀
        self.assertEqual(state.player.power("strength"), 2,
                         "无限之刃生成的飞刀应当触发军械库")


class EnergyCostModifierTest(BuffPowerTestCase):
    """能量费用修改族（``TryModifyEnergyCostInCombat`` / ``…Late``）。

    两个阶段：非 Late（`CuriousPower` 减费）→ Late（免费族 + 腐败归零）。
    顺序反了会得到"腐败下打纠缠的技能牌还要付费"这类错误。
    """

    def _cost(self, state, cid):
        from sts2_sim.core import CardInstance, play_cost
        return play_cost(state, CardInstance(cid))

    def test_free_attack_zeroes_attacks_only(self):
        state = self.combat()
        state.player.add_power("free_attack", 1)
        self.assertEqual(self._cost(state, "strike_ironclad"), 0)
        self.assertGreater(self._cost(state, "defend_ironclad"), 0, "技能牌不该免费")

    def test_free_skill_zeroes_skills_only(self):
        state = self.combat()
        state.player.add_power("free_skill", 1)
        self.assertEqual(self._cost(state, "defend_ironclad"), 0)
        self.assertGreater(self._cost(state, "strike_ironclad"), 0)

    def test_free_power_zeroes_power_cards_only(self):
        state = self.combat()
        state.player.add_power("free_power", 1)
        self.assertEqual(self._cost(state, "inflame"), 0)
        self.assertGreater(self._cost(state, "defend_ironclad"), 0)

    def test_veilpiercer_only_zeroes_ethereal_cards(self):
        """⭐ 判据是**关键字** ``Ethereal``，不是牌型。"""
        from sts2_sim import content
        state = self.combat()
        state.player.add_power("veilpiercer", 1)
        self.assertEqual(self._cost(state, "apparition"), 0)
        ethereal = content.CARD_DB["apparition"]
        self.assertIn("Ethereal", tuple(ethereal.keywords or ()))
        self.assertGreater(self._cost(state, "defend_ironclad"), 0)

    def test_curious_reduces_power_card_cost_by_amount(self):
        state = self.combat()
        state.player.add_power("curious", 1)
        self.assertEqual(self._cost(state, "demon_form"), 2, "3 费 − 1 = 2")
        state.player.add_power("curious", 5)
        self.assertEqual(self._cost(state, "demon_form"), 0, "减到负数要钳到 0")
        self.assertGreater(self._cost(state, "strike_ironclad"), 0, "只对能力牌生效")

    def test_void_form_frees_only_the_first_amount_cards(self):
        """⭐ 层数决定"免费几张"：打满之后就要正常付费。"""
        state = self.combat()
        state.player.add_power("void_form", 2)
        state.cards_played_started_this_turn = 0
        self.assertEqual(self._cost(state, "strike_ironclad"), 0)
        state.cards_played_started_this_turn = 2
        self.assertGreater(self._cost(state, "strike_ironclad"), 0,
                           "本回合已打完 2 张，第 3 张要付费")

    def test_late_stage_wins_over_earlier_surcharge(self):
        """腐败（Late）把纠缠（非 Late）加的那一笔一并清零 = 免费。

        顺序反了会得到"腐败下打纠缠技能牌还要付 1 费"。
        """
        state = self.combat()
        state.player.add_power("corruption", 1)
        state.player.add_power("tangled", 3)
        self.assertEqual(self._cost(state, "defend_ironclad"), 0)


class PowerAmountChangedTest(BuffPowerTestCase):
    """``AfterPowerAmountChanged`` 一族（``ShroudPower`` / ``SleightOfFleshPower``）。

    真机在**两个**调用点通知（施加 ``PowerCmd.cs:160``、修改 ``:249``），而
    ``Decrement`` 走的是后者的 ``applier: null`` —— 所以"谁上的"是一条硬条件。
    """

    def _apply(self, state, target, pid, amount, applier):
        from sts2_sim import powers as power_rules
        power_rules.apply_power_to(state, target, pid, amount, [], applier=applier)

    def test_shroud_gives_block_when_owner_applies_doom(self):
        state = self.combat()
        state.player.add_power("shroud", 3)
        self._apply(state, state.enemies[0], "doom", 5, state.player)
        self.assertEqual(state.player.block, 3)

    def test_shroud_ignores_doom_applied_by_someone_else(self):
        """``applier == base.Owner``：别人上的末日不给格挡。"""
        state = self.combat()
        state.player.add_power("shroud", 3)
        self._apply(state, state.enemies[0], "doom", 5, state.enemies[0])
        self.assertEqual(state.player.block, 0)

    def test_shroud_ignores_other_powers_and_decay(self):
        """递减（``Decrement`` → ``applier=None``）与**其它能力**都不该触发。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("shroud", 3)
        self._apply(state, state.enemies[0], "vulnerable", 2, state.player)
        self.assertEqual(state.player.block, 0, "不是末日 → 不给格挡")
        power_rules.apply_power_to(state, state.enemies[0], "doom", 4, [],
                                   applier=state.player)
        state.player.block = 0
        state.enemies[0].add_power("doom", -1)        # 递减：applier=None
        self.assertEqual(state.player.block, 0, "递减不给格挡")

    def test_sleight_of_flesh_hits_enemy_when_owner_applies_debuff(self):
        state = self.combat()
        state.player.add_power("sleight_of_flesh", 4)
        enemy = state.enemies[0]
        before = enemy.hp
        self._apply(state, enemy, "vulnerable", 2, state.player)
        self.assertEqual(enemy.hp, before - 4)

    def test_sleight_of_flesh_ignores_buffs_and_player_targets(self):
        """四条条件：非零变动 + **减益** + 目标是**敌人** + 自己上的。"""
        state = self.combat()
        state.player.add_power("sleight_of_flesh", 4)
        enemy = state.enemies[0]
        before = enemy.hp
        self._apply(state, enemy, "strength", 2, state.player)    # 增益 → 不触发
        self.assertEqual(enemy.hp, before)
        before = state.player.hp
        self._apply(state, state.player, "vulnerable", 2, state.player)  # 目标是玩家
        self.assertEqual(state.player.hp, before)


class CardPlayResultLocationTest(BuffPowerTestCase):
    """``ModifyCardPlayResultLocation`` 一族（``ReboundPower`` / ``NostalgiaPower``）。

    默认落点是**弃牌堆**；这两个能力把它改成**抽牌堆顶**，
    并且只有"真的改过"时才走 ``AfterModifyingCardPlayResultLocation``。
    """

    def test_rebound_sends_the_card_to_the_top_of_the_draw_pile(self):
        state = self.combat()
        state.player.add_power("rebound", 1)
        self.play(state, "strike_ironclad")
        self.assertEqual(state.draw_pile[-1].cid, "strike_ironclad",
                         "应当放回抽牌堆**顶**")
        self.assertEqual(state.player.power("rebound"), 0, "改过之后递减 1 层")

    def test_without_rebound_the_card_discards(self):
        state = self.combat()
        self.play(state, "strike_ironclad")
        self.assertEqual(state.discard[-1].cid, "strike_ironclad")

    def test_exhausting_cards_are_not_rerouted(self):
        """消耗牌在落点查询**之前**就进了消耗堆 —— 弹回不该把它救回来。"""
        from sts2_sim import content
        state = self.combat()
        exhausting = next(cid for cid, d in content.CARD_DB.items()
                          if d.exhaust and d.card_type == "skill")
        state.player.add_power("rebound", 1)
        self.play(state, exhausting)
        self.assertEqual(state.exhaust[-1].cid, exhausting)
        self.assertEqual(state.player.power("rebound"), 1, "没改过就不该掉层")

    def test_nostalgia_only_reroutes_attack_and_skill(self):
        state = self.combat()
        state.player.add_power("nostalgia", 1)
        self.play(state, "defend_ironclad")           # 技能牌 ✓
        self.assertEqual(state.draw_pile[-1].cid, "defend_ironclad")
        self.play(state, "inflame")                   # 能力牌 ✗
        self.assertEqual(state.discard[-1].cid, "inflame")

    def test_nostalgia_stops_after_amount_cards(self):
        """⭐ 层数限的是**本回合这类牌的张数**：打满之后要正常进弃牌堆。"""
        state = self.combat()
        state.player.add_power("nostalgia", 1)
        self.play(state, "strike_ironclad")
        self.assertEqual(state.draw_pile[-1].cid, "strike_ironclad")
        self.play(state, "strike_ironclad")
        self.assertEqual(state.discard[-1].cid, "strike_ironclad",
                         "第 2 张不该再回抽牌堆")


class AttackCommandModifierTest(BuffPowerTestCase):
    """攻击指令族（``BeforeAttack`` / ``AfterAttack``）：`LethalityPower` / `GigantificationPower`。

    两个能力的条件都挂在"这是**第几条**攻击指令"上，所以每条测试都配了
    "第二张牌不该再吃到"的对照 —— 少了对照，"只加成第一张"和"每张都加成"
    在只打一张牌的测试里长得一模一样。
    """

    def test_lethality_card_applies_50(self):
        """卡牌 ``Lethality`` 自己：1 费能力牌、Ethereal、给 50 层（= +50%）。"""
        state = self.combat()
        self.play(state, "lethality")
        self.assertEqual(state.player.power("lethality"), 50)

    def test_lethality_buffs_only_the_first_attack_of_the_turn(self):
        state = self.combat()
        state.player.add_power("lethality", 50)
        enemy = state.enemies[0]
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(enemy.hp, before - 9, "本回合第一张攻击牌 ×1.5（6 → 9）")
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(enemy.hp, before - 6, "第二张攻击牌回到原值")

    def test_lethality_ignores_skills(self):
        """技能牌不是"第一张攻击牌"：先打防御，打击照样吃满加成。"""
        state = self.combat()
        state.player.add_power("lethality", 50)
        enemy = state.enemies[0]
        self.play(state, "defend_ironclad")
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(enemy.hp, before - 9)

    def test_lethality_not_on_repeat_play(self):
        """⭐ "多打一次"的第二遍不再算第一张（源码 ``CurrentPlayIndex > 0``）。

        真机每个 playIndex 都登记一条 ``CardPlayStarted``，所以第二遍读到的是
        "本回合已有 2 条攻击记录" —— 引擎靠"逐次打出 +1"的同一个量恒等表达。
        """
        state = self.combat()
        state.player.add_power("lethality", 50)
        state.player.add_power("one_two_punch", 1)
        enemy = state.enemies[0]
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(enemy.hp, before - 15, "第一遍 9（×1.5）+ 第二遍 6（原值）")

    def test_gigantification_triples_the_next_attack(self):
        state = self.combat()
        state.player.add_power("gigantification", 1)
        enemy = state.enemies[0]
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(enemy.hp, before - 18, "源码是字面量 3m：6 × 3")
        self.assertEqual(state.player.power("gigantification"), 0, "指令结算完递减 1 层")

    def test_gigantification_ignores_skills(self):
        """技能牌不吃也不消耗：**只有**攻击指令才结账。"""
        state = self.combat()
        state.player.add_power("gigantification", 1)
        enemy = state.enemies[0]
        self.play(state, "defend_ironclad")
        self.assertEqual(state.player.power("gigantification"), 1, "没结算过就不该掉层")
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(enemy.hp, before - 18)

    def test_gigantification_amount_counts_uses(self):
        """层数 = 还能用几次（药水给 1 层）。"""
        state = self.combat()
        state.player.add_power("gigantification", 2)
        enemy = state.enemies[0]
        for _ in range(2):
            before = enemy.hp
            self.play(state, "strike_ironclad")
            self.assertEqual(enemy.hp, before - 18)
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(enemy.hp, before - 6, "两次用完 → 第三张恢复正常")
        self.assertEqual(state.player.power("gigantification"), 0)


class InstanceDataPowersTest(BuffPowerTestCase):
    """第八批：需要**实例数据**（"记下当时的层数 / 剩几张 / 当前的量"）的能力。

    ``SubroutinePower`` / ``GravityPower`` 一族用 ``Dictionary<CardModel,int>`` 记
    "这张牌开始结算时的层数"，``RollingBoulderPower`` 用自增量，``WitheringPresencePower``
    用 ``CardsLeft`` 倒计数 —— 全是 ``InitInternalData`` 那一类实例状态，
    引擎放进 ``power_flags``（与 ``AmountOnTurnStart`` 同一套做法）。

    ⚠️ 每条都配了反向对照："不是能力牌 / 不是同伴打的不该触发"，
    因为这类能力的错误方向都是**多触发**（静默变强）。
    """

    def test_subroutine_refunds_energy_for_power_cards(self):
        """``BeforeCardPlayed`` 记层数 → ``AfterCardPlayed`` 还回**等量**能量。"""
        from sts2_sim import content
        state = self.combat()
        state.player.add_power("subroutine", 2)
        cost = content.CARD_DB["inflame"].cost
        self.play(state, "inflame")
        self.assertEqual(state.energy, 9 - cost + 2, "能力牌结算完返还 2 点")

    def test_subroutine_ignores_skills(self):
        """反向对照：技能牌不进字典 → 一点都不返（源码按 ``CardPlay`` 的牌型过滤）。"""
        from sts2_sim import content
        state = self.combat()
        state.player.add_power("subroutine", 2)
        cost = content.CARD_DB["defend_ironclad"].cost
        self.play(state, "defend_ironclad")
        self.assertEqual(state.energy, 9 - cost)

    def test_rolling_boulder_hits_all_enemies_and_grows(self):
        """``AfterPlayerTurnStart``：全体可打敌人吃 Amount 点 Unpowered 伤害，然后 +5。"""
        from sts2_sim import core
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("rolling_boulder", 5)
        before = [e.hp for e in state.enemies]
        core.start_player_turn(state, [])
        self.assertEqual([e.hp for e in state.enemies],
                         [hp - 5 for hp in before], "两个敌人都吃 5 点")
        self.assertEqual(state.player.power("rolling_boulder"), 10, "Amount += 5")
        before = [e.hp for e in state.enemies]
        core.start_player_turn(state, [])
        self.assertEqual([e.hp for e in state.enemies],
                         [hp - 10 for hp in before], "第二回合滚到 10")

    def test_rolling_boulder_damage_is_unpowered(self):
        """``ValueProp.Unpowered``：力量/易伤都不该改变它。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("rolling_boulder", 5)
        state.player.add_power("strength", 10)
        state.enemies[0].add_power("vulnerable", 3)
        before = state.enemies[0].hp
        core.start_player_turn(state, [])
        self.assertEqual(state.enemies[0].hp, before - 5)

    def test_withering_presence_adds_a_wither_after_six_cards(self):
        """``AfterCardPlayed``：打满 6 张 → 往玩家手里塞 1 张「凋零」并把计数复位。"""
        state = self.combat()
        state.enemies[0].add_power("withering_presence", 1)
        for i in range(5):
            self.play(state, "defend_ironclad")
            self.assertEqual([c.cid for c in state.hand].count("wither"), 0,
                             f"第 {i + 1} 张还不该给凋零")
        self.play(state, "defend_ironclad")
        self.assertEqual([c.cid for c in state.hand].count("wither"), 1,
                         "第 6 张给 1 张凋零")
        self.assertEqual(state.enemies[0].power_flags["withering_cards_left"], 6,
                         "计数复位成 6")

    def test_shadow_step_grants_double_damage_then_leaves(self):
        """``AfterSideTurnStart``：拥有者阵营回合开始 → +Amount 双倍伤害并整条移除。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("shadow_step", 1)
        core.start_player_turn(state, [])
        self.assertEqual(state.player.power("double_damage"), 1)
        self.assertEqual(state.player.power("shadow_step"), 0, "用完就没了")
        enemy = state.enemies[0]
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(enemy.hp, before - 12, "6 × 2")

    def test_underworld_punishes_ally_attacks_only(self):
        """``AfterDamageGiven``：**同伴**的有效攻击 → 给目标上 ``总伤害 × Amount`` 末日。"""
        from sts2_sim import core
        state = self.combat(("nibbit", "nibbit"))
        state.enemies[0].add_power("underworld", 2)
        player = state.player
        core.deal_damage(state, state.enemies[1], player, 5, [])
        self.assertEqual(player.power("doom"), 10, "队友打的 5 × 2 层")
        # 反向 1：拥有者**自己**打的伤害不算（源码 dealer != base.Owner）
        player.add_power("doom", -player.power("doom"))
        core.deal_damage(state, state.enemies[0], player, 5, [])
        self.assertEqual(player.power("doom"), 0)
        # 反向 2：**玩家**打敌人不算（dealer.Side != Owner.Side）
        core.deal_damage(state, player, state.enemies[1], 5, [])
        self.assertEqual(player.power("doom"), 0)

    def test_underworld_removed_at_enemy_turn_end(self):
        """``AfterSideTurnEnd``：``side == Enemy`` → 整条移除。"""
        from sts2_sim import core
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.enemies[0].add_power("underworld", 1)
        power_rules.on_side_turn_end(state, [], "player")
        self.assertEqual(state.enemies[0].power("underworld"), 1, "玩家回合结束不移除")
        power_rules.on_side_turn_end(state, [], "enemy")
        self.assertEqual(state.enemies[0].power("underworld"), 0, "敌方回合结束移除")
        _ = core


class EnergyBlockOrbHookTest(BuffPowerTestCase):
    """第十批：三个**新调用点**上的能力（能量消耗 / 格挡加法条件 / 激发球）。

    | 能力 | 新调用点 | 最容易写错的地方 |
    |---|---|---|
    | `orbit` | `play_card` 扣费之后 | 按"花掉的能量取模"记账 → 层数中途变化时补发/漏发 |
    | `fasten` | `compute_block` 的加法阶段 | 漏掉 `Defend` 标签条件 → 所有格挡都 +Amount（静默变强） |
    | `thunder` | `orbs.evoke` 之后（带激发目标） | 漏掉"只认闪电球" → 任何球都追加伤害 |
    """

    def test_orbit_grants_energy_every_four_spent(self):
        """``AfterEnergySpent``：累计花满 4 点 → 回 Amount 点能量。"""
        from sts2_sim.core import Action, CardInstance, step
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("orbit", 2)
        state.energy = 4
        for _ in range(4):
            card = CardInstance("defend_ironclad")
            state.hand.append(card)
            step(state, Action("play_card", state.hand.index(card), 0))
        self.assertEqual(state.energy, 2, "4 点花完 → 立刻回 2 点")

    def test_orbit_does_not_grant_early(self):
        """反向对照：只花了 3 点不该给（阈值是源码里的私有常量 4）。"""
        from sts2_sim.core import Action, CardInstance, step
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("orbit", 2)
        state.energy = 4
        for _ in range(3):
            card = CardInstance("defend_ironclad")
            state.hand.append(card)
            step(state, Action("play_card", state.hand.index(card), 0))
        self.assertEqual(state.energy, 1, "花 3 点还剩 1，不触发")

    def test_fasten_only_counts_defend_tagged_cards(self):
        """``ModifyBlockAdditive``：卡牌来源必须带 ``Defend`` 标签。"""
        state = self.combat()
        state.player.add_power("fasten", 3)
        state.player.block = 0
        self.play(state, "defend_ironclad")            # 5 格挡，标签 Defend
        self.assertEqual(state.player.block, 8, "5 + 3")
        state.player.block = 0
        self.play(state, "iron_wave")                  # 5 格挡，**没有** Defend 标签
        self.assertEqual(state.player.block, 5, "非 Defend 标签不加")

    def test_fasten_without_defend_tag_cards_is_noop(self):
        """反向对照：没有 Defend 标签的牌一次都不该被加。"""
        state = self.combat()
        state.player.add_power("fasten", 3)
        state.player.block = 0
        self.play(state, "shrug_it_off")               # 8 格挡，标签为空
        self.assertEqual(state.player.block, 8)

    def test_thunder_adds_damage_to_a_lightning_evoke(self):
        """``AfterOrbEvoked``：激发闪电球 → 对**这次激发的目标**再打 Amount 点。"""
        from sts2_sim import orbs
        state = self.combat()
        state.player.add_power("thunder", 4)
        orbs.channel(state, "lightning", [])
        enemy = state.enemies[0]
        before = enemy.hp
        orbs.evoke(state, 0, [])
        self.assertEqual(enemy.hp, before - (8 + 4), "闪电激发 8 + 雷 4")

    def test_thunder_ignores_other_orbs(self):
        """反向对照：冰球激发不触发雷（源码里 ``orb is LightningOrb``）。"""
        from sts2_sim import orbs
        state = self.combat()
        state.player.add_power("thunder", 4)
        orbs.channel(state, "frost", [])
        state.player.block = 0
        before = state.enemies[0].hp
        orbs.evoke(state, 0, [])
        self.assertEqual(state.enemies[0].hp, before, "冰球不触发")
        self.assertEqual(state.player.block, 5, "冰球照常给格挡")


class BlockSidePowersTest(BuffPowerTestCase):
    """第十一批：格挡侧的两个能力（`BlurPower` / `ShadowmeldPower`）。

    两个都在"格挡怎么算 / 留不留"这条链上，而这条链在引擎里只有一个入口
    （``gain_block`` / ``start_player_turn`` 的清格挡那一步），所以测试直接用
    这两个入口驱动。
    """

    def test_blur_prevents_the_block_clear_then_expires(self):
        """``ShouldClearBlock`` → 回合开始不清格挡；``AfterSideTurnStart`` 递减 1 层。"""
        from sts2_sim import core
        state = self.combat()
        state.player.block = 10
        state.player.add_power("blur", 1)
        state.turn = 2                                   # 第 1 回合本来就豁免清格挡
        core.start_player_turn(state, [])
        self.assertEqual(state.player.block, 10, "潜伏生效：格挡留住了")
        self.assertEqual(state.player.power("blur"), 0, "回合开始递减 1 层")
        state.player.block = 10
        core.start_player_turn(state, [])
        self.assertEqual(state.player.block, 0, "层数用完 → 照常清空")

    def test_without_blur_block_is_cleared(self):
        """反向对照：没有潜伏时第 2 回合格挡必须清空。"""
        from sts2_sim import core
        state = self.combat()
        state.player.block = 10
        state.turn = 2
        core.start_player_turn(state, [])
        self.assertEqual(state.player.block, 0)

    def test_shadowmeld_multiplies_block_by_two_per_layer(self):
        """``ModifyBlockMultiplicative``：×``2^Amount``。"""
        state = self.combat()
        state.player.add_power("shadowmeld", 2)
        state.player.block = 0
        self.play(state, "defend_ironclad")
        self.assertEqual(state.player.block, 20, "5 × 2²")

    def test_shadowmeld_also_multiplies_unpowered_block(self):
        """⭐ 源码**不看 ``ValueProp``** —— 药水/充能球的 Unpowered 格挡同样要乘。"""
        from sts2_sim.core import use_potion
        state = self.combat()
        state.player.add_power("shadowmeld", 1)
        state.player.block = 0
        use_potion(state, "block_potion")
        self.assertEqual(state.player.block, 12 * 2, "药水的 12 点也要 ×2")

    def test_shadowmeld_removed_at_owner_side_turn_end(self):
        """``AfterSideTurnEnd`` 且 ``participants.Contains(Owner)`` → 整条移除。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("shadowmeld", 1)
        power_rules.on_side_turn_end(state, [], "enemy")
        self.assertEqual(state.player.power("shadowmeld"), 1, "敌方回合结束不移除")
        power_rules.on_turn_end(state.player, state, [])
        self.assertEqual(state.player.power("shadowmeld"), 0, "自己阵营回合结束移除")


    def test_well_laid_plans_keeps_the_hand_at_turn_end(self):
        """``ShouldFlush`` → false：回合结束手牌**不弃**（与 `RetainHand` 判据相同）。

        ⚠️ 断言看的是**"有没有发生弃手牌"**与**张数**，不是"手牌内容"：
        回合结束时弃掉的牌马上会被下一回合的抽牌重新洗回来（同一个实例、同一个 uid），
        比内容等于什么都没测出来。
        """
        from sts2_sim.core import Action, step
        state = self.combat()
        step(state, Action("end_turn"))
        self.assertTrue(any("回合结束弃掉" in line for line in state.log),
                        "反向对照：没有能力时必须弃手牌")
        state2 = self.combat()
        state2.player.add_power("well_laid_plans", 1)
        step(state2, Action("end_turn"))
        self.assertFalse(any("回合结束弃掉" in line for line in state2.log),
                         "未雨绸缪：一张都不弃")
        self.assertEqual(len(state2.hand), 10, "留住的 5 张 + 新抽的 5 张")


class PowerInstanceVarTest(BuffPowerTestCase):
    """第十二批：``set_power_var``（能力实例变量）与 ``AfterBlockCleared``。

    这一批同时补了**抽取器的一个静默丢失**：``(await PowerCmd.Apply<X>(…)).SetDamage(值)``
    以前抽不出来也不报错 —— 卡看起来"干净"，效果表里却少了"打多少"。
    所以两条测试都钉在**数值来自卡牌动态变量**这件事上（含升级）。
    """

    def test_the_bomb_uses_the_card_variable(self):
        """``TheBomb``：3 回合后对全体造成**卡牌 BombDamage** 点伤害并移除。"""
        from sts2_sim import core, powers as power_rules
        state = self.combat(("nibbit", "nibbit"))
        self.play(state, "the_bomb")
        self.assertEqual(state.player.power("the_bomb"), 3, "3 回合倒计时")
        self.assertEqual(state.player.power_vars["the_bomb"]["Damage"], 40)
        before = [e.hp for e in state.enemies]
        for _ in range(2):
            power_rules.on_before_side_turn_end(state, [], "player")
        self.assertEqual([e.hp for e in state.enemies], before, "还没到点，不炸")
        self.assertEqual(state.player.power("the_bomb"), 1, "递减到 1")
        power_rules.on_before_side_turn_end(state, [], "player")
        self.assertEqual([e.hp for e in state.enemies], [hp - 40 for hp in before],
                         "炸：全体 40 点")
        self.assertEqual(state.player.power("the_bomb"), 0, "炸完移除")
        _ = core

    def test_the_bomb_damage_is_unpowered(self):
        """``ValueProp.Unpowered``：力量与易伤都不该改变爆炸伤害。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("strength", 10)
        state.enemies[0].add_power("vulnerable", 3)
        self.play(state, "the_bomb")
        state.player.add_power("the_bomb", -2)          # 直接压到 1 层
        before = state.enemies[0].hp
        power_rules.on_before_side_turn_end(state, [], "player")
        self.assertEqual(state.enemies[0].hp, before - 40)

    def test_the_bomb_ignores_enemy_turn_end(self):
        """反向对照：判据是 ``participants.Contains(Owner)`` —— 敌方回合结束不炸。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        self.play(state, "the_bomb")
        state.player.add_power("the_bomb", -2)
        before = state.enemies[0].hp
        power_rules.on_before_side_turn_end(state, [], "enemy")
        self.assertEqual(state.enemies[0].hp, before, "敌方结束不该引爆")
        self.assertEqual(state.player.power("the_bomb"), 1)

    def test_toric_toughness_repeats_the_actual_block_gained(self):
        """``SetBlock`` 收到的是**上次实际获得的格挡**（含敏捷/脆弱），不是卡面的 5。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("dexterity", 3)          # 5 + 3 = 8
        self.play(state, "toric_toughness")
        self.assertEqual(state.player.power("toric_toughness"), 2, "2 回合")
        self.assertEqual(state.player.power_vars["toric_toughness"]["Block"], 8,
                         "记录的是**实际**获得的 8，不是卡面 5")
        state.player.block = 0
        core.start_player_turn(state, [])               # 清格挡 → 触发
        self.assertEqual(state.player.block, 8, "再给 8 点（Unpowered）")
        self.assertEqual(state.player.power("toric_toughness"), 1, "递减 1 层")

    def test_toric_toughness_block_is_unpowered(self):
        """给的是 ``ValueProp.Unpowered`` 格挡：**不再**吃敏捷、也不被脆弱打折。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("dexterity", 3)
        self.play(state, "toric_toughness")
        state.player.add_power("dexterity", 10)         # 之后涨的敏捷不该影响已记录值
        state.player.add_power("frail", 3)
        state.player.block = 0
        state.turn = 2
        core.start_player_turn(state, [])
        self.assertEqual(state.player.block, 8, "Unpowered：不重算、不打折")

    def test_self_forming_clay_gives_block_then_leaves(self):
        """``AfterBlockCleared``：自己的格挡被清空 → 获得 Amount 点格挡并移除。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("self_forming_clay", 6)
        state.turn = 2
        core.start_player_turn(state, [])
        self.assertEqual(state.player.block, 6, "清空之后给 6 点")
        self.assertEqual(state.player.power("self_forming_clay"), 0, "用完就没了")

    def test_self_forming_clay_only_on_own_clear(self):
        """反向对照：判据是 ``creature == base.Owner`` —— 敌人清格挡不该触发。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("self_forming_clay", 6)
        state.player.block = 0
        core._run_enemy_turn(state, [])
        self.assertEqual(state.player.block, 0, "敌方清格挡不触发")
        self.assertEqual(state.player.power("self_forming_clay"), 6, "层数还在")


class CardEnteredCombatTest(BuffPowerTestCase):
    """``AfterCardEnteredCombat``：**战斗中途生成的牌**要不要补苦痛。

    真机把苦痛族拆成两半：``AfterApplied`` 管"贴能力那一刻在场的牌"，
    ``AfterCardEnteredCombat`` 管"之后新进场的牌"（``CardPileCmd.cs:515``，
    判据是"此前没有任何牌堆"= 新造出来的牌）。引擎原来只有前一半 ——
    于是"生成一张攻击牌"在 `tangled` 下**不会被缠住**（打出去照常只花原费用），
    而日志与计数一切正常。这一组测试专门钉后一半。

    ⚠️ 反向对照（类型不符 / 本回合没打过技能）同样重要：`smoggy` 的判据是
    ``CardPlaysStarted.Any(type == Skill)``，"打过任意牌"不能顶替它。
    """

    def _enter(self, state, cid):
        """走引擎**生成牌的唯一致**入口（``CardPileCmd.AddGeneratedCardToCombat``）。"""
        from sts2_sim import core
        card = core.CardInstance(cid)
        core._add_generated_card(state, card, "hand")
        return card

    def test_tangled_afflicts_a_generated_attack(self):
        state = self.combat()
        state.player.add_power("tangled", 1)
        card = self._enter(state, "strike_ironclad")
        self.assertEqual(card.affliction, "entangled", "新生成的攻击牌要被缠住")

    def test_tangled_ignores_a_generated_skill(self):
        state = self.combat()
        state.player.add_power("tangled", 1)
        card = self._enter(state, "defend_ironclad")
        self.assertIsNone(card.affliction, "只贴攻击牌")

    def test_ringing_afflicts_any_generated_card(self):
        state = self.combat()
        state.player.add_power("ringing", 1)
        self.assertEqual(self._enter(state, "defend_ironclad").affliction, "ringing")

    def test_hex_afflicts_a_generated_card(self):
        state = self.combat()
        state.player.add_power("hex", 1)
        self.assertEqual(self._enter(state, "defend_ironclad").affliction, "hexed")

    def test_vital_spark_afflicts_a_generated_skill(self):
        state = self.combat()
        state.player.add_power("vital_spark", 1)
        self.assertEqual(self._enter(state, "defend_ironclad").affliction, "tainted")
        self.assertIsNone(self._enter(state, "strike_ironclad").affliction,
                          "只贴技能牌")

    def test_galvanic_afflicts_a_generated_power(self):
        state = self.combat()
        state.player.add_power("galvanic", 1)
        self.assertEqual(self._enter(state, "inflame").affliction, "galvanized")
        self.assertIsNone(self._enter(state, "defend_ironclad").affliction,
                          "只贴能力牌")

    def test_smoggy_requires_a_skill_played_this_turn(self):
        """⭐ 判据是"**本回合打过技能牌**" —— 没打过就不贴。"""
        state = self.combat()
        state.player.add_power("smoggy", 1)
        self.assertIsNone(self._enter(state, "defend_ironclad").affliction,
                          "这一回合还没打过技能牌 → 不贴")
        self.play(state, "defend_ironclad")             # 打一张技能牌
        card = self._enter(state, "defend_ironclad")
        self.assertEqual(card.affliction, "smog", "打过技能牌之后 → 新技能牌要被贴")

    def test_already_afflicted_cards_are_kept(self):
        """源码的条件里有 ``card.Affliction == null``：已有苦痛的不覆盖。"""
        state = self.combat()
        state.player.add_power("tangled", 1)
        from sts2_sim import core
        card = core.CardInstance("strike_ironclad")
        card.affliction = "tainted"
        core._add_generated_card(state, card, "hand")
        self.assertEqual(card.affliction, "tainted", "不该被覆盖成 entangled")


class InstanceKeywordTest(BuffPowerTestCase):
    """`PhantomBladesPower`：**卡实例级关键字**（``CardCmd.ApplyKeyword``）。

    卡牌关键字有两层：``CardDef.keywords`` 是**固有**的（改不得），
    ``CardCmd.ApplyKeyword`` 加的是**战斗内、单张牌**的（引擎存在
    ``CardInstance.keywords``，战斗结束随副本丢弃）。少了这一层，
    "幻影之刃给飞刀加保留"就只能改卡牌定义 —— 那会污染**所有**飞刀，
    而且下一场战斗还在。
    """

    def _shiv(self, state):
        from sts2_sim import core
        card = core.CardInstance("shiv")
        core._add_generated_card(state, card, "hand")
        return card

    def test_new_shiv_gets_retain(self):
        state = self.combat()
        state.player.add_power("phantom_blades", 9)
        card = self._shiv(state)
        self.assertIn("Retain", card.keywords, "飞刀应当获得保留")

    def test_retain_is_instance_level(self):
        """⭐ 只作用于**那一张**飞刀，不污染同名的另一张（也不是改卡牌定义）。"""
        state = self.combat()
        state.player.add_power("phantom_blades", 9)
        marked = self._shiv(state)
        plain = self._shiv(state)            # 已经贴过能力的飞刀 → 这张也会被贴
        self.assertIn("Retain", marked.keywords)
        self.assertIn("Retain", plain.keywords)
        # 反向：能力移除之后再生成的飞刀不带保留（关键字不是卡牌固有属性）
        state.player.add_power("phantom_blades", -9)
        fresh = self._shiv(state)
        self.assertNotIn("Retain", fresh.keywords)
        from sts2_sim import content
        self.assertNotIn("Retain", tuple(getattr(content.CARD_DB["shiv"], "keywords", ()) or ()),
                         "卡牌定义不该被改（否则会污染所有飞刀）")

    def test_retain_keeps_the_shiv_at_turn_end(self):
        from sts2_sim.core import Action, step
        state = self.combat()
        state.player.add_power("phantom_blades", 9)
        card = self._shiv(state)
        step(state, Action("end_turn"))
        self.assertIn(card, state.hand, "带保留的飞刀回合结束要留在手里")

    def test_first_shiv_each_turn_gets_the_bonus(self):
        """⭐ 源码数 ``CardPlaysFinished`` 里的飞刀：**第一张**吃 +Amount，第二张不吃。"""
        from sts2_sim import content
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("phantom_blades", 9)
        base = sum(e.amount for e in content.CARD_DB["shiv"].effects
                   if e.op == "damage")
        enemy = state.enemies[0]
        before = enemy.hp
        self.play(state, "shiv", target=0)
        self.assertEqual(before - enemy.hp, base + 9, "第一张飞刀 +9")
        before = enemy.hp
        self.play(state, "shiv", target=0)
        self.assertEqual(before - enemy.hp, base, "第二张不再加成")

    def test_without_the_power_no_bonus(self):
        """反向对照：没有这个能力时飞刀就是原伤害。"""
        from sts2_sim import content
        state = self.combat()
        base = sum(e.amount for e in content.CARD_DB["shiv"].effects
                   if e.op == "damage")
        enemy = state.enemies[0]
        before = enemy.hp
        self.play(state, "shiv", target=0)
        self.assertEqual(before - enemy.hp, base)


class UnwiredBehaviorTest(BuffPowerTestCase):
    """已注册但**行为没人消费**的能力（审计 ``tools/audit_power_hooks.py`` 找出来的）。

    这一组来自"注册了、描述了、却没有任何消费者"的扫描结果：
    ``curl_up`` / ``hard_to_kill`` / ``slow`` 当时**整条无效**；
    ``double_damage`` / ``colossus`` / ``intangible`` / ``covered`` 没有到期时机
    （永久生效 = 静默变强）；``block_next_turn`` 绕开了 ``gain_block``
    （"获得格挡时"的订阅者不触发）。每一条都能用"数值应该是多少"钉死。
    """

    def test_curl_up_gains_block_after_the_card_finishes(self):
        """``CurlUpPower``：被**牌**打中 → 那张牌**结算完**才获得 Amount 点格挡。"""
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("curl_up", 6)
        enemy.block = 0
        self.play(state, "strike_ironclad")
        self.assertEqual(enemy.block, 6, "被打之后蜷缩：+6 格挡")
        self.assertEqual(enemy.power("curl_up"), 0, "用完就移除")

    def test_curl_up_needs_a_card_source(self):
        """反向对照：``cardSource == null``（怪物招式 / 药水）不触发。"""
        from sts2_sim import core
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("curl_up", 6)
        enemy.block = 0
        core.deal_damage(state, state.player, enemy, 5, [])       # 没有 card
        self.assertEqual(enemy.block, 0)
        self.assertEqual(enemy.power("curl_up"), 6, "没触发就还在")

    def test_hard_to_kill_caps_each_hit(self):
        """``ModifyDamageCap``：打它的伤害**每次**最多 Amount 点（管线第 3 步）。"""
        from sts2_sim import core
        state = self.combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.add_power("hard_to_kill", 5)
        core.deal_damage(state, state.player, enemy, 30, [])
        self.assertEqual(100 - enemy.hp, 5, "30 点伤害被压到 5")
        core.deal_damage(state, state.player, enemy, 4, [])
        self.assertEqual(100 - enemy.hp, 9, "低于上限的伤害不受影响")

    def test_slow_amplifies_damage_against_its_holder(self):
        """``SlowPower``：每打出一张牌累计 1 格 → 它**受到的**有效攻击 ×(1+0.1×格数)。

        ⚠️ 数的是**本回合已经打完的牌**（``AfterCardPlayed`` 在伤害之后），
        所以正在打的那一张不算；格数太小时乘完仍会被 ``int()`` 截掉，
        这里用 3 格让差别显出来（6 × 1.3 = 7.8 → 7）。
        """
        from sts2_sim import powers as power_rules
        state = self.combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        enemy.add_power("slow", 1)
        for _ in range(3):
            self.play(state, "defend_ironclad")
        self.assertEqual(enemy.power_flags.get("slow_amount"), 3)
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(before - enemy.hp, 7, "6 × 1.3 = 7.8 → 7")
        power_rules.on_turn_start(enemy, state, [])    # 它自己回合开始 → 归零
        self.assertEqual(enemy.power_flags.get("slow_amount"), 0)
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(before - enemy.hp, 6, "归零之后恢复原值")

    def test_double_damage_expires_at_owner_turn_end(self):
        """⭐ ``DoubleDamagePower`` 只持续到自己回合结束（少了它 = 永久双倍）。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        state.player.add_power("double_damage", 1)
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(before - enemy.hp, 12, "在场的这一回合：6 × 2")
        power_rules.on_turn_end(state.player, state, [])
        self.assertEqual(state.player.power("double_damage"), 0, "自己回合结束递减")
        before = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(before - enemy.hp, 6, "之后恢复原值")

    def test_colossus_and_intangible_expire_at_enemy_turn_end(self):
        from sts2_sim import core, powers as power_rules
        state = self.combat()
        player = state.player
        player.add_power("colossus", 1)
        player.add_power("intangible", 1)
        before = player.hp
        core.deal_damage(state, state.enemies[0], player, 10, [])
        self.assertEqual(before - player.hp, 1, "无形：掉血压到 1")
        power_rules.on_side_turn_end(state, [], "player")
        self.assertEqual(player.power("intangible"), 1, "玩家回合结束不递减")
        power_rules.on_side_turn_end(state, [], "enemy")
        self.assertEqual(player.power("intangible"), 0, "敌方回合结束递减")
        self.assertEqual(player.power("colossus"), 0)
        before = player.hp
        core.deal_damage(state, state.enemies[0], player, 10, [])
        self.assertEqual(before - player.hp, 10, "到期之后照常吃满")

    def test_covered_expires_at_enemy_turn_end(self):
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("covered", 1)
        power_rules.on_side_turn_end(state, [], "player")
        self.assertEqual(state.player.power("covered"), 1, "不是敌方回合 → 不移除")
        power_rules.on_side_turn_end(state, [], "enemy")
        self.assertEqual(state.player.power("covered"), 0, "敌方回合结束整条移除")

    def test_block_next_turn_fires_the_block_gained_hook(self):
        """⭐ ``BlockNextTurnPower`` 走 ``GainBlock`` 通道 → 板甲（Juggernaut）要触发。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("block_next_turn", 5)
        state.player.add_power("juggernaut", 7)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        state.player.block = 0
        core.start_player_turn(state, [])
        self.assertGreaterEqual(state.player.block, 5, "留下 5 点格挡")
        self.assertEqual(100 - enemy.hp, 7, "板甲应当响应（旧实现直接加 block，静默不触发）")


class ApplierDeathTest(BuffPowerTestCase):
    """``AfterDeath`` 的**第三段作用域**：**施加者**死了 → 减益结束。

    真机 ``ConstrictPower`` / ``ShrinkPower`` / ``HexPower`` 的结束条件都是
    ``if (creature == base.Applier) Remove(this)`` —— 与"自己死 / 队友死"正交。
    引擎原来只记层数、不记施加者，于是"蛇死了还在缠你"（引擎比真机**苛刻**）。
    """

    def _constricted(self, state, pid="constrict", amount=3):
        """让**敌人**给玩家施加一条减益（这样施加者与拥有者不同）。

        ⚠️ 必须走 :func:`sts2_sim.powers.apply_power_to`（= 真机的 ``PowerCmd.Apply``）
        而不是 ``add_power``：只有前者会分发 ``AfterApplied``。
        ``add_power`` 是"改层数"的落点（``PowerCmd.Decrement`` 走它），不负责施加。
        """
        from sts2_sim import powers as power_rules
        enemy = state.enemies[0]
        power_rules.apply_power_to(state, state.player, pid, amount, [], applier=enemy)
        return enemy

    def test_constrict_ends_when_the_applier_dies(self):
        from sts2_sim import core
        state = self.combat()
        enemy = self._constricted(state)
        self.assertIs(state.player.powers_applier["constrict"], enemy,
                      "施加者要被记下来")
        core.mark_dead(state, enemy, [])
        self.assertEqual(state.player.power("constrict"), 0, "蛇死了，缠绕结束")

    def test_shrink_ends_when_the_applier_dies(self):
        from sts2_sim import core
        state = self.combat()
        enemy = self._constricted(state, pid="shrink", amount=1)
        core.mark_dead(state, enemy, [])
        self.assertEqual(state.player.power("shrink"), 0)

    def test_hex_ends_and_clears_the_affliction(self):
        """``HexPower``：移除的同时必须**清掉 Hexed**（否则那些牌继续带虚无）。"""
        from sts2_sim import core
        state = self.combat()
        enemy = self._constricted(state, pid="hex", amount=1)
        self.assertTrue(any(c.affliction == "hexed" for c in state.hand),
                        "施加上时要贴到手里的牌")
        core.mark_dead(state, enemy, [])
        self.assertEqual(state.player.power("hex"), 0)
        self.assertFalse(any(c.affliction == "hexed" for c in state.hand),
                         "移除时必须清苦痛")

    def test_debuffs_survive_an_unrelated_death(self):
        """反向对照：死的**不是**施加者时不该移除（别把"有东西死了"当成条件）。"""
        from sts2_sim import core
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("constrict", 3, applier=state.enemies[0])
        core.mark_dead(state, state.enemies[1], [])
        self.assertEqual(state.player.power("constrict"), 3, "另一个敌人死了不影响")

    def test_killing_the_holder_clears_its_own_applier_record(self):
        """层数归零时施加者记录也要清掉（否则复用同一 pid 会带着旧施加者）。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        enemy = state.enemies[0]
        power_rules.apply_power_to(state, state.player, "constrict", 3, [],
                                   applier=enemy)
        state.player.add_power("constrict", -3)
        self.assertNotIn("constrict", state.player.powers_applier)
        self.assertEqual(state.player.power("constrict"), 0)


class BoundAfflictionTest(BuffPowerTestCase):
    """``ChainsOfBindingPower`` 的 ``ShouldPlay``：**第一张** Bound 能打，其余打不出。

    真机分三处：

    * ``ShouldPlay(card)`` → ``!boundCardPlayed``（**查询**，卡在"能不能打"上，不是"打完不算"）；
    * ``BeforeCardPlayed`` → 翻 ``boundCardPlayed``；
    * ``BeforeSideTurnEnd`` → 复位标志 + **清掉所有 Bound**。

    ⚠️ 只接"每回合最多贴 N 张"那一半（原来的状态）等于**没有惩罚** ——
    被束缚的牌照样能打，而界面上还标着 Bound。
    """

    def _bind(self, state, cid="strike_ironclad"):
        from sts2_sim import core
        card = core.CardInstance(cid)
        card.affliction = "bound"
        state.hand.append(card)
        return card

    def _combat_with_chains(self, amount=2):
        state = self.combat()
        state.player.add_power("chains_of_binding", amount)
        return state

    def test_first_bound_card_is_playable_and_the_rest_are_not(self):
        from sts2_sim import core
        from sts2_sim.powers import affliction_blocks_play
        state = self._combat_with_chains()
        first = self._bind(state)
        second = self._bind(state)
        self.assertIsNone(affliction_blocks_play(state, first),
                          "第一张 Bound 永远能打")
        from sts2_sim.core import Action, step
        state.energy = 9
        step(state, Action("play_card", state.hand.index(first), 0))
        self.assertEqual(affliction_blocks_play(state, second), "chains_of_binding",
                         "打过一张之后，其余 Bound 打不出")

    def test_blocked_bound_card_is_masked_out_of_legal_actions(self):
        """**mask 是硬性要求**：打不出的牌不能出现在合法动作里（``docs/03`` §3.5）。"""
        from sts2_sim import core
        from sts2_sim.core import Action, step
        state = self._combat_with_chains()
        first = self._bind(state)
        second = self._bind(state)
        state.energy = 9
        step(state, Action("play_card", state.hand.index(first), 0))
        playable_indices = [a.hand_index for a in core.legal_actions(state)
                            if a.kind == "play_card"]
        self.assertNotIn(state.hand.index(second), playable_indices,
                         "打过一张 Bound 之后，其余 Bound 不该出现在合法动作里")

    def test_bound_does_not_block_without_the_power(self):
        """反向对照：没有这条能力时，带 Bound 的牌照常能打。"""
        from sts2_sim.powers import affliction_blocks_play
        state = self.combat()
        card = self._bind(state)
        state.player.power_flags["bound_played_this_turn"] = 1
        self.assertIsNone(affliction_blocks_play(state, card))

    def test_turn_end_clears_bound_and_the_flag(self):
        """``BeforeSideTurnEnd``：复位标志**并清掉所有 Bound**。"""
        from sts2_sim import powers as power_rules
        state = self._combat_with_chains()
        card = self._bind(state)
        state.player.power_flags["bound_played_this_turn"] = 1
        state.player.power_flags["bound_this_turn"] = 2
        power_rules.on_before_side_turn_end(state, [], "player")
        self.assertEqual(state.player.power_flags["bound_played_this_turn"], 0)
        self.assertEqual(state.player.power_flags["bound_this_turn"], 0)
        self.assertIsNone(card.affliction, "束缚只持续到这一回合结束")

    def test_card_drawn_gets_bound_up_to_the_cap(self):
        """``AfterCardDrawn``：每回合最多贴 ``Amount`` 张。"""
        from sts2_sim import core
        from sts2_sim import powers as power_rules
        state = self._combat_with_chains(amount=2)
        state.hand.clear()
        state.draw_pile = [core.CardInstance("strike_ironclad") for _ in range(4)]
        drawn = []
        for _ in range(4):
            card = state.draw_pile.pop()
            state.hand.append(card)
            power_rules.on_draw_card(state, card, [])
            drawn.append(card)
        self.assertEqual(sum(1 for c in drawn if c.affliction == "bound"), 2,
                         "最多 2 张（Amount）")


class SleepWakeTest(BuffPowerTestCase):
    """``asleep`` / ``slumber``：挨打醒来 + ``Stun(owner, WakeUpMove, nextMoveId)``。

    真机三段：``AfterDamageReceived``（掉血就醒）、``AfterSideTurnEnd``（倒计时，
    归零也醒），醒来时调 ``WakeUpMove`` 并 ``Stun`` 一个临时招式：
    **这一回合走唤醒动作**（唯一的数值部分是"移除镀层"）、**下一招固定**
    （拉瓦金 `SLASH_MOVE`、睡甲虫 `ROLL_OUT_MOVE`）。
    """

    def test_asleep_wakes_on_unblocked_damage(self):
        from sts2_sim import core
        state = self.combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.add_power("asleep", 3)
        enemy.add_power("plating", 8)
        core.deal_damage(state, state.player, enemy, 5, [])
        self.assertEqual(enemy.power("asleep"), 0, "掉血就醒")
        self.assertEqual(enemy.power("plating"), 0, "醒来时镀层被击碎"
                         "（这一条是 `AfterDamageReceived` 里**当下**就做的）")
        self.assertEqual(enemy.power_flags.get("stunned"), 1, "醒来这一回合被晕住")
        self.assertEqual(enemy.stun_follow_up, "slash", "下一招固定猛击")

    def test_asleep_does_not_wake_on_fully_blocked_damage(self):
        """⭐ 判据是 ``UnblockedDamage != 0``：被完全挡下打不醒。"""
        from sts2_sim import core
        state = self.combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.add_power("asleep", 3)
        enemy.block = 50
        core.deal_damage(state, state.player, enemy, 5, [])
        self.assertEqual(enemy.power("asleep"), 3, "全挡下 → 不醒")
        self.assertEqual(enemy.block, 45)

    def test_asleep_drops_plating_the_turn_before_waking(self):
        """``BeforeSideTurnEndVeryEarly``：倒计时到 1 时先扔掉镀层。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("asleep", 1)
        enemy.add_power("plating", 8)
        power_rules.on_before_side_turn_end(state, [], "enemy")
        self.assertEqual(enemy.power("plating"), 0, "醒来前一步掉镀层")
        self.assertEqual(enemy.power("asleep"), 1, "能力还在（下一回合才醒）")

    def test_slumber_wakes_when_damage_drains_the_counter(self):
        from sts2_sim import core
        state = self.combat(("nibbit", "nibbit"))
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.add_power("slumber", 2)
        enemy.add_power("plating", 6)
        core.deal_damage(state, state.player, enemy, 3, [])
        self.assertEqual(enemy.power("slumber"), 1, "掉血递减一层")
        core.deal_damage(state, state.player, enemy, 3, [])
        self.assertEqual(enemy.power("slumber"), 0, "归零 → 醒")
        self.assertEqual(enemy.stun_follow_up, "roll_out", "下一招固定滚撞")
        # ⚠️ 移除镀层发生在**被晕的那一回合**（真机是 `WakeUpMove` 里做的），
        # 不是醒来的瞬间 —— 所以这里要走一次出招。
        self.assertEqual(enemy.power("plating"), 6, "此时镀层还在")
        core._choose_intent(state, enemy)
        self.assertEqual(enemy.power("plating"), 0, "唤醒动作：移除镀层")
        self.assertEqual(enemy.intent.mid, "STUNNED_MOVE")

    def test_slumber_removes_plating_when_waking_by_countdown(self):
        """``SlumberingBeetle.WakeUpMove`` 里也有"移除镀层"。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("slumber", 1)
        enemy.add_power("plating", 6)
        power_rules.on_turn_end(enemy, state, [])
        self.assertEqual(enemy.power("slumber"), 0)
        self.assertEqual(enemy.power("plating"), 0)

    def test_stun_forces_the_follow_up_move_next_turn(self):
        """``nextMoveId`` 的兑现：**下一回合**固定走那一招，之后恢复状态机。"""
        from sts2_sim import core
        state = self.combat()
        enemy = state.enemies[0]
        enemy.power_flags["stunned"] = 1
        enemy.stun_follow_up = enemy.definition().moves[0].mid
        core._choose_intent(state, enemy)              # 被晕的这一回合：空招
        self.assertEqual(enemy.intent.mid, "STUNNED_MOVE")
        self.assertEqual(enemy.forced_move, enemy.definition().moves[0].mid)
        core._choose_intent(state, enemy)              # 下一回合：固定那一招
        self.assertEqual(enemy.intent.mid, enemy.definition().moves[0].mid)
        self.assertEqual(enemy.forced_move, "", "兑现一次就清空")
        core._choose_intent(state, enemy)              # 再下一回合：恢复状态机
        self.assertNotEqual(enemy.intent.mid, "STUNNED_MOVE")


class AutoPrePlayAndSlyTest(BuffPowerTestCase):
    """``MayhemPower``（AutoPrePlay 时机）与 ``MasterPlannerPower``（卡实例级 Sly）。

    两者都靠"时机/关键字到底落在哪一份状态上"：

    * `mayhem` 用的是 ``AfterAutoPrePlayPhaseEntered``（抽完牌、即将出牌），
      **不是**回合结束那个 ``AfterAutoPostPlayPhaseEntered``（`stampede` 用）；
    * `master_planner` 用 ``CardCmd.ApplyKeyword(card, Sly)`` 给**单张牌**加狡诈，
      所以读的那一侧也必须看实例关键字，否则加了等于没加。
    """

    def test_mayhem_autoplays_the_top_of_the_draw_pile(self):
        """⚠️ 时机在**抽完牌之后**（`AutoPrePlay`）——所以抽牌堆要先够抽。

        抽 5 张、还剩 3 张 → 自动打出顶上的 2 张（各 6 点），第 3 张留着。
        """
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("mayhem", 2)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        state.hand.clear()
        state.draw_pile = [core.CardInstance("strike_ironclad") for _ in range(8)]
        before = enemy.hp
        core.start_player_turn(state, [])
        self.assertEqual(before - enemy.hp, 12, "抽完之后自动打出两张打击")
        self.assertEqual(len(state.draw_pile), 1, "只打 Amount 张，其余留着")

    def test_mayhem_needs_the_power(self):
        """反向对照：没有这条能力时，抽完牌不会自动打任何东西。"""
        from sts2_sim import core
        state = self.combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        state.hand.clear()
        state.draw_pile = [core.CardInstance("strike_ironclad") for _ in range(8)]
        before = enemy.hp
        core.start_player_turn(state, [])
        self.assertEqual(before - enemy.hp, 0)
        self.assertEqual(len(state.draw_pile), 3, "没人自动打出 → 三张都还在")

    def test_master_planner_gives_sly_to_played_skills(self):
        from sts2_sim import core, keywords
        state = self.combat()
        state.player.add_power("master_planner", 1)
        card = core.CardInstance("defend_ironclad")
        state.hand.append(card)
        state.energy = 9
        from sts2_sim.core import Action, step
        step(state, Action("play_card", state.hand.index(card), 0))
        self.assertIn("Sly", card.keywords, "打出的技能牌获得狡诈（卡实例级）")
        self.assertTrue(keywords.is_sly_card(card))

    def test_master_planner_ignores_attacks(self):
        """反向对照：只给**技能牌**加（源码判 ``Type != Skill`` 直接返回）。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("master_planner", 1)
        card = core.CardInstance("strike_ironclad")
        state.hand.append(card)
        state.energy = 9
        from sts2_sim.core import Action, step
        step(state, Action("play_card", state.hand.index(card), 0))
        self.assertNotIn("Sly", card.keywords)

    def test_sly_instance_keyword_makes_the_discard_autoplay(self):
        """⭐ 实例关键字要**真的生效**：带 Sly 的技能牌被弃掉时自动打出。

        这条是"两侧必须一致"的检查 —— 只加关键字而读取侧只看卡牌定义，
        效果会**静默失效**（牌被弃掉就没了）。
        """
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("master_planner", 1)
        card = core.CardInstance("defend_ironclad")
        state.hand.append(card)
        state.energy = 9
        from sts2_sim.core import Action, step
        step(state, Action("play_card", state.hand.index(card), 0))
        self.assertIn("Sly", card.keywords)
        # 把它从弃牌堆"再弃一次"：真机 `CardCmd.DiscardAndDraw` 会对 Sly 牌自动打出
        state.player.block = 0
        state.hand.clear()
        state.discard.remove(card)
        state.hand.append(card)
        core._autoplay_sly(state, card, [])
        self.assertGreater(state.player.block, 0, "Sly 牌被弃掉时应自动打出（获得格挡）")


class ZeroCostPlayTest(BuffPowerTestCase):
    """`feral` / `one_for_all`：判据都是"**这一张实际花了多少能量**"。

    真机读的是 ``CardPlay.Resources.EnergyValue``（本次出牌实际花掉的钱，
    免费能力减到 0 也算），引擎记在 ``state.current_play_energy_spent``。
    两条都要配"不是 0 费就不生效"的对照 —— 只测 0 费那一半，
    "所有攻击牌都吃加成"这种错也能通过。
    """

    def _play(self, state, cid, target=0):
        from sts2_sim.core import Action, CardInstance, step
        card = CardInstance(cid)
        state.hand.append(card)
        state.energy = 9
        step(state, Action("play_card", state.hand.index(card), target))
        return card

    def test_feral_returns_zero_cost_attacks_to_hand(self):
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("feral", 1)
        card = self._play(state, "anger")             # 0 费攻击
        self.assertIn(card, state.hand, "0 费攻击牌回手牌")
        self.assertNotIn(card, state.discard)
        self.assertEqual(state.player.power_flags.get("feral_returned"), 1)

    def test_feral_ignores_costing_attacks(self):
        """反向对照：1 费攻击照常进弃牌堆。"""
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("feral", 1)
        card = self._play(state, "strike_ironclad")
        self.assertNotIn(card, state.hand)
        self.assertIn(card, state.discard)

    def test_feral_caps_per_turn(self):
        """每回合只有 ``Amount`` 次（源码 `zeroCostAttacksPlayed >= Amount → 不改`）。"""
        from sts2_sim.core import Action, step
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("feral", 1)
        first = self._play(state, "anger")
        self.assertIn(first, state.hand)
        # 再打同一张：这一回合的额度已经用完了
        state.energy = 9
        step(state, Action("play_card", state.hand.index(first), 0))
        self.assertIn(first, state.discard, "额度用完 → 正常进弃牌堆")

    def test_feral_resets_each_turn(self):
        from sts2_sim import core
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("feral", 1)
        self._play(state, "anger")
        self.assertEqual(state.player.power_flags["feral_returned"], 1)
        state.player.block = 0
        core.start_player_turn(state, [])
        self.assertEqual(state.player.power_flags["feral_returned"], 0, "回合开始归零")

    def test_feral_skips_x_cost_attacks(self):
        """X 费牌花的是"全部能量" —— 即使当前能量是 0，也**不是** 0 费牌。

        真机的判据是 ``resources.EnergyValue > 0 → 不改``；引擎里 X 费卡的
        `play_cost` 就是当前能量，所以能量为 0 时它也可能是 0 —— 这一条测试
        钉住"X 费卡按它自己的费用走"这个边界（源码 ``CostsX`` 在
        `OneForAllPower` 里是**显式排除**的）。
        """
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("feral", 1)
        state.energy = 3
        from sts2_sim.core import Action, CardInstance, step
        card = CardInstance("whirlwind")
        state.hand.append(card)
        step(state, Action("play_card", state.hand.index(card), 0))
        self.assertIn(card, state.discard, "X 费卡不参与回手牌")

    def test_one_for_all_adds_damage_only_to_zero_cost_attacks(self):
        from sts2_sim import content
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("one_for_all", 3)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        base = sum(e.amount for e in content.CARD_DB["anger"].effects
                   if e.op == "damage")
        before = enemy.hp
        self._play(state, "anger")                    # 0 费 → +3
        self.assertEqual(before - enemy.hp, base + 3)
        before = enemy.hp
        self._play(state, "strike_ironclad")          # 1 费 → 不加
        self.assertEqual(before - enemy.hp, 6)

    def test_one_for_all_skips_x_cost_attacks(self):
        state = self.combat(("nibbit", "nibbit"))
        state.player.add_power("one_for_all", 3)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        # whirlwind 的伤害是 X 段，这里只看"它有没有被额外加 3"
        from sts2_sim.core import Action, CardInstance, step
        card = CardInstance("whirlwind")
        state.hand.append(card)
        state.energy = 2
        before = enemy.hp
        step(state, Action("play_card", state.hand.index(card), 0))
        # X=2：两段各 5 点（值来自卡表）——不含 one_for_all 的 +3
        self.assertGreater(before - enemy.hp, 0, "X 费卡照常结算")
        self.assertNotEqual((before - enemy.hp) % 5, 3 % 5,
                            "每段都不该多出 3 点（X 费被显式排除）")


class StarCostTest(BuffPowerTestCase):
    """星费（摄政王资源）与挂在它上面的两个能力。

    以前引擎**根本没有星费**：23 张带星费的卡（`comet` 要 5 星）只要能量够就能打
    —— 门禁只看"效果能不能落地"，所以这是"静默变强"里最不显眼的一类。
    """

    def test_card_needs_enough_stars(self):
        from sts2_sim import core
        state = self.combat()
        card = core.CardInstance("comet")            # CanonicalStarCost => 5
        state.hand.append(card)
        state.energy = 9
        state.stars = 0
        self.assertEqual(core.star_cost(state, card), 5)
        self.assertNotIn(action_index(state, card), playable_hand(state),
                         "星不够 → 不该出现在合法动作里")
        state.stars = 5
        self.assertIn(action_index(state, card), playable_hand(state))

    def test_playing_spends_the_stars(self):
        from sts2_sim.core import Action, CardInstance, step
        state = self.combat()
        card = CardInstance("comet")
        state.hand.append(card)
        state.energy = 9
        state.stars = 7
        step(state, Action("play_card", state.hand.index(card), 0))
        self.assertEqual(state.stars, 2, "5 星被扣掉")

    def test_child_of_the_stars_gains_block_per_star(self):
        from sts2_sim.core import Action, CardInstance, step
        state = self.combat()
        state.player.add_power("child_of_the_stars", 3)
        state.player.block = 0
        card = CardInstance("comet")
        state.hand.append(card)
        state.energy = 9
        state.stars = 5
        step(state, Action("play_card", state.hand.index(card), 0))
        self.assertEqual(state.player.block, 15, "3 × 花掉的 5 星")

    def test_child_of_the_stars_block_is_unpowered(self):
        """``ValueProp.Unpowered``：敏捷/脆弱都不该改变它。"""
        from sts2_sim.core import Action, CardInstance, step
        state = self.combat()
        state.player.add_power("child_of_the_stars", 2)
        state.player.add_power("dexterity", 5)
        state.player.add_power("frail", 3)
        state.player.block = 0
        card = CardInstance("comet")
        state.hand.append(card)
        state.energy = 9
        state.stars = 5
        step(state, Action("play_card", state.hand.index(card), 0))
        self.assertEqual(state.player.block, 10)

    def test_black_hole_hits_when_stars_are_gained(self):
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("black_hole", 4)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        core.gain_stars(state, 2, [])
        self.assertEqual(100 - enemy.hp, 4, "获得星 → 全体 4 点 Unpowered 伤害")

    def test_black_hole_hits_after_a_star_card_finishes(self):
        """⭐ ``AfterCardPlayed`` + ``StarsSpent > 0``：**牌结算完**才炸。"""
        from sts2_sim import content
        from sts2_sim.core import Action, CardInstance, step
        state = self.combat()
        state.player.add_power("black_hole", 4)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        card = CardInstance("comet")
        state.hand.append(card)
        state.energy = 9
        state.stars = 5
        comet_damage = sum(e.amount for e in content.CARD_DB["comet"].effects
                           if e.op == "damage")
        before = enemy.hp
        step(state, Action("play_card", state.hand.index(card), 0))
        self.assertEqual(before - enemy.hp, comet_damage + 4,
                         "彗星的伤害 + 黑洞的 4 点")

    def test_black_hole_ignores_free_cards(self):
        """反向对照：没花星的牌打完不触发。"""
        from sts2_sim.core import Action, CardInstance, step
        state = self.combat()
        state.player.add_power("black_hole", 4)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        card = CardInstance("strike_ironclad")
        state.hand.append(card)
        state.energy = 9
        before = enemy.hp
        step(state, Action("play_card", state.hand.index(card), 0))
        self.assertEqual(before - enemy.hp, 6, "只有打击本身的 6 点")


def playable_hand(state):
    """合法动作里**可打出**的手牌下标（用来验 mask）。"""
    from sts2_sim import core
    return {a.hand_index for a in core.legal_actions(state) if a.kind == "play_card"}


def action_index(state, card):
    return state.hand.index(card)


class GeneratedCardPowersTest(BuffPowerTestCase):
    """第十九批：`calamity`（造攻击牌）/ `consuming_shadow`（激发最后一个球）/ `soulbound`（Soul 连锁）。

    三个都靠"**生成物落位**"与"**牌身份**"这类细节，所以每条都配了反向对照：
    非攻击牌不造牌、激发的是**最右边**那个球、以及 soulbound 的**防自激**。
    """

    def test_calamity_generates_attack_cards_after_an_attack(self):
        from sts2_sim import content
        state = self.combat()
        state.player.add_power("calamity", 2)
        state.hand.clear()
        self.play(state, "strike_ironclad")
        added = [c for c in state.hand if c.cid != "strike_ironclad"]
        self.assertEqual(len(added), 2, "打攻击牌 → 造 2 张")
        for card in added:
            self.assertEqual(content.CARD_DB[card.cid].card_type, "attack",
                             "只从**攻击牌**里抽")

    def test_calamity_ignores_skills(self):
        """反向对照：技能牌不触发（源码判 ``Type != Attack`` 直接返回）。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("calamity", 2)
        state.hand.clear()
        self.play(state, "defend_ironclad")
        self.assertEqual([c.cid for c in state.hand], [], "技能牌不造牌")
        _ = core

    def test_consuming_shadow_evokes_the_last_orb(self):
        from sts2_sim import orbs, powers as power_rules
        state = self.combat()
        state.player.add_power("consuming_shadow", 1)
        orbs.channel(state, "lightning", [])
        orbs.channel(state, "frost", [])
        self.assertEqual([o.oid for o in state.orbs], ["lightning", "frost"])
        state.player.block = 0
        power_rules.on_turn_end(state.player, state, [])
        self.assertEqual([o.oid for o in state.orbs], ["lightning"],
                         "激发的是**最后一个**（最右边）")
        self.assertEqual(state.player.block, 5, "冰球的被动生效了")

    def test_consuming_shadow_needs_orbs(self):
        """反向对照：队列为空时什么都不做（源码在循环**之前**判空）。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("consuming_shadow", 3)
        state.player.block = 0
        power_rules.on_turn_end(state.player, state, [])
        self.assertEqual(state.player.block, 0)

    def test_soulbound_adds_souls_when_the_applier_creates_one(self):
        from sts2_sim import core, powers as power_rules
        state = self.combat()
        enemy = state.enemies[0]
        power_rules.apply_power_to(state, enemy, "soulbound", 2, [],
                                   applier=state.player)
        state.draw_pile.clear()
        soul = core.CardInstance("soul")
        core._add_generated_card(state, soul, "hand")
        souls = [c for c in state.draw_pile if c.cid == "soul"]
        self.assertEqual(len(souls), 2, "施加者造出 Soul → 再塞 2 张进抽牌堆")
        self.assertNotIn(soul, state.draw_pile, "原来那张留在手牌，不被搬走")

    def test_soulbound_does_not_recurse_forever(self):
        """⭐ ``IsAddingSoul`` 防自激：塞进去的 Soul 自己也会触发钩子，必须挡住。"""
        from sts2_sim import core, powers as power_rules
        state = self.combat()
        enemy = state.enemies[0]
        power_rules.apply_power_to(state, enemy, "soulbound", 1, [],
                                   applier=state.player)
        state.draw_pile.clear()
        core._add_generated_card(state, core.CardInstance("soul"), "hand")
        self.assertEqual(len([c for c in state.draw_pile if c.cid == "soul"]), 1,
                         "只加 1 张 —— 不是无限递归")

    def test_soulbound_ignores_other_generated_cards(self):
        from sts2_sim import core, powers as power_rules
        state = self.combat()
        enemy = state.enemies[0]
        power_rules.apply_power_to(state, enemy, "soulbound", 2, [],
                                   applier=state.player)
        state.draw_pile.clear()
        core._add_generated_card(state, core.CardInstance("shiv"), "hand")
        self.assertEqual([c for c in state.draw_pile if c.cid == "soul"], [],
                         "只有 Soul 才连锁")


if __name__ == "__main__":
    unittest.main()
