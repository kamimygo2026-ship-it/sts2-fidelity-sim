"""充能球机制的回归测试（``docs/09`` L4 角色机制）。

数值全部抄自 ``MegaCrit.Sts2.Core.Models.Orbs.*``。这里锁住的是**三条容易
搞错、错了就静默偏掉**的语义：

1. 球的伤害/格挡带 ``ValueProp.Unpowered`` —— 不受力量/虚弱/**易伤**影响，
   但**可以被格挡**。按普通伤害走会让 Defect 的输出凭空变化。
2. ``ModifyOrbValue`` 才是 Focus 的作用点，而 Plasma 用裸值 ——
   "Plasma 不受 Focus 影响"是**源码事实**，不是平衡说明。
3. Dark 与 Glass 有**跨回合状态**：Dark 的 evoke 值逐回合累加，Glass 的
   passive 值逐回合衰减。当成无状态会让 Dark 永远 evoke 6、Glass 永远不减。
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"


def restore_builtin() -> None:
    from sts2_sim.content import load_builtin
    from sts2_sim.featurize import rebuild_vocab
    load_builtin()
    rebuild_vocab()


def setup_content():
    if not CONTENT.exists():
        raise unittest.SkipTest(f"缺内容目录 {CONTENT}")
    from sts2_sim.featurize import configure_content
    configure_content(str(CONTENT))
    from sts2_sim.content import STARTING_DECK
    return STARTING_DECK


class TestOrbBasics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self, enemy: str = "nibbit"):
        from sts2_sim.core import start_combat
        state = start_combat(self.deck, (enemy,), seed=5)
        state.player.hp = state.player.max_hp = 200
        state.hand = []
        return state

    def test_all_five_orbs_are_registered(self):
        from sts2_sim import orbs
        self.assertEqual(sorted(orbs.ORB_DEFS),
                         ["dark", "frost", "glass", "lightning", "plasma"])

    def test_channel_fills_slots(self):
        from sts2_sim import orbs
        state = self._combat()
        for _ in range(orbs.DEFAULT_ORB_SLOTS):
            orbs.channel(state, "lightning", [])
        self.assertEqual(len(state.orbs), orbs.DEFAULT_ORB_SLOTS)

    def test_channel_overflows_by_evoking_the_leftmost(self):
        """槽位满了要**先激发最左边那个**（FIFO），不是丢弃新球。

        搞反的话球的节奏完全变样：激发会打伤害，而丢弃不会。
        """
        from sts2_sim import orbs
        state = self._combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.block = 0
        for _ in range(3):
            orbs.channel(state, "lightning", [])
        before = enemy.hp
        orbs.channel(state, "frost", [])
        self.assertEqual(before - enemy.hp, 8, "最左边的闪电应被激发（8 点）")
        self.assertEqual([o.oid for o in state.orbs],
                         ["lightning", "lightning", "frost"])

    def test_evoke_removes_the_orb(self):
        from sts2_sim import orbs
        state = self._combat()
        orbs.channel(state, "frost", [])
        orbs.evoke(state, 0, [])
        self.assertEqual(state.orbs, [])


class TestOrbValues(unittest.TestCase):
    """逐个球的 Passive / Evoke 数值（对照源码）。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self):
        from sts2_sim.core import start_combat
        state = start_combat(self.deck, ("nibbit",), seed=5)
        state.player.hp = state.player.max_hp = 200
        state.hand = []
        return state

    def test_static_values_match_source(self):
        from sts2_sim import orbs
        expected = {"lightning": (3, 8), "frost": (2, 5), "plasma": (1, 2)}
        for oid, (passive, evoke) in expected.items():
            with self.subTest(orb=oid):
                state = self._combat()
                orbs.channel(state, oid, [])
                self.assertEqual(orbs.passive_value(state, state.orbs[0]), passive)
                self.assertEqual(orbs.evoke_value(state, state.orbs[0]), evoke)

    def test_dark_accumulates_its_evoke_value(self):
        """``DarkOrb``：``_evokeVal`` 初值 6，每次回合结束 ``+= PassiveVal``。"""
        from sts2_sim import orbs
        state = self._combat()
        orbs.channel(state, "dark", [])
        self.assertEqual(orbs.evoke_value(state, state.orbs[0]), 6)
        orbs.on_turn_end(state, [])
        self.assertEqual(orbs.evoke_value(state, state.orbs[0]), 12)
        orbs.on_turn_end(state, [])
        self.assertEqual(orbs.evoke_value(state, state.orbs[0]), 18)

    def test_dark_growth_scales_with_focus(self):
        """Dark 累加的是 ``PassiveVal``（吃 Focus 的那个值），不是裸 6。"""
        from sts2_sim import orbs
        state = self._combat()
        state.player.add_power("focus", 2)
        orbs.channel(state, "dark", [])
        orbs.on_turn_end(state, [])
        self.assertEqual(orbs.evoke_value(state, state.orbs[0]), 6 + 8)

    def test_negative_focus_clamps_orb_values_at_zero(self):
        """``FocusPower.ModifyOrbValue`` 是 ``Math.Max(value + Amount, 0m)``。

        ``AllowNegative => true`` 且 ``BiasedCognitionPower`` / ``TemporaryFocusPower``
        会施加**负**专注，所以下限这一钳是源码行为：闪电被动 3 点配 −5 专注，
        真机是 **0**，不是 **−2**（负伤害会一路带着往下走，谁也查不出来）。
        """
        from sts2_sim import orbs
        state = self._combat()
        state.player.add_power("focus", -5)
        orbs.channel(state, "lightning", [])
        self.assertEqual(orbs.passive_value(state, state.orbs[0]), 0, "钳到 0")
        state.player.add_power("focus", 5 + 2)          # 净 +2
        self.assertEqual(orbs.passive_value(state, state.orbs[0]), 3 + 2)

    def test_glass_decays_each_trigger(self):
        """``GlassOrb``：每次触发 ``_passiveVal = max(0, value - 1)``。"""
        from sts2_sim import orbs
        state = self._combat()
        orbs.channel(state, "glass", [])
        self.assertEqual(orbs.passive_value(state, state.orbs[0]), 4)
        orbs.on_turn_end(state, [])
        self.assertEqual(orbs.passive_value(state, state.orbs[0]), 3)
        for _ in range(6):
            orbs.on_turn_end(state, [])
        self.assertEqual(orbs.passive_value(state, state.orbs[0]), 0, "不该降到负数")

    def test_glass_evoke_is_double_passive(self):
        from sts2_sim import orbs
        state = self._combat()
        orbs.channel(state, "glass", [])
        self.assertEqual(orbs.evoke_value(state, state.orbs[0]), 8)


class TestFocusInteraction(unittest.TestCase):
    """``ModifyOrbValue`` 才是 Focus 的作用点。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self):
        from sts2_sim.core import start_combat
        state = start_combat(self.deck, ("nibbit",), seed=5)
        state.player.hp = state.player.max_hp = 200
        state.hand = []
        return state

    def test_focus_boosts_lightning_and_frost(self):
        from sts2_sim import orbs
        state = self._combat()
        state.player.add_power("focus", 2)
        for oid, passive, evoke in (("lightning", 5, 10), ("frost", 4, 7)):
            with self.subTest(orb=oid):
                state.orbs.clear()
                orbs.channel(state, oid, [])
                self.assertEqual(orbs.passive_value(state, state.orbs[0]), passive)
                self.assertEqual(orbs.evoke_value(state, state.orbs[0]), evoke)

    def test_plasma_ignores_focus(self):
        """``PlasmaOrb`` 用的是裸 ``1m``/``2m``，**不走** ``ModifyOrbValue``。

        给它加上 Focus 会让每回合凭空多出能量 —— 而能量是全局节奏的源头。
        """
        from sts2_sim import orbs
        state = self._combat()
        state.player.add_power("focus", 5)
        orbs.channel(state, "plasma", [])
        self.assertEqual(orbs.passive_value(state, state.orbs[0]), 1)
        self.assertEqual(orbs.evoke_value(state, state.orbs[0]), 2)


class TestOrbHooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self):
        from sts2_sim.core import start_combat
        state = start_combat(self.deck, ("nibbit",), seed=5)
        state.player.hp = state.player.max_hp = 200
        state.hand = []
        return state

    def test_frost_gives_block_at_turn_end(self):
        from sts2_sim import orbs
        state = self._combat()
        orbs.channel(state, "frost", [])
        state.player.block = 0
        orbs.on_turn_end(state, [])
        self.assertEqual(state.player.block, 2)

    def test_plasma_gives_energy_at_turn_start(self):
        """等离子挂在 ``AfterTurnStartOrbTrigger`` —— 是**回合开始**，不是结束。"""
        from sts2_sim import orbs
        state = self._combat()
        orbs.channel(state, "plasma", [])
        state.energy = 0
        orbs.on_turn_start(state, [])
        self.assertEqual(state.energy, 1)
        state.energy = 0
        orbs.on_turn_end(state, [])
        self.assertEqual(state.energy, 0, "回合结束不该给能量")

    def test_plasma_energy_arrives_through_a_real_turn(self):
        """走真实回合流程验证时机：能量重置**之后**才拿到那 1 点。"""
        from sts2_sim import orbs
        from sts2_sim.core import BASE_ENERGY, Action, step
        state = self._combat()
        orbs.channel(state, "plasma", [])
        step(state, Action(kind="end_turn"))
        self.assertEqual(state.energy, BASE_ENERGY + 1)


class TestUnpoweredDamage(unittest.TestCase):
    """球的伤害/格挡是 ``Unpowered``：可格挡，但不受力量/虚弱/易伤影响。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self):
        from sts2_sim.core import start_combat
        state = start_combat(self.deck, ("nibbit",), seed=5)
        state.player.hp = state.player.max_hp = 200
        state.hand = []
        return state

    def test_orb_damage_is_blockable(self):
        from sts2_sim import orbs
        state = self._combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.block = 50
        orbs.channel(state, "lightning", [])
        orbs.evoke(state, 0, [])
        self.assertEqual(enemy.hp, 100, "球伤害应被格挡完全吃掉")
        self.assertEqual(enemy.block, 42)

    def test_orb_damage_ignores_strength_and_vulnerable(self):
        """带 ``Unpowered``：力量与易伤都**不**参与。

        用普通伤害通道会给球的输出乘上 1.5 —— Defect 的伤害大半来自球，
        误差会直接反映成胜率差。
        """
        from sts2_sim import orbs
        state = self._combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.block = 0
        enemy.add_power("vulnerable", 3)
        state.player.add_power("strength", 5)
        orbs.channel(state, "lightning", [])
        orbs.evoke(state, 0, [])
        self.assertEqual(100 - enemy.hp, 8, "易伤/力量都不该改变球的伤害")

    def test_frost_block_ignores_dexterity_and_frail(self):
        """霜的格挡同样 ``Unpowered``：不吃敏捷，也不被脆弱打折。"""
        from sts2_sim import orbs
        state = self._combat()
        state.player.add_power("dexterity", 5)
        state.player.add_power("frail", 3)
        orbs.channel(state, "frost", [])
        state.player.block = 0
        orbs.on_turn_end(state, [])
        self.assertEqual(state.player.block, 2)


class TestOrbCardIntegration(unittest.TestCase):
    """充能球的卡在引擎里要真的采纳源码效果。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_orb_cards_are_adopted_from_source(self):
        from sts2_sim import content
        for cid in ("ball_lightning", "chill", "cold_snap", "fusion",
                    "capacitor", "dualcast", "zap"):
            with self.subTest(cid=cid):
                card = content.CARD_DB[cid]
                self.assertEqual(card.effect_source, "source",
                                 f"{cid} 应采纳源码效果")

    def test_channel_effect_carries_the_orb_id(self):
        from sts2_sim import content
        card = content.CARD_DB["zap"]
        channel = next(e for e in card.effects if e.op == "channel")
        self.assertEqual(channel.orb, "lightning")

    def test_random_orb_is_not_guessed(self):
        """`chaos` 引导**随机**球，泛型实参抽不出来 —— 不能猜成某种球。"""
        from sts2_sim import content
        card = content.CARD_DB["chaos"]
        self.assertNotEqual(card.effect_source, "source")
        self.assertTrue(card.effects_incomplete)

    def test_playing_zap_channels_a_lightning_orb(self):
        from sts2_sim.core import Action, CardInstance, start_combat, step
        from sts2_sim import orbs
        state = start_combat(self.deck, ("nibbit",), seed=5)
        state.hand = [CardInstance("zap")]
        state.energy = 5
        step(state, Action("play_card", 0, -1))
        self.assertEqual([o.oid for o in state.orbs], ["lightning"])

    def test_capacitor_adds_a_slot(self):
        """``Capacitor``（电容器）：``OrbCmd.AddSlots(owner, Repeat=2)`` → **+2 个槽**。

        ⚠️ 这条断言以前写的是 ``+1`` —— 那是**错的**，而且错在抽取器：
        `AddSlots` 的签名是 ``AddSlots(Player owner, int count)``，
        旧实现读的是**第 1 个**参数（``base.Owner``），永远抽不出量、退回 1。
        见 `docs/12` §3.2.0。升级 ``Repeat.UpgradeValueBy(1)`` → **+3**。
        """
        from sts2_sim.core import Action, CardInstance, start_combat, step
        from sts2_sim.content import CARD_DB
        state = start_combat(self.deck, ("nibbit",), seed=5)
        state.hand = [CardInstance("capacitor")]
        state.energy = 5
        before = state.orb_slots
        step(state, Action("play_card", 0, -1))
        self.assertEqual(state.orb_slots, before + 2)
        # 升级版再加一个（源码：`Repeat.UpgradeValueBy(1m)`）
        upgraded = start_combat(self.deck, ("nibbit",), seed=5)
        upgraded.hand = [CardInstance("capacitor", upgraded=True)]
        upgraded.energy = 5
        before = upgraded.orb_slots
        step(upgraded, Action("play_card", 0, -1))
        self.assertEqual(upgraded.orb_slots, before + 3)
        self.assertEqual(CARD_DB["capacitor"].effects[0].amount, 2)


class TestXCostCards(unittest.TestCase):
    """X 费卡（``docs/09`` L4）—— 语义全部抄自源码。

    ``CardEnergyCost.GetAmountToSpend``：X 费花的是**玩家当前全部能量**；
    ``CardEnergyCost.Canonical = (!CostsX) ? canonicalCost : 0``：规范费用强制为 0；
    ``HasEnoughResourcesFor`` 用 ``GetWithModifiers``（X 费提前返回 0）——
    所以 **0 能量也能打出**（X=0，效果为空）。

    ⚠️ 不能靠费用数字判断，两个方向都会错：普通 X 费卡写 ``base(0, …)``
    （会被当成**免费**），``Cascade`` 写 ``base(-1, …)``（会被当成**不可打出**）。
    """

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_x_cost_declared_in_source(self):
        """X 费的权威声明是源码里的 ``HasEnergyCostX => true``。

        ⚠️ ``stardust`` 不在其中：它用的是 ``HasStarCostX`` —— **星费** X
        （摄政王的资源），与能量 X 是两套机制，别混在一起。
        """
        from sts2_sim import content
        expected = {"cascade", "dirge", "eradicate", "heavenly_drill", "malaise",
                    "multi_cast", "skewer", "tempest", "volley", "whirlwind"}
        marked = {c.cid for c in content.CARD_DB.values() if c.is_x_cost}
        self.assertTrue(expected <= marked, f"漏标：{expected - marked}")
        self.assertNotIn("stardust", marked, "stardust 是星费 X，不是能量 X")

    def test_x_cost_canonical_is_zero(self):
        """``Canonical = (!CostsX) ? canonicalCost : 0``：X 费卡费用显示为 0。"""
        from sts2_sim import content
        for cid in ("cascade", "whirlwind", "tempest"):
            with self.subTest(cid=cid):
                self.assertEqual(content.CARD_DB[cid].cost, 0)

    def test_x_cost_spends_all_energy(self):
        from sts2_sim.core import Action, CardInstance, start_combat, step
        state = start_combat(self.deck, ("nibbit",), seed=1)
        state.hand = [CardInstance("whirlwind")]
        state.draw_pile, state.discard = [], []
        state.energy = 3
        step(state, Action("play_card", 0, 0))
        self.assertEqual(state.energy, 0)

    def test_whirlwind_hits_x_times(self):
        """``WithHitCount(ResolveEnergyXValue())``：打 **X** 次，不是 1 次。

        只认整数字面量会让次数默认成 1 —— 强度差 X 倍，而且看起来完全正常。
        """
        from sts2_sim.core import Action, CardInstance, start_combat, step
        state = start_combat(self.deck, ("nibbit",), seed=1)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        enemy.block = 0
        state.hand = [CardInstance("whirlwind")]
        state.draw_pile, state.discard = [], []
        state.energy = 3
        step(state, Action("play_card", 0, 0))
        self.assertEqual(200 - enemy.hp, 15, "5 点伤害 × X=3 次")

    def test_x_zero_is_still_playable(self):
        """0 能量也能打出 X 费牌（真机 `HasEnoughResourcesFor` 对 X 只看 0）。"""
        from sts2_sim.core import CardInstance, legal_actions, start_combat
        state = start_combat(self.deck, ("nibbit",), seed=1)
        state.hand = [CardInstance("whirlwind")]
        state.draw_pile, state.discard = [], []
        state.energy = 0
        self.assertTrue(any(a.kind == "play_card" for a in legal_actions(state)))

    def test_dualcast_evokes_twice_with_first_keeping_the_orb(self):
        """``DualCast`` 调用 ``EvokeNext`` **两次**，第一次 ``dequeue: false``。

        真机签名是 ``EvokeNext(choiceContext, player, dequeue)`` ——
        第 2 个参数是玩家、第 3 个是 dequeue，**都不是次数**。
        把参数当次数会把"激发两次"变成"激发一次"。
        """
        from sts2_sim import orbs
        from sts2_sim.core import Action, CardInstance, start_combat, step
        state = start_combat(self.deck, ("nibbit",), seed=1)
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        enemy.block = 0
        state.hand = [CardInstance("dualcast")]
        state.draw_pile, state.discard = [], []
        state.energy = 3
        orbs.channel(state, "lightning", [])
        step(state, Action("play_card", 0, 0))
        self.assertEqual(200 - enemy.hp, 16, "闪电激发 8 点 × 2 次")
        self.assertEqual(state.orbs, [], "第二次激发会把球移走")

    def test_tempest_channels_x_orbs(self):
        """``Tempest`` 的 X 在**循环上界**里（``for (i < numOfOrbs)``）。"""
        from sts2_sim import content
        channel = next(e for e in content.CARD_DB["tempest"].effects
                       if e.op == "channel")
        self.assertTrue(channel.times_x)

    def test_unresolvable_x_cards_are_flagged_not_guessed(self):
        """`cascade` 的效果建不出来 → 必须标残缺，不能猜。

        ⚠️ `multi_cast` **不在**此列：它的效果是
        ``evoke_next(times_x=True)``（循环里调用 X 次），已经能正确建模。
        """
        from sts2_sim import content
        card = content.CARD_DB["cascade"]
        self.assertNotEqual(card.effect_source, "source")
        self.assertTrue(card.effects_incomplete)

    def test_multi_cast_is_modelled_as_x_repeats(self):
        from sts2_sim import content
        card = content.CARD_DB["multi_cast"]
        self.assertEqual(card.effect_source, "source")
        evokes = [e for e in card.effects if e.op == "evoke_next"]
        self.assertTrue(evokes)
        self.assertTrue(all(e.times_x for e in evokes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
