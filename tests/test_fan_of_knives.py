"""刀扇必须连同小刀的目标消费者交付，不能只注册一个无效标记。

源码：FanOfKnives.OnPlay/OnUpgrade、FanOfKnivesPower、Shiv.TargetType/OnPlay。
这里验证源码用例、动作与恢复契约，不代表已完成真机对拍。
"""

import unittest
from pathlib import Path

from sts2_sim import core, content, eligibility, featurize
from sts2_sim.observe import observe


class TestFanOfKnives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        featurize.configure_content(str(Path(__file__).resolve().parents[1]
                                       / "data/content/repo"))
        cls.addClassCleanup(cls._restore_content)

    @staticmethod
    def _restore_content():
        content.load_builtin()
        featurize.rebuild_vocab()

    def combat(self):
        state = core.start_combat(content.STARTING_DECK, ("nibbit", "nibbit"), seed=7)
        state.hand = []
        state.draw_pile = []
        state.discard = []
        state.energy = 9
        state.player.powers.clear()
        for enemy in state.enemies:
            enemy.hp = enemy.max_hp = 100
            enemy.block = 0
            enemy.powers.clear()
        return state

    def test_admission_requires_the_marker_and_its_consumer(self):
        """FanOfKnives.OnPlay 依赖能力与 Shiv；未知的其它能力不能随之放行。"""
        self.assertTrue(eligibility.card_admission("fan_of_knives").admitted)
        self.assertTrue(eligibility.card_admission("shiv").admitted)
        # ⭐ 这条原来断言 ``parry`` **被拒**（当时它只是一条没有消费者的登记）。
        # W03（`docs/12` §2.44）给 `SovereignBlade` 接上了消费者 ——
        # `CalculatedBlockVar.WithMultiplier(... GetOwnerParryAmount ...)`，
        # 于是它现在**应该**放行。这正是本测试的本意：
        # "标记能力 + 它的消费者"一起实现才放行。
        self.assertTrue(eligibility.card_admission("parry").admitted)
        # ⚠️ 这条要拿**仍然缺能力**的卡来钉。原先是 ``anticipate``，它属于临时属性族
        # （``TemporaryDexterityPower``）；那一族在 `docs/12` §2.28 被归并实现后
        # ``anticipate`` 已经放行，继续断言 False 就变成"测能力没做"的反向断言了。
        # 换成 ``beacon_of_hope``（能力仍只在源码里覆写 ``Type``，引擎没实现），
        # 保持测试本意：能力没实现 → 整张卡被拒。
        self.assertFalse(eligibility.card_admission("beacon_of_hope").admitted)

    def test_card_generates_four_or_five_shivs_after_applying_power(self):
        """FanOfKnives：Shivs=4，升级 +1；生成的小刀本身不升级。"""
        for upgraded, count in ((False, 4), (True, 5)):
            with self.subTest(upgraded=upgraded):
                state = self.combat()
                state.hand = [core.CardInstance("fan_of_knives", upgraded=upgraded)]
                core.step(state, core.Action("play_card", 0, -1))
                self.assertEqual(state.player.power("fan_of_knives"), 1)
                self.assertEqual([(c.cid, c.upgraded) for c in state.hand],
                                 [("shiv", False)] * count)
                self.assertEqual(state.energy, 7)
                self.assertEqual(len({c.uid for c in state.hand}), count)
                self.assertTrue(all(a.target == -1 for a in core.legal_actions(state)
                                    if a.kind == "play_card"))

    def test_shiv_remains_single_target_without_power(self):
        """Shiv.HasFanOfKnives=false：目标必须选择，基础伤害仅落在该敌人。"""
        state = self.combat()
        state.hand = [core.CardInstance("shiv")]
        plays = [a for a in core.legal_actions(state) if a.kind == "play_card"]
        self.assertEqual([a.target for a in plays], [0, 1])
        core.step(state, plays[1])
        self.assertEqual([e.hp for e in state.enemies], [100, 96])

    def test_shiv_hits_all_enemies_and_preserves_upgrade_and_exhaust(self):
        """Shiv.OnPlay/OnUpgrade：群攻仍是 4/6 点，关键字仍为 Exhaust。"""
        for upgraded, damage in ((False, 4), (True, 6)):
            with self.subTest(upgraded=upgraded):
                state = self.combat()
                state.player.add_power("fan_of_knives", 1)
                card = core.CardInstance("shiv", upgraded=upgraded)
                state.hand = [card]
                plays = [a for a in core.legal_actions(state) if a.kind == "play_card"]
                self.assertEqual(plays, [core.Action("play_card", 0, -1)])
                core.step(state, plays[0])
                self.assertEqual([e.hp for e in state.enemies], [100-damage] * 2)
                self.assertIn(card, state.exhaust)
                self.assertEqual(content.CARD_DB["shiv"].target, "enemy")
                self.assertEqual(content.CARD_DB["shiv"].effects[0].target, "enemy")

    def test_power_is_owner_local_and_does_not_change_other_attacks(self):
        """Shiv.HasFanOfKnives 只看拥有者；不存在按所有 Attack 改目标的规则。"""
        state = self.combat()
        state.enemies[0].add_power("fan_of_knives", 1)
        state.hand = [core.CardInstance("shiv")]
        self.assertEqual(len([a for a in core.legal_actions(state)
                              if a.kind == "play_card"]), 2)
        state.player.add_power("fan_of_knives", 1)
        state.hand = [core.CardInstance("strike_ironclad")]
        self.assertEqual([a.target for a in core.legal_actions(state)
                          if a.kind == "play_card"], [0, 1])

    def test_autoplay_does_not_roll_a_target_for_area_shiv(self):
        """CardCmd.AutoPlay 仅 AnyEnemy 随机取目标；群攻不能额外消费 combat_targets。"""
        state = self.combat()
        state.player.add_power("fan_of_knives", 1)
        state.hand = [core.CardInstance("shiv")]
        before = state.hidden.rng.streams["combat_targets"].getstate()
        self.assertTrue(core.autoplay_card(state, state.hand[0], []))
        self.assertEqual([e.hp for e in state.enemies], [96, 96])
        self.assertEqual(state.hidden.rng.streams["combat_targets"].getstate(), before)

    def test_area_shiv_preserves_repeat_and_card_damage_modifiers(self):
        """Shiv.FromCard + OneTwoPunch/Accuracy：两次群攻均保留卡牌来源和加伤。"""
        state = self.combat()
        for pid, amount in (("fan_of_knives", 1), ("one_two_punch", 1),
                            ("accuracy", 3), ("strength", 2)):
            state.player.add_power(pid, amount)
        state.hand = [core.CardInstance("shiv")]
        core.step(state, core.Action("play_card", 0, -1))
        self.assertEqual([e.hp for e in state.enemies], [82, 82])
        self.assertEqual(len(state.exhaust), 1)

    def test_area_shiv_skips_reviving_enemies(self):
        """Shiv.TargetingAllOpponents + CreatureCmd.Damage：不可命中对象不能受伤。"""
        state = self.combat()
        state.player.add_power("fan_of_knives", 1)
        state.enemies[0].hp = 0
        state.enemies[0].reviving = True
        state.hand = [core.CardInstance("shiv")]
        core.step(state, core.Action("play_card", 0, -1))
        self.assertEqual([e.hp for e in state.enemies], [0, 96])
        self.assertTrue(state.enemies[0].reviving)

    def test_full_hand_overflow_keeps_all_generated_shivs(self):
        """FanOfKnives.OnPlay + CardPileCmd.Add：满手牌时其余生成牌进入弃牌堆。"""
        state = self.combat()
        state.hand = [core.CardInstance("fan_of_knives")]
        state.hand += [core.CardInstance("defend_ironclad") for _ in range(9)]
        core.step(state, core.Action("play_card", 0, -1))
        self.assertEqual(sum(c.cid == "shiv" for c in state.hand), 1)
        self.assertEqual(sum(c.cid == "shiv" for c in state.discard), 3)

    def test_observation_and_encoding_include_power_and_area_action(self):
        """公开的刀扇标记和无单体目标的动作必须同时可供策略读取。"""
        state = self.combat()
        state.player.add_power("fan_of_knives", 1)
        state.hand = [core.CardInstance("shiv")]
        obs = observe(state)
        actions = core.legal_actions(state)
        encoded = featurize.encode_observation(obs, actions)
        self.assertIn(("fan_of_knives", 1), obs.player_powers)
        power_entities = encoded["token_entity"][
            encoded["token_type"] == featurize.TOKEN_POWER]
        self.assertIn(featurize.POWER_VOCAB["fan_of_knives"], power_entities)
        self.assertEqual(actions[0], core.Action("play_card", 0, -1))
        self.assertEqual(encoded["act_target"][0], -1)
        self.assertTrue(obs.hand[0].playable)

    def test_snapshot_restore_and_power_removal_restore_target_semantics(self):
        """SL 必须恢复标记；移除标记后小刀立即恢复单体，不污染静态定义。"""
        state = self.combat()
        state.player.add_power("fan_of_knives", 1)
        state.hand = [core.CardInstance("shiv")]
        saved = core.snapshot(state)
        restored = core.restore(saved)
        result = core.step(state, core.Action("play_card", 0, -1))
        replay = core.step(restored, core.Action("play_card", 0, -1))
        self.assertEqual(observe(state), observe(restored))
        self.assertEqual(result.events, replay.events)
        without_power = core.restore(saved)
        without_power.player.add_power("fan_of_knives", -1)
        self.assertEqual([a.target for a in core.legal_actions(without_power)
                          if a.kind == "play_card"], [0, 1])
