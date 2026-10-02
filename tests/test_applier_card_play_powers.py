"""**施加者的出牌 → 在我身上结算**两个能力的源码级回归测试（`docs/12` §2.31）。

源码（两条**逐字同构**，只差"结算成什么"）：

``StranglePower`` / ``OblivionPower``（都是 ``InstanceType.InstancedPerApplier``）::

    BeforeCardPlayed(cardPlay):
        if (Applier?.Player == null) return;
        if (cardPlay.Card.Owner != Applier.Player) return;
        data.amountsForPlayedCards.Add(cardPlay.Card, base.Amount);
    AfterCardPlayed(cardPlay):
        if (data.amountsForPlayedCards.Remove(cardPlay.Card, out var value))
            // strangle: Damage(base.Owner, value, Unblockable|Unpowered, null, null)
            // oblivion: Apply<DoomPower>(base.Owner, value, base.Applier, null)

| 容易漏掉的那一条 | 漏掉的后果 |
|---|---|
| 记的是 **BeforeCardPlayed 那一刻**的层数 | 结算中途叠层会用到错的量 |
| 没记过的牌（能力贴上之前就开始结算的）**不结算** | 每张牌都触发，强度凭空翻倍 |
| ``Unblockable | Unpowered``（strangle） | 被格挡吃掉 / 被力量与易伤放大 |
| 移除时机：strangle 看**拥有者阵营**、oblivion 看**玩家阵营** | 一个提前消失、一个永久留下 |

本批还顺带修了一处引擎静默错误（§2.31.1）：**卡牌施加能力时不记施加者**
（``Combatant.powers_applier`` 空着），于是"读施加者"的能力经由卡牌施加时全部失效。
下面的 ``ApplierRecordedTest`` 就是钉这一条的。

这些断言只说明"引擎按源码执行"，不代表已与真机对拍（``docs/09`` §5）。
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


class _Case(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, enemies=("nibbit",), seed: int = 5):
        from sts2_sim import core
        state = core.start_combat(self.deck, tuple(enemies), seed=seed)
        state.hand = []
        state.draw_pile = []
        state.discard = []
        state.energy = 9
        state.player.powers.clear()
        state.player.block = 0
        for enemy in state.enemies:
            enemy.hp = enemy.max_hp = 100
            enemy.block = 0
            enemy.powers.clear()
        return state

    def play(self, state, cid: str, target: int = 0):
        from sts2_sim import core
        card = core.CardInstance(cid)
        state.hand.append(card)
        state.energy = 9
        return core.step(state, core.Action("play_card", state.hand.index(card),
                                            target))


class StrangleTest(_Case):
    """``StranglePower``：施加者每打一张牌 → 拥有者掉 ``Amount`` 点（不可格挡/无修正）。"""

    def test_card_applies_8_damage_and_two_layers(self):
        state = self.combat()
        enemy = state.enemies[0]
        hp = enemy.hp
        self.play(state, "strangle")
        self.assertEqual(hp - enemy.hp, 8, "卡本身的 8 点伤害")
        self.assertEqual(enemy.power("strangle"), 2)

    def test_every_card_the_applier_plays_costs_two_hp(self):
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "strangle")
        for cid in ("defend_ironclad", "strike_ironclad"):
            with self.subTest(cid=cid):
                hp = enemy.hp
                self.play(state, cid)
                # 攻击牌自己还有伤害，所以只断言"至少掉了 2 点勒杀伤害"
                self.assertGreaterEqual(hp - enemy.hp, 2)

    def test_skill_card_costs_exactly_two(self):
        """技能牌没有伤害，所以能精确断言勒杀那 2 点。"""
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "strangle")
        hp = enemy.hp
        self.play(state, "defend_ironclad")
        self.assertEqual(hp - enemy.hp, 2)
        self.assertEqual(state.player.block, 5, "防御牌照常给格挡")

    def test_damage_ignores_block(self):
        """``Unblockable``：敌人有多少格挡都拦不住勒杀。"""
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "strangle")
        enemy.block = 50
        hp = enemy.hp
        self.play(state, "defend_ironclad")
        self.assertEqual(hp - enemy.hp, 2)
        self.assertEqual(enemy.block, 50, "不可格挡的伤害不吃格挡")

    def test_damage_ignores_strength_and_vulnerable(self):
        """``Unpowered``：力量 / 易伤都改不动它。"""
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "strangle")
        state.player.add_power("strength", 5)
        enemy.add_power("vulnerable", 3)
        hp = enemy.hp
        self.play(state, "defend_ironclad")
        self.assertEqual(hp - enemy.hp, 2, "既不加力量也不吃易伤")

    def test_stacking_uses_the_recorded_amount(self):
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "strangle")
        self.play(state, "strangle")
        self.assertEqual(enemy.power("strangle"), 4)
        hp = enemy.hp
        self.play(state, "defend_ironclad")
        self.assertEqual(hp - enemy.hp, 4)

    def test_removed_at_the_owners_side_turn_end(self):
        """``participants.Contains(Owner)``：拥有者是敌人 → **敌方**回合结束移除。"""
        from sts2_sim import core
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "strangle")
        core.tick_powers(state.player, state, [])
        self.assertIn("strangle", enemy.powers, "玩家阵营结束不移除")
        core._end_enemy_side_turn(state, [])
        self.assertNotIn("strangle", enemy.powers)

    def test_only_the_appliers_own_cards_trigger_it(self):
        """源码条件 ``cardPlay.Card.Owner != Applier.Player`` → 直接跳过。

        钉住这一条：把"施加强的人"判丢了，"谁贴的"就无所谓了 ——
        敌人贴在我们身上的勒杀会因为我们出牌而反过来扣我们的血。
        """
        from sts2_sim import powers
        state = self.combat()
        enemy = state.enemies[0]
        powers.apply_power_to(state, state.player, "strangle", 3, [],
                              applier=enemy)
        self.assertEqual(state.player.power("strangle"), 3)
        hp = state.player.hp
        # ⚠️ 用**不加格挡**的牌：`defend_ironclad` 会给 5 点格挡，
        # 而"勒杀伤害变成普通伤害"这种变异正好会被那 5 点格挡吃掉 ——
        # 于是这条测试就测不出"施加者闸门被删掉"了（实测踩过）。
        self.play(state, "strike_ironclad")
        self.assertEqual(state.player.hp, hp, "敌人贴的勒杀：我们的牌不触发")


class OblivionTest(_Case):
    """``OblivionPower``：施加者每打一张牌 → 给拥有者上等量末日。"""

    def test_card_applies_three_layers(self):
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "oblivion")
        self.assertEqual(enemy.power("oblivion"), 3)
        self.assertEqual(enemy.power("doom"), 0, "卡本身不上末日")

    def test_each_card_gives_three_doom(self):
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "oblivion")
        self.play(state, "defend_ironclad")
        self.assertEqual(enemy.power("doom"), 3)
        self.play(state, "defend_ironclad")
        self.assertEqual(enemy.power("doom"), 6)

    def test_doom_keeps_the_original_applier(self):
        """``Apply<DoomPower>(..., base.Applier, null)``：施加者是**原施加者**。"""
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "oblivion")
        self.play(state, "defend_ironclad")
        self.assertIs(enemy.powers_applier.get("doom"), state.player)

    def test_removed_at_the_player_side_turn_end(self):
        """条件是 ``side == CombatSide.Player`` —— 与同族的 strangle **相反**。"""
        from sts2_sim import core
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "oblivion")
        core._end_enemy_side_turn(state, [])
        self.assertIn("oblivion", enemy.powers, "敌方阵营结束不移除")
        core.step(state, core.Action("end_turn"))
        self.assertNotIn("oblivion", state.enemies[0].powers)


class ApplierRecordedTest(_Case):
    """卡牌施加能力时必须记下**施加者**（`docs/12` §2.31.1 修掉的静默错误）。"""

    def test_card_applied_power_records_the_player_as_applier(self):
        from sts2_sim import core, powers
        state = self.combat()
        enemy = state.enemies[0]
        powers.apply_power_to(state, enemy, "shrink", 2, [], applier=state.player)
        self.assertIs(enemy.powers_applier.get("shrink"), state.player,
                      "apply_power_to 这条路一直是记的（对照组）")
        enemy.powers.clear()
        enemy.powers_applier.clear()
        self.play(state, "strangle")
        self.assertIs(enemy.powers_applier.get("strangle"), state.player,
                      "卡牌路径也必须记（曾经是空的）")

    def test_applier_death_ends_the_power_from_the_card_path(self):
        """``ConstrictPower.after_death``：`creature == base.Applier` → 移除。

        卡牌路径不记施加者时，这条**永远不会触发**（能力永久留下）。
        这里用 ``strangle``（同一份施加者数据）确认字段真的写进去了。
        """
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "strangle")
        applier = enemy.powers_applier.get("strangle")
        self.assertIsNotNone(applier)
        self.assertIs(applier, state.player)


class AdmissionTest(_Case):
    def test_the_two_cards_are_admitted(self):
        from sts2_sim import eligibility
        for cid in ("strangle", "oblivion"):
            with self.subTest(cid=cid):
                admission = eligibility.card_admission(cid)
                self.assertTrue(admission.admitted, f"{cid} 仍在拒：{admission.reasons()}")


if __name__ == "__main__":
    unittest.main()
