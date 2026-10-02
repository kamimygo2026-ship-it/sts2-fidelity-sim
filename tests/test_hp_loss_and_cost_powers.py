"""**掉血阶段 / 战斗内加价 / 掉血处决**三条能力的源码级回归测试（`docs/12` §2.30）。

| 能力 | 源码 | 容易漏掉的那一条 | 漏掉的后果 |
|---|---|---|---|
| ``buffer`` | ``BufferPower.ModifyHpLostAfterOstyLate`` | 晚段压到 **0**（盖过无形的"压到 1"），且**每抵消一次减 1 层** | 只压不递减 = 永久免疫（静默变强）；把递减挂进纯查询 = 被**空放** |
| ``borrowed_time`` | ``BorrowedTimePower.TryModifyEnergyCostInCombat`` | 加价与减费**同一阶段**；回合结束**整条移除** | 忘了移除 = 加价永久（静默变强） |
| ``the_gambit`` | ``TheGambitPower.AfterDamageReceived`` | 三个条件：被打的是自己、**有效攻击**、**UnblockedDamage > 0** | 只看"被打了" → 格挡流随手自尽；不判 powered → 中毒直接打死 |

同时钉住本批顺带修掉的两条"注释与代码不一致"：

* ``deal_raw_damage`` 的 docstring 一直写着"``IsPoweredAttack()`` 为假"，但调用时
  **漏了** ``powered=False``，默认值把它变成 True（`docs/12` §2.30.1）；
* ``play_cost`` 一直写着"X 费卡参与腐败归零——源码里根本没判 ``CostsX``"，
  而真机 ``CardEnergyCost.GetWithModifiers`` 在 ``_base < 0`` 与 ``CostsX``
  两处**提前返回**，X 费卡两个阶段都不参与（§2.30.2）。写 `borrowed_time` 时
  这条错会立刻变成可见故障：X 费卡在引擎里是"当前全部能量"，再加 1 就永远打不出来。

这些断言只说明"引擎按源码执行"，不代表已与真机对拍（``docs/09`` §5）。
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"

#: 本批新实现的能力引用的卡（能力做完 → 卡应放行）。
BATCH_CARDS = ("buffer", "borrowed_time", "the_gambit")


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
        state.player.hp = state.player.max_hp
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

    def hit(self, state, amount: int, **kwargs):
        """让第一只怪对玩家造成一次伤害（走 ``deal_damage``）。"""
        from sts2_sim import core
        events: list[str] = []
        core.deal_damage(state, state.enemies[0], state.player, amount, events,
                         **kwargs)
        return events


class BufferTest(_Case):
    """``BufferPower``：掉血压到 0，每抵消一次减 1 层。"""

    def test_prevents_one_hp_loss_and_is_consumed(self):
        state = self.combat()
        self.play(state, "buffer")
        self.assertEqual(state.player.power("buffer"), 1)
        hp = state.player.hp
        self.hit(state, 9, unblockable=True)
        self.assertEqual(state.player.hp, hp, "掉血被完全免掉")
        self.assertNotIn("buffer", state.player.powers, "抵消一次后 0 层 → 移除")
        self.hit(state, 9, unblockable=True)
        self.assertEqual(state.player.hp, hp - 9, "缓冲用光后掉血照常")

    def test_block_is_still_consumed(self):
        """缓冲管的是**掉血**那一段：格挡该扣还是要扣。"""
        state = self.combat()
        self.play(state, "buffer")
        state.player.block = 30
        hp = state.player.hp
        self.hit(state, 40)
        self.assertEqual(state.player.block, 0, "30 点格挡被吃光")
        self.assertEqual(state.player.hp, hp, "剩下 10 点掉血由缓冲免掉")

    def test_not_consumed_when_the_hit_is_fully_blocked(self):
        """掉血本来就是 0（被完全格挡）→ 缓冲**不该**被空放掉。"""
        state = self.combat()
        self.play(state, "buffer")
        state.player.block = 50
        self.hit(state, 40)
        self.assertEqual(state.player.power("buffer"), 1)
        self.assertEqual(state.player.block, 10)

    def test_buffer_wins_over_intangible(self):
        """缓冲是**晚段**（``…AfterOstyLate``），结果覆盖无形的"压到 1"。"""
        state = self.combat()
        self.play(state, "buffer")
        state.player.add_power("intangible", 2)
        hp = state.player.hp
        self.hit(state, 10, unblockable=True)
        self.assertEqual(state.player.hp, hp, "0 点掉血（不是无形的 1 点）")
        self.assertNotIn("buffer", state.player.powers, "被消耗的是缓冲")
        self.assertEqual(state.player.power("intangible"), 2,
                         "无形的层数由回合末 duration 递减，不在这里减")

    def test_lucky_tonic_potion_now_really_prevents_damage(self):
        """药水侧的同一条路：`lucky_tonic` 曾经"能喝但什么都不发生"。"""
        from sts2_sim import core
        state = self.combat()
        core.use_potion(state, "lucky_tonic")
        self.assertEqual(state.player.power("buffer"), 1)
        hp = state.player.hp
        self.hit(state, 12, unblockable=True)
        self.assertEqual(state.player.hp, hp)


class BorrowedTimeTest(_Case):
    """``BorrowedTimePower``：所有牌 +Amount，回合结束整条移除。"""

    def test_adds_to_every_card_cost(self):
        from sts2_sim import core
        state = self.combat()
        self.play(state, "borrowed_time")
        self.assertEqual(state.player.power("borrowed_time"), 1)
        self.assertEqual(core.play_cost(state, core.CardInstance("strike_ironclad")), 2)
        self.assertEqual(core.play_cost(state, core.CardInstance("defend_ironclad")), 2,
                         "不判牌型：技能牌同样 +1")

    def test_stacks(self):
        from sts2_sim import core
        state = self.combat()
        self.play(state, "borrowed_time")
        self.play(state, "borrowed_time")
        self.assertEqual(core.play_cost(state, core.CardInstance("strike_ironclad")), 3)

    def test_removed_at_own_side_turn_end_only(self):
        from sts2_sim import core
        state = self.combat()
        self.play(state, "borrowed_time")
        core._end_enemy_side_turn(state, [])
        self.assertIn("borrowed_time", state.player.powers,
                      "敌方阵营回合结束不移除（源码判 participants.Contains(Owner)）")
        core.tick_powers(state.player, state, [])
        self.assertNotIn("borrowed_time", state.player.powers)
        self.assertEqual(core.play_cost(state, core.CardInstance("strike_ironclad")), 1)

    def test_x_cost_cards_are_untouched(self):
        """X 费卡两个阶段都不参与（``CardEnergyCost.GetWithModifiers`` 提前返回）。

        ⚠️ 这条必须钉住：引擎里 X 费卡的 ``effective_cost`` 是"当前全部能量"，
        加价 1 点就等于**永远打不出来**（X 费卡在本批之前会被 `corruption` 错误归零）。
        """
        from sts2_sim import core, content
        x_cards = [cid for cid, definition in content.CARD_DB.items()
                   if definition.is_x_cost and not definition.effects_incomplete]
        self.assertTrue(x_cards, "内容里应当有可用的 X 费卡")
        state = self.combat()
        self.play(state, "borrowed_time")
        state.energy = 3
        for cid in x_cards:
            with self.subTest(cid=cid):
                self.assertEqual(core.play_cost(state, core.CardInstance(cid)), 3,
                                 f"{cid} 是 X 费卡：费用仍应是全部能量")


class XCostGuardTest(_Case):
    """X 费卡不吃任何一个战斗内费用钩子（`docs/12` §2.30.2）。"""

    def test_corruption_does_not_zero_x_cost_skills(self):
        from sts2_sim import core, content
        x_skills = [cid for cid, definition in content.CARD_DB.items()
                    if definition.is_x_cost and definition.card_type == "skill"
                    and not definition.effects_incomplete]
        self.assertTrue(x_skills, "内容里应当有可用的 X 费技能牌")
        state = self.combat()
        state.player.add_power("corruption", 1)
        state.energy = 4
        for cid in x_skills:
            with self.subTest(cid=cid):
                self.assertEqual(core.play_cost(state, core.CardInstance(cid)), 4,
                                 f"{cid} 是 X 费卡：腐败**不**归零、按全部能量算")

    def test_corruption_still_zeroes_normal_skills(self):
        """反向钉：普通技能牌照旧被腐败归零（别把闸门开成"谁都不改"）。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("corruption", 1)
        self.assertEqual(core.play_cost(state, core.CardInstance("defend_ironclad")), 0)


class TheGambitTest(_Case):
    """``TheGambitPower``：被打掉血 → 处决拥有者。"""

    def test_fully_blocked_hit_does_not_trigger(self):
        state = self.combat()
        self.play(state, "the_gambit")
        self.assertEqual(state.player.block, 50, "0 费拿 50 格挡")
        self.hit(state, 10)
        self.assertTrue(state.player.alive(), "被完全格挡 = UnblockedDamage 0 → 不触发")
        self.assertIn("the_gambit", state.player.powers)

    def test_penetrating_powered_hit_kills_the_owner(self):
        state = self.combat()
        self.play(state, "the_gambit")
        self.hit(state, 60)
        self.assertFalse(state.player.alive(), "掉血了 → 处决拥有者")
        self.assertNotIn("the_gambit", state.player.powers, "同时移除自己")

    def test_unpowered_hp_loss_does_not_trigger(self):
        """中毒 / 自伤不是**有效攻击**：不该触发"掉血即死"。

        ⚠️ 这条同时钉住本批修掉的一处静默错误：``deal_raw_damage`` 的 docstring
        写着 ``IsPoweredAttack()`` 为假，但调用时漏传了 ``powered=False``，
        默认值把它变成 True —— 于是中毒会直接把带孤注一掷的玩家打死。
        """
        from sts2_sim import core
        state = self.combat()
        self.play(state, "the_gambit")
        events: list[str] = []
        core.deal_raw_damage(state, state.player, 10, events, "中毒")
        self.assertTrue(state.player.alive(), "中毒是 Unpowered，不该触发孤注一掷")
        self.assertIn("the_gambit", state.player.powers)

    def test_thorns_does_not_retaliate_on_poison(self):
        """同一处修正的第二个观察点：荆棘只反**有效攻击**。"""
        from sts2_sim import core
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("thorns", 3)
        hp = enemy.hp
        core.deal_raw_damage(state, enemy, 10, [], "中毒")
        self.assertEqual(enemy.hp, hp - 10, "中毒照常掉血")
        self.assertEqual(state.player.hp, state.player.max_hp,
                         "但荆棘**不**反伤（powered=False）")


class AdmissionTest(_Case):
    def test_the_three_cards_are_admitted(self):
        from sts2_sim import eligibility
        for cid in BATCH_CARDS:
            with self.subTest(cid=cid):
                admission = eligibility.card_admission(cid)
                self.assertTrue(admission.admitted, f"{cid} 仍在拒：{admission.reasons()}")


if __name__ == "__main__":
    unittest.main()
