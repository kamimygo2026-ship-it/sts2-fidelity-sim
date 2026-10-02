"""KnowledgeDemon 的四种「知识诅咒」能力回归（``docs/17`` P01–P04，批次 W01）。

四个能力都由 ``KnowledgeDemon.CurseOfKnowledgeMove`` 生成的三组二选一卡牌施加
（``KnowledgeDemon.cs:36-52``），所以它们共用一条入口链：

======================  ==========================================================
能力                     本体的源码依据
======================  ==========================================================
``MindRot``             ``MindRotPower.ModifyHandDraw``（``MindRotPower.cs:14-21``）
``WasteAway``           ``WasteAwayPower.ModifyMaxEnergy``（``WasteAwayPower.cs:16-23``）
``Sloth``               ``SlothPower.ShouldPlay/BeforeCardPlayed/BeforeSideTurnStart``
``Disintegration``      ``DisintegrationPower.AfterSideTurnEndLate``
======================  ==========================================================

本文件钉的是**能力本体**（``docs/17`` §1.1 的 W01 前半）：抽牌链、能量链、
出牌上限与回合末 Late 自伤。KnowledgeDemon 三轮二选一的**入口**属于同一批的后半，
需要 S02 的续执行帧，本文件不覆盖 —— 它仍是已知缺口。

⚠️ 测试只走 ``core`` 的公开流程（``start_player_turn`` / ``step`` / ``legal_actions`` /
钩子入口）。直接改数值就等于绕过被验证的那条路径。
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


class CurseTestCase(unittest.TestCase):
    """公共脚手架：一副起手牌 + 一只怪。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, enemies=("nibbit",), seed: int = 5):
        from sts2_sim.core import start_combat
        return start_combat(self.deck, tuple(enemies), seed=seed)

    def draw_turn(self, state, powers=()):
        """把开局手牌塞回抽牌堆再开一个新回合，返回这一回合抽到的张数。

        ⚠️ 不能只 ``hand.clear()``：那等于把那 5 张牌从模拟里抹掉，
        抽牌堆只剩 5 张 → 想抽 6 张也只抽得到 5。另外手牌上限是 10，
        留着开局手牌会让多抽的部分被截断（与 ``test_buff_powers`` 同一套路）。
        """
        from sts2_sim import core
        state.draw_pile.extend(state.hand)
        state.hand.clear()
        for pid, amount in powers:
            state.player.add_power(pid, amount)
        core.start_player_turn(state, [])
        return len(state.hand)

    def hold(self, state, cid: str, energy: int = 9):
        """把一张牌塞进手里（并给足能量），返回它的手牌下标。"""
        from sts2_sim.core import CardInstance
        card = CardInstance(cid)
        state.hand.append(card)
        state.energy = energy
        return state.hand.index(card)


class MindRotTest(CurseTestCase):
    """P01 ``MindRotPower.ModifyHandDraw``（``MindRotPower.cs:14-21``）。"""

    def test_amount_subtracts_from_the_regular_draw(self):
        """``Math.Max(0m, count - Amount)``：基础的 5 张里减掉层数。"""
        state = self.combat()
        self.assertEqual(self.draw_turn(state, [("mind_rot", 2)]), 3, "5 - 2")

    def test_clamps_at_zero_instead_of_going_negative(self):
        """⭐ 层数超过额度 → 抽 **0** 张，不是负数（源码包着 ``Math.Max``）。"""
        state = self.combat()
        self.assertEqual(self.draw_turn(state, [("mind_rot", 9)]), 0)

    def test_chain_order_is_the_apply_order(self):
        """⭐ 链式 + **施加顺序**：截零在链中间，顺序不同结果就差 1。

        ``Hook.ModifyHandDraw`` 把上一个人的返回值喂给下一个人，listener 顺序 =
        ``Creature.Powers`` = 施加顺序。所以：

        * 先 ``clarity`` 后 ``mind_rot``：``max(0, 5+1-9) = 0``
        * 先 ``mind_rot`` 后 ``clarity``：``max(0, 5-9)+1 = 1``

        这条用例同时钉住两件事：**不能求和**（求和两种顺序都是 -3 → 截成 0），
        **不能按固定的表顺序**（那会把两种顺序算成同一个数）。
        """
        first = self.combat()
        self.assertEqual(
            self.draw_turn(first, [("clarity", 1), ("mind_rot", 9)]), 0,
            "clarity 先：max(0, 5+1-9) = 0")
        second = self.combat()
        self.assertEqual(
            self.draw_turn(second, [("mind_rot", 9), ("clarity", 1)]), 1,
            "mind_rot 先：max(0, 5-9)+1 = 1")

    def test_effect_extra_draw_is_not_reduced(self):
        """药水 / 卡牌的额外抽牌走 ``core.draw_cards``，**不经过** ``ModifyHandDraw``。

        出处：``CardPileCmd.Draw`` 的调用方是效果，不是 ``SetupPlayerTurn``
        那条 ``Hook.ModifyHandDraw``。把减益错误地做成"全局抽牌惩罚"，
        会让"抽 2 张"的牌在苦痛下变成 0 张 —— 而真机照抽。
        """
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("mind_rot", 3)
        state.draw_pile.extend(state.hand)
        state.hand.clear()
        core.draw_cards(state, 2, [])
        self.assertEqual(len(state.hand), 2)

    def test_enemy_side_mind_rot_does_not_touch_the_player(self):
        """``if (player != base.Owner.Player) return count;`` —— 挂在敌人身上不生效。"""
        state = self.combat()
        state.enemies[0].add_power("mind_rot", 3)
        self.assertEqual(self.draw_turn(state), 5)


class WasteAwayTest(CurseTestCase):
    """P02 ``WasteAwayPower.ModifyMaxEnergy``（``WasteAwayPower.cs:16-23``）。"""

    def test_applying_does_not_change_current_energy(self):
        """施加能力只改**上限**，不动当前能量（源码里 ``Apply`` 与 ``MaxEnergy`` 两条路）。"""
        state = self.combat()
        before = state.energy
        state.player.add_power("waste_away", 1)
        self.assertEqual(state.energy, before)

    def test_next_turn_reset_uses_the_reduced_maximum(self):
        """``PlayerCombatState.ResetEnergy`` 是 ``Energy = MaxEnergy``。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("waste_away", 1)
        state.draw_pile.extend(state.hand)
        state.hand.clear()
        core.start_player_turn(state, [])
        self.assertEqual(state.energy, state.max_energy - 1)

    def test_negative_maximum_is_not_clamped(self):
        """⭐ 源码此处**没有** ``max(0, …)``：``MaxEnergy`` 只是 ``(int)Hook.ModifyMaxEnergy``，
        ``ResetEnergy`` 与 ``Energy`` 的 setter 都不截零（``PlayerCombatState.cs:84-101``）。

        只有 ``GainEnergy`` / ``LoseEnergy`` 才 clamp。凭直觉补一个截零，
        "能量上限被压成负数"这个真机行为就会在模拟器里消失。
        """
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("waste_away", state.max_energy + 2)
        state.draw_pile.extend(state.hand)
        state.hand.clear()
        core.start_player_turn(state, [])
        self.assertEqual(state.energy, -2)

    def test_removing_the_power_restores_the_maximum(self):
        from sts2_sim import core, powers as power_rules
        state = self.combat()
        state.player.add_power("waste_away", 2)
        self.assertEqual(power_rules.max_energy_bonus(state), -2)
        state.player.add_power("waste_away", -2)
        self.assertEqual(power_rules.max_energy_bonus(state), 0)
        state.draw_pile.extend(state.hand)
        state.hand.clear()
        core.start_player_turn(state, [])
        self.assertEqual(state.energy, state.max_energy)

    def test_stacks_with_friendship(self):
        """名单是**求和**：加了减益之后 ``pyre`` / ``friendship`` 不能被漏掉。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("friendship", 3)
        state.player.add_power("waste_away", 1)
        self.assertEqual(power_rules.max_energy_bonus(state), 2)


class SlothTest(CurseTestCase):
    """P03 ``SlothPower``（``SlothPower.cs:22-51``）。"""

    def play(self, state, cid: str, target: int = 0):
        from sts2_sim.core import Action, step
        index = self.hold(state, cid)
        return step(state, Action("play_card", index, target))

    def test_allows_exactly_amount_cards(self):
        """``return _cardsPlayedThisTurn < base.Amount;``：层数 2 → 前 2 张放行。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("sloth", 2)
        self.play(state, "strike_ironclad")
        self.play(state, "strike_ironclad")
        self.assertEqual(power_rules.sloth_cards_played(state.player), 2)

    def test_blocks_the_card_after_the_limit(self):
        """⭐ 上限之后：``card_playable`` / ``legal_actions`` / 直接 ``step`` **三处一致**。

        只封锁动作掩码是不够的 —— 观察层与直接执行都读 ``card_playable``，
        漏一处就会出现"界面说不能打、直接执行却成功了"。
        """
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("sloth", 2)
        self.play(state, "strike_ironclad")
        self.play(state, "strike_ironclad")
        index = self.hold(state, "strike_ironclad")
        card = state.hand[index]
        self.assertFalse(core.card_playable(state, card), "可打性判据")
        actions = core.legal_actions(state)
        self.assertFalse(
            any(a.kind == "play_card" and a.hand_index == index for a in actions),
            "合法动作里不该有它")
        with self.assertRaises(ValueError):
            core.step(state, core.Action("play_card", index, 0))

    def test_refusal_costs_nothing(self):
        """被拒的那次不花能量、牌还在手里、计数也不再涨。"""
        from sts2_sim import core, powers as power_rules
        state = self.combat()
        state.player.add_power("sloth", 1)
        self.play(state, "strike_ironclad")
        index = self.hold(state, "strike_ironclad", energy=9)
        energy = state.energy
        with self.assertRaises(ValueError):
            core.step(state, core.Action("play_card", index, 0))
        self.assertEqual(state.energy, energy)
        self.assertIn(state.hand[index].cid, ("strike_ironclad",))
        self.assertEqual(power_rules.sloth_cards_played(state.player), 1)

    def test_counter_resets_at_the_owner_side_turn_start(self):
        """``BeforeSideTurnStart`` 里 ``participants.Contains(Owner)`` → 归零。"""
        from sts2_sim import core, powers as power_rules
        state = self.combat()
        state.player.add_power("sloth", 2)
        self.play(state, "strike_ironclad")
        self.assertEqual(power_rules.sloth_cards_played(state.player), 1)
        state.draw_pile.extend(state.hand)
        state.hand.clear()
        core.start_player_turn(state, [])
        self.assertEqual(power_rules.sloth_cards_played(state.player), 0,
                         "新回合计数归零")
        self.assertEqual(state.player.power("sloth"), 2, "层数是上限，不归零")

    def test_applied_mid_turn_starts_from_zero(self):
        """中途首次施加从 **0** 开始 —— 不能复制整回合的全局出牌统计。"""
        from sts2_sim import core
        state = self.combat()
        self.play(state, "strike_ironclad")
        state.player.add_power("sloth", 2)
        index = self.hold(state, "strike_ironclad")
        self.assertTrue(core.card_playable(state, state.hand[index]),
                        "已打的那张不算进怠惰的私有计数")

    def test_more_layers_raise_the_limit_without_resetting_the_counter(self):
        """叠层改上限、**不重置**计数（源码里只有 ``BeforeSideTurnStart`` 归零）。"""
        from sts2_sim import core, powers as power_rules
        state = self.combat()
        state.player.add_power("sloth", 1)
        self.play(state, "strike_ironclad")
        state.player.add_power("sloth", 1)
        self.assertEqual(power_rules.sloth_cards_played(state.player), 1)
        index = self.hold(state, "strike_ironclad")
        self.assertTrue(core.card_playable(state, state.hand[index]), "上限升到 2")

    def test_autoplay_is_also_limited(self):
        """⭐ ``ShouldPlay`` 丢弃 ``autoPlayType``（参数名 ``_``）—— 自动打出同样受限。

        与 ``Enthralled``（放行自动打出）相反，所以两张表不能共用一条"自动免检"。
        """
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("sloth", 1)
        self.play(state, "strike_ironclad")
        index = self.hold(state, "strike_ironclad")
        card = state.hand.pop(index)
        events: list[str] = []
        hp = state.enemies[0].hp
        self.assertTrue(core.autoplay_card(state, card, events))
        self.assertEqual(state.enemies[0].hp, hp, "被否决 → 只落牌堆、不结算")
        self.assertTrue(any("否决" in e for e in events), events)

    def test_enemy_side_sloth_does_not_limit_the_player(self):
        """``card.Owner.Creature != base.Owner`` → 放行：敌人身上的怠惰不影响玩家出牌。"""
        from sts2_sim import core
        state = self.combat()
        state.enemies[0].add_power("sloth", 1)
        index = self.hold(state, "strike_ironclad")
        self.assertTrue(core.card_playable(state, state.hand[index]))


class DisintegrationTest(CurseTestCase):
    """P04 ``DisintegrationPower.AfterSideTurnEndLate``（``DisintegrationPower.cs:19-26``）。"""

    def test_hurts_the_owner_at_their_own_side_turn_end(self):
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("disintegration", 6)
        hp = state.player.hp
        power_rules.on_side_turn_end_late(state, [], "enemy")   # 敌方结束：不该触发
        self.assertEqual(state.player.hp, hp, "敌方阵营结束不触发")
        power_rules.on_side_turn_end_late(state, [], "player")  # 玩家结束：触发
        self.assertEqual(hp - state.player.hp, 6)

    def test_the_end_turn_flow_dispatches_the_late_pass(self):
        """完整 ``step(end_turn)`` 流程里也要走到第二遍 —— 不能只有直接调用才触发。

        （数值断言放在上一条里：这里跑的是完整回合，敌方出招也会掉血。）
        """
        from sts2_sim import core, hooks
        state = self.combat()
        state.player.add_power("disintegration", 6)
        with hooks.trace() as record:
            core.step(state, core.Action("end_turn"))
        self.assertIn("on_side_turn_end_late", record.names())

    def test_block_absorbs_it(self):
        """``ValueProp.Unpowered`` ≠ ``Unblockable``：格挡**可以**吸收。"""
        from sts2_sim import core, powers as power_rules
        state = self.combat()
        state.player.add_power("disintegration", 6)
        state.player.block = 10
        hp = state.player.hp
        power_rules.on_side_turn_end_late(state, [], "player")
        self.assertEqual(state.player.hp, hp, "全格挡")
        self.assertEqual(state.player.block, 4)

    def test_strength_does_not_increase_it(self):
        """``Unpowered``：力量不加成、也不吃易伤倍率。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("disintegration", 6)
        state.player.add_power("strength", 5)
        hp = state.player.hp
        power_rules.on_side_turn_end_late(state, [], "player")
        self.assertEqual(hp - state.player.hp, 6)

    def test_does_not_decrement(self):
        """源码里没有 ``Decrement``：层数**一直在**，每回合都掉这么多。"""
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("disintegration", 6)
        power_rules.on_side_turn_end_late(state, [], "player")
        power_rules.on_side_turn_end_late(state, [], "player")
        self.assertEqual(state.player.power("disintegration"), 6)

    def test_is_dispatched_after_the_first_pass(self):
        """⭐ ``AfterSideTurnEndLate`` 是**第二遍**，必须排在 ``AfterSideTurnEnd`` 之后。

        真机 ``Hook.AfterSideTurnEnd`` 在同一个函数里先跑完第一遍、再跑这一遍
        （``Hook.cs``）。错序（例如与第一遍交错）会让自伤插到别的回合末效果中间。
        """
        from sts2_sim import hooks, powers as power_rules
        state = self.combat()
        state.player.add_power("disintegration", 6)
        with hooks.trace() as record:
            power_rules.on_side_turn_end(state, [], "player")
            power_rules.on_side_turn_end_late(state, [], "player")
        order = [name for name in record.names() if name.startswith("on_side_turn_end")]
        self.assertIn("on_side_turn_end_late", order)
        first_late = order.index("on_side_turn_end_late")
        last_early = max(i for i, name in enumerate(order) if name == "on_side_turn_end")
        self.assertLess(last_early, first_late,
                        f"第二遍必须整体排在第一遍之后，实际 {order}")

    def test_lethal_self_damage_ends_the_combat(self):
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.hp = 3
        state.player.add_power("disintegration", 6)
        power_rules.on_side_turn_end_late(state, [], "player")
        self.assertEqual(state.player.hp, 0)
        self.assertFalse(state.player.alive())


if __name__ == "__main__":
    unittest.main()
