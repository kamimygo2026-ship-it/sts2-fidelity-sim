"""**W02 · P06** ``ForegoneConclusionPower``：抽牌前挑牌入手，然后移除自己。

真机出处 ``ForegoneConclusionPower.cs:19-28``::

    if (player == base.Owner.Player) {
        await CardPileCmd.ShuffleIfNecessary(choiceContext, base.Owner.Player);
        await CardPileCmd.Add(await CardSelectCmd.FromCombatPile(
            choiceContext, PileType.Draw.GetPile(base.Owner.Player), base.Owner.Player,
            new CardSelectorPrefs(SelectionScreenPrompt, base.Amount)), PileType.Hand);
        await PowerCmd.Remove(this);
    }

它与 P05（`Stratagem`）是一对：都从抽牌堆挑牌入手，但

* 时机在 ``BeforeHandDraw``（**抽手牌之前**）而不是洗完牌之后；
* 会 ``ShuffleIfNecessary``（抽牌堆空才洗）—— 而**洗牌又会**触发
  ``StratagemPower.AfterShuffle``，于是两个能力会**嵌套挂起**；
* 结束时 ``PowerCmd.Remove(this)``：**整个移除**自己，不是减层。

本文件钉住这四条，以及"抽牌之前"这个顺序。
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


class ForegoneConclusionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, enemies=("nibbit",), seed: int = 5):
        from sts2_sim.core import start_combat
        return start_combat(self.deck, tuple(enemies), seed=seed)

    def start_turn(self, state, powers=(), empty_draw: bool = True):
        """开一个回合。``empty_draw`` = 把抽牌堆清空（逼出 `ShuffleIfNecessary`）。"""
        from sts2_sim import core
        state.discard.extend(state.hand)
        state.hand.clear()
        if empty_draw:
            state.discard.extend(state.draw_pile)
            state.draw_pile.clear()
        for pid, amount in powers:
            state.player.add_power(pid, amount)
        core.start_player_turn(state, [])
        return state


class OrderTest(ForegoneConclusionTestCase):
    """挑牌发生在**抽手牌之前**（`BeforeHandDraw`）。"""

    def test_nothing_is_drawn_before_the_pick(self):
        state = self.start_turn(self.combat(), [("foregone_conclusion", 2)])
        self.assertIsNotNone(state.pending)
        self.assertEqual(len(state.hand), 0, "挂起时一张手牌都还没有")
        self.assertEqual(state.draw_frame.remaining, -1,
                         "-1 = 连抽几张都还没算（ModifyHandDraw 排在它之后）")

    def test_shuffle_if_necessary_only_when_the_draw_pile_is_empty(self):
        """抽牌堆非空 → **不**洗牌，但仍然挑牌。"""
        state = self.combat()
        events: list[str] = []
        self.start_turn(state, [("foregone_conclusion", 1)], empty_draw=False)
        self.assertIsNotNone(state.pending)
        self.assertFalse(any("洗牌" in line for line in state.log),
                         "抽牌堆非空时不该洗牌")
        events.clear()

    def test_the_pick_lands_in_hand_and_the_draw_still_happens(self):
        from sts2_sim import core
        state = self.start_turn(self.combat(), [("foregone_conclusion", 1)])
        pool = len(state.draw_pile)
        chosen = state.pending.candidates(state)[0]
        core.step(state, core.Action("select_card", 0))
        self.assertIn(chosen, state.hand)
        self.assertEqual(len(state.hand), 6, "1 张挑入手 + 5 张正常抽")
        self.assertEqual(len(state.draw_pile), pool - 1 - 5)


class RemovalTest(ForegoneConclusionTestCase):
    """``PowerCmd.Remove(this)``：整个移除，且**在选完之后**。"""

    def test_the_power_is_still_there_while_choosing(self):
        from sts2_sim import core
        state = self.start_turn(self.combat(), [("foregone_conclusion", 2)])
        self.assertEqual(state.player.power("foregone_conclusion"), 2,
                         "挂起期间能力还在（真机是选完才 Remove）")
        self.assertEqual(state.pending.remove_power_after, "foregone_conclusion")
        core.step(state, core.Action("select_card", 0))
        self.assertEqual(state.player.power("foregone_conclusion"), 2,
                         "还差一张，不能提前移除")
        core.step(state, core.Action("select_card", 0))
        self.assertEqual(state.player.power("foregone_conclusion"), 0,
                         "选完整个移除（2 层一起没了，不是减到 1）")

    def test_no_candidates_still_removes_the_power(self):
        """全空牌堆：真机的 `Add` 是空操作，但 `PowerCmd.Remove` **无条件**执行。

        不修这一条的话，这张牌打出去之后能力永远留着，每回合空问一次 ——
        而玩家看到的是"牌打出去了但什么都没发生"。
        """
        from sts2_sim import core
        state = self.combat()
        state.hand.clear()
        state.discard.clear()
        state.draw_pile.clear()
        state.player.add_power("foregone_conclusion", 1)
        core.start_player_turn(state, [])
        self.assertIsNone(state.pending)
        self.assertEqual(state.player.power("foregone_conclusion"), 0)

    def test_removal_does_not_fire_power_amount_changed(self):
        """⭐ `PowerCmd.Remove` 与"减层"是两条路：前者**不**派发 `AfterPowerAmountChanged`。

        用错（`add_power(pid, -层数)`）会让"层数变化时触发"的能力多响应一次。
        """
        from sts2_sim import core, hooks
        state = self.start_turn(self.combat(), [("foregone_conclusion", 1)])
        with hooks.trace() as record:
            core.step(state, core.Action("select_card", 0))
        self.assertEqual(state.player.power("foregone_conclusion"), 0)
        self.assertEqual(record.count("on_power_amount_changed"), 0,
                         "整个移除不该走层数变化那条路径")


class NestingTest(ForegoneConclusionTestCase):
    """两个能力同时在场：`Stratagem` 先挂、`ForegoneConclusion` 排队。"""

    def test_the_two_choices_happen_in_source_order(self):
        """真机是嵌套 `await`：`ShuffleIfNecessary` 里的 `AfterShuffle`
        先让 `Stratagem` 挑完，才轮到 `ForegoneConclusion` 自己挑。"""
        from sts2_sim import core
        state = self.start_turn(self.combat(),
                                [("stratagem", 1), ("foregone_conclusion", 2)])
        picks: list[tuple[str, str]] = []
        while state.pending is not None:
            picks.append((state.pending.purpose,
                          state.pending.remove_power_after or "-"))
            core.step(state, core.Action("select_card", 0))
        self.assertEqual(picks, [("to_hand", "-"),
                                 ("to_hand", "foregone_conclusion"),
                                 ("to_hand", "foregone_conclusion")],
                         "先把 Stratagem 的一次挑完，再挑 ForegoneConclusion 的两张")
        self.assertEqual(state.player.power("stratagem"), 1)
        self.assertEqual(state.player.power("foregone_conclusion"), 0)
        self.assertEqual(len(state.hand), 8, "1 + 2 张挑入手 + 5 张正常抽")
        self.assertEqual(state.pending_tasks, [])


class SnapshotsTest(ForegoneConclusionTestCase):
    def test_snapshot_mid_choice_keeps_the_pending_removal(self):
        from sts2_sim import core
        state = self.start_turn(self.combat(), [("foregone_conclusion", 1)])
        saved = core.snapshot(state)
        restored = core.restore(saved)
        self.assertEqual(restored.pending.remove_power_after, "foregone_conclusion")
        self.assertEqual(restored.draw_frame.remaining, -1)
        core.step(restored, core.Action("select_card", 0))
        self.assertEqual(restored.player.power("foregone_conclusion"), 0)
        self.assertEqual(len(restored.hand), 6)


if __name__ == "__main__":
    unittest.main()
