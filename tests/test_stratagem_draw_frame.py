"""**W02 · P05** ``StratagemPower``（计策）：洗牌后从抽牌堆挑牌入手。

真机出处 ``StratagemPower.cs:14-27``::

    public override async Task AfterShuffle(PlayerChoiceContext ctx, Player player)
    {
        if (player != base.Owner.Player) return;
        foreach (CardModel item in await CardSelectCmd.FromCombatPile(
                     ctx, PileType.Draw.GetPile(base.Owner.Player), base.Owner.Player,
                     new CardSelectorPrefs(base.SelectionScreenPrompt, base.Amount)))
            await CardPileCmd.Add(item, PileType.Hand);
    }

钩子时机是 ``Hook.AfterShuffle``（``CardPileCmd.cs:1131``，在 ``Shuffle`` 末尾）——
而 ``Shuffle`` 在**抽牌循环中间**也会被调到（``DrawInternal`` 每轮
``await ShuffleIfNecessary``，``CardPileCmd.cs:1047``）。所以这条能力考验的是
**抽牌中途挂起**：真机靠 ``await`` 停在原地，引擎靠 ``core.DrawFrame``。

本文件钉住四件事：挂起的位置、候选来源与顺序、**入手不是抽牌**、
以及选完之后抽牌与回合开始剩余步骤都接着跑完。
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


class StratagemTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, enemies=("nibbit",), seed: int = 5):
        from sts2_sim.core import start_combat
        return start_combat(self.deck, tuple(enemies), seed=seed)

    def draw_turn_with_shuffle(self, state, powers=(), hand: int = 0):
        """把整副牌塞进弃牌堆、清空抽牌堆，再开一个回合 —— 抽牌时**必然**洗牌。

        ``hand`` 是保留在手里的张数（默认 0）：留牌会让抽牌数被手牌上限截断。
        """
        from sts2_sim import core
        state.discard.extend(state.hand)
        state.hand.clear()
        kept = []
        for _ in range(hand):
            kept.append(state.discard.pop())
        state.hand.extend(kept)
        state.discard.extend(state.draw_pile)
        state.draw_pile.clear()
        for pid, amount in powers:
            state.player.add_power(pid, amount)
        core.start_player_turn(state, [])
        return state


class SuspendTest(StratagemTestCase):
    """挂起发生在**洗牌那一刻**，而不是抽完之后。"""

    def test_the_draw_suspends_right_after_the_shuffle(self):
        state = self.combat()
        self.draw_turn_with_shuffle(state, [("stratagem", 1)])
        self.assertIsNotNone(state.pending, "洗牌触发的选择必须把抽牌停住")
        self.assertIsNotNone(state.draw_frame, "必须记下还要抽几张")
        self.assertEqual(state.draw_frame.remaining, 5)
        self.assertTrue(state.draw_frame.resume_turn_start,
                        "这是回合开始的手牌抽取 → 选完要接着跑回合开始的剩余步骤")
        self.assertEqual(len(state.hand), 0, "挂起时一张都还没抽")

    def test_candidates_come_from_the_draw_pile_in_canonical_order(self):
        """候选来自**抽牌堆**，且顺序按 ``(cid, upgraded)`` 规范排序。

        抽牌堆顺序是 L3 隐藏信息：拿它当候选顺序就是把隐藏信息编码进观测。
        """
        from sts2_sim import core
        state = self.combat()
        self.draw_turn_with_shuffle(state, [("stratagem", 1)])
        candidates = state.pending.candidates(state)
        self.assertEqual(state.pending.source_pile, "draw")
        self.assertEqual([c.cid for c in candidates],
                         sorted(c.cid for c in candidates))
        self.assertEqual(len(candidates), 10, "整副起手牌都被洗进抽牌堆")

    def test_no_candidates_means_no_suspension(self):
        """牌堆全空时不挂起 —— 挂一个"没有候选"的选择会让 `legal_actions` 返回空。"""
        from sts2_sim import core
        state = self.combat()
        state.hand.clear()
        state.discard.clear()
        state.draw_pile.clear()
        state.player.add_power("stratagem", 1)
        core.start_player_turn(state, [])
        self.assertIsNone(state.pending)
        self.assertIsNone(state.draw_frame)


class ResumeTest(StratagemTestCase):
    """选完之后：该抽的牌抽完、回合开始的剩余步骤跑完。"""

    def test_the_chosen_card_enters_hand_and_the_draw_continues(self):
        from sts2_sim import core
        state = self.combat()
        self.draw_turn_with_shuffle(state, [("stratagem", 1)])
        pool = len(state.draw_pile)
        chosen = state.pending.candidates(state)[0]
        core.step(state, core.Action("select_card", 0))
        self.assertIn(chosen, state.hand, "选中的那张必须进手牌")
        self.assertNotIn(chosen, state.draw_pile, "移动入手，不能同时留在抽牌堆")
        self.assertEqual(len(state.hand), 6, "1 张选入手 + 5 张正常抽")
        self.assertEqual(len(state.draw_pile), pool - 1 - 5)
        self.assertIsNone(state.draw_frame, "帧必须被消费掉")

    def test_entering_hand_is_a_move_not_a_draw(self):
        """⭐ 入手走 ``CardPileCmd.Add``，**不**派发"抽到牌"的钩子。

        用 ``ConfusedPower``（``AfterCardDrawn`` 时随机改费）当探针：
        选入手的牌如果被改了费，就说明引擎把它当成了"抽牌"。
        """
        from sts2_sim import core
        state = self.combat()
        self.draw_turn_with_shuffle(state, [("stratagem", 1), ("confused", 1)])
        chosen = state.pending.candidates(state)[0]
        core.step(state, core.Action("select_card", 0))
        self.assertIsNone(chosen.combat_cost_override,
                          "移动入手不该触发 AfterCardDrawn（苦痛/混乱是抽牌时机）")
        # 而**抽**上来的牌必须被混乱改费 —— 反向验证探针本身有效
        drawn = [c for c in state.hand if c is not chosen]
        self.assertTrue(any(c.combat_cost_override is not None for c in drawn),
                        "同一批抽上来的牌应当被混乱改费")

    def test_the_power_is_kept_and_does_not_decrement(self):
        state = self.combat()
        self.draw_turn_with_shuffle(state, [("stratagem", 2)])
        from sts2_sim import core
        while state.pending is not None:
            core.step(state, core.Action("select_card", 0))
        self.assertEqual(state.player.power("stratagem"), 2, "能力保留、层数不减")

    def test_two_layers_ask_for_two_cards(self):
        from sts2_sim import core
        state = self.combat()
        self.draw_turn_with_shuffle(state, [("stratagem", 2)])
        picked = 0
        while state.pending is not None:
            core.step(state, core.Action("select_card", 0))
            picked += 1
        self.assertEqual(picked, 2)

    def test_the_rest_of_the_turn_start_still_runs(self):
        """⭐ 抽牌挂起**不能**把回合开始的剩余步骤吞掉。

        真机整个 ``SetupPlayerTurn`` 停在那个 ``await`` 上，选完接着跑
        ``Hook.AfterPlayerTurnStart`` / ``AfterSideTurnStart`` / 球的回合开始 /
        ``AutoPrePlay``。少跑任何一段都不报错，只是那一回合"少了一段流程"。
        """
        from sts2_sim import core, hooks
        state = self.combat()
        state.discard.extend(state.hand)
        state.hand.clear()
        state.discard.extend(state.draw_pile)
        state.draw_pile.clear()
        state.player.add_power("stratagem", 1)
        with hooks.trace() as record:
            core.start_player_turn(state, [])
            self.assertIsNotNone(state.pending)
            core.step(state, core.Action("select_card", 0))
        names = record.names()
        for expected in ("on_after_player_turn_start", "on_owner_turn_start",
                         "on_auto_pre_play"):
            self.assertIn(expected, names, f"续跑后仍然要分发 {expected}")

    def test_the_full_end_turn_flow_also_suspends(self):
        """完整 ``step(end_turn)`` 链路：敌方回合跑完之后、新回合抽牌时挂起。"""
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("stratagem", 1)
        state.player.max_hp = 3000
        state.player.hp = 3000
        # 把抽牌堆清空（牌留在手里，回合末会进弃牌堆）→ 新回合抽牌必然洗牌
        state.discard.extend(state.draw_pile)
        state.draw_pile.clear()
        result = core.step(state, core.Action("end_turn"))
        self.assertFalse(result.done)
        self.assertIsNotNone(state.pending)
        self.assertIsNotNone(state.draw_frame)
        actions = core.legal_actions(state)
        self.assertEqual([a.kind for a in actions].count("select_card"),
                         len(state.pending.candidates(state)))
        core.step(state, core.Action("select_card", 0))
        self.assertEqual(state.phase, "player")
        self.assertIsNone(state.draw_frame)


class SnapshotsTest(StratagemTestCase):
    """挂起时的快照与恢复（SL 的物理基础）。"""

    def test_snapshot_mid_draw_restores_the_frame(self):
        from sts2_sim import core
        state = self.combat()
        self.draw_turn_with_shuffle(state, [("stratagem", 1)])
        saved = core.snapshot(state)

        restored = core.restore(saved)
        self.assertIsNotNone(restored.pending)
        self.assertEqual(restored.draw_frame, state.draw_frame)
        self.assertEqual([c.cid for c in restored.pending.candidates(restored)],
                         [c.cid for c in state.pending.candidates(state)])

        core.step(restored, core.Action("select_card", 0))
        self.assertEqual(len(restored.hand), 6)
        self.assertEqual(restored.player.power("stratagem"), 1)

    def test_the_frame_is_pure_data(self):
        import copy
        state = self.combat()
        self.draw_turn_with_shuffle(state, [("stratagem", 1)])
        clone = copy.deepcopy(state)
        self.assertEqual(clone.draw_frame.remaining, state.draw_frame.remaining)


if __name__ == "__main__":
    unittest.main()
