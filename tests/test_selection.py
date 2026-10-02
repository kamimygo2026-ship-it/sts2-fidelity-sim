"""玩家选牌动作的回归测试（``docs/09`` L3「选牌」）。

真机的效果序列是 `抽牌 → 选牌 → 弃牌`，所以选牌必须**落在序列的正确位置**上，
而且在选牌期间：
  * 只能选牌（其它动作全非法），
  * 打出的那张牌**还不能**落进弃牌堆（续跑还没跑完），
  * **重开语义依然成立**（SL 重开后必须面对同一副手牌与同一组候选）。

对应已真实发生过的 bug：

| 症状 | 根因 | 对应测试 |
|---|---|---|
| `Prepared` 变成"先弃后选" | 选牌**嵌套**在外层 `CardCmd.Discard(...)` 的参数里，按文本位置排序会颠倒 | `test_nested_selection_keeps_source_order` |
| 选牌时 bot 返回 `end_turn`（非法动作） | `RuleBot.act` 只找 `play_card`，没找 `select_card` | `test_rule_bot_answers_a_selection` |
| 加特征列后整个模块 import 就炸 | `NUM_FEATURES` 是手写魔数，与 `FEATURE_LAYOUT` 重复维护 | `test_feature_layout_is_the_single_source` |
| lint 把合法算子报成"未知" | `lint` 里另抄了一份算子表 | `test_lint_accepts_select_card` |
"""

from __future__ import annotations

import json
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


def _source_card(cid: str) -> dict:
    cards = json.loads((CONTENT / "cards_source.json").read_text(encoding="utf-8"))
    return next(c for c in cards if c["cid"] == cid)


class TestSelectionExtraction(unittest.TestCase):
    """抽取层：顺序与嵌套关系。"""

    @classmethod
    def setUpClass(cls):
        if not CONTENT.exists():
            raise unittest.SkipTest(f"缺内容目录 {CONTENT}")

    def test_acrobatics_draws_before_discarding(self):
        """分句写法：`Draw(…); … FromHandForDiscard(…); CardCmd.Discard(…)`。

        顺序必须是"先抽后弃"—— 反了的话手里可弃的牌少 3 张，选择空间完全不同。
        """
        ops = [e["op"] for e in _source_card("acrobatics")["effects"]]
        self.assertEqual(ops, ["draw", "select_card"])

    def test_nested_selection_keeps_source_order(self):
        """嵌套写法：``CardCmd.Discard(ctx, await CardSelectCmd.FromHandForDiscard(…))``。

        `CardCmd.Discard` 在**文本里更靠前**，按位置排序会变成"先弃后选"。
        必须靠**括号包含关系**认出它只是外层包装。
        """
        ops = [e["op"] for e in _source_card("prepared")["effects"]]
        self.assertEqual(ops, ["draw", "select_card"],
                         "嵌套选牌的顺序被按文本位置排反了")

    def test_selection_encodes_purpose_and_count(self):
        for cid, purpose in (("acrobatics", "discard"), ("brand", "exhaust")):
            with self.subTest(cid=cid):
                effect = next(e for e in _source_card(cid)["effects"]
                              if e["op"] == "select_card")
                self.assertEqual(effect["purpose"], purpose)
                self.assertEqual(effect["from"], "hand")
                self.assertGreaterEqual(effect["count"], 1)

    def test_wrapping_discard_is_not_emitted_twice(self):
        """选牌的后果已经由 `select_card` 表达，包装层的 discard/exhaust 要丢掉。"""
        for cid in ("acrobatics", "prepared", "survivor", "brand", "true_grit"):
            with self.subTest(cid=cid):
                ops = [e["op"] for e in _source_card(cid)["effects"]]
                self.assertNotIn("discard", ops)
                self.assertNotIn("exhaust", ops)


class TestSelectionRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat_with(self, cid: str, hand_size: int = 4):
        from sts2_sim.core import CardInstance, start_combat
        state = start_combat(self.deck, ("nibbit",), seed=7)
        state.hand = [CardInstance(cid)] + [
            CardInstance("strike_ironclad") for _ in range(hand_size)]
        state.draw_pile = [CardInstance("defend_ironclad") for _ in range(6)]
        state.energy = 5
        return state

    def test_playing_a_selection_card_pauses(self):
        from sts2_sim.core import Action, legal_actions, step
        state = self._combat_with("acrobatics")
        step(state, Action("play_card", 0, -1))
        self.assertIsNotNone(state.pending)
        self.assertEqual(state.pending.purpose, "discard")
        self.assertTrue(all(a.kind == "select_card" for a in legal_actions(state)),
                        "挂起期间只允许选牌")

    def test_played_card_does_not_land_until_selection_finishes(self):
        """挂起时打出的牌留在 ``play_area`` —— 它还没算打完。"""
        from sts2_sim.core import Action, step
        state = self._combat_with("acrobatics")
        step(state, Action("play_card", 0, -1))
        self.assertIn("acrobatics", [c.cid for c in state.play_area])
        self.assertNotIn("acrobatics", [c.cid for c in state.discard])

    def test_selection_resumes_and_finishes_the_play(self):
        from sts2_sim.core import Action, step
        state = self._combat_with("acrobatics")
        step(state, Action("play_card", 0, -1))
        drawn = [c.cid for c in state.hand]
        step(state, Action("select_card", 0))
        self.assertIsNone(state.pending)
        self.assertEqual(state.play_area, [])
        self.assertIn("acrobatics", [c.cid for c in state.discard])
        self.assertEqual(len(drawn) - 1, len(state.hand), "应恰好少一张（被弃掉）")

    def test_draw_happens_before_the_discard(self):
        """⭐ 顺序语义：抽到的牌**也在**候选里。

        "先弃后抽"与"先抽后弃"的区别是实打实的：后者多 3 个候选。
        """
        from sts2_sim.core import Action, step
        state = self._combat_with("acrobatics")
        hand_before = len(state.hand)
        step(state, Action("play_card", 0, -1))
        candidates = state.pending.candidates(state)
        self.assertEqual(len(candidates), hand_before - 1 + 3,
                         "打出 1 张、抽 3 张之后候选应是 原手牌-1+3")

    def test_candidates_are_recomputed_not_snapshotted(self):
        """候选**每次现算**：选一张之后下标移位，快照会让引擎与观测错位。"""
        from sts2_sim.core import Action, step
        state = self._combat_with("acrobatics")
        step(state, Action("play_card", 0, -1))
        first = list(state.pending.candidates(state))
        step(state, Action("select_card", 2))
        self.assertIsNone(state.pending)          # 只选 1 张，已完成
        self.assertNotIn(first[2], state.hand)    # 被选中的那张确实离开了手牌

    def test_rule_bot_answers_a_selection(self):
        """⚠️ bot 只找 `play_card` 的话，选牌时会返回 `end_turn`（**非法动作**）。

        `survivor`（格挡 + 弃牌）这类卡被 bot 打出去之后，下一次调用就会炸。
        """
        from sts2_sim.bot import RuleBot
        from sts2_sim.core import Action, legal_actions, step
        from sts2_sim.observe import observe
        state = self._combat_with("survivor")
        bot = RuleBot()
        for _ in range(6):
            legal = legal_actions(state)
            if not legal:
                break
            action = bot.act(observe(state), legal)
            self.assertIn(action, legal, f"bot 给出了非法动作 {action}")
            step(state, action)
            if state.finished():
                break

    def test_invalid_selection_index_raises(self):
        from sts2_sim.core import Action, step
        state = self._combat_with("acrobatics")
        step(state, Action("play_card", 0, -1))
        with self.assertRaises(ValueError):
            step(state, Action("select_card", 999))


class TestSelectionAndSL(unittest.TestCase):
    """⭐ SL 语义在选牌状态下必须依然成立。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_restart_preserves_the_same_candidates(self):
        """重开后必须面对**同一副手牌与同一组候选**。

        选牌是"已揭示信息"的来源之一：重开后候选若变了，
        模型就能靠重开刷出想要的候选 —— 那是作弊，不是学习。
        """
        from sts2_sim.core import Action, legal_actions, step
        from sts2_sim.core import CardInstance, start_combat

        def candidates_after_play():
            state = start_combat(self.deck, ("nibbit",), seed=99)
            state.hand = [CardInstance("acrobatics")] + [
                CardInstance("strike_ironclad") for _ in range(4)]
            state.draw_pile = [CardInstance("defend_ironclad") for _ in range(6)]
            state.energy = 5
            step(state, Action("play_card", 0, -1))
            return [c.cid for c in state.pending.candidates(state)]

        first = candidates_after_play()
        second = candidates_after_play()
        self.assertEqual(first, second, "同一副牌序下候选必须一致")


class TestFeatureLayoutConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_feature_layout_is_the_single_source(self):
        """``NUM_FEATURES`` 必须由布局推导，不能另写一个数字。

        实测加了三列选牌特征后手写的 `NUM_FEATURES` 仍是 20，
        `assert len(FEATURE_LAYOUT) == NUM_FEATURES` 直接让整个模块 import 失败。
        """
        from sts2_sim import featurize
        self.assertEqual(featurize.NUM_FEATURES, len(featurize.FEATURE_LAYOUT))
        self.assertIn("selecting", featurize.FEATURE_LAYOUT)
        self.assertIn("selecting", featurize.FEATURE_USAGE["global"])

    def test_selection_is_visible_in_the_observation(self):
        """选牌状态必须进观测：屏幕上写着"选择一张牌弃置"。"""
        from sts2_sim.core import Action, CardInstance, start_combat, step
        from sts2_sim.observe import observe
        state = start_combat(self.deck, ("nibbit",), seed=7)
        state.hand = [CardInstance("acrobatics")] + [
            CardInstance("strike_ironclad") for _ in range(4)]
        state.energy = 5
        self.assertEqual(observe(state).selecting_purpose, "")
        step(state, Action("play_card", 0, -1))
        view = observe(state)
        self.assertEqual(view.selecting_purpose, "discard")
        self.assertGreaterEqual(view.selecting_remaining, 1)

    def test_action_kind_vocab_includes_selection(self):
        from sts2_sim import featurize
        self.assertIn("select_card", featurize.ACTION_KINDS)

    def test_lint_accepts_select_card(self):
        """lint 必须与 `ENGINE_OPS` 用**同一张**算子表，不能另抄一份。"""
        from sts2_sim import content
        self.assertIn("select_card", content.ENGINE_OPS)
        problems = [p for p in content.lint() if "select_card" in p]
        self.assertEqual(problems, [], "lint 把合法算子报成了未知")


if __name__ == "__main__":
    unittest.main(verbosity=2)
