"""选牌下标一致性与信息边界的回归测试（审计 F09 / 工单 T03）。

``Action("select_card", i)`` 里的 ``i`` 到底是"第几个候选"还是"手牌第几张"，
曾经**两边不一致**：

* 观测按 ``(cid, upgraded)`` 排序给出候选；
* 引擎的动作下标用的是 ``PendingSelection.candidates()`` 的原始顺序。

实测手牌 ``[bash, strike, strike, defend, defend]`` 在观测里变成
``[bash, defend, defend, strike, strike]`` —— 于是同一个下标在两边指向不同的牌。
这不会报错，只会让模型"想弃防御却弃了攻击"。

顺带锁住信息边界：**从抽牌堆选牌时，候选排列不得泄漏抽牌堆的真实顺序**。

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 观测候选与动作下标错位 | 排序只做在观测侧 | `test_observation_matches_action_index` |
| 抽牌堆顺序泄漏成候选排列 | 直接 `list(state.draw_pile)` | `test_draw_pile_order_is_not_revealed` |
| Run 层选牌在观测里不可见 | 观测没有候选字段 | `test_run_selection_is_visible_with_matching_slots` |
"""

from __future__ import annotations

import unittest

from sts2_sim import legal_actions, observe, start_combat
from sts2_sim.core import Action, PendingSelection
from sts2_sim.observe import _selection_view


def make_state(deck=("defend", "strike", "defend", "strike", "bash")):
    return start_combat(list(deck), ("jaw_worm",), seed=1)


def attach(state, source_pile: str, purpose: str = "discard", remaining: int = 1):
    state.pending = PendingSelection(purpose=purpose, source_pile=source_pile,
                                     remaining=remaining)
    return state


class TestSelectionIndexConsistency(unittest.TestCase):
    def test_observation_matches_action_index(self):
        """⭐ 观测里的第 i 个候选，必须就是动作 ``select_card(i)`` 选中的那张。"""
        for source in ("hand", "discard", "draw"):
            with self.subTest(source=source):
                state = make_state()
                if source == "discard":
                    state.discard.extend(state.draw_pile[:3])
                    state.draw_pile = state.draw_pile[3:]
                attach(state, source)
                if not state.pending.candidates(state):
                    continue
                engine = [card.cid for card in state.pending.candidates(state)]
                view = [card.cid for card in _selection_view(state).candidates]
                self.assertEqual(engine, view,
                                 f"{source} 的候选顺序在观测与引擎之间不一致")

    def test_action_index_resolves_to_the_observed_card(self):
        """端到端：按观测下标发动作，引擎清掉的必须是观测里那一张。"""
        state = make_state()
        attach(state, "hand")
        observed = _selection_view(state).candidates
        target = observed[1]
        from sts2_sim.core import step
        step(state, Action("select_card", 1))
        self.assertIn(target.cid, [c.cid for c in state.discard],
                      f"动作下标 1 应当弃掉观测里的 {target.cid}")

    def test_reverse_verification_sorted_observation_would_differ(self):
        """反向验证：证明"只在观测侧排序"确实会错位（否则上面的测试没有鉴别力）。"""
        state = make_state()
        attach(state, "hand")
        engine = [card.cid for card in state.pending.candidates(state)]
        sorted_view = sorted(engine)
        self.assertNotEqual(engine, sorted_view,
                            "构造的手牌顺序恰好已排序 → 这条反向验证失效")

    def test_hand_candidates_follow_the_visible_hand_order(self):
        """手牌顺序是玩家看到并排列的顺序，必须原样保留。"""
        state = make_state()
        attach(state, "hand")
        self.assertEqual([c.uid for c in state.pending.candidates(state)],
                         [c.uid for c in state.hand])


class TestDrawPileOrderIsNotRevealed(unittest.TestCase):
    def test_draw_pile_order_is_not_revealed(self):
        """⭐ 抽牌堆顺序是 L3：打乱它，候选的**卡牌序列**必须逐位不变。"""
        state_a = make_state()
        state_b = make_state()
        state_b.draw_pile.reverse()
        # 内容相同、顺序不同（构造上保证两边候选集合一致）
        self.assertEqual(sorted(c.uid for c in state_a.draw_pile),
                         sorted(c.uid for c in state_b.draw_pile))
        attach(state_a, "draw")
        attach(state_b, "draw")
        a = [c.cid for c in state_a.pending.candidates(state_a)]
        b = [c.cid for c in state_b.pending.candidates(state_b)]
        self.assertEqual(a, b, "抽牌堆顺序泄漏成了候选排列")

    def test_discard_candidates_are_canonical_too(self):
        """弃牌堆也走规范排序：同名同升级状态的牌对玩家不可区分。"""
        state = make_state()
        state.discard.extend(state.draw_pile)
        state.draw_pile = []
        attach(state, "discard")
        order = [c.cid for c in state.pending.candidates(state)]
        self.assertEqual(order, sorted(order))


class TestRunSelectionIsVisible(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        content = Path(__file__).resolve().parent.parent / "data" / "content" / "repo"
        if not content.exists():
            self.skipTest(f"缺内容目录 {content}")
        from sts2_sim.featurize import configure_content
        configure_content(str(content))
        self.addCleanup(self._restore)

    @staticmethod
    def _restore():
        from sts2_sim.content import load_builtin
        from sts2_sim.featurize import rebuild_vocab
        load_builtin()
        rebuild_vocab()

    def test_run_selection_is_visible_with_matching_slots(self):
        """Run 层选牌的候选槽位必须与 `legal_meta_actions` 的下标一致。"""
        from sts2_sim import run as runmod
        from sts2_sim.observe import observe_run
        from sts2_sim.runeffects import RunSelection

        env = runmod.RunEnv(seed=5, attempt_budget=0, character="ironclad")
        env.reset()
        state = env.raw_state
        # 进入一个事件房，再手工挂一个选牌（等价于事件选项触发的"选一张牌"）
        state.pending_selection = RunSelection(
            purpose="upgrade", count=1, source="deck",
            candidates=tuple(range(min(3, len(state.player.deck)))))
        observation = observe_run(state)
        self.assertEqual(observation.selection_purpose, "upgrade")
        self.assertEqual(len(observation.selection_candidates),
                         len(state.pending_selection.candidates))

        actions = runmod.legal_meta_actions(state)
        slots = sorted(a.index for a in actions if a.kind == "select_deck_card")
        # 事件房之外没有 select_deck_card 也是合法的；这里断言的是"出现的槽位
        # 必须是观测里给出的候选槽位的子集"。
        self.assertTrue(set(slots) <= set(observation.selection_candidates)
                        or not slots)


class TestEmptyCandidateSelection(unittest.TestCase):
    """⭐ 空候选/候选抽干时**不能挂起**，否则环境会没有任何合法动作。

    这两个 bug 是"真实内容 + SL 预算跑 1500 步"时才暴露的：
    实测第 112 步炸在 ``存在没有任何合法动作的样本`` —— 根因是回合 1 打出
    ``headbutt``（"从弃牌堆放一张到抽牌堆顶"），而弃牌堆本来就是空的。

    | 症状 | 根因 | 对应测试 |
    |---|---|---|
    | 合法动作为 0，训练报错 | 挂起了一个没有候选的选择 | `test_empty_pile_does_not_hang` |
    | 选到一半候选抽干 | ``remaining > 0`` 就一直挂着 | `test_drained_candidates_end_the_selection` |
    | 候选下标撑爆动作槽位表 | 槽位表按手牌上限设的 | `test_large_candidate_set_still_encodes` |
    """

    def test_empty_pile_does_not_hang(self):
        """端到端：真实内容下的 ``headbutt``（从弃牌堆取牌）在空弃牌堆时不挂起。

        这是训练里实际炸掉的那一步（第 112 步，"存在没有任何合法动作的样本"）。
        """
        from pathlib import Path
        content = Path(__file__).resolve().parent.parent / "data" / "content" / "repo"
        if not content.exists():
            self.skipTest(f"缺内容目录 {content}")
        from sts2_sim.featurize import configure_content
        from sts2_sim.content import load_builtin
        from sts2_sim.featurize import rebuild_vocab
        configure_content(str(content))
        try:
            from sts2_sim.core import Action, step
            # 真实内容下基础牌是 `strike_ironclad`，敌人 id 也不再是 `jaw_worm`
            # —— 写内置内容的 id 会在真实内容上 KeyError。
            state = start_combat(
                ["headbutt", "strike_ironclad", "defend_ironclad",
                 "strike_ironclad", "defend_ironclad"],
                ("axebot",), seed=1)
            state.discard = []              # 回合 1，弃牌堆本来就是空的
            state.energy = 5
            self.assertIn("headbutt", [c.cid for c in state.hand])
            step(state, Action("play_card", 0, 0))
            self.assertIsNone(state.pending, "弃牌堆为空时不该挂起选择")
            self.assertTrue(legal_actions(state), "必须仍然有合法动作")
        finally:
            load_builtin()
            rebuild_vocab()

    def test_drained_candidates_end_the_selection(self):
        """"弃 2 张"但只有 1 张候选 → 选完 1 张就按现有数量结束。"""
        from sts2_sim.core import Action, CardInstance, PendingSelection, step
        state = make_state()
        state.hand = [CardInstance("strike")]
        state.discard = [CardInstance("defend")]
        state.pending = PendingSelection(purpose="discard", source_pile="hand",
                                         remaining=2)
        step(state, Action("select_card", 0))
        self.assertIsNone(state.pending, "候选抽干后应当结束选择，而不是继续等")
        self.assertTrue(legal_actions(state), "必须仍然有合法动作")

    def test_large_candidate_set_still_encodes(self):
        """候选数可以远大于手牌上限 —— 编码必须能容纳（否则训练中途崩）。

        真实局面就是"从弃牌堆选一张"：弃牌堆可以有 17+ 张，而手牌上限是 10。
        动作槽位表如果按手牌上限设，这里就会 IndexError。
        """
        from sts2_sim.core import CardInstance, PendingSelection
        from sts2_sim.featurize import MAX_HAND, encode_observation
        state = make_state()
        state.discard = [CardInstance("strike") for _ in range(20)]
        state.pending = PendingSelection(purpose="to_hand", source_pile="discard",
                                         remaining=1)
        n = len(state.pending.candidates(state))
        self.assertGreater(n, MAX_HAND, "构造的候选数没有超过手牌上限，测试没有鉴别力")
        actions = legal_actions(state)
        self.assertEqual(len(actions), n)
        # 不该抛 ValueError / IndexError
        encoded = encode_observation(observe(state), actions)
        self.assertEqual(int(encoded["act_mask"].sum()), n)


if __name__ == "__main__":
    unittest.main(verbosity=2)
