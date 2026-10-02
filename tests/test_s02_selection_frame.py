"""**S02 · 可挂起的执行帧**：怪物出招中途等玩家选择的回归。

交付对象是 ``docs/17`` §2 的 S02 与 W01 后半（KnowledgeDemon 的三轮二选一）所需的
**引擎底座**。真机的怪物动作是 async 序列：``KnowledgeDemon.CurseOfKnowledgeMove``
在循环里 ``await`` 玩家的二选一，await 期间整个战斗状态原样停住
（``KnowledgeDemon.cs:157-176``）。引擎的 ``step`` 是同步的，所以必须把
"跑到哪了"存成**数据**（``core.EnemyActionFrame``），否则 ``snapshot()``
（``copy.deepcopy``）装不下闭包/生成器，SL 直接废掉。

本文件用一条**手工构造**的二选一怪物招式驱动，不依赖抽取器 ——
抽取器把 ``CurseOfKnowledgeMove`` 抽成 DSL 是下一批的工作。这样 S02 的
挂起 / 续跑 / 快照三条语义可以**先被钉住**，数据到位时不必再动引擎。
"""

from __future__ import annotations

import unittest
from dataclasses import replace
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


def curse_choice_effect():
    """一条 ``choose_one``：两个候选分别给玩家上 ``mind_rot`` / ``waste_away``。

    ⚠️ ``target="enemy"`` 是**怪物视角**（真机的效果按施法者视角解释）——
    这正是本文件最有鉴别力的一处：如果续跑时把 ``source`` 错当成玩家，
    这两个效果会打到**怪物自己**身上，而且不会报任何错。
    """
    from sts2_sim.content import Effect
    return Effect("choose_one", choices=(
        ("mind_rot", (Effect("apply_power", 2, "mind_rot", target="enemy"),)),
        ("waste_away", (Effect("apply_power", 1, "waste_away", target="enemy"),)),
    ))


class SelectionFrameTest(unittest.TestCase):
    """挂起 → 选牌 → 续跑 → 推进回合的完整链路。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def setUp(self):
        from sts2_sim import content
        self._original = content.ENEMY_DB["nibbit"]
        self.addCleanup(content.ENEMY_DB.__setitem__, "nibbit", self._original)
        definition = self._original
        move = content.MoveDef(mid="curse_test", name="Curse Test", intent="debuff",
                               value=0, times=1, effects=(curse_choice_effect(),))
        content.ENEMY_DB["nibbit"] = replace(
            definition, moves=tuple(definition.moves) + (move,))

    def combat(self, enemies=("nibbit",), seed: int = 5, choosers=None):
        """开一场战斗，把指定怪物的意图钉成 ``curse_test``。

        ``choosers`` 是"哪几只怪用二选一招"，默认全部；其余怪保持原本的意图。
        """
        from sts2_sim import core
        state = core.start_combat(self.deck, tuple(enemies), seed=seed)
        for index, enemy in enumerate(state.enemies):
            if choosers is None or enemy.eid in choosers:
                enemy.intent = core.Intent("debuff", "curse_test", 0, 1)
        return state

    # ---- 挂起 ------------------------------------------------------------
    def test_enemy_turn_suspends_on_the_choice(self):
        """``step(end_turn)`` 撞上二选一时必须**当场停住**，不许把回合跑完。

        停住的三个证据：``state.pending``（等谁）、``state.enemy_action``
        （跑到哪了）、``state.phase == "enemy"``（还在敌方回合内）。
        """
        from sts2_sim import core
        state = self.combat()
        result = core.step(state, core.Action("end_turn"))
        self.assertFalse(result.done, "挂在选择上时这一步不算结束")
        self.assertIsNotNone(state.pending)
        self.assertIsNotNone(state.enemy_action, "必须记下敌方回合的进度")
        self.assertEqual(state.phase, "enemy")
        self.assertTrue(state.selecting())

    def test_legal_actions_are_exactly_the_two_candidates(self):
        """挂起期间合法动作**只有**那两个候选（其它动作全部非法）。"""
        from sts2_sim import core
        state = self.combat()
        core.step(state, core.Action("end_turn"))
        actions = core.legal_actions(state)
        self.assertEqual([a.kind for a in actions], ["select_card", "select_card"])
        candidates = state.pending.candidates(state)
        self.assertEqual([c.cid for c in candidates], ["mind_rot", "waste_away"],
                         "候选顺序 = 界面顺序，不得排序")

    # ---- 选中与续跑 ------------------------------------------------------
    def test_chosen_option_applies_to_the_correct_side(self):
        """选中的候选执行自己的效果，**按怪物视角**解释目标（打到玩家）。"""
        from sts2_sim import core
        state = self.combat()
        core.step(state, core.Action("end_turn"))
        core.step(state, core.Action("select_card", 0))
        self.assertEqual(state.player.power("mind_rot"), 2, "玩家吃到 2 层")
        self.assertEqual(state.enemies[0].power("mind_rot"), 0, "怪物自己不该吃到")

    def test_unselected_option_does_not_take_effect(self):
        from sts2_sim import core
        state = self.combat()
        core.step(state, core.Action("end_turn"))
        core.step(state, core.Action("select_card", 1))
        self.assertEqual(state.player.power("waste_away"), 1)
        self.assertEqual(state.player.power("mind_rot"), 0)

    def test_the_frame_clears_and_the_turn_advances(self):
        """续跑完：帧清空、敌方回合收尾、回合号 +1、玩家回合正常开始。"""
        from sts2_sim import core
        state = self.combat()
        core.step(state, core.Action("end_turn"))
        core.step(state, core.Action("select_card", 0))
        self.assertIsNone(state.pending)
        self.assertIsNone(state.enemy_action)
        self.assertEqual(state.phase, "player")
        self.assertEqual(state.turn, 2)
        self.assertTrue(state.hand, "新回合应当照常抽牌")

    def test_ending_the_player_turn_does_not_duplicate_the_move(self):
        """续跑**不能**把已经出过手的怪再打一次。

        `_run_enemy_actions` 从 `frame.enemy_index + 1` 接着跑，所以这一招的
        历史只有一条。从 0 重跑会让日志看起来只是"多打了一下"，不报错。
        """
        from sts2_sim import core
        state = self.combat()
        core.step(state, core.Action("end_turn"))
        core.step(state, core.Action("select_card", 0))
        self.assertEqual(state.enemies[0].move_history.count("curse_test"), 1)

    def test_a_second_enemy_still_acts_after_the_resume(self):
        """第一只怪挂起、续跑之后，**第二只怪也要照常出招**（不能被跳过）。"""
        from sts2_sim import core
        state = self.combat(enemies=("nibbit", "axebot"), choosers=("nibbit",))
        core.step(state, core.Action("end_turn"))
        self.assertIsNotNone(state.pending)
        core.step(state, core.Action("select_card", 0))
        self.assertTrue(state.enemies[1].move_history, "第二只怪必须出过招")
        self.assertEqual(state.turn, 2)

    # ---- SL / 快照 -------------------------------------------------------
    def test_snapshot_mid_selection_restores_the_choice(self):
        """挂起时快照 → 恢复 → 候选一致、两条分支各自成立（SL 的物理基础）。"""
        from sts2_sim import core
        state = self.combat()
        core.step(state, core.Action("end_turn"))
        saved = core.snapshot(state)

        restored = core.restore(saved)
        self.assertIsNotNone(restored.pending)
        self.assertIsNotNone(restored.enemy_action)
        self.assertEqual(
            [c.cid for c in restored.pending.candidates(restored)],
            [c.cid for c in state.pending.candidates(state)])

        # 恢复出来的世界选第 0 张；原世界走另一条分支 —— 互不干扰。
        core.step(restored, core.Action("select_card", 0))
        self.assertEqual(restored.player.power("mind_rot"), 2)
        self.assertEqual(restored.turn, 2)

        core.step(state, core.Action("select_card", 1))
        self.assertEqual(state.player.power("waste_away"), 1)
        self.assertEqual(state.player.power("mind_rot"), 0)

    def test_snapshot_round_trip_keeps_the_frame_serializable(self):
        """帧必须是**纯数据**（``deepcopy`` 装得下）—— 塞闭包就会在这里炸。"""
        import copy
        from sts2_sim import core
        state = self.combat()
        core.step(state, core.Action("end_turn"))
        clone = copy.deepcopy(state)
        self.assertEqual(clone.enemy_action.move_mid, "curse_test")
        self.assertEqual(clone.pending.purpose, "choose")

    # ---- 观测 ------------------------------------------------------------
    def test_observation_offers_the_two_cards(self):
        # ⚠️ `sts2_sim.observe` 是**函数**（`__init__.py` 的导出），不是模块 ——
        # 必须从 `sts2_sim.observe` 子模块导入。
        from sts2_sim import core
        from sts2_sim.observe import observe
        state = self.combat()
        core.step(state, core.Action("end_turn"))
        view = observe(state)
        self.assertIsNotNone(view.selection)
        self.assertEqual(len(view.selection.candidates), 2)
        self.assertEqual(view.selection.source_pile, "offered")


class ChooseOneParsingTest(unittest.TestCase):
    """``choose_one`` 的数据侧：JSON → ``Effect.choices``。"""

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_choices_are_parsed_recursively(self):
        from sts2_sim import content
        raw = [{"op": "choose_one", "choices": [
            {"card": "mind_rot",
             "effects": [{"op": "apply_power", "amount": 1, "power": "mind_rot",
                          "target": "enemy"}]},
            {"card": "waste_away",
             "effects": [{"op": "apply_power", "amount": 1, "power": "waste_away",
                          "target": "enemy"}]},
        ]}]
        resolved = content._resolve_effects(raw, {})
        self.assertIsNotNone(resolved)
        effect = resolved[0]
        self.assertEqual([label for label, _ in effect.choices],
                         ["mind_rot", "waste_away"])
        self.assertEqual(effect.choices[0][1][0].power, "mind_rot")

    def test_a_candidate_that_cannot_be_resolved_rejects_the_whole_effect(self):
        """候选里有落不进 DSL 的效果 → **整条** ``choose_one`` 返回 None。

        不能"跳过那个候选"：那等于**偷偷改掉玩家的选项**（而且报告是干净的）。
        """
        from sts2_sim import content
        raw = [{"op": "choose_one", "choices": [
            # `needs_runtime` 的量抽不出来 → 这个候选解析失败
            {"card": "mind_rot", "effects": [
                {"op": "apply_power", "amount": 1, "power": "mind_rot",
                 "target": "enemy", "needs_runtime": True}]},
            {"card": "waste_away", "effects": [
                {"op": "apply_power", "amount": 1, "power": "waste_away",
                 "target": "enemy"}]},
        ]}]
        self.assertIsNone(content._resolve_effects(raw, {}))

    def test_an_empty_choice_list_is_rejected(self):
        from sts2_sim import content
        self.assertIsNone(content._resolve_effects(
            [{"op": "choose_one", "choices": []}], {}))


if __name__ == "__main__":
    unittest.main()
