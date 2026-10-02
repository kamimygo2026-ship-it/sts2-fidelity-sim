"""**W01 后半**：KnowledgeDemon 的 Curse of Knowledge 三轮二选一（入口链路）。

`docs/17` P04 的入口。依赖两件本批刚交付的东西：

* ``docs/12`` §2.38 的 **S02 可挂起执行帧**（怪物出招中途停住等玩家选择）；
* 抽取器把 ``CurseOfKnowledgeMove`` 抽成 ``effects_by_counter``（三组候选 + 6/7/8）
  与 ``counter_increment``（招式走完递增 ``_curseOfKnowledgeCounter``）。

真机出处（``KnowledgeDemon.cs``）：

* ``:36-52`` 静态表：``_disintegrationDamageValues = { 6, 7, 8 }`` 与
  ``_curseOfKnowledgeSets`` 三组各两张卡
  （``Disintegration`` + ``MindRot`` / ``Sloth`` / ``WasteAway``）；
* ``:157-176`` ``CurseOfKnowledgeMove``：对每个 target 跑 ``ChooseCurse``
  （造两张卡 → ``FromChooseACardScreen`` → 选中的 ``OnChosen``），
  战斗仍进行时 ``CurseOfKnowledgeCounter++``；
* ``:210-211`` 分支：``Respawns``-式的计数器 ``< 3`` 才继续出 ``CURSE_OF_KNOWLEDGE``。

⚠️ 本文件同时钉住**抽取器**的一条危险回归：只有**调用选择辅助方法**的那一招
才能拿到候选载荷。少了这条判据，整只怪的每一条攻击招式都会被贴上同一份候选
（实测：Slap / KnowledgeOverwhelming / Ponder 在 ``counter < 3`` 时全部变成
"不打伤害、改去让玩家选卡"），而日志完全正常。
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"
DEMON = (ROOT / "data" / "decompiled" / "sts2"
         / "MegaCrit.Sts2.Core.Models.Monsters" / "KnowledgeDemon.cs")


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


class KnowledgeDemonEntryTest(unittest.TestCase):
    """三轮二选一的完整链路（挂起 → 选 → 施加 → 计数 → 分支）。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, hp: int = 3000):
        """一场知识恶魔战。默认把玩家血量拉高：本文件测**机制**，不是测能不能打赢。"""
        from sts2_sim import core
        state = core.start_combat(self.deck, ("knowledge_demon",), seed=5)
        state.player.max_hp = hp
        state.player.hp = hp
        return state

    def end_turn(self, state, choose: int = 0):
        """跑一个玩家回合结束，遇到二选一时选第 ``choose`` 张。

        返回这一次出现的候选（没有选择就返回 ``None``）。
        """
        from sts2_sim import core
        core.step(state, core.Action("end_turn"))
        offered = None
        while state.pending is not None:
            candidates = state.pending.candidates(state)
            if offered is None:
                offered = [card.cid for card in candidates]
            core.step(state, core.Action("select_card", min(choose, len(candidates) - 1)))
        return offered

    def test_the_three_rounds_offer_the_right_pairs(self):
        """候选对按源码顺序：Disintegration + MindRot / Sloth / WasteAway。"""
        state = self.combat()
        rounds = []
        for _ in range(13):
            offered = self.end_turn(state)
            if offered:
                rounds.append(offered)
            if len(rounds) == 3:
                break
        self.assertEqual(rounds, [["disintegration", "mind_rot"],
                                  ["disintegration", "sloth"],
                                  ["disintegration", "waste_away"]])

    def test_disintegration_carries_the_six_seven_eight_values(self):
        """三轮都选 Disintegration：6 → +7 → +8，层数累计 21。

        数值来自 ``_disintegrationDamageValues``，而且第 n 轮是**运行时**
        改写的 ``BaseValue``（同一个 cid、三份不同载荷）—— 所以候选效果必须
        随轮次走，不能靠 cid 回查卡片数据。
        """
        from sts2_sim import powers as power_rules
        state = self.combat()
        seen = []
        for _ in range(13):
            self.end_turn(state, choose=0)
            seen.append(state.player.power("disintegration"))
            if state.enemies[0].ai_counters.get("_curseOfKnowledgeCounter") == 3:
                break
        self.assertEqual(state.player.power("disintegration"), 21, f"逐轮层数 {seen}")
        self.assertEqual(power_rules.sloth_cards_played(state.player), 0)

    def test_the_alternative_option_applies_its_own_power(self):
        """选第二张：第 1 轮 MindRot 1 层、第 2 轮 Sloth 3 层、第 3 轮 WasteAway 1 层。

        层数来自各自卡牌的 ``CanonicalVars``（``PowerVar<XxxPower>(Nm)``）。
        """
        state = self.combat()
        picks = []
        for _ in range(13):
            offered = self.end_turn(state, choose=1)
            if offered:
                picks.append(tuple(offered))
            if state.enemies[0].ai_counters.get("_curseOfKnowledgeCounter") == 3:
                break
        self.assertEqual(state.player.power("mind_rot"), 1)
        self.assertEqual(state.player.power("sloth"), 3)
        self.assertEqual(state.player.power("waste_away"), 1)
        self.assertEqual(state.player.power("disintegration"), 0, "没选它就不该有层数")

    def test_counter_advances_and_the_branch_switches_after_three(self):
        """三轮之后 ``>= 3`` 分支生效：不再出 ``curse_of_knowledge``。

        这条同时证明"招式走完的收尾"真的执行了 —— 计数器不递增的话，
        ``counter < 3`` 恒真、这只 Boss 会永远重复二选一。
        """
        state = self.combat()
        for _ in range(13):
            self.end_turn(state)
            if state.enemies[0].ai_counters.get("_curseOfKnowledgeCounter") == 3:
                break
        self.assertEqual(state.enemies[0].ai_counters["_curseOfKnowledgeCounter"], 3)
        history = state.enemies[0].move_history
        self.assertEqual(history.count("curse_of_knowledge"), 3, f"历史 {history}")
        # 之后 3 个回合只会走 slap / overwhelming / ponder
        for _ in range(3):
            self.end_turn(state)
        self.assertEqual(state.enemies[0].move_history.count("curse_of_knowledge"), 3)

    def test_each_curse_happens_once_per_cycle(self):
        """一个循环里只有一次选择（不是每回合都问）。"""
        state = self.combat()
        picks = 0
        for _ in range(4):
            if self.end_turn(state):
                picks += 1
        self.assertEqual(picks, 1, "一个 4 回合循环里只该出现一次二选一")

    def test_the_counter_is_not_incremented_after_the_combat_is_over(self):
        """⭐ 真机的自增带 ``if (base.CombatState.IsLiveCombat())`` 守卫。

        战斗已分出胜负还记账，会让"第 3 次"这类分支在战斗结束后被多推一格。
        """
        from sts2_sim import core
        state = self.combat()
        enemy = state.enemies[0]
        move = next(m for m in enemy.definition().moves if m.mid == "curse_of_knowledge")
        before = enemy.ai_counters["_curseOfKnowledgeCounter"]
        state.phase = "won"
        core._finish_enemy_move(state, enemy, move, [])
        self.assertEqual(enemy.ai_counters["_curseOfKnowledgeCounter"], before)
        state.phase = "player"
        core._finish_enemy_move(state, enemy, move, [])
        self.assertEqual(enemy.ai_counters["_curseOfKnowledgeCounter"], before + 1)

    def test_snapshot_mid_choice_keeps_the_same_round(self):
        """挂起时快照 → 恢复 → 同一轮的候选与选后结果都一致（SL 语义）。"""
        from sts2_sim import core
        state = self.combat()
        core.step(state, core.Action("end_turn"))
        self.assertIsNotNone(state.pending)
        saved = core.snapshot(state)

        restored = core.restore(saved)
        self.assertEqual([c.cid for c in restored.pending.candidates(restored)],
                         ["disintegration", "mind_rot"])
        core.step(restored, core.Action("select_card", 1))
        self.assertEqual(restored.player.power("mind_rot"), 1)
        self.assertEqual(restored.enemies[0].ai_counters["_curseOfKnowledgeCounter"], 1)

        core.step(state, core.Action("select_card", 0))
        self.assertEqual(state.player.power("disintegration"), 6)


class ExtractionShapeTest(unittest.TestCase):
    """抽取器只给**真的会问玩家**的那一招贴候选（本轮抓到的危险回归）。"""

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_only_the_choosing_move_gets_the_payload(self):
        from tools.extract_moves import _choice_helper_names, extract_choose_sets
        from tools.extract_monsters import method_body
        source = DEMON.read_text(encoding="utf-8", errors="replace")
        helpers = _choice_helper_names(source)
        self.assertIn("ChooseCurse", helpers, "选择命令的辅助方法必须被认出来")

        curse = method_body(source, "CurseOfKnowledgeMove")
        self.assertIsNotNone(extract_choose_sets(source, curse, helpers),
                             "招式体调用 ChooseCurse → 它才是二选一招式")
        for handler in ("SlapMove", "KnowledgeOverwhelmingMove", "PonderMove"):
            body = method_body(source, handler)
            self.assertIsNone(
                extract_choose_sets(source, body, helpers),
                f"{handler} 没有问玩家，不该拿到候选载荷")

    def test_counter_names_are_normalized_to_the_registry(self):
        """``CurseOfKnowledgeCounter``（属性）→ ``_curseOfKnowledgeCounter``（登记名）。

        不规范化的话，运行时查表永远落到默认值 0 —— 每一轮都给第一组载荷，
        而且日志完全正常（实测踩过）。
        """
        from sts2_sim import monster_ai
        from tools.extract_moves import _counter_key
        key = _counter_key("CurseOfKnowledgeCounter")
        self.assertEqual(key, "_curseOfKnowledgeCounter")
        self.assertIn(key, monster_ai.COUNTER_SOURCES)

    def test_loaded_move_has_three_groups_and_a_counter_increment(self):
        from sts2_sim.content import ENEMY_DB
        moves = {m.mid: m for m in ENEMY_DB["knowledge_demon"].moves}
        curse = moves["curse_of_knowledge"]
        self.assertEqual(curse.effects_by_counter[0], "_curseOfKnowledgeCounter")
        self.assertEqual(len(curse.effects_by_counter[1]), 3)
        self.assertEqual(curse.counter_increment, "_curseOfKnowledgeCounter")
        self.assertTrue(curse.counter_increment_live_only)
        for mid in ("slap", "knowledge_overwhelming", "ponder"):
            self.assertFalse(moves[mid].effects_by_counter, f"{mid} 不该有二选一载荷")


if __name__ == "__main__":
    unittest.main()
