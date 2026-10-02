"""**W02 · P08** ``EntropyPower``：回合开始挑手牌**原位随机转化**。

两张源码：

* ``EntropyPower.cs:19-33``（能力）—— ``AfterPlayerTurnStart`` 里
  ``CardSelectCmd.FromHand(prefs(TransformSelectionPrompt, Amount))``，
  然后逐张 ``CardCmd.TransformToRandom(item, RunState.Rng.CombatCardSelection)``；
* ``CardFactory.cs:170-212``（候选池）—— 原卡的 ``Pool``（Quest/Event/Ancient/Token
  换 ``ColorlessCardPool``）→ 原卡不是 Status/Curse 时只允许 Common/Uncommon/Rare
  （判据 ``(uint)(rarity - 8) > 1u``）→ ``CanBeGeneratedInCombat`` → 排除原卡 id；
* ``CardCmd.cs:371-407``（执行）—— 记下原卡的堆与下标、移除、**原位插回**。

本文件钉住四件事：挂起的时机与位置、**原位替换**、候选池的过滤链、
以及"转化不算新增牌"（一换一，`added_cards` 不变）。
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


class EntropyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, powers=(), seed: int = 5):
        from sts2_sim import core
        state = core.start_combat(self.deck, ("nibbit",), seed=seed)
        state.player.max_hp = state.player.hp = 3000
        for pid, amount in powers:
            state.player.add_power(pid, amount)
        return state


class SuspendTest(EntropyTestCase):
    """挂起发生在**回合开始、抽牌之后**（`AfterPlayerTurnStart`）。"""

    def test_the_turn_start_suspends_on_the_transform_choice(self):
        from sts2_sim import core
        state = self.combat([("entropy", 2)])
        core.start_player_turn(state, [])
        self.assertIsNotNone(state.pending)
        self.assertEqual(state.pending.purpose, "transform")
        self.assertEqual(state.pending.source_pile, "hand")
        self.assertEqual(state.pending.remaining, 2)
        self.assertTrue(state.turn_start_step,
                        "要记下回合开始流程跑到第几步，选完接着跑")

    def test_the_hand_is_already_drawn_when_it_asks(self):
        """时机在**抽牌之后**：挂起时手牌已经抽好了（不是空的）。"""
        from sts2_sim import core
        state = self.combat([("entropy", 1)])
        core.start_player_turn(state, [])
        self.assertGreaterEqual(len(state.hand), 5)

    def test_the_rest_of_the_turn_start_still_runs_after_the_choice(self):
        """⭐ 选完之后，回合开始的**剩余步骤**必须照跑。

        真机整个 ``SetupPlayerTurn`` 停在那个 ``await`` 上。少跑一段不报错，
        只是那一回合"少了一段流程"（如 `AfterSideTurnStart` 的下回合格挡）。
        """
        from sts2_sim import core, hooks
        state = self.combat([("entropy", 1)])
        with hooks.trace() as record:
            core.start_player_turn(state, [])
            self.assertIsNotNone(state.pending)
            core.step(state, core.Action("select_card", 0))
        names = record.names()
        for expected in ("on_owner_turn_start", "on_auto_pre_play"):
            self.assertIn(expected, names, f"续跑后仍然要分发 {expected}")
        self.assertEqual(state.turn_start_step, 0, "续跑完要把下标清零")


class TransformTest(EntropyTestCase):
    """`CardCmd.Transform` 的语义：原位替换、一换一。"""

    def test_the_chosen_card_is_replaced_in_place(self):
        """⭐ 位置不变：真机先记下标、移除、再插回**原位**。

        换成"先移出手牌再追加到末尾"会让玩家的手牌顺序乱掉 ——
        而观测里手牌是**有序**的（`observe.py` 用手牌顺序当候选顺序）。
        """
        from sts2_sim import core
        state = self.combat([("entropy", 1)])
        core.start_player_turn(state, [])
        index = 2
        target = state.pending.candidates(state)[index]
        others = [c.uid for i, c in enumerate(state.hand) if i != index]
        core.step(state, core.Action("select_card", index))
        self.assertEqual(len(state.hand), 10, "张数不变（一换一）")
        self.assertNotIn(target.uid, [c.uid for c in state.hand])
        self.assertEqual([c.uid for i, c in enumerate(state.hand) if i != index],
                         others, "其余牌的位置一动不动")

    def test_the_replacement_is_a_brand_new_instance(self):
        """新卡是**新实例**（真机 `CreateCard`）：不继承升级 / 苦痛。"""
        from sts2_sim import core
        state = self.combat([("entropy", 1)])
        target = state.hand[0]
        target.upgraded = True
        target.affliction = "tangled"
        replacement = core.transform_card_random(state, target, [])
        self.assertIsNotNone(replacement)
        self.assertFalse(replacement.upgraded)
        self.assertIsNone(replacement.affliction)
        self.assertNotEqual(replacement.uid, target.uid)

    def test_transform_is_not_a_generated_card(self):
        """一换一 → **不**计入 `added_cards`（守恒校验靠它）。"""
        from sts2_sim import core
        state = self.combat([("entropy", 1)])
        before = state.added_cards
        total_before = len(state.all_piles())
        core.start_player_turn(state, [])
        core.step(state, core.Action("select_card", 0))
        self.assertEqual(state.added_cards, before)
        self.assertEqual(len(state.all_piles()), total_before)

    def test_the_power_is_kept(self):
        """源码里没有 `PowerCmd.Remove`：层数 = 每回合转几张。"""
        from sts2_sim import core
        state = self.combat([("entropy", 2)])
        core.start_player_turn(state, [])
        while state.pending is not None:
            core.step(state, core.Action("select_card", 0))
        self.assertEqual(state.player.power("entropy"), 2)


class OptionsTest(EntropyTestCase):
    """候选池 = 真机 `CardFactory.GetDefaultTransformationOptions` 的过滤链。"""

    def test_it_excludes_the_original_id(self):
        from sts2_sim import core
        state = self.combat()
        for card in list(state.hand):
            options = core.transformation_options(state, card)
            self.assertNotIn(card.cid, options, f"{card.cid} 不该出现在自己的候选里")

    def test_only_common_uncommon_rare_for_a_normal_card(self):
        """原卡不是 Status/Curse → 候选只允许 Common/Uncommon/Rare。

        出处：``(uint)(rarity - 8) > 1u``（``CardFactory.cs:193``），
        枚举里 8 = Status、9 = Curse。
        """
        from sts2_sim import content, core
        state = self.combat()
        card = next(c for c in state.hand if c.cid == "strike_ironclad")
        options = core.transformation_options(state, card)
        self.assertTrue(options)
        rarities = {content.CARD_DB[cid].rarity for cid in options}
        self.assertTrue(rarities <= {"common", "uncommon", "rare"}, rarities)

    def test_cards_that_cannot_be_generated_in_combat_are_excluded(self):
        """⭐ `CanBeGeneratedInCombat => false` 的牌不能作为转化结果。

        少了这一条，转化会发出真机永远发不出的牌（本地构建里 19 张，
        包括 `nightmare` / `mind_rot` 这类专属牌）。
        """
        from sts2_sim import content, core
        state = self.combat()
        card = next(c for c in state.hand if c.cid == "strike_ironclad")
        options = set(core.transformation_options(state, card))
        banned = {cid for cid, d in content.CARD_DB.items()
                  if not d.can_be_generated_in_combat}
        self.assertFalse(options & banned, options & banned)
        self.assertIn("nightmare", banned, "探针本身要有效")

    def test_a_status_card_draws_from_its_own_pool(self):
        """原卡是 Status → 候选来自**它自己的池**（StatusCardPool），不是无色池。

        出处 `CardFactory.cs:172`：换无色池的条件是 ``CardType.Quest``
        或稀有度 ∈ {Event, Ancient, Token} —— **不含** Status/Curse。
        把它们也算进去会多发一堆状态牌（实测踩过）。
        """
        from sts2_sim import content, core
        state = self.combat()
        wound = content.CARD_DB.get("wound")
        if wound is None or wound.rarity != "status":
            self.skipTest("内容里没有可用的状态牌")
        from sts2_sim.core import CardInstance
        options = set(core.transformation_options(state, CardInstance("wound")))
        status_pool = set(content.CARD_POOLS["StatusCardPool"])
        self.assertTrue(options <= status_pool - {"wound"},
                        f"越界的候选：{options - status_pool}")

    def test_a_quest_card_draws_from_the_colorless_pool(self):
        """⭐ `CardType.Quest` 看的是**卡类型**（不是稀有度）→ 换成无色池。"""
        from sts2_sim import content, core
        state = self.combat()
        quest = next((cid for cid, d in content.CARD_DB.items()
                      if d.card_type == "quest"), None)
        if quest is None:
            self.skipTest("内容里没有任务牌")
        from sts2_sim.core import CardInstance
        options = set(core.transformation_options(state, CardInstance(quest)))
        colorless = set(content.CARD_POOLS["ColorlessCardPool"])
        self.assertTrue(options <= colorless - {quest},
                        f"越界的候选：{options - colorless}")

    def test_an_empty_pool_is_refused_loudly(self):
        """⭐ 候选为空时必须**如实拒绝**，不能静默不转。

        真机在这里抛 ``All transformation options provided are invalid!``。
        """
        from sts2_sim import core
        state = self.combat()
        original = core.transformation_options
        core.transformation_options = lambda state_, card_: []
        try:
            events: list[str] = []
            card = state.hand[0]
            result = core.transform_card_random(state, card, events)
        finally:
            core.transformation_options = original
        self.assertIsNone(result)
        self.assertTrue(any("候选池为空" in line for line in events), events)
        self.assertIn(card, state.hand, "拒绝时不能动那张牌")


class SnapshotTest(EntropyTestCase):
    def test_snapshot_mid_choice_keeps_the_turn_start_step(self):
        from sts2_sim import core
        state = self.combat([("entropy", 1)])
        core.start_player_turn(state, [])
        saved = core.snapshot(state)
        restored = core.restore(saved)
        self.assertEqual(restored.pending.purpose, "transform")
        self.assertEqual(restored.turn_start_step, state.turn_start_step)
        core.step(restored, core.Action("select_card", 0))
        self.assertIsNone(restored.pending)
        self.assertEqual(restored.turn_start_step, 0)


if __name__ == "__main__":
    unittest.main()
