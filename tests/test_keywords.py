"""卡牌关键字（``CardKeyword``）的行为测试。

铁律：**每条断言都要能追到源码**（``docs/09`` §5.2 R1/R3）。
用例按关键字分组，注释里给出真机执行点；测试名里带关键字名，便于按关键字筛选。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from sts2_sim import content, keywords
from sts2_sim.core import Action, CardInstance, start_combat, start_player_turn, step

CONTENT = Path(__file__).resolve().parent.parent / "data" / "content" / "repo"


def setUpModule() -> None:               # noqa: N802 - unittest 约定
    """内容层必须加载：这些用例全部在真实卡表上跑，不用手写假数据。"""
    content.load_content_dir(str(CONTENT))


def combat(deck: list[str], enemies: tuple[str, ...] = ("calcified_cultist",), **kw):
    """起一场战斗；牌序由种子决定，用例自己把牌摆好再断言。"""
    return start_combat(deck, list(enemies), seed=7, **kw)


class TestKeywordTable(unittest.TestCase):
    """关键字表本身：七个关键字必须有实现、有出处。"""

    def test_all_seven_keywords_are_implemented_and_cited(self):
        """``CardKeyword.cs`` 里的 7 个取值，一个不少、且都能跑。"""
        expected = {"Exhaust", "Ethereal", "Innate", "Unplayable", "Retain",
                    "Sly", "Eternal"}
        self.assertEqual({k.name for k in keywords.KEYWORDS}, expected)
        self.assertEqual(set(keywords.ENGINE_KEYWORDS), expected)
        for spec in keywords.KEYWORDS:
            with self.subTest(keyword=spec.name):
                self.assertTrue(spec.source, "关键字必须带源码出处（铁律 R1）")
                self.assertIn(".cs:", spec.source)
                self.assertTrue(spec.behavior)

    def test_every_keyword_in_content_is_known(self):
        """内容里出现的关键字全部在引擎的表里（否则会被标残缺，不该静默）。"""
        unknown = {k for card in content.CARD_DB.values()
                   for k in card.keywords if k not in keywords.ENGINE_KEYWORDS}
        self.assertEqual(unknown, set())


class TestUnplayable(unittest.TestCase):
    """``CardModel.CanPlay``（CardModel.cs:1734）→ ``HasUnplayableKeyword``。

    ⚠️ 真机这一条**与费用无关**：``CanPlay`` 先看关键字、再看资源。
    现在内容里 27 张 Unplayable 恰好都是 -1 费，所以"负费用"这条判据也能挡；
    但那是**巧合**，不是机制 —— 用例用一张临时把费用改成 0 的牌来锁住这一点
    （真机里附魔/费用修正可以让它变成正费用，``EnchantmentModel.cs:285``
    就专门处理了"牌组里的不可打出牌"）。
    """

    def test_unplayable_card_with_non_negative_cost_is_still_rejected(self):
        from dataclasses import replace
        from unittest import mock

        real = content.CARD_DB["wound"]
        patched = replace(real, cost=0, effects=real.effects or ())
        with mock.patch.dict(content.CARD_DB, {"wound_zero_cost": patched}):
            self.assertFalse(CardInstance("wound_zero_cost").playable(),
                             "带 Unplayable 的牌即使费用非负也不能打出")

    def test_unplayable_card_is_not_a_legal_action(self):
        from sts2_sim.core import legal_actions
        state = combat(["wound", "strike_ironclad"] * 4,
                       enemies=("calcified_cultist",))
        wound = CardInstance("wound")
        state.hand = [wound]
        plays = [a for a in legal_actions(state) if a.kind == "play_card"]
        self.assertEqual(plays, [], "不可打出的牌不能出现在合法动作里")


class TestEthereal(unittest.TestCase):
    """``CombatManager.DoTurnEnd``（CombatManager.cs:1612-1626）→ 消耗。"""

    def test_ethereal_card_is_exhausted_at_turn_end_not_discarded(self):
        state = combat(["strike_ironclad"] * 30,
                       enemies=("calcified_cultist",))
        dazed = CardInstance("dazed")
        state.hand.append(dazed)
        step(state, Action(kind="end_turn"))
        self.assertIn(dazed, state.exhaust, "虚无牌回合结束应进消耗堆")
        self.assertNotIn(dazed, state.discard, "虚无牌不该进弃牌堆")

    def test_non_ethereal_card_is_discarded(self):
        """对照组：没有虚无关键字的牌照常弃掉。

        ⚠️ 牌组要做大：牌组太小时，下回合抽牌会把弃牌堆洗回抽牌堆，
        "弃没弃"就看不出差别了（这是测试写法问题，不是引擎问题）。
        """
        state = combat(["strike_ironclad"] * 30,
                       enemies=("calcified_cultist",))
        strike = state.hand[0]
        self.assertNotIn("Ethereal", strike.definition().keywords)
        step(state, Action(kind="end_turn"))
        self.assertIn(strike, state.discard)


class TestRetain(unittest.TestCase):
    """``CombatManager.FlushPlayerHand``（CombatManager.cs:1795-1805）→ 不弃。"""

    RETAINED = "eradicate"          # 0 费、带 Retain、可训练

    def test_retain_keyword_card_stays_in_hand(self):
        card = CardInstance(self.RETAINED)
        self.assertIn("Retain", card.definition().keywords, "前提：这张牌带保留")
        state = combat(["strike_ironclad"] * 30,
                       enemies=("calcified_cultist",))
        state.hand.append(card)
        step(state, Action(kind="end_turn"))
        self.assertIn(card, state.hand, "带保留的牌回合结束必须留在手里")

    def test_other_cards_still_flush_while_one_is_retained(self):
        """⚠️ 判据是**逐张**的：不能因为有一张保留牌就整手不清。"""
        state = combat(["strike_ironclad"] * 30,
                       enemies=("calcified_cultist",))
        retained = CardInstance(self.RETAINED)
        state.hand.append(retained)
        other = state.hand[0]
        step(state, Action(kind="end_turn"))
        self.assertIn(retained, state.hand, "保留牌留下")
        self.assertIn(other, state.discard, "没有保留关键字的牌该被弃掉")


class TestInnate(unittest.TestCase):
    """``CombatManager.SetupPlayerTurn``（CombatManager.cs:908-923）→ 起手。"""

    def test_innate_card_is_drawn_on_turn_one(self):
        # 牌组做大叫它不可能靠 5 张随机抽上来蒙对（1/13 左右，且要连中）。
        deck = ["backstab"] + ["strike_ironclad"] * 14
        innate = CardInstance("backstab")
        self.assertIn("Innate", innate.definition().keywords, "前提：带固有")
        state = combat(deck, enemies=("calcified_cultist",))
        self.assertIn("backstab", [c.cid for c in state.hand],
                      "固有点必须在第 1 回合上手（MoveToTop + max(handDraw, 张数)）")

    def test_innate_hoist_only_on_first_turn(self):
        """``TurnNumber == 1`` 才做 —— 第 2 回合起不再抬牌。"""
        deck = ["backstab"] + ["strike_ironclad"] * 14
        state = combat(deck, enemies=("calcified_cultist",))
        state.turn = 2
        events: list[str] = []
        start_player_turn(state, events)
        self.assertFalse(any("固有点" in e for e in events),
                         "第 2 回合不该再抬固有牌")


class TestSly(unittest.TestCase):
    """``CardCmd.DiscardAndDraw``（CardCmd.cs:184-204）→ 弃掉后自动打出。"""

    def test_sly_card_is_autoplayed_after_discard(self):
        from sts2_sim.core import _autoplay_sly
        state = combat(["strike_ironclad"], enemies=("calcified_cultist",))
        state.hand = []
        state.draw_pile = [CardInstance("strike_ironclad") for _ in range(4)]
        card = CardInstance("reflex")          # Sly，效果是抽 2 张
        state.discard.append(card)
        events: list[str] = []
        _autoplay_sly(state, card, events)
        self.assertTrue(any("狡诈" in e for e in events),
                        f"应记录一次狡诈自动打出：{events}")
        self.assertEqual(len(state.hand), 2, "自动打出必须真的结算效果（抽 2）")

    def test_non_sly_card_is_untouched(self):
        from sts2_sim.core import _autoplay_sly
        state = combat(["strike_ironclad"], enemies=("calcified_cultist",))
        card = CardInstance("strike_ironclad")
        state.discard.append(card)
        events: list[str] = []
        _autoplay_sly(state, card, events)
        self.assertEqual(events, [], "非狡诈牌不该被自动打出")

    def test_unplayable_sly_card_does_not_resolve_effects(self):
        """``CardCmd.AutoPlay``（CardCmd.cs:58-62）→ 不可打出只落牌堆、**不结算**。

        内容里目前没有"既狡诈又不可打出"的牌（``MasterPlannerPower`` 是给
        **已打出的牌**贴狡诈），所以这里用一张临时卡把这条分支锁住 ——
        机制在源码里存在，就必须有测试覆盖。
        """
        from dataclasses import replace
        from unittest import mock

        from sts2_sim.core import _autoplay_sly
        sly_unplayable = replace(content.CARD_DB["reflex"], cost=-1,
                                 keywords=("Sly", "Unplayable"))
        state = combat(["strike_ironclad"], enemies=("calcified_cultist",))
        state.hand = []
        state.draw_pile = [CardInstance("strike_ironclad") for _ in range(4)]
        card = CardInstance("reflex")
        with mock.patch.dict(content.CARD_DB, {"reflex": sly_unplayable}):
            state.discard.append(card)
            events: list[str] = []
            _autoplay_sly(state, card, events)
        self.assertEqual(state.hand, [], "不可打出的狡诈牌不该结算抽牌效果")
        self.assertTrue(any("不可打出" in e for e in events), events)


class TestEternal(unittest.TestCase):
    """``CardModel.IsRemovable``（CardModel.cs:738）→ 不能被移除。"""

    def test_eternal_card_cannot_be_removed_from_deck(self):
        from sts2_sim.content import Effect
        from sts2_sim.rng import RngSet
        from sts2_sim.runeffects import apply_run_effects

        class FakePlayer:
            def __init__(self) -> None:
                self.hp = 80
                self.max_hp = 80
                self.gold = 0
                self.deck = [CardInstance("ascenders_bane")]

        class FakeRun:
            def __init__(self) -> None:
                self.player = FakePlayer()
                self.hidden = type("H", (), {"rng": RngSet(1)})()

        run = FakeRun()
        self.assertIn("Eternal",
                      run.player.deck[0].definition().keywords, "前提：这张牌是永恒")
        events: list[str] = []
        apply_run_effects(run, (Effect(op="remove_card_from_deck",
                                       card="ascenders_bane"),), events)
        self.assertEqual([c.cid for c in run.player.deck], ["ascenders_bane"],
                         "永恒牌不能被移除")
        self.assertTrue(any("永恒" in e for e in events), events)


if __name__ == "__main__":
    unittest.main(verbosity=2)
