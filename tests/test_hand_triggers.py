"""手牌触发器、duration 时机、牌堆移动的回归测试（``docs/09`` L3 续）。

对应已真实发生过的 bug，每个都属于"卡看起来还在、效果却没了"那一类：

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 手里的燃烧/腐朽牌**毫无代价**，模型学会囤牌 | `CardDef` 根本没有 `triggers` 字段，12 张卡的触发器被静默丢弃 | `test_burn_damages_at_end_of_turn` |
| 只带触发器、没有 OnPlay 效果的卡**整条被跳过** | 采用条件写成"没有效果就跳过"，而这些状态牌的正常效果就在触发器里 | `test_trigger_only_cards_are_adopted` |
| "每次掉 13 点血"变成"掉 0 点" | 量与变量都缺失时静默归零 | `test_unresolvable_amount_is_not_zeroed` |
| 玩家的易伤在**玩家回合结束**就递减（太早） | 真机三个 duration 能力的递减条件是 `side == CombatSide.Enemy` | `test_duration_ticks_at_enemy_side_turn_end` |
| Ethereal 的牌被弃掉而不是消耗 → 一直在牌库循环 | 弃手牌时没区分 Ethereal | `test_ethereal_cards_are_exhausted` |
| `Dredge` 变成"选完放进弃牌堆"（正好相反） | `resolve_pile` 在整串参数里搜 `PileType`，搜到的是**嵌套选牌的来源**牌堆 | `test_pile_target_is_the_top_level_argument` |
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


def _source(cid: str) -> dict:
    cards = json.loads((CONTENT / "cards_source.json").read_text(encoding="utf-8"))
    return next(c for c in cards if c["cid"] == cid)


class TestHandTriggers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat_with_hand(self, cids: list[str]):
        from sts2_sim.core import CardInstance, start_combat
        state = start_combat(self.deck, ("nibbit",), seed=5)
        state.player.hp = state.player.max_hp = 200
        state.hand = [CardInstance(c) for c in cids]
        return state

    def _hp_loss_over_a_turn(self, cids: list[str]) -> int:
        """跑完一个回合的掉血量。**必须减去敌人的攻击** —— 直接断言绝对值会把
        敌人那一下算进来（实测 2 张燃烧期望 4、实际 12，差的就是敌人打的 8）。"""
        from sts2_sim.core import Action, step
        state = self._combat_with_hand(cids)
        before = state.player.hp
        step(state, Action(kind="end_turn"))
        return before - state.player.hp

    def test_burn_damages_at_end_of_turn(self):
        """``Burn.OnTurnEndInHand``：回合结束时若还在手里，每张造成 2 点伤害。

        不接触发器的话状态牌毫无代价 —— 模拟器**比真机简单**，
        模型会学到"囤着燃烧牌不管"，而这在真机里是要掉血的。
        """
        without = self._hp_loss_over_a_turn(["strike_ironclad"])
        with_burn = self._hp_loss_over_a_turn(["strike_ironclad", "burn", "burn"])
        self.assertEqual(with_burn - without, 4, "两张燃烧应各多造成 2 点")

    def test_burn_source_value_is_two(self):
        """源码里 ``Burn`` 的触发器量是**变量引用**（``DynamicVars.Damage``），
        所以要看**解析后**的数值，不是 JSON 里的 ``amount``。"""
        from sts2_sim import content
        card = content.CARD_DB["burn"]
        hook, effects = card.triggers[0]
        self.assertEqual(hook, "on_turn_end_in_hand")
        self.assertEqual(effects[0].op, "damage")
        self.assertEqual(effects[0].amount, 2)

    def test_trigger_only_cards_are_adopted(self):
        """只带触发器、没有 OnPlay 效果的卡**不能**被当成"没有效果"跳过。

        `burn` / `decay` / `infection` / `toxic` / `wither` 全是这样 ——
        它们的正常效果就在触发器里。
        """
        from sts2_sim import content
        for cid in ("burn", "decay", "infection", "toxic", "wither"):
            with self.subTest(cid=cid):
                card = content.CARD_DB[cid]
                self.assertTrue(card.triggers, f"{cid} 的触发器被丢掉了")
                self.assertEqual(card.triggers[0][0], "on_turn_end_in_hand")

    def test_triggers_run_before_the_hand_is_discarded(self):
        """真机顺序：手牌触发 → 再弃手牌。反了的话触发永远不生效。"""
        from sts2_sim.core import Action, step
        without = self._hp_loss_over_a_turn(["strike_ironclad"])
        with_burn = self._hp_loss_over_a_turn(["strike_ironclad", "burn"])
        self.assertEqual(with_burn - without, 2)
        state = self._combat_with_hand(["burn"])
        step(state, Action(kind="end_turn"))
        self.assertNotIn("burn", [c.cid for c in state.hand], "回合结束后手牌应清空")

    def test_ethereal_cards_are_exhausted(self):
        """``ResolveTurnEndCardEffects``：Ethereal 的牌走消耗，不是弃牌。

        一律弃掉会让这 18 张牌一直在牌库循环 —— 等于**变相加强**。
        """
        from sts2_sim.core import Action, step
        from sts2_sim import content
        ethereal = [c.cid for c in content.CARD_DB.values()
                    if "Ethereal" in c.keywords and c.cost >= 0 and c.effects]
        self.assertTrue(ethereal, "应当有可打出的 Ethereal 牌")
        cid = ethereal[0]
        state = self._combat_with_hand([cid])
        step(state, Action(kind="end_turn"))
        self.assertIn(cid, [c.cid for c in state.exhaust])
        self.assertNotIn(cid, [c.cid for c in state.discard])

    def test_normal_cards_are_discarded(self):
        """对照组：普通牌回合结束应进弃牌堆。"""
        from sts2_sim.core import Action, step
        state = self._combat_with_hand(["strike_ironclad"])
        step(state, Action(kind="end_turn"))
        self.assertIn("strike_ironclad", [c.cid for c in state.discard])

    def test_unsupported_trigger_hooks_are_flagged(self):
        """引擎没实现的触发钩子必须标记，不能静默当成"没有触发"。"""
        from sts2_sim import content
        card = content.CARD_DB["bolas"]          # 触发器在 before_hand_draw
        self.assertTrue(card.effects_incomplete,
                        "未实现的触发钩子必须让这张卡被标记为残缺")


class TestNoSilentZero(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_unresolvable_amount_is_not_zeroed(self):
        """量与变量都缺失时**不能**默认成 0。

        `regret` 的"每次掉 13 点血"若变成"掉 0 点"，卡看起来还在、效果却没了 ——
        而且没有任何信号。宁可整张卡标残缺。
        """
        from sts2_sim import content
        card = content.CARD_DB["regret"]
        self.assertTrue(card.effects_incomplete,
                        "量解析不出的触发式卡必须标记残缺，不能静默变 0")
        for _hook, effects in card.triggers:
            for effect in effects:
                self.assertNotEqual(effect.amount, 0,
                                    "解析不出的量不该以 0 的形式进效果表")

    def test_source_effects_have_no_zero_amounts_for_damage(self):
        """源码采纳的卡里，伤害类效果不该出现 0 点（那是"解析失败"的痕迹）。

        ⚠️ 例外：**运行期公式**（``calc_kind``，`docs/12` §2.33）的量本来就是
        卡面 0、结算那一刻才算（`BodySlam` 打出的量等于当前格挡）。
        真正的判据是"既没有量、也没有公式、也不是 X"。
        """
        from sts2_sim import content
        suspicious = []
        for card in content.CARD_DB.values():
            if card.effect_source != "source":
                continue
            for effect in card.effects:
                if effect.op not in ("damage", "damage_all", "block"):
                    continue
                if effect.amount == 0 and not effect.calc_kind and not effect.amount_x:
                    suspicious.append(f"{card.cid}:{effect.op}")
        self.assertEqual(suspicious, [], f"这些卡有 0 点伤害/格挡，疑似解析失败：{suspicious}")


class TestDurationTiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self):
        from sts2_sim.core import start_combat
        state = start_combat(self.deck, ("nibbit",), seed=5)
        state.player.hp = state.player.max_hp = 300
        return state

    def test_duration_ticks_at_enemy_side_turn_end(self):
        """⭐ 真机三个 duration 能力的递减条件是 ``side == CombatSide.Enemy``。

        在**玩家回合结束**就递减会让"敌人先上易伤再打你"少算一整个回合的量：
        玩家结束回合 → 若此时递减，敌人这回合的攻击就享受不到易伤加成。
        """
        from sts2_sim.core import Action, step
        state = self._combat()
        state.player.add_power("vulnerable", 3)
        state.player.block = 999
        step(state, Action(kind="end_turn"))
        # 一个完整回合（玩家结束 + 敌方回合结束）之后才递减 1
        self.assertEqual(state.player.power("vulnerable"), 2)

    def test_enemy_debuffs_tick_too(self):
        from sts2_sim.core import Action, step
        state = self._combat()
        enemy = state.enemies[0]
        enemy.add_power("weak", 3)
        step(state, Action(kind="end_turn"))
        self.assertEqual(enemy.power("weak"), 2)

    def test_vulnerable_is_active_during_the_enemy_turn(self):
        """易伤在敌方回合**必须仍然生效**（这正是递减时机的作用）。"""
        from sts2_sim.core import Action, step
        state = self._combat()
        state.player.add_power("vulnerable", 2)
        state.player.block = 0
        before = state.player.hp
        step(state, Action(kind="end_turn"))
        self.assertGreater(before - state.player.hp, 0)


class TestPileMoves(unittest.TestCase):
    """`CardPileCmd.Add`：把选中的牌移到指定牌堆。"""

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_pile_target_is_the_top_level_argument(self):
        """``resolve_pile`` 只能看**顶层参数**。

        ``CardPileCmd.Add(await CardSelectCmd.FromCombatPile(ctx, PileType.Discard…),
        PileType.Hand)`` 里有**两个** ``PileType`` —— 整串正则搜到的是嵌套选牌的
        **来源**牌堆，于是 `Dredge` 被判成"选完放进弃牌堆"，正好相反。
        """
        from tools.extract_cards import resolve_pile
        args = ('await CardSelectCmd.FromCombatPile(ctx, PileType.Discard.GetPile(x),'
                ' y, prefs), PileType.Hand')
        self.assertEqual(resolve_pile(args), ("hand", ""))

    def test_pile_position_is_extracted(self):
        from tools.extract_cards import resolve_pile
        self.assertEqual(resolve_pile("cardModel, PileType.Draw, CardPilePosition.Top"),
                         ("draw", "top"))

    def test_selection_carries_the_target_pile(self):
        """选完之后去哪，由**外层** `CardPileCmd.Add` 的 PileType 决定。"""
        expected = {
            "headbutt": ("to_draw", "top", "discard"),   # 从弃牌堆挑一张放抽牌堆顶
            "hologram": ("to_hand", "", "discard"),      # 从弃牌堆挑一张回手牌
            "dredge": ("to_hand", "", "discard"),
            "thinking_ahead": ("to_draw", "top", "hand"),  # 手牌放回抽牌堆顶
        }
        for cid, (purpose, position, source) in expected.items():
            with self.subTest(cid=cid):
                selection = next(e for e in _source(cid)["effects"]
                                 if e["op"] == "select_card")
                self.assertEqual(selection["purpose"], purpose)
                self.assertEqual(selection.get("position", ""), position)
                self.assertEqual(selection["from"], source)

    def test_selection_from_hand_selects_from_hand(self):
        selection = next(e for e in _source("acrobatics")["effects"]
                         if e["op"] == "select_card")
        self.assertEqual(selection["from"], "hand")


class TestSelectionMoveRuntime(unittest.TestCase):
    """选牌后移到别的牌堆，在引擎里要真的动。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_choose_from_discard_into_hand(self):
        from sts2_sim.core import (Action, CardInstance, legal_actions,
                                   start_combat, step)
        state = start_combat(self.deck, ("nibbit",), seed=7)
        state.hand = [CardInstance("hologram")]
        state.discard = [CardInstance("strike_ironclad"),
                         CardInstance("defend_ironclad")]
        state.draw_pile = []
        state.energy = 5
        step(state, Action("play_card", 0, -1))
        self.assertIsNotNone(state.pending, "应从弃牌堆选牌")
        self.assertEqual(len(legal_actions(state)), 2)
        # ⚠️ 候选顺序由 `PendingSelection.candidates()` **规范排序**（审计 F09：
        # 观测与动作必须共用同一套下标），所以这里按身份找下标，不写死 1 ——
        # 写死数字会让测试去锁定一个与"选哪张牌"无关的实现细节。
        index = [c.cid for c in state.pending.candidates(state)].index(
            "defend_ironclad")
        step(state, Action("select_card", index))
        self.assertIn("defend_ironclad", [c.cid for c in state.hand])
        self.assertNotIn("defend_ironclad", [c.cid for c in state.discard])


if __name__ == "__main__":
    unittest.main(verbosity=2)
