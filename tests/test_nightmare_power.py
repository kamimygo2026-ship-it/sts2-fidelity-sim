"""**W02 · P07** ``NightmarePower``：把选中的那张牌克隆 3 份，下回合入手。

两张源码：

* ``Nightmare.cs:57-77``（卡）—— 从**手牌**选 1 张，然后
  ``(await PowerCmd.Apply<NightmarePower>(…, 3m, …)).SetSelectedCard(selectedCard)``；
* ``NightmarePower.cs:49-63``（能力，``Instanced``）—— ``BeforeHandDraw`` 里把那份
  快照 ``CreateClone()`` ``Amount`` 次、``AddGeneratedCardToCombat(…, Hand)``，
  最后 ``PowerCmd.Remove(this)`` **只移除这一个实例**；``SetSelectedCard`` 自己
  做 ``CreateClone()`` + ``ClearAffliction``（`:73-79`）。

三件事最容易漏，本文件逐个钉住：

1. 选牌**不移动**那张牌（真机只克隆快照，牌还在手里）；
2. 克隆的是**快照**，原卡之后被升级/转化都不该改到它；
3. 兑现完只移除**那一个实例**（S01），两张 Nightmare 各自结算。
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


class NightmareTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, enemies=("nibbit",), seed: int = 5, hp: int = 3000):
        from sts2_sim.core import start_combat
        state = start_combat(self.deck, tuple(enemies), seed=seed)
        state.player.max_hp = hp
        state.player.hp = hp
        return state

    def play_nightmare(self, state, choose_index: int = 0):
        """打出一张 ``nightmare`` 并完成选牌，返回**被选中的那张卡**。"""
        from sts2_sim.core import Action, CardInstance, step
        card = CardInstance("nightmare")
        state.hand.append(card)
        state.energy = 9
        step(state, Action("play_card", state.hand.index(card), 0))
        chosen = state.pending.candidates(state)[choose_index]
        step(state, Action("select_card", choose_index))
        return chosen

    def next_turn(self, state):
        """结束回合并把新回合开始的（可能的）选牌都处理掉。"""
        from sts2_sim.core import Action, step
        step(state, Action("end_turn"))
        while state.pending is not None:
            step(state, Action("select_card", 0))


class EntryTest(NightmareTestCase):
    """`Nightmare.cs` 的入口：选牌 + `SetSelectedCard`。"""

    def test_choosing_does_not_move_the_card(self):
        """⭐ 真机只做 `CreateClone()`：被选中的牌**留在手牌里**。

        走成 `select_card(purpose="to_hand"/"exhaust")` 之类会把牌搬走 ——
        玩家会发现自己"选了一张牌，结果那张牌不见了"。
        """
        state = self.combat()
        chosen = self.play_nightmare(state)
        self.assertTrue(any(c.uid == chosen.uid for c in state.hand),
                        "被选中的牌必须还在手牌里")
        self.assertFalse(state.pending, "选完就该继续结算")

    def test_the_power_is_applied_with_the_snapshot_payload(self):
        """`Apply<NightmarePower>(…, 3)` + `SetSelectedCard`：一个实例、载荷是克隆。"""
        from sts2_sim.core import _instance_type
        state = self.combat()
        chosen = self.play_nightmare(state)
        self.assertEqual(_instance_type("nightmare"), "instanced")
        self.assertEqual(state.player.power("nightmare"), 3)
        instances = state.player.power_instances["nightmare"]
        self.assertEqual([i.amount for i in instances], [3])
        payload = instances[0].vars["selected_card"]
        self.assertEqual(payload.cid, chosen.cid)
        self.assertNotEqual(payload.uid, chosen.uid, "存的是克隆快照，不是原实例")

    def test_the_snapshot_survives_later_changes_to_the_original(self):
        """⭐ 快照与手牌那张牌**彻底脱钩**：原卡后来被升级，快照不受影响。"""
        state = self.combat()
        chosen = self.play_nightmare(state)
        chosen.upgraded = True                      # 原卡在场上升级
        payload = state.player.power_instances["nightmare"][0].vars["selected_card"]
        self.assertFalse(payload.upgraded, "快照不该跟着原卡变")

    def test_an_upgraded_card_is_snapshotted_as_upgraded(self):
        from sts2_sim.core import Action, CardInstance, step
        state = self.combat()
        card = CardInstance("nightmare")
        upgraded = CardInstance("strike_ironclad", upgraded=True)
        state.hand.extend([card, upgraded])
        state.energy = 9
        step(state, Action("play_card", state.hand.index(card), 0))
        candidates = state.pending.candidates(state)
        index = next(i for i, c in enumerate(candidates) if c.uid == upgraded.uid)
        step(state, Action("select_card", index))
        payload = state.player.power_instances["nightmare"][0].vars["selected_card"]
        self.assertTrue(payload.upgraded, "升级形态要跟着快照走")


class DeliveryTest(NightmareTestCase):
    """`NightmarePower.BeforeHandDraw`：下回合克隆 `Amount` 份入手并移除自己。"""

    def test_three_copies_arrive_next_turn_and_the_power_is_removed(self):
        """数**新生成**的牌（`state.added_cards`），不数"手牌里有几张同名牌" ——
        后者会把手牌抽取抽上来的同名牌一起算进去（起手牌组里本来就有 5 张打击）。"""
        state = self.combat()
        self.play_nightmare(state)
        before = state.added_cards
        self.next_turn(state)
        self.assertEqual(state.added_cards - before, 3, "克隆 3 份入手")
        self.assertEqual(state.player.power("nightmare"), 0, "兑现完移除自己")

    def test_copies_are_generated_cards(self):
        """走的是 `AddGeneratedCardToCombat` → 派发"生成牌/进入战斗"两个钩子。

        用 `smokestack` 当探针不现实（它只认状态牌），这里直接看
        `state.added_cards` 的增量 —— 它就是"战斗中新生成的牌"的账。
        """
        state = self.combat()
        self.play_nightmare(state)
        before = state.added_cards
        self.next_turn(state)
        self.assertEqual(state.added_cards - before, 3)

    def test_two_nightmares_each_deliver_their_own_snapshot(self):
        """⭐ 两个实例各自一份载荷：一个选 `strike`、一个选 `defend`，
        下回合各自克隆 3 份，且**第一份兑现不会清掉第二个实例**。"""
        from sts2_sim.core import Action, CardInstance, step
        state = self.combat()
        strikes = [c for c in state.hand if c.cid == "strike_ironclad"]
        defends = [c for c in state.hand if c.cid == "defend_ironclad"]
        self.assertTrue(strikes and defends)

        for target in (strikes[0], defends[0]):
            card = CardInstance("nightmare")
            state.hand.append(card)
            state.energy = 9
            step(state, Action("play_card", state.hand.index(card), 0))
            candidates = state.pending.candidates(state)
            index = next(i for i, c in enumerate(candidates) if c.uid == target.uid)
            step(state, Action("select_card", index))

        instances = state.player.power_instances["nightmare"]
        self.assertEqual(len(instances), 2, "两个 Nightmare 是两个实例")
        self.assertEqual([i.vars["selected_card"].cid for i in instances],
                         ["strike_ironclad", "defend_ironclad"])

        # 清掉手里的同名牌，只看这一回合新生成了多少（两张 Nightmare 各 3 份）
        state.hand = [c for c in state.hand
                      if c.cid not in ("strike_ironclad", "defend_ironclad")]
        before = state.added_cards
        self.next_turn(state)
        self.assertEqual(state.added_cards - before, 6, "两个实例各克隆 3 份")
        self.assertEqual(state.player.power("nightmare"), 0)


class SnapshotTest(NightmareTestCase):
    def test_snapshot_keeps_the_payload_and_the_instances(self):
        from sts2_sim import core
        state = self.combat()
        chosen = self.play_nightmare(state)
        saved = core.snapshot(state)
        restored = core.restore(saved)
        instances = restored.player.power_instances["nightmare"]
        self.assertEqual([i.amount for i in instances], [3])
        self.assertEqual(instances[0].vars["selected_card"].cid, chosen.cid)
        self.next_turn(restored)
        self.assertEqual(restored.player.power("nightmare"), 0)


if __name__ == "__main__":
    unittest.main()
