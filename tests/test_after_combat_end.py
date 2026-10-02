"""``AfterCombatEnd``（战斗结束的**额外奖励**）能力回归。

真机签名是 ``AfterCombatEnd(CombatRoom room)``，能力在里面对房间追加奖励
（``room.AddExtraReward(player, reward)``）。引擎里"房间 / 奖励队列 / 牌组 / 金币"
都在 Run 层，所以分层是：

* **战斗侧**（``sts2_sim/powers.py`` 的 :func:`after_combat_end`）只产出**描述**：
  ``{"kind": "gold" | "remove_card_choices" | "upgrade_random_deck_cards", "amount": N}``；
* **Run 侧**（``RunEnv._apply_after_combat_end_powers``）把它翻译成 Run 层效果。

这里的测试正好按这层分工分两组：先钉描述，再用真 RunEnv 打一场**真的**战斗，
从"战斗胜利"这个入口验证落地（不直接调私有方法，否则调用点没人管）。
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


class AfterCombatEndDescriptionTest(unittest.TestCase):
    """第一层：战斗侧只谈"谁该给什么"。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self):
        from sts2_sim.core import start_combat
        return start_combat(self.deck, ("nibbit",), seed=5)

    def test_describes_extra_rewards(self):
        from sts2_sim import powers
        state = self._combat()
        state.player.add_power("royalties", 30)
        state.player.add_power("forbidden_grimoire", 2)
        state.player.add_power("improvement", 1)
        events: list[str] = []
        described = {item["power"]: item for item in powers.after_combat_end(state, events)}
        self.assertEqual(described["royalties"]["kind"], "gold")
        self.assertEqual(described["royalties"]["amount"], 30)
        self.assertEqual(described["forbidden_grimoire"]["kind"], "remove_card_choices")
        self.assertEqual(described["forbidden_grimoire"]["amount"], 2)
        self.assertEqual(described["improvement"]["kind"], "upgrade_random_deck_cards")

    def test_no_powers_no_rewards(self):
        """反向对照：身上没有这些能力时**一条描述都不该产出**。"""
        from sts2_sim import powers
        self.assertEqual(powers.after_combat_end(self._combat(), []), [])


class AfterCombatEndRunTest(unittest.TestCase):
    """第二层：真 RunEnv 打一场真战斗，从"战斗胜利"入口验证落地。"""

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def _win_a_combat(self, powers=()):
        """进一个怪物节点 → 把怪压到 1 血 → 用攻击牌收掉 → 返回 ``state``。

        把**节点的常规奖励清零**（金币 / 三选一），这样断言到的金币差额只可能
        来自被验证的能力 —— 否则"节点本身给的金币"会把结论冲淡。
        """
        from sts2_sim.content import CARD_DB
        from sts2_sim.core import Action
        from sts2_sim.run import PHASE_COMBAT, RunEnv

        env = RunEnv(seed=11, attempt_budget=1)
        env.reset()
        state = env.raw_state
        node = next(n for n in state.map.nodes if n.kind == "monster")
        env._enter_node(node.node_id)
        self.assertEqual(env.phase, PHASE_COMBAT, "应当进入战斗")
        contents = state.hidden.node_contents.get(state.position)
        if contents is not None:
            contents.gold = 0
            contents.card_reward = ()
        combat = state.combat
        combat.enemies[0].hp = 1
        for pid, amount in powers:
            combat.player.add_power(pid, amount)
        gold_before = state.player.gold
        for _ in range(40):
            if env.phase != PHASE_COMBAT:
                break
            hand = [i for i, card in enumerate(combat.hand)
                    if CARD_DB[card.cid].card_type == "attack"]
            if hand:
                env.step(Action("play_card", hand[0], 0))
            else:
                env.step(Action("end_turn"))
        self.assertNotEqual(env.phase, PHASE_COMBAT, "这场战斗没打完")
        return env, state, gold_before

    def test_royalties_pays_gold_at_combat_end(self):
        _env, state, gold_before = self._win_a_combat((("royalties", 30),))
        self.assertEqual(state.player.gold - gold_before, 30,
                         "战斗结束应当额外发 30 金币（节点奖励已清零）")

    def test_without_royalties_no_extra_gold(self):
        """反向对照：没有这个能力时，节点奖励清零 → 金币一点不涨。"""
        _env, state, gold_before = self._win_a_combat()
        self.assertEqual(state.player.gold, gold_before)

    def test_improvement_upgrades_a_deck_card(self):
        from sts2_sim.content import CARD_DB
        _env, state, _gold = self._win_a_combat((("improvement", 2),))
        upgraded = [c for c in state.player.deck if c.upgraded]
        self.assertEqual(len(upgraded), 2, "随机升级 2 张（源码按 Amount 次数循环）")
        for card in upgraded:
            self.assertTrue(CARD_DB[card.cid].upgrade, "只升级**有升级版**的牌")

    def test_forbidden_grimoire_asks_the_player_to_choose(self):
        """``CardRemovalReward``：**玩家自己选**一张移除（不是替玩家随机删）。"""
        from sts2_sim.run import PHASE_COMBAT, legal_meta_actions
        env, state, _gold = self._win_a_combat((("forbidden_grimoire", 1),))
        self.assertIsNotNone(state.pending_selection, "应当挂起等玩家选牌")
        actions = legal_meta_actions(state)
        self.assertTrue(actions, "挂起选牌时**必须有**合法动作（否则 run 堵死）")
        self.assertTrue(all(a.kind == "select_deck_card" for a in actions))
        before = len(state.player.deck)
        env.step(actions[0])
        self.assertEqual(len(state.player.deck), before - 1, "选中的那张被移除")
        self.assertNotEqual(env.phase, PHASE_COMBAT)


if __name__ == "__main__":
    unittest.main()
