"""**W03 · P11** `conqueror`（征服者）：只在**来自御剑**的攻击上 ×2，且每回合递减。

``ConquerorPower.cs``：:

    public override decimal ModifyDamageMultiplicative(Creature? target, decimal amount,
        ValueProp props, Creature? dealer, CardModel? cardSource, CardPlay? cardPlay)
    {
        if (!(cardSource is SovereignBlade)) return 1m;      // ← 卡来源
        if (!props.IsPoweredAttack()) return 1m;             // ← 有效攻击
        if (target != base.Owner) return 1m;                 // ← 打的是拥有者自己
        return 2m;
    }

**三个条件缺一不可**。最容易漏的是第一条：不判卡来源的话，**任何**攻击打在
有征服者的敌人身上都会翻倍 —— 那不是"少一个倍率"，而是"整场伤害翻倍"，
而且日志完全正常。`test_only_the_sovereign_blade_is_amplified` 就是钉它的。
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


class ConquerorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, seed: int = 5):
        from sts2_sim import core
        state = core.start_combat(self.deck, ("axebot",), seed=seed)
        state.player.max_hp = state.player.hp = 3000
        return state

    def hit(self, state, card_cid: str, *, conqueror: int = 0) -> int:
        """把敌人恢复成干净状态，用**一张指定牌**打它，返回实际掉血。"""
        from sts2_sim import core
        from sts2_sim.core import Action, CardInstance
        enemy = state.enemies[0]
        enemy.max_hp = enemy.hp = 500
        enemy.block = 0
        enemy.powers.clear()
        state.player.powers.clear()
        if conqueror:
            enemy.add_power("conqueror", conqueror)
        card = CardInstance(card_cid)
        state.hand.append(card)
        state.energy = 9
        before = enemy.hp
        core.step(state, Action("play_card", state.hand.index(card), 0))
        return before - enemy.hp


class DamageTest(ConquerorTest):
    def test_the_sovereign_blade_is_doubled(self):
        state = self.combat()
        self.assertEqual(self.hit(state, "sovereign_blade"), 10, "基线 10")
        self.assertEqual(self.hit(state, "sovereign_blade", conqueror=3), 20, "×2")

    def test_only_the_sovereign_blade_is_amplified(self):
        """⭐ 条件一（卡来源）：普通攻击**不**翻倍。

        少了这一条，任何攻击打在它身上都会 ×2 —— 整场伤害翻倍，而且不报错。
        """
        state = self.combat()
        plain = self.hit(state, "strike_ironclad")
        with_conqueror = self.hit(state, "strike_ironclad", conqueror=3)
        self.assertEqual(plain, with_conqueror,
                         f"普通打击不该被征服者放大（{plain} vs {with_conqueror}）")

    def test_seeking_edge_does_not_change_the_multiplier(self):
        """锐锋换的是**目标**（单体→全体），不是倍率；两者同时在场仍各自生效。"""
        state = self.combat()
        state.player.add_power("seeking_edge", 1)
        self.assertEqual(self.hit(state, "sovereign_blade", conqueror=1), 20)


class DecrementTest(ConquerorTest):
    def test_it_decrements_at_the_enemy_side_turn_end(self):
        """`AfterSideTurnEnd` + `participants.Contains(base.Owner)`：拥有者在**敌方**
        阵营，所以敌方阵营回合结束时递减一层。"""
        from sts2_sim import core
        state = self.combat()
        state.enemies[0].add_power("conqueror", 3)
        core.step(state, core.Action("end_turn"))
        while state.pending is not None:
            core.step(state, core.Action("select_card", 0))
        self.assertEqual(state.enemies[0].power("conqueror"), 2)

    def test_it_is_registered_as_a_duration_power(self):
        from sts2_sim import powers
        self.assertIn("conqueror", powers.IMPLEMENTED)
        self.assertIn("conqueror", powers.DECREMENTS_AT_ENEMY_SIDE_TURN_END)


if __name__ == "__main__":
    unittest.main()
