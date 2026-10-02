"""伤害/格挡管线测试——严格对照反编译源码。

真相来源：

* ``Hook.ModifyDamageInternal``：**加法 → 乘法（累乘）→ 钳制**，全程 decimal
* ``Creature.cs``：``(int)Math.Clamp(amount, 0m, 999999999m)`` —— **只在最后截断一次**
* ``VulnerablePower.cs``：``DamageIncrease = 1.5``
* ``WeakPower.cs``：``DamageDecrease = 0.75``
* ``FrailPower.cs``：``ModifyBlockMultiplicative`` 返回 ``0.75``
"""

from __future__ import annotations

import unittest

from sts2_sim.core import VALUE_CAP, Combatant, compute_block, compute_damage


def fighter(*powers: tuple[str, int]) -> Combatant:
    unit = Combatant("x", 100, 100)
    for name, amount in powers:
        unit.add_power(name, amount)
    return unit


class TestSingleTruncation(unittest.TestCase):
    """⭐ 只在最后截断一次——逐次截断会系统性偏低。"""

    def test_weak_and_vulnerable_multiply_together(self):
        attacker = fighter(("weak", 1))
        defender = fighter(("vulnerable", 2))
        # floor(base × 1.5 × 0.75) = floor(base × 1.125)
        for base, expected in ((7, 7), (9, 10), (13, 14), (20, 22)):
            self.assertEqual(compute_damage(base, attacker, defender), expected,
                             f"base {base} 的虚弱+易伤结算不符")

    def test_sequential_truncation_would_differ(self):
        """反向验证：证明"逐次截断"确实会算错，否则上面的测试没有鉴别力。"""
        attacker = fighter(("weak", 1))
        defender = fighter(("vulnerable", 2))
        sequential = int(int(9 * 0.75) * 1.5)      # 旧实现
        self.assertNotEqual(compute_damage(9, attacker, defender), sequential)

    def test_strength_is_additive(self):
        attacker = fighter(("strength", 3))
        self.assertEqual(compute_damage(6, attacker, fighter()), 9)

    def test_strength_applies_before_multipliers(self):
        """加法在先、乘法在后：(6+2) × 1.5 = 12，而不是 6×1.5+2 = 11。"""
        attacker = fighter(("strength", 2))
        defender = fighter(("vulnerable", 1))
        self.assertEqual(compute_damage(6, attacker, defender), 12)

    def test_weak_only(self):
        self.assertEqual(compute_damage(8, fighter(("weak", 1)), fighter()), 6)

    def test_vulnerable_only(self):
        self.assertEqual(compute_damage(8, fighter(), fighter(("vulnerable", 1))), 12)

    def test_negative_strength_clamps_to_zero(self):
        self.assertEqual(compute_damage(2, fighter(("strength", -5)), fighter()), 0)

    def test_value_cap_matches_creature_cs(self):
        attacker = fighter(("strength", VALUE_CAP))
        self.assertEqual(compute_damage(1, attacker, fighter()), VALUE_CAP)


class TestAdditiveStageOrder(unittest.TestCase):
    """⭐ 审计 F06：``tainted`` 必须落在**加法阶段**，与力量同层。

    真机 ``Hook.ModifyDamageInternal`` 先把所有 ``ModifyDamageAdditive`` 跑完，
    再跑 ``ModifyDamageMultiplicative``。``TaintedPower`` 走的是 Additive，
    所以正确结果是 ``int((10+3)×1.5) = 19``；旧实现放在乘法之后得到
    ``int(10×1.5)+3 = 18``——**只在同时有易伤/虚弱时**才暴露。
    """

    def test_tainted_is_added_before_vulnerable_multiplies(self):
        defender = fighter(("vulnerable", 1), ("tainted", 3))
        self.assertEqual(compute_damage(10, fighter(), defender), 19)

    def test_tainted_without_multipliers_is_plain_addition(self):
        self.assertEqual(compute_damage(10, fighter(), fighter(("tainted", 3))), 13)

    def test_tainted_and_weak_multiply_the_whole_sum(self):
        """int((6+2) × 1.5 × 0.75) = int(9.0) = 9（不是 6×1.125+2 = 8）。"""
        attacker = fighter(("weak", 1))
        defender = fighter(("vulnerable", 1), ("tainted", 2))
        self.assertEqual(compute_damage(6, attacker, defender), 9)

    def test_tainted_and_strength_are_the_same_stage(self):
        """int((6+2+3) × 1.5) = int(16.5) = 16 —— 两者相加后才乘。"""
        attacker = fighter(("strength", 2))
        defender = fighter(("vulnerable", 1), ("tainted", 3))
        self.assertEqual(compute_damage(6, attacker, defender), 16)

    def test_reverse_verification_old_ordering_would_differ(self):
        """反向验证：证明旧顺序（乘法后再加）确实给出不同的数字。"""
        old = int(10 * 1.5) + 3
        self.assertNotEqual(
            compute_damage(10, fighter(), fighter(("vulnerable", 1), ("tainted", 3))),
            old)


class TestFrailBlock(unittest.TestCase):
    def test_frail_multiplies_block_by_075(self):
        self.assertEqual(compute_block(8, fighter(("frail", 2))), 6)
        self.assertEqual(compute_block(5, fighter(("frail", 1))), 3)

    def test_no_frail_is_unchanged(self):
        self.assertEqual(compute_block(8, fighter()), 8)

    def test_block_truncates_once(self):
        """11 × 0.75 = 8.25 → 8（只截断一次）。"""
        self.assertEqual(compute_block(11, fighter(("frail", 1))), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
