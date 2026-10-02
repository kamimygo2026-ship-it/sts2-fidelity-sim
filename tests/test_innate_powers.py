"""怪物固有属性与神器（Artifact）的回归测试（``docs/09`` L2/L4）。

真机有 **46 只怪（40%）** 自带固有属性，在入场时施加（``AfterAddedToRoom``）。
这一组锁住两件事：

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 46 只怪开局**拿不到**自己的固有属性 | 导入器只留了"未建模能力的 id 列表"，**把数值丢了**；引擎也从不在开战时施加 | `test_innate_powers_are_applied_at_combat_start` |
| 带神器的怪被轻易上满 debuff | `ArtifactPower.TryModifyPowerAmountReceived` 未实现 | `test_artifact_negates_a_debuff` |
| "给敌人上负力量"在神器前被错误抵消 | `GetTypeForAmount` 会让**带负值的类型翻转** | `test_negative_amount_flips_power_type` |
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


class TestInnatePowersData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CONTENT.exists():
            raise unittest.SkipTest(f"缺内容目录 {CONTENT}")

    def test_import_keeps_amounts(self):
        """归一化数据必须**连数值一起**保留固有属性。

        旧代码只留了一个 id 列表 —— `artifact 3` 与 `artifact 1`
        是完全不同的战斗，丢了数值等于这只怪的强度凭空变化。
        """
        monsters = json.loads((CONTENT / "monsters.json").read_text(encoding="utf-8"))
        with_innate = [m for m in monsters if m.get("innate_powers")]
        self.assertGreaterEqual(len(with_innate), 40,
                                "应当有约 46 只怪带固有属性")
        for entry in with_innate:
            for power in entry["innate_powers"]:
                self.assertIn("power", power)
                self.assertIn("amount", power,
                              f"{entry['eid']} 的固有属性缺少数值")

    def test_aeonglass_artifact_is_three(self):
        monsters = {m["eid"]: m for m in
                    json.loads((CONTENT / "monsters.json").read_text(encoding="utf-8"))}
        innate = {(p["power"], p["amount"])
                  for p in monsters["aeonglass"]["innate_powers"]}
        self.assertIn(("artifact", 3), innate)


class TestInnatePowersRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self, eid: str):
        from sts2_sim.core import start_combat
        return start_combat(self.deck, (eid,), seed=1)

    def test_innate_powers_are_applied_at_combat_start(self):
        """⭐ 真机在入境时施加固有属性；不施加等于**换了一只怪**。"""
        state = self._combat("aeonglass")
        self.assertEqual(state.enemies[0].power("artifact"), 3)

    def test_multiple_monsters_keep_their_innate_powers(self):
        expected = {"chomper": ("artifact", 2), "bowlbug_rock": ("imbalanced", 1),
                    "bygone_effigy": ("slow", 1)}
        for eid, (power, amount) in expected.items():
            with self.subTest(eid=eid):
                state = self._combat(eid)
                self.assertEqual(state.enemies[0].power(power), amount)

    def test_monster_without_innate_powers_starts_clean(self):
        """对照组：没有固有属性的怪不该凭空多出能力。"""
        from sts2_sim import content
        clean = [e.eid for e in content.ENEMY_DB.values() if not e.innate_powers]
        self.assertTrue(clean)
        state = self._combat(clean[0])
        self.assertEqual(state.enemies[0].powers, {})


class TestArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self, eid: str = "chomper"):
        from sts2_sim.core import start_combat
        return start_combat(self.deck, (eid,), seed=1)

    def test_artifact_negates_a_debuff(self):
        """``ArtifactPower``：抵消一次**可见 Debuff**，并消耗 1 层。"""
        from sts2_sim.core import Action, CardInstance, step
        state = self._combat()
        enemy = state.enemies[0]
        self.assertEqual(enemy.power("artifact"), 2)
        state.hand = [CardInstance("bash")]      # 8 伤害 + 2 易伤
        state.energy = 5
        step(state, Action("play_card", 0, 0))
        self.assertEqual(enemy.power("vulnerable"), 0, "易伤应被完全抵消")
        self.assertEqual(enemy.power("artifact"), 1, "神器应消耗 1 层")

    def test_artifact_is_consumed_one_layer_at_a_time(self):
        from sts2_sim.core import Action, CardInstance, step
        state = self._combat()
        enemy = state.enemies[0]
        state.energy = 99
        for _ in range(2):
            state.hand = [CardInstance("bash")]
            step(state, Action("play_card", 0, 0))
        self.assertEqual(enemy.power("artifact"), 0)
        self.assertEqual(enemy.power("vulnerable"), 0)
        # 神器用完后，第三次应该真的上上去
        state.hand = [CardInstance("bash")]
        step(state, Action("play_card", 0, 0))
        self.assertEqual(enemy.power("vulnerable"), 2)

    def test_artifact_does_not_negate_buffs(self):
        """神器只挡 Debuff —— 给自己上力量不该被挡。"""
        from sts2_sim import powers
        state = self._combat()
        enemy = state.enemies[0]
        self.assertFalse(powers.negates_debuff(enemy, "strength", 3))
        self.assertTrue(powers.negates_debuff(enemy, "vulnerable", 2))

    def test_artifact_ignores_zero_amounts(self):
        from sts2_sim import powers
        state = self._combat()
        self.assertFalse(powers.negates_debuff(state.enemies[0], "vulnerable", 0))

    def test_no_artifact_means_no_negation(self):
        """对照组：没有神器的怪不该抵消任何东西。"""
        from sts2_sim import powers
        state = self._combat("nibbit")
        self.assertEqual(state.enemies[0].power("artifact"), 0)
        self.assertFalse(powers.negates_debuff(state.enemies[0], "vulnerable", 2))


class TestPowerTypeForAmount(unittest.TestCase):
    """``PowerModel.GetTypeForAmount``：带负值时正负性会翻转。"""

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_negative_flips_counter_with_allow_negative(self):
        from sts2_sim import powers
        # strength 是 allow_negative 的 Counter 能力 → 负数算 Debuff
        self.assertEqual(
            powers.type_for_amount(-3, "buff", "counter", True), "debuff")

    def test_negative_flips_debuff_without_allow_negative(self):
        from sts2_sim import powers
        self.assertEqual(
            powers.type_for_amount(-3, "debuff", "counter", False), "buff")

    def test_positive_amounts_keep_their_type(self):
        from sts2_sim import powers
        self.assertEqual(powers.type_for_amount(3, "buff", "counter", True), "buff")
        self.assertEqual(powers.type_for_amount(3, "debuff", "counter", False),
                         "debuff")

    def test_negative_strength_consumes_artifact(self):
        """给带神器的怪上**负力量**算 Debuff，应当被抵消。"""
        from sts2_sim import powers
        from sts2_sim.core import start_combat
        from sts2_sim.content import STARTING_DECK
        state = start_combat(STARTING_DECK, ("chomper",), seed=1)
        self.assertTrue(powers.negates_debuff(state.enemies[0], "strength", -3))


class TestCoverageReporting(unittest.TestCase):
    #: 已知台账：**还有多少只怪的固有属性没实现**。只允许往下调。
    #: 2026-09 的敌人 buff 专项之后归零（46 只怪的固有属性全部实现）。
    KNOWN_INNATE_GAP = 0

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_innate_gap_is_visible(self):
        """还没实现的固有属性必须能被报出来，而不是静默生效一半。

        ⚠️ 现在这个缺口是 **0**（46 只怪的固有属性全部实现），所以断言改成
        **台账式**：数字不许变多；真变多了说明新抽出了没实现的能力，
        而且报告里必须列得出来。反过来，如果哪天又归零，这条测试也要能过。
        """
        from sts2_sim import content
        coverage = content.engine_coverage()
        self.assertGreaterEqual(coverage["monsters_with_innate_powers"], 40)
        gap = coverage["monsters_with_unimplemented_innate"]
        self.assertLessEqual(
            gap, self.KNOWN_INNATE_GAP,
            f"有固有属性缺口的怪变多了（台账 {self.KNOWN_INNATE_GAP} → {gap}）："
            f"{coverage['unimplemented_innate_ranked'][:8]}")
        if gap:
            self.assertTrue(coverage["unimplemented_innate_ranked"],
                            "有缺口就必须列得出来，否则训练时无从排除")
        else:
            self.assertEqual(coverage["unimplemented_innate_ranked"], [],
                             "缺口为 0 时清单必须是空的")

    def test_artifact_is_counted_as_implemented(self):
        from sts2_sim import powers
        self.assertIn("artifact", powers.IMPLEMENTED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
