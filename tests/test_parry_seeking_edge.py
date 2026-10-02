"""**W03 · P09/P10** `parry`（招架）与 `seeking_edge`（锐锋）——两个**纯标记**能力。

源码里两个能力的类注释都写着"自己不做事，由 `SovereignBlade` 读它"：

* ``ParryPower.cs``：*"This power doesn't actually do anything on its own. Instead,
  the SovereignBlade card checks for its existence and modifies its block."*
  —— `SovereignBlade.CanonicalVars` 的
  ``new CalculatedBlockVar(ValueProp.Move).WithMultiplier((card, _) => GetOwnerParryAmount(card))``，
  而 `GetOwnerParryAmount` 就是 ``card.Owner.Creature.GetPowerAmount<ParryPower>()``；
* ``SeekingEdgePower.cs``：*"Sovereign blade cards checks for this power and changes
  its behavior based off of that"* —— `SovereignBlade.TargetType` 在拥有者身上有它时
  从 ``AnyEnemy`` 变成 ``AllEnemies``。

所以这两个能力的"验收"必须看**消费者**：本文件直接往手牌塞一张
``sovereign_blade`` 来验，不依赖 Forge（那属于后续批次）。
"""

from __future__ import annotations

import json
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


class BladeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, enemies=("nibbit",), seed: int = 5):
        from sts2_sim import core
        state = core.start_combat(self.deck, tuple(enemies), seed=seed)
        state.player.max_hp = state.player.hp = 3000
        return state

    def hold_blade(self, state, *, energy: int = 9):
        from sts2_sim.core import CardInstance
        blade = CardInstance("sovereign_blade")
        state.hand.append(blade)
        state.energy = energy
        return blade

    def play(self, state, blade, target: int = 0):
        from sts2_sim import core
        return core.step(state, core.Action("play_card",
                                            state.hand.index(blade), target))


class ParryTest(BladeTestCase):
    """`parry` 的消费者：`SovereignBlade` 的格挡 = 招架层数。"""

    def test_the_layers_become_block(self):
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("parry", 3)
        blade = self.hold_blade(state)
        self.play(state, blade)
        self.assertEqual(state.player.block, 3, "3 层招架 → 3 点格挡")

    def test_layers_stack(self):
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("parry", 2)
        state.player.add_power("parry", 3)
        self.assertEqual(state.player.power("parry"), 5)
        blade = self.hold_blade(state)
        self.play(state, blade)
        self.assertEqual(state.player.block, 5)

    def test_no_parry_means_no_block(self):
        """⭐ `SovereignBlade.GainsBlock => GetOwnerParryAmount(this) > 0m`：
        没有招架时这张牌**不给格挡**（格挡效果整条不生效）。

        测的是"消费者真的读了那个能力"——只登记能力而没接消费者时这里会是 0
        与"有招架时也是 0"无法区分，所以两条测试必须成对看。
        """
        from sts2_sim import core
        state = self.combat()
        blade = self.hold_blade(state)
        self.play(state, blade)
        self.assertEqual(state.player.block, 0)

    def test_it_is_registered_as_a_plain_marker(self):
        from sts2_sim import content, powers
        self.assertIn("parry", powers.IMPLEMENTED)
        self.assertEqual(content.POWERS["parry"].stack_type, "counter")


class SeekingEdgeTest(BladeTestCase):
    """`seeking_edge` 的消费者：`SovereignBlade` 的目标从单体变全体。"""

    def test_target_becomes_all_enemies(self):
        from sts2_sim import core
        state = self.combat(enemies=("nibbit", "axebot"))
        blade = self.hold_blade(state)
        self.assertEqual(core.card_target(state, blade), "enemy",
                         "没有锐锋时是单体")
        state.player.add_power("seeking_edge", 1)
        self.assertEqual(core.card_target(state, blade), "all_enemies")

    def test_damage_hits_every_enemy(self):
        """⭐ 伤害落点跟着 `card_target` 一起变（只改一处，三处一起生效）。"""
        state = self.combat(enemies=("nibbit", "axebot"))
        state.player.add_power("seeking_edge", 1)
        blade = self.hold_blade(state)
        before = [enemy.hp for enemy in state.enemies]
        self.play(state, blade)
        after = [enemy.hp for enemy in state.enemies]
        self.assertEqual([b - a for b, a in zip(before, after)], [10, 10])

    def test_without_it_only_the_target_is_hit(self):
        state = self.combat(enemies=("nibbit", "axebot"))
        blade = self.hold_blade(state)
        before = [enemy.hp for enemy in state.enemies]
        self.play(state, blade, target=1)
        after = [enemy.hp for enemy in state.enemies]
        self.assertEqual(before[0] - after[0], 0, "没被指定的敌人不该掉血")
        self.assertEqual(before[1] - after[1], 10)

    def test_block_and_all_target_can_coexist(self):
        state = self.combat(enemies=("nibbit", "axebot"))
        state.player.add_power("seeking_edge", 1)
        state.player.add_power("parry", 4)
        blade = self.hold_blade(state)
        before = [enemy.hp for enemy in state.enemies]
        self.play(state, blade)
        self.assertEqual([b - e.hp for b, e in zip(before, state.enemies)], [10, 10])
        self.assertEqual(state.player.block, 4)

    def test_it_is_registered_as_a_single_stack_marker(self):
        from sts2_sim import content, powers
        self.assertIn("seeking_edge", powers.IMPLEMENTED)
        self.assertEqual(content.POWERS["seeking_edge"].stack_type, "single")


class ExtractorTest(BladeTestCase):
    """抽取器侧：`SovereignBlade` 的格挡公式必须被认出来。"""

    def test_the_block_formula_is_recorded(self):
        """``CalculatedBlockVar.WithMultiplier((card, _) => GetOwnerParryAmount(card))``
        → ``calc_kind="self_power"`` + ``calc_arg="parry"``。

        ⚠️ 这个 lambda 体是**方法调用**，能力藏在 `GetOwnerParryAmount` 的实现里 ——
        抽取器要跟着那一层回源码抓 `GetPowerAmount<ParryPower>()`。
        不跟的话整条会被判"量由运行期公式决定"，卡被排除出训练集。
        """
        from sts2_sim import content
        blocks = [e for e in content.CARD_DB["sovereign_blade"].effects
                  if e.op == "block"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].calc_kind, "self_power")
        self.assertEqual(blocks[0].calc_arg, "parry")
        self.assertEqual(blocks[0].calc_base, 0)
        self.assertEqual(blocks[0].calc_extra, 1, "基 0 + 增量 1 × 招架层数")

    def test_the_card_has_no_remaining_gap(self):
        """`SovereignBlade` 不该再有任何 `unsupported`。

        它原来报的两条都要消失：``量由运行期公式决定：block``（本批修）与
        ``未扫描的钩子：AfterCardChangedPiles``（那段全是 VFX，见
        ``PRESENTATION_COMMANDS``）。
        """
        records = json.loads(
            (CONTENT / "cards_source.json").read_text(encoding="utf-8"))
        record = next(r for r in records if r["cid"] == "sovereign_blade")
        self.assertEqual(record["unsupported"], [])
        ops = [e["op"] for e in record["effects"]]
        self.assertEqual(ops, ["damage", "block"])


if __name__ == "__main__":
    unittest.main()
