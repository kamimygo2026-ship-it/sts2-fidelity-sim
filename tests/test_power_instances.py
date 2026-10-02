"""**S01 · 能力实例**：`InstanceType != None` 的能力每次施加新建实例。

真机出处（``PowerModel.cs:144`` 的默认实现 + ``PowerInstanceType`` 枚举的文档注释）::

    public virtual PowerInstanceType InstanceType => PowerInstanceType.None;

    /// None                —— 同类能力合成一个，再施加就叠层（242 个）
    /// Instanced           —— 每次施加新建实例（21 个）。枚举注释里的例子正是
    ///                        TheBombPower："放第二个炸弹，你要的是**另一个从 3 开始
    ///                        倒数**的炸弹，而不是把第一个的层数变成 6"
    /// InstancedPerApplier —— 每个**施加者**一个实例，同一施加者再施加则叠到那一个
    ///                        （2 个：OblivionPower / StranglePower）

引擎侧的表达：``Combatant.power_instances[pid]`` 存实例表，
``Combatant.powers[pid]`` 是它们的**总和镜像**（观察层与既有查询一行都不用改）。

本文件钉住四件事：施加语义（新建 / 按施加者合并 / 普通叠层）、
**载荷随实例走**、钩子**逐实例**派发、以及 ``Remove(this)`` 只摘一个实例。
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"
POWER_DIR = (ROOT / "data" / "decompiled" / "sts2"
             / "MegaCrit.Sts2.Core.Models.Powers")


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


class InstanceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, enemies=("nibbit",), seed: int = 5):
        from sts2_sim.core import start_combat
        return start_combat(self.deck, tuple(enemies), seed=seed)

    def amounts(self, combatant, pid: str) -> list[int]:
        return [item.amount for item in combatant.power_instances.get(pid, [])]


class InstanceTypeTableTest(InstanceTestCase):
    """清单本身必须与源码一致（引擎不硬编码这件事）。"""

    def test_the_instanced_list_matches_the_decompiled_sources(self):
        """`content.INSTANCED_POWERS` 必须等于源码里声明了 ``InstanceType`` 的那批。

        上游新增一个 Instanced 能力而清单没跟上时，那个能力会**静默**退回
        "叠层"语义 —— 这正是这条测试要挡的事。
        """
        from sts2_sim import content
        from tools.extract_powers import instance_type_of, read_sources

        sources = read_sources(POWER_DIR)
        expected = {pid for pid, p in content.POWERS.items()
                    if p.instance_type != "none"}
        self.assertEqual(expected, set(content.INSTANCED_POWERS))
        # 抽查两个出处明确的：TheBombPower 是 Instanced、OblivionPower 是 PerApplier
        self.assertEqual(instance_type_of(sources, "TheBombPower"), "instanced")
        self.assertEqual(instance_type_of(sources, "OblivionPower"),
                         "instancedperapplier")
        self.assertEqual(instance_type_of(sources, "StrengthPower"), "none")

    def test_the_table_is_not_empty_and_has_both_flavours(self):
        from sts2_sim import content
        kinds = {content.POWERS[pid].instance_type
                 for pid in content.INSTANCED_POWERS}
        self.assertEqual(kinds, {"instanced", "instancedperapplier"})


class ApplySemanticsTest(InstanceTestCase):
    """三种 `InstanceType` 的施加语义。"""

    def test_instanced_creates_a_new_instance_each_time(self):
        """⭐ 两个炸弹是**两个实例各 3 层**，不是"一个 6 层的炸弹"。

        镜像 `powers["the_bomb"]` 仍是 6（观察层读它），但实例表里是两块。
        """
        state = self.combat()
        state.player.add_power("the_bomb", 3)
        state.player.add_power("the_bomb", 3)
        self.assertEqual(self.amounts(state.player, "the_bomb"), [3, 3])
        self.assertEqual(state.player.power("the_bomb"), 6, "镜像是总和")

    def test_normal_power_still_stacks_into_one_number(self):
        """非分实例能力不受影响：还是叠成一个数字，没有实例表。"""
        state = self.combat()
        state.player.add_power("strength", 2)
        state.player.add_power("strength", 3)
        self.assertEqual(state.player.power("strength"), 5)
        self.assertFalse(state.player.power_instances.get("strength"))

    def test_instanced_per_applier_merges_only_within_one_applier(self):
        """⭐ `OblivionPower`：同一个施加者再施加**叠到那一个**，换个施加者才新建。"""
        state = self.combat()
        state.player.add_power("oblivion", 2, applier=state.player)
        state.player.add_power("oblivion", 3, applier=state.player)
        self.assertEqual(self.amounts(state.player, "oblivion"), [5],
                         "同一施加者 → 叠到同一个实例")
        state.player.add_power("oblivion", 1, applier=state.enemies[0])
        self.assertEqual(self.amounts(state.player, "oblivion"), [5, 1],
                         "另一个施加者 → 新实例")

    def test_each_instance_keeps_its_own_payload(self):
        """⭐ **载荷随实例走**：一张升级过、一张没升级的两颗炸弹伤害不同。

        共用一份 `power_vars[pid]` 时后者会覆盖前者 —— 先放的那颗用错数值，
        而且日志完全正常。
        """
        state = self.combat()
        state.player.add_power("the_bomb", 3)
        state.player.power_vars["the_bomb"]["Damage"] = 40
        state.player.add_power("the_bomb", 3)
        state.player.power_vars["the_bomb"]["Damage"] = 50
        instances = state.player.power_instances["the_bomb"]
        self.assertEqual([i.vars.get("Damage") for i in instances], [40, 50])
        # `power_vars[pid]` 按约定指向**最新实例**（`Apply(...).SetXxx()` 的落点）
        self.assertIs(state.player.power_vars["the_bomb"], instances[-1].vars)


class DispatchTest(InstanceTestCase):
    """钩子**逐实例**派发，`base.Amount` 是实例自己的层数。"""

    def test_the_two_bombs_tick_independently(self):
        """一个 1 层、一个 3 层：前者当场引爆并被移除，后者只减到 2。

        ⚠️ 只派发一次（拿总和 4）会让它们**同时**炸掉，或者都不炸。
        """
        from sts2_sim import powers as power_rules
        state = self.combat()
        state.player.add_power("the_bomb", 1)
        state.player.add_power("the_bomb", 3)
        events: list[str] = []
        power_rules.on_before_side_turn_end(state, events, "player")
        self.assertEqual(self.amounts(state.player, "the_bomb"), [2],
                         "1 层那颗引爆后消失，3 层那颗减到 2")
        self.assertTrue(any("炸弹引爆" in line for line in events), events)

    def test_removing_one_instance_keeps_the_other(self):
        """`Remove(this)` 只摘掉**指定的那一个**，不是整条 `pop(pid)`。

        直接测 API：两颗炸弹层数不同（3 / 5），摘掉先放的那颗之后
        镜像必须跟着变成 5 —— 用 `add_power(pid, -amount)` 会从**最后一个**扣，
        减错对象而且不报错（所以才有 `remove_power_instance`）。
        """
        state = self.combat()
        state.player.add_power("the_bomb", 3)
        state.player.add_power("the_bomb", 5)
        first = state.player.power_instances["the_bomb"][0]
        state.player.remove_power_instance("the_bomb", first)
        self.assertEqual(self.amounts(state.player, "the_bomb"), [5])
        self.assertEqual(state.player.power("the_bomb"), 5, "镜像跟着走")

    def test_each_instance_sees_its_own_amount(self):
        """处理器拿到的第二个实参是**实例的层数**，不是总和。"""
        from sts2_sim import core, powers as power_rules
        state = self.combat()
        state.player.add_power("toric_toughness", 2)
        state.player.power_vars["toric_toughness"]["Block"] = 7
        state.player.add_power("toric_toughness", 2)
        state.player.power_vars["toric_toughness"]["Block"] = 9
        state.player.block = 0
        # `AfterBlockCleared` 要 `creature is owner` —— 直接派发那个钩子
        power_rules.on_block_cleared(state.player, state, [])
        # 两个实例各自给一次格挡（7 + 9），各自减 1 层
        self.assertEqual(state.player.block, 16)
        self.assertEqual(self.amounts(state.player, "toric_toughness"), [1, 1])


class SnapshotTest(InstanceTestCase):
    def test_instances_survive_the_snapshot_round_trip(self):
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("the_bomb", 1)
        state.player.power_vars["the_bomb"]["Damage"] = 41
        state.player.add_power("the_bomb", 3)
        saved = core.snapshot(state)
        restored = core.restore(saved)
        self.assertEqual(self.amounts(restored.player, "the_bomb"), [1, 3])
        self.assertEqual([i.vars.get("Damage")
                          for i in restored.player.power_instances["the_bomb"]],
                         [41, None])
        # 快照后两个世界互不影响
        restored.player.add_power("the_bomb", -1, instance=restored.player.power_instances["the_bomb"][0])
        self.assertEqual(self.amounts(restored.player, "the_bomb"), [3])
        self.assertEqual(self.amounts(state.player, "the_bomb"), [1, 3])


if __name__ == "__main__":
    unittest.main()
