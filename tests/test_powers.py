"""能力注册表对照测试（``docs/09`` L2）。

真相来源：``powers.json``（由真机 ``PowerModel`` 派生），
以及 ``PowerStackType`` / ``PowerType`` 枚举。
"""

from __future__ import annotations

import json
import pathlib
import unittest

from sts2_sim.content import POWERS, PowerDef
from sts2_sim.core import Combatant, _allows_negative

CONTENT = pathlib.Path("data/content/repo")
RAW = pathlib.Path("data/raw/spire-codex/repo/powers.json")


def restore_builtin() -> None:
    """还原内置占位内容。

    ⚠️ **加载真机内容会改全局状态**，不还原会污染同一进程里的其它测试——
    实测把 `test_sim.py` / `test_run.py` 里假设"12 张占位卡"的用例全带崩了。
    凡是调用 `configure_content` 的测试类都必须注册这个清理。
    """
    from sts2_sim.content import load_builtin
    from sts2_sim.featurize import rebuild_vocab
    load_builtin()
    rebuild_vocab()


def loaded() -> bool:
    if not (CONTENT / "powers.json").exists():
        raise unittest.SkipTest("缺 content/repo/powers.json")
    if not POWERS:
        from sts2_sim.featurize import configure_content
        configure_content(str(CONTENT))
    return True


class TestPowerRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loaded()
        cls.addClassCleanup(restore_builtin)

    def test_all_powers_loaded(self):
        """社区库 + **源码补缺** = 全部能力。

        ⚠️ 社区库 ``powers.json`` 只有 257 个，而反编译源码里有 **265 个**
        非抽象能力类 —— 少的 10 个是卡牌专属能力（``hyperbeam_focus_down`` /
        ``cacophony`` / ``concoct`` …）。不补的话这些卡一抽取出来就报
        "未知 power"，把真正的数据错误淹掉（实测 lint 一次报 12 条）。
        """
        raw = json.loads((CONTENT / "powers.json").read_text(encoding="utf-8"))
        source = json.loads((CONTENT / "powers_source.json").read_text(encoding="utf-8"))
        codex_ids = {p["id"].lower() for p in raw}
        source_only = {p["id"].lower() for p in source} - codex_ids
        self.assertGreater(len(source_only), 5,
                           "源码里应该有一批社区库没有的能力")
        for pid in source_only:
            with self.subTest(pid=pid):
                self.assertIn(pid, POWERS,
                              f"{pid} 只在源码里有，必须被补进能力表")
        self.assertEqual(len(POWERS), len(codex_ids | {p["id"].lower() for p in source}))

    def test_source_powers_carry_stack_semantics(self):
        """**全部**能力的叠加语义都必须与源码一致（源码优先于社区库）。

        ``HyperbeamFocusDownPower : TemporaryFocusPower`` 自己不声明
        ``StackType``，只看自己的文件取不到 —— 所以必须沿继承链取。
        """
        source = {p["id"].lower(): p
                  for p in json.loads((CONTENT / "powers_source.json").read_text(encoding="utf-8"))}
        # 条件式叠加只有极少数（属性体里 if/return），静态无唯一值
        unresolved = [pid for pid, p in source.items() if p["stack_type"] is None]
        self.assertLessEqual(len(unresolved), 3, f"条件式叠加应当极少：{unresolved}")
        mismatched = [pid for pid, entry in source.items()
                      if entry["stack_type"] and pid in POWERS
                      and POWERS[pid].stack_type != entry["stack_type"]]
        self.assertEqual(mismatched, [],
                         "叠加语义必须以源码为准（社区库在这一项上有分歧）")

    def test_source_is_authoritative_for_kind(self):
        """正负性也必须与源码一致 —— 真机的机制是 ``Type => PowerType.X``。

        ⚠️ 我第一版抽的是 ``IsPositive``（字段找错了），结果 44 个能力和社区库
        对不上。改为 ``PowerType`` 后分歧降到 **0**（255 个共有能力全对）。
        """
        source = {p["id"].lower(): p for p in
                  json.loads((CONTENT / "powers_source.json").read_text(encoding="utf-8"))}
        codex = {p["id"].lower(): p for p in
                 json.loads((CONTENT / "powers.json").read_text(encoding="utf-8"))}
        self.assertGreater(len(codex), 200)
        for pid, entry in codex.items():
            if not source.get(pid, {}).get("type"):
                continue
            with self.subTest(pid=pid):
                self.assertEqual(entry.get("type", "").lower(), source[pid]["type"],
                                 "源码与社区库在正负性上不应有分歧")
        unexplained = [pid for pid, e in source.items()
                       if e["type"] is None and pid in POWERS
                       and POWERS[pid].kind == "buff"]
        # 条件式 Type 的能力会被社区库的值兜住，这是可接受的；数目应极少
        self.assertLessEqual(len(unexplained), 25)

    def test_kind_and_stack_distribution(self):
        """分布必须与**加载规则**逐条算出来的结果一致，而不是钉一个魔数。

        规则是"源码优先、社区库补缺"（``_load_powers``）。把规则抄成断言，
        数据源更新时测试会自己跟着走；钉魔数则每次都要手改，改错了还看不出来。
        """
        codex = {p["id"].lower(): p
                 for p in json.loads((CONTENT / "powers.json").read_text(encoding="utf-8"))}
        source = {p["id"].lower(): p
                  for p in json.loads((CONTENT / "powers_source.json").read_text(encoding="utf-8"))}

        expected_kind: dict[str, str] = {}
        expected_stack: dict[str, str] = {}
        for pid in set(codex) | set(source):
            src, cx = source.get(pid, {}), codex.get(pid, {})
            expected_kind[pid] = str(src.get("type") or cx.get("type")
                                     or "").lower() or "unknown"
            expected_stack[pid] = str(src.get("stack_type")
                                      or cx.get("stack_type") or "").lower() or "counter"

        self.assertEqual({p.pid: p.kind for p in POWERS.values()}, expected_kind)
        self.assertEqual({p.pid: p.stack_type for p in POWERS.values()}, expected_stack)

        debuffs = sum(1 for k in expected_kind.values() if k == "debuff")
        self.assertGreater(debuffs, 40,
                           "负面能力不该只有个位数 —— 那是正负性字段抽错了"
                           "（真机写 `Type => PowerType.Debuff`，不是 `IsPositive`）")

    def test_negative_powers_match_the_source(self):
        """⭐ ``allow_negative`` 恰好是这 5 个。

        硬编码容易漏——我第一版只写了 strength / dexterity，
        漏掉 focus / shriek / shrink，表现为"这些能力被削到 0 就不再往下走"。
        """
        negative = sorted(p.pid for p in POWERS.values() if p.allow_negative)
        self.assertEqual(negative, ["dexterity", "focus", "shriek", "shrink", "strength"])

    def test_registry_is_the_source_of_truth(self):
        self.assertTrue(_allows_negative("focus"))
        self.assertTrue(_allows_negative("shrink"))
        self.assertFalse(_allows_negative("vulnerable"))
        self.assertFalse(_allows_negative("weak"))

    def test_monster_innate_powers_all_known(self):
        """怪物固有属性必须都能在注册表里查到，否则就是没建模的能力。"""
        from sts2_sim.content import ENEMY_DB
        unknown = sorted({p for e in ENEMY_DB.values()
                          for p in e.unmodeled_innate_powers if p not in POWERS})
        self.assertEqual(unknown, [], f"怪物用到了未注册的能力：{unknown}")


class TestNegativeStacking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loaded()
        cls.addClassCleanup(restore_builtin)

    def _unit(self) -> Combatant:
        return Combatant("x", 50, 50)

    def test_steamroll_negative_value(self):
        unit = self._unit()
        unit.add_power("strength", 3)
        unit.add_power("strength", -5)
        self.assertEqual(unit.power("strength"), -2, "力量允许为负，不该归零")

    def test_allow_negative_power_keeps_going_down(self):
        unit = self._unit()
        unit.add_power("focus", 2)
        unit.add_power("focus", -3)
        self.assertEqual(unit.power("focus"), -1)

    def test_counter_power_disappears_at_zero(self):
        unit = self._unit()
        unit.add_power("vulnerable", 2)
        unit.add_power("vulnerable", -2)
        self.assertEqual(unit.power("vulnerable"), 0)
        self.assertNotIn("vulnerable", unit.powers, "归零的能力应当从字典里移除")

    def test_debuff_cannot_go_negative(self):
        unit = self._unit()
        unit.add_power("weak", 1)
        unit.add_power("weak", -5)
        self.assertEqual(unit.power("weak"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
