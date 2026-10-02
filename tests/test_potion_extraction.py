"""药水抽取的回归测试（审计 F11 的"不能静默丢弃" + 事件代理留下的跨文件依赖）。

``FoulPotion`` 暴露了两个**静默**问题：

1. ``TargetType`` 是**条件属性**（按"是否在战斗中"返回不同枚举值），
   ``property_expr`` 只认单个表达式 → 整瓶药被丢掉。实测 65 个药水类只抽出 64 个，
   而社区库有 65 个（缺的正是 ``foul_potion``），事件 ``potion_courier``
   因此报"奖励药水不在药水表里"而不可用。
2. ``OnUse`` 有三个分支（战斗内 / 商人 / 假商人事件）。把整个方法体丢给效果
   抽取器会把三个分支**混在一起** —— 战斗内用这瓶药会一边打伤害一边白拿 100 金币。

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 65 个药水类只抽出 64 个 | 条件属性被当成"声明缺失"整瓶丢弃 | `test_no_potion_class_is_silently_dropped` |
| 战斗内用药水白拿 100 金币 | 三个分支的效果被合并 | `test_combat_branch_excludes_the_merchant_branch` |
| 事件 potion_courier 不可用 | foul_potion 不在药水表里 | `test_foul_potion_exists_and_is_marked_incomplete` |
"""

from __future__ import annotations

import collections
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"
POTION_SRC = ROOT / "data" / "decompiled" / "sts2" / "MegaCrit.Sts2.Core.Models.Potions"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestConditionalPotionDeclaration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not POTION_SRC.exists():
            raise unittest.SkipTest(f"缺源码目录 {POTION_SRC}")
        sys.path.insert(0, str(ROOT))
        cls.tool = _load("extract_potions_under_test",
                         ROOT / "tools" / "extract_potions.py")

    def test_conditional_property_values_are_collected(self):
        source = (POTION_SRC / "FoulPotion.cs").read_text(
            encoding="utf-8", errors="replace")
        values = self.tool.conditional_enum_values(source, "TargetType")
        self.assertIn("AllEnemies", values)
        self.assertIn("TargetedNoCreature", values)

    def test_combat_branch_excludes_the_merchant_branch(self):
        """⭐ 只抽战斗内分支：否则战斗内用药水会白拿 100 金币。"""
        from tools.extract_cards import extract_effects, build_key_map, method_body, parse_vars
        source = (POTION_SRC / "FoulPotion.cs").read_text(
            encoding="utf-8", errors="replace")
        whole = method_body(source, "OnUse")
        branch, branched = self.tool.combat_branch(whole)
        self.assertTrue(branched, "这个文件应当被识别为「含非战斗分支」")

        whole_ops = [e["op"] for e in
                     extract_effects(whole, build_key_map(parse_vars(source)))[0]]
        branch_ops = [e["op"] for e in
                      extract_effects(branch, build_key_map(parse_vars(source)))[0]]
        # 反向验证：整段抽会混进 `gain_gold`（商人分支），切分支之后不会。
        self.assertIn("gain_gold", whole_ops,
                      "构造前提不成立：整段抽居然没混进商人分支")
        self.assertNotIn("gain_gold", branch_ops,
                         "战斗内分支里不该有金币收益")

    def test_no_potion_class_is_silently_dropped(self):
        """⭐ 每个药水类都必须产出记录 —— 静默丢弃是本项目明令禁止的。"""
        report: collections.Counter = collections.Counter()
        classes = sorted(POTION_SRC.glob("*.cs"))
        parsed = [p for p in (self.tool.parse_potion(f, report) for f in classes) if p]
        self.assertEqual(len(parsed), len(classes),
                         f"{len(classes)} 个类只抽出 {len(parsed)} 个；"
                         f"缺的是 {sorted({f.stem for f in classes} - {p['class'] for p in parsed})}")

    def test_declaration_gaps_are_recorded_not_swallowed(self):
        """申报缺口要进 `unsupported`（content 按它判 `effects_incomplete`）。"""
        report: collections.Counter = collections.Counter()
        record = self.tool.parse_potion(POTION_SRC / "FoulPotion.cs", report)
        self.assertIsNotNone(record)
        self.assertTrue(record["unsupported"],
                        "条件声明必须被记录成缺口，不能悄悄采用一个值")
        self.assertTrue(any("TargetType" in item or "分支" in item
                            for item in record["unsupported"]))
        self.assertGreater(report["conditional_target_type"], 0)


class TestFoulPotionInContent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CONTENT.exists():
            raise unittest.SkipTest(f"缺内容目录 {CONTENT}")
        from sts2_sim.featurize import configure_content
        configure_content(str(CONTENT))
        cls.addClassCleanup(cls._restore)

    @staticmethod
    def _restore():
        from sts2_sim.content import load_builtin
        from sts2_sim.featurize import rebuild_vocab
        load_builtin()
        rebuild_vocab()

    def test_foul_potion_exists_and_is_marked_incomplete(self):
        """它必须在表里（事件要引用），但**不能**被判成可用（效果没建模全）。"""
        from sts2_sim.content import POTIONS
        potion = POTIONS.get("foul_potion")
        self.assertIsNotNone(potion, "foul_potion 必须进药水表，否则事件引用不到")
        self.assertTrue(potion.effects_incomplete,
                        "非战斗分支没建模，必须标成不完整")

    def test_no_usable_potion_is_incomplete(self):
        """口径检查：可用的药水里不允许混进不完整的。"""
        from sts2_sim.content import POTIONS
        for potion in POTIONS.values():
            if potion.effects_incomplete:
                continue
            self.assertTrue(potion.effects or potion.usage == "automatic", potion.pid)

    def test_potion_courier_becomes_usable(self):
        """事件代理留下的 blocker：奖励药水不在表里 → 事件不可用。"""
        from sts2_sim.content import EVENT_DB
        definition = EVENT_DB.get("potion_courier")
        if definition is None:
            self.skipTest("当前内容里没有 potion_courier")
        self.assertTrue(definition.usable,
                        f"补上 foul_potion 之后它应当可用，实际 reasons={definition.reasons}")

    def test_potion_source_has_no_duplicate_pids(self):
        records = json.loads((CONTENT / "potions_source.json").read_text(
            encoding="utf-8"))
        pids = [record["pid"] for record in records]
        self.assertEqual(len(pids), len(set(pids)), "药水表有重复 pid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
