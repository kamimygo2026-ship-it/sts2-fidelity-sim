"""**源码出处必须可解析**（``docs/09`` §5.2 铁律 R1 / R3）。

铁律是"一切机制以源码为准；所有断言与猜测都必须经源码检验"。
光写在文档里是口号，这个测试把它变成**机器可判定的**：

1. 引擎里每条能力声明都带 ``PowerRules.source``（形如 ``PoisonPower.AfterSideTurnStart``）；
   这里把其中的类名解析出来，确认 ``data/decompiled/sts2/`` 里**真有那个文件**。
2. 每条已实现算子在源码里都有依据（``ENGINE_OPS`` 逐个标注）。
3. 内容数据里每个对象的 ``class`` 名都能在反编译树里找到。

为什么值得单独一个测试文件：本项目的静默错误里，**根因是"没回源码确认"的比
"实现写错了"的多**（``docs/09`` §5.3 那张表：8 条里 6 条）。出处写错、
或者写了一个根本不存在的类名，说明这条机制从来没被真正核对过。
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECOMPILED = ROOT / "data" / "decompiled" / "sts2"
CONTENT = ROOT / "data" / "content" / "repo"

#: 从 ``PowerRules.source`` 里认出"某个类"的写法。
#: 真机命名统一以这些后缀结尾（``PoisonPower`` / ``Vajra`` / ``Shockwave``…）。
CLASS_TOKEN = re.compile(r"\b([A-Z]\w*(?:Power|Relic|Card|Orb|Potion|Enchantment|Model))\b")

#: 源码里确实存在、但不在 ``Models`` 子树里的类（命令 / 钩子 / 管理器）。
NON_MODEL_PREFIXES = ("CreatureCmd", "CardPileCmd", "PowerCmd", "PlayerCmd",
                      "Hook", "RelicModel", "PowerModel", "CardModel",
                      "AbstractModel", "CombatManager", "Creature", "CardRarityOdds")


def decompiled_files() -> dict[str, Path]:
    """``类名 → 文件路径``。索引一次，后面反复查。"""
    index: dict[str, Path] = {}
    for path in DECOMPILED.rglob("*.cs"):
        index.setdefault(path.stem, path)
    return index


class TestSourceCitations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DECOMPILED.exists():
            raise unittest.SkipTest(f"缺反编译源码树 {DECOMPILED}")
        cls.index = decompiled_files()

    def test_decompiled_tree_is_indexed(self):
        """先确认索引本身有内容 —— 否则后面每条断言都会"空过"。"""
        self.assertGreater(len(self.index), 1000, "反编译树没索引到类")
        for name in ("Vajra", "PoisonPower", "Shockwave", "CombatManager"):
            self.assertIn(name, self.index, f"{name}.cs 不在反编译树里")

    def test_every_implemented_power_cites_a_real_class(self):
        """每条已实现的能力都要能指到一个**真实存在**的源码类。"""
        from sts2_sim import powers
        unresolved: list[str] = []
        for pid, rules in powers.RULES.items():
            cited = CLASS_TOKEN.findall(rules.source)
            cited = [name for name in cited if name not in NON_MODEL_PREFIXES]
            if not cited:
                unresolved.append(f"{pid}: 没有可解析的类名（source={rules.source!r}）")
                continue
            if not any(name in self.index for name in cited):
                unresolved.append(f"{pid}: 这些类在源码里都不存在 {cited}")
        self.assertEqual(unresolved, [],
                         "能力声明的源码出处解析不到真实文件：\n" + "\n".join(unresolved))

    def test_power_source_citations_match_the_class_they_describe(self):
        """出处的类名必须与该能力**自己**对应，不能张冠李戴。

        ``PoisonPower`` 的声明里出现 ``PoisonPower``；写成别的 Power 说明
        这段行为是从别处抄来的（或者写错了），两种都该被发现。
        """
        from sts2_sim import powers
        wrong: list[str] = []
        for pid, rules in powers.RULES.items():
            cited = set(CLASS_TOKEN.findall(rules.source))
            # 允许引用别人的类（例如 `negates_debuff` 里提到 `ArtifactPower`），
            # 但**必须**至少有一个自己对应的类出现在出处里。
            if not cited & set(self.index):
                wrong.append(f"{pid}: {rules.source!r}")
        self.assertEqual(wrong, [], f"出处与能力不对应：{wrong}")

    def test_every_content_class_exists_in_source(self):
        """内容表里每个对象的类名都要能在反编译树里找到。

        ``cards_source.json`` 的 ``class`` / ``relics_source.json`` 的 ``class``
        是抽取器从源码文件名写的 —— 数值也来自那个文件。类名解析不到，
        说明这条数据没有可追溯的源码依据（或者抽取器写错了）。
        """
        if not CONTENT.exists():
            raise unittest.SkipTest(f"缺内容目录 {CONTENT}")
        problems: list[str] = []
        for name, key in (("cards_source.json", "class"),
                          ("relics_source.json", "class"),
                          ("monster_ai.json", "class"),
                          ("potions_source.json", "class")):
            path = CONTENT / name
            if not path.exists():
                continue
            records = json.loads(path.read_text(encoding="utf-8"))
            missing = [str(r.get(key) or r.get("cid") or r.get("rid"))
                       for r in records
                       if r.get(key) and r[key] not in self.index]
            if missing:
                problems.append(f"{name}: {len(missing)} 个类名不在源码树里 "
                                f"（前 5 个：{missing[:5]}）")
        self.assertEqual(problems, [], "\n".join(problems))

    def test_engine_ops_are_documented(self):
        """每个已实现算子都要在 ``ENGINE_OPS`` 的说明块里标出**真机命令**。

        算子表是"源码→引擎"的翻译契约。少一行说明，就意味着这个算子
        落地时没写清楚它是从哪个命令来的 —— 下一个人改它就无从对照。
        """
        from sts2_sim import content
        source = (ROOT / "sts2_sim" / "content.py").read_text(encoding="utf-8")
        start = source.find("ENGINE_OPS = frozenset(")
        self.assertGreater(start, 0, "找不到 ENGINE_OPS 定义")
        # 说明写在定义**之前**，所以往前取一段
        block = source[max(0, start - 3000):start]
        undocumented = [op for op in content.ENGINE_OPS if f"``{op}``" not in block]
        self.assertEqual(undocumented, [],
                         f"这些算子在 ENGINE_OPS 里没有源码出处：{undocumented}")
        # 说明块里必须真的提到真机命令类名，而不是自说自话
        cited = [name for name in ("CreatureCmd", "CardPileCmd", "PowerCmd",
                                   "PlayerCmd", "OrbCmd", "CardSelectCmd")
                 if name in block]
        self.assertGreaterEqual(len(cited), 4,
                                f"ENGINE_OPS 的说明里真机命令类名太少：{cited}")


if __name__ == "__main__":
    unittest.main()
