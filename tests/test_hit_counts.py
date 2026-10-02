"""抽取器的 ``WithHitCount`` 回归（"打 N 次"的次数从哪来）。

``DamageCmd.Attack(伤害).WithHitCount(次数)`` 是攻击卡的常见写法，而**次数**有四种来源，
抽取器必须区别对待（``tools/extract_cards.py`` 的 :func:`resolve_times`）：

| 写法 | 例子 | 该怎么处理 |
|---|---|---|
| 字面量 | ``WithHitCount(3)`` | 直接展开 |
| ``ResolveEnergyXValue()`` | `whirlwind` | 标 ``times_x``，打出时按 X 展开 |
| **卡牌动态变量** | ``WithHitCount(base.DynamicVars.Repeat.IntValue)``（`peck`） | 标 ``times_var``，引擎按具体数值展开 |
| **运行期公式** | ``WithHitCount((int)((CalculatedVar)…).Calculate(target))`` | **必须报缺口**，绝不能当成常量变量 |

最后一行是这一批最危险的一处：`CalculatedVar` 的外壳也是 ``DynamicVars["CalculatedHits"]``，
"顺手"按常量变量处理会让"打 N 次"变成"按 base 值打" —— 而卡在报告里是**干净**的。
（本文件就是先踩到、再修掉之后钉住的。）
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"


def _record(cid: str) -> dict:
    records = json.loads((CONTENT / "cards_source.json").read_text(encoding="utf-8"))
    for item in records:
        if item.get("cid") == cid:
            return item
    raise AssertionError(f"cards_source.json 里没有 {cid}")


def _damage_effects(cid: str) -> list[dict]:
    return [e for e in _record(cid)["effects"]
            if e.get("op") in ("damage", "damage_all")]


class TestWithHitCount(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CONTENT.exists():
            raise unittest.SkipTest(f"缺内容目录 {CONTENT}")

    def test_dynamic_var_count_becomes_times_var(self):
        """``base.DynamicVars.Repeat.IntValue`` → ``times_var="Repeat"``（不是认不出）。"""
        for cid in ("peck", "celestial_might", "sword_boomerang"):
            effects = _damage_effects(cid)
            self.assertTrue(effects, f"{cid} 没有伤害效果")
            self.assertEqual(effects[0].get("times_var"), "Repeat", cid)
            self.assertNotIn("WithHitCount", " ".join(_record(cid).get("unsupported") or []))

    def test_runtime_formula_count_is_still_a_gap(self):
        """⭐ ``CalculatedVar`` 是运行期公式 —— 必须继续报缺口，不能当常量变量。"""
        for cid in ("finisher", "flechettes", "radiate", "tear_asunder"):
            unsupported = " ".join(_record(cid).get("unsupported") or [])
            self.assertIn("WithHitCount", unsupported,
                          f"{cid} 的『按局面算次数』被当成常量变量了（会静默打错次数）")
            for effect in _damage_effects(cid):
                self.assertIsNone(effect.get("times_var"),
                                  f"{cid} 不该有 times_var")

    def test_local_variable_count_is_still_a_gap(self):
        """局部变量里藏着条件表达式（``dismantle`` 的 ``hitCount``）同样要报缺口。"""
        unsupported = " ".join(_record("dismantle").get("unsupported") or [])
        self.assertIn("WithHitCount", unsupported)

    def test_engine_expands_the_repeat_count(self):
        """引擎侧：``Repeat=3`` 的 `peck` 展开成 3 次伤害，升级 +1 变 4 次。"""
        from sts2_sim.featurize import configure_content
        configure_content(str(CONTENT))
        from sts2_sim import content
        definition = content.CARD_DB["peck"]
        self.assertEqual([e.amount for e in definition.effects], [2, 2, 2])
        self.assertEqual([e.amount for e in definition.upgrade], [2, 2, 2, 2],
                         "升级加的是 **Repeat**（次数），不是伤害")


if __name__ == "__main__":
    unittest.main()
