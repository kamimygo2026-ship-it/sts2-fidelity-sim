"""抽取器的**链式 setter** 回归（``PowerCmd.Apply<X>(…).SetDamage(…)``）。

背景（``docs/12`` 的"静默丢失"一族）：命令扫描只认 ``\\w+Cmd\\.\\w+``，
``SetDamage`` 这种**链式 setter** 一个都不匹配 —— 于是它在报告里既不算
"未支持命令"、也不进效果表，卡看起来**干净**，实际少了"打多少伤害"这一块。
`TheBomb`（炸弹伤害）与 `ToricToughness`（下次给多少格挡）都是这样被悄悄削弱的。

钉住四件事：

1. 卡牌动态变量能抽成 ``set_power_var``（``TheBomb`` 的 ``BombDamage``）；
2. "上一条命令的返回值"能识别成 ``from="last_block_gain"``，**不会**被误解析成
   卡面那个数（``ToricToughness`` 的 ``SetBlock(blockAmount)``）；
3. "选中的那张卡"能识别成 ``from="remembered_card"``（``Nightmare`` 的
   ``SetSelectedCard`` —— 见 `docs/12` §2.42）；
4. 仍然认不出的 setter 要**显式**记进 ``unsupported``，绝不静默放过
   （这条原则不变，载体换成合成源码，因为 ``Nightmare`` 那条已经实现了）。
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


class TestChainedSetter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CONTENT.exists():
            raise unittest.SkipTest(f"缺内容目录 {CONTENT}")

    def _setter(self, cid: str) -> dict:
        effects = [e for e in _record(cid)["effects"]
                   if e.get("op") == "set_power_var"]
        self.assertTrue(effects, f"{cid} 没有抽出 set_power_var")
        return effects[0]

    def test_the_bomb_records_the_card_variable(self):
        effect = self._setter("the_bomb")
        self.assertEqual(effect["power"], "the_bomb")
        self.assertEqual(effect["var"], "Damage")
        self.assertEqual(effect["amount_var"], "BombDamage",
                         "量必须挂在**卡牌变量** BombDamage 上（升级 +10 才会跟着变）")

    def test_toric_toughness_uses_the_actual_block_gain(self):
        """⭐ 关键：``blockAmount`` 的局部定义里**含有** ``DynamicVars.Block``。

        直接走数值解析会解析成卡面的 5 —— 而真机拿到的是 ``GainBlock`` 的返回值
        （已经过了敏捷/脆弱/倍率）。必须解析成 ``from="last_block_gain"``。
        """
        effect = self._setter("toric_toughness")
        self.assertEqual(effect["var"], "Block")
        self.assertEqual(effect.get("from"), "last_block_gain")
        self.assertIsNone(effect.get("amount_var"),
                          "不能退化成'卡面 Block 变量'")

    def test_nightmare_records_the_selected_card_payload(self):
        """``SetSelectedCard(<选牌结果>)`` → ``from="remembered_card"``（不再是缺口）。

        出处：``Nightmare.cs:57-77`` 的 ``Apply<NightmarePower>(…).SetSelectedCard(card)``
        与 ``NightmarePower.cs:73-79`` 的 ``CreateClone()`` + ``ClearAffliction``。
        """
        record = _record("nightmare")
        setters = [e for e in record["effects"] if e.get("op") == "set_power_var"]
        self.assertEqual(len(setters), 1)
        self.assertEqual(setters[0]["power"], "nightmare")
        self.assertEqual(setters[0]["var"], "selected_card")
        self.assertEqual(setters[0]["from"], "remembered_card")
        # 选牌本身也要在，而且用途是"记住"（不移动牌）
        selections = [e for e in record["effects"] if e.get("op") == "select_card"]
        self.assertEqual([e["purpose"] for e in selections], ["remember"])
        self.assertEqual(record.get("unsupported") or [], [],
                         "这一条不再是缺口")

    def test_an_unrecognised_setter_is_still_reported(self):
        """认不出的链式 setter 仍然要**显式**报缺口（原则不变）。

        ⚠️ 载体换过了：原来是拿 ``Nightmare`` 的 ``SetSelectedCard`` 当例子，
        而它**已经实现**（见上一条）。所以这里用一段合成源码直接测抽取器本体 ——
        ``SetSomeModelRef(…)`` 传的是模型引用，``resolve_amount`` 解析不出。
        """
        from tools import extract_cards
        body = ("var p = (await PowerCmd.Apply<FooPower>(ctx, base.Owner.Creature, 1m,"
                " base.Owner.Creature, this)).SetSomeModelRef(someModel);")
        match = extract_cards.COMMAND_CALL.search(body)
        self.assertIsNotNone(match)
        effect, reason = extract_cards.chained_setter(body, match, "foo", {}, {})
        self.assertIsNone(effect, "认不出就不能产出效果")
        self.assertIn("量认不出", reason or "", "必须显式报缺口")


if __name__ == "__main__":
    unittest.main()
