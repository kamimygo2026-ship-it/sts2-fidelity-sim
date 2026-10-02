"""``Repeat`` 机制（"把这段效果重复 N 次"）的回归测试。

真机大量卡写成"循环 N 次"，N 来自 ``CanonicalVars``：

```csharp
IceLance:       for (int i = 0; i < base.DynamicVars.Repeat.IntValue; i++)
                    await OrbCmd.Channel<FrostOrb>(choiceContext, base.Owner);
Quadcast:       for (int i = 0; i < base.DynamicVars.Repeat.IntValue; i++)
                    await OrbCmd.EvokeNext(choiceContext, base.Owner,
                                           i == base.DynamicVars.Repeat.IntValue - 1);
CloakAndDagger: GainBlock(Block); for (int i = 0; i < Cards.IntValue; i++) Shiv.CreateInHand(…);
```

抽取器原来只认"上界依赖 X"的循环，这类**常量变量上界**的循环被整个忽略 ——
循环体里的命令只算一次，于是卡明显变弱（冰枪该引导 3 个冰球却只引导 1 个），
而且**没有任何标记**（`unsupported` 是空的），直接混进训练集。

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 冰枪只引导 1 个冰球 | 常量变量上界的 `for` 循环没被识别 | `test_ice_lance_channels_repeat_times` |
| 四重施法第一次就把球移走 | `dequeue` 没跟着循环变量走 | `test_quadcast_only_dequeues_on_the_last_evoke` |
| 变量查不到时次数静默变成 1 | 解析层没有 `times_var` 分支 | `test_unresolved_times_var_marks_the_card_incomplete` |
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from sts2_sim.content import CARD_DB

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"
CARD_SRC = ROOT / "data" / "decompiled" / "sts2" / "MegaCrit.Sts2.Core.Models.Cards"


class TestRepeatMechanic(unittest.TestCase):
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

    def test_ice_lance_channels_repeat_times(self):
        """``IceLance``：``Repeat=3`` → **3 个冰球**（外加 19 点伤害）。"""
        card = CARD_DB["ice_lance"]
        channels = [e for e in card.effects if e.op == "channel"]
        self.assertEqual(len(channels), 3, f"应引导 3 个冰球，实际 {len(channels)}")
        self.assertTrue(all(e.orb == "frost" for e in channels))
        self.assertEqual([e.amount for e in card.effects if e.op == "damage"], [19])

    def test_quadcast_only_dequeues_on_the_last_evoke(self):
        """``Quadcast``：``EvokeNext(…, i == Repeat-1)`` → **前 3 次不移除、最后 1 次移除**。

        ⚠️ 这条是"次数正确但语义错误"的典型：只看参数里有没有 ``false``
        会得出"每次都移除"，于是第一次就把球移走、后面三次激发空气 ——
        比"只激发 1 次"更错。
        """
        card = CARD_DB["quadcast"]
        evokes = [e for e in card.effects if e.op == "evoke_next"]
        self.assertEqual(len(evokes), 4, f"应激发 4 次，实际 {len(evokes)}")
        self.assertEqual([e.dequeue for e in evokes], [False, False, False, True])

    def test_refract_repeats_both_parts(self):
        """``Refract``：2 段伤害（``WithHitCount``）+ ``Repeat=2`` 个玻璃球。"""
        card = CARD_DB["refract"]
        self.assertEqual(len([e for e in card.effects if e.op == "damage"]), 2)
        channels = [e for e in card.effects if e.op == "channel"]
        self.assertEqual(len(channels), 2)
        self.assertTrue(all(e.orb == "glass" for e in channels))

    def test_repeat_cards_are_admitted_again(self):
        """机制实现之后，卡应当**自动**回到可训练集（门禁与实现同一套判据）。"""
        from sts2_sim import eligibility
        for cid in ("ice_lance", "quadcast", "refract"):
            self.assertTrue(eligibility.card_admission(cid).admitted,
                            f"{cid} 的 Repeat 已经实现，应当重新准入")

    def test_unresolved_times_var_marks_the_card_incomplete(self):
        """反向验证：``times_var`` 查不到时必须**整张卡标残缺**，不能默认成 1。

        默认成 1 就是"静默变弱" —— 正是这一整类 bug 的根源。
        """
        from sts2_sim.content import _resolve_effects
        record = {"effects": [{"op": "channel", "amount": 1, "amount_var": None,
                               "power": None, "target": "self", "times": 1,
                               "orb": "frost", "times_var": "__missing__"}]}
        self.assertIsNone(_resolve_effects(record["effects"], {}),
                          "变量查不到时必须返回 None（→ 标记残缺）")

    def test_loop_bound_is_parsed(self):
        """``loop_bound`` 从 ``for`` 头部取出上界表达式。"""
        tool = self._tool()
        self.assertEqual(tool.loop_bound("int i = 0; i < base.DynamicVars.Repeat.IntValue; i++"),
                         "base.DynamicVars.Repeat.IntValue")
        self.assertEqual(tool.loop_bound("int i = 0; i <= 3; i++"), "3")
        self.assertIsNone(tool.loop_bound("var x in list"))

    @staticmethod
    def _tool():
        path = ROOT / "tools" / "extract_cards.py"
        spec = importlib.util.spec_from_file_location("extract_cards_repeat", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["extract_cards_repeat"] = module
        spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main(verbosity=2)
