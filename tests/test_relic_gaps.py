"""遗物钩子门禁的回归测试（``docs/12`` §2.17）。

本批次修的是**两处抽取器误报** —— 它们把本来可以采纳的遗物钩子判成了缺口：

| 误报 | 受影响 | 症状 |
|---|---|---|
| `while (…)` 循环体里的 `if (…) { break; }` 被当成**触发条件** | `delicate_frond` 等 | "战斗开始前把空药水槽买满"整条被拒 |
| `SetToFreeThisTurn` 这个**方法名**被"私有回合计数器"正则命中 | `vexing_puzzlebox` 等 | 明明只是"生成的牌本回合免费"，却被判成 stateful 拒绝 |

⚠️ 修第一处时**不能顺手放宽**：`HappenedThisTurn` / `LostHpThisTurn` /
`AnyCardsPlayedThisTurn` 也是"方法调用"，但它们是**运行期历史查询**
（引擎没有战斗历史）—— 放宽了会把 `ripple_basin` 的"本回合没打出过攻击牌"
当成已建模，而它在 `BeforeSideTurnEnd` 时机上根本算不出 `card_type`，
于是"被采纳却永不触发"。这类判据必须**按名字白名单**，不能按"后面有没有括号"。

⚠️ 这些测试证明的是"钩子被正确采纳/拒绝"，**不是**数值与真机一致
（真机对拍为 0，见 ``docs/13`` 头部）。
"""

from __future__ import annotations

import collections
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"
RELICS = (ROOT / "data" / "decompiled" / "sts2"
          / "MegaCrit.Sts2.Core.Models.Relics")


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


@unittest.skipUnless(RELICS.is_dir(), "缺反编译源码")
class TestRelicGuardExtraction(unittest.TestCase):
    @staticmethod
    def _body(name: str, hook: str) -> str:
        from tools.extract_relics import method_body
        path = next(RELICS.rglob(f"{name}.cs"))
        body = method_body(path.read_text(encoding="utf-8", errors="replace"), hook)
        assert body is not None, f"{name}.{hook} 没有方法体"
        return body

    def test_break_guard_is_not_a_trigger_condition(self):
        """``while`` 里的 ``if (…) { break; }`` 只是循环控制流，不是触发守卫。

        ``DelicateFrond.BeforeCombatStart``::

            while (base.Owner.HasOpenPotionSlots) {
                potion = PotionFactory.CreateRandomPotionOutOfCombat(…);
                if (!(await PotionCmd.TryToProcure(potion, …)).success) break;
            }

        把它当触发条件 → 整个钩子被判"触发条件认不出"，两条缺口叠加
        （`PotionCmd.TryToProcure` + 触发条件）—— 而它其实一条都不缺。
        """
        from tools.extract_relics import if_conditions, unrecognized_guard
        body = self._body("DelicateFrond", "BeforeCombatStart")
        conditions = if_conditions(body)
        self.assertNotIn("!(await PotionCmd.TryToProcure(potion, base.Owner)).success",
                         conditions, "break 守卫不该进条件表")
        self.assertFalse(unrecognized_guard(body),
                         "这个钩子的触发条件应当全部认得")

    def test_set_to_free_method_name_is_not_a_counter(self):
        """``SetToFreeThisTurn()`` 是**方法调用**，不是私有回合计数器。

        ``VexingPuzzlebox.AfterPlayerTurnStart`` 里唯一带 ``ThisTurn`` 的标识符
        就是它 —— 被当成计数器之后，这个遗物整条判 stateful 拒绝。
        """
        from tools.extract_relics import hook_scope
        body = self._body("VexingPuzzlebox", "AfterPlayerTurnStart")
        self.assertEqual(hook_scope(body).get("stateful"), [],
                         "SetToFreeThisTurn 不该被算成状态条件")

    def test_runtime_history_queries_still_are_gaps(self):
        """⚠️ 但 ``HappenedThisTurn`` 这类**历史查询**必须仍是缺口。

        ``RippleBasin.BeforeSideTurnEnd`` 的条件是"本回合**没有**打出过攻击牌"
        （``!History.CardPlaysFinished.Any(e => e.HappenedThisTurn(…) && …)``）。
        引擎没有战斗历史；放宽判据会让它被"采纳"，
        而钩子里那个 `card_type` 守卫在 `BeforeSideTurnEnd` 时机上算不出
        → 永远不触发（"报告说完全复刻、实际什么都不做"）。
        """
        from tools.extract_relics import hook_scope
        body = self._body("RippleBasin", "BeforeSideTurnEnd")
        self.assertIn("per_turn_counter", hook_scope(body).get("stateful") or [],
                      "运行期历史查询必须继续算作状态缺口")


class TestAdoptedRelicHooks(unittest.TestCase):
    """两个遗物从"被误拒"变成"有行为"。"""

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_delicate_frond_fills_empty_potion_slots(self):
        from sts2_sim import content
        relic = content.RELICS.get("delicate_frond")
        self.assertIsNotNone(relic, "delicate_frond 应当在遗物表里")
        self.assertTrue(relic.has_behavior, "它现在应当有已复刻行为")
        effects = [e for hook in relic.hooks for e in hook.effects]
        self.assertTrue(any(e.op == "procure_random_potion" for e in effects),
                        f"应当是'填满空药水槽'的效果：{effects}")
        procure = next(e for e in effects if e.op == "procure_random_potion")
        self.assertEqual(procure.amount, 0,
                         "amount=0 表示 while 循环（填满所有空槽）")

    def test_vexing_puzzlebox_generates_a_free_card(self):
        from sts2_sim import content
        relic = content.RELICS.get("vexing_puzzlebox")
        self.assertIsNotNone(relic, "vexing_puzzlebox 应当在遗物表里")
        self.assertTrue(relic.has_behavior, "它现在应当有已复刻行为")
        effects = [e for hook in relic.hooks for e in hook.effects]
        generated = [e for e in effects if e.op == "generate_card"]
        self.assertTrue(generated, f"应当是'生成一张牌'的效果：{effects}")
        self.assertTrue(generated[0].free, "生成的牌本回合免费（SetToFreeThisTurn）")
        self.assertEqual(dict(generated[0].card_filter), {"pool": "character"},
                         "从**角色池**里随机生成")


if __name__ == "__main__":
    unittest.main()
