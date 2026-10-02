"""静态造牌工厂（``X.CreateInHand``）的回归测试。

真机有一批卡的效果**不是** ``*Cmd.*`` 调用，而是**卡牌类的静态工厂**：

    Shiv.CreateInHand(base.Owner, base.CombatState);                 // BladeDance.cs:29
    Shiv.CreateInHand(base.Owner, base.DynamicVars.Cards.IntValue, …) // HiddenDaggers.cs:32
    Soul.Create(base.Owner, 3, base.CombatState)                      // Severance.cs:31

命令扫描（``COMMAND_CALL``）只认 ``*Cmd.*``，所以这些句子在旧实现里**整条看不见**。
后果是本项目最难发现的一类错误 —— **报告干净、卡却比真机弱**：

| 卡 | 旧抽取结果 | 真机 |
|---|---|---|
| ``cloak_and_dagger`` | 只有 6 点格挡 | 6 点格挡 **+ 1 张 Shiv 入手** |
| ``leading_strike`` | 只有 3 点伤害 | 3 点伤害 **+ 2 张 Shiv 入手** |
| ``blade_dance`` | 空卡 | 3 张 Shiv 入手 |

本文件把三件事钉住：

1. 工厂调用被抽成 ``add_card``，**张数与落位**都对（``pile=hand``）；
2. 抽不出的（X 张 / 循环×张数 / 运行期张数）**照旧报缺口**，不许猜；
3. 顺手补上的"无条件静默缺口扫描"真的会把 ``EnergyCost.AddThisCombat``
   这类**改机制的成员调用**报出来 —— 否则补上造牌之后 `up_my_sleeve`
   的「每次打出费用 -1」就变成静默丢失了。
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


def _record(cid: str) -> dict:
    cards = json.loads((CONTENT / "cards_source.json").read_text(encoding="utf-8"))
    return next(c for c in cards if c["cid"] == cid)


def _added(cid: str) -> list[dict]:
    return [e for e in _record(cid)["effects"] if e.get("op") == "add_card"]


def _unsupported(cid: str) -> str:
    return " | ".join(_record(cid).get("unsupported") or [])


class TestFactoryExtraction(unittest.TestCase):
    """抽取层：工厂调用变成 ``add_card``，张数与落位来自**源码**。"""

    @classmethod
    def setUpClass(cls):
        if not CONTENT.exists():
            raise unittest.SkipTest(f"缺内容目录 {CONTENT}")

    def test_loop_bound_factory_becomes_counted_add(self):
        """``for i < DynamicVars.Cards`` + 两参 ``CreateInHand`` → 加 ``Cards`` 张。

        出处：``BladeDance.OnPlay``（BladeDance.cs:27-31）与
        ``Shiv.CreateInHand(Player, ICombatState, Player?)``（Shiv.cs:83，
        两参重载恒为**一张**）。循环把"一张"重复 ``Cards`` 次。
        """
        for cid, var in (("blade_dance", "Cards"),
                         ("cloak_and_dagger", "Cards"),
                         ("leading_strike", "Shivs")):
            with self.subTest(cid=cid):
                added = _added(cid)
                self.assertEqual(len(added), 1, cid)
                self.assertEqual(added[0]["card"], "shiv", cid)
                self.assertEqual(added[0]["pile"], "hand", cid)
                self.assertEqual(added[0]["amount_var"], var, cid)
                self.assertEqual(_unsupported(cid), "", cid)

    def test_explicit_count_factory_becomes_counted_add(self):
        """三参重载：``Shiv.CreateInHand(owner, DynamicVars["Shivs"].IntValue, state)``。

        出处：``HiddenDaggers.OnPlay``（HiddenDaggers.cs:32）。这张卡另外还
        **升级生成的 Shiv**（``CardCmd.Upgrade(item)``，写在
        ``if (!base.IsUpgraded) return;`` 之后），引擎的 ``add_card`` 造的是
        基础形态 —— 所以它必须继续被拒，而不是"看起来干净"。

        ⚠️ 这条断言在改造牌算子那一批（``docs/12`` §2.14/§2.15）之后**换了形式**：
        升级那一步不再留在效果表里（``upgrade_card`` 引擎没有这个算子），
        而是被 ``upgraded_only_spans`` 认成"**升级版专属效果**"缺口。
        卡照样被拒 —— 被验的是"它不能看起来干净"，不是缺口的具体写法。
        """
        added = _added("hidden_daggers")
        self.assertEqual([e["card"] for e in added], ["shiv"])
        self.assertEqual(added[0]["amount_var"], "Shivs")
        gaps = " ".join(_record("hidden_daggers").get("unsupported") or [])
        self.assertIn("升级版专属效果", gaps,
                      "升级生成的 Shiv 那一步必须留下缺口（否则基础版会被当成"
                      "已升级，静默变强）")
        self.assertNotEqual(_unsupported("hidden_daggers"), "")

    def test_direct_factory_argument_resolves(self):
        """实参**直接是**工厂调用：``AddGeneratedCardsToCombat(Soul.Create(owner, 3, …))``。

        出处：``GraveWarden.OnPlay``（GraveWarden.cs:33）。旧实现只认"裸变量名"，
        这种写法整条报成"卡 id 解析不出" —— 而它一点都不含糊。
        """
        added = _added("grave_warden")
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["card"], "soul")
        self.assertEqual(added[0]["amount_var"], "Cards")
        self.assertEqual(added[0]["pile"], "draw")
        self.assertEqual(added[0]["position"], "random")

    def test_indexed_collection_elements_resolve(self):
        """``souls[0] / souls[1] / souls[2]`` 分别落到抽牌堆 / 弃牌堆 / 手牌。

        出处：``Severance.OnPlay``（Severance.cs:31-34）。元素类型来自
        **定义那个集合的工厂**（``Soul.Create(…)``），不是按下标猜的。
        """
        added = _added("severance")
        self.assertEqual([e["card"] for e in added], ["soul", "soul", "soul"])
        self.assertEqual([e["pile"] for e in added], ["draw", "discard", "hand"])
        self.assertEqual([e["amount"] for e in added], [1, 1, 1])
        self.assertEqual(added[0]["position"], "random")
        self.assertEqual(_unsupported("severance"), "")

    def test_runtime_count_is_still_a_gap(self):
        """运行期张数（``handSize``）**不许猜** —— 猜成 1 就是强度差一大截。"""
        self.assertIn("Shiv.CreateInHand", _unsupported("storm_of_steel"))
        self.assertIn("张数抽不出", _unsupported("storm_of_steel"))


class TestSilentGapScan(unittest.TestCase):
    """无条件静默缺口扫描：补上造牌之后，别把降费/免费弄丢。"""

    @classmethod
    def setUpClass(cls):
        if not CONTENT.exists():
            raise unittest.SkipTest(f"缺内容目录 {CONTENT}")

    def test_energy_cost_change_is_reported(self):
        """``base.EnergyCost.AddThisCombat(-1)``（``UpMySleeve``，UpMySleeve.cs:46）。

        这条以前是靠"一条效果都没抽到"的兜底扫描报出来的；补上 ``CreateInHand``
        造牌之后效果表非空，兜底不再触发 —— 必须改成**无条件**扫描才不丢。
        """
        self.assertIn("战斗内改费用", _unsupported("up_my_sleeve"))
        self.assertTrue(_added("up_my_sleeve"), "造牌那半边仍要抽出来（两边互不替代）")

    def test_set_to_free_is_reported(self):
        """``cardModel.SetToFreeThisTurn()``（``Discovery``，Discovery.cs:36）。"""
        self.assertIn("免费", _unsupported("discovery"))

    def test_clone_is_reported(self):
        """``selection.CreateClone()``（``DualWield`` / ``HeirloomHammer``）。"""
        self.assertIn("克隆", _unsupported("dual_wield"))

    def test_plain_container_calls_are_not_reported(self):
        """反向：``list.Add`` / 表现层 ``AddChildSafely`` **不算**缺口。

        口径与 ``EFFECT_VERBS`` 一致 —— "结论对、理由错"的告警比没有告警更糟。
        """
        self.assertNotIn("list.Add", _unsupported("crash_landing"))
        self.assertNotIn("AddChildSafely", _unsupported("bombardment"))


class TestFactoryRuntime(unittest.TestCase):
    """引擎侧：打出去之后牌**真的**进了该进的牌堆。"""

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)
        from sts2_sim.content import STARTING_DECK
        cls.deck = STARTING_DECK

    def _combat_with(self, cid: str):
        from sts2_sim.core import CardInstance, start_combat
        state = start_combat(self.deck, ("nibbit",), seed=7)
        state.hand = [CardInstance(cid)] + [CardInstance("defend_ironclad")]
        state.draw_pile = [CardInstance("defend_ironclad") for _ in range(6)]
        state.discard = []
        state.energy = 5
        return state

    def _play(self, cid: str):
        from sts2_sim.core import Action, step
        state = self._combat_with(cid)
        # 攻击牌要指定目标（真机 ``TargetType.AnyEnemy`` 必须选一个敌人）；
        # 格挡 / 技能牌用 -1。
        from sts2_sim.content import CARD_DB
        target = -1 if CARD_DB[cid].target == "self" else 0
        step(state, Action("play_card", 0, target))
        return state

    @staticmethod
    def _first_amount(cid: str, op: str) -> int:
        """从**卡牌表**（引擎实际用的数值）取第一个该算子的量 —— 不手抄数字。"""
        from sts2_sim.content import CARD_DB
        return next(e.amount for e in CARD_DB[cid].effects if e.op == op)

    def test_blade_dance_adds_three_shivs(self):
        """``blade_dance``（1 费）：手牌 +3 张 ``shiv``；升级版 +4。"""
        state = self._play("blade_dance")
        shivs = [c for c in state.hand if c.cid == "shiv"]
        self.assertEqual(len(shivs), 3)
        self.assertFalse(any(c.upgraded for c in shivs))

    def test_cloak_and_dagger_blocks_and_adds_a_shiv(self):
        """``cloak_and_dagger``（1 费）：6 点格挡 **且** 1 张 Shiv。"""
        state = self._play("cloak_and_dagger")
        self.assertEqual(state.player.block, self._first_amount("cloak_and_dagger", "block"))
        self.assertEqual([c.cid for c in state.hand].count("shiv"), 1)

    def test_leading_strike_hits_and_adds_two_shivs(self):
        """``leading_strike``（1 费）：3 点伤害 **且** 2 张 Shiv。"""
        state = self._play("leading_strike")
        damage = self._first_amount("leading_strike", "damage")
        self.assertEqual(state.enemies[0].hp, state.enemies[0].max_hp - damage)
        self.assertEqual([c.cid for c in state.hand].count("shiv"), 2)

    def test_grave_warden_puts_souls_into_the_draw_pile(self):
        """``grave_warden``：8 点格挡 + ``Cards(=1)`` 张 ``soul`` 进**抽牌堆**。

        ⚠️ 张数取自源码的 ``new CardsVar(1)``（GraveWarden.cs:19）——
        社区描述写的是 2 张，**源码是唯一真相**。
        """
        from sts2_sim.content import CARD_DB
        state = self._play("grave_warden")
        self.assertEqual(state.player.block, self._first_amount("grave_warden", "block"))
        expected = next(e.amount for e in CARD_DB["grave_warden"].effects
                        if e.op == "add_card")
        self.assertEqual(expected, 1, "源码 GraveWarden.cs:19 写的是 CardsVar(1)")
        self.assertEqual([c.cid for c in state.draw_pile].count("soul"), expected)
        self.assertEqual([c.cid for c in state.hand].count("soul"), 0)

    def test_severance_splits_three_souls_across_piles(self):
        """``severance``：1 张进抽牌堆、1 张进弃牌堆、1 张进手牌。"""
        state = self._play("severance")
        self.assertEqual([c.cid for c in state.draw_pile].count("soul"), 1)
        self.assertEqual([c.cid for c in state.discard].count("soul"), 1)
        self.assertEqual([c.cid for c in state.hand].count("soul"), 1)


class TestMultiplayerOnlyCards(unittest.TestCase):
    """多人专用卡**排除**而不是近似实现（``docs/09`` 铁律 5）。"""

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_all_allies_cards_are_excluded(self):
        """``TargetType.AllAllies`` 的牌作用于**队友**，单人局里队友恒为空集。

        ⚠️ 抽取器把效果目标一律落成 ``self`` / ``enemy`` 两档，所以在旧实现里
        ``energy_surge``（给队友能量）会变成**白给自己能量** —— 比真机强，
        而且报告里看不出来。``rally``（给队友格挡）同理。

        判据落在 ``content._source_reject_reason`` 上（门禁的唯一入口）：
        这条规则命中时源码版不被采纳，卡退回文本版 / 占位版并标
        ``effects_incomplete``，于是**离开训练集**。
        """
        import json
        from sts2_sim import content, eligibility
        records = {r["cid"]: r for r in json.loads(
            (CONTENT / "cards_source.json").read_text(encoding="utf-8"))}
        admitted = eligibility.admitted_cards()
        for cid in ("energy_surge", "rally", "blade_symphony"):
            with self.subTest(cid=cid):
                reason = content._source_reject_reason(records[cid]) or ""
                self.assertIn("all_allies", reason, cid)
                self.assertNotIn(cid, admitted, f"{cid} 是多人专用卡，不该进训练集")

    def test_multiplayer_declaration_matches_target(self):
        """源码声明与 ``TargetType`` 一致：声明 ``MultiplayerOnly`` 的牌都是 ``all_allies``。

        出处：``CardModel.MultiplayerConstraint``（37 张卡声明
        ``CardMultiplayerConstraint.MultiplayerOnly``），其中 8 张的构造参数是
        ``TargetType.AllAllies``。这条断言把"两边口径一致"钉住 ——
        将来只改一边会被立刻发现。
        """
        src = ROOT / "data" / "decompiled" / "sts2" / "MegaCrit.Sts2.Core.Models.Cards"
        if not src.is_dir():
            self.skipTest("缺反编译源码")
        from tools.extract_cards import snake
        declared = {snake(p.stem) for p in src.glob("*.cs")
                    if "MultiplayerConstraint.MultiplayerOnly"
                    in p.read_text(encoding="utf-8", errors="replace")}
        self.assertIn("blade_symphony", declared)
        self.assertNotIn("blade_dance", declared)


if __name__ == "__main__":
    unittest.main()
