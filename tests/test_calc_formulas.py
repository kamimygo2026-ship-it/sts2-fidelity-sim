"""**运行期公式**（``CalculatedVar``）的源码级回归测试（`docs/12` §2.33）。

真机算式（``CalculatedVar.WithMultiplier`` 的 docstring）：

    CalculatedVar = CalculationBase + ExtraDamage × multiplier(lambda)

例子（每个都回源码核过）：

| 卡 | 公式 | 引擎的 ``calc_kind`` |
|---|---|---|
| ``BodySlam`` | ``0 + 1 × 当前格挡`` | ``own_block`` |
| ``Mimic`` | ``0 + 1 × 被打者的格挡`` | ``target_block`` |
| ``MindBlast`` | ``0 + 1 × 抽牌堆张数`` | ``draw_pile_count`` |
| ``AshenStrike`` | ``6 + 3 × 消耗堆张数`` | ``exhaust_count`` |
| ``Bully`` | ``4 + 2 × 被打者的易伤层数`` | ``target_power`` |
| ``TimesUp`` | ``0 + 1 × 被打者的末日层数`` | ``target_power`` |
| ``ExpectAFight`` | ``15 + 5 × max(0, 自己的力量)`` | ``self_power`` |
| ``PerfectedStrike`` | ``6 + 2 × 带 Strike 标签的牌数`` | ``all_cards_tag`` |
| ``GoldAxe`` | ``0 + 1 × 本场打完的牌数`` | ``card_plays_total`` |

**认不出的形状不进这张表**：抽取器会发现场公式却认不出 lambda 时继续报
"量由运行期公式决定"，整张卡排除出训练集（宁缺勿猜）。
本文件同时钉住"两张表必须一致"与"未知 kind 必须被拒"。

这些断言只说明"引擎按源码执行"，不代表已与真机对拍（``docs/09`` §5）。
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


class CalcTableTest(unittest.TestCase):
    """门禁表与求值表必须是同一套；未知 kind 必须被拒。"""

    def test_engine_tables_agree(self):
        from sts2_sim import content, powers
        self.assertEqual(powers.CALC_KINDS, content.ENGINE_CALC_KINDS)

    def test_extractor_only_emits_known_kinds(self):
        if not CONTENT.exists():
            self.skipTest("缺内容目录")
        from sts2_sim import powers
        kinds = set()
        for record in json.loads((CONTENT / "cards_source.json").read_text(
                encoding="utf-8")):
            for effect in record.get("effects") or []:
                if effect.get("calc_kind"):
                    kinds.add(effect["calc_kind"])
        self.assertTrue(kinds, "抽取产物里应当有 calc_kind（本批 10 张卡）")
        self.assertTrue(kinds <= powers.CALC_KINDS,
                        f"抽取器写出了引擎不认识的公式：{kinds - powers.CALC_KINDS}")

    def test_unknown_kind_is_rejected_not_zeroed(self):
        """认得出、算不了的公式 → **拒绝**，不许静默算成 0。"""
        from sts2_sim import content, eligibility
        effect = content.Effect(op="damage", amount=0, calc_kind="history_count",
                                calc_base=1, calc_extra=1)
        reasons = eligibility._effect_reasons((effect,), "base")
        self.assertIn("unsupported_calc:history_count@base", reasons)


class CalcFormulaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, enemies=("nibbit",), seed: int = 5):
        from sts2_sim import core
        state = core.start_combat(self.deck, tuple(enemies), seed=seed)
        state.hand = []
        state.draw_pile = []
        state.discard = []
        state.exhaust = []
        state.energy = 9
        state.player.powers.clear()
        state.player.block = 0
        for enemy in state.enemies:
            enemy.hp = enemy.max_hp = 200
            enemy.block = 0
            enemy.powers.clear()
        return state

    def play(self, state, cid: str, target: int = 0, upgraded: bool = False):
        from sts2_sim import core
        card = core.CardInstance(cid, upgraded=upgraded)
        state.hand.append(card)
        state.energy = 9
        return core.step(state, core.Action("play_card", state.hand.index(card),
                                            target))

    def test_own_block(self):
        state = self.combat()
        state.player.block = 12
        self.play(state, "body_slam")
        self.assertEqual(200 - state.enemies[0].hp, 12)

    def test_target_block(self):
        """``Mimic``：自己获得格挡、量取**被打者**的格挡（公式 target ≠ 效果 target）。"""
        state = self.combat()
        state.enemies[0].block = 17
        self.play(state, "mimic")
        self.assertEqual(state.player.block, 17)

    def test_draw_pile_count(self):
        from sts2_sim import core
        state = self.combat()
        state.draw_pile = [core.CardInstance("strike_ironclad") for _ in range(7)]
        self.play(state, "mind_blast")
        self.assertEqual(200 - state.enemies[0].hp, 7)

    def test_exhaust_count(self):
        from sts2_sim import core
        state = self.combat()
        state.exhaust = [core.CardInstance("defend_ironclad") for _ in range(2)]
        self.play(state, "ashen_strike")
        self.assertEqual(200 - state.enemies[0].hp, 6 + 3 * 2)

    def test_target_power(self):
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("vulnerable", 3)
        self.play(state, "bully")
        # (4 + 2×3) × 1.5（易伤）= 15
        self.assertEqual(200 - enemy.hp, 15)

    def test_target_power_doom(self):
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("doom", 9)
        self.play(state, "times_up")
        self.assertEqual(200 - enemy.hp, 9)

    def test_self_power_floors_at_zero(self):
        """源码写 ``Math.Max(0, …)``：负力量当 0（不是负数扣格挡）。"""
        state = self.combat()
        state.player.add_power("strength", -5)
        self.play(state, "expect_a_fight")
        self.assertEqual(state.player.block, 15)
        state2 = self.combat()
        state2.player.add_power("strength", 2)
        self.play(state2, "expect_a_fight")
        self.assertEqual(state2.player.block, 15 + 5 * 2)

    def test_all_cards_tag_counts_the_played_card_itself(self):
        """``PerfectedStrike`` 自己带 ``Strike`` 标签 → 它把自己也算进去。

        这不是 bug：真机 ``AllCards`` 包含出牌区，而这张牌的名字里就有 "Strike"
        （StS1 同样如此）。少算它反而与真机不符。
        """
        from sts2_sim import core
        state = self.combat()
        state.draw_pile = [core.CardInstance("strike_ironclad") for _ in range(3)]
        state.discard = [core.CardInstance("strike_ironclad")]
        self.play(state, "perfected_strike")
        self.assertEqual(200 - state.enemies[0].hp, 6 + 2 * 5,
                         "4 张堆里的打击 + 它自己 = 5")

    def test_card_plays_total_spans_turns(self):
        """``GoldAxe`` 读的是**整场**打完的牌数（不是本回合）。

        第一回合打两张，第二回合打金斧 → 伤害 2。
        如果误用"本回合已打出的牌数"（``cards_played_started_this_turn``），
        第二回合是 0 → 伤害 0，而日志一切正常。
        """
        from sts2_sim import core
        state = self.combat()
        self.play(state, "defend_ironclad")
        self.play(state, "defend_ironclad")
        core.step(state, core.Action("end_turn"))
        self.assertEqual(state.cards_played_started_this_turn, 0, "新回合的本回合计数已清零")
        state.energy = 9
        state.hand = []
        hp = state.enemies[0].hp
        self.play(state, "gold_axe")
        self.assertEqual(hp - state.enemies[0].hp, 2,
                         "整场 2 张（本张还没收尾，不算自己）")

    def test_upgrade_delta_applies_to_the_extra_term(self):
        """升级改的是 ``ExtraDamage``（增量）—— 没套上就会"升了级还是基础值"。"""
        from sts2_sim import core
        state = self.combat()
        state.draw_pile = [core.CardInstance("strike_ironclad") for _ in range(4)]
        self.play(state, "perfected_strike", upgraded=True)
        # 升级后 ExtraDamage 2 → 3：6 + 3×5（含它自己）
        self.assertEqual(200 - state.enemies[0].hp, 6 + 3 * 5)


class ApplyPowerCalcTest(unittest.TestCase):
    """⭐ 公式也能给**上多少层**用（`docs/12` §2.34）。

    这三张卡的量都是"当场算"的：`Hang` 的 ``max(2, 目标身上的绞刑层数)``、
    `Dominate` 的"目标当前的易伤层数"、`Synchronize` 的"不同球种数"。
    公式没接上时它们会按 **0 层**结算 —— 打出去什么都不发生，日志却正常。
    """

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def combat(self, enemies=("nibbit", "nibbit"), seed: int = 5):
        from sts2_sim import core
        state = core.start_combat(self.deck, tuple(enemies), seed=seed)
        state.hand = []
        state.draw_pile = []
        state.discard = []
        state.exhaust = []
        state.energy = 9
        state.player.powers.clear()
        state.player.block = 0
        for enemy in state.enemies:
            enemy.hp = enemy.max_hp = 500
            enemy.block = 0
            enemy.powers.clear()
        return state

    def play(self, state, cid: str, target: int = 0):
        from sts2_sim import core
        card = core.CardInstance(cid)
        state.hand.append(card)
        state.energy = 9
        return core.step(state, core.Action("play_card", state.hand.index(card),
                                            target))

    def test_hang_escalates_by_its_own_layers(self):
        """``Hang``：伤害 × 层数；层数每次取 ``max(2, 当前层数)``。

        10 → 20 → 40（层数 2 → 4 → 8）。真机的两步顺序是**先伤害、后加层**，
        所以第二张吃的是第一张留下的 2 层。
        """
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "hang")
        self.assertEqual(500 - enemy.hp, 10)
        self.assertEqual(enemy.power("hang"), 2)
        hp = enemy.hp
        self.play(state, "hang")
        self.assertEqual(hp - enemy.hp, 20, "10 × 2 层")
        self.assertEqual(enemy.power("hang"), 4, "max(2, 2) = 2 → 2+2 = 4")
        hp = enemy.hp
        self.play(state, "hang")
        self.assertEqual(hp - enemy.hp, 40, "10 × 4 层")
        self.assertEqual(enemy.power("hang"), 8)

    def test_hang_multiplier_only_applies_to_the_hang_card(self):
        """``cardSource is Hang``：别的攻击牌不许吃这个倍率。

        这条最容易写错成"谁身上有绞刑就乘谁"—— 那会让**所有**攻击都翻好几倍。
        """
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "hang")
        hp = enemy.hp
        self.play(state, "strike_ironclad")
        self.assertEqual(hp - enemy.hp, 6, "打击牌只有自己的 6 点")

    def test_hang_stops_stacking_at_source_limit(self):
        """Hang.OnPlay：powerAmount + num > 999999999 时只补到上限。"""
        for layers in (499999999, 500000000, 999999998, 999999999):
            with self.subTest(layers=layers):
                state = self.combat()
                enemy = state.enemies[0]
                enemy.hp = enemy.max_hp = 100000000000
                enemy.add_power("hang", layers)
                self.play(state, "hang")
                self.assertEqual(enemy.power("hang"), min(999999999, layers * 2))

    def test_hang_does_not_leak_to_another_enemy(self):
        """倍率挂在**被打的那个**敌人身上（循环遍历全部敌人时最容易多乘一次）。"""
        state = self.combat()
        first, second = state.enemies
        self.play(state, "hang")
        hp = second.hp
        self.play(state, "hang", target=0)
        self.assertEqual(second.hp, hp, "第二只怪既不挨打也不该被乘")
        self.assertEqual(second.power("hang"), 0)

    def test_dominate_uses_the_vulnerable_it_just_applied(self):
        """``Dominate``：先上 1 层易伤，再按**目标当前**的易伤层数给自己加力量。"""
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "dominate")
        self.assertEqual(enemy.power("vulnerable"), 1)
        self.assertEqual(state.player.power("strength"), 1, "1 层易伤 → +1 力量")
        self.play(state, "dominate")
        self.assertEqual(enemy.power("vulnerable"), 2)
        self.assertEqual(state.player.power("strength"), 3, "累计 1 + 2")

    def test_synchronize_counts_distinct_orb_types(self):
        """``Synchronize``：``Orbs group by orb.Id`` → **种数**，不是球总数。"""
        state = self.combat()
        state.orbs = ["lightning", "lightning", "frost"]
        self.play(state, "synchronize")
        self.assertEqual(state.player.power("focus"), 2,
                         "3 个球但只有 2 种")
        state2 = self.combat()
        state2.orbs = ["lightning"] * 4
        self.play(state2, "synchronize")
        self.assertEqual(state2.player.power("focus"), 1, "4 个同种球 = 1")

    def test_compile_driver_draws_by_distinct_orb_types(self):
        """同一个公式也能给 ``draw`` 用。"""
        from sts2_sim import core
        state = self.combat()
        state.orbs = ["lightning", "frost", "glass"]
        state.draw_pile = [core.CardInstance("defend_ironclad") for _ in range(5)]
        self.play(state, "compile_driver")
        self.assertEqual(len(state.hand), 3, "抽 = 3 种球")


class AdmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_formula_cards_are_admitted(self):
        from sts2_sim import eligibility
        for cid in ("body_slam", "mind_blast", "perfected_strike", "bully",
                    "ashen_strike", "gold_axe", "expect_a_fight", "times_up"):
            with self.subTest(cid=cid):
                admission = eligibility.card_admission(cid)
                self.assertTrue(admission.admitted, f"{cid} 仍在拒：{admission.reasons()}")

    def test_upgrade_audit_checks_formula_coefficients(self):
        """五张牌 OnUpgrade 改的是公式系数，审计不能只比 amount=0。"""
        from dataclasses import replace
        from unittest.mock import patch
        from sts2_sim import content
        from tools.verify_implemented import check_upgrade_versions
        cids = ["ashen_strike", "bully", "expect_a_fight", "perfected_strike",
                "synchronize"]
        self.assertEqual(check_upgrade_versions(cids), [])
        for cid in cids:
            with self.subTest(cid=cid):
                card = content.CARD_DB[cid]
                with patch.dict(content.CARD_DB, {cid: replace(card, upgrade=card.effects)}):
                    self.assertEqual(len(check_upgrade_versions([cid])), 1)

    def test_source_audit_detects_formula_only_corruption(self):
        """CalculatedVar 的 kind/arg/base/extra 任一个被改都要被源码记录核对发现。"""
        from dataclasses import replace
        from unittest.mock import patch
        from sts2_sim import content
        from tools.verify_implemented import check_card_source_agreement
        card = content.CARD_DB["bully"]
        for changes in ({"calc_kind": "self_power"}, {"calc_arg": "strength"},
                        {"calc_base": 999}, {"calc_extra": 999}):
            with self.subTest(changes=changes):
                broken = replace(card, effects=(replace(card.effects[0], **changes),))
                with patch.dict(content.CARD_DB, {"bully": broken}):
                    self.assertEqual(len(check_card_source_agreement(["bully"])), 1)

    def test_multiplayer_only_formula_cards_stay_rejected(self):
        """``Mimic`` / ``DemonicShield`` 是**多人专用** —— 公式做完了也不该进单人池。"""
        from sts2_sim import eligibility
        for cid in ("mimic", "demonic_shield"):
            with self.subTest(cid=cid):
                self.assertIn("multiplayer_only", eligibility.card_reasons(cid))

    def test_unrecognised_formulas_keep_the_old_gap(self):
        """认不出的形状**仍然**报"运行期公式"（宁缺勿猜，不许悄悄放行）。"""
        records = {r["cid"]: r for r in json.loads(
            (CONTENT / "cards_source.json").read_text(encoding="utf-8"))}
        # `Protector` 的公式依赖 Osty（`FromOsty()` + 守护者血量），本批不认。
        self.assertTrue(any("运行期公式" in u
                            for u in records["protector"].get("unsupported") or []))
        self.assertFalse(any(e.get("calc_kind")
                             for e in records["protector"].get("effects") or []))


if __name__ == "__main__":
    unittest.main()
