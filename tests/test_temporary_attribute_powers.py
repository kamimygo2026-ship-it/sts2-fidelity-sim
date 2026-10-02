"""**临时属性族**（``Temporary{Strength,Dexterity,Focus}Power``）的源码级回归测试。

源码：``MegaCrit.Sts2.Core.Models.Powers.TemporaryStrengthPower.cs`` /
``TemporaryDexterityPower.cs`` / ``TemporaryFocusPower.cs``（三份**逐字同构**，只差
``InternallyAppliedPower``），以及各自的子类（``AnticipatePower.cs`` 等整份只有
``OriginModel``；负向的那 7 个多一行 ``IsPositive => false``）。

父类的三个回调，缺一条都会静默错：

| 回调 | 源码 | 漏掉的后果 |
|---|---|---|
| ``BeforeApplied`` | ``PowerCmd.Apply`` → 立刻 ``Apply<Xxx>(Sign*amount)`` | 上了能力但不加属性（整张牌白打） |
| ``AfterPowerAmountChanged`` | ``power == this && amount != base.Amount`` → ``Apply<Xxx>(Sign*amount)`` | 叠加的部分不加属性（第二张白打）；递减撤不回 |
| ``AfterSideTurnEnd`` | ``participants.Contains(Owner)`` → 移除并 ``Apply<Xxx>(-Sign*base.Amount)`` | 属性永久留下（或提前消失） |

⚠️ 最容易被写成"看起来对"的一条是**两条施加路径重叠**：真机里"新造实例"走
``BeforeApplied``（``PowerCmd.cs:136/159``），"叠加"走 ``ModifyAmount``（``:119``），
后者**不调** ``BeforeApplied``、只发 ``AfterPowerAmountChanged``（``:249``）。
引擎里 ``on_applied`` 两条路径都发（有些能力按"打出次数"计数，例如
``CrimsonMantlePower`` 的 ``SelfDamage``，叠加也必须发），所以靠 ``fresh`` 区分。
两边都做就会把力量加两遍 —— 本文件第 3、4 条正是钉这个。

这些断言只说明"引擎按源码执行"，不代表已与真机对拍（``docs/09`` §5）。
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"
DECOMPILED = ROOT / "data" / "decompiled" / "sts2"

#: 本批（`docs/12` §2.28）批量登记的 18 个子类。来源：
#: ``data/decompiled/sts2/**/*.cs`` 里 ``class X : Temporary*Power`` 的全集
#: 减去此前已单独登记的 ``setup_strike`` / ``mangle`` /
#: ``monarchs_gaze_strength_down``。
BATCH_PIDS = (
    "anticipate", "coordinate", "crush_under", "dark_shackles", "dying_star",
    "enfeebling_touch", "fade", "feeding_frenzy", "focused_strike", "hotfix",
    "hyperbeam_focus_down", "piercing_wail", "synchronize",
    "flex_potion", "shackling_potion", "speed_potion",
    "helical_dart", "reptile_trinket",
)

#: 被这些能力放行的卡（"能力实现"这一条不再拦它们）。
#: ``dying_star`` 仍有源码抽取缺口；Synchronize.OnPlay 的球种数公式已经接入。
ADMITTED_CARDS = (
    "anticipate", "crush_under", "dark_shackles",
    "enfeebling_touch", "feeding_frenzy", "focused_strike",
    "hotfix", "hyperbeam", "piercing_wail", "synchronize",
)
STILL_BLOCKED_FOR_OTHER_REASONS = ("dying_star",)

#: 同族里声明了 ``MultiplayerOnly`` 的两张 —— 它们**不该**被"能力做完"放行
#: （`docs/12` §2.31.3 测出、§2.32 加的门禁）。
MULTIPLAYER_ONLY_CARDS = ("coordinate", "fade")


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


class TemporaryAttributeTestCase(unittest.TestCase):
    """公共脚手架：一副起手牌 + 一只怪，以及"把手牌打出去"与"推一方回合结束"。"""

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
        state.energy = 9
        state.player.powers.clear()
        for enemy in state.enemies:
            enemy.hp = enemy.max_hp = 100
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

    def player_side_end(self, state):
        """推**玩家阵营**回合结束（``AfterSideTurnEnd`` 的玩家那一次分发）。"""
        from sts2_sim import core
        core.tick_powers(state.player, state, [])

    def enemy_side_end(self, state):
        """推**敌方阵营**回合结束（同一钩子的敌方那一次分发）。"""
        from sts2_sim import core
        core._end_enemy_side_turn(state, [])


class FamilyRegistrationTest(TemporaryAttributeTestCase):
    """族完整性：源码里每个同构子类都必须在 ``RULES`` 里、且三个回调齐全。"""

    def test_table_covers_every_source_subclass(self):
        """``data/decompiled`` 里 ``class X : Temporary*Power`` 的全集 = 引擎表。

        少了任何一行，源码里就有个"临时属性能力"在引擎里**整条不存在** ——
        施加它的卡会被门禁拒（看得见），但如果那张卡还有别的理由被拒，
        这条缺口就**静默**了（`docs/12` §2.26.3 的那类问题）。
        """
        if not DECOMPILED.exists():
            self.skipTest(f"缺反编译源码 {DECOMPILED}")
        from sts2_sim import powers
        found = set()
        pattern = re.compile(r"class\s+(\w+)\s*:\s*Temporary(?:Strength|Dexterity|Focus)Power\b")
        for path in DECOMPILED.rglob("*.cs"):
            text = path.read_text(encoding="utf-8", errors="replace")
            found.update(pattern.findall(text))
        found = {re.sub(r"Power$", "", name) for name in found
                 if not name.startswith("Mock")}
        found = {_snake(name) for name in found}
        self.assertEqual(found, set(powers._TEMPORARY_ATTRIBUTE_POWERS))

    def test_every_family_power_has_all_three_callbacks(self):
        """三条回调缺一不可（少第二条 = 叠加白打；少第三条 = 属性永久留下）。"""
        from sts2_sim import powers
        for pid, (attr, sign, _apply, _expire) in (
                powers._TEMPORARY_ATTRIBUTE_POWERS.items()):
            with self.subTest(pid=pid):
                rule = powers.RULES.get(pid)
                self.assertIsNotNone(rule, f"{pid} 没进 RULES（门禁会拒掉它的卡）")
                self.assertIsNotNone(rule.on_applied, f"{pid} 缺 BeforeApplied")
                self.assertIsNotNone(rule.on_power_amount_changed,
                                     f"{pid} 缺 AfterPowerAmountChanged")
                self.assertIsNotNone(rule.on_owner_turn_end,
                                     f"{pid} 缺 AfterSideTurnEnd")
                self.assertIn(attr, ("strength", "dexterity", "focus"))
                self.assertIn(sign, (-1, 1))


class FirstApplicationTest(TemporaryAttributeTestCase):
    """``BeforeApplied``：**首次**施加立刻按 ``Sign × Amount`` 改内部属性。"""

    def test_positive_strength_card(self):
        state = self.combat()
        self.play(state, "feeding_frenzy")
        self.assertEqual(state.player.power("feeding_frenzy"), 5)
        self.assertEqual(state.player.power("strength"), 5)

    def test_negative_strength_card_lands_on_the_enemy(self):
        state = self.combat()
        self.play(state, "dark_shackles")
        self.assertEqual(state.enemies[0].power("strength"), -9,
                         "IsPositive=false → Sign=-1，扣的是目标的力量")
        self.assertEqual(state.player.power("strength"), 0)

    def test_every_family_power_uses_its_own_attribute(self):
        """18 个入口逐个过一遍：施加 4 层就必须出现 ±4 点**对应**属性。

        这条同时挡住"参数抄错"：把 ``fade``（敏捷）写成力量、或把
        ``hyperbeam_focus_down`` 的符号写反，都只会在这一条上露出来。
        """
        from sts2_sim import powers
        for pid, (attr, sign, _a, _e) in powers._TEMPORARY_ATTRIBUTE_POWERS.items():
            with self.subTest(pid=pid):
                state = self.combat()
                target = state.player
                events: list[str] = []
                powers.apply_power_to(state, target, pid, 4, events)
                self.assertEqual(target.power(pid), 4)
                for other in ("strength", "dexterity", "focus"):
                    expected = sign * 4 if other == attr else 0
                    self.assertEqual(target.power(other), expected,
                                     f"{pid} 应当只改 {attr}")


class StackingTest(TemporaryAttributeTestCase):
    """叠加走 ``AfterPowerAmountChanged``：只加**差额**，且不能与首次路径重叠。"""

    def test_two_cards_add_the_sum_not_more(self):
        """两张 +5：属性 10（不是 15，也不是 5）。"""
        state = self.combat()
        self.play(state, "feeding_frenzy")
        self.play(state, "feeding_frenzy")
        self.assertEqual(state.player.power("feeding_frenzy"), 10)
        self.assertEqual(state.player.power("strength"), 10)

    def test_layer_change_path_syncs_delta_only(self):
        """``add_power`` 叠加（真机 ``ModifyAmount``）只同步这一次的差额。"""
        from sts2_sim import powers
        state = self.combat()
        powers.apply_power_to(state, state.player, "feeding_frenzy", 5, [])
        state.player.add_power("feeding_frenzy", 2)
        self.assertEqual(state.player.power("strength"), 7)
        state.player.add_power("feeding_frenzy", -3)
        self.assertEqual(state.player.power("strength"), 4, "递减也要撤回去")
        self.assertNotEqual(state.player.power("strength"), 0,
                            "撤的是差额，不是整条")

    def test_decrement_to_zero_restores_everything(self):
        state = self.combat()
        from sts2_sim import powers
        powers.apply_power_to(state, state.player, "fade", 3, [])
        self.assertEqual(state.player.power("dexterity"), 3)
        state.player.add_power("fade", -3)
        self.assertEqual(state.player.power("dexterity"), 0)
        self.assertNotIn("fade", state.player.powers)
        self.assertEqual(state.player.power("strength"), 0, "别串到别的属性")


class ExpiryTest(TemporaryAttributeTestCase):
    """``AfterSideTurnEnd``：只有**拥有者所在阵营**的回合结束才撤。"""

    def test_player_owned_expires_on_player_side_end_only(self):
        state = self.combat()
        self.play(state, "anticipate")
        self.assertEqual(state.player.power("dexterity"), 2)
        self.enemy_side_end(state)
        self.assertEqual(state.player.power("dexterity"), 2,
                         "敌人回合结束不该撤玩家身上的临时敏捷")
        self.player_side_end(state)
        self.assertEqual(state.player.power("dexterity"), 0)
        self.assertNotIn("anticipate", state.player.powers)

    def test_enemy_owned_expires_on_enemy_side_end_only(self):
        state = self.combat()
        self.play(state, "enfeebling_touch")
        enemy = state.enemies[0]
        self.assertLess(enemy.power("strength"), 0)
        self.player_side_end(state)
        self.assertLess(enemy.power("strength"), 0,
                        "玩家回合结束不该撤敌人身上的临时力量下降")
        self.enemy_side_end(state)
        self.assertEqual(enemy.power("strength"), 0)
        self.assertNotIn("enfeebling_touch", enemy.powers)

    def test_expiry_restores_the_whole_stacked_amount(self):
        """撤销的是**当前层数** ``base.Amount``：叠到 10 就还 10，不是每次还 5。"""
        state = self.combat()
        self.play(state, "feeding_frenzy")
        self.play(state, "feeding_frenzy")
        self.player_side_end(state)
        self.assertEqual(state.player.power("strength"), 0)

    def test_focus_expires_too(self):
        """三个父类同构，专注那一路也要走到（只测力量会漏掉 Sign/属性抄错）。"""
        state = self.combat()
        self.play(state, "hotfix")
        self.assertEqual(state.player.power("focus"), 2)
        self.player_side_end(state)
        self.assertEqual(state.player.power("focus"), 0)


class OtherEntryPointsTest(TemporaryAttributeTestCase):
    """药水/遗物走的是同一条父类行为，不能只有卡牌那条路生效。"""

    def test_potions_apply_and_expire(self):
        from sts2_sim import core
        cases = (("flex_potion", "strength", 5),
                 ("speed_potion", "dexterity", 5),
                 ("shackling_potion", "strength", -7))
        for pid, attr, amount in cases:
            with self.subTest(potion=pid):
                state = self.combat()
                core.use_potion(state, pid)
                holder = (state.enemies[0] if pid == "shackling_potion"
                          else state.player)
                self.assertEqual(holder.power(attr), amount)
                self.assertEqual(holder.power(pid), abs(amount))
                if holder is state.player:
                    self.player_side_end(state)
                else:
                    self.enemy_side_end(state)
                self.assertEqual(holder.power(attr), 0)

    def test_relic_powers_are_registered_even_though_the_relics_are_gapped(self):
        """``helical_dart`` / ``reptile_trinket`` 的能力已登记。

        两件遗物本身仍缺钩子（``AfterCardPlayed`` 条件抽不出 / ``AfterPotionUsed``
        引擎没有），所以它们的**卡（遗物）**还没放行；这里只钉"能力这一层已完成"，
        免得以后接上遗物钩子时发现能力又没了。
        """
        from sts2_sim import powers
        for pid in ("helical_dart", "reptile_trinket"):
            with self.subTest(pid=pid):
                self.assertIn(pid, powers.RULES)
                self.assertEqual(powers._TEMPORARY_ATTRIBUTE_POWERS[pid][0],
                                 "dexterity" if pid == "helical_dart" else "strength")


class AdmissionTest(TemporaryAttributeTestCase):
    """门禁：能力做完的卡要放行，没做完的卡要仍然被拒。"""

    def test_cards_using_the_family_are_admitted(self):
        from sts2_sim import eligibility
        for cid in ADMITTED_CARDS:
            with self.subTest(cid=cid):
                admission = eligibility.card_admission(cid)
                self.assertTrue(admission.admitted,
                                f"{cid} 仍在拒：{admission.reasons()}")

    def test_two_family_cards_are_multiplayer_only(self):
        """⚠️ `coordinate` / `fade` 是**多人专用**卡，单人局里不会出现。

        它们在本批（§2.28）曾经被放行 —— 那时准入门禁还没有"多人专用"这一条
        （`docs/12` §2.31.3 测出、§2.32 修好）。现在它们必须被这条理由拒掉：
        "能力实现完了"不等于"这张牌该进单人池"。
        """
        from sts2_sim import eligibility
        for cid in MULTIPLAYER_ONLY_CARDS:
            with self.subTest(cid=cid):
                reasons = eligibility.card_admission(cid).reasons()
                self.assertIn("multiplayer_only", reasons,
                              f"{cid} 应当因多人专用被拒：{reasons}")

    def test_remaining_card_still_blocked_but_not_by_the_power(self):
        """``dying_star`` 的**能力**已实现，缺口在别处。

        这张卡的效果只从社区文本解析（源码抽取抽不出），所以仍然不能进池 ——
        但它们**不该**再带 ``unimplemented_power`` 这条理由，否则说明本批的能力
        登记没有真正生效。
        """
        from sts2_sim import eligibility
        for cid in STILL_BLOCKED_FOR_OTHER_REASONS:
            with self.subTest(cid=cid):
                reasons = eligibility.card_admission(cid).reasons()
                self.assertTrue(reasons, f"{cid} 不该被放行")
                self.assertFalse([r for r in reasons
                                  if r.startswith("unimplemented_power")],
                                 f"{cid} 的能力已实现，却仍被能力门禁拒：{reasons}")


class PotionPowerGateTest(TemporaryAttributeTestCase):
    """PowderedDemise.OnUse 的依赖已实现，药水门禁不再允许未实现能力。"""

    def test_usable_potions_never_reference_unimplemented_powers(self):
        from sts2_sim import content, powers
        silent = {}
        for pid, potion in sorted(content.POTIONS.items()):
            if potion.effects_incomplete:
                continue
            for effect in potion.effects:
                if effect.op == "apply_power" and effect.power not in powers.IMPLEMENTED:
                    silent[pid] = effect.power
        self.assertEqual(silent, {},
                         "可用药水引用的未实现能力变了：先补能力或补药水门禁")


def _snake(name: str) -> str:
    """``Anticipate`` → ``anticipate``；``HyperbeamFocusDown`` → ``hyperbeam_focus_down``。"""
    out = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return out


if __name__ == "__main__":
    unittest.main()
