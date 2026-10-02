"""**每回合施加族**与两个 **duration 型减益**（`docs/12` §2.29）的源码级回归测试。

本文件覆盖 5 个新实现的能力，每一条都对应源码里一个**容易被漏掉的条件**：

| 能力 | 源码 | 容易漏掉的那一条 | 漏掉的后果 |
|---|---|---|---|
| ``biased_cognition`` | ``BiasedCognitionPower.AfterSideTurnStart`` | **永不撤销**（不是临时属性） | 当成临时专注 → 每回合白赚回来 |
| ``wraith_form`` | ``WraithFormPower.AfterSideTurnStart`` | 同上，扣的是**敏捷** | 同上 |
| ``neurosurge`` | ``NeurosurgePower.AfterSideTurnStart`` | 走 ``Apply`` 路径（正向、可被神器抵消） | 用 ``add_power`` 会漏掉 ``AfterApplied`` |
| ``debilitate`` | ``DebilitatePower`` 的两个**公开方法** | 易伤抬、虚弱**压**，方向相反且**分别**看被打者/攻击者 | 只做一半，或两边都抬 |
| ``no_block`` | ``NoBlockPower.ModifyBlockMultiplicative`` | 只对**卡牌来源**生效（药水 / 能力给的格挡不归零） | 药水格挡也变成 0（引擎比真机**弱**） |

``debilitate`` / ``no_block`` 的到期时机在 ``DECREMENTS_AT_ENEMY_SIDE_TURN_END``
（与易伤/虚弱/脆弱同一张表），因此它们没有 ``AfterSideTurnEnd`` 钩子 ——
这条已记在 ``tests/test_power_hook_coverage.py`` 的 ``EQUIVALENT`` 里。

这些断言只说明"引擎按源码执行"，不代表已与真机对拍（``docs/09`` §5）。
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"
DECOMPILED = ROOT / "data" / "decompiled" / "sts2"

#: 源码扫描得到的同族全集（``AfterSideTurnStart``/``AfterSideTurnEnd`` 里
#: ``Apply<Xxx>(..., base.Owner, ±base.Amount, ...)`` 的那些）。
PERIODIC_FAMILY = (
    "biased_cognition", "demon_form", "high_voltage", "neurosurge", "prep_time",
    "ritual", "shadow_step", "territorial", "wraith_form",
)

#: 本批新实现的 5 个能力引用的卡（能力做完 → 卡应放行）。
BATCH_CARDS = ("biased_cognition", "wraith_form", "neurosurge", "panic_button",
               "debilitate")


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


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


_METHOD = re.compile(r"(?:public|protected|internal|private)\s+override\s+"
                     r"[\w<>?,\[\]\. ]+\s+(\w+)\s*\([^)]*\)\s*\{")


def _method_bodies(text: str) -> dict[str, list[str]]:
    """按**花括号配对**切出每个覆写方法的函数体。

    不能只看"文件里出现过某个模式"：``PaleBlueDotPower`` 的
    ``Apply<DrawCardsNextTurnPower>(..., base.Owner, base.Amount, ...)`` 在
    ``AfterCardPlayed`` 里，与"每回合开始施加"完全是两回事 ——
    按文件匹配会把它错算进这个族。
    """
    out: dict[str, list[str]] = {}
    for match in _METHOD.finditer(text):
        start = match.end() - 1
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    break
        out.setdefault(match.group(1), []).append(text[start:index + 1])
    return out


class _Case(unittest.TestCase):
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
        state.energy = 9
        state.player.powers.clear()
        state.player.block = 0
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


class PeriodicAttributeFamilyTest(_Case):
    """每回合按层数再施加一次的族（与临时属性族**不是**一回事）。"""

    def test_source_family_is_fully_registered(self):
        """源码里同族的每一条都必须在 ``RULES`` 里。

        判据：**某个覆写的 ``AfterSideTurnStart``/``AfterSideTurnEnd`` 方法体里**
        出现 ``Apply<Xxx>(..., base.Owner, ±base.Amount, ...)``。
        少了任何一条，那张卡会被门禁拒（看得见）；但如果它还有别的拒绝理由，
        这条缺口就静默了 —— 与临时属性族同一个教训。
        """
        if not DECOMPILED.exists():
            self.skipTest(f"缺反编译源码 {DECOMPILED}")
        from sts2_sim import powers
        found = set()
        pattern = re.compile(r"Apply<\w+>\([^;]*?base\.Owner,\s*-?base\.Amount", re.S)
        for path in DECOMPILED.rglob("*Power.cs"):
            bodies = _method_bodies(path.read_text(encoding="utf-8", errors="replace"))
            for hook in ("AfterSideTurnStart", "AfterSideTurnEnd"):
                if any(pattern.search(body) for body in bodies.get(hook, [])):
                    found.add(_snake(re.sub(r"Power$", "", path.stem)))
        self.assertEqual(sorted(found), sorted(PERIODIC_FAMILY))
        for pid in PERIODIC_FAMILY:
            with self.subTest(pid=pid):
                self.assertIn(pid, powers.RULES)

    def test_biased_cognition_loses_focus_every_turn(self):
        """打出 +5 专注，之后**每回合** −1（层数 1 → 一次扣 1）。"""
        from sts2_sim import core
        state = self.combat()
        self.play(state, "biased_cognition")
        self.assertEqual(state.player.power("focus"), 5)
        self.assertEqual(state.player.power("biased_cognition"), 1)
        core.start_player_turn(state, [])
        self.assertEqual(state.player.power("focus"), 4)
        core.start_player_turn(state, [])
        self.assertEqual(state.player.power("focus"), 3, "第 3 回合还在扣（永不撤销）")
        self.assertEqual(state.player.power("biased_cognition"), 1, "能力层数不变")

    def test_wraith_form_loses_dexterity_every_turn(self):
        from sts2_sim import core
        state = self.combat()
        self.play(state, "wraith_form")
        self.assertEqual(state.player.power("intangible"), 2, "卡的正面效果照常")
        self.assertEqual(state.player.power("dexterity"), 0)
        core.start_player_turn(state, [])
        self.assertEqual(state.player.power("dexterity"), -1)
        core.start_player_turn(state, [])
        self.assertEqual(state.player.power("dexterity"), -2)

    def test_neurosurge_gains_doom_every_turn(self):
        """0 费：+3 能量、抽 2、给自己 3 层能力；回合开始 +3 层末日。"""
        from sts2_sim import core
        state = self.combat()
        self.play(state, "neurosurge")
        self.assertEqual(state.player.power("doom"), 0, "卡本身不上末日")
        self.assertEqual(state.player.power("neurosurge"), 3)
        core.start_player_turn(state, [])
        self.assertEqual(state.player.power("doom"), 3)
        core.start_player_turn(state, [])
        self.assertEqual(state.player.power("doom"), 6, "每回合累加，不是只加一次")

    def test_stacking_increases_the_per_turn_amount(self):
        """两张偏执认知 = 每回合扣 2（真机 ``-base.Amount`` 读的是**当前层数**）。"""
        from sts2_sim import core
        state = self.combat()
        self.play(state, "biased_cognition")
        self.play(state, "biased_cognition")
        self.assertEqual(state.player.power("biased_cognition"), 2)
        before = state.player.power("focus")
        core.start_player_turn(state, [])
        self.assertEqual(state.player.power("focus"), before - 2)


class DebilitateTest(_Case):
    """``DebilitatePower``：抬高易伤、压低虚弱，随后按回合递减。"""

    def test_vulnerable_multiplier_is_raised(self):
        from sts2_sim import powers
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("vulnerable", 3)
        self.assertEqual(powers.damage_multipliers(state.player, enemy, None, state),
                         [("vulnerable", "1.5")])
        enemy.add_power("debilitate", 1)
        self.assertEqual(powers.damage_multipliers(state.player, enemy, None, state),
                         [("vulnerable", "2.0")],
                         "源码是 amount + (amount − 1)：1.5 → 2.0")

    def test_weak_multiplier_is_lowered(self):
        from sts2_sim import powers
        state = self.combat()
        player = state.player
        player.add_power("weak", 3)
        self.assertEqual(powers.damage_multipliers(player, state.enemies[0], None, state),
                         [("weak", "0.75")])
        player.add_power("debilitate", 1)
        self.assertEqual(powers.damage_multipliers(player, state.enemies[0], None, state),
                         [("weak", "0.50")],
                         "源码是 amount − (1 − amount)：0.75 → 0.5")

    def test_directions_do_not_leak_to_the_other_side_of_the_pair(self):
        """打成"两边都抬/都压"最容易发生：两个方向各用一个干净局面钉。"""
        from sts2_sim import powers
        # ① 攻击者身上有 debilitate、被打者身上有易伤 → 易伤**不**被抬高
        state = self.combat()
        player, enemy = state.player, state.enemies[0]
        player.add_power("debilitate", 1)
        enemy.add_power("vulnerable", 3)
        self.assertEqual(powers.damage_multipliers(player, enemy, None, state),
                         [("vulnerable", "1.5")],
                         "易伤看的是**被打者**身上的 debilitate")
        # ② 被打者身上有 debilitate、攻击者身上有虚弱 → 虚弱**不**被压低
        state = self.combat()
        player, enemy = state.player, state.enemies[0]
        enemy.add_power("debilitate", 1)
        player.add_power("weak", 3)
        self.assertEqual(powers.damage_multipliers(player, enemy, None, state),
                         [("weak", "0.75")],
                         "虚弱看的是**攻击者**身上的 debilitate")

    def test_end_to_end_damage(self):
        """走到 ``compute_damage`` 的组合：易伤被抬高、虚弱被压低。"""
        from sts2_sim import core
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("vulnerable", 3)
        self.assertEqual(core.compute_damage(10, state.player, enemy, None, state), 15)
        enemy.add_power("debilitate", 1)
        self.assertEqual(core.compute_damage(10, state.player, enemy, None, state), 20,
                         "被打者身上的 debilitate 把易伤抬到 2.0")
        state.player.add_power("weak", 3)
        self.assertEqual(core.compute_damage(10, state.player, enemy, None, state), 15,
                         "10 × 2.0（易伤） × 0.75（虚弱）；攻击者没有 debilitate，"
                         "所以虚弱仍是 0.75")
        state.player.add_power("debilitate", 1)
        self.assertEqual(core.compute_damage(10, state.player, enemy, None, state), 10,
                         "攻击者也有了 debilitate → 虚弱被压到 0.5：10 × 2.0 × 0.5")

    def test_card_deals_damage_then_applies_the_power(self):
        from sts2_sim import core
        state = self.combat()
        enemy = state.enemies[0]
        self.play(state, "debilitate")
        self.assertEqual(100 - enemy.hp, 10, "先结算伤害（此时还没有 debilitate）")
        self.assertEqual(enemy.power("debilitate"), 2)
        core._end_enemy_side_turn(state, [])
        self.assertEqual(enemy.power("debilitate"), 1)
        core._end_enemy_side_turn(state, [])
        self.assertNotIn("debilitate", enemy.powers)

    def test_decays_at_enemy_side_end_not_player_side(self):
        state = self.combat()
        enemy = state.enemies[0]
        enemy.add_power("debilitate", 2)
        from sts2_sim import core
        core.tick_powers(state.player, state, [])
        self.assertEqual(enemy.power("debilitate"), 2,
                         "玩家阵营回合结束不该递减（源码条件不是 `side == Player`）")
        core._end_enemy_side_turn(state, [])
        self.assertEqual(enemy.power("debilitate"), 1)


class NoBlockTest(_Case):
    """``NoBlockPower``：卡牌来源的格挡 ×0，随后按回合递减。"""

    def test_card_block_is_zeroed_but_unpowered_block_is_not(self):
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("no_block", 2)
        self.play(state, "defend_ironclad")
        self.assertEqual(state.player.block, 0, "卡牌来源的格挡归零")
        core.gain_block(state, state.player, 7, [], unpowered=True, label="药水")
        self.assertEqual(state.player.block, 7,
                         "Unpowered 来源（药水/能力）**不**归零（源码 `props.HasFlag(Unpowered)`）")
        self.play(state, "defend_ironclad")
        self.assertEqual(state.player.block, 7, "再打一张仍然不涨")

    def test_block_works_again_after_the_power_is_gone(self):
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("no_block", 1)
        self.play(state, "defend_ironclad")
        self.assertEqual(state.player.block, 0)
        core._end_enemy_side_turn(state, [])
        self.assertNotIn("no_block", state.player.powers)
        self.play(state, "defend_ironclad")
        self.assertEqual(state.player.block, 5)

    def test_decays_at_enemy_side_end_not_player_side(self):
        from sts2_sim import core
        state = self.combat()
        state.player.add_power("no_block", 2)
        core.tick_powers(state.player, state, [])
        self.assertEqual(state.player.power("no_block"), 2)
        core._end_enemy_side_turn(state, [])
        self.assertEqual(state.player.power("no_block"), 1)

    def test_card_gives_its_own_block_before_the_power_lands(self):
        """``PanicButton`` 的效果顺序：先 30 点格挡，**再**上 NoBlock（所以那 30 点留着）。"""
        state = self.combat()
        self.play(state, "panic_button")
        self.assertEqual(state.player.block, 30)
        self.assertEqual(state.player.power("no_block"), 2)
        self.play(state, "defend_ironclad")
        self.assertEqual(state.player.block, 30, "之后拿的格挡才是 0")


class AdmissionTest(_Case):
    def test_the_five_cards_are_admitted(self):
        from sts2_sim import eligibility
        for cid in BATCH_CARDS:
            with self.subTest(cid=cid):
                admission = eligibility.card_admission(cid)
                self.assertTrue(admission.admitted, f"{cid} 仍在拒：{admission.reasons()}")


if __name__ == "__main__":
    unittest.main()
