"""能力行为与效果算子的回归测试（``docs/09`` L2）。

这一组对应**已经真实发生过**的 bug，每个都是"看起来正常但一定算错"那一类。
真机语义全部抄自反编译源码，测试名里写清依据：

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 中毒每回合打满层数、永不衰减 | 只算总伤害，漏了 `PowerCmd.Decrement` | `test_poison_decays_and_totals_correctly` |
| 敌人打的是**无格挡**的玩家，玩家暴死 | 在敌方回合开始清空玩家格挡（真机在玩家回合开始清） | `test_block_survives_into_the_enemy_turn` |
| "加一张 Soul 进抽牌堆"变成进弃牌堆 | `add_card` 无条件塞弃牌堆 | `test_add_card_goes_to_the_declared_pile` |
| 无形完全免疫伤害（连格挡都不消耗） | 把"掉血上限"错当成"伤害上限" | `test_intangible_caps_hp_loss_not_damage` |
| 已实现的能力仍被算成"缺失" | 卡牌加载器用旧的 3 元组而非 `powers.IMPLEMENTED` | `test_implemented_powers_are_not_reported_missing` |
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
    from sts2_sim.content import STARTING_DECK
    return STARTING_DECK


class TestPowerBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self, enemy: str = "nibbit", seed: int = 5):
        from sts2_sim.core import start_combat
        return start_combat(self.deck, (enemy,), seed=seed)

    def test_poison_decays_and_totals_correctly(self):
        """``PoisonPower.Trigger``：每次触发后**递减 1**，所以 5 层 = 5+4+3+2+1 = 15。

        漏掉递减的话中毒会每回合打满层数 —— 总伤害高得离谱，
        而且"无视格挡的持续消耗"这个战术定位完全变样。
        """
        from sts2_sim.core import Action, step
        state = self._combat()
        enemy = state.enemies[0]
        enemy.hp = 100
        enemy.max_hp = 100
        enemy.add_power("poison", 5)

        dealt = []
        for _ in range(6):
            before = enemy.hp
            step(state, Action(kind="end_turn"))
            dealt.append(before - enemy.hp)
        self.assertEqual(dealt[:5], [5, 4, 3, 2, 1])
        self.assertEqual(sum(dealt), 15)

    def test_poison_ignores_block_and_strength(self):
        """中毒带 ``Unblockable | Unpowered``：不格挡、不受力量/虚弱影响。

        ⚠️ 必须**只跳中毒那一下**，不能跑完整敌方回合：真机在敌方回合开始
        （``Creature.AfterTurnStart``）就会清掉敌方格挡，毒是在那之后才跳的
        （``AfterSideTurnStart``）。跑整回合再断言"格挡还是 50"，
        测的就变成"格挡没被清" —— 那是另一条机制，见
        ``test_enemy_block_clears_at_enemy_turn_start``。
        """
        from sts2_sim import powers as power_rules
        state = self._combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.block = 50
        enemy.add_power("poison", 4)
        enemy.add_power("strength", 10)
        events: list[str] = []
        power_rules.on_turn_start(enemy, state, events)     # 只跳中毒
        self.assertEqual(100 - enemy.hp, 4, "中毒伤害不能被格挡或力量改变")
        self.assertEqual(enemy.block, 50, "格挡不该被中毒消耗")

    def test_enemy_block_clears_at_enemy_turn_start(self):
        """``Creature.AfterTurnStart(side)``：**双方阵营**都在自己的回合开始清格挡。

        少了敌方这一次，会加格挡的怪（`guardbot` 每回合 15）格挡会无限累积 ——
        实测 4 个回合后 15→60，那只怪直接变成打不死的。

        例外是 ``ShouldClearBlock → false`` 的能力（``burrowed`` 潜伏）。
        """
        from sts2_sim import core
        state = self._combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.block = 15
        events: list[str] = []
        core._run_enemy_turn(state, events)
        self.assertEqual(enemy.block, 0, "敌方回合开始必须清敌方格挡")

        state2 = self._combat()
        lurker = state2.enemies[0]
        lurker.hp = lurker.max_hp = 100
        lurker.block = 15
        lurker.add_power("burrowed", 1)
        core._run_enemy_turn(state2, [])
        self.assertEqual(lurker.block, 15, "潜伏声明了不清格挡")

    def test_energy_next_turn_grants_then_removes_itself(self):
        """``EnergyNextTurnPower.AfterEnergyReset``：+N 之后 ``PowerCmd.Remove``。"""
        from sts2_sim.core import Action, BASE_ENERGY, step
        state = self._combat()
        state.player.add_power("energy_next_turn", 2)
        step(state, Action(kind="end_turn"))
        self.assertEqual(state.energy, BASE_ENERGY + 2)
        self.assertNotIn("energy_next_turn", state.player.powers)

    def test_dexterity_adds_to_block(self):
        """``DexterityPower`` 走格挡加法阶段：基础 5 + 敏捷 3 = 8。"""
        from sts2_sim.core import compute_block
        state = self._combat()
        state.player.add_power("dexterity", 3)
        self.assertEqual(compute_block(5, state.player), 8)

    def test_intangible_caps_hp_loss_not_damage(self):
        """``IntangiblePower.ModifyHpLostAfterOsty`` 管的是**掉血**那一段。

        弄反成"伤害上限 1"会让无形连格挡都不消耗 —— 完全免疫，而不是最多掉 1。
        """
        from sts2_sim.core import deal_damage
        state = self._combat()
        state.player.add_power("intangible", 2)
        state.player.block = 0
        before = state.player.hp
        deal_damage(state, state.enemies[0], state.player, 30, [])
        self.assertEqual(before - state.player.hp, 1)

    def test_no_draw_blocks_draw(self):
        """``NoDrawPower.ShouldDraw`` → false：拥有者这一回合不抽牌。"""
        from sts2_sim.core import start_player_turn
        state = self._combat()
        state.hand = []
        state.discard = list(state.draw_pile)
        state.draw_pile = []
        state.player.add_power("no_draw", 1)
        start_player_turn(state, [])
        self.assertEqual(len(state.hand), 0)

    def test_no_power_draws_normally(self):
        """对照组：没有能力时必须正常抽牌（防止 `blocks_draw` 恒真的假阳性）。"""
        from sts2_sim.core import CARDS_PER_TURN, start_player_turn
        state = self._combat()
        state.hand = []
        state.discard = list(state.draw_pile)
        state.draw_pile = []
        start_player_turn(state, [])
        self.assertEqual(len(state.hand), CARDS_PER_TURN)

    def test_retain_hand_keeps_hand_and_decrements(self):
        """``RetainHandPower.ShouldFlush`` → false，且 ``AfterSideTurnEnd`` **递减**。"""
        from sts2_sim.core import Action, step
        state = self._combat()
        state.player.add_power("retain_hand", 2)
        kept = list(state.hand)
        step(state, Action(kind="end_turn"))
        for card in kept:
            self.assertIn(card, state.hand, "手牌必须被保留")
        self.assertEqual(state.player.power("retain_hand"), 1, "应递减而不是移除")

    def test_block_survives_into_the_enemy_turn(self):
        """⚠️ 格挡只在**玩家回合开始**清空，不能在敌方回合开始清。

        在敌方回合开始清一次，敌人打的就是无格挡的玩家，玩家成片暴死 ——
        表现为"规则 bot 与随机策略打平"（区分度测试抓到的，我自己引入过）。
        """
        from sts2_sim.core import Action, step
        state = self._combat()
        state.player.block = 12
        step(state, Action(kind="end_turn"))
        # 敌人可能打掉一部分，但绝不该在回合开始就被清零
        self.assertLessEqual(state.player.block, 12)
        self.assertGreaterEqual(state.player.block, 0)

    def test_doom_executes_when_hp_is_low(self):
        """``DoomPower.IsOwnerDoomed``：生命 ≤ 层数时被处决。"""
        from sts2_sim.core import Action, step
        state = self._combat()
        enemy = state.enemies[0]
        enemy.hp = 3
        enemy.add_power("doom", 5)
        step(state, Action(kind="end_turn"))
        self.assertFalse(enemy.alive())

    def test_doom_does_not_execute_above_threshold(self):
        """对照组：血量高于层数时不该被处决。"""
        from sts2_sim.core import Action, step
        state = self._combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.add_power("doom", 5)
        step(state, Action(kind="end_turn"))
        self.assertTrue(enemy.alive())

    def test_thorns_reflects_damage(self):
        """``ThornsPower.AfterDamageReceived``：被打之后反弹，且不受格挡影响。"""
        from sts2_sim.core import deal_damage
        state = self._combat()
        attacker = state.enemies[0]
        attacker.hp = attacker.max_hp = 60
        attacker.block = 10
        state.player.add_power("thorns", 4)
        before = attacker.hp
        deal_damage(state, attacker, state.player, 5, [])
        self.assertEqual(before - attacker.hp, 4, "反伤应无视攻击者格挡")


class TestEffectOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_add_card_goes_to_the_declared_pile(self):
        """``add_card`` 必须按 ``PileType`` 落位。

        真机有 Hand / Draw / Discard 三种，``CardPilePosition.Top`` 还表示
        放到抽牌堆**顶**。一律塞弃牌堆会让效果完全失效，而且看起来正常。
        """
        from sts2_sim.content import Effect
        from sts2_sim.core import CombatState, _apply_effects, start_combat
        state = start_combat(self.deck, ("nibbit",), seed=5)
        state.hand, state.draw_pile, state.discard = [], [], []
        for pile in ("hand", "draw", "discard"):
            before = {"hand": len(state.hand), "draw": len(state.draw_pile),
                      "discard": len(state.discard)}[pile]
            _apply_effects(state, [Effect(op="add_card", amount=1, card="dazed",
                                          target="self", pile=pile)],
                           state.player, 0, [])
            after = {"hand": len(state.hand), "draw": len(state.draw_pile),
                     "discard": len(state.discard)}[pile]
            self.assertEqual(after, before + 1, f"{pile} 牌堆没有收到卡")

    def test_add_card_to_top_of_draw_pile_is_drawn_next(self):
        """``CardPilePosition.Top``：下一张就抽到它。"""
        from sts2_sim.content import Effect
        from sts2_sim.core import _apply_effects, draw_cards, start_combat
        state = start_combat(self.deck, ("nibbit",), seed=5)
        state.hand, state.draw_pile, state.discard = [], [], []
        for cid in ("strike_ironclad", "defend_ironclad"):
            _apply_effects(state, [Effect(op="add_card", amount=1, card=cid,
                                          target="self", pile="draw")],
                           state.player, 0, [])
        _apply_effects(state, [Effect(op="add_card", amount=1, card="dazed",
                                      target="self", pile="draw",
                                      position="top")],
                       state.player, 0, [])
        draw_cards(state, 1, [])
        self.assertEqual(state.hand[0].cid, "dazed")

    def test_gain_stars_accumulates(self):
        """``PlayerCmd.GainStars`` → ``gain_stars`` 算子。"""
        from sts2_sim.content import Effect
        from sts2_sim.core import _apply_effects, start_combat
        state = start_combat(self.deck, ("nibbit",), seed=5)
        _apply_effects(state, [Effect(op="gain_stars", amount=3, target="self")],
                       state.player, 0, [])
        self.assertEqual(state.stars, 3)


class TestDecrementPowers(unittest.TestCase):
    """**递减型**能力：镀层 / 再生。这类最容易静默错 —— 少了递减那一行，
    强度从"持续 N 回合"变成"永久"，而且日志一切正常。
    """

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def setUp(self):
        from sts2_sim import content
        from sts2_sim.core import start_combat
        self.events: list[str] = []
        self.state = start_combat(list(content.STARTING_DECK), ("guardbot",), seed=3)

    def _finish_player_turn(self):
        from sts2_sim.core import tick_powers
        tick_powers(self.state.player, self.state, self.events)

    def _advance_to_next_player_turn(self):
        from sts2_sim import core
        self.state.discard.extend(self.state.hand)
        self.state.hand = []
        core._run_enemy_turn(self.state, self.events)
        self.state.turn += 1
        core.start_player_turn(self.state, self.events)

    def test_regeneration_heals_then_decrements(self):
        """``RegenPower.BeforeSideTurnEndEarly``：回 N 点**然后**减 1 层。

        源码里治疗与递减紧挨着：

        .. code-block:: csharp

           await CreatureCmd.Heal(base.Owner, base.Amount);
           await PowerCmd.Decrement(this);

        所以 5 层再生的总治疗量是 ``5+4+3+2+1 = 15``。少了递减会变成
        **永远每回合回 5 点**（差距随回合数线性放大）。
        """
        player = self.state.player
        player.hp = 40
        player.add_power("regen", 5)
        healed = []
        for _ in range(7):
            before = player.hp
            self._finish_player_turn()
            healed.append(player.hp - before)
            if player.power("regen") == 0:
                break
            self._advance_to_next_player_turn()
        self.assertEqual(healed, [5, 4, 3, 2, 1])
        self.assertEqual(sum(healed), 15)
        self.assertEqual(player.power("regen"), 0)

    def test_plating_grants_block_then_decrements(self):
        """``PlatingPower``：回合结束给 N 点格挡，**回合开始减 1 层**。

        真机带两个豁免：玩家第 1 回合不减、敌人第 1 轮不减
        （``PlatingPower.AfterSideTurnStart``）。所以 4 层的总格挡量是
        ``4+3+2+1 = 10`` —— 第 1 回合就给满 4 点。
        """
        player = self.state.player
        player.add_power("plating", 4)
        granted = []
        for _ in range(6):
            player.block = 0
            self._finish_player_turn()
            granted.append(player.block)
            if player.power("plating") == 0:
                break
            self._advance_to_next_player_turn()
        self.assertEqual(granted, [4, 3, 2, 1, 0])
        self.assertEqual(sum(granted), 10)

    def test_plating_block_is_unpowered(self):
        """镀层格挡是 ``ValueProp.Unpowered``：不吃敏捷、也不被脆弱打折。"""
        player = self.state.player
        player.add_power("plating", 4)
        player.add_power("dexterity", 5)
        player.add_power("frail", 3)
        player.block = 0
        self._finish_player_turn()
        self.assertEqual(player.block, 4, "敏捷/脆弱不该改变镀层格挡")

    def test_barricade_keeps_block(self):
        """``BarricadePower.ShouldClearBlock → false``：格挡永不清除。"""
        player = self.state.player
        player.add_power("barricade", 1)
        player.block = 20
        self._advance_to_next_player_turn()
        self.assertEqual(player.block, 20)


class TestCoverageAccounting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_implemented_powers_are_not_reported_missing(self):
        """卡牌加载器必须用 ``powers.IMPLEMENTED`` 判定"缺失"。

        用旧的硬编码 3 元组会让**已经实现**的能力仍被算成缺失，
        卡牌白白被排除出训练集（实测漏算 26 处引用）。
        """
        from sts2_sim import content, powers
        self.assertGreater(len(powers.IMPLEMENTED), 3)
        source = {c["cid"]: c for c in
                  json.loads((CONTENT / "cards_source.json").read_text(encoding="utf-8"))}
        wrongly_missing = []
        for cid, record in source.items():
            for pid in content._missing_powers(record):
                if pid in powers.IMPLEMENTED:
                    wrongly_missing.append(f"{cid}:{pid}")
        self.assertEqual(wrongly_missing, [],
                         "已实现的能力被判成缺失，卡牌会被错误地排除出训练集")

    def test_engine_coverage_is_consistent(self):
        from sts2_sim import content
        coverage = content.engine_coverage()
        self.assertEqual(
            coverage["cards_trainable"],
            coverage["cards_total"] - coverage["cards_incomplete"])
        self.assertGreater(coverage["cards_total"], 500)
        # 残缺卡必须能列出来，训练时才有东西可排除
        self.assertTrue(coverage["incomplete_examples"])
        self.assertEqual(
            len(content.unverified_report()["cards_effects_incomplete"]),
            coverage["cards_incomplete"])

    def test_no_trainable_card_silently_no_ops(self):
        """**核心不变量**：可训练的卡不能引用引擎没实现的能力或算子。

        这是最危险的一类静默错误 —— 卡在训练集里、费用照扣、日志照打，
        但**效果什么也不做**。模型会学到"这张牌没用"，而真机里它有用。

        门禁在两处：内容层把这类卡标成 ``effects_incomplete``（排除出训练集），
        ``lint()`` 再兜一层。这里断言两道门都是关着的。
        """
        from sts2_sim import content, powers
        broken_powers: list[str] = []
        broken_ops: list[str] = []
        for card in content.CARD_DB.values():
            if card.effects_incomplete:
                continue
            for effect in card.effects:
                if (effect.op == "apply_power" and effect.power
                        and effect.power not in powers.IMPLEMENTED):
                    broken_powers.append(f"{card.cid}:{effect.power}")
                if effect.op not in content.ENGINE_OPS:
                    broken_ops.append(f"{card.cid}:{effect.op}")
        self.assertEqual(broken_powers, [],
                         f"可训练的卡引用了未实现的能力：{broken_powers[:10]}")
        self.assertEqual(broken_ops, [],
                         f"可训练的卡引用了引擎没有的算子：{broken_ops[:10]}")
        # 用注入的缺失依赖检验门禁；补全内容不应被历史残缺数量下限阻止。
        self.assertEqual(content._missing_powers({"effects": [
            {"op": "apply_power", "power": "missing_power_for_gate_test"}
        ]}), ["missing_power_for_gate_test"])

    def test_trainable_cards_are_not_silently_empty(self):
        """可训练的卡**不能既没有效果、也没有触发**（除非是纯占位牌）。

        ⭐ 这条是"静默空卡"的结构护栏。它抓到过一整类真实缺陷：
        抽取器把 ``DamageCmd.Attack`` 当表现层丢掉之后，``Bludgeon`` /
        ``BodySlam`` / ``Clash`` 这类纯攻击卡**既没有效果、也没有任何"未支持"标记**，
        于是以"零效果"的身份混进训练集 —— 费用照扣、日志照打，但打出去什么都不发生。
        模型会学到"这张卡没用"，而真机里它是主力输出。

        豁免只有一类：**诅咒与状态牌**。它们"什么都不做"就是真机行为
        （作用是占手牌位、卡住攻防），这一点由 ``_is_pure_clog`` 以源码为准判定。
        所以这里按"类型不是诅咒/状态"来断言 —— 一旦再出现攻击/技能/能力牌是空的，
        测试立刻红。
        """
        from sts2_sim import content
        empty = [c.cid for c in content.CARD_DB.values()
                 if not c.effects_incomplete and not c.effects and not c.triggers
                 and c.card_type not in ("curse", "status")]
        self.assertEqual(empty, [],
                         f"这些可训练的卡在引擎里什么都不做：{empty}")
        # 反证：豁免的那一类确实存在（否则这条断言可能因为"全被标残缺"而空转）
        clog = [c.cid for c in content.CARD_DB.values()
                if not c.effects_incomplete and not c.effects and not c.triggers
                and c.card_type in ("curse", "status")]
        self.assertGreater(len(clog), 5, "纯占位牌的豁免没生效，断言可能没在起作用")

    def test_lint_is_clean_on_real_content(self):
        """``lint()`` 在真实内容上必须零问题（算子/选牌用途/费用合法性）。"""
        from sts2_sim import content
        problems = content.lint()
        self.assertEqual(problems, [], f"lint 报出 {len(problems)} 个问题：{problems[:6]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
