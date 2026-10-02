"""造牌算子（``generate_card``）与"本回合免费"的回归测试。

对应整改批次见 ``docs/12`` §2.14。这一批补的是**两座桥**加**四条静默错误门禁**：

| 症状 | 根因 | 对应测试 |
|---|---|---|
| `white_noise` / `infernal_blade` / `bundle_of_joy` 在引擎里"打出去什么都不发生" | 抽取器不认 ``CardFactory.GetDistinctForCombat``，而引擎早就有对应实现（``powers.generate_cards_to_hand``） | `test_white_noise_generates_a_power_card`、`test_generated_pool_respects_admission_gate` |
| 生成的牌本回合要花钱（真机是 0 费） | ``SetToFreeThisTurn()`` 被当成"未实现"整条丢掉 | `test_generated_card_is_free_this_turn` |
| **X 费卡被"免费"变成白嫖** | 源码 ``SetThisTurnOrUntilPlayed`` 对 ``Canonical < 0``（X 费）**直接跳过**那道门 | `test_free_does_not_apply_to_x_cost_cards` |
| 免费跨回合留着 | ``EndOfTurnCleanup`` 没实现 | `test_free_clears_at_turn_end` |
| `fight_through` 少加一张 Wound、`bouncing_flask` 只毒 1 次 | ``count_loop_spans`` 用圆括号配平取法取**花括号**循环体，体内第一个 ``)`` 就截断 | `test_loop_body_takes_two_wounds` |
| 球数/条件决定次数的循环被当成"跑一次" | 引擎没有"按集合长度重复"的算子 | `test_runtime_loop_is_reported_as_gap` |
| **基础版白拿升级版的效果**（`spinner` 白得一个玻璃球、`true_grit` 基础版本应随机消耗） | 引擎只建模**数值**升级，`if (base.IsUpgraded)` 里的结构性效果被无条件抽出 | `test_upgraded_only_branch_is_reported_as_gap` |
| 药水一用就 ``ValueError: 未知算子`` | ``_load_potions`` 不校验算子白名单（卡牌才走 ``_source_reject_reason``） | `test_no_usable_potion_carries_an_unimplemented_op` |

⚠️ 这些测试证明的是"引擎按声明的规则执行、残缺内容进不了动作空间"，
**不是**数值与真机一致（真机对拍为 0，见 ``docs/13`` 头部）。
"""

from __future__ import annotations

import collections
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"
DECOMPILED = ROOT / "data" / "decompiled" / "sts2" / "MegaCrit.Sts2.Core.Models.Cards"


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


class TestGenerateCardRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self, seed: int = 3):
        from sts2_sim.core import start_combat
        state = start_combat(self.deck, ("nibbit",), seed=seed)
        # 只手牌与能量可控：抽牌堆留空，免得"生成"与"抽牌"混在一起数不清。
        state.draw_pile = []
        state.discard = []
        state.energy = 9
        return state

    def test_white_noise_generates_a_power_card(self):
        """``WhiteNoise``：生成一张**能力**牌进手牌，且**本回合 0 费**。

        真机 ``WhiteNoise.cs:23-29``：``GetDistinctForCombat(角色池 where
        c.Type == CardType.Power, 1, …)`` → ``SetToFreeThisTurn()`` → 进手牌。
        抽取器认不出来时这张牌在引擎里是**空卡**（打出去什么都不发生）。
        """
        from sts2_sim.content import CARD_DB
        from sts2_sim.core import Action, CardInstance, play_cost, step
        state = self._combat()
        state.hand = [CardInstance("white_noise")]
        step(state, Action("play_card", 0, -1))
        self.assertEqual(len(state.hand), 1, "应当恰好生成 1 张牌")
        generated = state.hand[0]
        self.assertEqual(CARD_DB[generated.cid].card_type, "power",
                         f"生成的不是能力牌：{generated.cid}")
        self.assertTrue(generated.free_this_turn,
                        "真机对生成的牌调了 SetToFreeThisTurn()")
        self.assertEqual(play_cost(state, generated), 0,
                         "本回合免费必须体现在**实际费用**上（观察层读的也是它）")

    def test_infernal_blade_generates_an_attack_card(self):
        """``InfernalBlade`` 只给**攻击**牌 —— 过滤条件丢掉就变成"任意牌"。"""
        from sts2_sim.content import CARD_DB
        from sts2_sim.core import Action, CardInstance, step
        state = self._combat()
        state.hand = [CardInstance("infernal_blade")]
        step(state, Action("play_card", 0, -1))
        self.assertEqual(len(state.hand), 1)
        self.assertEqual(CARD_DB[state.hand[0].cid].card_type, "attack")

    def test_generated_pool_respects_admission_gate(self):
        """生成池必须过**准入门禁**（``docs/13`` §7）：不许凭空发一张引擎算不对的牌。"""
        from sts2_sim import eligibility
        from sts2_sim.powers import generate_cards_to_hand
        state = self._combat()
        admitted = eligibility.admitted_cards()
        for _ in range(20):
            state.hand = []
            for cid in generate_cards_to_hand(state, 3, [], pool="IroncladCardPool"):
                self.assertIn(cid, admitted)
                self.assertIn(cid, {c.cid for c in state.hand})

    def test_generate_card_effects_need_a_known_pool(self):
        """池 / 牌型认不出来时必须**整张卡标残缺**（当成"任意牌"是静默变强）。"""
        from sts2_sim.content import _source_reject_reason
        base = {"cost": 1, "target": "self", "effects": [
            {"op": "generate_card", "amount": 1, "target": "self",
             "filter": [["pool", "unknown_pool"]], "pile": "hand"}]}
        self.assertIsNotNone(_source_reject_reason(base))
        ok = {"cost": 1, "target": "self", "effects": [
            {"op": "generate_card", "amount": 1, "target": "self",
             "filter": [["pool", "character"], ["card_type", "attack"]],
             "pile": "hand"}]}
        self.assertIsNone(_source_reject_reason(ok))


    def test_cunning_potion_gives_upgraded_shivs(self):
        """``cunning_potion``：3 张**已升级**的 Shiv 进手牌。

        真机 ``CunningPotion.cs`` 逐张 ``CardCmd.Upgrade(item)``，**没有**
        ``IsUpgraded`` 条件 —— 所以基础版药水给出的就该是升级版 Shiv。
        引擎的 ``add_card`` 原来造的是基础形态（静默变弱一档）。
        """
        from sts2_sim.core import start_combat, use_potion
        state = start_combat(self.deck, ("nibbit",), seed=7)
        state.hand = []
        use_potion(state, "cunning_potion")
        shivs = [card for card in state.hand if card.cid == "shiv"]
        self.assertEqual(len(shivs), 3, f"应当恰好 3 张 Shiv：{state.hand}")
        self.assertTrue(all(card.upgraded for card in shivs),
                        "真机对每张 Shiv 都调了 CardCmd.Upgrade")

    def test_potion_of_capacity_uses_its_declared_variable(self):
        """``potion_of_capacity``：加球槽的量来自源码声明的 ``RepeatVar``。

        旧快照把它抽成常量 1（与真机声明的变量脱钩）；现在量跟着变量走。
        这里只钉住"量是正数且与源码声明的 Repeat 一致"。
        """
        import json
        from sts2_sim import content
        record = {r["pid"]: r for r in json.loads(
            (CONTENT / "potions_source.json").read_text(encoding="utf-8"))}[
                "potion_of_capacity"]
        declared = {v["name"]: v["base"] for v in record.get("vars") or []}
        slots = [e for e in content.POTIONS["potion_of_capacity"].effects
                 if e.op == "add_orb_slots"]
        self.assertTrue(slots, content.POTIONS["potion_of_capacity"].effects)
        self.assertEqual(slots[0].amount, declared.get("Repeat"),
                         "加槽数必须等于源码声明的 Repeat 变量值")


class TestFreeThisTurn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self, seed: int = 5):
        from sts2_sim.core import start_combat
        state = start_combat(self.deck, ("nibbit",), seed=seed)
        state.draw_pile = []
        state.discard = []
        return state

    def test_free_does_not_apply_to_x_cost_cards(self):
        """⚠️ 源码 ``CardEnergyCost.SetThisTurnOrUntilPlayed`` 有
        ``if (cost != 0 || Canonical >= 0)`` 这道门（``CardEnergyCost.cs:197``）——
        **X 费卡的 Canonical 是 -1，这条修饰被直接丢弃**。

        不照抄那道门，X 费卡会变成"费用 0 却按当前能量结算 X"：
        白嫖 X 张牌 / X 次伤害，而且日志完全正常。
        """
        from sts2_sim.core import CardInstance, play_cost
        state = self._combat()
        state.energy = 6
        card = CardInstance("whirlwind")
        card.free_this_turn = True
        self.assertTrue(card.definition().is_x_cost, "whirlwind 应当是 X 费卡")
        self.assertEqual(play_cost(state, card), 6,
                         "X 费卡不受 SetToFreeThisTurn 影响（照抄源码那道门）")

    def test_free_clears_at_turn_end(self):
        """``PlayerCombatState.EndOfTurnCleanup``（``CombatManager.cs:1813``）：
        回合结束时把"本回合有效"的临时费用修饰清掉 —— 不清就是**免费跨回合**。"""
        from sts2_sim.core import Action, step
        state = self._combat()
        card = state.hand[0]
        card.free_this_turn = True
        step(state, Action("end_turn"))
        self.assertFalse(card.free_this_turn,
                         "回合结束后这张牌的免费修饰必须已经清掉")

    def test_free_applies_the_first_turn_only_for_a_generated_card(self):
        """生成的免费牌在**下一个玩家回合**要恢复原价。"""
        from sts2_sim.core import Action, CardInstance, play_cost, step
        state = self._combat()
        state.hand = [CardInstance("white_noise")]
        state.energy = 9
        step(state, Action("play_card", 0, -1))
        generated = state.hand[0]
        self.assertEqual(play_cost(state, generated), 0)
        step(state, Action("end_turn"))
        self.assertFalse(generated.free_this_turn)


class TestStackOps(unittest.TestCase):
    """三个"整堆 / 自动打出 / 采购"算子（``docs/12`` §2.16）。

    | 算子 | 真机命令 | 使用者 |
    |---|---|---|
    | `shuffle` | `CardPileCmd.Shuffle` | `BottledPotential`（药水）、`Reboot`（卡） |
    | `autoplay_from_draw` | `CardPileCmd.AutoPlayFromDrawPile` | `DistilledChaos`、`Cascade`、`Havoc`、`IAmInvincible` |
    | `procure_random_potion` | `PotionCmd.TryToProcure` + `CreateRandomPotionOutOfCombat` | `EntropicBrew`（药水）、`DelicateFrond`（遗物） |
    """

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self, seed: int = 11):
        from sts2_sim.core import start_combat
        return start_combat(self.deck, ("nibbit",), seed=seed)

    def test_bottled_potential_moves_hand_into_draw_pile(self):
        """``bottled_potential``：手牌**全部**进抽牌堆 → 洗牌 → 抽回 ``Cards`` 张。

        真机 ``BottledPotential.cs:28-30``。旧快照把前两步判成
        `CardPileCmd.Add(集合，谓词认不出)` + `Shuffle` 两条缺口整瓶拒掉。
        """
        from sts2_sim.core import use_potion
        state = self._combat()
        state.draw_pile = []
        state.discard = []
        before = len(state.hand)
        self.assertGreater(before, 0)
        use_potion(state, "bottled_potential")
        # 手牌全进抽牌堆又抽回 5 张：手牌数不变、抽牌堆被抽空。
        self.assertEqual(len(state.hand), before,
                         "洗牌后应当把手牌抽回原来的数量")
        self.assertEqual(len(state.draw_pile), 0,
                         f"5 张牌应当全部抽回：{len(state.draw_pile)} 张留在抽牌堆")

    def test_reboot_does_the_same_thing_as_bottled_potential(self):
        """``Reboot``（卡）与 ``BottledPotential`` 是同一件事的两种写法。

        真机 ``Reboot.cs:24-29`` 用 ``foreach`` 逐张搬，语义等于整堆搬 ——
        只认集合表达式的抽取器会把它整条拒掉。
        """
        from sts2_sim.content import CARD_DB
        from sts2_sim.core import Action, CardInstance, step
        state = self._combat()
        state.hand = [CardInstance("reboot")]
        state.draw_pile = []
        state.discard = []
        state.energy = 9
        target = -1 if CARD_DB["reboot"].target == "self" else 0
        step(state, Action("play_card", 0, target))
        self.assertEqual(len(state.draw_pile), 0, "洗牌后应当把牌抽回来")

    def test_distilled_chaos_autoplays_from_the_draw_pile(self):
        """``distilled_chaos``：从抽牌堆**顶**自动打出 3 张（不花能量）。"""
        from sts2_sim.core import CardInstance, use_potion
        state = self._combat()
        state.hand = []
        state.discard = []
        state.draw_pile = [CardInstance("strike_ironclad") for _ in range(3)]
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        use_potion(state, "distilled_chaos")
        self.assertEqual(len(state.draw_pile), 0, "抽牌堆的 3 张应当都被打出")
        self.assertLess(enemy.hp, 200, "打出的打击应当造成伤害")

    def test_havoc_autoplay_exhausts_the_card(self):
        """``Havoc``：``forceExhaust: true`` —— 自动打出的牌进**消耗堆**。

        丢掉这个标志，那张牌会进弃牌堆、还能再被抽到（比真机强），
        而日志完全正常。
        """
        from sts2_sim.content import CARD_DB
        from sts2_sim.core import Action, CardInstance, step
        state = self._combat()
        state.hand = [CardInstance("havoc")]
        state.draw_pile = [CardInstance("strike_ironclad")]
        state.discard = []
        state.energy = 9
        target = -1 if CARD_DB["havoc"].target == "self" else 0
        step(state, Action("play_card", 0, target))
        self.assertIn("strike_ironclad", [card.cid for card in state.exhaust],
                      f"自动打出的牌应当被消耗：{state.exhaust}")

    def test_entropic_brew_fills_every_empty_slot(self):
        """``entropic_brew``：``while (HasOpenPotionSlots)`` → **填满**所有空槽。"""
        from sts2_sim.core import use_potion
        state = self._combat()
        state.potions = [None, "block_potion", None]
        use_potion(state, "entropic_brew")
        self.assertNotIn(None, state.potions, f"空槽应当被填满：{state.potions}")
        self.assertEqual(state.potions[1], "block_potion", "已有药水的槽不许被动")

    def test_procured_potions_are_engine_executable(self):
        """采购到的药水必须**引擎算得对**（不许发一瓶用不了的）。"""
        from sts2_sim import content
        from sts2_sim.core import use_potion
        state = self._combat()
        state.potions = [None, None, None]
        use_potion(state, "entropic_brew")
        for pid in state.potions:
            self.assertIsNotNone(pid)
            self.assertFalse(content.POTIONS[pid].effects_incomplete,
                             f"{pid} 引擎执行不了，不该被发出来")


class TestPotionOperatorGate(unittest.TestCase):
    """药水必须和卡牌走**同一张**算子门禁（修 ``ValueError`` 崩溃）。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_no_usable_potion_carries_an_unimplemented_op(self):
        """"可用"的药水不许带引擎没有的算子。

        实测（本批次之前）``blessing_of_the_forge``（``upgrade_card``）、
        ``cunning_potion``（``upgrade_card``）、``glowwater_potion``
        （``exhaust``）都在"45 瓶可用"里，一旦抽到就用 ``ValueError``
        把训练循环炸掉 —— 而且只在抽到那瓶药时炸。
        """
        from sts2_sim import content
        offenders = []
        for pid, potion in content.POTIONS.items():
            if potion.effects_incomplete:
                continue
            for effect in potion.effects:
                if effect.op not in content.ENGINE_OPS:
                    offenders.append((pid, effect.op))
        self.assertEqual(offenders, [],
                         f"这些'可用'药水带引擎没有的算子（一用即崩）：{offenders}")

    def test_crashing_potions_are_marked_incomplete(self):
        from sts2_sim import content
        for pid in ("blessing_of_the_forge", "glowwater_potion"):
            potion = content.POTIONS.get(pid)
            self.assertIsNotNone(potion, f"{pid} 应当在药水表里")
            self.assertTrue(potion.effects_incomplete,
                            f"{pid} 的算子引擎没实现，必须标残缺")

    def test_generated_cards_can_be_upgraded_on_creation(self):
        """``CardCmd.Upgrade(刚生成的牌)`` → 造出来的就是**升级版实例**。

        真机两处写法（都是"造出来就是升级版"）：

        * ``CunningPotion``：``foreach (item in await Shiv.CreateInHand(…)) CardCmd.Upgrade(item);``
        * ``CosmicConcoction``：``foreach (item in distinctForCombat) { CardCmd.Upgrade(item); AddGeneratedCardToCombat(item, …); }``

        丢掉这个标志 = 给出的牌比真机**弱一档**（未升级的 Shiv / 无色牌），
        而报告完全正常。这两瓶药现在都应当是**可用**的。
        """
        from sts2_sim import content
        for pid in ("cunning_potion", "cosmic_concoction"):
            potion = content.POTIONS.get(pid)
            self.assertIsNotNone(potion, f"{pid} 应当在药水表里")
            self.assertFalse(potion.effects_incomplete,
                             f"{pid} 现在是完整可用的，不该再标残缺")
            self.assertTrue(any(e.upgrade for e in potion.effects),
                            f"{pid} 的效果必须带 upgrade 标志：{potion.effects}")

    def test_other_generated_potions_are_now_complete(self):
        """本批次另外两瓶也从"造牌认不出"变成可用。"""
        from sts2_sim import content
        for pid in ("orobic_acid", "pot_of_ghouls"):
            potion = content.POTIONS.get(pid)
            self.assertIsNotNone(potion, f"{pid} 应当在药水表里")
            self.assertFalse(potion.effects_incomplete, f"{pid} 应当可用")
            self.assertTrue(potion.effects, f"{pid} 必须有效果")

    def test_potions_whose_effects_were_wrong_are_now_incomplete(self):
        """4 瓶"看着可用、行为其实错"的药水现在**如实残缺**（§2.15 的修正）。

        | 药水 | 旧快照（错的） | 真机 |
        |---|---|---|
        | `ambergris` | 回复 50 点生命 | 回复**最大生命的 50%** |
        | `essence_of_darkness` | 引导 1 个黑暗球 | 按**球槽数**逐个引导 |
        | `liquid_memories` | 选一张回手（原价） | 选一张回手 **+ 本回合免费** |
        | `snecko_oil` | （改费用整段丢失） | 手牌费用随机 0–3 |
        """
        from sts2_sim import content
        for pid in ("ambergris", "essence_of_darkness", "liquid_memories",
                    "snecko_oil"):
            potion = content.POTIONS.get(pid)
            self.assertIsNotNone(potion, f"{pid} 应当在药水表里")
            self.assertTrue(potion.effects_incomplete,
                            f"{pid} 的效果与真机不一致，必须如实标残缺")

    def test_deprecated_potion_is_not_loaded_at_all(self):
        """``DeprecatedPotion`` 是**已从游戏移除**的占位类 → 不加载、不计缺口。

        出处：``DeprecatedPotion.cs`` 的类注释
        *"Represents a potion that has been removed from the game.
        Mostly used for the run history."*，且 ``Rarity => PotionRarity.None``
        （`rarity == "none"`）。它没有 `OnUse`，也不该进任何池 ——
        让它留在表里只会一直占一个"残缺"名额（`docs/12` §2.26）。
        """
        from sts2_sim import content
        self.assertNotIn("deprecated_potion", content.POTIONS,
                         "已移除的占位类不该进引擎的药水表")

    def test_using_an_incomplete_potion_is_refused_not_crashed(self):
        from sts2_sim.core import start_combat, use_potion
        state = start_combat(self.deck, ("nibbit",), seed=2)
        state.potions = ["blessing_of_the_forge", None, None]
        with self.assertRaises(ValueError):
            use_potion(state, "blessing_of_the_forge")
        # 但**动作空间**里不许出现它（`potion_actions` 只看 `effects_incomplete`）。
        from sts2_sim.core import legal_actions, potion_actions
        self.assertEqual(potion_actions(state), [])
        self.assertNotIn("use_potion",
                         {action.kind for action in legal_actions(state)})


@unittest.skipUnless(DECOMPILED.is_dir(), "缺反编译源码")
class TestExtractorGuards(unittest.TestCase):
    """抽取器一侧的门禁：**认不出就必须报缺口**，不许静默按"跑一次"处理。"""

    @staticmethod
    def _unsupported(name: str) -> list[str]:
        from tools.extract_cards import parse_card
        path = next(DECOMPILED.rglob(f"{name}.cs"))
        record = parse_card(path, collections.Counter())
        return list(record.get("unsupported") or [])

    @staticmethod
    def _effects(name: str) -> list[dict]:
        from tools.extract_cards import parse_card
        path = next(DECOMPILED.rglob(f"{name}.cs"))
        record = parse_card(path, collections.Counter())
        return list(record.get("effects") or [])

    def test_loop_body_takes_two_wounds(self):
        """``FightThrough`` 的 ``for (i < 2)`` 必须真的算 **2 次**。

        回归的是"取花括号循环体却用了圆括号配平"这个 bug：循环体在体内第一个
        ``)`` 处被截断，于是 `FightThrough` 只加 1 张 Wound（真机 2 张）。
        这张卡**当时已经在训练集里** —— 属于"报告干净、比真机弱"。
        """
        effects = self._effects("FightThrough")
        wound = [e for e in effects if e.get("card") == "wound"]
        self.assertEqual(len(wound), 1, f"应当恰好一条加 Wound 的效果：{effects}")
        self.assertEqual(wound[0].get("times"), 2)

    def test_repeat_loop_keeps_its_variable(self):
        """``BouncingFlask`` 的"重复 3 次"来自 ``DynamicVars.Repeat``。"""
        effects = self._effects("BouncingFlask")
        poison = [e for e in effects if e.get("power") == "poison"]
        self.assertTrue(poison, f"应当有条中毒效果：{effects}")
        self.assertEqual(poison[0].get("times_var"), "Repeat")

    def test_runtime_loop_is_reported_as_gap(self):
        """球数 / 条件决定次数的循环：引擎没有对应算子 → 必须报缺口。

        ``Shatter``（按球数激发）与 ``EvilEye``（``WasCardExhaustedThisTurn ? 2 : 1``）
        在修复前都被当成"跑一次"放进了训练集。
        """
        for name in ("Shatter", "EvilEye"):
            unsupported = self._unsupported(name)
            self.assertTrue(
                any("循环次数取决于运行期状态" in item for item in unsupported),
                f"{name} 应当报运行期次数缺口，实际：{unsupported}")

    def test_darkness_keeps_its_other_gap(self):
        """运行期循环**不许吞掉**这条命令本来能报的更具体缺口。"""
        unsupported = self._unsupported("Darkness")
        self.assertTrue(any("OrbCmd.Passive" in item for item in unsupported),
                        f"Darkness 的 OrbCmd.Passive 缺口被吞了：{unsupported}")

    def test_upgraded_only_branch_is_reported_as_gap(self):
        """``if (base.IsUpgraded)`` 里的结构性效果必须报缺口。

        引擎只建模**数值**升级，所以把它无条件抽出来等于"基础版比真机强"：
        ``Spinner`` 不升级也白得一个玻璃球、``TrueGrit`` 基础版本该**随机**
        消耗一张牌却变成了"玩家选"。
        """
        for name in ("Spinner", "TrueGrit"):
            unsupported = self._unsupported(name)
            self.assertTrue(
                any("升级版专属效果" in item for item in unsupported),
                f"{name} 应当报升级版专属效果缺口，实际：{unsupported}")

    def test_set_to_free_outside_generation_still_is_a_gap(self):
        """``SetToFreeThisTurn`` 只对**生成牌**那条路径实现了。

        其它用法（``BulletTime`` 的手牌全体免费、``MummifiedHand`` 的抽到能力牌免费）
        仍是缺口 —— 这条名单**不能**因为实现了生成牌那半就从缺口表里删掉：
        删掉之后 `BulletTime` 从"残缺"变成"干净"，而"本回合手牌全免费"整段丢失。
        """
        unsupported = self._unsupported("BulletTime")
        self.assertTrue(any("SetToFreeThisTurn" in item for item in unsupported),
                        f"BulletTime 的手牌免费应当仍是缺口：{unsupported}")

    def test_early_return_upgraded_guard_is_reported(self):
        """``if (!base.IsUpgraded) return;`` 这种**早退**守卫也要认。

        ``StormOfSteel`` / ``HiddenDaggers`` 写的是"没升级就直接返回"，
        后面才是"逐张升级刚造的牌"。只认 ``if (base.IsUpgraded) { … }``
        会让这两张卡的升级被当成**无条件** —— 基础版于是静默变强。
        """
        for name in ("StormOfSteel", "HiddenDaggers"):
            unsupported = self._unsupported(name)
            self.assertTrue(
                any("升级版专属效果" in item for item in unsupported),
                f"{name} 的早退守卫应当被认成升级版专属效果，实际：{unsupported}")

    def test_deck_wide_upgrade_is_not_swallowed(self):
        """``foreach (card in 牌组) CardCmd.Upgrade(card)`` **不是**造牌自带升级。

        回归的是本批次踩过的坑：把判据放宽成"循环变量被 Upgrade"之后，
        `Apotheosis` / `Dirge` / `KnifeTrap` / `Jackpot` / `DrainPower` /
        `StormOfSteel` 等 11 张卡的升级效果被当成造牌附带的升级**消费掉**，
        效果整段消失（`apotheosis` 直接变成空卡）。
        """
        from tools.extract_cards import generated_upgrades, method_body
        source = next(DECOMPILED.rglob("Apotheosis.cs")).read_text(
            encoding="utf-8", errors="replace")
        body = method_body(source, "OnPlay")
        names, _starts = generated_upgrades(body)
        self.assertEqual(names, set(),
                         "Apotheosis 遍历的是**牌组**，不是刚生成的牌")
        self.assertTrue(any(e.get("op") == "upgrade_card"
                            for e in self._effects("Apotheosis")),
                        "Apotheosis 的升级效果不许被吞掉")

    def test_in_combat_potion_pool_is_still_a_gap(self):
        """``Alchemize`` 用的是 ``CreateRandomPotionInCombat`` —— **另一种池**。

        InCombat 版额外过滤 ``CanBeGeneratedInCombat``，与 OutOfCombat 版
        （`EntropicBrew` / `DelicateFrond`）不是同一批药水。没实现就照旧报缺口，
        绝不按 OutOfCombat 近似 —— 那会让该卡发出真机发不出的药水。
        """
        gaps = " ".join(self._unsupported("Alchemize"))
        self.assertIn("战斗内药水池未实现", gaps)
        self.assertNotIn("procure_random_potion",
                         [e.get("op") for e in self._effects("Alchemize")])

    def test_autoplay_with_runtime_count_is_a_gap(self):
        """``Cascade`` 的张数是 ``ResolveEnergyXValue()`` 加升级分支 → 如实报缺口。"""
        gaps = " ".join(self._unsupported("Cascade"))
        self.assertIn("张数抽不出", gaps)

    def test_variable_count_selection_is_rejected(self):
        """可变张数选牌（``CardSelectorPrefs(prompt, 0, MAX)``）必须整条拒绝。

        引擎的挂起只有"选够 N 张"，**没有"结束选择"动作**；按 ``max(1, 0)``
        处理会把 `gamblers_brew`（弃任意张并抽等量）变成"弃 1 张、不抽"。
        """
        from sts2_sim.content import _source_reject_reason
        record = {"unsupported": [], "choice_commands": 0,
                  "effects": [{"op": "select_card", "from": "hand",
                               "purpose": "discard", "amount": 0,
                               "target": "self"}]}
        self.assertIsNotNone(_source_reject_reason(record, require_cost=False))


if __name__ == "__main__":
    unittest.main()
