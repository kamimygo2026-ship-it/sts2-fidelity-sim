"""Run 层效果的回归测试（``docs/09`` §19）。

Run 层效果作用于**牌组 / 金币 / 最大生命**，与战斗内效果共用一套 DSL、
但结算对象不同。这一层的两个坑：

| 症状 | 根因 |
|---|---|
| 增删一个战后遗物要改引擎 | `burning_blood` 被**硬编码**在 `run.py` 里 |
| "升级一张牌"变成了"升级一张随机牌" | 真机是 `CardSelectCmd.FromDeckForUpgrade`（**玩家选**），抽取器没识别成选牌 |
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"


def setup_content():
    if not CONTENT.exists():
        raise unittest.SkipTest(f"缺内容目录 {CONTENT}")
    from sts2_sim.featurize import configure_content
    configure_content(str(CONTENT))


def restore_builtin() -> None:
    from sts2_sim.content import load_builtin
    from sts2_sim.featurize import rebuild_vocab
    load_builtin()
    rebuild_vocab()


class FakePlayer:
    def __init__(self):
        self.hp = 40
        self.max_hp = 80
        self.gold = 99
        self.deck: list = []
        self.relics: list = []


class FakeRun:
    def __init__(self):
        from sts2_sim.rng import RngSet

        class Hidden:
            rng = RngSet(1)

        self.player = FakePlayer()
        self.hidden = Hidden()


class RunEffectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def run_state(self):
        return FakeRun()

    def test_gain_max_hp_also_heals(self):
        """``CreatureCmd.GainMaxHp`` **会回血** —— 加多少上限就回多少血。

        .. code-block:: csharp

           public static async Task GainMaxHp(Creature creature, decimal amount)
           {
               decimal num = await SetMaxHp(creature, (decimal)creature.MaxHp + amount);
               ...
               await Heal(creature, num);        // ← 在方法体**最后一行**
           }

        ⚠️ 这一条是被用户纠正后补的：我一开始只读了方法**前十行**
        （到 `SetMaxHp` 为止），于是实现成"只加上限、不回血"，
        症状是"所有加最大生命的遗物都少回一次血"。
        `Heal` 在**方法末尾**，截断的 grep 窗口看不到它。
        """
        from sts2_sim import runeffects
        from sts2_sim.content import Effect
        state = self.run_state()
        events: list[str] = []
        runeffects.apply_run_effects(
            state, [Effect(op="gain_max_hp", amount=20)], events)
        self.assertEqual(state.player.max_hp, 100)
        self.assertEqual(state.player.hp, 60, "加 20 上限应当同时回 20 血")

    def test_gain_max_hp_heal_is_clamped(self):
        """回血量受**新上限**约束（`Heal` 内部钳制，不会超上限）。"""
        from sts2_sim import runeffects
        from sts2_sim.content import Effect
        state = self.run_state()
        state.player.hp = 79
        events: list[str] = []
        runeffects.apply_run_effects(
            state, [Effect(op="gain_max_hp", amount=20)], events)
        self.assertEqual(state.player.max_hp, 100)
        self.assertEqual(state.player.hp, 99, "79+20 会被上限截到 99")

    def test_lose_max_hp_only_shrinks_the_tank(self):
        """``CreatureCmd.LoseMaxHp``：**只砍血槽**，当前血量不按全额扣。

        .. code-block:: csharp

           decimal newMaxHp = (decimal)creature.MaxHp - amount;
           if (newMaxHp < (decimal)creature.CurrentHp)
               await Damage(context, creature, CurrentHp - newMaxHp, Unblockable|Unpowered, …);
           await SetMaxHp(creature, Math.Max(1.0m, newMaxHp));

        两条分支都要测：

        * 当前血 **低于**新上限 → 血不变，只缩槽
        * 当前血 **高于**新上限 → 掉到新上限（掉的正好是超出部分）
        """
        from sts2_sim import runeffects
        from sts2_sim.content import Effect
        # 分支一：血 40、上限 80，砍 30 → 40/50
        state = self.run_state()
        runeffects.apply_run_effects(
            state, [Effect(op="lose_max_hp", amount=30)], [])
        self.assertEqual((state.player.hp, state.player.max_hp), (40, 50))
        # 分支二：血 70、上限 80，砍 30 → 50/50
        state = self.run_state()
        state.player.hp = 70
        runeffects.apply_run_effects(
            state, [Effect(op="lose_max_hp", amount=30)], [])
        self.assertEqual((state.player.hp, state.player.max_hp), (50, 50))

    def test_lose_max_hp_never_goes_below_one(self):
        """上限最低砍到 **1**（源码 ``Math.Max(1m, newMaxHp)``），不会归零。"""
        from sts2_sim import runeffects
        from sts2_sim.content import Effect
        state = self.run_state()
        runeffects.apply_run_effects(
            state, [Effect(op="lose_max_hp", amount=999)], [])
        self.assertEqual(state.player.max_hp, 1)

    def test_heal_and_lose_hp_clamp(self):
        from sts2_sim import runeffects
        from sts2_sim.content import Effect
        state = self.run_state()
        events: list[str] = []
        runeffects.apply_run_effects(state, [Effect(op="heal", amount=999)], events)
        self.assertEqual(state.player.hp, 80, "回复不该超过上限")
        runeffects.apply_run_effects(state, [Effect(op="lose_hp", amount=999)], events)
        self.assertEqual(state.player.hp, 0, "失去生命可以扣到 0")

    def test_add_card_to_deck(self):
        from sts2_sim import runeffects
        from sts2_sim.content import Effect
        state = self.run_state()
        events: list[str] = []
        runeffects.apply_run_effects(
            state, [Effect(op="add_card", amount=1, card="enthralled", pile="deck")],
            events)
        self.assertEqual([c.cid for c in state.player.deck], ["enthralled"])

    def test_unknown_card_is_reported_not_silently_added(self):
        """要加的卡不在卡表里 → **报出来**，不加一张空卡。"""
        from sts2_sim import runeffects
        from sts2_sim.content import Effect
        state = self.run_state()
        events: list[str] = []
        runeffects.apply_run_effects(
            state, [Effect(op="add_card", amount=1, card="not_a_card", pile="deck")],
            events)
        self.assertEqual(state.player.deck, [])
        self.assertTrue(any("不在卡表" in e for e in events), events)

    def test_choice_ops_are_refused_not_approximated(self):
        """**需要玩家选择**的算子不能自动结算。

        ``Whetstone``（升级一张）/ ``Astrolabe``（转化）/ ``PandorasBox``（移除）
        在真机里都是"让玩家从牌组里挑"。引擎在"恰好只有一种可选"时才自动做，
        否则**如实拒绝**并记一笔 —— 随机挑一张等于把机制换掉了。
        """
        from sts2_sim import runeffects
        from sts2_sim.content import Effect
        from sts2_sim.core import CardInstance
        state = self.run_state()
        state.player.deck = [CardInstance("strike"), CardInstance("defend"),
                             CardInstance("bash")]
        events: list[str] = []
        runeffects.apply_run_effects(
            state, [Effect(op="upgrade_card", amount=1, target="self")], events)
        self.assertEqual([c.upgraded for c in state.player.deck], [False] * 3,
                         "多张候选时不该擅自升级一张")
        self.assertTrue(any("需要玩家选择" in e for e in events), events)


class RelicRunHookTest(unittest.TestCase):
    """遗物的 Run 层时机：拾取 / 战斗结束 / 战斗胜利。"""

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_combat_victory_relics_heal(self):
        """``AfterCombatVictory``：燃烧之血回 6、黑血回 12，**两件都生效**。

        ⚠️ 以前 ``burning_blood`` 是**硬编码**在 ``run.py`` 里的
        （``if "burning_blood" in relics: hp += 6``）—— 增删一个战后遗物要改引擎，
        而且漏掉的不报错、只是不生效。
        """
        from sts2_sim import relics
        state = FakeRun()
        state.player.hp = 30
        state.player.relics = ["burning_blood", "black_blood"]
        events: list[str] = []
        relics.apply_run_hook(state, "combat_victory", state.player.relics, events)
        self.assertEqual(state.player.hp, 48, f"应当 30+6+12，实际 {state.player.hp}")

    def test_obtained_hook_adds_a_curse(self):
        """``blood_soaked_rose`` 拾取时往牌组塞一张 ``enthralled``。"""
        from sts2_sim import relics
        state = FakeRun()
        state.player.relics = ["blood_soaked_rose"]
        events: list[str] = []
        relics.apply_run_hook(state, "obtained", state.player.relics, events)
        self.assertEqual([c.cid for c in state.player.deck], ["enthralled"])

    def test_run_timings_are_on_the_bus(self):
        """Run 层时机也要在钩子总线上有声明的钩子（不然那一层的遗物全不生效）。"""
        from sts2_sim import hooks
        for timing in ("obtained", "combat_end", "combat_victory"):
            hook = hooks.RELIC_TIMING_TO_HOOK.get(timing)
            self.assertIsNotNone(hook, f"{timing} 没有对应的总线钩子")
            self.assertIn(hook, hooks.HOOKS)

    def test_no_hardcoded_relic_ids_in_run_layer(self):
        """结构护栏：``run.py`` 里不许再出现硬编码的遗物 id。"""
        source = (ROOT / "sts2_sim" / "run.py").read_text(encoding="utf-8")
        for rid in ("burning_blood", "black_blood", "blood_soaked_rose"):
            self.assertNotIn(f'"{rid}"', source,
                             f"run.py 里硬编码了遗物 {rid}，应走 Run 层钩子")


class RewardOfferTest(unittest.TestCase):
    """`RewardsCmd.OfferCustom`：提供一组奖励让玩家挑。

    真机是弹奖励界面，引擎把它表达成 `RunState.pending_rewards` +
    `take_reward` / `skip_reward` 动作，与"卡牌三选一"走同一套交互。
    """

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def make_env(self, seed=3):
        from sts2_sim.run import RunEnv
        env = RunEnv(seed=seed, attempt_budget=0)
        env.reset()
        return env

    def test_calling_bell_offers_three_relic_rarities(self):
        """召唤铃：``new RelicReward(RelicRarity.X)`` × 3（普通 / 罕见 / 稀有）。

        源码（``CallingBell.GenerateRewards``）里三个奖励各写一次稀有度，
        不是"随机三个遗物"。抽错会把"必得一个稀有"变成"可能三个普通"。
        """
        from sts2_sim import content
        bell = content.RELICS["calling_bell"]
        hooks = [h for h in bell.hooks if h.timing == "obtained"]
        rarities = sorted(dict(e.card_filter).get("rarity")
                          for h in hooks for e in h.effects
                          if e.op == "offer_rewards")
        self.assertEqual(rarities, ["common", "rare", "uncommon"])

    def test_offer_puts_the_run_into_the_reward_phase(self):
        from sts2_sim import run as runmod
        env = self.make_env()
        state = env._state
        state.player.relics.append("calling_bell")
        env._fire_relic_run_hook("obtained", ["calling_bell"])
        self.assertEqual(state.room.kind, runmod.PHASE_REWARD)
        self.assertEqual(len(state.pending_rewards), 3)
        actions = runmod.legal_meta_actions(state)
        self.assertIn("take_reward", [a.kind for a in actions])
        self.assertIn("skip_reward", [a.kind for a in actions])

    def test_taking_a_reward_grants_exactly_that_relic(self):
        """⭐ **回归护栏**：领一次奖励不能级联。

        曾经 `AfterObtained` 传的是**全部**遗物，于是每次领取都重新触发
        召唤铃自己的 ``AfterObtained``，又生成 3 个候选 ——
        实测领一次奖励后玩家从 1 个遗物变成 **120 个**。

        真机的 ``AfterObtained`` 是**逐个遗物**的回调，只对刚获得的那一个触发。
        """
        from sts2_sim import run as runmod
        env = self.make_env()
        state = env._state
        # ⚠️ 基线要**从当前遗物数算**，不能写死 1：审计 F02 修好之后，
        # 角色会正确地带上自己的初始遗物（铁甲战士 = burning_blood），
        # 于是"领完应当是 4 个"这种绝对数字会在换角色时失效。
        before = len(state.player.relics)
        state.player.relics.append("calling_bell")
        env._fire_relic_run_hook("obtained", ["calling_bell"])
        while state.pending_rewards:
            env._resolve_reward(runmod.MetaAction("take_reward", 0))
        # 铃铛本身 + 3 个领到的遗物
        self.assertEqual(len(state.player.relics), before + 4,
                         f"应当只多 4 个遗物，实际 {len(state.player.relics) - before}")
        self.assertEqual(state.room.kind, runmod.PHASE_MAP, "领完回到地图")

    def test_skipping_closes_the_offer(self):
        from sts2_sim import run as runmod
        env = self.make_env()
        state = env._state
        state.player.relics.append("calling_bell")
        env._fire_relic_run_hook("obtained", ["calling_bell"])
        before = len(state.player.relics)
        while state.pending_rewards:
            env._resolve_reward(runmod.MetaAction("skip_reward"))
        self.assertEqual(len(state.player.relics), before, "跳过不该给东西")
        self.assertEqual(state.room.kind, runmod.PHASE_MAP)

    def test_potion_offer_fills_a_slot(self):
        """大锅：提供 N 瓶药水（``new PotionReward`` 在循环里）。"""
        from sts2_sim import run as runmod
        env = self.make_env(seed=5)
        state = env._state
        state.player.relics.append("cauldron")
        env._fire_relic_run_hook("obtained", ["cauldron"])
        self.assertTrue(state.pending_rewards, "大锅应当提供药水奖励")
        env._resolve_reward(runmod.MetaAction("take_reward", 0))
        self.assertIsNotNone(state.player.potions[0], "应当占掉第一个空槽")

    def test_reward_kinds_come_from_the_source(self):
        """奖励类型必须来自源码的 ``new XxxReward(...)``，不能猜。"""
        from sts2_sim import content
        expected = {"calling_bell": "relic", "cauldron": "potion",
                    "glass_eye": "card", "toy_box": "relic"}
        for rid, kind in expected.items():
            relic = content.RELICS[rid]
            kinds = {dict(e.card_filter).get("kind")
                     for h in relic.hooks if h.timing == "obtained"
                     for e in h.effects if e.op == "offer_rewards"}
            self.assertIn(kind, kinds, f"{rid} 应当是 {kind} 奖励，实际 {kinds}")


if __name__ == "__main__":
    unittest.main()
