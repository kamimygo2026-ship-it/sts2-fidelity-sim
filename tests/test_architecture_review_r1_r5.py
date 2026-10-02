"""外部架构复核（``docs/15`` R1–R5）的回归护栏。

这五条都是**复现出来**的真问题，不是"看起来不优雅"：

| 编号 | 症状（修之前实测） | 根因 | 守护测试 |
|---|---|---|---|
| R1 | 自动打出遇到选牌时立刻收尾：格挡先到手、牌先进弃牌堆、**同一个 uid 出现在两个牌堆**；``OneTwoPunch`` 对自动打出不生效（手动 12 点 / 自动 6 点） | 手动 / 自动 / 狡诈三条出牌路径**各写一份**，各自漏掉"打出次数"与"挂起时不许收尾" | :class:`TestUnifiedPlayProtocol` |
| R2 | ``CombatPool(character="defect")`` 里 ``CombatState.character == "ironclad"``：生成能力牌时按**错的卡池**取牌 | ``SpireEnv`` 只传 deck/encounter/seed/hp，角色等上下文一个都不传 | :class:`TestCombatInitContract` |
| R3 | ``creative_ai`` 被准入却生成 ``biased_cognition``（引擎执行不了的牌）；``draw_card_reward(RngSet(0))`` 发出 ``demonic_shield``；``normality`` 在手里毫无代价 | 运行时生成与奖励**绕过门禁**；卡牌级 ``ShouldPlay`` 整批没实现 | :class:`TestGateCoversEveryEntrance` |
| R4 | "腐败 + 0 能量 + 一张防御"：合法动作里有出牌，观测却写 ``cost=1, playable=False``；加 9 层中毒 / 球槽 +2 后编码数组**一字不差** | 观察层自己抄了一份可打性；张量化只编码 strength/vulnerable/weak 三个能力 | :class:`TestObservationMatchesRules` |
| R5 | ``restart_node()`` 后 ``env._rng is not state.hidden.rng``；同一节点快照重开出**28 个不同**的第二幕 | Run 的随机状态有**两个所有者**（外壳字段 + ``Hidden``） | :class:`TestRunRngSingleOwner` |

⚠️ 这些测试只证明"引擎与自己的规则一致"，**不**证明与真机一致（真机对拍仍为 0）。
"""

from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

import numpy as np

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


class TestUnifiedPlayProtocol(unittest.TestCase):
    """R1：``CardCmd.AutoPlay`` 与手动出牌共用 ``CardModel.OnPlayWrapper`` 的四步。

    源码（``CardModel.cs:1882-1997``）：算落点 → 算 ``GeneratePlayCount`` →
    登记 ``History.CardPlayStarted`` → 循环内每一次打出分发
    ``BeforeCardPlayed`` / ``OnPlay`` / ``AfterCardPlayed`` → 按落点入堆。
    三条入口（手动 / 通用自动 / 狡诈）必须**同一条**执行器。
    """

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def _headbutt_autoplay(self):
        """``Headbutt`` 自动打出，且弃牌堆里有一张牌（它的效果要选牌）。"""
        from sts2_sim import core
        state = core.start_combat(["headbutt", "defend_ironclad"], ["axebot"], seed=1)
        card = next(x for x in state.hand if x.cid == "headbutt")
        state.hand = [card]
        state.discard = [core.CardInstance("defend_ironclad")]
        state.player.powers["afterimage"] = 1
        events: list[str] = []
        done = core.autoplay_card(state, card, events)
        return state, card, done

    def test_autoplay_reports_pause_and_does_not_finish(self):
        """挂起时必须**返回 False**，且不许做任何"这张牌打完了"的动作。

        修之前：``AfterCardPlayed`` 立刻触发（Afterimage 先给 1 格挡）、
        ``_finish_played_card`` 把牌塞进弃牌堆、出牌区清空 —— 而效果链其实
        还挂在选牌上。``AfterCardPlayed`` 的时机是审计 F06 专门修过的，
        自动打出这条路当时漏了同一件事。
        """
        state, card, done = self._headbutt_autoplay()
        self.assertIsNotNone(state.pending, "这张牌的效果要选牌，必须挂起")
        self.assertFalse(done, "挂起时执行器必须返回 False（调用方据此停手）")
        self.assertEqual(state.player.block, 0,
                         "选牌还没完成，AfterCardPlayed 不该已经触发（Afterimage 给了格挡）")
        self.assertIn(card, state.play_area, "挂起期间打出的牌留在出牌区")
        self.assertNotIn(card, state.discard, "挂起期间不该已经落堆")

    def test_card_is_not_a_candidate_of_its_own_selection(self):
        """挂在选牌上时，打出的那张牌**不能**出现在自己的候选里。

        修之前它已经被塞进弃牌堆，于是 ``Headbutt`` 可以从自己的弃牌堆里
        挑回自己 —— 候选集合是公开信息，这直接改变了动作空间。
        """
        state, card, _done = self._headbutt_autoplay()
        candidates = state.pending.candidates(state)
        self.assertNotIn(card, candidates)

    def test_resuming_does_not_duplicate_the_card(self):
        """续跑完成后，同一个 ``uid`` 只能属于**一个**牌堆。

        修之前：挂起时先落堆一次、续跑又落堆一次 → 同一 uid 出现两次，
        牌区不变量被破坏（而日志一切正常）。
        """
        from sts2_sim import core
        state, card, _done = self._headbutt_autoplay()
        core.step(state, core.legal_actions(state)[0])
        self.assertIsNone(state.pending)
        self.assertEqual(sum(x.uid == card.uid for x in state.all_piles()), 1)

    def test_autoplay_uses_generate_play_count(self):
        """``OneTwoPunchPower``（``GeneratePlayCount``）对自动打出**同样**生效。

        修之前基础打击手动 12 点、自动只有 6 点 —— 自动打出整条路没经过
        ``card_play_count``。出处：``CardModel.OnPlayWrapper`` 对自动与手动
        都算 playCount；``OneTwoPunchPower`` 没有排除自动打出。
        """
        from sts2_sim import core

        def damage(auto: bool) -> int:
            state = core.start_combat(["strike_ironclad"], ["axebot"], seed=1)
            state.player.powers["one_two_punch"] = 1
            before = state.enemies[0].hp
            if auto:
                core.autoplay_card(state, state.hand[0], [])
            else:
                core.step(state, core.Action("play_card", 0, 0))
            return before - state.enemies[0].hp

        self.assertEqual(damage(True), damage(False))
        self.assertEqual(damage(True), 12, "打击 6 点 × 打两次")

    def test_manual_and_auto_share_one_executor(self):
        """反向验证：手动出牌也走同一个执行器（函数是**唯一**入口）。"""
        import inspect
        from sts2_sim import core
        source = inspect.getsource(core.step)
        self.assertIn("play_card(state, card, events", source,
                      "手动出牌必须调用统一执行器 play_card")
        auto = inspect.getsource(core.autoplay_card)
        self.assertIn("play_card(", auto)
        sly = inspect.getsource(core._autoplay_sly)
        self.assertIn("play_card(", sly)

    def test_sly_autoplay_goes_through_the_same_executor(self):
        """狡诈自动打出：落在**弃牌堆**的牌要先取出，再走完整流程（不重复入堆）。"""
        from sts2_sim import core
        state = core.start_combat(["defend_ironclad"], ["axebot"], seed=3)
        sly = core.CardInstance("strike_ironclad")
        sly.keywords = ("Sly",)
        state.hand = []
        state.discard = [sly]
        state.energy = 5
        done = core._autoplay_sly(state, sly, [])
        self.assertTrue(done)
        self.assertEqual(sum(x.uid == sly.uid for x in state.all_piles()), 1)


class TestCombatInitContract(unittest.TestCase):
    """R2：Run / 训练 / 回放三条入口的**战斗上下文**必须一致。

    出处：``CardModel`` 生成牌时按 ``Owner.Character.CardPool`` 取池
    （``CreativeAiPower.cs:27``），所以"声明了 defect 却在跑 ironclad"
    不是小事：生成的牌来自**另一个角色**的池子。
    """

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_training_pool_passes_the_declared_character(self):
        from sts2_rl.ppo import CombatPool
        for character in ("ironclad", "defect", "silent", "necrobinder", "regent"):
            with self.subTest(character=character):
                pool = CombatPool(n_envs=1, base_seed=1, character=character)
                state = pool.envs[0].raw_state
                self.assertEqual(state.character, character)

    def test_max_hp_comes_from_the_character_table(self):
        """最大生命按角色表补齐：defect 是 75，给 80 会造出真机不存在的状态。"""
        from sts2_rl.ppo import CombatPool
        from sts2_sim.content import CHARACTERS
        for character in ("ironclad", "defect", "silent", "necrobinder", "regent"):
            with self.subTest(character=character):
                pool = CombatPool(n_envs=1, base_seed=1, character=character)
                state = pool.envs[0].raw_state
                self.assertEqual(state.player.max_hp, CHARACTERS[character].hp)
                self.assertLessEqual(state.player.hp, state.player.max_hp)

    def test_combat_init_is_the_single_source(self):
        """``combat_init()`` 报的就是战斗里实际生效的那份上下文。"""
        from sts2_sim.env import SpireEnv
        env = SpireEnv(seed=5, character="defect", encounter=("axebot",))
        env.reset()
        init = env.combat_init()
        state = env.raw_state
        self.assertEqual(init.character, state.character)
        self.assertEqual(init.player_max_hp, state.player.max_hp)
        self.assertEqual(init.max_energy, state.max_energy)
        self.assertEqual(init.room, state.room)

    def test_evaluation_entry_passes_character_too(self):
        """评估入口（``make_env``）同样不许静默退回默认角色。"""
        import random
        from sts2_rl.ppo import make_env
        env = make_env(1, random.Random(0), 0, character="defect")
        env.reset()
        self.assertEqual(env.raw_state.character, "defect")

    def test_replay_entry_matches_reset(self):
        """同一份上下文下，``reset`` 与 ``replay`` 得到同一个角色与上限。"""
        from sts2_sim.env import SpireEnv
        env = SpireEnv(seed=7, character="defect", encounter=("axebot",))
        env.reset()
        replayed = env.replay([], seed=7)
        self.assertEqual(replayed.character, env.raw_state.character)
        self.assertEqual(replayed.player.max_hp, env.raw_state.player.max_hp)


class TestGateCoversEveryEntrance(unittest.TestCase):
    """R3：**所有**产出内容对象的入口都要过门禁，被动查询也要门禁。

    ``docs/13`` §7 的"闭包可执行"要求"可能生成、召唤、转化、发放到的对象
    也满足准入"。运行时随机生成与奖励发放原先都绕过了这条。
    """

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_runtime_generation_never_produces_an_inadmissible_card(self):
        """能力造牌（``CardFactory.GetDistinctForCombat``）必须只发合格卡。

        修之前种子 8 会生成 ``biased_cognition`` —— 一张门禁以
        ``effects_incomplete / unimplemented_power`` 拒绝的牌。
        """
        from sts2_sim import core, eligibility, powers
        from sts2_sim.rng import RngSet
        for seed in range(12):
            with self.subTest(seed=seed):
                state = core.start_combat(["defend_defect"], ["axebot"], seed=1,
                                          character="defect")
                state.hidden.rng = RngSet(seed)
                made = powers.generate_cards_to_hand(state, 2, [], card_type="power")
                for cid in made:
                    self.assertTrue(eligibility.card_admission(cid).admitted, cid)

    def test_reward_pool_is_gated(self):
        """卡牌奖励不许发出引擎执行不了的牌（修之前 ``RngSet(0)`` 发 ``demonic_shield``）。"""
        from sts2_sim import eligibility
        from sts2_sim.run import draw_card_reward
        from sts2_sim.rng import RngSet
        for seed in range(30):
            for kind in ("monster", "elite", "boss", "shop"):
                with self.subTest(seed=seed, kind=kind):
                    for cid in draw_card_reward(RngSet(seed), kind):
                        self.assertTrue(eligibility.card_admission(cid).admitted, cid)

    def test_shop_and_event_pools_are_gated_too(self):
        """商店进货走的是同一个 ``reward_pool``，不该有第二个池子。"""
        from sts2_sim import eligibility
        from sts2_sim.run import draw_shop_stock
        from sts2_sim.rng import RngSet
        for seed in range(10):
            with self.subTest(seed=seed):
                for cid, _price in draw_shop_stock(RngSet(seed)):
                    self.assertTrue(eligibility.card_admission(cid).admitted, cid)

    def test_normality_blocks_the_fourth_play(self):
        """``Normality`` 在**手里**且本回合已打出 3 张 → 第 4 张打不出。

        出处：``Normality.ShouldPlay``（Normality.cs:40-52）+
        ``CardsPlayedThisTurn`` = ``History.CardPlaysStarted`` 本回合计数（:33）。
        修之前这条覆写**根本没实现**，于是这张诅咒是纯收益。
        """
        from sts2_sim import core
        state = core.start_combat(["defend_ironclad"] * 4 + ["normality"],
                                  ["axebot"], seed=1)
        state.hand = ([c for c in state.hand if c.cid == "normality"]
                      + [c for c in state.hand if c.cid != "normality"])
        state.energy = 9
        played = 0
        for _ in range(4):
            actions = [a for a in core.legal_actions(state) if a.kind == "play_card"]
            if not actions:
                break
            core.step(state, actions[0])
            played += 1
        self.assertEqual(played, 3, "Normality 手里时每回合最多 3 张")

    def test_enthralled_blocks_other_cards_but_not_itself(self):
        """``Enthralled``：手里的其它牌打不出；它自己可以打出。

        出处：``Enthralled.ShouldPlay``（Enthralled.cs:21-41）——
        ``card is Enthralled`` 放行、``autoPlayType != None`` 放行、其余否决。

        ⚠️ "它自己可以打出"这一半要求引擎承认"**没有 ``OnPlay`` 也能打出**"：
        真机 ``CanPlay`` 不要求"有效果"，而引擎的"空卡不可打出"只是模拟器约定。
        见 ``core.card_playable`` 里那条窄例外。
        """
        from sts2_sim import core
        state = core.start_combat(["strike_ironclad", "enthralled"],
                                  ["axebot"], seed=2)
        state.hand = [core.CardInstance("enthralled"),
                      core.CardInstance("strike_ironclad")]
        state.energy = 9
        playable = {a.hand_index for a in core.legal_actions(state)
                    if a.kind == "play_card"}
        self.assertEqual(playable, {0}, "只有 Enthralled 自己能打出")
        # 自动打出不受它限制（`autoPlayType != AutoPlayType.None` 直接放行）
        auto = core.CardInstance("strike_ironclad")
        state.hand.append(auto)
        self.assertTrue(core.hand_should_play_allows(state, auto, auto=True))
        self.assertFalse(core.hand_should_play_allows(state, auto, auto=False))

    def test_should_play_table_is_source_backed(self):
        """覆写方只有两张牌 —— 与源码里 ``override bool ShouldPlay`` 的数量一致。"""
        from sts2_sim import core
        self.assertEqual(set(core.CARD_SHOULD_PLAY), {"normality", "enthralled"})

    def test_generation_pool_shrink_is_reported_not_silent(self):
        """缩池必须**说出来**（事件日志），不许静默近似。

        ``docs/13`` §7 的口径：受限课程预先缩池并标注分布变化，
        完整范围模式遇到缺内容记 ``unsupported`` —— 两件事都不能靠"发出来再重抽"掩盖。
        """
        from sts2_sim import core, powers
        from sts2_sim.rng import RngSet
        state = core.start_combat(["defend_defect"], ["axebot"], seed=1,
                                  character="defect")
        state.hidden.rng = RngSet(8)
        events: list[str] = []
        powers.generate_cards_to_hand(state, 1, events, card_type="power")
        self.assertTrue(any("准入门禁缩小" in e for e in events),
                        f"缩池没有记录：{events}")


class TestObservationMatchesRules(unittest.TestCase):
    """R4：公开观察必须由**规则查询**产生，且公开状态不能消失。

    两半都要测，而且**反向**也要测（``docs/15`` §4 的验收口径）：
    "公开状态变化应改变相应编码；只改变隐藏状态而公开结果不变时，编码与候选必须不变"。
    """

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_observed_cost_is_the_actual_cost(self):
        """腐败生效时实际费用是 0，观测也必须写 0 且可打出。"""
        from sts2_sim import core
        from sts2_sim.observe import observe
        state = core.start_combat(["defend_ironclad"], ["axebot"], seed=1)
        state.player.powers["corruption"] = 1
        state.energy = 0
        view = observe(state).hand[0]
        self.assertEqual(view.cost, core.play_cost(state, state.hand[0]))
        self.assertEqual(view.cost, 0)
        self.assertTrue(view.playable)
        self.assertTrue(any(a.kind == "play_card"
                            for a in core.legal_actions(state)))

    def test_observed_playable_agrees_with_legal_actions(self):
        """逐张对照：观测的 ``playable`` 与合法动作**必须**一致。

        两个信号互相否认时 mask 补救不了：模型会学到"playable 这一列不可信"。
        """
        from sts2_sim import core
        from sts2_sim.observe import observe
        cases = [
            (["defend_ironclad", "strike_ironclad", "comet"], {}),
            (["defend_ironclad", "strike_ironclad"], {"corruption": 1}),
            (["comet"], {"void_form": 1}),
        ]
        for hand, powers in cases:
            with self.subTest(hand=hand, powers=powers):
                state = core.start_combat(list(hand), ["axebot"], seed=1)
                for pid, amount in powers.items():
                    state.player.powers[pid] = amount
                state.energy = 0
                legal = {a.hand_index for a in core.legal_actions(state)
                         if a.kind == "play_card"}
                view = observe(state)
                for index, card in enumerate(view.hand):
                    self.assertEqual(
                        card.playable, index in legal,
                        f"第 {index} 张（{card.cid}）观测与合法动作不一致")

    def test_poison_is_visible_in_the_encoding(self):
        """给玩家加 9 层中毒必须改变编码（修之前数组一字不差）。

        中毒是公开信息，也是"该不该继续出牌"的核心依据；197 个能力里
        原先只有 strength/vulnerable/weak 编码得出来。
        """
        from sts2_sim import core
        from sts2_sim.featurize import encode_observation
        from sts2_sim.observe import observe
        state = core.start_combat(["defend_ironclad"], ["axebot"], seed=1)
        legal = core.legal_actions(state)
        base_obs = observe(state)
        base = encode_observation(base_obs, legal)
        changed = dataclasses.replace(
            base_obs, player_powers=base_obs.player_powers + (("poison", 9),))
        after = encode_observation(changed, legal)
        self.assertTrue(any(not np.array_equal(base[k], after[k]) for k in base),
                        "中毒没有进入编码")

    def test_orb_slots_are_visible_in_the_encoding(self):
        """**空**球槽也是公开的（``orb_count_x5`` 只数已充能的球）。"""
        from sts2_sim import core
        from sts2_sim.featurize import encode_observation
        from sts2_sim.observe import observe
        state = core.start_combat(["defend_ironclad"], ["axebot"], seed=1)
        legal = core.legal_actions(state)
        base_obs = observe(state)
        base = encode_observation(base_obs, legal)
        changed = dataclasses.replace(base_obs, orb_slots=base_obs.orb_slots + 2)
        after = encode_observation(changed, legal)
        self.assertTrue(any(not np.array_equal(base[k], after[k]) for k in base),
                        "球槽数没有进入编码")

    def test_enemy_powers_are_visible_too(self):
        """敌人身上的能力同样要看得见（它决定"打不打得动"）。"""
        from sts2_sim import core
        from sts2_sim.featurize import encode_observation
        from sts2_sim.observe import observe
        state = core.start_combat(["defend_ironclad"], ["axebot"], seed=1)
        legal = core.legal_actions(state)
        base_obs = observe(state)
        base = encode_observation(base_obs, legal)
        enemy = base_obs.enemies[0]
        changed = dataclasses.replace(
            base_obs, enemies=(dataclasses.replace(
                enemy, powers=enemy.powers + (("artifact", 2),)),)
            + base_obs.enemies[1:])
        after = encode_observation(changed, legal)
        self.assertTrue(any(not np.array_equal(base[k], after[k]) for k in base),
                        "敌人的能力没有进入编码")

    def test_feature_layout_covers_the_new_columns(self):
        """新增列必须出现在布局里（``NUM_FEATURES`` 由布局推导，不许手写）。"""
        from sts2_sim.featurize import FEATURE_LAYOUT, NUM_FEATURES
        for name in ("orb_slots_x5", "power_amount_x10", "power_on_player"):
            self.assertIn(name, FEATURE_LAYOUT)
        self.assertEqual(NUM_FEATURES, len(FEATURE_LAYOUT))

    def test_hidden_state_changes_do_not_change_the_encoding(self):
        """**反向**验收：只改隐藏状态（未揭示的牌序）时编码必须逐位不变。"""
        from sts2_sim import core
        from sts2_sim.featurize import encode_observation
        from sts2_sim.observe import observe
        state = core.start_combat(["defend_ironclad", "strike_ironclad"],
                                  ["axebot"], seed=1)
        legal = core.legal_actions(state)
        base = encode_observation(observe(state), legal)
        state.draw_pile = list(reversed(state.draw_pile))       # 隐藏顺序
        state.hidden.rng = type(state.hidden.rng)(999)          # 隐藏随机状态
        after = encode_observation(observe(state), legal)
        for key in base:
            self.assertTrue(np.array_equal(base[key], after[key]),
                            f"隐藏状态改变了编码字段 {key}")


class TestRunRngSingleOwner(unittest.TestCase):
    """R5：Run 的持久随机状态**只有一个所有者**（``RunState.hidden.rng``）。"""

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_rng_is_the_same_object_before_and_after_restart(self):
        from sts2_sim.run import RunEnv
        env = RunEnv(seed=4, attempt_budget=3)
        env.reset()
        self.assertIs(env._rng, env.raw_state.hidden.rng)
        env.step(next(a for a in env._step_view().legal if a.kind == "choose_node"))
        env.restart_node()
        self.assertIs(env._rng, env.raw_state.hidden.rng,
                      "重开之后出现了第二个随机数所有者")

    def test_restart_reproduces_the_same_act_contents(self):
        """同一节点快照重开 → 跨幕生成的内容必须**逐节点相同**。

        修之前 seed=4 下 28 个节点的遭遇/卡牌奖励/金币与第一次全不同。
        """
        from sts2_sim.run import RunEnv
        env = RunEnv(seed=4, attempt_budget=3)
        env.reset()
        env.step(next(a for a in env._step_view().legal if a.kind == "choose_node"))

        def contents():
            return {k: (v.encounter, v.card_reward, v.gold)
                    for k, v in env.raw_state.hidden.node_contents.items()}

        env._begin_act(2)
        first = contents()
        env.restart_node()
        env._begin_act(2)
        second = contents()
        self.assertTrue(first, "第二幕没有生成任何节点内容")
        self.assertEqual(first, second)

    def test_rng_is_read_only(self):
        """``_rng`` 是**只读属性** —— 谁再想引入第二个所有者就立刻炸。"""
        from sts2_sim.run import RunEnv
        env = RunEnv(seed=1)
        env.reset()
        with self.assertRaises(AttributeError):
            env._rng = type(env._rng)(1)  # type: ignore[misc]

    def test_rng_before_reset_is_a_loud_error(self):
        """还没 ``reset()`` 时读 ``_rng`` 要**报错**，不能返回 ``None`` 让人误判。"""
        from sts2_sim.run import RunEnv
        env = RunEnv(seed=1)
        with self.assertRaises(RuntimeError):
            _ = env._rng


if __name__ == "__main__":
    unittest.main()
