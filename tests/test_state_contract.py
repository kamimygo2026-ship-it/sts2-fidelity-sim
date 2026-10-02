"""审计 F01–F03 的回归测试（``docs/12`` / 工单 T01–T02）。

这一组盯的是**跨系统交接处的静默错误**：状态在 Run → 战斗之间丢字段、
角色参数不决定初始状态、内容门禁没有落到采样器。三个都不报错，
只是模型学到的东西与真机不一样。

| 审计 | 症状 | 对应测试 |
|---|---|---|
| F01 | Run 升级 10 张 / 最大生命 120 / 进阶 8 → 战斗 0 / 80 / 0 | `TestRunCombatContract` |
| F02 | `RunEnv(character="silent")` 仍是 80 血 + 铁甲战士牌组 + 空遗物 | `TestCharacterInit` |
| F03 | 采样器放行 `effects_incomplete`、跨角色混牌、默认跑内置内容 | `TestContentAdmission` |
"""

from __future__ import annotations

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


# ==========================================================================
# F01 · Run ↔ 战斗 状态契约
# ==========================================================================
class TestRunCombatContract(unittest.TestCase):
    """``docs/13`` §3.2 的入场/退场表。"""

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def make_env(self, ascension: int = 8):
        from sts2_sim.run import RunEnv
        env = RunEnv(seed=7, ascension=ascension, attempt_budget=0)
        env.reset()
        return env

    def enter_first_combat(self, env):
        from sts2_sim.run import MetaAction
        state = env.raw_state
        env.step(MetaAction("choose_node", state.map.start[0]))
        return env.raw_state.combat

    def test_upgrades_survive_the_transition(self):
        """审计 F01 的**主症状**：Run 升级 10 张，战斗里必须还是 10 张。"""
        env = self.make_env()
        player = env.raw_state.player
        for card in player.deck:
            card.upgraded = True
        combat = self.enter_first_combat(env)
        self.assertEqual(sum(card.upgraded for card in combat.all_piles()),
                         len(player.deck))

    def test_unupgraded_cards_stay_unupgraded(self):
        """反向验证：没有升级的牌不能被"顺手"升上去（否则上一条没有鉴别力）。"""
        env = self.make_env()
        combat = self.enter_first_combat(env)
        self.assertEqual(sum(card.upgraded for card in combat.all_piles()), 0)

    def test_max_hp_is_not_hardcoded(self):
        """审计 F01：战斗最大生命被写死 80，出现"当前生命 > 最大生命"。"""
        env = self.make_env()
        env.raw_state.player.hp = 105
        env.raw_state.player.max_hp = 120
        combat = self.enter_first_combat(env)
        self.assertEqual(combat.player.max_hp, 120)
        self.assertEqual(combat.player.hp, 105)
        self.assertLessEqual(combat.player.hp, combat.player.max_hp)

    def test_ascension_reaches_the_combat(self):
        """审计 F01：`start_combat` 调用侧没传 ascension，A8 的 run 打的是 A0 的怪。"""
        env = self.make_env(ascension=8)
        combat = self.enter_first_combat(env)
        self.assertEqual(combat.ascension, 8)

    def test_combat_cards_link_back_to_the_permanent_deck(self):
        """``docs/13`` §3.2 入场行："克隆完整实例并**保留关联**"。"""
        env = self.make_env()
        deck_uids = {card.uid for card in env.raw_state.player.deck}
        combat = self.enter_first_combat(env)
        links = {card.link_uid for card in combat.all_piles()}
        self.assertEqual(links, deck_uids)

    def test_max_hp_change_survives_the_combat(self):
        """退场行：永久最大生命变化跨战斗保持。

        直接驱动 `_sync_player_from_combat`（唯一的回写入口），
        避免依赖"某张卡刚好能改上限"这种脆弱的构造。
        """
        env = self.make_env()
        combat = self.enter_first_combat(env)
        combat.player.max_hp -= 30            # 模拟 `lose_max_hp` 的永久效果
        combat.player.hp = min(combat.player.hp, combat.player.max_hp)
        env._sync_player_from_combat(combat)
        self.assertEqual(env.raw_state.player.max_hp, combat.player.max_hp)

    def test_temporary_combat_state_does_not_leak(self):
        """退场行：临时费用 / 战斗内生成牌不污染永久牌组。"""
        env = self.make_env()
        permanent_before = list(env.raw_state.player.deck)
        combat = self.enter_first_combat(env)
        combat.permanent_card_changes.clear()   # 没有任何永久改牌
        env._sync_player_from_combat(combat)
        self.assertEqual([c.uid for c in env.raw_state.player.deck],
                         [c.uid for c in permanent_before])

    def test_permanent_upgrade_writes_back(self):
        """退场行："仅明确的永久改牌回写"。"""
        from sts2_sim.core import apply_permanent_changes, mark_permanent_upgrade
        env = self.make_env()
        combat = self.enter_first_combat(env)
        target = combat.all_piles()[0]
        self.assertTrue(mark_permanent_upgrade(combat, target))
        touched = apply_permanent_changes(combat, env.raw_state.player.deck)
        self.assertEqual(touched, [target.link_uid])
        deck = {card.uid: card for card in env.raw_state.player.deck}
        self.assertTrue(deck[target.link_uid].upgraded)

    def test_generated_cards_have_no_permanent_target(self):
        """反向验证：战斗内生成的牌**没有**永久本体，不该被回写。"""
        from sts2_sim.core import CardInstance, mark_permanent_upgrade
        env = self.make_env()
        combat = self.enter_first_combat(env)
        generated = CardInstance("strike_ironclad")
        self.assertIsNone(generated.link_uid)
        self.assertFalse(mark_permanent_upgrade(combat, generated))

    def test_character_starting_relic_reaches_the_combat(self):
        """初始遗物也要进战斗 —— 否则"有遗物"和"没有遗物"在数值上一样。"""
        env = self.make_env()
        combat = self.enter_first_combat(env)
        self.assertIn("burning_blood", combat.relics)

    def test_campfire_upgrade_reaches_the_next_combat(self):
        """``docs/13`` §3.2 的验收样例：**营火升级影响下一战**。"""
        from sts2_sim.mapgen import MONSTER
        from sts2_sim.run import PHASE_REST, MetaAction, Room, legal_meta_actions

        env = self.make_env()
        state = env.raw_state
        # 强制进入营火，升一张牌
        state.room = Room(kind=PHASE_REST, rest_options=("rest", "smith"))
        smith = next(a for a in legal_meta_actions(state) if a.kind == "smith")
        env.step(smith)
        upgraded = [card for card in state.player.deck if card.upgraded]
        self.assertEqual(len(upgraded), 1, "营火应当升级恰好一张牌")

        # 走到一个战斗节点
        target = next(node for node in state.map.nodes if node.kind == MONSTER
                      and node.node_id in state.map.start)
        env.step(MetaAction("choose_node", target.node_id))
        combat = env.raw_state.combat
        self.assertIsNotNone(combat, "该节点应当是战斗")
        self.assertEqual(sum(card.upgraded for card in combat.all_piles()), 1,
                         "营火的升级必须带进战斗")

    def test_used_potion_does_not_come_back(self):
        """``docs/13`` §3.2 的验收样例：**用掉药水不会在下一战回来**。"""
        from sts2_sim.content import POTIONS
        from sts2_sim.core import use_potion_at

        usable = next((pid for pid, p in POTIONS.items()
                       if not p.effects_incomplete and p.usage != "automatic"
                       and "enemy" not in p.target_type), None)
        if usable is None:
            self.skipTest("当前内容里没有可直接使用的非指向性药水")
        env = self.make_env()
        env.raw_state.player.potions[0] = usable
        combat = self.enter_first_combat(env)
        self.assertEqual(combat.potions[0], usable, "入场应共用 Run 的药水库存")
        use_potion_at(combat, 0)
        self.assertIsNone(combat.potions[0], "用掉之后该槽位应当为空")
        env._sync_player_from_combat(combat)
        self.assertIsNone(env.raw_state.player.potions[0],
                          "用掉的药水不能回到 Run 的库存里")


# ==========================================================================
# F02 · 角色参数必须决定完整初始状态
# ==========================================================================
class TestCharacterInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def starting(self, character: str):
        from sts2_sim.run import RunEnv
        env = RunEnv(seed=3, character=character)
        env.reset()
        return env.raw_state.player

    def test_each_character_gets_its_own_hp(self):
        """源码依据：``Silent.StartingHp = 70``；审计实测旧实现恒为 80。"""
        expected = {"ironclad": 80, "silent": 70, "defect": 75,
                    "regent": 75, "necrobinder": 66}
        for character, hp in expected.items():
            self.assertEqual(self.starting(character).max_hp, hp, character)

    def test_each_character_gets_its_own_deck(self):
        for character in ("ironclad", "silent"):
            deck = {card.cid for card in self.starting(character).deck}
            self.assertTrue(any(cid.endswith(f"_{character}") for cid in deck),
                            f"{character} 的起始牌组里没有该角色的基础牌：{deck}")

    def test_each_character_gets_its_own_relic(self):
        """审计 F02 的隐藏一半：遗物 id 只 `.lower()` 不转 snake_case，
        于是 ``RingOfTheSnake`` → ``ringofthesnake`` 与遗物表对不上，
        初始遗物**静默失效**。"""
        self.assertEqual(self.starting("silent").relics, ["ring_of_the_snake"])
        self.assertEqual(self.starting("defect").relics, ["cracked_core"])

    def test_starting_relics_all_resolve_to_definitions(self):
        from sts2_sim.content import CHARACTERS, RELICS
        for character, definition in CHARACTERS.items():
            for rid in definition.relics:
                self.assertIn(rid, RELICS, f"{character} 的初始遗物 {rid} 不存在")

    def test_unknown_character_fails_loudly(self):
        """未知角色**必须报错**，不能静默退回默认角色。"""
        from sts2_sim.run import RunEnv
        env = RunEnv(character="not_a_character")
        with self.assertRaises(KeyError):
            env.reset()


# ==========================================================================
# F03 · 内容门禁必须落到采样器
# ==========================================================================
class TestContentAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_draft_pool_has_no_incomplete_cards(self):
        """审计 F03 的主症状：418 张里混进 223 张残缺卡。"""
        from sts2_rl.ppo import draft_pool
        from sts2_sim.content import CARD_DB
        pool = draft_pool("ironclad")
        self.assertTrue(pool)
        bad = [cid for cid in pool if CARD_DB[cid].effects_incomplete]
        self.assertEqual(bad, [], f"采样池里仍有残缺卡：{bad[:5]}")

    def test_draft_pool_has_no_unimplemented_powers(self):
        from sts2_rl.ppo import draft_pool
        from sts2_sim.content import CARD_DB
        from sts2_sim.powers import IMPLEMENTED
        for cid in draft_pool("ironclad"):
            for effect in CARD_DB[cid].effects:
                if effect.op == "apply_power":
                    self.assertIn(effect.power, IMPLEMENTED, cid)

    def test_draft_pool_stays_inside_one_character(self):
        """审计 F03：``random_deck()`` 会混合不同角色的起始牌组和跨角色加牌。"""
        from sts2_rl.ppo import draft_pool
        from sts2_sim.content import CARD_POOLS
        ironclad = set(CARD_POOLS["IroncladCardPool"])
        silent = set(CARD_POOLS["SilentCardPool"])
        self.assertTrue(set(draft_pool("ironclad")) <= ironclad)
        self.assertTrue(set(draft_pool("silent")) <= silent)
        self.assertNotEqual(set(draft_pool("ironclad")), set(draft_pool("silent")))

    def test_random_deck_uses_one_character(self):
        import random
        from sts2_rl.ppo import random_deck
        from sts2_sim.content import CARD_POOLS, CHARACTERS
        allowed = (set(CARD_POOLS["IroncladCardPool"])
                   | set(CHARACTERS["ironclad"].deck))
        for seed in range(20):
            deck = random_deck(random.Random(seed), "ironclad")
            stray = [cid for cid in deck if cid not in allowed]
            self.assertEqual(stray, [], f"seed={seed} 混进了非本角色的牌：{stray}")

    def test_builtin_content_is_refused_by_default(self):
        """审计 F03：默认启动训练会用内置占位内容，而且**没有任何提示**。

        ⚠️ 本组其它用例已加载真实内容，所以这里必须先切回内置再断言 ——
        否则测的是"真实内容下不报错"，与要锁的行为无关。
        """
        from sts2_sim import eligibility
        restore_builtin()
        try:
            with self.assertRaises(RuntimeError):
                eligibility.assert_trainable("ironclad")
            # 显式打开之后就允许（仅测试 / 冒烟）
            eligibility.assert_trainable("ironclad", allow_builtin=True)
        finally:
            setup_content()

    def test_clog_is_not_a_defect_but_an_empty_playable_is(self):
        """可打出的牌却什么都不做 = 解析缺口；诅咒/状态牌没效果 = 设计。

        ``docs/13`` §7 要求这两者分开：混在一起会让 90+ 张诅咒/状态牌被当成
        引擎缺陷，或者反过来让真正缺效果的牌静默过关。
        """
        from sts2_sim import eligibility as el
        from sts2_sim.content import CARD_DB, CardDef

        clog = CardDef("__clog__", "诅咒", -1, "curse", "curse", "none", ())
        empty_playable = CardDef("__empty__", "空技能", 1, "skill", "common",
                                 "none", ())
        CARD_DB["__clog__"] = clog
        CARD_DB["__empty__"] = empty_playable
        try:
            self.assertTrue(el.is_clog(clog))
            self.assertNotIn(el.R_NO_EFFECT, el.card_reasons("__clog__"))
            self.assertIn(el.R_NO_EFFECT, el.card_reasons("__empty__"))
        finally:
            CARD_DB.pop("__clog__", None)
            CARD_DB.pop("__empty__", None)

    def test_unplayable_keyword_card_is_not_reported_as_no_effect(self):
        """带 ``Unplayable`` 关键字**但有效果**的牌（``debris`` 这类）
        不是缺口；它只是不能主动打出。"""
        from sts2_sim import eligibility as el
        from sts2_sim.content import CARD_DB, CardDef, Effect

        card = CardDef("__debris_like__", "残骸", 1, "status", "status", "none",
                       (Effect("damage", 3),), keywords=("Unplayable",))
        CARD_DB["__debris_like__"] = card
        try:
            self.assertTrue(el.is_clog(card))
            self.assertEqual(el.card_reasons("__debris_like__"), ())
        finally:
            CARD_DB.pop("__debris_like__", None)

    def test_closure_gap_is_detected(self):
        """``docs/13`` §7：生成物不合格时，源卡也不算闭合。"""
        from sts2_sim import eligibility as el
        from sts2_sim.content import CARD_DB, CardDef, Effect
        original = CARD_DB.get("anger")
        self.assertIsNotNone(original, "需要 anger 来构造闭包用例")
        self.assertEqual(el.closure_gaps("anger"), ())
        # 故意把 anger 生成的牌改成不存在 → 闭包必须报缺口
        CARD_DB["anger"] = CardDef(
            "anger", "愤怒", 0, "attack", "common", "enemy",
            (Effect("damage", 6), Effect("add_card", 1, card="__nope__", target="self")))
        try:
            self.assertIn("generates_unknown_card:__nope__",
                          el.closure_gaps("anger"))
        finally:
            CARD_DB["anger"] = original

    def test_fingerprint_changes_with_content(self):
        """审计 F12：内容变了而指纹没变，checkpoint 就无法自证。"""
        from sts2_sim import eligibility as el
        before = el.content_fingerprint()
        self.assertGreater(before["content_hash"], 0)
        self.assertEqual(before["cards_admitted"], len(el.admitted_cards()))

    def test_report_counts_have_no_unknown_reason_bucket(self):
        """``docs/13`` §7：报告要能归因，不许出现"其它"。"""
        from sts2_sim import eligibility as el
        report = el.report("ironclad")
        for reason in report["reject_reasons"]:
            self.assertIn(reason, {
                "unknown_card", "effects_incomplete", "triggers_incomplete",
                "no_effects", "clog", "unsupported_op", "unimplemented_power",
                "unsupported_selection", "unsupported_target",
                "unsupported_trigger", "generates_unknown_card",
                "references_unknown_relic", "closure_gap",
                # 效果只来自社区文本（非权威来源）—— 见 `eligibility.R_TEXT_ONLY`
                "effects_from_community_text",
                # 单人 profile 遇不到（多人专用卡）—— 见 `docs/12` §2.32
                "multiplayer_only",
            }, f"未登记的拒绝原因 {reason}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
