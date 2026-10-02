"""特征化层的反作弊测试。

``observe.py`` 的投影对了，不代表策略看到的就干净——**特征化是另一条可以漏水的管子**。
所以这里把同一套判据在编码结果上再验一遍。
"""

from __future__ import annotations

import ast
import pathlib
import unittest

import numpy as np

from sts2_sim import legal_actions, observe, reroll_hidden, start_combat
from sts2_sim.content import STARTING_DECK
from sts2_sim.core import Action
from sts2_sim.featurize import (
    CARD_VOCAB, MAX_ACTIONS, MAX_ENEMIES, MAX_HAND, MAX_TOKENS,
    batch_encode, encode_observation, shapes,
)

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent.parent / "sts2_sim"

#: 特征化模块禁止触及的标识符（与 bot.py 同一套门禁）
FORBIDDEN_IDENTIFIERS = frozenset({
    "Hidden", "RngSet", "CombatState", "Snapshot", "raw_state", "snapshot",
    "restore", "reroll_hidden", "master_seed", "draw_pile", "hidden",
})


def make_case(seed: int = 5, encounter=("jaw_worm", "red_louse"), deck=STARTING_DECK):
    state = start_combat(deck, encounter, seed=seed)
    obs = observe(state)
    return state, obs, legal_actions(state)


class TestEncodingContract(unittest.TestCase):
    def test_shapes_match_declared_constants(self):
        _state, obs, legal = make_case()
        encoded = encode_observation(obs, legal)
        for key, shape in shapes().items():
            self.assertEqual(encoded[key].shape, shape, f"{key} 形状不符")
        self.assertEqual(encoded["token_type"].shape[0], MAX_TOKENS)
        self.assertEqual(encoded["act_mask"].shape[0], MAX_ACTIONS)

    def test_token_mask_reflects_population(self):
        _state, obs, legal = make_case()
        encoded = encode_observation(obs, legal)
        # 全局 token + 手牌数 + 敌人数
        expected = 1 + len(obs.hand) + len(obs.enemies)
        self.assertEqual(int(encoded["token_mask"].sum()), expected)

    def test_action_count_matches_mask(self):
        _state, obs, legal = make_case()
        encoded = encode_observation(obs, legal)
        self.assertEqual(int(encoded["n_actions"]), len(legal))
        self.assertEqual(int(encoded["act_mask"].sum()), len(legal))

    def test_action_card_matches_hand_slot(self):
        _state, obs, legal = make_case()
        encoded = encode_observation(obs, legal)
        for i, action in enumerate(legal):
            if action.kind != "play_card":
                continue
            expected = CARD_VOCAB[obs.hand[action.hand_index].cid]
            self.assertEqual(int(encoded["act_card"][i]), expected)

    def test_end_turn_is_always_available(self):
        _state, obs, legal = make_case()
        kinds = [a.kind for a in legal]
        self.assertIn("end_turn", kinds)

    def test_encoding_is_deterministic(self):
        _state, obs, legal = make_case()
        a = encode_observation(obs, legal)
        b = encode_observation(obs, legal)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    def test_batch_encode_stacks(self):
        state_a, obs_a, legal_a = make_case(seed=1)
        _sa, obs_b, legal_b = make_case(seed=2)
        batch = batch_encode([(obs_a, legal_a), (obs_b, legal_b)])
        self.assertEqual(batch["token_num"].shape[0], 2)
        self.assertEqual(batch["act_mask"].shape[0], 2)

    def test_too_many_actions_is_loud(self):
        """动作数超限必须报错，不能静默截断（静默截断会悄悄改变策略语义）。"""
        _state, obs, legal = make_case()
        with self.assertRaises(ValueError):
            encode_observation(obs, list(legal) * (MAX_ACTIONS + 1))


class TestFeaturizerLeakage(unittest.TestCase):
    """⭐ 核心：隐藏信息怎么变，编码都必须逐位不变。"""

    def test_hidden_reroll_does_not_change_encoding(self):
        state, obs, legal = make_case(seed=11)
        before = encode_observation(obs, legal)

        reroll_hidden(state, 999_999)
        state.hidden.move_queue = ["bellow", "chomp", "thrash"]
        state.hidden.future_drops = ["relic:boss"]
        after = encode_observation(observe(state), legal_actions(state))

        for key in before:
            np.testing.assert_array_equal(before[key], after[key],
                                          err_msg=f"{key} 泄漏了隐藏信息")

    def test_pile_order_does_not_change_encoding(self):
        """抽牌堆 / 弃牌堆的**顺序**不得进入编码（只允许多重集）。"""
        state, obs, legal = make_case(seed=13)
        before = encode_observation(obs, legal)

        state.draw_pile.reverse()
        state.discard.reverse()
        after = encode_observation(observe(state), legal_actions(state))

        np.testing.assert_array_equal(before["pile_counts"], after["pile_counts"])
        np.testing.assert_array_equal(before["token_num"], after["token_num"])

    def test_pile_counts_are_a_multiset(self):
        _state, obs, legal = make_case(seed=3)
        encoded = encode_observation(obs, legal)
        total = encoded["pile_counts"].sum()
        hand = len(obs.hand)
        self.assertAlmostEqual(float(total) + hand, float(len(STARTING_DECK)))

    def test_hand_order_is_preserved_but_piles_are_not(self):
        """手牌顺序是可见信息（人类按顺序读牌），必须保留；牌堆顺序必须丢掉。

        这条测试同时锁住两个方向——只测一个方向都可能掩盖错误。
        """
        _state, obs, legal = make_case(seed=17)
        encoded = encode_observation(obs, legal)
        slots = encoded["token_num"][1:1 + len(obs.hand), 14]
        expected = np.arange(len(obs.hand), dtype=np.float32) / 10.0
        np.testing.assert_allclose(slots, expected)


class TestFeaturizerModuleIsolation(unittest.TestCase):
    def test_no_forbidden_identifiers(self):
        path = PACKAGE_DIR / "featurize.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        used: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.ImportFrom):
                used.update(alias.name for alias in node.names)
        leaked = used & FORBIDDEN_IDENTIFIERS
        # observe.py 是唯一投影出口，特征化层不允许直接读 state
        self.assertFalse(leaked, f"featurize.py 触及了隐藏/内部标识符：{leaked}")

    def test_only_imports_from_public_modules(self):
        """只约束**包内**依赖：特征化层只允许 import 公开的数据/协议模块。

        允许的四个各有理由：

        * ``content`` —— 卡牌/遗物/药水的**静态定义**（L1 固定知识）
        * ``core``    —— 只取 ``Action`` 这个公开动作类型
        * ``observe`` —— 唯一投影出口
        * ``orbs``    —— 充能球的**静态定义表**（``ORB_DEFS``），同样是 L1；
          审计 F04 要求把球编进观测，所以词表必须能从引擎注册表构建。
        * ``powers``  —— 能力的**静态注册表**（``powers.RULES``），同样只提供
          "哪些能力 id 存在"这一层 L1 信息（外部复核 R4 要求把能力编成实体 token）。
          它**不持有**任何战斗状态：运行时状态在 ``CombatState`` 里，由 ``core`` 持有。

        ⚠️ 这几个模块都**不持有隐藏状态**；真正要挡住的是 ``hidden`` / ``rng`` /
        牌堆顺序这类东西，由上一条 AST 标识符门禁负责。

        标准库与第三方（typing、numpy）不受限。
        """
        path = PACKAGE_DIR / "featurize.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        allowed_modules = {"content", "core", "observe", "orbs", "powers"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = (node.module or "").lstrip(".")
            is_internal = node.level > 0 or module.startswith("sts2_sim")
            if not is_internal:
                continue
            self.assertIn(module, allowed_modules,
                          f"featurize.py 不应依赖包内模块 {module}")


class TestTokenBudget(unittest.TestCase):
    def test_capacity_overflow_is_loud_not_silent(self):
        """⭐ 审计 F09：超出容量必须**报错**，不能静默截断。

        静默截断让策略看到的世界**比真实世界小**，而且是系统性的：
        手牌 12 张时它只看到前 10 张，"少打了一张牌"没有任何信号。
        """
        import dataclasses
        _state, obs, legal = make_case(seed=19)
        extra = obs.hand[:1]
        broken = dataclasses.replace(obs, hand=obs.hand + extra * (MAX_HAND + 1))
        with self.assertRaises(ValueError):
            encode_observation(broken, legal)

        broken_enemies = dataclasses.replace(
            obs, enemies=obs.enemies + obs.enemies * (MAX_ENEMIES + 1))
        with self.assertRaises(ValueError):
            encode_observation(broken_enemies, legal)


class TestPublicStateIsEncoded(unittest.TestCase):
    """⭐ 审计 F04：已经实现的机制必须**进入模型输入**。

    实测旧实现里"只给公开观察增加 9 层中毒，最终网络输入完全不变"——
    环境按这些机制结算，策略却没有条件选择不同动作。
    """

    def _encode_with(self, **changes):
        import dataclasses
        _state, obs, legal = make_case(seed=23)
        return encode_observation(dataclasses.replace(obs, **changes), legal), obs, legal

    def test_relics_change_the_encoding(self):
        """遗物必须**占 token 位**（审计 F04：以前是空占位）。"""
        import dataclasses
        from sts2_sim.featurize import TOKEN_RELIC
        base, obs, legal = self._encode_with()
        with_relic = encode_observation(
            dataclasses.replace(obs, relics=("burning_blood", "vajra")), legal)
        self.assertNotEqual(float(base["token_mask"].sum()),
                            float(with_relic["token_mask"].sum()))
        self.assertIn(TOKEN_RELIC, set(int(t) for t in with_relic["token_type"]))

    def test_relic_identity_uses_its_own_vocab(self):
        """遗物身份走独立词表：与卡牌/敌人下标空间**不能混用**。"""
        from sts2_sim.featurize import RELIC_VOCAB
        if not RELIC_VOCAB:
            self.skipTest("内置占位内容没有遗物表（load_builtin 会清空 RELICS）")
        import dataclasses
        from sts2_sim.featurize import TOKEN_RELIC
        _state, obs, legal = make_case(seed=23)
        rid = sorted(RELIC_VOCAB)[0]
        encoded = encode_observation(dataclasses.replace(obs, relics=(rid,)), legal)
        slot = int(np.nonzero(encoded["token_type"] == TOKEN_RELIC)[0][0])
        self.assertEqual(int(encoded["token_entity"][slot]), RELIC_VOCAB[rid])

    def test_potions_change_the_encoding(self):
        import dataclasses
        base, obs, legal = self._encode_with()
        with_potion = encode_observation(
            dataclasses.replace(obs, potions=((0, "fire_potion"),)), legal)
        self.assertFalse(np.array_equal(base["token_mask"], with_potion["token_mask"]))

    def test_stars_change_the_global_features(self):
        import dataclasses
        base, obs, legal = self._encode_with()
        with_stars = encode_observation(dataclasses.replace(obs, stars=7), legal)
        self.assertNotEqual(float(base["token_num"][0, 23]),
                            float(with_stars["token_num"][0, 23]))

    def test_upgraded_cards_in_a_pile_are_distinguishable(self):
        """牌堆编码以前忽略升级标记（审计 F04）。"""
        import dataclasses
        from sts2_sim.observe import BagView
        _base, obs, legal = self._encode_with()
        plain = BagView((("strike", False, 3),))
        upgraded = BagView((("strike", True, 3),))
        a = encode_observation(dataclasses.replace(obs, draw_bag=plain), legal)
        b = encode_observation(dataclasses.replace(obs, draw_bag=upgraded), legal)
        np.testing.assert_array_equal(a["pile_counts"], b["pile_counts"])
        self.assertFalse(np.array_equal(a["pile_upgraded"], b["pile_upgraded"]),
                         "升级过的牌必须在编码上与未升级的区分开")

    def test_pile_counts_still_totals_everything(self):
        """``pile_counts`` 仍是"全部"，``pile_upgraded`` 是其中一个子集。"""
        import dataclasses
        from sts2_sim.observe import BagView
        _base, obs, legal = self._encode_with()
        mixed = BagView((("strike", False, 2), ("strike", True, 3)))
        encoded = encode_observation(dataclasses.replace(obs, draw_bag=mixed), legal)
        self.assertAlmostEqual(float(encoded["pile_counts"][0].sum()), 5.0)
        self.assertAlmostEqual(float(encoded["pile_upgraded"][0].sum()), 3.0)

    def test_selection_candidates_come_from_the_right_pile(self):
        """⭐ 审计 F09：`select_card` 的下标是**候选**下标，不是手牌下标。"""
        import dataclasses
        from sts2_sim.observe import CardView, SelectionView
        _state, obs, legal = make_case(seed=29)
        discard_card = obs.hand[0].cid          # 保证在词表里
        selection = SelectionView(purpose="discard", source_pile="discard",
                                  remaining=1,
                                  candidates=(CardView(discard_card, False, 1, True),))
        obs = dataclasses.replace(obs, selection=selection,
                                  selecting_purpose="discard",
                                  selecting_remaining=1)
        encoded = encode_observation(obs, [Action("select_card", 0)])
        self.assertEqual(int(encoded["act_card"][0]), CARD_VOCAB[discard_card])
        self.assertGreater(float(encoded["token_num"][0, 26]), 0.0,
                           "选牌来源牌堆必须被编码")


if __name__ == "__main__":
    unittest.main(verbosity=2)
