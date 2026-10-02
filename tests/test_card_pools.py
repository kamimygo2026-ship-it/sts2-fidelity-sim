"""卡池与卡牌奖励的回归测试（``docs/09`` L5）。

这一组锁住一个**Run 层的重大保真缺陷**：奖励原先从 ``CARD_DB`` 全体里按稀有度抽，
于是**铁甲战士会被给到静默 / 缺陷 / 摄政王的牌**（实测确认），
而奖励选牌是整局最重要的决策 —— 池子错了等于在另一个游戏里训练。

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 铁甲战士拿到静默的牌 | 奖励池用的是全体卡牌，不是角色卡池 | `test_reward_stays_inside_the_character_pool` |
| 同一批奖励出现重复牌 | 没有 distinct 保证 | `test_reward_has_no_duplicates` |
| Boss 打完没有卡牌奖励 | `if kind != BOSS` 把 Boss 奖励整个抹掉了 | `test_boss_reward_is_always_rare` |
| 卡池建不出来 | 社区库 ``cards.json`` **没有 color 字段** | `test_pools_are_extracted_from_source` |
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


class TestCardPools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_pools_are_extracted_from_source(self):
        """卡池必须从**源码**抽：社区库没有 color 字段，建不出来。"""
        pools = json.loads((CONTENT / "card_pools.json").read_text(encoding="utf-8"))
        names = {p["pool"] for p in pools}
        for expected in ("IroncladCardPool", "SilentCardPool", "DefectCardPool",
                         "NecrobinderCardPool", "RegentCardPool",
                         "ColorlessCardPool"):
            with self.subTest(pool=expected):
                self.assertIn(expected, names)
        for pool in pools:
            self.assertGreater(len(pool["cards"]), 0)

    def test_character_pools_are_mutually_exclusive(self):
        """五个角色的卡池不该互相重叠 —— 重叠说明成员抽错了。"""
        from sts2_sim import content
        races = ["IroncladCardPool", "SilentCardPool", "DefectCardPool",
                 "NecrobinderCardPool", "RegentCardPool"]
        sets = {name: set(content.CARD_POOLS[name]) for name in races}
        for i, first in enumerate(races):
            for second in races[i + 1:]:
                with self.subTest(pair=(first, second)):
                    self.assertEqual(sets[first] & sets[second], set(),
                                     f"{first} 与 {second} 有重叠")

    def test_pool_ids_match_the_card_table(self):
        """池里的 id 必须能在卡牌表里找到（类名驼峰 → 下划线 id）。"""
        from sts2_sim import content
        missing = sorted({cid for pool in content.CARD_POOLS.values()
                          for cid in pool if cid not in content.CARD_DB})
        self.assertEqual(missing, [], f"池里有卡牌表里没有的 id：{missing[:8]}")

    def test_placeholder_content_has_a_pool(self):
        """占位内容也要有卡池 —— 否则 Run 层会直接报错（而不是退回错的路）。"""
        from sts2_sim import content
        content.load_builtin()
        self.assertTrue(content.CARD_POOLS)
        self.assertIn("IroncladCardPool", content.CARD_POOLS)


class TestCardReward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def _reward(self, kind: str, character: str, seed: int = 0, count: int = 3):
        from sts2_sim.rng import RngSet
        from sts2_sim.run import draw_card_reward
        return draw_card_reward(RngSet(seed), kind, count, character=character)

    def test_reward_stays_inside_the_character_pool(self):
        """⭐ 铁甲战士的奖励里**不能**出现别的角色的牌。"""
        from sts2_sim import content
        pool = set(content.CARD_POOLS["IroncladCardPool"])
        outside = []
        for seed in range(40):
            for cid in self._reward("monster", "ironclad", seed):
                if cid not in pool:
                    outside.append(cid)
        self.assertEqual(outside, [], f"奖励跑出了角色卡池：{sorted(set(outside))[:8]}")

    def test_silent_reward_does_not_leak_ironclad_cards(self):
        from sts2_sim import content
        ironclad = set(content.CARD_POOLS["IroncladCardPool"])
        for seed in range(20):
            for cid in self._reward("monster", "silent", seed):
                self.assertNotIn(cid, ironclad, "静默的奖励里混进了铁甲的牌")

    def test_reward_has_no_duplicates(self):
        for seed in range(50):
            picked = self._reward("monster", "ironclad", seed)
            self.assertEqual(len(set(picked)), len(picked),
                             f"seed={seed} 的奖励里有重复：{picked}")

    def test_boss_reward_is_always_rare(self):
        """``bossRareOdds = 1f``：Boss 奖励**必定**稀有，而且**存在**。"""
        from sts2_sim import content
        for seed in range(20):
            picked = self._reward("boss", "ironclad", seed)
            self.assertEqual(len(picked), 3, "Boss 也应有奖励")
            for cid in picked:
                self.assertEqual(content.CARD_DB[cid].rarity, "rare")

    def test_normal_reward_is_mostly_common(self):
        """普通战斗以普通牌为主（common 0.6 / uncommon 0.37 / rare 0.03）。"""
        from sts2_sim import content
        counts: dict[str, int] = {}
        for seed in range(200):
            for cid in self._reward("monster", "ironclad", seed):
                rarity = content.CARD_DB[cid].rarity
                counts[rarity] = counts.get(rarity, 0) + 1
        total = sum(counts.values())
        common_ratio = counts.get("common", 0) / total
        rare_ratio = counts.get("rare", 0) / total
        self.assertGreater(common_ratio, 0.5)
        self.assertLess(rare_ratio, 0.08, "稀有率明显偏高")

    def test_unknown_character_raises(self):
        from sts2_sim.rng import RngSet
        from sts2_sim.run import draw_card_reward
        with self.assertRaises(KeyError):
            draw_card_reward(RngSet(0), "monster", 3, character="nobody")

    def test_full_run_offers_only_character_cards(self):
        """整局跑下来，所有奖励都必须在角色卡池里。"""
        from sts2_sim import content
        from sts2_sim.rng import RngSet
        from sts2_sim.run import RunEnv
        from sts2_sim.runbot import RunBot
        pool = set(content.CARD_POOLS["IroncladCardPool"])
        bot = RunBot()
        for seed in range(6):
            env = RunEnv(seed=seed, attempt_budget=1, character="ironclad")
            step = env.reset()
            for _ in range(600):
                if step.phase in ("won", "lost"):
                    break
                # `raw_state` 仅供测试使用（策略路径不得读取）
                for cid in env.raw_state.room.card_reward:
                    self.assertIn(cid, pool, f"奖励 {cid} 不在铁甲池里")
                step, _r, done, _t, _i = env.step(bot.act(step))
                if done:
                    break


if __name__ == "__main__":
    unittest.main(verbosity=2)
