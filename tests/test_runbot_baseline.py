"""Run 层规则基线的回归测试（``docs/04`` §4.1）。

规则 bot 不是"跑得通就行"的装饰 —— 它是**判断策略有没有进步的那个基线**。
``docs/07`` §7.1 的失败实验教训就是：没有可解释基线的 RL 无法判断到底学到了什么。

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 换真实内容后 bot 永远选第一张 | 评分表写死了内置占位卡 id | `test_scoring_is_content_agnostic` |
| 基线与随机策略打平 | bot 没有区分力 | `test_rule_bot_beats_a_random_policy` |
"""

from __future__ import annotations

import random
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


class TestScoringIsContentAgnostic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_scoring_is_content_agnostic(self):
        """真实内容的卡 id（``strike_ironclad`` 等）必须拿到**非默认**分数。"""
        from sts2_sim.content import CARD_DB
        from sts2_sim.runbot import card_score

        attack = next(cid for cid, c in CARD_DB.items()
                      if c.rarity == "basic" and c.card_type == "attack")
        self.assertGreater(card_score(attack), 0.0,
                           f"{attack} 应当拿到正的攻击分")

    def test_curses_are_strongly_negative(self):
        from sts2_sim.content import CARD_DB
        from sts2_sim.runbot import card_score
        curses = [cid for cid, c in CARD_DB.items() if c.rarity == "curse"]
        if not curses:
            self.skipTest("当前内容里没有诅咒牌")
        for cid in curses[:5]:
            self.assertLess(card_score(cid), 0.0, cid)

    def test_reverse_verification_a_hardcoded_table_would_default(self):
        """反向验证：写死内置 id 的表对真实内容 id 只能返回默认值。

        这正是旧实现的形态 —— 换内容之后每张牌分数一样，
        bot 退化成"永远选第一张"，而且没有任何信号。
        """
        from sts2_sim.content import CARD_DB
        from sts2_sim.runbot import card_score
        builtin_only = {"strike", "defend", "bash", "anger", "cleave"}
        real_ids = [cid for cid in CARD_DB if cid not in builtin_only]
        self.assertTrue(real_ids, "真实内容里应当有非内置 id 的卡")
        scores = {card_score(cid) for cid in real_ids}
        self.assertGreater(len(scores), 1,
                           "所有真实卡的分数都一样 → 评分没有真的按内容算")


class TestRuleBotHasDiscriminatingPower(unittest.TestCase):
    """基线的价值在于"能分辨好坏策略"（``docs/README`` 铁律 6）。"""

    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def _run(self, chooser, seeds: int) -> float:
        from sts2_sim.run import PHASE_LOST, PHASE_WON, RunEnv
        floors = []
        for seed in range(seeds):
            env = RunEnv(seed=1000 + seed, attempt_budget=1, character="ironclad")
            step = env.reset()
            for _ in range(400):
                if step.phase in (PHASE_WON, PHASE_LOST):
                    break
                if not step.legal:
                    break
                step, _r, done, _t, _i = env.step(chooser(step))
                if done:
                    break
            floors.append(env.raw_state.floor)
        return sum(floors) / max(1, len(floors))

    def test_rule_bot_beats_a_random_policy(self):
        from sts2_sim.runbot import RunBot

        bot = RunBot()
        rng = random.Random(0)

        def smart(step):
            return bot.act(step)

        def dumb(step):
            return rng.choice(step.legal)

        bot_floor = self._run(smart, seeds=12)
        random_floor = self._run(dumb, seeds=12)
        self.assertGreater(bot_floor, random_floor,
                           f"规则 bot 平均到达 {bot_floor:.1f} 层，"
                           f"随机策略 {random_floor:.1f} 层 —— 基线没有区分力")


if __name__ == "__main__":
    unittest.main(verbosity=2)
