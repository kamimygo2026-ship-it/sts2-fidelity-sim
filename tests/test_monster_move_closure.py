"""出招表闭包的回归测试（``docs/13`` §7 的"闭包可执行"）。

两个**真实**问题由它锁住：

1. ``torch_head_amalgam`` 的状态机首个状态是 ``STRONG_TACKLE_MOVE``（伤害 26），
   而社区库给的招式列表是 ``tackle / tackle_2 / beam / tackle_3 / tackle_4`` ——
   **根本没有 26 点那一下**。运行到第 1 招就抛"状态机指向未知招式"，
   实测在整局 fuzz 的 seed=134 直接把 Run 层打断（而且是在训练跑到一半时才发生）。

2. 手牌上限（见 ``tests/test_hand_limit.py``）之外，这类"内容表里有洞、
   运行时才炸"的问题必须能在**加载期**被发现 —— 所以有 :func:`_enemy_move_gaps`
   与 :func:`_drop_encounters_with_gaps` 两层闸门。

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 整局在某只怪出场时崩 | 社区库招式列表缺项，状态机对不上 | `test_every_move_state_resolves_to_a_move` |
| 缺的招式数值凭空来 | 没有从 `move_values` 取源码数值 | `test_created_move_takes_its_value_from_source` |
| 带洞的怪照样进遭遇 | 加载期没有闭包校验 | `test_encounters_with_gaps_are_dropped` |
"""

from __future__ import annotations

import unittest
from pathlib import Path

from sts2_sim import content
from sts2_sim.content import ENEMY_DB

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"


class TestMoveClosure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CONTENT.exists():
            raise unittest.SkipTest(f"缺内容目录 {CONTENT}")
        from sts2_sim.featurize import configure_content
        configure_content(str(CONTENT))
        cls.addClassCleanup(cls._restore)

    @staticmethod
    def _restore():
        from sts2_sim.content import load_builtin
        from sts2_sim.featurize import rebuild_vocab
        load_builtin()
        rebuild_vocab()

    def test_every_move_state_resolves_to_a_move(self):
        """⭐ 全员闭包：任何一只怪的**每个** move 状态都必须能落到一条招式。"""
        gaps = content._enemy_move_gaps()
        self.assertEqual(gaps, {}, f"仍有出招闭包缺口：{gaps}")

    def test_the_known_reconciliation_is_present(self):
        """``torch_head_amalgam`` 的那条缺失招式必须被补出来（来自源码）。"""
        enemy = ENEMY_DB["torch_head_amalgam"]
        mids = {move.mid for move in enemy.moves}
        self.assertIn("strong_tackle_move", mids,
                      "状态机需要的那条招式没有被补出来")

    def test_created_move_takes_its_value_from_source(self):
        """补出来的招式，数值/意图必须来自 ``move_values``（不是猜的）。"""
        enemy = ENEMY_DB["torch_head_amalgam"]
        move = next(m for m in enemy.moves if m.mid == "strong_tackle_move")
        info = (enemy.ai_record.get("move_values") or {})["StrongTackleMove"]
        self.assertEqual(move.intent, "attack")
        self.assertEqual(move.value, int(info["normal"]))
        self.assertEqual(move.times, max(1, int(info.get("times") or 1)))

    def test_no_encounter_contains_a_monster_with_a_gap(self):
        """遭遇表里不许有"带洞的怪"（否则战斗一定崩）。"""
        gaps = set(content._enemy_move_gaps())
        for kind, table in content.ENCOUNTERS.items():
            for members in table:
                self.assertFalse(set(members) & gaps,
                                 f"{kind} 里的 {members} 含出招缺口怪")

    def test_state_mid_matching_accepts_all_three_naming_schemes(self):
        """状态 id / 去后缀 / 处理函数名 三种写法都要能对上。"""
        self.assertIn("claw_move", content._state_mids("CLAW_MOVE", "ClawMove"))
        self.assertIn("claw", content._state_mids("CLAW_MOVE", "ClawMove"))
        self.assertIn("clawmove", content._state_mids("CLAW_MOVE", "ClawMove"))


class TestGapDetectionHasTeeth(unittest.TestCase):
    """反向验证：缺口检测必须真的能报出问题（否则它只是恒真的装饰）。"""

    @classmethod
    def setUpClass(cls):
        if not CONTENT.exists():
            raise unittest.SkipTest(f"缺内容目录 {CONTENT}")
        from sts2_sim.featurize import configure_content
        configure_content(str(CONTENT))
        cls.addClassCleanup(cls._restore)

    @staticmethod
    def _restore():
        from sts2_sim.content import load_builtin
        from sts2_sim.featurize import rebuild_vocab
        load_builtin()
        rebuild_vocab()

    def test_gap_detector_flags_a_missing_move(self):
        from dataclasses import replace
        enemy = ENEMY_DB["torch_head_amalgam"]
        # 把补出来的那条拿掉 → 检测器必须重新报缺口
        stripped = replace(enemy, moves=tuple(
            move for move in enemy.moves if move.mid != "strong_tackle_move"))
        gaps = content._move_state_gaps(stripped)
        self.assertEqual([state_id for state_id, _h in gaps],
                         ["STRONG_TACKLE_MOVE"])

    def test_gap_detector_is_quiet_when_complete(self):
        self.assertEqual(content._move_state_gaps(ENEMY_DB["torch_head_amalgam"]), [])

    def test_encounters_with_gaps_are_dropped(self):
        """闭包闸门：含缺口怪的遭遇必须被摘掉并计数。"""
        eid = "torch_head_amalgam"
        original = {kind: tuple(table) for kind, table in content.ENCOUNTERS.items()}
        try:
            # 只留一个种类，计数才可断言（闸门会遍历全部种类）。
            content.ENCOUNTERS["monster"] = (("torch_head_amalgam",), ("axebot",))
            content.ENCOUNTERS["elite"] = ()
            content.ENCOUNTERS["boss"] = ()
            removed = content._drop_encounters_with_gaps({eid})
            self.assertEqual(removed, 1)
            self.assertEqual(content.ENCOUNTERS["monster"], (("axebot",),))
        finally:
            for kind, table in original.items():
                content.ENCOUNTERS[kind] = table

    def test_intent_map_only_covers_attacks(self):
        """意图类名映射表只映射攻击 —— 其余不许猜（猜错会错报公开信息）。"""
        self.assertEqual(set(content.MONSTER_INTENT_KINDS),
                         {"SingleAttackIntent", "MultiAttackIntent"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
