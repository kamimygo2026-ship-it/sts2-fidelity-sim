"""药水与 ``ValueProp`` 的回归测试（``docs/09`` L5）。

药水与卡牌同构，所以复用同一套效果 DSL；但药水（以及充能球）的伤害/格挡
带 ``ValueProp.Unpowered`` —— **不受力量/虚弱/易伤，也不受敏捷/脆弱**，
但**仍然可以被格挡**。忽略这一条会让数值静默偏掉：

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 火焰药水打出 25 而不是 20 | 忽略了 ``ValueProp.Unpowered``，把玩家的力量算了进去 | `test_potion_damage_is_unpowered` |
| 格挡药水被敏捷/脆弱改变 | 同上（格挡那一侧） | `test_potion_block_is_unpowered` |
| 残缺药水被当成完整药水用 | 没有拒绝机制 | `test_incomplete_potion_is_refused` |
| 药水 id 与社区库一个都对不上 | 类名驼峰直接用 ``.lower()``，而 id 是下划线 | `test_potion_ids_match_the_codex` |
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


class TestPotionData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def test_potion_ids_match_the_codex(self):
        """药水 id 必须与社区库对齐：类名是驼峰、id 是下划线。

        直接 ``.lower()`` 会让 64 个药水**一个都对不上** —— 而且不报错，
        只是全都变成"社区库没有的新药水"。
        """
        source = json.loads((CONTENT / "potions_source.json").read_text(encoding="utf-8"))
        codex = {p["id"].lower() for p in
                 json.loads((CONTENT / "potions.json").read_text(encoding="utf-8"))}
        ours = {p["pid"] for p in source}
        matched = ours & codex
        self.assertGreater(len(matched), 55,
                           f"只对上 {len(matched)} 个，id 命名八成又错了")

    def test_potions_are_loaded(self):
        from sts2_sim import content
        self.assertGreaterEqual(len(content.POTIONS), 55)


class TestPotionRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()
        cls.addClassCleanup(restore_builtin)

    def _combat(self):
        from sts2_sim.core import start_combat
        return start_combat(self.deck, ("nibbit",), seed=1)

    def test_block_potion_grants_its_block(self):
        from sts2_sim.core import use_potion
        state = self._combat()
        state.player.block = 0
        use_potion(state, "block_potion")
        self.assertEqual(state.player.block, 12)

    def test_dexterity_potion_grants_the_power(self):
        from sts2_sim.core import use_potion
        state = self._combat()
        use_potion(state, "dexterity_potion")
        self.assertEqual(state.player.power("dexterity"), 2)

    def test_potion_damage_is_unpowered(self):
        """``DamageVar(20m, ValueProp.Unpowered)``：**不受力量/易伤影响**。

        实测忽略它会打出 25 而不是 20（玩家的力量被算进去了）。
        """
        from sts2_sim.core import use_potion
        state = self._combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.block = 0
        state.player.add_power("strength", 5)
        enemy.add_power("vulnerable", 3)
        use_potion(state, "fire_potion", target_index=0)
        self.assertEqual(100 - enemy.hp, 20, "药水伤害不受力量与易伤影响")

    def test_potion_damage_is_still_blockable(self):
        """``Unpowered`` ≠ ``Unblockable``：药水伤害**仍然可以被格挡吃掉**。"""
        from sts2_sim.core import use_potion
        state = self._combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.block = 50
        use_potion(state, "fire_potion", target_index=0)
        self.assertEqual(enemy.hp, 100)
        self.assertEqual(enemy.block, 30)

    def test_potion_block_is_unpowered(self):
        """格挡那一侧同理：药水格挡不吃敏捷、也不被脆弱打折。"""
        from sts2_sim.core import use_potion
        state = self._combat()
        state.player.block = 0
        state.player.add_power("dexterity", 5)
        state.player.add_power("frail", 3)
        use_potion(state, "block_potion")
        self.assertEqual(state.player.block, 12)

    def test_gigantification_potion_triples_the_next_attack(self):
        """``GigantificationPotion`` → 1 层 ``gigantification``：下一张攻击牌 ×3。

        ⚠️ 这个药水在 ``content_status`` 里**一直算"可用"**（效果解析得出来），
        但 1 层 ``gigantification`` 在能力没实现时是**打出去什么都不发生** ——
        "可用 45/65"这个口径看不见这种空转。所以这里按**真实数值**钉住。
        """
        from sts2_sim.core import Action, CardInstance, step, use_potion
        state = self._combat()
        enemy = state.enemies[0]
        use_potion(state, "gigantification_potion")
        self.assertEqual(state.player.power("gigantification"), 1)
        card = CardInstance("strike_ironclad")
        state.hand.append(card)
        state.energy = 9
        before = enemy.hp
        step(state, Action("play_card", state.hand.index(card), 0))
        self.assertEqual(enemy.hp, before - 18, "6 × 3")
        self.assertEqual(state.player.power("gigantification"), 0, "用掉 1 层")

    def test_card_damage_is_still_powered(self):
        """对照组：**普通卡牌**的伤害必须继续吃力量与易伤。

        把 ``Unpowered`` 一刀切到所有效果上就废掉了力量体系。
        """
        from sts2_sim.core import Action, CardInstance, step
        state = self._combat()
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.block = 0
        state.player.add_power("strength", 5)
        state.hand = [CardInstance("strike_ironclad")]
        state.energy = 3
        step(state, Action("play_card", 0, 0))
        self.assertEqual(100 - enemy.hp, 11, "6 + 力量 5")

    def test_incomplete_potion_is_refused(self):
        """效果残缺的药水**不能用** —— 用一瓶"少了一半效果"的药水
        等于给模型一个错误的强度，比不让用危险得多。"""
        from sts2_sim import content
        from sts2_sim.core import use_potion
        broken = [p.pid for p in content.POTIONS.values() if p.effects_incomplete]
        self.assertTrue(broken, "应当有残缺药水")
        state = self._combat()
        with self.assertRaises(ValueError):
            use_potion(state, broken[0])

    def test_unknown_potion_raises(self):
        from sts2_sim.core import use_potion
        with self.assertRaises(KeyError):
            use_potion(self._combat(), "no_such_potion")


class TestValuePropExtraction(unittest.TestCase):
    """``ValueProp`` 必须从源码抽出来并传染给引用它的效果。"""

    @classmethod
    def setUpClass(cls):
        if not CONTENT.exists():
            raise unittest.SkipTest(f"缺内容目录 {CONTENT}")

    def _potion(self, pid: str) -> dict:
        rows = json.loads((CONTENT / "potions_source.json").read_text(encoding="utf-8"))
        return next(p for p in rows if p["pid"] == pid)

    def test_unpowered_is_recorded_on_potion_effects(self):
        for pid in ("fire_potion", "explosive_ampoule", "block_potion"):
            with self.subTest(pid=pid):
                effects = self._potion(pid)["effects"]
                self.assertTrue(any(e.get("unpowered") for e in effects),
                                f"{pid} 的效果应带 unpowered")

    def test_common_attack_cards_are_powered(self):
        """对照组：普通攻击牌用 ``ValueProp.Move``，**不该**被标 unpowered。"""
        cards = json.loads((CONTENT / "cards_source.json").read_text(encoding="utf-8"))
        strike = next(c for c in cards if c["cid"] == "strike_ironclad")
        self.assertFalse(any(e.get("unpowered") for e in strike["effects"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
