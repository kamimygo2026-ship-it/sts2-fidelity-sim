"""**钩子总线**的结构测试（``docs/11`` §11.3）。

这一组测试守的是"钩子调用点分散"这个结构性风险。真机的钩子只有一个入口
（``Hook.AfterDeath``），漏调用在结构上不可能；引擎是手写调用点，
所以要用测试把"漏调用"变成**不可能通过**，而不是靠人记。

已经因此踩到的坑（``docs/11`` §11.3）：一次抓到 4 条死亡路径绕过钩子，
症状全是静默的 —— 滑溜的墨宝会变成永久免伤，而所有测试都是绿的。
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
    from sts2_sim.content import STARTING_DECK
    return list(STARTING_DECK)


def restore_builtin() -> None:
    from sts2_sim.content import load_builtin
    from sts2_sim.featurize import rebuild_vocab
    load_builtin()
    rebuild_vocab()


class HookRegistryTest(unittest.TestCase):
    """声明表本身必须自洽。"""

    def test_declarations_match_the_power_rule_fields(self):
        """``hooks.HOOKS`` 与 ``PowerRules`` 的钩子位必须一一对应。

        对不上的两种情况都会静默失效：声明了但没有字段（永远不会被调用），
        或字段没有声明（`fire` 会拒绝，但只有跑到那一步才知道）。
        """
        from sts2_sim import hooks
        self.assertEqual(hooks.audit(), [])

    def test_every_hook_cites_the_real_source(self):
        """每条声明都要带真机出处（铁律 R1）与触发时机。"""
        from sts2_sim import hooks
        for spec in hooks.HOOKS.values():
            self.assertTrue(spec.source, f"{spec.name} 缺少源码出处")
            self.assertTrue(spec.timing, f"{spec.name} 缺少触发时机说明")
            self.assertIn(spec.scope, hooks.HOLDER_SCOPES)

    def test_every_declared_call_site_exists(self):
        """声明里写的调用点必须在引擎源码里真实存在（含 ``orbs.py``）。

        这条是"声明与实现不漂移"的护栏：重构时改了函数名，
        声明没跟着改就会失败，而不是安静地指向一个不存在的函数。
        ``orbs.py`` 也在扫描范围内 —— 充能球侧的调用点
        （``evoke`` 分发 ``AfterOrbEvoked``）同样会被改坏。
        """
        from sts2_sim import hooks
        sources = "\n".join(
            (ROOT / "sts2_sim" / name).read_text(encoding="utf-8")
            for name in ("core.py", "powers.py", "relics.py", "run.py", "orbs.py"))
        missing = []
        for spec in hooks.HOOKS.values():
            for site in spec.call_sites:
                if f"def {site}(" not in sources:
                    missing.append(f"{spec.name} → {site}")
        self.assertEqual(missing, [], f"声明里的调用点不存在：{missing}")

    def test_unknown_hook_name_is_rejected_loudly(self):
        """拼错钩子名必须**立刻报错**，不能安静地什么都不做。

        这正是总线存在的意义：以前每个分发函数各写一份遍历，
        名字写错只会表现为"这个能力好像没用"。
        """
        from sts2_sim import hooks
        with self.assertRaises(KeyError):
            hooks.fire("on_no_such_hook", None, [], lambda *a: None)

    def test_missing_hook_argument_is_rejected_loudly(self):
        """钩子声明需要实参却没传 → 报错。

        ``on_attacked`` 需要 ``unblocked`` / ``from_card``：少了它们
        ``SlipperyPower`` 与 ``SkittishPower`` 永远不触发，而战斗照常进行。
        """
        from sts2_sim import hooks

        class Fake:
            player = None
            enemies: tuple = ()

        with self.assertRaises(TypeError):
            hooks.fire("on_attacked", Fake(), [], lambda *a: None, owner=None)


class RelicOnTheBusTest(unittest.TestCase):
    """遗物与能力共用同一张总线表（它们重写的是同一批 `AbstractModel` 虚方法）。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()

    @classmethod
    def tearDownClass(cls):
        restore_builtin()

    def test_every_relic_timing_has_a_declared_hook(self):
        """每个遗物时机都要在总线上有声明 —— 否则那个时机的遗物全部不生效。"""
        from sts2_sim import hooks, relics
        for timing in relics.TIMING_ORDER:
            hook = hooks.RELIC_TIMING_TO_HOOK.get(timing)
            self.assertIsNotNone(hook, f"遗物时机 {timing} 没有对应的总线钩子")
            self.assertIn(hook, hooks.HOOKS, f"{hook} 没在总线上声明")
            self.assertEqual(hooks.HOOKS[hook].scope, "relics")

    def test_relic_hooks_are_dispatched_in_a_turn(self):
        """一个回合里，四个回合级遗物时机都要被分发到。"""
        from sts2_sim import core, hooks
        state = core.start_combat(self.deck, ["crusher"], 7, relics=["lantern"])
        with hooks.trace() as record:
            state.discard.extend(state.hand)
            state.hand = []
            core._run_enemy_turn(state, [])
            state.turn += 1
            core.start_player_turn(state, [])
        fired = set(record.names())
        for timing in ("relic_before_side_turn_start", "relic_energy_reset",
                       "relic_after_player_turn_start", "relic_side_turn_start"):
            self.assertIn(timing, fired, f"回合里应当分发 {timing}")

    def test_combat_start_relic_hook_is_dispatched(self):
        """开局遗物时机在**第 1 回合之前**分发（真机 `CombatManager.cs:594`）。"""
        from sts2_sim import core, hooks
        state = core.start_combat(self.deck, ["crusher"], 7, relics=["vajra"])
        self.assertEqual(state.player.power("strength"), 1, "金刚杵应当生效")

    def test_a_responding_relic_shows_up_in_the_trace(self):
        """有人响应的遗物要能在轨迹里看到（区分"分发"与"响应"）。

        用**提灯**：它的效果是"第 1 回合 +1 能量"，走 ``after_side_turn_start``
        这个时机钩子。（准备背包那种改抽牌数的遗物是**数值修正**，
        不走钩子 —— 它们由 ``relics.hand_draw_bonus()`` 在流程里直接查询。）
        """
        from sts2_sim import core, hooks
        state = core.start_combat(self.deck, ["crusher"], 7, relics=["lantern"])
        state.discard.extend(state.hand)
        state.hand = []
        state.turn = 1                          # 提灯只在第 1 回合生效
        with hooks.trace() as record:
            core.start_player_turn(state, [])
        responders = record.responders()
        self.assertTrue(any(name.startswith("relic_") for name, _ in responders),
                        f"提灯应当响应，实际 {responders}")


class HookTraceTest(unittest.TestCase):
    """钩子轨迹：让"这一跳触发了哪些钩子"变成可断言的事实。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()

    @classmethod
    def tearDownClass(cls):
        restore_builtin()

    def test_one_combat_turn_fires_the_expected_hooks(self):
        """打一个回合，断言**分发过**的钩子集合。

        这条测试的价值在于它写死了时序：能量重置、回合开始、抽牌、
        出牌、受伤、回合结束 —— 少任何一个，或者顺序错了，都会看出来。
        （断言"分发"而不是"响应"：这些测试里的怪没有能力，
        响应者可能是空的，但**流程必须走到每一个时机**。）
        """
        from sts2_sim import core, hooks
        state = core.start_combat(self.deck, ["crusher"], 7)
        with hooks.trace() as record:
            state.discard.extend(state.hand)
            state.hand = []
            core._run_enemy_turn(state, [])
            state.turn += 1
            core.start_player_turn(state, [])
        fired = set(record.names())
        for expected in ("on_energy_reset", "on_owner_turn_start", "on_card_drawn",
                         "on_owner_turn_end"):
            self.assertIn(expected, fired, f"一个回合里应当触发 {expected}")
        print(f"\n[钩子轨迹] 一个回合分发 {len(record.fired)} 次，"
              f"其中 {len(record.responders())} 次有人响应：{sorted(fired)}")

    def test_responding_power_shows_up_in_the_trace(self):
        """**有人响应**的那一路：挂上能力之后，轨迹里要能看到它。

        与上一条相对 —— 它测的是"能力真的被触发了"，而不是"流程走到了"。
        """
        from sts2_sim import core, hooks
        state = core.start_combat(self.deck, ["crusher"], 7)
        enemy = state.enemies[0]
        enemy.add_power("thorns", 3)
        player_hp = state.player.hp
        with hooks.trace() as record:
            core.deal_damage(state, state.player, enemy, 5, [])
        self.assertEqual(record.count_responded("on_attacked"), 1,
                         "荆棘应当在被打时响应")
        self.assertLess(state.player.hp, player_hp, "荆棘应当反伤")

    def test_damage_fires_both_sides_of_the_hook(self):
        """造成伤害时，**攻防两侧**的钩子都要分发。

        只触发一侧是这类 bug 的典型形态：``PainfulStabsPower``（攻方）
        或 ``SlipperyPower``（守方）会静默失效。
        """
        from sts2_sim import core, hooks
        state = core.start_combat(self.deck, ["crusher"], 7)
        with hooks.trace() as record:
            core.deal_damage(state, state.player, state.enemies[0], 5, [])
        self.assertEqual(record.count("on_attacked"), 1, "受伤方的钩子")
        self.assertEqual(record.count("on_damage_given"), 1, "攻击方的钩子")

    def test_death_fires_self_and_ally_scopes(self):
        """死亡要分发两个作用域：死者自己的、队友的。"""
        from sts2_sim import core, hooks
        state = core.start_combat(self.deck, ["crusher", "axebot"], 7)
        dying = state.enemies[0]
        with hooks.trace() as record:
            core.mark_dead(state, dying, [])
        self.assertEqual(record.count("on_self_death"), 1, "死者自己的钩子")
        self.assertEqual(record.count("on_ally_death"), 1, "队友的钩子")

    def test_raw_damage_also_fires_damage_received(self):
        """掉血（中毒 / Unblockable）也要分发 `AfterDamageReceived`。

        真机两条路共用同一条钩子链，只是 `IsPoweredAttack()` 为假。
        """
        from sts2_sim import core, hooks
        state = core.start_combat(self.deck, ["crusher"], 7)
        with hooks.trace() as record:
            core.deal_raw_damage(state, state.enemies[0], 3, [], label="中毒")
        self.assertEqual(record.count("on_attacked"), 1)


if __name__ == "__main__":
    unittest.main()
