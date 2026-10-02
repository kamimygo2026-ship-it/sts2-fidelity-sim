"""出招状态机解释器的**运行期**测试（``docs/09`` L1）。

上面的 ``test_monster_ai.py`` 验证"提取对不对"；这里验证"跑起来对不对"——
严格对着 ``MonsterMoveStateMachine.cs`` 的语义。
"""

from __future__ import annotations

import json
import pathlib
import random
import unittest

from sts2_sim.monster_ai import (
    CANNOT_REPEAT, CAN_REPEAT_FOREVER, CAN_REPEAT_X_TIMES, USE_ONLY_ONCE,
    Branch, MonsterAiError, MonsterContext, MoveStateMachine, State,
    UnknownCondition, build, evaluate_condition,
)
from sts2_sim.rng import RngSet

AI_PATH = pathlib.Path("data/content/repo/monster_ai.json")


def records() -> dict[str, dict]:
    if not AI_PATH.exists():
        raise unittest.SkipTest(f"缺少 {AI_PATH}")
    return {m["eid"]: m for m in json.loads(AI_PATH.read_text(encoding="utf-8"))}


def make(mid: str, seed: int = 1) -> MoveStateMachine:
    machine = build(records()[mid])
    machine._rng = RngSet(seed)["monster_ai"]
    return machine


def play(machine: MoveStateMachine, ctx: MonsterContext, turns: int) -> list[str]:
    """模拟真机时序：RollMove 决定招式 → 执行完调用 OnMovePerformed。"""
    moves = []
    for _ in range(turns):
        state = machine.roll_move(ctx, machine._rng)
        moves.append(state.state_id)
        machine.on_move_performed()
    return moves


class TestRepeatConstraints(unittest.TestCase):
    """逐条验证 ``RandomBranchState.GetStateWeight`` 的算法。"""

    def _machine(self, branch: Branch, turns_log: list[str]) -> MoveStateMachine:
        states = {
            "INIT": State("INIT", "move", move="init", follow_up="RAND"),
            "RAND": State("RAND", "random", branches=(branch,)),
            "TARGET": State("TARGET", "move", move="t", follow_up="RAND"),
        }
        machine = MoveStateMachine(states, "INIT")
        machine.state_log = ["INIT"] + list(turns_log)
        return machine

    def _weight(self, branch: Branch, log: list[str]) -> float:
        machine = self._machine(branch, log)
        return machine._branch_weight(branch, machine.states["RAND"])

    def test_can_repeat_forever_never_decays(self):
        branch = Branch("TARGET", CAN_REPEAT_FOREVER, weight=1.0)
        self.assertEqual(self._weight(branch, ["TARGET"] * 5), 1.0)

    def test_cannot_repeat_blocks_only_immediately_previous(self):
        branch = Branch("TARGET", CANNOT_REPEAT, weight=1.0)
        self.assertEqual(self._weight(branch, ["TARGET"]), 0.0, "上一次就是它 → 权重 0")
        self.assertEqual(self._weight(branch, ["OTHER", "TARGET"]), 0.0)
        self.assertEqual(self._weight(branch, ["TARGET", "OTHER"]), 1.0,
                         "上次不是它 → 可以再出")
        self.assertEqual(self._weight(branch, []), 1.0, "还没出过 → 可以出")

    def test_use_only_once_dies_after_first_use(self):
        branch = Branch("TARGET", USE_ONLY_ONCE, weight=1.0)
        self.assertEqual(self._weight(branch, []), 1.0)
        self.assertEqual(self._weight(branch, ["OTHER"]), 1.0)
        self.assertEqual(self._weight(branch, ["TARGET"]), 0.0)
        self.assertEqual(self._weight(branch, ["TARGET", "OTHER", "OTHER"]), 0.0,
                         "用过一次就永久为 0")

    def test_can_repeat_x_times_stops_after_streak(self):
        branch = Branch("TARGET", CAN_REPEAT_X_TIMES, max_times=2, weight=1.0)
        self.assertEqual(self._weight(branch, []), 1.0)
        self.assertEqual(self._weight(branch, ["TARGET"]), 1.0, "连用 1 次还能再用")
        self.assertEqual(self._weight(branch, ["TARGET", "TARGET"]), 0.0,
                         "连用满 2 次 → 0")
        self.assertEqual(self._weight(branch, ["TARGET", "TARGET", "OTHER"]), 1.0,
                         "被打断后重新可用")

    def test_cooldown_blocks_recent_use(self):
        branch = Branch("TARGET", CAN_REPEAT_FOREVER, weight=1.0, cooldown=2)
        self.assertEqual(self._weight(branch, ["OTHER", "TARGET"]), 0.0)
        self.assertEqual(self._weight(branch, ["TARGET", "OTHER", "OTHER"]), 1.0)

    def test_weight_multiplies(self):
        branch = Branch("TARGET", CAN_REPEAT_FOREVER, weight=0.4)
        self.assertAlmostEqual(self._weight(branch, []), 0.4)


class TestMachineSemantics(unittest.TestCase):
    def test_initial_state_is_the_first_move(self):
        """源码：`(!_performedFirstMove && IsMove) → return`，所以第一招就是初始状态。"""
        machine = make("mawler")
        self.assertEqual(machine.roll_move(MonsterContext(), machine._rng).state_id,
                         "CLAW_MOVE")

    def test_empty_next_returns_to_initial(self):
        """源码：`SetCurrentState(string.IsNullOrEmpty(next) ? _initialState : States[next])`。"""
        states = {
            "A_MOVE": State("A_MOVE", "move", follow_up=""),        # next 为空
            "B_MOVE": State("B_MOVE", "move", follow_up=None),
        }
        machine = MoveStateMachine(states, "A_MOVE")
        self.assertEqual(machine.roll_move(MonsterContext(), RngSet(1)["monster_ai"]).state_id,
                         "A_MOVE")
        machine.on_move_performed()
        # 从 A 转移：next 为空 → 回到初始状态（仍然是 A）
        self.assertEqual(machine.roll_move(MonsterContext(), RngSet(1)["monster_ai"]).state_id,
                         "A_MOVE")

    def test_move_state_without_follow_up_raises(self):
        """源码里 MoveState.GetNextState 在无后继时抛异常，我们必须同样抛。"""
        states = {"A_MOVE": State("A_MOVE", "move", follow_up=None)}
        machine = MoveStateMachine(states, "A_MOVE")
        machine.on_move_performed()
        with self.assertRaises(MonsterAiError):
            machine.roll_move(MonsterContext(), RngSet(1)["monster_ai"])

    def test_must_perform_once_blocks_transition(self):
        """4 只怪用这个：没执行过就不许转移。"""
        states = {
            "STUN_MOVE": State("STUN_MOVE", "move", follow_up="NEXT_MOVE",
                               must_perform_once=True),
            "NEXT_MOVE": State("NEXT_MOVE", "move", follow_up="STUN_MOVE"),
        }
        machine = MoveStateMachine(states, "STUN_MOVE")
        rng = RngSet(1)["monster_ai"]
        self.assertEqual(machine.roll_move(MonsterContext(), rng).state_id, "STUN_MOVE")
        machine.on_move_performed()
        self.assertEqual(machine.roll_move(MonsterContext(), rng).state_id, "NEXT_MOVE")

    def test_conditional_branch_picks_by_flag(self):
        machine = make("bowlbug_rock")
        ctx = MonsterContext(flags={"IsOffBalance": True})
        first = machine.roll_move(ctx, machine._rng)
        machine.on_move_performed()
        self.assertEqual(machine.roll_move(ctx, machine._rng).state_id, "DIZZY_MOVE")

        machine2 = make("bowlbug_rock")
        ctx2 = MonsterContext(flags={"IsOffBalance": False})
        machine2.roll_move(ctx2, machine2._rng)
        machine2.on_move_performed()
        self.assertEqual(machine2.roll_move(ctx2, machine2._rng).state_id, "HEADBUTT_MOVE")

    def test_conditional_with_no_match_raises(self):
        states = {
            "A_MOVE": State("A_MOVE", "move", follow_up="COND"),
            "COND": State("COND", "conditional", conditions=(("A_MOVE", "UnknownThing"),)),
        }
        machine = MoveStateMachine(states, "A_MOVE")
        machine.on_move_performed()
        with self.assertRaises(UnknownCondition):
            machine.roll_move(MonsterContext(), RngSet(1)["monster_ai"])

    def test_dangling_state_is_rejected(self):
        with self.assertRaises(MonsterAiError):
            MoveStateMachine({"A_MOVE": State("A_MOVE", "move", follow_up="GHOST")}, "A_MOVE")


class TestMawlerBehavior(unittest.TestCase):
    """随机分支 + 重复约束的真实行为。"""

    def test_roar_is_used_at_most_once(self):
        machine = make("mawler")
        moves = play(machine, MonsterContext(), turns=60)
        self.assertEqual(moves.count("ROAR_MOVE"), 1,
                         "ROAR 标了 UseOnlyOnce，整场只能出一次")

    def test_no_two_identical_moves_in_a_row(self):
        """CLAW 与 RIP_AND_TEAR 都标了 CannotRepeat。"""
        machine = make("mawler")
        moves = play(machine, MonsterContext(), turns=60)
        repeats = [(a, b) for a, b in zip(moves, moves[1:]) if a == b]
        self.assertEqual(repeats, [], f"出现了连续重复出招：{repeats[:5]}")

    def test_all_moves_are_reachable(self):
        machine = make("mawler", seed=7)
        moves = set(play(machine, MonsterContext(), turns=60))
        self.assertEqual(moves, {"CLAW_MOVE", "ROAR_MOVE", "RIP_AND_TEAR_MOVE"})

    def test_first_move_is_claw(self):
        machine = make("mawler")
        self.assertEqual(play(machine, MonsterContext(), turns=1), ["CLAW_MOVE"])


class TestConditions(unittest.TestCase):
    """32 个真机条件表达式必须全部可求值——**不能有静默的 False**。"""

    def test_every_condition_in_the_data_evaluates(self):
        # ⚠️ 标志表必须**完整**：`MonsterContext.flag` 对缺值抛 UnsupportedMechanic
        # （审计 F05 —— 默认 False 会让 Fabricator 的出招悄悄走错分支）。
        # 这里用 `resolve_flags` 造一份"全部已知标志都有值"的上下文。
        from sts2_sim.monster_ai import resolve_flags
        ctx = MonsterContext(
            slot_name="first", is_front=True, ally_count=1,
            hp=50, max_hp=100, powers=frozenset({"asleep"}),
            flags={**resolve_flags([]), "IsOffBalance": True,
                   "HasAmalgamDied": True, "HasBeetleCharged": True},
            counters={"_curseOfKnowledgeCounter": 3, "Respawns": 2})
        unknown = []
        for monster in records().values():
            for entries in monster.get("conditionals", {}).values():
                for entry in entries:
                    try:
                        evaluate_condition(entry["condition"], ctx)
                    except UnknownCondition:
                        unknown.append((monster["eid"], entry["condition"]))
        self.assertEqual(unknown, [], f"{len(unknown)} 个条件未实现")

    def test_slot_conditions(self):
        self.assertTrue(evaluate_condition('base.Creature.SlotName == "first"',
                                           MonsterContext(slot_name="first")))
        self.assertFalse(evaluate_condition('base.Creature.SlotName == "second"',
                                            MonsterContext(slot_name="first")))

    def test_power_conditions(self):
        ctx = MonsterContext(powers=frozenset({"asleep"}))
        self.assertTrue(evaluate_condition("base.Creature.HasPower<AsleepPower>()", ctx))
        self.assertFalse(evaluate_condition("!base.Creature.HasPower<AsleepPower>()", ctx))
        self.assertFalse(evaluate_condition("base.Creature.HasPower<SlumberPower>()", ctx))

    def test_cast_and_member_conditions(self):
        self.assertTrue(evaluate_condition("((Nibbit)base.Creature.Monster).IsFront",
                                           MonsterContext(is_front=True)))
        self.assertTrue(evaluate_condition("!((Nibbit)base.Creature.Monster).IsFront",
                                           MonsterContext(is_front=False)))

    def test_compound_conditions(self):
        ctx = MonsterContext(hp=60, max_hp=100, flags={"HasBeetleCharged": False})
        self.assertTrue(evaluate_condition(
            "!HasBeetleCharged && base.Creature.CurrentHp >= base.Creature.MaxHp / 2", ctx))
        ctx2 = MonsterContext(hp=40, max_hp=100, flags={"HasBeetleCharged": False})
        self.assertTrue(evaluate_condition(
            "!HasBeetleCharged && base.Creature.CurrentHp < base.Creature.MaxHp / 2", ctx2))

    def test_ally_and_negation(self):
        self.assertTrue(evaluate_condition("GetAllyCount() > 0", MonsterContext(ally_count=2)))
        self.assertFalse(evaluate_condition("GetAllyCount() > 0", MonsterContext(ally_count=0)))
        self.assertTrue(evaluate_condition("!CanFabricate",
                                           MonsterContext(flags={"CanFabricate": False})))

    def test_unknown_condition_raises_loudly(self):
        with self.assertRaises(UnknownCondition):
            evaluate_condition("SomeBrandNewFlag", MonsterContext())


class TestSeededReproducibility(unittest.TestCase):
    def test_same_seed_same_moves(self):
        """同种子必须完全可复现——这是对拍与反作弊的基础。"""
        a = play(make("mawler", seed=42), MonsterContext(), turns=30)
        b = play(make("mawler", seed=42), MonsterContext(), turns=30)
        self.assertEqual(a, b)

    def test_different_seeds_diverge(self):
        traces = {tuple(play(make("mawler", seed=s), MonsterContext(), turns=30))
                  for s in range(12)}
        self.assertGreater(len(traces), 5, "出招序列似乎没有真正随机化")


class TestRuntimeContextIsComplete(unittest.TestCase):
    """⭐ 审计 F05：条件读取必须来自**真实运行状态**，缺值必须报错。

    | 症状 | 根因 | 对应测试 |
    |---|---|---|
    | 单独生成 Fabricator 初始意图是 disintegrate | 专属标志缺失被默认成 False | `test_fabricator_can_fabricate_when_alone` |
    | 三只怪的第 3 只被叫成 second | 槽位被简化成 first/second | `test_slot_names_come_from_position` |
    | 缺标志时静默走错分支 | 访问器默认 False/0 | `test_missing_flag_raises` |
    """

    @classmethod
    def setUpClass(cls):
        from sts2_sim.content import load_builtin
        from sts2_sim.featurize import rebuild_vocab
        if not AI_PATH.exists():
            raise unittest.SkipTest(f"缺少 {AI_PATH}")
        from sts2_sim.featurize import configure_content
        configure_content("data/content/repo")
        cls.addClassCleanup(lambda: (load_builtin(), rebuild_vocab()))

    def combat(self, enemies: tuple[str, ...]):
        from sts2_sim.core import start_combat
        return start_combat(["strike_ironclad"] * 5 + ["defend_ironclad"] * 4,
                            list(enemies), seed=1, player_hp=70)

    def test_fabricator_can_fabricate_when_alone(self):
        """``CanFabricate`` = 存活队友（**含自己**）< 4 → 单只时必须走随机分支。

        源码：``Fabricator.cs:51`` + ``CombatState.GetTeammatesOf``
        （doc comment 写明 "including the creature itself"）。
        """
        state = self.combat(("fabricator",))
        intent = state.enemies[0].intent.mid
        self.assertIn(intent, ("fabricate", "fabricating_strike"),
                      f"单独一只 Fabricator 应当能造机器人，实际意图 {intent}")

    def test_fabricator_cannot_fabricate_with_a_full_team(self):
        """反向验证：队友满 5 只时 ``CanFabricate`` 为假 → 走 disintegrate。"""
        state = self.combat(("fabricator", "stabbot", "guardbot",
                             "noisebot", "zapbot"))
        self.assertEqual(state.enemies[0].intent.mid, "disintegrate")

    def test_slot_names_come_from_position(self):
        """槽位是**序数**（``Exoskeleton`` 的条件是 ``SlotName == "third"``）。"""
        state = self.combat(("exoskeleton", "myte", "guardbot"))
        self.assertEqual([e.slot_name for e in state.enemies],
                         ["first", "second", "third"])

    def test_exoskeleton_uses_its_slot_branch(self):
        """三只怪里第 3 只 Exoskeleton 才会走 ENRAGE 分支。"""
        state = self.combat(("guardbot", "zapbot", "exoskeleton"))
        self.assertEqual(state.enemies[2].slot_name, "third")
        self.assertEqual(state.enemies[2].intent.mid, "enrage")

    def test_instance_flags_are_initialised_not_defaulted(self):
        """实例标志必须**显式初始化**（出厂值 False，对应 C# 字段默认值）。"""
        state = self.combat(("frog_knight",))
        self.assertIn("HasBeetleCharged", state.enemies[0].ai_flags)

    def test_missing_flag_raises_instead_of_defaulting_false(self):
        """``docs/13`` §5：缺必需值必须报错 —— 默认 False 会静默走错分支。"""
        from sts2_sim.monster_ai import UnsupportedMechanic
        with self.assertRaises(UnsupportedMechanic):
            evaluate_condition("CanFabricate", MonsterContext())
        with self.assertRaises(UnsupportedMechanic):
            MonsterContext(counters={}).counter("Respawns")

    def test_slot_overflow_is_rejected_loudly(self):
        """真机最多 5 个敌人槽位；第 6 个没有槽位名，必须报错而不是编号 5。"""
        from sts2_sim.monster_ai import UnsupportedMechanic, slot_name_for
        self.assertEqual(slot_name_for(4), "fifth")
        with self.assertRaises(UnsupportedMechanic):
            slot_name_for(5)

    def test_encounters_beyond_slot_capacity_are_excluded(self):
        """遭遇表里成员超过 5 的记录必须被排除（那是一场真机不存在的战斗）。"""
        from sts2_sim.content import ENCOUNTERS, MAX_ENEMY_SLOTS
        for kind, table in ENCOUNTERS.items():
            for members in table:
                self.assertLessEqual(len(members), MAX_ENEMY_SLOTS,
                                     f"{kind} 里的 {members} 超出槽位上限")


if __name__ == "__main__":
    unittest.main(verbosity=2)
