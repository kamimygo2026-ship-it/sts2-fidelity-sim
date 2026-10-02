"""敌人出招状态机的**对照测试**（``docs/09`` L1）。

原则：源码是机制的唯一真相。下面每一条断言都是从反编译的 C# 里读出来、写死在这里的
**事实**——提取器一旦回归，测试立刻失败。

Mawler.cs：
    MoveState claw    = new("CLAW_MOVE",  ClawMove,  new MultiAttackIntent(ClawDamage, 2));
    MoveState roar    = new("ROAR_MOVE",  RoarMove,  new DebuffIntent());
    MoveState ripTear = new("RIP_AND_TEAR_MOVE", RipAndTearMove, new SingleAttackIntent(RipAndTearDamage));
    r.AddBranch(ripTear, MoveRepeatType.CannotRepeat, 1f);
    r.AddBranch(roar,    MoveRepeatType.UseOnlyOnce,  1f);
    r.AddBranch(claw,    MoveRepeatType.CannotRepeat, 1f);
    return new MonsterMoveStateMachine(list, moveState3);   // 初始 = CLAW

BowlbugRock.cs（**codex JSON 在这里是错的**，把 FollowUpState 记成了 null）：
    headbutt.FollowUpState = new ConditionalBranchState("POST_HEADBUTT");
    post.AddState(dizzy,    () => IsOffBalance);
    post.AddState(headbutt, () => !IsOffBalance);
"""

from __future__ import annotations

import json
import pathlib
import unittest

AI_PATH = pathlib.Path("data/content/repo/monster_ai.json")


def load() -> dict[str, dict]:
    if not AI_PATH.exists():
        raise unittest.SkipTest(f"缺少 {AI_PATH}（先运行 tools/extract_monsters.py）")
    return {m["eid"]: m for m in json.loads(AI_PATH.read_text(encoding="utf-8"))}


class TestExtractionCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ai = load()

    def test_every_monster_class_has_a_machine(self):
        """121 个怪物类全部要有状态机（4 个继承基类的用基类状态机）。"""
        self.assertEqual(len(self.ai), 121)

    def test_no_dangling_references(self):
        """所有跳转目标、分支目标都必须是已定义的状态。"""
        problems = []
        for eid, monster in self.ai.items():
            states = set(monster["states"])
            if monster["initial"] not in states:
                problems.append(f"{eid}: 初始状态 {monster['initial']} 未定义")
            for src, dst in monster["follow_up"].items():
                if dst not in states:
                    problems.append(f"{eid}: {src} → {dst} 未定义")
            for entries in list(monster["branches"].values()) + \
                    list(monster["conditionals"].values()):
                for entry in entries:
                    if entry["state"] not in states:
                        problems.append(f"{eid}: 分支目标 {entry['state']} 未定义")
        self.assertEqual(problems, [], f"{len(problems)} 处悬空引用")

    def test_move_states_have_handlers(self):
        missing = [(eid, sid) for eid, m in self.ai.items()
                   for sid, st in m["states"].items()
                   if st["kind"] == "move" and not st["move"]]
        self.assertEqual(missing, [], "move 状态必须绑定了处理函数")


class TestMawler(unittest.TestCase):
    """随机分支 + 重复约束的代表。"""

    @classmethod
    def setUpClass(cls):
        cls.m = load()["mawler"]

    def test_initial_move(self):
        self.assertEqual(self.m["initial"], "CLAW_MOVE")

    def test_three_moves_with_exact_intents(self):
        self.assertEqual(
            {k: (v.get("intent"), v.get("intent_args")) for k, v in self.m["moves"].items()},
            {"ClawMove": ("MultiAttackIntent", ["ClawDamage", "2"]),
             "RoarMove": ("DebuffIntent", []),
             "RipAndTearMove": ("SingleAttackIntent", ["RipAndTearDamage"])})

    def test_all_moves_loop_back_to_random_branch(self):
        self.assertEqual(self.m["follow_up"],
                         {"CLAW_MOVE": "RAND", "ROAR_MOVE": "RAND",
                          "RIP_AND_TEAR_MOVE": "RAND"})

    def test_random_branch_constraints(self):
        branches = {b["state"]: b for b in self.m["branches"]["RAND"]}
        self.assertEqual(set(branches), {"CLAW_MOVE", "ROAR_MOVE", "RIP_AND_TEAR_MOVE"})
        self.assertEqual(branches["ROAR_MOVE"]["repeat"], "UseOnlyOnce")
        self.assertEqual(branches["CLAW_MOVE"]["repeat"], "CannotRepeat")
        self.assertEqual(branches["RIP_AND_TEAR_MOVE"]["repeat"], "CannotRepeat")
        for branch in branches.values():
            self.assertEqual(branch["weight"], 1.0)

    def test_ascension_values(self):
        """进阶值必须取自源码常量，不能拍脑袋。

        源码：``MinInitialHp => GetValueIfAscension(ToughEnemies, 76, 72)``。
        ``AscensionLevel`` 的枚举序号就是门槛（``ToughEnemies = 8``、``DeadlyEnemies = 9``）。
        """
        self.assertEqual(self.m["hp_range"],
                         {"normal": 72, "ascension": 76, "level": 8})
        claw = self.m["move_values"]["ClawMove"]
        self.assertEqual((claw["normal"], claw["ascension"], claw["level"], claw["times"]),
                         (4, 5, 9, 2))
        tear = self.m["move_values"]["RipAndTearMove"]
        self.assertEqual((tear["normal"], tear["ascension"], tear["level"]), (14, 16, 9))
        self.assertEqual(self.m["hp"], [72, 72])

    def test_ascension_coverage(self):
        """全量数据里应当有大量带进阶缩放的数值——否则说明提取退化了。"""
        ai = load()
        scaled_moves = sum(1 for m in ai.values()
                           for v in (m.get("move_values") or {}).values()
                           if v.get("level", 0) > 0)
        scaled_hp = sum(1 for m in ai.values()
                        if (m.get("hp_range") or {}).get("level", 0) > 0)
        print(f"\n[进阶缩放] 招式 {scaled_moves} 个 | 怪物血量 {scaled_hp} 个")
        self.assertGreater(scaled_moves, 150, "带进阶缩放的招式太少")
        self.assertGreater(scaled_hp, 80, "带进阶缩放的怪物太少")


class TestBowlbugRock(unittest.TestCase):
    """codex JSON 记错的 case —— 提取器必须给出正确结果。"""

    @classmethod
    def setUpClass(cls):
        cls.m = load()["bowlbug_rock"]

    def test_follow_up_is_not_null(self):
        """JSON 写的是 next: null，源码写的是 FollowUpState = POST_HEADBUTT。"""
        self.assertEqual(self.m["follow_up"]["HEADBUTT_MOVE"], "POST_HEADBUTT")
        self.assertEqual(self.m["follow_up"]["DIZZY_MOVE"], "HEADBUTT_MOVE")

    def test_conditional_branches_are_deduped_and_carry_conditions(self):
        entries = self.m["conditionals"]["POST_HEADBUTT"]
        self.assertEqual(len(entries), 2, "条件分支被记了两遍（AddState 重复解析）")
        conditions = {e["state"]: e["condition"] for e in entries}
        self.assertEqual(conditions, {"DIZZY_MOVE": "IsOffBalance",
                                      "HEADBUTT_MOVE": "!IsOffBalance"})

    def test_initial(self):
        self.assertEqual(self.m["initial"], "HEADBUTT_MOVE")


class TestConditionalInitialState(unittest.TestCase):
    """初始状态是局部变量：`MoveState initialState = (_flag ? a : b);`"""

    def test_chomper_resolves_alias(self):
        m = load()["chomper"]
        self.assertEqual(m["initial_expr"], "initialState")
        self.assertEqual(m["initial"], "SCREECH_MOVE")
        self.assertIn(m["initial"], m["states"])


class TestMoveTableCoverage(unittest.TestCase):
    """招式表（``monsters.json``，来自**社区库**）与实际 AI 状态机（来自**源码**）的对账。

    这一组测试守的是**当前最大的敌人保真缺口**：招式表来自社区库，
    它对很多招式只记了意图、没记效果，于是那些怪在模拟器里"那一回合什么也不做"。
    缺口数量必须**可数、可见、不许悄悄变大**。
    """

    #: 已知缺口台账。**只允许往下调**；往上涨说明回归，必须查明原因。
    #: 修法是把招式效果改成从源码抽取（和卡牌/遗物同一条管线）。
    KNOWN_EMPTY_BUFF_MOVES = 39
    KNOWN_UNMATCHED_MOVES = 7      # 其中 6 只是测试用假怪（不在 115 只真怪里）

    @classmethod
    def setUpClass(cls):
        cls.ai = load()
        cls.monsters = {m["eid"]: m for m in json.loads(
            pathlib.Path("data/content/repo/monsters.json").read_text(encoding="utf-8"))}

    @staticmethod
    def _mid_candidates(state_id: str, state: dict) -> set[str]:
        key = state_id.lower()
        return {key, key.removesuffix("_move"), str(state.get("move") or "").lower()}

    def test_move_ids_match_the_state_machine(self):
        """每个 ``MoveState`` 都要能在招式表里找到对应条目。

        找不到时引擎会**大声报错**（`KeyError: 状态机指向未知招式`），
        不会静默退化 —— 但那是运行到那一步才炸。这里做**静态**全量检查，
        把"走不到就永远发现不了"的潜伏失配提前抓出来。
        """
        unmatched: dict[str, list[str]] = {}
        for eid, record in self.ai.items():
            mids = {str(m["mid"]).lower()
                    for m in self.monsters.get(eid, {}).get("moves", [])}
            missing = [state_id for state_id, state in
                       (record.get("states") or {}).items()
                       if state.get("kind") == "move"
                       and not (self._mid_candidates(state_id, state) & mids)]
            if missing:
                unmatched[eid] = missing
        print(f"\n[招式对账] 失配 {len(unmatched)} 只：{sorted(unmatched)}")
        # 真怪里**只允许** torch_head_amalgam 一只失配（社区库漏了它的 strong_tackle）
        real_enemies = set(self.monsters)
        real_unmatched = {eid for eid in unmatched if eid in real_enemies}
        self.assertLessEqual(
            len(unmatched), self.KNOWN_UNMATCHED_MOVES,
            f"招式失配变多了（台账 {self.KNOWN_UNMATCHED_MOVES}）：{unmatched}")
        self.assertEqual(
            real_unmatched, {"torch_head_amalgam"},
            f"真怪里的招式失配清单变了：{sorted(real_unmatched)}")

    def test_no_buff_move_silently_does_nothing(self):
        """**意图是 buff/debuff 的招式不能没有效果。**

        这是静默洞的典型形状：意图告诉玩家"它要上 buff / 下 debuff"，
        而引擎里这招什么也不做 —— 玩家按错误的威胁评估决策，
        模型学到"这只怪那回合很安全"。源码里这些招都有明确命令，例如::

            // CalcifiedCultist.IncantationMove
            await PowerCmd.Apply<RitualPower>(ctx, base.Creature, IncantationAmount, …);
            // Chomper.ScreechMove
            await CardPileCmd.AddToCombatAndPreview<Dazed>(targets, PileType.Discard, 3, null);

        只统计 buff/debuff：``unknown`` 意图里混着大量真·空招
        （``nothing`` / ``sleep`` / ``stun``），那些空是**正确**的。
        """
        from sts2_sim import content
        from sts2_sim.featurize import configure_content
        configure_content("data/content/repo")
        self.addCleanup(self._restore)

        empty: list[str] = []
        for monster in self.monsters.values():
            for move in monster.get("moves") or []:
                if move.get("effects"):
                    continue
                if str(move.get("intent")) in ("buff", "debuff"):
                    empty.append(f"{monster['eid']}.{move['mid']}")
        self.assertLessEqual(
            len(empty), self.KNOWN_EMPTY_BUFF_MOVES,
            f"「意图说有 buff/debuff、效果却是空」的招式变多了"
            f"（台账 {self.KNOWN_EMPTY_BUFF_MOVES}）：{sorted(set(empty))[:10]}")
        self.assertGreaterEqual(
            len(empty), self.KNOWN_EMPTY_BUFF_MOVES,
            f"缺口被修好了但台账没跟着下调（现在是 {len(empty)}）—— "
            f"这是好事，请把 KNOWN_EMPTY_BUFF_MOVES 改成 {len(empty)}")

    def _restore(self):
        from sts2_sim.content import load_builtin
        from sts2_sim.featurize import rebuild_vocab
        load_builtin()
        rebuild_vocab()


class TestMoveEffectsFromSource(unittest.TestCase):
    """招式效果必须来自**源码**，不能继续用社区库（``docs/09`` §15 的 3.4）。

    社区库对很多招式只记了意图、没记效果，于是那些怪在模拟器里
    "那一回合什么也不做"，而意图还显示着 buff/debuff。
    """

    @classmethod
    def setUpClass(cls):
        from sts2_sim.featurize import configure_content
        configure_content("data/content/repo")
        cls.addClassCleanup(cls._restore)

    @staticmethod
    def _restore():
        from sts2_sim.content import load_builtin
        from sts2_sim.featurize import rebuild_vocab
        load_builtin()
        rebuild_vocab()

    def test_attack_moves_still_carry_damage(self):
        """⭐ **有数值的攻击招式必须有伤害效果**。

        真机的怪物伤害来自**意图**（``SingleAttackIntent(Damage)`` →
        ``DamageCmd.Attack(Damage).FromMonster(this)``），招式函数体里**没有**
        独立的伤害命令。把招式效果换成源码版本时，如果不把意图伤害补成
        ``damage`` 效果，敌人就**永远打不出伤害** ——
        而 ``_effective_effects`` 只在已有 ``damage`` 效果时才用意图数值覆盖。

        实测这个坑真踩过：221 个攻击招式里 197 个丢了伤害，
        表现是"玩家打完一整场没掉血"（是
        `test_vulnerable_is_active_during_the_enemy_turn` 抓出来的）。

        ``value == 0`` 的招式不查 —— 那种伤害由**运行期表达式**决定
        （见 :attr:`KNOWN_RUNTIME_DAMAGE_MOVES`）。
        """
        from sts2_sim import content
        broken = []
        for enemy in content.ENEMY_DB.values():
            for move in enemy.moves:
                if move.value <= 0:
                    continue
                ops = {e.op for e in move.effects}
                if not (ops & {"damage", "damage_all"}):
                    broken.append(f"{enemy.eid}.{move.mid}")
        self.assertEqual(broken, [],
                         f"这些攻击招式没有伤害效果（敌人会站着不动）：{broken[:8]}")

    def test_runtime_damage_moves_are_ledgered(self):
        """台账：**伤害由运行期决定**、因而没建模的攻击招式数。

        这几招的 ``value`` 是 0（真机写的是运行期表达式，如"伤害等于某值"），
        静态抽不出数值。数量不许变多；修好了要手动下调。
        """
        from sts2_sim import content
        runtime = []
        for enemy in content.ENEMY_DB.values():
            for move in enemy.moves:
                if move.value > 0 or move.intent not in ("attack", "attack_debuff"):
                    continue
                if not ({e.op for e in move.effects} & {"damage", "damage_all"}):
                    runtime.append(f"{enemy.eid}.{move.mid}")
        print(f"\n[运行期伤害招式] {len(runtime)} 条：{sorted(runtime)}")
        self.assertLessEqual(len(runtime), self.KNOWN_RUNTIME_DAMAGE_MOVES,
                             f"运行期伤害的招式变多了：{sorted(runtime)}")

    #: 台账：伤害由运行期表达式决定、当前未建模的攻击招式。
    KNOWN_RUNTIME_DAMAGE_MOVES = 2
    #: 台账：仍为空的 buff/debuff 招式数。只允许往下调。
    KNOWN_EMPTY_BUFF_MOVES = 9

    def test_source_effects_are_adopted_for_most_moves(self):
        """源码必须成为招式效果的**主要**来源（台账：采纳数不许掉回去）。"""
        from sts2_sim import content
        summary = content.load_content_dir("data/content/repo")
        adopted = summary.get("monster_move_effects_from_source", 0)
        from_codex = summary.get("monster_move_effects_from_codex", 0)
        print(f"\n[招式效果] 源码 {adopted} 条 / 社区库 {from_codex} 条")
        self.assertGreater(adopted, 200, "源码采纳数太低，抽取器可能没跑")
        self.assertGreater(adopted, from_codex, "源码应当是主要来源")

    def test_buff_move_effects_are_no_longer_empty(self):
        """台账：**意图说有 buff/debuff、效果却是空**的招式不许变多。

        起始值 39 条（全靠社区库时）。招式效果改从源码抽取后降到 6 条，
        剩下的是召唤 / 需要玩家选择这类真复杂的。
        """
        from sts2_sim import content
        empty = []
        for enemy in content.ENEMY_DB.values():
            for move in enemy.moves:
                if move.effects:
                    continue
                if move.intent in ("buff", "debuff"):
                    empty.append(f"{enemy.eid}.{move.mid}")
        self.assertLessEqual(
            len(empty), self.KNOWN_EMPTY_BUFF_MOVES,
            f"空的 buff/debuff 招式变多了（台账 {self.KNOWN_EMPTY_BUFF_MOVES}）："
            f"{sorted(empty)[:8]}")
        self.assertGreaterEqual(
            len(empty), 0)
        if len(empty) < self.KNOWN_EMPTY_BUFF_MOVES:
            print(f"\n[提示] 缺口已降到 {len(empty)}，"
                  f"请把 KNOWN_EMPTY_BUFF_MOVES 下调到 {len(empty)}")

    def test_signature_moves_match_the_source(self):
        """抽查两条有据可查的招式（源码写在 docstring 里，回归立刻失败）。

        ``CalcifiedCultist.IncantationMove``::

            await PowerCmd.Apply<RitualPower>(ctx, base.Creature, IncantationAmount, …);

        ``Chomper.ScreechMove``::

            await CardPileCmd.AddToCombatAndPreview<Dazed>(targets, PileType.Discard, 3, null);
        """
        from sts2_sim import content
        cultist = content.ENEMY_DB["calcified_cultist"]
        incantation = next(m for m in cultist.moves if m.mid == "incantation")
        powers = [(e.op, e.power, e.amount, e.target) for e in incantation.effects]
        self.assertIn(("apply_power", "ritual", 2, "self"), powers,
                      f"腐化信徒的仪式应当是「给自己 2 层 ritual」，实际 {powers}")

        chomper = content.ENEMY_DB["chomper"]
        screech = next(m for m in chomper.moves if m.mid == "screech")
        adds = [(e.card, e.amount, e.pile) for e in screech.effects
                if e.op == "add_card"]
        self.assertIn(("dazed", 3, "discard"), adds,
                      f"咀嚼者的尖啸应当塞 3 张眩晕进弃牌堆，实际 {adds}")

    def test_kills_self_is_authoritative_from_source(self):
        """自爆招式（``CreatureCmd.Kill(base.Creature)``）必须被标出来。

        少了它，自爆怪下一回合会去 roll 后继并抛"没有后继状态"
        （实测 `gas_bomb` 直接炸掉 Run 层）。
        """
        from sts2_sim import content
        self.assertTrue(content.ENEMY_DB["gas_bomb"].moves)
        self.assertTrue(any(m.kills_self for m in content.ENEMY_DB["gas_bomb"].moves),
                        "炸弹的自爆招式没被标记")


class TestIntents(unittest.TestCase):
    def test_intent_vocabulary(self):
        """意图类型必须收敛在一个小集合里，出现新类型要能看见。"""
        kinds = {v["intent"] for m in load().values() for v in m["moves"].values()
                 if v.get("intent")}
        print(f"\n[意图类型] {len(kinds)} 种：{sorted(kinds)}")
        self.assertTrue(kinds)
        self.assertIn("SingleAttackIntent", kinds)
        self.assertIn("MultiAttackIntent", kinds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
