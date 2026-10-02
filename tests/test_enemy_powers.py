"""敌方自我强化类能力的**对照测试**（``docs/09`` L1 / §5.4）。

敌人的很多机制本质上是"给自己上一个特殊 buff"。每一条断言都从反编译
C# 里读出来，并把源码片段抄在 docstring 里 —— 提取器或实现一旦回归，测试立刻失败。

| 能力 | 源码行为 | 对应测试 |
|---|---|---|
| `RitualPower` | 拥有者回合结束 +Amount 力量；**刚获得的那一回合跳过** | `test_ritual_skips_the_turn_it_was_applied` |
| `HighVoltagePower` | 拥有者回合结束 +Amount 力量（无跳过） | `test_high_voltage_has_no_skip` |
| `EnragePower` | 玩家打出**技能牌** → +Amount 力量 | `test_enrage_only_triggers_on_skills` |
| `CrabRagePower` | 同阵营队友阵亡 → +6 力量 +99 格挡，**移除自身** | `test_crab_rage_triggers_on_ally_death` |
| `MinionPower` | 主敌人死亡且队友全是次级敌人 → 队友一起死 | `test_minions_die_with_their_leader` |
| `SlipperyPower` | 掉血压到 1；受未格挡伤害减 1 层 | `test_slippery_caps_hp_loss_and_decrements` |
| `SkittishPower` | 每回合一次；**卡牌**来源 + 真的掉血 → +Amount 格挡 | `test_skittish_once_per_turn_and_card_only` |
| `RampartPower` | **玩家回合开始**时己方炮台 +Amount 格挡（Unpowered） | `test_rampart_grants_block_on_player_turn_start` |
| `BattlewornDummyTimeLimitPower` | 倒计时归零 → `CreatureCmd.Escape`（**逃跑 ≠ 死亡**） | `test_time_limit_escapes_instead_of_dying` |
| `EscapeArtistPower` | 纯视觉倒计时，无战斗效果 | `test_escape_artist_has_no_gameplay_effect` |
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


class EnemyBuffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()

    @classmethod
    def tearDownClass(cls):
        restore_builtin()

    def fresh(self, enemies, *, clear_powers=True):
        """开一场战斗；默认清掉固有属性，让每个测试只测自己那一条。"""
        from sts2_sim import core
        state = core.start_combat(self.deck, list(enemies), 7)
        if clear_powers:
            for enemy in state.enemies:
                enemy.powers.clear()
                enemy.power_applied_turn.clear()
                enemy.power_used_turn.clear()
        self.events: list[str] = []
        return state

    # ---- 回合结束加力量 ----
    def test_ritual_skips_the_turn_it_was_applied(self):
        """``RitualPower``：回合结束 +Amount 力量，但**刚获得的那一次跳过**。

        .. code-block:: csharp

           public override Task AfterApplied(...)
           {
               if (base.Owner.IsEnemy) WasJustAppliedByEnemy = true;
           }
           public override async Task AfterSideTurnEnd(...)
           {
               if (WasJustAppliedByEnemy) { WasJustAppliedByEnemy = false; return; }
               await PowerCmd.Apply<StrengthPower>(..., base.Amount, ...);
           }

        仪式是敌人在**自己回合里**上的，同回合的回合结束紧接着就跑 ——
        不跳过的话"上仪式那一回合"白赚一层，觉醒邪教徒第一个回合就多 1 点力量。
        """
        from sts2_sim import core, powers
        state = self.fresh(["calcified_cultist"])
        enemy = state.enemies[0]
        enemy.add_power("ritual", 2)
        enemy.power_applied_turn["ritual"] = state.turn      # 本回合刚上
        powers.on_turn_end(enemy, state, self.events)
        self.assertEqual(enemy.power("strength"), 0, "刚获得的那一回合不该触发")
        powers.on_turn_end(enemy, state, self.events)
        self.assertEqual(enemy.power("strength"), 2)
        powers.on_turn_end(enemy, state, self.events)
        self.assertEqual(enemy.power("strength"), 4)

    def test_high_voltage_has_no_skip(self):
        """``HighVoltagePower``：同样回合结束 +力量，**没有**"刚获得跳过"那条规则。"""
        from sts2_sim import powers
        state = self.fresh(["calcified_cultist"])
        enemy = state.enemies[0]
        enemy.add_power("high_voltage", 3)
        enemy.power_applied_turn["high_voltage"] = state.turn
        powers.on_turn_end(enemy, state, self.events)
        self.assertEqual(enemy.power("strength"), 3)

    # ---- 触发型 ----
    def test_enrage_only_triggers_on_skills(self):
        """``EnragePower.AfterCardPlayed``：**只有技能牌**触发。

        .. code-block:: csharp

           if (cardPlay.Card.Type == CardType.Skill)
               await PowerCmd.Apply<StrengthPower>(choiceContext, base.Owner, base.Amount, ...);

        一律触发会把"愤怒"从"别打技能牌"变成"别出牌"，是另一种游戏。
        """
        from sts2_sim import powers
        state = self.fresh(["calcified_cultist"])
        enemy = state.enemies[0]
        enemy.add_power("enrage", 2)
        powers.on_card_played(state, self.events, "attack")
        self.assertEqual(enemy.power("strength"), 0, "攻击牌不该触发")
        powers.on_card_played(state, self.events, "power")
        self.assertEqual(enemy.power("strength"), 0, "能力牌不该触发")
        powers.on_card_played(state, self.events, "skill")
        self.assertEqual(enemy.power("strength"), 2, "技能牌应当触发")

    def test_crab_rage_triggers_on_ally_death(self):
        """``CrabRagePower.AfterDeath``：同阵营队友阵亡 → +6 力量 +99 格挡，然后**移除自身**。

        .. code-block:: csharp

           if (creature != base.Owner && creature.Side == base.Owner.Side)
           {
               await PowerCmd.Apply<StrengthPower>(..., 6, ...);
               await CreatureCmd.GainBlock(base.Owner, 99m, null);
               await PowerCmd.Remove(this);
           }

        6 与 99 是写死的 ``DynamicVars``，**不受层数影响**；一次性效果。
        """
        from sts2_sim import powers
        state = self.fresh(["crusher", "crusher"])
        survivor, dying = state.enemies
        survivor.add_power("crab_rage", 1)
        dying.hp = 0
        powers.on_any_death(state, dying, self.events)
        self.assertEqual(survivor.power("strength"), 6)
        self.assertEqual(survivor.block, 99)
        self.assertEqual(survivor.power("crab_rage"), 0, "触发后应移除自身")

    # ---- 死亡语义 ----
    def test_minions_die_with_their_leader(self):
        """``MinionPower``：**主敌人倒下且队友全是次级敌人 → 队友一起倒下**。

        .. code-block:: csharp

           if (creature.Side == CombatSide.Enemy)
               if (isPrimaryEnemy && teammates.Count != 0
                   && teammates.All(t => t.IsSecondaryEnemy))
                   await Kill(teammates);

        ``IsSecondaryEnemy`` = 身上有 ``OwnerIsSecondaryEnemy`` 的能力
        （``MinionPower`` / ``IllusionPower``）。漏了这条，带小怪的 Boss 战
        会变成"必须把每只小怪都清掉"。
        """
        from sts2_sim import powers
        state = self.fresh(["kin_follower", "kin_follower"])
        minion, leader = state.enemies
        minion.add_power("minion", 1)
        leader.hp = 0
        powers.on_any_death(state, leader, self.events)
        self.assertEqual(minion.hp, 0, "次级敌人应随首领倒下")

    def test_minions_survive_when_a_non_secondary_ally_dies(self):
        """队友里还有**非次级**敌人时，小怪**不该**跟着死（否则会误杀）。"""
        from sts2_sim import powers
        state = self.fresh(["kin_follower", "kin_follower", "kin_follower"])
        minion, other, leader = state.enemies
        minion.add_power("minion", 1)
        leader.hp = 0
        powers.on_any_death(state, leader, self.events)
        self.assertGreater(minion.hp, 0, "还有正常队友时不该连锁死亡")
        self.assertGreater(other.hp, 0)

    # ---- 防御型 ----
    def test_slippery_caps_hp_loss_and_decrements(self):
        """``SlipperyPower``：掉血压到 1，且每次**未格挡**伤害减 1 层。"""
        from sts2_sim import core
        state = self.fresh(["inklet"])
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.add_power("slippery", 2)
        self.assertEqual(core.deal_damage(state, state.player, enemy, 30, self.events), 1)
        self.assertEqual(enemy.power("slippery"), 1)
        self.assertEqual(core.deal_damage(state, state.player, enemy, 30, self.events), 1)
        self.assertEqual(enemy.power("slippery"), 0)
        self.assertEqual(core.deal_damage(state, state.player, enemy, 30, self.events), 30,
                         "层数用完后应打满")

    def test_skittish_once_per_turn_and_card_only(self):
        """``SkittishPower``：每回合一次；**必须是卡牌来源**且**真的掉了血**。

        三个条件缺一不可（源码）：

        * ``!HasGainedBlockThisTurn`` —— 每回合一次
        * ``command.DamageProps.HasFlag(ValueProp.Move)`` —— 招式伤害
        * ``command.ModelSource is CardModel`` —— 来源是卡牌
        * ``damageResult.UnblockedDamage != 0`` —— 真的掉了血
        """
        from sts2_sim import core
        state = self.fresh(["phantasmal_gardener"])
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        enemy.add_power("skittish", 4)
        core.deal_damage(state, state.player, enemy, 10, self.events)
        self.assertEqual(enemy.block, 4)
        enemy.block = 0
        core.deal_damage(state, state.player, enemy, 10, self.events)
        self.assertEqual(enemy.block, 0, "同一回合只触发一次")
        enemy.power_used_turn.clear()
        core.deal_damage(state, state.player, enemy, 10, self.events)
        self.assertEqual(enemy.block, 4, "新回合可以再触发")

    def test_skittish_ignores_non_card_damage(self):
        """非卡牌来源（中毒 / 反伤 / 怪物互殴）**不触发**胆怯。"""
        from sts2_sim import core
        state = self.fresh(["phantasmal_gardener"])
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        enemy.add_power("skittish", 4)
        core.deal_raw_damage(state, enemy, 10, self.events, label="中毒")
        self.assertEqual(enemy.block, 0)

    def test_skittish_ignores_fully_blocked_hits(self):
        """完全被格挡的攻击**不触发**（``UnblockedDamage != 0`` 那条）。"""
        from sts2_sim import core
        state = self.fresh(["phantasmal_gardener"])
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        enemy.add_power("skittish", 4)
        enemy.block = 100
        core.deal_damage(state, state.player, enemy, 10, self.events)
        self.assertEqual(enemy.block, 90, "格挡被消耗但没掉血 → 不补格挡")

    def test_rampart_grants_block_on_player_turn_start(self):
        """``RampartPower``：**玩家回合开始**时己方炮台 +Amount 格挡（``Unpowered``）。"""
        from sts2_sim import powers
        state = self.fresh(["turret_operator"])
        turret = state.enemies[0]
        turret.add_power("rampart", 5)
        turret.add_power("dexterity", 9)          # Unpowered：敏捷不该影响
        powers.on_player_turn_start(state, self.events)
        self.assertEqual(turret.block, 5)

    # ---- 倒计时 / 逃跑 ----
    def test_time_limit_escapes_instead_of_dying(self):
        """``BattlewornDummyTimeLimitPower``：归零 → **逃跑**，不是死亡。

        逃跑与死亡在结算上不同（不算击杀、不进击杀统计），所以引擎里用一个
        显式标记区分，而不是"血量归零就当死了"。
        """
        from sts2_sim import powers
        state = self.fresh(["battle_friend_v1"])
        dummy = state.enemies[0]
        dummy.add_power("battleworn_dummy_time_limit", 2)
        powers.on_turn_end(dummy, state, self.events)
        self.assertEqual(dummy.power("battleworn_dummy_time_limit"), 1)
        self.assertGreater(dummy.hp, 0, "还有时间时不该走")
        powers.on_turn_end(dummy, state, self.events)
        self.assertEqual(dummy.hp, 0)
        self.assertTrue(dummy.escaped, "应当是逃跑而不是普通死亡")
        self.assertEqual(dummy.power("battleworn_dummy_time_limit"), 0)

    def test_escape_artist_has_no_gameplay_effect(self):
        """``EscapeArtistPower`` 是**纯视觉倒计时**：只递减层数，不改任何数值。

        源码注释：*"Just a visual timer for when ThievingHopper will escape."*
        照实实现 —— 不给它硬塞一个不存在效果。
        """
        from sts2_sim import powers
        state = self.fresh(["thieving_hopper"])
        hopper = state.enemies[0]
        hopper.add_power("escape_artist", 3)
        hp, block = hopper.hp, hopper.block
        powers.on_turn_end(hopper, state, self.events)
        self.assertEqual(hopper.power("escape_artist"), 2)
        self.assertEqual((hopper.hp, hopper.block), (hp, block), "不该改任何数值")

    # ---- 减伤族（乘法阶段）----
    def test_flutter_halves_damage_and_decrements(self):
        """``FlutterPower``：受有效攻击伤害 **×0.50**，每受一次有效伤害减 1 层。

        .. code-block:: csharp

           public override decimal ModifyDamageMultiplicative(...)
           {
               if (target != base.Owner) return 1m;
               if (!props.IsPoweredAttack()) return 1m;
               return base.DynamicVars["DamageDecrease"].BaseValue / 100m;   // 50/100
           }

        ``IsPoweredAttack()`` = 带 ``Move`` 且非 ``Unpowered``
        （``ValuePropExtensions.cs:5-12``），所以中毒 / 药水伤害**不吃**这个减伤。
        """
        from sts2_sim import core
        state = self.fresh(["thieving_hopper"])
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        enemy.add_power("flutter", 2)
        self.assertEqual(core.deal_damage(state, state.player, enemy, 20, self.events), 10)
        self.assertEqual(enemy.power("flutter"), 1)
        self.assertEqual(core.deal_damage(state, state.player, enemy, 20, self.events), 10)
        self.assertEqual(enemy.power("flutter"), 0)
        self.assertEqual(core.deal_damage(state, state.player, enemy, 20, self.events), 20,
                         "层数用完后不再减伤")

    def test_soar_halves_damage_permanently(self):
        """``SoarPower``：同样 ×0.50，但**不减层**（``StackType.Single``，无递减钩子）。"""
        from sts2_sim import core
        state = self.fresh(["thieving_hopper"])
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        enemy.add_power("soar", 1)
        self.assertEqual(core.deal_damage(state, state.player, enemy, 20, self.events), 10)
        self.assertEqual(core.deal_damage(state, state.player, enemy, 20, self.events), 10)
        self.assertEqual(enemy.power("soar"), 1)

    def test_shrink_reduces_owner_outgoing_damage(self):
        """``ShrinkPower``：减的是**拥有者造成**的伤害（``Owner != dealer`` 时才放过）。

        ``return (100m - DamageDecrease) / 100m`` = ``0.70``。
        与易伤等其它乘法修正是**累乘**关系：``20 × 0.7 × 1.5 = 21``。
        """
        from sts2_sim import core
        state = self.fresh(["thieving_hopper"])
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        player = state.player
        player.add_power("shrink", 1)
        self.assertEqual(core.deal_damage(state, player, enemy, 20, self.events), 14)
        enemy.add_power("vulnerable", 1)
        self.assertEqual(core.deal_damage(state, player, enemy, 20, self.events), 21)

    def test_damage_multipliers_are_not_applied_twice(self):
        """**回归护栏**：易伤 / 虚弱只能乘一次。

        乘法阶段从"写死两条"改成表驱动时，如果忘了删旧的两行，就会乘两次
        （易伤变成 ×2.25）。这个测试专门盯这种"重构留下的双份"。
        """
        from sts2_sim import core
        state = self.fresh(["thieving_hopper"])
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 400
        enemy.add_power("vulnerable", 1)
        self.assertEqual(core.deal_damage(state, state.player, enemy, 20, self.events), 30,
                         "易伤只乘一次 1.5 → 30，乘两次会变成 45")
        state.player.add_power("weak", 1)
        self.assertEqual(core.deal_damage(state, state.player, enemy, 20, self.events), 22,
                         "20×0.75×1.5 = 22.5 → 截断 22")

    def test_surrounded_only_multiplies_from_behind(self):
        """``SurroundedPower``：只在**背后**被打时 ×1.5，正面无效。

        .. code-block:: csharp

           case Direction.Right:
               if (!dealer.HasPower<BackAttackLeftPower>()) return 1m;
               break;
           ...
           return 1.5m;
        """
        from sts2_sim import core
        state = self.fresh(["crusher"])
        boss = state.enemies[0]
        boss.hp = boss.max_hp = 400
        boss.add_power("surrounded", 1)
        boss.power_facing = "right"
        attacker = state.player
        self.assertEqual(core.deal_damage(state, attacker, boss, 20, self.events), 20,
                         "朝右、攻击者没有 back_attack_left → 不增伤")
        attacker.add_power("back_attack_left", 1)
        self.assertEqual(core.deal_damage(state, attacker, boss, 20, self.events), 30,
                         "背后攻击应 ×1.5")

    # ---- 倒计时 / 开关 ----
    def test_nemesis_alternates_intangible(self):
        """``NemesisPower.AfterSideTurnEnd``：**隔回合**切换无形（不是永久免伤）。

        .. code-block:: csharp

           _shouldApplyIntangible = !_shouldApplyIntangible;
           if (_shouldApplyIntangible) Apply<IntangiblePower>(owner, 1);
           else if (owner.HasPower<IntangiblePower>()) Remove(...);

        只写"加"不写"减"会让复仇之影**永久免伤**，那是最难打的怪变成无敌。
        """
        from sts2_sim import powers
        state = self.fresh(["crusher"])
        enemy = state.enemies[0]
        enemy.add_power("nemesis", 1)
        seen = []
        for _ in range(4):
            powers.on_turn_end(enemy, state, self.events)
            seen.append(enemy.power("intangible"))
        self.assertEqual(seen, [1, 0, 1, 0])

    def test_constrict_damages_its_own_owner_unpowered(self):
        """``ConstrictPower.AfterSideTurnEnd``：拥有者回合结束受 ``Amount`` 点**无源**伤害。

        ``ValueProp.Unpowered`` → 不吃力量/易伤，但**可以格挡**。
        """
        from sts2_sim import powers
        state = self.fresh(["crusher"])
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.add_power("constrict", 7)
        powers.on_turn_end(enemy, state, self.events)
        self.assertEqual(enemy.hp, 93)
        enemy.hp, enemy.block = 100, 5
        powers.on_turn_end(enemy, state, self.events)
        self.assertEqual(enemy.hp, 98, "格挡应当抵消一部分")
        self.assertEqual(enemy.block, 0)

    def test_hatch_is_a_pure_countdown(self):
        """``HatchPower.AfterSideTurnEnd``：只递减，行为在别处（孵化）。"""
        from sts2_sim import powers
        state = self.fresh(["crusher"])
        enemy = state.enemies[0]
        enemy.add_power("hatch", 3)
        for _ in range(4):
            powers.on_turn_end(enemy, state, self.events)
        self.assertEqual(enemy.power("hatch"), 0)

    def test_painful_stabs_adds_wounds(self):
        """``PainfulStabsPower.AfterAttack``：**自己**打出有效攻击 → 往玩家弃牌堆塞伤口。"""
        from sts2_sim import core
        state = self.fresh(["crusher"])
        enemy = state.enemies[0]
        enemy.add_power("painful_stabs", 1)
        core.deal_damage(state, enemy, state.player, 5, self.events)
        wounds = [c.cid for c in state.discard if c.cid == "wound"]
        self.assertEqual(len(wounds), 1)
        # 玩家打敌人**不**触发（只有拥有者造成伤害才触发）
        core.deal_damage(state, state.player, enemy, 5, self.events)
        self.assertEqual(sum(1 for c in state.discard if c.cid == "wound"), 1)

    def test_personal_hive_adds_dazed_to_draw_pile(self):
        """``PersonalHivePower``：被**卡牌**打中 → 往玩家**抽牌堆随机位置**塞 ``Amount`` 张眩晕。

        位置必须是随机的：放牌堆顶会让"下次抽牌必中眩晕"，强度完全不同。
        所以只断言"进了抽牌堆"，不断言位置。
        """
        from sts2_sim import core
        state = self.fresh(["entomancer"])
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        enemy.add_power("personal_hive", 2)
        core.deal_damage(state, state.player, enemy, 5, self.events)
        self.assertEqual(sum(1 for c in state.draw_pile if c.cid == "dazed"), 2)

    def test_back_attack_markers_are_registered(self):
        """``back_attack_left`` / ``back_attack_right`` 是**纯标记**能力，供 ``surrounded`` 查询。

        它们自己没有任何钩子（源码里一个 override 都没有），所以"注册了但看起来
        什么都不做"是**正确**的 —— 这条测试把这个意图钉住，免得后人以为漏实现了。
        """
        from sts2_sim import powers
        for pid in ("back_attack_left", "back_attack_right"):
            rules = powers.RULES[pid]
            self.assertIn(pid, powers.IMPLEMENTED)
            self.assertIsNone(rules.on_owner_turn_end)
            self.assertIsNone(rules.on_attacked)

    # ---- 眩晕 / 召唤原语 ----
    def test_stun_skips_exactly_one_turn(self):
        """``CreatureCmd.Stun``：**只跳过一次**行动，之后恢复。

        ⚠️ 引擎把"眩晕"表达成"意图指向招式表里不存在的 id" ——
        ``_run_enemy_turn`` 找不到招式就不施加效果，正是"被晕住"。
        写成"移除意图"会让怪**永久不行动**，是另一种游戏。
        """
        from sts2_sim import core, powers
        state = self.fresh(["crusher"])
        enemy = state.enemies[0]
        powers.stun(enemy, state, self.events, source="测试")
        core._choose_intent(state, enemy)
        self.assertEqual(enemy.intent.mid, "STUNNED_MOVE")
        core._run_enemy_turn(state, self.events)
        self.assertEqual(state.player.hp, state.player.max_hp, "眩晕回合不该造成伤害")
        self.assertNotEqual(enemy.intent.mid, "STUNNED_MOVE", "下回合必须恢复正常出招")

    def test_imbalanced_triggers_only_when_fully_blocked(self):
        """``ImbalancedPower.AfterDamageGiven``：自己打出的伤害被**完全格挡**才失衡。

        .. code-block:: csharp

           if (dealer == base.Owner && result.WasFullyBlocked) { ... }

        "打中了但伤害很小"**不**算 —— 条件是"一点血都没掉"。
        """
        from sts2_sim import core
        state = self.fresh(["bowlbug_rock"])
        enemy = state.enemies[0]
        enemy.add_power("imbalanced", 1)
        core.deal_damage(state, enemy, state.player, 5, self.events)
        self.assertIsNone(enemy.power_flags.get("off_balance"), "没被格挡 → 不触发")
        state.player.block = 99
        core.deal_damage(state, enemy, state.player, 5, self.events)
        self.assertEqual(enemy.power_flags.get("off_balance"), 1)

    def test_suck_gains_strength_on_unblocked_damage(self):
        """``SuckPower.AfterAttack``：只有**真的造成掉血**才 +``Amount`` 力量。"""
        from sts2_sim import core
        state = self.fresh(["fossil_stalker"])
        enemy = state.enemies[0]
        enemy.add_power("suck", 2)
        state.player.block = 99
        core.deal_damage(state, enemy, state.player, 5, self.events)
        self.assertEqual(enemy.power("strength"), 0, "被完全格挡 → 不汲取")
        state.player.block = 0
        core.deal_damage(state, enemy, state.player, 5, self.events)
        self.assertEqual(enemy.power("strength"), 2)

    def test_shriek_stuns_when_hp_drops_below_threshold(self):
        """``ShriekPower``：``Amount`` 是**血量阈值**，不是伤害量。

        掉血后 ``CurrentHp <= Amount`` 才击晕自己并移除能力。
        """
        from sts2_sim import core
        state = self.fresh(["crusher"])
        enemy = state.enemies[0]
        enemy.hp = 30
        enemy.add_power("shriek", 25)
        core.deal_damage(state, state.player, enemy, 3, self.events)
        self.assertIsNone(enemy.power_flags.get("stunned"), "血量还高于阈值 → 不晕")
        core.deal_damage(state, state.player, enemy, 5, self.events)
        self.assertEqual(enemy.power_flags.get("stunned"), 1)
        self.assertEqual(enemy.power("shriek"), 0, "触发后应移除自身")

    def test_stock_spawns_a_replacement_on_death(self):
        """``StockPower.AfterDeath``：自己在原地补一只，层数 -1。

        .. code-block:: csharp

           axebot.StockAmount = base.Amount - 1;
           await CreatureCmd.Add(axebot, base.CombatState, base.Owner.Side, base.Owner.SlotName);

        新怪必须**带好意图**（否则会在它自己回合抛"没有出招"）。
        """
        from sts2_sim import powers
        state = self.fresh(["axebot"])
        enemy = state.enemies[0]
        enemy.add_power("stock", 2)
        before = len(state.enemies)
        powers.on_any_death(state, enemy, self.events)
        self.assertEqual(len(state.enemies), before + 1)
        spawned = state.enemies[-1]
        self.assertEqual(spawned.eid, "axebot")
        self.assertEqual(spawned.power("stock"), 1)
        self.assertIsNotNone(spawned.intent, "召唤出来的怪必须有意图")

    def test_infested_spawns_four_stunned_wrigglers(self):
        """``InfestedPower.AfterDeath``：召唤 4 只**眩晕状态**的扭动虫。

        ``wriggler.StartStunned = true`` → 它们先挨打一回合。
        """
        from sts2_sim import powers
        state = self.fresh(["phrog_parasite"])
        enemy = state.enemies[0]
        enemy.add_power("infested", 1)
        before = len(state.enemies)
        powers.on_any_death(state, enemy, self.events)
        spawned = state.enemies[before:]
        self.assertEqual(len(spawned), 4)
        self.assertTrue(all(s.eid == "wriggler" for s in spawned))
        self.assertTrue(all(s.power_flags.get("stunned") for s in spawned),
                        "召唤出的扭动虫应当处于眩晕")

    def test_surprise_spawns_two_gremlins(self):
        """``SurprisePower.AfterDeath``：召唤胖地精 + 鬼祟地精。"""
        from sts2_sim import powers
        state = self.fresh(["gremlin_merc"])
        enemy = state.enemies[0]
        enemy.add_power("surprise", 1)
        before = len(state.enemies)
        powers.on_any_death(state, enemy, self.events)
        eids = [e.eid for e in state.enemies[before:]]
        self.assertEqual(eids, ["fat_gremlin", "sneaky_gremlin"])

    def test_ravenous_eats_an_ally(self):
        """``RavenousPower.AfterDeath``：**队友**阵亡 → 拥有者被击晕并 +``Amount`` 力量。"""
        from sts2_sim import core, powers
        state = self.fresh(["corpse_slug", "corpse_slug"])
        dying, eater = state.enemies
        eater.add_power("ravenous", 2)
        dying.hp = 0
        powers.on_any_death(state, dying, self.events)
        self.assertEqual(eater.power("strength"), 2)
        self.assertEqual(eater.power_flags.get("stunned"), 1)

    def test_plow_clears_temporary_strength_and_stuns(self):
        """``PlowPower``：掉到阈值以下 → 清掉**临时**力量并击晕自己。"""
        from sts2_sim import core
        state = self.fresh(["crusher"])
        enemy = state.enemies[0]
        enemy.hp = 30
        enemy.add_power("plow", 25)
        enemy.add_power("temporary_strength", 4)
        core.deal_damage(state, state.player, enemy, 5, self.events)
        self.assertEqual(enemy.power("temporary_strength"), 0)
        self.assertEqual(enemy.power_flags.get("stunned"), 1)
        self.assertEqual(enemy.power("plow"), 0)

    def test_hardened_shell_is_a_per_turn_budget(self):
        """``HardenedShellPower``：掉血上限是**每回合总预算**，不是"每次压到 1"。

        .. code-block:: csharp

           return Math.Min(amount, base.Amount - damageReceivedThisTurn);

        ``BeforeSideTurnStart`` 把 ``damageReceivedThisTurn`` 清零 → 每回合恢复。
        写成"每次最多 1 点"会让硬壳强十倍（30 点攻击掉 1 而不是 10）。
        """
        from sts2_sim import core
        state = self.fresh(["crusher"])
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 200
        enemy.add_power("hardened_shell", 10)
        self.assertEqual(core.deal_damage(state, state.player, enemy, 30, self.events), 10)
        self.assertEqual(core.deal_damage(state, state.player, enemy, 30, self.events), 0,
                         "本回合预算用完 → 完全不掉血")
        enemy.power_flags["hardened_shell_received"] = 0     # 模拟新回合
        self.assertEqual(core.deal_damage(state, state.player, enemy, 30, self.events), 10)

    def test_paper_cuts_removes_max_hp(self):
        """``PaperCutsPower``：``CreatureCmd.LoseMaxHp`` —— **永久的**最大生命，不是伤害。

        不吃格挡、不吃力量；而且**必须真的掉了血**才触发（``UnblockedDamage > 0``）。
        """
        from sts2_sim import core
        state = self.fresh(["crusher"])
        enemy = state.enemies[0]
        enemy.add_power("paper_cuts", 3)
        before = state.player.max_hp
        core.deal_damage(state, enemy, state.player, 5, self.events)
        self.assertEqual(state.player.max_hp, before - 3)
        state.player.block = 99
        core.deal_damage(state, enemy, state.player, 5, self.events)
        self.assertEqual(state.player.max_hp, before - 3, "被完全格挡 → 不触发")

    # ---- 覆盖率台账 ----
    def test_implemented_enemy_powers_are_registered(self):
        """这 10 个敌方能力必须都进了 ``IMPLEMENTED``（否则卡牌/固有属性会被误判残缺）。"""
        from sts2_sim import powers
        for pid in ("ritual", "high_voltage", "enrage", "crab_rage", "minion",
                    "slippery", "skittish", "rampart",
                    "battleworn_dummy_time_limit", "escape_artist",
                    "flutter", "soar", "shrink", "surrounded"):
            self.assertIn(pid, powers.IMPLEMENTED, f"{pid} 没注册进能力表")

    def test_every_new_power_cites_a_real_source_class(self):
        """新增能力都要带**可解析的源码出处**（``docs/09`` §5.2 铁律 R1）。"""
        import re
        from sts2_sim import powers
        tree = ROOT / "data" / "decompiled" / "sts2"
        index = {p.stem for p in tree.rglob("*.cs")}
        for pid in ("ritual", "high_voltage", "enrage", "crab_rage", "minion",
                    "slippery", "skittish", "rampart",
                    "battleworn_dummy_time_limit", "escape_artist",
                    "flutter", "soar", "shrink", "surrounded"):
            rules = powers.RULES[pid]
            names = re.findall(r"\b([A-Z]\w*Power)\b", rules.source)
            self.assertTrue(names, f"{pid} 的出处里没有类名：{rules.source!r}")
            self.assertTrue(any(n in index for n in names),
                            f"{pid} 的出处类在源码树里不存在：{names}")


if __name__ == "__main__":
    unittest.main()
