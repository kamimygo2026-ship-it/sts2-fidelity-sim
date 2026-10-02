"""遗物层的回归测试（``docs/09`` L5）。

遗物是**最容易静默出错**的一层：它不在卡牌里、不在怪物里，效果全靠钩子触发，
而钩子的**时机**和**条件极性**都在反编译源码的函数体里。错了不会报错，
只会让数值整体偏掉。这张表列出已经踩过的坑：

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 提灯"第 2 回合起"每回合 +1 能量 | 把**包含式**条件 `if (TurnNumber <= 1) { 给能量 }` 当成了排除式守卫 | `TestGuardPolarity` |
| 弹珠袋永远不上易伤 | 同上（条件不满足时被当成"命中则不生效"） | `test_bag_of_marbles_applies_on_turn_one` |
| 锚的 10 点格挡开局就没了 | 回合开始无条件清格挡；真机对**玩家第 1 回合豁免** | `test_anchor_block_survives_turn_one` |
| 准备背包变成"第 2 回合起多抽 2 张" | 把 early-return 守卫读反 | `test_bag_of_preparation_only_turn_one` |
| 贤者之石/弹珠袋只影响第一个敌人 | `all_enemies` 被塌缩成单体目标 | `test_all_enemies_targets_every_enemy` |
| 增删遗物要改引擎代码，漏了不报错 | 引擎里硬编码了 3 个遗物 id | `test_engine_has_no_hardcoded_relics` |
| 面包（Bread）只拿到好处没拿到代价 | 一个遗物的数值半边被采纳、另一半没实现 | `test_partially_modeled_relic_is_not_adopted` |

**铁律**：遗物一旦有战斗阶段的钩子没复刻出来，它的数值部分（每回合能量/抽牌）
就不能采纳 —— 只拿一半会让遗物比真机更强或更弱。
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


def new_combat(deck, enemies=("aeonglass",), relics=(), room="monster", hp=80):
    from sts2_sim import core
    return core.start_combat(deck, list(enemies), 7, player_hp=hp,
                             relics=relics, room=room)


def next_turn(state):
    """推进到下一个玩家回合（把当前手牌丢掉，跳过敌人出招）。"""
    from sts2_sim import core
    state.discard.extend(state.hand)
    state.hand = []
    state.turn += 1
    events: list[str] = []
    core.start_player_turn(state, events)
    return events


class RelicDataTest(unittest.TestCase):
    """数据层：299 个遗物都要有定义，且关键字段来自源码而不是猜。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()

    @classmethod
    def tearDownClass(cls):
        restore_builtin()

    def test_all_relics_loaded(self):
        from sts2_sim import content
        self.assertGreaterEqual(len(content.RELICS), 290)
        missing = [r for r in ("vajra", "anchor", "lantern", "bag_of_marbles")
                   if r not in content.RELICS]
        self.assertEqual(missing, [], f"基础遗物缺失：{missing}")

    def test_unconditional_energy_relics(self):
        """无条件 +1 能量：贤者之石 / 添水 / 灵质。"""
        from sts2_sim import content, relics
        for rid in ("philosophers_stone", "sozu", "ectoplasm", "velvet_choker"):
            definition = content.RELICS.get(rid)
            self.assertIsNotNone(definition, f"{rid} 缺失")
        # 天鹅绒项圈有战斗阶段的钩子（每回合 6 张上限）没复刻 → 数值不采纳
        self.assertEqual(relics.energy_bonus(["sozu", "ectoplasm"], 1), 2)
        self.assertEqual(relics.energy_bonus(["velvet_choker"], 1), 0,
                         "有战斗阶段缺口的遗物不能只采纳数值半边")

    def test_guard_polarity_is_recorded_from_source(self):
        """两个方向的守卫都要按源码判定：包含式 vs 排除式。

        源码里这两种写法都真实存在，判反了整批遗物会反着生效：

        * ``Lantern``：``if (… && TurnNumber <= 1) { 给能量 }`` → 包含式（第 1 回合）
        * ``TwistedFunnel``：``if (… || TurnNumber > 1) { return; }`` → 排除式（第 1 回合）
        * ``BagOfPreparation``：``if (TurnNumber > 1) return count;`` → 排除式（第 1 回合）
        """
        from sts2_sim import content
        lantern = [h for h in content.RELICS["lantern"].hooks
                   if h.timing == "after_side_turn_start"][0]
        self.assertEqual([(g[0], g[1], g[2]) for g in lantern.guards],
                         [("turn_lte", 1, "require")])
        funnel = [h for h in content.RELICS["twisted_funnel"].hooks
                  if h.timing == "before_side_turn_start"][0]
        self.assertEqual([(g[0], g[1], g[2]) for g in funnel.guards],
                         [("turn_gt", 1, "skip_when")])
        prep = content.RELICS["bag_of_preparation"].hand_draw
        self.assertEqual(prep[0], "Cards")
        self.assertEqual(prep[2], (("turn_gt", 1, "skip_when"),))

    def test_no_relic_silently_loses_its_numeric_effect(self):
        """**核心不变量**：带数值钩子的遗物，要么被采纳，要么被报出来。

        没有这条，一个"条件认不出"的遗物会静默地不加能量/不多抽牌 ——
        训练时它只是一张"没有遗物"，没人会发现。三种结局必须覆盖全集：

        * 采纳（``max_energy``/``hand_draw`` 非空）
        * 报了但没采纳（进 ``relics_unresolved_numeric``）
        * 源码里根本没有这个钩子（两者都没有，但原始数据里也没有该字段）
        """
        from sts2_sim import content
        summary = content.load_content_dir(str(CONTENT))
        unresolved = set(summary.get("relics_unresolved_numeric") or [])
        records = json.loads((CONTENT / "relics_source.json").read_text(encoding="utf-8"))
        silent: list[str] = []
        for record in records:
            rid = record["rid"]
            for field in ("max_energy", "hand_draw"):
                if not record.get(field):
                    continue                     # 源码里没有这个钩子
                adopted = getattr(content.RELICS[rid], field) is not None
                if not adopted and f"{rid}.{field}" not in unresolved:
                    silent.append(f"{rid}.{field}")
        self.assertEqual(silent, [], f"这些遗物的数值被静默丢弃：{silent}")
        # 抽样确认这条不变量有鉴别力（真的存在被报出来的条目）
        self.assertIn("pocketwatch.hand_draw", unresolved)
        self.assertIn("pendulum.hand_draw", unresolved)

    def test_unresolved_guards_are_never_adopted(self):
        """跨回合计数器（``TurnsSeen`` / ``KindleCount``）算不出 → 不许生效。

        ⚠️ 两种"不生效"都要覆盖，而且都必须**可见**：

        * **不采纳**：``pocketwatch`` / ``ring_of_the_drake`` —— 进
          ``relics_unresolved_numeric``，字段是 ``None``
        * **采纳但算不出**：``pumpkin_candle``（守卫是 ``kindle_lte``）——
          字段有值，但求值时返回 0，并且要出现在 ``relics.unevaluable()`` 里

        只测第一种会让第二种（"看起来实现了、其实每回合都算 0"）漏过去。
        """
        from sts2_sim import content, relics
        summary = content.load_content_dir(str(CONTENT))
        unresolved = summary.get("relics_unresolved_numeric") or []
        # ⚠️ 这一组是**当前**确实算不出的条目（会随抽取器能力变化而变），
        # 所以只做"必须被报出来"的断言，不写死全部名单。
        for entry in ("pocketwatch.hand_draw", "ring_of_the_drake.hand_draw",
                      "bread.max_energy", "velvet_choker.max_energy"):
            self.assertIn(entry, unresolved,
                          f"{entry} 的条件战斗内算不出或被阻挡，必须报出来而不是猜")
        # ⭐ 反向验证：抽取器**能**解释守卫之后，数值必须真的被采纳。
        # `booming_conch` 的 `ModifyHandDraw` 守卫只是 `TurnNumber > 1`（"第 1 回合多抽"），
        # 外加一个单人恒真的 `participants` 前缀与精英房范围 —— 这些现在都认得出来，
        # 所以它**不该**继续躺在"算不出"名单里（那会让一条正确行为被静默丢掉）。
        self.assertNotIn("booming_conch.hand_draw", unresolved,
                         "守卫已经能解释了，就不该再算作算不出")
        self.assertIsNotNone(content.RELICS["booming_conch"].hand_draw)
        for name in ("pocketwatch", "ring_of_the_drake", "bread", "velvet_choker"):
            self.assertIsNone(content.RELICS[name].hand_draw)
            self.assertIsNone(content.RELICS[name].max_energy)
        # `paels_flesh` 是**采纳**的例子，而且应当算得对：
        # 它的四个钩子（`BeforeCombatStart` / `BeforeSideTurnStart` /
        # `AfterSideTurnStart` / `AfterCombatEnd`）在源码里**只做显示与状态**
        # （`InvokeDisplayAmountChanged` / `RelicStatus` / `Flash`），没有任何机制。
        # 所以"第 3 回合起 +1 能量"可以安全采纳 —— 守卫 `turn_lt 3` 算得出。
        self.assertEqual([relics.energy_bonus(["paels_flesh"], t) for t in (1, 2, 3, 4)],
                         [0, 0, 1, 1], "第 3 回合起才 +1 能量")
        # 采纳但算不出：字段有值，但求值为 0，且被列出来
        candle = content.RELICS.get("pumpkin_candle")
        if candle is not None and candle.max_energy is not None:
            self.assertEqual(relics.energy_bonus(["pumpkin_candle"], 3), 0,
                             "算不出的守卫不该让遗物白白生效")
            self.assertIn("pumpkin_candle.max_energy",
                          relics.unevaluable(["pumpkin_candle"]),
                          "采纳但算不出的遗物必须能被列出来（不许静默算 0）")

    def test_relic_vars_are_resolved(self):
        """具名变量写法 ``new EnergyVar("GainEnergy", 1)`` 的值必须抽到。

        ⚠️ 只把第一个实参当数字会把具名形式的**名字和数值双双丢掉**，
        77 处具名变量、66 张卡受影响。
        """
        from sts2_sim import content
        self.assertEqual(content.RELICS["lantern"].values.get("Energy"), 1)
        self.assertEqual(content.RELICS["vajra"].values.get("Strength"), 1)
        self.assertEqual(content.RELICS["anchor"].values.get("Block"), 10)


class RelicCombatTest(unittest.TestCase):
    """战斗内行为：时机、条件、目标。"""

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()

    @classmethod
    def tearDownClass(cls):
        restore_builtin()

    def test_vajra_grants_strength_at_combat_start(self):
        state = new_combat(self.deck, relics=["vajra"])
        self.assertEqual(state.player.power("strength"), 1)
        # 第 2 回合不应再给（它不是"每回合"效果）
        next_turn(state)
        self.assertEqual(state.player.power("strength"), 1)

    def test_anchor_block_survives_turn_one(self):
        """锚的 10 点格挡：真机 ``Creature.AfterTurnStart`` 对**玩家第 1 回合**豁免清格挡。

        无条件清零会让**所有**开局格挡遗物静默失效（锚、假锚、铜鳞的格挡版本…）。
        """
        state = new_combat(self.deck, relics=["anchor"])
        self.assertEqual(state.player.block, 10)
        next_turn(state)
        self.assertEqual(state.player.block, 0, "第 2 回合起格挡应被清空")

    def test_bag_of_preparation_only_turn_one(self):
        state = new_combat(self.deck, relics=["bag_of_preparation"])
        self.assertEqual(len(state.hand), 7, "第 1 回合应抽 5+2 张")
        next_turn(state)
        self.assertEqual(len(state.hand), 5, "第 2 回合只应抽 5 张")

    def test_big_mushroom_draws_fewer_on_turn_one(self):
        """大蘑菇：第 1 回合**少抽 2 张**（``sign = -1``）。

        ⚠️ 负号不能丢，否则一个纯负面遗物会变成正面遗物。
        """
        state = new_combat(self.deck, relics=["big_mushroom"])
        self.assertEqual(len(state.hand), 3, "第 1 回合应抽 5-2 张")
        next_turn(state)
        self.assertEqual(len(state.hand), 5)

    def test_lantern_energy_only_turn_one(self):
        """提灯：**只在第 1 回合** +1 能量。

        条件 `TurnNumber <= 1` 是**包含式**的。当成排除式守卫会得到
        "第 2 回合起每回合 +1 能量" —— 强度高一个数量级。
        """
        from sts2_sim import core
        state = new_combat(self.deck, relics=["lantern"])
        self.assertEqual(state.energy, core.BASE_ENERGY + 1)
        next_turn(state)
        self.assertEqual(state.energy, core.BASE_ENERGY)

    def test_akabeko_vigor_only_turn_one(self):
        state = new_combat(self.deck, relics=["akabeko"])
        self.assertEqual(state.player.power("vigor"), 8)
        next_turn(state)
        self.assertEqual(state.player.power("vigor"), 8, "不应叠加到 16")

    def test_bag_of_marbles_applies_on_turn_one(self):
        """弹珠袋：第 1 回合给**所有**敌人上 1 层易伤。"""
        state = new_combat(self.deck, enemies=("architect",), relics=["bag_of_marbles"])
        self.assertEqual(state.enemies[0].power("vulnerable"), 1)

    def test_all_enemies_targets_every_enemy(self):
        """``all_enemies`` 必须展开到**每一个**敌人。

        塌缩成单体目标会让"给全体敌人 +1 力量"（贤者之石）变成只给第一个 ——
        日志照常打印"获得 1 点力量"。
        """
        state = new_combat(self.deck, enemies=("architect", "axebot"),
                           relics=["philosophers_stone"])
        self.assertEqual(len(state.enemies), 2)
        for enemy in state.enemies:
            self.assertEqual(enemy.power("strength"), 1,
                             f"{enemy.name} 没拿到贤者之石的力量")

    def test_artifact_absorbs_relic_debuff(self):
        """神器抵消遗物上的负面能力（``aeonglass`` 固有神器 3）。"""
        state = new_combat(self.deck, enemies=("aeonglass",), relics=["bag_of_marbles"])
        enemy = state.enemies[0]
        self.assertEqual(enemy.power("vulnerable"), 0, "神器应抵消易伤")
        self.assertEqual(enemy.power("artifact"), 2, "神器应减 1 层")

    def test_room_scoped_relics(self):
        """房间种类决定生效：赤備只在精英战、万用表只在 Boss 战。"""
        normal = new_combat(self.deck, relics=["sling_of_courage"], room="monster")
        elite = new_combat(self.deck, relics=["sling_of_courage"], room="elite")
        self.assertEqual(normal.player.power("strength"), 0)
        self.assertEqual(elite.player.power("strength"), 2)
        # 万用表是 Boss 战回血 25：Boss 战会顶到上限，普通战不变
        boss = new_combat(self.deck, relics=["pantograph"], room="boss", hp=50)
        plain = new_combat(self.deck, relics=["pantograph"], room="monster", hp=50)
        self.assertEqual(boss.player.hp, 75)
        self.assertEqual(plain.player.hp, 50)

    def test_powers_relic_grants_are_routed_through_power_rules(self):
        """新采纳的这批开局遗物都要真的生效（不是"加载了但没用"）。"""
        state = new_combat(self.deck, relics=[
            "bronze_scales", "gorget", "oddly_smooth_stone",
            "data_disk", "sword_of_jade", "fake_anchor"])
        self.assertEqual(state.player.power("thorns"), 3)
        self.assertEqual(state.player.power("plating"), 4)
        self.assertEqual(state.player.power("dexterity"), 1)
        self.assertEqual(state.player.power("focus"), 1)
        self.assertEqual(state.player.power("strength"), 3)
        self.assertEqual(state.player.block, 4)

    def test_engine_has_no_hardcoded_relics(self):
        """引擎里**不许**再出现硬编码的遗物 id。

        旧版把 ``vajra`` / ``anchor`` / ``bag_of_preparation`` 三个 id 写死在
        ``core._apply_start_of_combat_relics`` 里 —— 增删遗物要改引擎，
        漏掉的遗物不报错、只是静默不生效。
        """
        source = (ROOT / "sts2_sim" / "core.py").read_text(encoding="utf-8")
        for rid in ("vajra", "anchor", "bag_of_preparation", "bronze_scales"):
            self.assertNotIn(f'"{rid}"', source,
                             f"core.py 里硬编码了遗物 {rid}，应改为数据驱动")

    def test_run_layer_passes_room_kind_and_relics_to_combat(self):
        """Run 层必须把**房间种类**和**遗物列表**都传进战斗。

        ``sling_of_courage``（精英战 +2 力量）与 ``pantograph``（Boss 战回血 25）
        靠房间种类区分；不传的话遗物在、效果不在，而且不报错。
        """
        from sts2_sim import run as run_module
        captured: list[dict] = []
        original = run_module.start_combat

        def spy(deck, encounter, **kwargs):
            captured.append({"room": kwargs.get("room"),
                             "relics": kwargs.get("relics")})
            return original(deck, encounter, **kwargs)

        run_module.start_combat = spy
        try:
            env = run_module.RunEnv(seed=0, attempt_budget=0)
            env.reset()
            state = env._state
            nodes = {n.kind: n.node_id for n in state.map.nodes}
            for kind in ("monster", "elite", "boss"):
                if kind not in nodes:
                    continue
                captured.clear()
                env._enter_node(nodes[kind])
                self.assertTrue(captured, f"{kind} 节点没有开战")
                self.assertEqual(captured[-1]["room"], kind,
                                 f"{kind} 节点传了错误的房间种类")
                self.assertIsNotNone(captured[-1]["relics"],
                                     "遗物列表必须传给战斗")
        finally:
            run_module.start_combat = original

    def test_relics_do_not_leak_into_other_combats(self):
        """遗物列表只在 ``start_combat`` 里设定；不传就是不生效。"""
        bare = new_combat(self.deck)
        self.assertEqual(bare.player.power("strength"), 0)
        self.assertEqual(bare.player.block, 0)
        self.assertEqual(len(bare.hand), 5)
        self.assertEqual(bare.relics, ())

    def test_unevaluable_is_reported_not_silent(self):
        """算不出的遗物必须能被列出来（不许静默跳过）。

        ``unevaluable`` 针对的是"**采纳了但条件算不出**"这一格；
        "条件算不出所以没采纳"那一格由 ``relics_unresolved_numeric`` 兜住
        （见 ``test_no_relic_silently_loses_its_numeric_effect``）。
        两格加起来必须覆盖全部，中间不能有静默的洞。
        """
        from sts2_sim import content, relics
        # 被挡在门外的（条件算不出 / 战斗阶段有缺口）→ 压根不采纳
        self.assertIsNone(content.RELICS["pocketwatch"].hand_draw)
        self.assertEqual(relics.unevaluable(["pocketwatch"]), [],
                         "没采纳的遗物不该出现在 unevaluable 里")
        # 采纳的必须真的算得出
        self.assertEqual(relics.energy_bonus(["sozu"], 1), 1)
        self.assertEqual(relics.unevaluable(["sozu"]), [])
        # 未实现的钩子要能列出来（sozu 的"不能再获得药水"没复刻）
        self.assertEqual(relics.unimplemented(["sozu"]), ["sozu"])
        self.assertEqual(relics.unimplemented(["vajra", "anchor"]), [])


class EveryNRelicTest(unittest.TestCase):
    """「每打出 N 张某类牌触发一次」的一族（`Kunai` / `Shuriken` /
    `LetterOpener` / `OrnamentalFan`）。

    四个遗物的源码**逐字相同**（只有最后那一条效果不同）::

        if (cardPlay.Card.Owner == base.Owner && CombatManager.Instance.IsInProgress
            && cardPlay.Card.Type == CardType.Attack)
        {
            AttacksPlayedThisTurn++;                                  // 先自增
            int intValue = base.DynamicVars.Cards.IntValue;
            if (AttacksPlayedThisTurn % intValue == 0) { …效果… }     // 再判取模
        }

    这里钉住三件事：

    1. **恰好**每 N 张触发一次（不是每张，也不是只触发一次）；
    2. 计数按**牌型**分开（`OrnamentalFan` 不该被技能牌推进）；
    3. 计数**每回合清零**（源码的计数器叫 ``…ThisTurn``，引擎同义）。
    """

    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()

    @classmethod
    def tearDownClass(cls):
        restore_builtin()

    def _combat(self, relics, hand):
        from sts2_sim.core import CardInstance
        state = new_combat(self.deck, relics=relics)
        state.hand = [CardInstance(cid) for cid in hand]
        state.draw_pile = [CardInstance("defend_ironclad") for _ in range(10)]
        state.discard = []
        state.energy = 99
        return state

    def _play_all(self, state, hand_size):
        """按顺序把手牌全打出去（攻击牌指定 0 号敌人）。"""
        from sts2_sim.core import Action, step
        for index in range(hand_size):
            definition = state.hand[0].definition()
            target = -1 if definition.target == "self" else 0
            step(state, Action("play_card", 0, target))

    def test_kunai_gains_dexterity_every_third_attack(self):
        """苦无：第 3、6 张攻击牌各 +1 敏捷；第 1、2、4、5 张不加。"""
        state = self._combat(["kunai"], ["strike_ironclad"] * 6)
        self._play_all(state, 6)
        self.assertEqual(state.player.power("dexterity"), 2)

    def test_shuriken_gains_strength_every_third_attack(self):
        state = self._combat(["shuriken"], ["strike_ironclad"] * 3)
        self._play_all(state, 3)
        self.assertEqual(state.player.power("strength"), 1)

    def test_ornamental_fan_blocks_every_third_attack(self):
        """华丽折扇：第 3 张攻击牌 +4 格挡。

        ⚠️ 前两张**不能**给格挡 —— 少了取模守卫就会变成"每张牌 +4"。
        """
        state = self._combat(["ornamental_fan"], ["strike_ironclad"] * 2)
        self._play_all(state, 2)
        self.assertEqual(state.player.block, 0, "前两张攻击牌不该给格挡")
        state = self._combat(["ornamental_fan"], ["strike_ironclad"] * 3)
        self._play_all(state, 3)
        self.assertEqual(state.player.block, 4)

    def test_letter_opener_counts_skills_and_hits_all_enemies(self):
        """拆信刀：第 3 张**技能**牌对**所有**敌人造成 5 点伤害（``ValueProp.Unpowered``）。"""
        state = new_combat(self.deck, enemies=("architect", "axebot"),
                           relics=["letter_opener"])
        from sts2_sim.core import CardInstance
        state.hand = [CardInstance("defend_ironclad") for _ in range(3)]
        state.energy = 99
        before = [enemy.hp for enemy in state.enemies]
        self._play_all(state, 3)
        self.assertEqual([enemy.hp for enemy in state.enemies],
                         [hp - 5 for hp in before])

    def test_attack_and_skill_counters_are_separate(self):
        """牌型分开计数：技能牌不推进「攻击牌计数」，反之亦然。"""
        state = self._combat(["kunai"], ["defend_ironclad"] * 3)
        self._play_all(state, 3)
        self.assertEqual(state.player.power("dexterity"), 0,
                         "技能牌不该推进苦无的攻击计数")

    def test_counter_resets_each_turn(self):
        """计数器每回合清零（源码的 ``AttacksPlayedThisTurn``）。"""
        from sts2_sim.core import CardInstance
        state = self._combat(["kunai"], ["strike_ironclad"] * 2)
        self._play_all(state, 2)
        self.assertEqual(state.player.power("dexterity"), 0)
        next_turn(state)
        state.hand = [CardInstance("strike_ironclad")]
        state.energy = 99
        self._play_all(state, 1)
        self.assertEqual(state.player.power("dexterity"), 0,
                         "跨回合累计会让第 3 张攻击牌在新回合就触发")

    def test_unevaluable_guard_is_not_adopted(self):
        """引擎算不出的守卫必须**在加载期被挡住**，而不是运行期静默跳过。

        少这一道闸门就会出现"报告说完全复刻、实际什么都不做"
        （``relics.hooks_at`` 见到 ``_applies(...) is not True`` 只是 continue）。
        """
        from sts2_sim import content
        body = {"effects": [{"op": "gain_energy", "amount": 1,
                             "amount_var": None, "power": None,
                             "target": "self", "times": 1, "amount_raw": "1m"}],
                "unsupported": [], "choices": 0,
                "scope": {"room": "any", "all_rooms": [], "map_point": [],
                          "dead_check": False, "stateful": [],
                          "turn_guards": [{"kind": "made_up_guard", "value": 1,
                                           "polarity": "require"}]}}
        self.assertIsNone(content._relic_hook("after_card_played", "AfterCardPlayed",
                                              body, {}),
                          "算不出的守卫必须让整个钩子被拒")
        # 反过来：引擎算得出的守卫（`every_n_turn`）要能过闸门。
        body["scope"]["turn_guards"] = [
            {"kind": "every_n_turn",
             "value": {"counter": "attack", "n": 3, "var": "Cards"},
             "polarity": "require"}]
        self.assertIsNotNone(content._relic_hook("after_card_played",
                                                "AfterCardPlayed", body, {}))


if __name__ == "__main__":
    unittest.main()
