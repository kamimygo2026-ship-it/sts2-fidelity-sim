"""遗物钩子的**类别划分与门禁**回归（审计 F11 的同类问题）。

这一组锁住两件容易"看起来在做、其实没做"的事：

1. **记账型钩子不是缺失行为。** ``ArtOfWar.AfterCombatEnd`` 只写自己的私有字段
   （``AnyAttacksPlayedLastTurn = false``），本身不改变任何游戏状态 ——
   它只是给同类里别的钩子记数。把它算成"遗物缺行为"会**高估缺口**：
   实测报出来的 280 处里 **55 处**是这一类。
2. **选牌用途也必须过门禁。** ``royal_stamp``（皇家印章）的 ``AfterObtained``
   是"从牌组里选一张附魔"，而引擎没实现 ``enchant`` 用途 —— 运行期
   ``runeffects`` 只会打一行日志跳过。旧实现把它算成"**完全复刻**"：
   报告说做好了，实际玩起来什么也不做。

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 缺口数虚高 | 把私有计数器重置算成行为 | `test_bookkeeping_hooks_are_not_counted_as_behavior` |
| "完全复刻"的遗物其实无效 | 选牌用途没过门禁 | `test_unsupported_selection_purpose_is_rejected` |
| 类别与钩子名单对不上 | 两处独立维护 | `test_every_unmodeled_hook_has_a_kind` |
"""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

from sts2_sim.content import (
    ENGINE_SELECTION_PURPOSES, RELICS, RUN_LEVEL_TIMINGS, _relic_hook,
)

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"
RELIC_SRC = ROOT / "data" / "decompiled" / "sts2" / "MegaCrit.Sts2.Core.Models.Relics"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestRelicContent(unittest.TestCase):
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

    def test_every_unmodeled_hook_has_a_kind(self):
        """类别表必须与未实现钩子名单**逐项对应**（两处独立维护必然漂移）。"""
        for relic in RELICS.values():
            names = {name for name, _kind in relic.unmodeled_hook_kinds}
            self.assertEqual(names, set(relic.unmodeled_hooks), relic.rid)
            for _name, kind in relic.unmodeled_hook_kinds:
                self.assertIn(kind, ("commands", "query", "bookkeeping", "unknown"),
                              relic.rid)

    def test_bookkeeping_hooks_are_not_counted_as_behavior(self):
        """``bookkeeping`` 不算进 ``effectful_gaps``。"""
        for relic in RELICS.values():
            for name, kind in relic.unmodeled_hook_kinds:
                if kind == "bookkeeping":
                    self.assertNotIn(name, relic.effectful_gaps, relic.rid)
                else:
                    self.assertIn(name, relic.effectful_gaps, relic.rid)

    def test_reverse_verification_unknown_kinds_count_as_real_gaps(self):
        """反向验证：认不出类别的必须**算作真缺口**，不能当记账型放过。"""
        for relic in RELICS.values():
            for name, kind in relic.unmodeled_hook_kinds:
                if kind == "unknown":
                    self.assertIn(name, relic.effectful_gaps, relic.rid)

    def test_bookkeeping_only_flag_is_consistent(self):
        for relic in RELICS.values():
            if relic.bookkeeping_only_gaps:
                self.assertTrue(relic.unmodeled_hooks, relic.rid)
                self.assertFalse(relic.effectful_gaps, relic.rid)

    def test_unsupported_selection_purpose_is_rejected(self):
        """⭐ 选牌用途不在可执行集合里 → 钩子必须被判为缺口，而不是"已复刻"。"""
        body = {
            "effects": [{"op": "select_card", "from": "deck",
                         "purpose": "enchant", "count": 1, "amount": 1,
                         "target": "self"}],
            "unsupported": [], "choices": 0,
            "scope": {"room": "any", "stateful": False, "map_point": False},
        }
        self.assertIsNone(
            _relic_hook("obtained", "AfterObtained", body, {}),
            "enchant 用途引擎没实现，不能被判成已复刻")

    def test_supported_run_selection_purpose_is_adopted(self):
        """反向验证：**支持**的用途（removal）必须照常采纳，别把门禁做成一律拒绝。"""
        body = {
            "effects": [{"op": "select_card", "from": "deck",
                         "purpose": "removal", "count": 1, "amount": 1,
                         "target": "self"}],
            "unsupported": [], "choices": 0,
            "scope": {"room": "any", "stateful": False, "map_point": False},
        }
        hook = _relic_hook("obtained", "AfterObtained", body, {})
        self.assertIsNotNone(hook, "removal 是 Run 层已实现的用途，不该被拒")

    def test_combat_timing_uses_the_combat_purpose_set(self):
        """战斗内时机用战斗侧用途表（两套本来就不同）。

        ``discard`` 只在战斗侧合法：它出现在 Run 层钩子里必须被拒。
        """
        self.assertIn("discard", ENGINE_SELECTION_PURPOSES)
        self.assertNotIn("discard", __import__(
            "sts2_sim.runeffects", fromlist=["RUN_SELECTION_PURPOSES"]
        ).RUN_SELECTION_PURPOSES)
        body = {
            "effects": [{"op": "select_card", "from": "hand",
                         "purpose": "discard", "count": 1, "amount": 1,
                         "target": "self"}],
            "unsupported": [], "choices": 0,
            "scope": {"room": "any", "stateful": False, "map_point": False},
        }
        self.assertIsNone(_relic_hook("obtained", "AfterObtained", body, {}),
                          "Run 层钩子里的战斗侧用途必须被拒")

    def test_run_level_timings_are_declared(self):
        self.assertTrue({"obtained", "combat_end", "combat_victory"}
                        <= set(RUN_LEVEL_TIMINGS))


class TestHookClassifier(unittest.TestCase):
    """分类器本身：直接对着 C# 源码验，防止它退化成恒返回同一类。"""

    @classmethod
    def setUpClass(cls):
        if not RELIC_SRC.exists():
            raise unittest.SkipTest(f"缺源码目录 {RELIC_SRC}")
        cls.tool = _load("extract_relics_under_test",
                         ROOT / "tools" / "extract_relics.py")

    def _classify(self, class_name: str, hook: str) -> str:
        from tools.extract_cards import method_body
        source = (RELIC_SRC / f"{class_name}.cs").read_text(
            encoding="utf-8", errors="replace")
        return self.tool.classify_hook(source, hook, method_body(source, hook))

    def test_pure_counter_reset_is_bookkeeping(self):
        """``ArtOfWar.AfterCombatEnd`` 只重置自己的计数器。"""
        self.assertEqual(self._classify("ArtOfWar", "AfterCombatEnd"),
                         "bookkeeping")

    def test_command_body_is_commands(self):
        """``Lantern.AfterSideTurnStart`` 真的发能量（有命令）。"""
        self.assertEqual(self._classify("Lantern", "AfterSideTurnStart"),
                         "commands")

    def test_query_hook_is_query(self):
        """``TryModify…`` 靠返回值改行为，没有命令也算真缺口。"""
        source = (RELIC_SRC / "PrayerWheel.cs").read_text(
            encoding="utf-8", errors="replace")
        from tools.extract_cards import method_body
        body = method_body(source, "TryModifyRewards")
        if body is None:
            self.skipTest("PrayerWheel.TryModifyRewards 不存在")
        self.assertEqual(self.tool.classify_hook(source, "TryModifyRewards", body),
                         "query")

    def test_classifier_does_not_collapse_to_one_bucket(self):
        """反向验证：分类器必须给出多于一种结果，否则它没有鉴别力。"""
        results = {self._classify("ArtOfWar", "AfterCombatEnd"),
                   self._classify("Lantern", "AfterSideTurnStart")}
        self.assertGreater(len(results), 1,
                           "分类器把所有钩子都归成了一类")

    def test_helper_inlining_is_used_for_classification_only(self):
        """``inline_zero_arg_helpers`` 只做分类用，展开物不进效果抽取。"""
        import inspect
        source = inspect.getsource(self.tool.inline_zero_arg_helpers)
        self.assertIn("只用于分类", inspect.getdoc(
            self.tool.inline_zero_arg_helpers) or "" + source)


class TestNoAdoptedHookDropsACondition(unittest.TestCase):
    """⭐⭐ **本批次最重要的护栏**：已采纳的遗物钩子，源码里不许还有未建模的条件。

    这一条覆盖的是一类**静默变强**：抽取器只认"回合数守卫"和四个跨回合计数器，
    于是下面这些条件整批丢失，钩子被当成**无条件生效**：

    ==================== ========================================== ==========================
    遗物                  源码条件                                    丢掉之后的错误行为
    ==================== ========================================== ==========================
    ``happy_flower``      ``(TurnsSeen + 1) % Turns == 0``            每 3 回合 +1 能量 → **每回合**
    ``kunai``             仅攻击牌 且 ``AttacksPlayedThisTurn % N``    每 3 张 +1 敏捷 → **每张**
    ``war_paint``         ``Type == Skill`` 且取 N 张随机              升级随机 N 张 → 可能升错牌型
    ``parrying_shield``   ``Rng.CombatTargets.NextItem``              打**指定**敌人 → 随机目标
    ==================== ========================================== ==========================

    实测被这一条判出来的有 **42 个遗物** —— 全都已经在模拟器里"无条件生效"了。

    判据与抽取器**同一份**（``tools.extract_relics.STATEFUL_CONDITIONS``），
    避免两处独立维护后漂移。

    ⚠️ 唯一允许忽略的是 ``participants.Contains(base.Owner.Creature)``：
    本模拟器是**单人**环境（``docs/13`` §5 的第一交付范围），玩家永远是自己
    阵营的 participants 成员，这个前缀恒真，丢掉它不改变任何数值。
    """

    @classmethod
    def setUpClass(cls):
        if not CONTENT.exists() or not RELIC_SRC.exists():
            raise unittest.SkipTest("缺内容目录或反编译源码")
        from sts2_sim.featurize import configure_content
        configure_content(str(CONTENT))
        cls.tool = _load("extract_relics_for_conditions",
                         ROOT / "tools" / "extract_relics.py")
        cls.addClassCleanup(cls._restore)

    @staticmethod
    def _restore():
        from sts2_sim.content import load_builtin
        from sts2_sim.featurize import rebuild_vocab
        load_builtin()
        rebuild_vocab()

    @staticmethod
    def _record(rid: str):
        import json
        records = {r["rid"]: r for r in json.loads(
            (CONTENT / "relics_source.json").read_text(encoding="utf-8"))}
        return records.get(rid)

    def _source_body(self, rid: str, timing: str):
        from tools.extract_cards import method_body
        from sts2_sim.content import HOOK_TIMING_BY_NAME
        record = self._record(rid) or {}
        path = RELIC_SRC / f"{record.get('class', '')}.cs"
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in record.get("hooks") or {}:
            if HOOK_TIMING_BY_NAME.get(name) == timing:
                return method_body(text, name)
        return None

    def test_no_adopted_hook_drops_a_condition(self):
        """已采纳的钩子，源码里不许还有**没建模**的条件。

        ⭐ 判据必须是"**剥掉已建模的条件之后**还剩不剩状态条件" ——
        ``Kunai`` / ``Shuriken`` / ``LetterOpener`` / ``OrnamentalFan`` 的取模计数
        已经由 :func:`tools.extract_relics.modulo_guard` 建成了可求值的
        ``every_n_turn`` 守卫（引擎按本回合张数判 ``% N == 0``），
        所以它们**不该**再算"条件丢了"。用原文去匹配 ``STATEFUL_CONDITIONS``
        会把"已经建好"误报成"静默变强"。

        反向也钉住：源码里有取模、而采纳出来的钩子里**没有** ``every_n_turn``
        守卫 —— 那才是真的把条件丢了。
        """
        offenders: list[str] = []
        for rid, relic in RELICS.items():
            record = self._record(rid) or {}
            variables = record.get("vars") or []
            for hook in relic.hooks:
                body = self._source_body(rid, hook.timing)
                if body is None:
                    continue
                guard_body, modulo = self.tool.modulo_guard(body, variables)
                if modulo is not None:
                    kinds = {g[0] for g in hook.guards}
                    self.assertIn("every_n_turn", kinds,
                                  f"{rid}.{hook.timing} 的取模条件没进守卫表")
                # ⚠️ "随机取 N 张不重复"已经由 `random_deck_cards` 完整表达
                # （谓词 + 张数 + **随机流** + 取法），所以"集合谓词 / 取 N 张 /
                # 随机采样"三类文本特征不再算"条件丢了" —— 与
                # `tools.extract_relics.parse_relic` 的剔除逻辑**同一口径**。
                modelled_random = any(e.op in ("random_deck_cards",
                                               "add_random_cards")
                                      for e in hook.effects)
                for pattern, label in self.tool.STATEFUL_CONDITIONS:
                    if not re.search(pattern, guard_body):
                        continue
                    if modelled_random and label in ("collection_predicate",
                                                     "collection_take",
                                                     "random_sample",
                                                     # `NextItem(...)` 的"随机取一个"
                                                     "random_target"):
                        continue
                    offenders.append(f"{rid}.{hook.timing}({label})")
                    break
        self.assertEqual(offenders, [],
                         f"{len(offenders)} 个已采纳的钩子在源码里带未建模条件，"
                         f"会静默变强：{offenders[:10]}")

    def test_known_silent_buffs_are_now_gaps(self):
        """逐个点名：**还没建模**的那几个必须是缺口，不能"已复刻"。

        ⚠️ `war_paint` / `whetstone` 从这张名单里**移出**了（`docs/12` §2.19）：
        它们的条件（"随机取 N 张攻击 / 技能牌"）已经**建出来了** ——
        抽取器产出 `random_deck_cards`（带谓词、**随机流**与取法），
        由引擎的 Run 层按同一条 `niche` 流复刻。见
        `test_random_pick_relics_are_modelled`。
        """
        expected_gap = {
            "happy_flower": "AfterSideTurnStart",
            "fake_happy_flower": "AfterSideTurnStart",
            "iron_club": "AfterCardPlayed",
            "parrying_shield": "AfterSideTurnEnd",
        }
        for rid, hook in expected_gap.items():
            relic = RELICS.get(rid)
            if relic is None:
                continue
            self.assertIn(hook, relic.unmodeled_hooks,
                          f"{rid}.{hook} 的条件建不出来，必须是缺口")

    def test_random_pick_relics_are_modelled(self):
        """已建模的"随机取 N 张"那一族：必须**不再**是缺口，且参数可核对。

        出处（四个逐字相同，见 ``tools.extract_cards.random_deck_picks``）::

            PileType.Deck.GetPile(owner).Cards.Where(c => … && c.Type == CardType.Attack
                                                          && c.IsUpgradable)
                .ToList().StableShuffle(owner.RunState.Rng.Niche)
                .Take(base.DynamicVars.Cards.IntValue);
            foreach (item in enumerable) CardCmd.Upgrade(item);

        ⚠️ **随机流必须照着抽**（`niche`）：换一条流，分布一样但序列与真机分叉。
        """
        from sts2_sim.rng import STREAMS
        expected = {
            "whetstone": "Attack",
            "war_paint": "Skill",
        }
        for rid, card_type in expected.items():
            relic = RELICS.get(rid)
            self.assertIsNotNone(relic, rid)
            self.assertEqual(list(relic.unmodeled_hooks), [],
                             f"{rid} 的随机取牌已建模，不该再算缺口")
            effects = [e for hook in relic.hooks for e in hook.effects
                       if e.op == "random_deck_cards"]
            self.assertEqual(len(effects), 1, rid)
            effect = effects[0]
            self.assertEqual(effect.purpose, "upgrade", rid)
            self.assertEqual(effect.rng, "niche", rid)
            self.assertIn(effect.rng, STREAMS, rid)
            self.assertEqual(effect.pick, "shuffle", rid)
            self.assertEqual(dict(effect.card_filter).get("type"), card_type, rid)

    def test_modelled_every_n_relics_are_no_longer_gaps(self):
        """已经建模的那一族（每 N 张某类牌）：必须**不再**是缺口，且守卫可求值。

        出处（四个逐字相同，见 ``tools.extract_relics.modulo_guard``）：
        ``Kunai.AfterCardPlayed`` / ``Shuriken.AfterCardPlayed`` /
        ``LetterOpener.AfterCardPlayed`` / ``OrnamentalFan.AfterCardPlayed``。
        """
        from sts2_sim import relics as relic_rules
        expected = {
            "kunai": ("attack", 3),
            "shuriken": ("attack", 3),
            "ornamental_fan": ("attack", 3),
            "letter_opener": ("skill", 3),
        }
        for rid, (counter, every) in expected.items():
            relic = RELICS.get(rid)
            self.assertIsNotNone(relic, rid)
            self.assertEqual(list(relic.unmodeled_hooks), [],
                             f"{rid} 的取模条件已建模，不该再算缺口")
            guards = [g for hook in relic.hooks for g in hook.guards
                      if g[0] == "every_n_turn"]
            self.assertEqual(len(guards), 1, rid)
            value = guards[0][1]
            self.assertEqual((value["counter"], value["n"]), (counter, every), rid)
            self.assertIn("every_n_turn", relic_rules.EVALUABLE_GUARDS)

    def test_participants_guard_is_documented_as_single_player_safe(self):
        """``participants.Contains`` 允许被忽略，但必须在代码里写明理由。"""
        import inspect
        doc = inspect.getdoc(self.tool.hook_scope) or ""
        source = inspect.getsource(self.tool)
        self.assertIn("participants", source,
                      "单人恒真的 participants 守卫必须被显式说明，而不是碰巧放过")
        self.assertTrue(doc)

    def test_condition_gate_has_teeth(self):
        """反向验证：门禁必须真的能报出问题（否则它只是恒真的装饰）。"""
        patterns = dict((label, pattern)
                        for pattern, label in self.tool.STATEFUL_CONDITIONS)
        self.assertIn("modulo_counter", patterns)
        self.assertTrue(re.search(patterns["modulo_counter"],
                                  "(TurnsSeen + 1) % base.DynamicVars[\"Turns\"].IntValue"))
        self.assertTrue(re.search(patterns["random_target"],
                                  "RunState.Rng.CombatTargets.NextItem(list)"))
        self.assertTrue(re.search(patterns["per_turn_counter"],
                                  "AttacksPlayedThisTurn += 1"))
        # ⚠️ `SkillsPlayed >= SkillsThreshold`（`TuningFork`）**不由模式捕获** ——
        # 它靠 `unrecognized_guard` 的"解释不了就拒绝"兜住（见下一个测试）。
        # 这正是为什么不能只靠标识符名字：名字可以随便变。
        self.assertTrue(re.search(patterns["per_turn_counter"],
                                  "ActivationCountThisTurn < 1"))
        self.assertTrue(re.search(patterns["persistent_counter"], "TurnsSeen == 0"))
        # 而单人恒真的 participants 守卫**不该**被任何模式命中
        for pattern, label in self.tool.STATEFUL_CONDITIONS:
            self.assertIsNone(
                re.search(pattern, "if (participants.Contains(base.Owner.Creature))"),
                f"participants 守卫被 {label} 误判成状态条件")

    def test_unrecognized_guard_rejects_unknown_conditions(self):
        """``unrecognized_guard`` 是**最可靠**的准入判据：解释不了就不采纳。

        实测它抓出四个"条件丢掉 = 每张牌都触发"的静默变强
        （``RainbowRing`` / ``TuningFork`` / ``HelicalDart`` / ``IvoryTile``），
        而按标识符名字猜的做法**四个全漏**。
        """
        gate = self.tool.unrecognized_guard
        # 认得的：归属 / 上下文 / 单人恒真 / 回合数 / 牌型 / 死亡安全网
        self.assertFalse(gate("if (cardPlay.Card.Owner == base.Owner) { }"))
        self.assertFalse(gate("if (CombatManager.Instance.IsInProgress) { }"))
        self.assertFalse(gate("if (participants.Contains(base.Owner.Creature)) { }"))
        self.assertFalse(gate("if (base.Owner.PlayerCombatState.TurnNumber <= 1) { }"))
        self.assertFalse(gate("if (cardPlay.Card.Type == CardType.Power) { }"))
        self.assertFalse(gate("if (!base.Owner.Creature.IsDead) { }"))
        # 认不出的：私有计数器 / 牌型标签 / 能量阈值 / 集合取样
        self.assertTrue(gate("if (SkillsPlayed >= SkillsThreshold) { }"))
        self.assertTrue(gate("if (cardPlay.Card.Tags.Contains(CardTag.Shiv)) { }"))
        self.assertTrue(gate("if (cardPlay.Resources.EnergyValue >= Th) { }"))
        self.assertTrue(gate("if (AnyAttacksPlayedLastTurn) { }"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
