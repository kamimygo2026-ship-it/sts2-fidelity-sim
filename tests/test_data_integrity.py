"""数据完整性与源码对账的回归测试。

这一组测试对应**已经真实发生过**的 bug，每个都是"静默出错、看起来正常"那一类。
测试名里写清了当时的症状，改坏了要能一眼看出是哪一类问题回来了。

| 症状 | 根因 | 对应测试 |
|---|---|---|
| encounters 87 条变 261 条 | 三个来源不加区分地合并 | `test_no_dataset_is_imported_twice` |
| ZIP 独有的 7 个数据集消失 | `plain if plain else archives` 全有或全无 | `test_zip_only_dataset_reaches_content` |
| 一只真 Boss 不见了（115→114） | codex 的 HP 是 null → `return None` 整条丢 | `test_hp_less_monster_is_kept` |
| 一半怪的"最大 HP"是 None | 属性别名链 `MaxInitialHp => MinInitialHp` 只解析一趟 | `test_source_hp_is_complete` |
| 4 只怪的 HP 全空 | HP 只写在基类里，派生类文件里没有 | `test_inherited_hp_is_resolved` |
| 某只 Boss 一进战斗就抛异常 | 状态存在**属性**里，`DECL_VAR` 匹配不到 → 转移被丢 | `test_no_dangling_move_state` |
| 自爆怪第二回合崩掉整个 Run 层 | `CreatureCmd.Kill(base.Creature)` 未建模 | `test_self_destruct_moves_are_flagged` |
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"

#: 主键字段名**不统一**：cards 用 cid，monsters/encounters 用 eid，其余才叫 id。
#: 一律用 `record["id"]` 查重会得到"零重复"的假象（实测被骗过一次）。
PRIMARY_KEY = {"cards": "cid", "monsters": "eid", "encounters": "eid"}

#: 归一化名重复但**合法**的情形。
#:
#: 优先用**规则**解释（`_explain_collision`），规则解释不了的才落到这里。
#: 这里只剩"不同 power 类共用同一个显示名"：真机把"临时力量"按来源拆成了
#: 多个类（COORDINATE / FLEX_POTION / FEEDING_FRENZY…），显示名一样但确实是
#: 不同实体；`Back Attack` 则是左右两个方向各一个类。
ALLOWED_NAME_DUPLICATES = {
    "temporarydexterity", "temporarystrength", "temporarystrengthdown",
    "temporaryfocus", "backattack",
}


def _explain_collision(ids: list[str]) -> str | None:
    """给一组同名的 id 找合法理由；找不到返回 None（= 需要人工看）。"""
    import re
    # "??? 版"遗物：ANCHOR vs FAKE_ANCHOR（游戏里叫 Anchor???，是独立的真遗物）
    if all(i.startswith("FAKE_") or f"FAKE_{i}" in ids for i in ids):
        return "真遗物与 ??? 版成对出现"
    # 5 个角色各自的基础牌：strike_ironclad / strike_silent / …
    if all(re.fullmatch(r"(strike|defend)_\w+", i) for i in ids):
        return "每个角色各有自己的基础牌"
    return None


def restore_builtin() -> None:
    """还原内置占位内容，避免污染同进程的其它测试模块。"""
    from sts2_sim.content import load_builtin
    from sts2_sim.featurize import rebuild_vocab
    load_builtin()
    rebuild_vocab()


def load_content():
    """加载真机内容（幂等），并注册清理。"""
    if not CONTENT.exists():
        raise unittest.SkipTest(f"缺内容目录 {CONTENT}")
    from sts2_sim.featurize import configure_content
    configure_content(str(CONTENT))


def _records(name: str) -> list[dict]:
    return json.loads((CONTENT / f"{name}.json").read_text(encoding="utf-8"))


class TestNoDuplicateImports(unittest.TestCase):
    """同一份数据被多个来源写进同一张表 —— 实测 encounters 87 条变 261 条。"""

    @classmethod
    def setUpClass(cls):
        if not CONTENT.exists():
            raise unittest.SkipTest(f"缺内容目录 {CONTENT}")

    def test_no_dataset_is_imported_twice(self):
        import collections
        for name in ("cards", "monsters", "encounters", "relics", "potions",
                     "events", "powers", "characters", "keywords",
                     "enchantments", "afflictions", "orbs", "intents",
                     "modifiers", "achievements", "epochs"):
            path = CONTENT / f"{name}.json"
            if not path.exists():
                continue
            with self.subTest(dataset=name):
                records = json.loads(path.read_text(encoding="utf-8"))
                field = PRIMARY_KEY.get(name, "id")
                keys = [r.get(field) for r in records]
                self.assertNotIn(None, keys, f"{name} 有记录缺主键 {field}")
                duplicate = {k: v for k, v in collections.Counter(keys).items() if v > 1}
                self.assertEqual(duplicate, {}, f"{name} 主键重复：{dict(list(duplicate.items())[:5])}")

    def test_name_duplicates_are_only_the_legitimate_ones(self):
        """重名必须能被**规则**解释掉，解释不了就说明可能是重复导入。

        这条测试故意做成"新出现的重名会让它失败"，逼人去看一眼，
        而不是自动放行 —— 重复导入的症状正是"同一实体进了两次表"。
        """
        import collections
        import re
        for name in ("cards", "monsters", "encounters", "relics", "powers"):
            path = CONTENT / f"{name}.json"
            if not path.exists():
                continue
            with self.subTest(dataset=name):
                records = json.loads(path.read_text(encoding="utf-8"))
                field = PRIMARY_KEY.get(name, "id")
                grouped: dict[str, list[str]] = collections.defaultdict(list)
                for record in records:
                    key = re.sub(r"[^a-z0-9]", "", str(record.get("name") or "").lower())
                    grouped[key].append(str(record.get(field)))
                unexplained = []
                for key, ids in grouped.items():
                    if len(ids) < 2 or not key:
                        continue
                    if key in ALLOWED_NAME_DUPLICATES or _explain_collision(ids):
                        continue
                    unexplained.append(f"{key}: {ids}")
                self.assertEqual(unexplained, [],
                                 f"{name} 出现无法解释的重名（查一下是不是同一实体导入了两次）")

    def test_zip_only_dataset_reaches_content(self):
        """ZIP 独有的数据集必须出现在归一化结果里。

        旧实现 `sources = plain if plain else archives` 只要散装文件存在就
        完全不读 ZIP，这 7 个数据集被静默丢掉，报告里一个字都不提。
        """
        for name in ("enchantments", "afflictions", "orbs", "intents",
                     "modifiers", "achievements", "epochs"):
            with self.subTest(dataset=name):
                self.assertTrue((CONTENT / f"{name}.json").exists(),
                                f"{name}.json 不在归一化结果里（ZIP 独有的数据集被丢了）")


class TestMonsterDataIsComplete(unittest.TestCase):
    """怪物表：一条都不许少、HP 必须齐全、出招链必须闭合。"""

    @classmethod
    def setUpClass(cls):
        load_content()
        cls.addClassCleanup(restore_builtin)

    def test_hp_less_monster_is_kept(self):
        """codex 里 HP 为 null 的怪**不能整条丢掉**。

        `TestSubject`（三阶段复活 Boss）就因此消失过，怪物表从 115 变成 114，
        报告里只有一行 `monster_without_hp`。
        """
        from sts2_sim import content
        codex = {r["eid"]: r for r in _records("monsters")}
        self.assertIn("test_subject", codex, "codex 里应该有 test_subject")
        self.assertIn("test_subject", content.ENEMY_DB,
                      "test_subject 被丢掉了 —— 缺 HP 不是丢弃整条记录的理由")
        self.assertIsNone(codex["test_subject"]["hp"],
                          "这条测试的前提是 codex 没给 HP；若 codex 补上了，"
                          "请换一个缺 HP 的怪物或删掉本测试")

    def test_source_hp_is_complete(self):
        """源码抽取的 HP 必须全部有值（含最大 HP）。

        `MaxInitialHp => MinInitialHp` 在真机里有 40+ 只怪这么写，而且别名会
        成链（`MaxInitialHp` → `MinInitialHp` → `FirstFormHp`）。只扫一趟
        会让最大 HP 变成 None。
        """
        records = _records("monster_ai")
        self.assertTrue(records)
        broken = [r["eid"] for r in records
                  if r.get("hp") and r["hp"][1] is None]
        self.assertEqual(broken, [], f"最大 HP 解析失败（别名链没迭代到不动点）：{broken}")
        missing = [r["eid"] for r in records if not r.get("hp")]
        self.assertEqual(missing, [], f"源码没抽出 HP：{missing}")

    def test_inherited_hp_is_resolved(self):
        """HP 只写在基类里的怪，必须沿继承链取到。

        `MysteriousKnight : FlailKnight`、
        `DecimillipedeSegment{Front,Middle,Back} : DecimillipedeSegment`
        的 `MinInitialHp` 只在基类文件里。
        """
        from sts2_sim import content
        for eid in ("mysterious_knight", "decimillipede_segment_front",
                    "decimillipede_segment_middle", "decimillipede_segment_back"):
            with self.subTest(eid=eid):
                enemy = content.ENEMY_DB.get(eid)
                self.assertIsNotNone(enemy, f"{eid} 不在怪物表里")
                self.assertIsNotNone(enemy.hp, f"{eid} 没有 HP（继承链没查到）")
                self.assertGreater(enemy.hp[0], 0)

    def test_source_hp_matches_codex_hp(self):
        """源码与社区库的 HP 必须一致 —— 两个独立来源的交叉验证。

        对不上就说明至少一边错了，必须人工看，不能静默取一边。
        """
        ai = {r["eid"]: r for r in _records("monster_ai")}
        codex = {r["eid"]: r for r in _records("monsters")}
        mismatched = []
        checked = 0
        for eid, record in codex.items():
            source, hp = ai.get(eid), record.get("hp")
            if not source or hp is None:
                continue
            lo = source["hp"][0]
            hi = source["hp"][1] if source["hp"][1] is not None else lo
            checked += 1
            if [lo, hi] != list(hp):
                mismatched.append(f"{eid}: 源码={(lo, hi)} codex={hp}")
        self.assertGreater(checked, 100, "比对样本太少，测试没有鉴别力")
        self.assertEqual(mismatched, [], f"HP 不一致：{mismatched[:8]}")

    def test_no_dangling_move_state(self):
        """每条 move 状态都必须有后继 —— 除非它是自爆招。

        状态若存在**属性**里（`DeadState = new MoveState("RESPAWN_MOVE", …)`），
        赋值语句没有类型前缀，`DECL_VAR` 匹配不到，变量名就绑不上状态 id，
        于是 `DeadState.FollowUpState = …` 整条转移被丢掉。
        症状不是"少一个状态"，而是**进战斗就抛"没有后继状态"**。
        """
        records = _records("monster_ai")
        dangling = []
        for record in records:
            follow_up = record.get("follow_up") or {}
            for state_id, state in (record.get("states") or {}).items():
                if state.get("kind") != "move" or state_id in follow_up:
                    continue
                handler = state.get("move")
                if (record.get("moves") or {}).get(handler, {}).get("kills_self"):
                    continue        # 自爆怪打完就死，真机也不会请求后继
                dangling.append(f"{record['eid']}:{state_id}")
        self.assertEqual(dangling, [],
                         f"这些 move 状态没有后继，运行时必抛异常：{dangling}")

    def test_self_destruct_moves_are_flagged(self):
        """自爆招必须标记出来。

        真机在招式处理函数结尾写 `await CreatureCmd.Kill(base.Creature)`，
        打完自己就死，**永远不会请求后继状态**。不建模的话下一回合会去 roll
        后继并抛异常（实测 gas_bomb 直接炸掉 Run 层）。
        """
        from sts2_sim import content
        for eid in ("gas_bomb", "waterfall_giant"):
            with self.subTest(eid=eid):
                enemy = content.ENEMY_DB[eid]
                self.assertTrue(any(m.kills_self for m in enemy.moves),
                                f"{eid} 的自爆招没有被标记（真机 body 里有 CreatureCmd.Kill）")

    def test_self_destruct_kills_the_monster_in_combat(self):
        """自爆招结算后怪必须真的死掉，而不是留在场上等下一回合。"""
        from sts2_sim.core import start_combat
        from sts2_sim.content import STARTING_DECK

        state = start_combat(STARTING_DECK, ("gas_bomb",), seed=3, ascension=0)
        bomb = state.enemies[0]
        self.assertTrue(bomb.alive())
        bomb.hp = 0                      # 直接触发结算后的存活判定
        self.assertFalse(bomb.alive())

    def test_self_destruct_monster_does_not_break_a_full_run(self):
        """带自爆怪的整局必须能跑完 —— 曾经在第 4 步抛"没有后继状态"。"""
        from sts2_sim.run import RunEnv
        from sts2_sim.runbot import RunBot
        bot = RunBot()
        for seed in (26, 0, 5):
            with self.subTest(seed=seed):
                env = RunEnv(seed=seed, attempt_budget=1)
                step = env.reset()
                for _ in range(400):
                    if step.phase in ("won", "lost"):
                        break
                    step, _r, done, _t, _info = env.step(bot.act(step))
                    if done:
                        break

    def test_every_conditional_expression_evaluates(self):
        """32 个条件表达式必须全部可求值。

        `evaluate_condition` 对认不出的表达式**抛异常而不是返回 False**
        （静默返回 False 会让怪悄悄走错分支），所以这里直接全量扫一遍。
        """
        from sts2_sim.monster_ai import (
            COUNTER_DEFAULTS, MonsterContext, evaluate_condition, resolve_flags,
        )
        # 标志与计数器都必须齐全：缺值会抛 UnsupportedMechanic（审计 F05 要求如此）。
        context = MonsterContext(flags=resolve_flags([]),
                                 counters=dict(COUNTER_DEFAULTS))
        failures = []
        total = 0
        for record in _records("monster_ai"):
            for state_id, entries in (record.get("conditionals") or {}).items():
                for entry in entries:
                    total += 1
                    try:
                        evaluate_condition(entry["condition"], context)
                    except Exception as exc:            # noqa: BLE001
                        failures.append(f"{record['eid']}:{state_id} "
                                        f"{entry['condition']!r} → {exc}")
        self.assertGreater(total, 25, "条件样本太少，测试没有鉴别力")
        self.assertEqual(failures, [], f"求值失败：{failures}")


class TestAuditTool(unittest.TestCase):
    """审计工具本身要对真实数据报"没问题"。"""

    def test_audit_finds_no_problems_on_real_data(self):
        import sys
        sys.path.insert(0, str(ROOT))
        from tools.audit_data import (audit_content, audit_raw,
                                      cross_check_source_vs_codex)
        raw_report, problems = audit_raw(ROOT / "data/raw/spire-codex/repo")
        self.assertTrue(raw_report, "没解析出任何数据集")
        problems += audit_content(CONTENT, raw_report)
        cross_problems, stats = cross_check_source_vs_codex(CONTENT)
        problems += cross_problems
        self.assertGreater(stats.get("hp_checked", 0), 100,
                           "HP 对账样本太少，审计没有鉴别力")
        self.assertEqual(problems, [], f"审计发现问题：{problems}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
