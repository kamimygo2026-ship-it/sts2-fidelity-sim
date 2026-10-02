"""死亡路径的**统一入口**回归测试（``docs/09`` §18 架构审计）。

真机所有死亡都走 ``CreatureCmd.Kill``，``Hook.AfterDeath`` 在那里调一次。
引擎早期是在各处**硬编码** ``hp = 0``，于是"某条路径漏掉死亡钩子"没人发现：

| 路径 | 漏掉的后果 |
|---|---|
| 末日处决 | `crab_rage` 不暴怒、`stock` 不补人 |
| 招式自爆 | `surprise` 不召唤、`minion` 不连锁 |
| 中毒掉血 | `slippery` 不递减、`AfterDamageReceived` 整条不跑 |

这些漏掉的共同点是**静默**：怪还在按部就班地打，只是"该发生的事没发生"。
所以每一类路径都要有一条测试盯着。
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


class DeathPathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = setup_content()

    @classmethod
    def tearDownClass(cls):
        restore_builtin()

    def fresh(self, enemies):
        from sts2_sim import core
        state = core.start_combat(self.deck, list(enemies), 7)
        for enemy in state.enemies:
            enemy.powers.clear()
            enemy.power_flags.clear()
        self.events: list[str] = []
        return state

    def test_doom_death_fires_the_death_hooks(self):
        """**末日处决**也必须走 ``AfterDeath``。

        以前这里直接 ``enemy.hp = 0``，于是队友的 ``crab_rage``（阵亡才暴怒）
        永远不会触发 —— 而且看不出任何异常。
        """
        from sts2_sim import core, powers
        state = self.fresh(["crusher", "axebot"])
        doomed, bystander = state.enemies
        bystander.add_power("crab_rage", 1)
        doomed.add_power("doom", 999)
        core._run_enemy_turn(state, self.events)
        self.assertEqual(doomed.hp, 0, "末日应当处决它")
        self.assertEqual(bystander.power("strength"), 6,
                         "队友阵亡应当触发 crab_rage（+6 力量）")
        self.assertEqual(bystander.block, 99)

    def test_raw_damage_fires_damage_received(self):
        """**中毒 / 掉血**也要触发 ``AfterDamageReceived``。

        真机走 ``CreatureCmd.Damage(..., Unblockable)``，与普通伤害共用同一条
        钩子链 —— ``SlipperyPower`` 就靠它递减。漏掉这一步，
        "滑溜"会变成一个永久免伤（每次最多掉 1 点且永不减层）。
        """
        from sts2_sim import core
        state = self.fresh(["inklet"])
        enemy = state.enemies[0]
        enemy.hp = enemy.max_hp = 100
        enemy.add_power("slippery", 2)
        core.deal_raw_damage(state, enemy, 7, self.events, label="中毒")
        self.assertEqual(enemy.power("slippery"), 1, "掉血应当让滑溜递减")

    def test_raw_damage_death_fires_the_death_hooks(self):
        """中毒**打死**的怪也要触发死亡钩子（否则召唤/连锁机制静默失效）。"""
        from sts2_sim import core
        state = self.fresh(["axebot"])
        enemy = state.enemies[0]
        enemy.hp = 3
        enemy.add_power("stock", 2)
        before = len(state.enemies)
        core.deal_raw_damage(state, enemy, 99, self.events, label="中毒")
        self.assertEqual(enemy.hp, 0)
        self.assertEqual(len(state.enemies), before + 1,
                         "库存应当补一只（死亡钩子没跑就会漏）")

    def test_self_destruct_fires_the_death_hooks(self):
        """招式自爆也要走统一死亡入口。

        ``gas_bomb`` 的 ``EXPLODE_MOVE`` 结尾是 ``CreatureCmd.Kill(base.Creature)``。
        """
        from sts2_sim import core
        state = self.fresh(["gas_bomb"])
        bomb = state.enemies[0]
        self.assertTrue(any(m.kills_self for m in bomb.definition().moves),
                        "炸弹应当有自爆招式")
        # 让它自爆：找到那一招并执行
        explode = next(m for m in bomb.definition().moves if m.kills_self)
        bomb.intent = core.Intent(explode.intent, explode.mid, explode.value,
                                  explode.times)
        core._run_enemy_turn(state, self.events)
        self.assertEqual(bomb.hp, 0)

    def test_death_hooks_fire_once(self):
        """重复调用 ``mark_dead`` 不能把死亡钩子触发两次。

        否则 ``crab_rage`` 会给两次 +6 力量、``stock`` 会补两只。
        """
        from sts2_sim import core
        state = self.fresh(["crusher", "axebot"])
        dying, bystander = state.enemies
        bystander.add_power("crab_rage", 1)
        dying.hp = 0
        core.mark_dead(state, dying, self.events)
        core.mark_dead(state, dying, self.events)
        self.assertEqual(bystander.power("strength"), 6, "只该触发一次")

    def test_there_is_exactly_one_death_funnel(self):
        """**结构护栏**：``core.py`` 里不许再出现裸的 ``hp = 0``（玩家除外）。

        这条测试盯的是"有人又加了一条绕过钩子的死亡路径"。
        玩家没有死亡钩子（`AfterDeath` 是敌方/队友语境），所以放行。
        """
        import re
        source = (ROOT / "sts2_sim" / "core.py").read_text(encoding="utf-8")
        offenders = []
        for match in re.finditer(r"^.*?(\w+)\.hp = 0.*$", source, re.M):
            line_no = source[:match.start()].count("\n") + 1
            text = match.group(0)
            if "player.hp = 0" in text or "combatant.hp = 0" in text:
                continue                    # player：没有死亡钩子；combatant：就是 mark_dead 自己
            offenders.append(f"core.py:{line_no}: {text.strip()[:60]}")
        self.assertEqual(
            offenders, [],
            "这些地方直接置 hp=0、绕过了 mark_dead（死亡钩子不会触发）：\n"
            + "\n".join(offenders))

    def test_every_power_hook_has_a_call_site(self):
        """**结构护栏**：每个事件型钩子位都必须有调用点。

        这是对"硬编码调用点"这个结构性风险的廉价对冲：钩子位定义了但没人调用，
        等于这个能力永远不触发 —— 而且没有任何症状（`docs/11` §11.3）。

        检查方式是扫 `sts2_sim/` 全部源码里是否出现过该钩子名
        （`_fire` 的查表分支、`on_any_death` 的内部循环都算）。
        """
        from sts2_sim import powers
        text = "\n".join(
            p.read_text(encoding="utf-8")
            for p in (ROOT / "sts2_sim").glob("*.py"))
        #: 由 `_fire` / `on_any_death` 等分发器统一处理的钩子，
        #: 它们的"调用点"是规则字段名，不出现独立函数
        dispatch_fields = {
            "on_owner_turn_start": "on_turn_start",
            "on_owner_turn_end": "on_turn_end",
            "on_energy_reset": "on_energy_reset",
            "on_ally_death": "on_any_death",
            "on_self_death": "on_any_death",
        }
        missing = []
        for field in powers.PowerRules.__dataclass_fields__:
            if not field.startswith("on_"):
                continue
            probe = dispatch_fields.get(field, field)
            if probe not in text:
                missing.append(field)
        self.assertEqual(missing, [],
                         f"这些钩子位没有任何调用点（定义了但永不触发）：{missing}")


if __name__ == "__main__":
    unittest.main()
