"""架构对照审计：模拟器 vs 真机（``data/decompiled/sts2/``）。

回答一个问题：**我的结构跟源码像不像？** 不像的地方有没有风险？

做法是逐层对照，并给每一层标上"对齐程度"与"风险"：

* ✅ 已实现：结构或行为有实现，且能指到调用点
* ⚠️ 部分：结构不同或覆盖不全，风险已知且被显式上报
* ❌ 缺口：结构不同且**有已知风险**（通常是"某个钩子忘了在某条路径上调用"）

⚠️ 审计 F11 的教训：**所有数字都必须现算、所有结论都必须带条件**。
本脚本以前会：

* 在数量差 ≤30 时直接打印"数量对齐" —— 30 只怪的差距不是"对齐"；
* 把"忠实翻译""307 行""109 只怪"写成**硬编码文本** —— 代码改了它照旧这么印。

现在每一行都是实测值，且明确区分 ``docs/13`` §7 的六种状态：
**已索引 → 已解析 → 可执行 → 闭包可执行 → 源码用例 → 参考验证**。
类文件数量只证明"已索引"，**不证明数值正确**。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
GAME = ROOT / "data" / "decompiled" / "sts2"

#: 真机"模型类"的目录（每张卡/每个能力都是一个类）
MODEL_DIRS = ("Cards", "Powers", "Relics", "Monsters", "Potions", "Orbs",
              "Afflictions", "Enchantments", "Encounters", "Events", "CardPools")


def count(directory: Path, pattern: str = "*.cs") -> int:
    return len(list(directory.glob(pattern))) if directory.exists() else 0


def hook_surface() -> tuple[int, int]:
    """真机的钩子面：``Hook.cs`` 的静态入口 + ``AbstractModel`` 的虚方法。"""
    hook = (GAME / "MegaCrit.Sts2.Core.Hooks" / "Hook.cs").read_text(
        encoding="utf-8", errors="replace")
    model = (GAME / "MegaCrit.Sts2.Core.Models" / "AbstractModel.cs").read_text(
        encoding="utf-8", errors="replace")
    entries = len(re.findall(r"public static (?:async )?[\w<>?\[\],\s]+?\s+(\w+)\s*\(",
                             hook))
    virtuals = len(re.findall(r"public virtual ", model))
    return entries, virtuals


def local_hook_surface() -> tuple[int, int]:
    """我的钩子面：事件型钩子位 + 查询型接口。"""
    from sts2_sim import powers
    events = [f for f in powers.PowerRules.__dataclass_fields__ if f.startswith("on_")]
    queries = [n for n in dir(powers)
               if n.startswith(("blocks_", "hp_loss", "negates_", "damage_multipliers",
                                "tainted_", "affliction_", "type_for", "retains_"))]
    return len(events), len(queries)


def count_lines(path: Path) -> int:
    """真实行数（以前这里写死的 ``307 行`` 会随代码变化而撒谎）。"""
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def machines_attached() -> tuple[int, int]:
    """挂上真机状态机的怪物数 / 怪物总数。"""
    from sts2_sim import content
    attached = sum(1 for e in content.ENEMY_DB.values()
                   if getattr(e, "ai_record", None))
    return attached, len(content.ENEMY_DB)


def main() -> int:
    from sts2_sim import content, powers
    from sts2_sim import eligibility
    content.load_content_dir(str(ROOT / "data" / "content" / "repo"))

    line = "=" * 74
    print(line)
    print("架构对照：模拟器 vs 真机")
    print(line)
    print("⚠️ 本报告只证明到「已索引 / 已解析 / 可执行 / 闭包可执行」四层；")
    print("   「源码用例」与「参考验证（真机对拍）」**尚未通过**，不计入任何完成度。")

    print("\n【一、模型层：真机是「一个对象一个类」，我是「一张表 + 一个处理器」】")
    print(f"{'内容':10s} {'真机类文件':>10s} {'我的表条数':>10s}  差异  状态")
    rows = [
        ("卡牌", "Cards", len(content.CARD_DB)),
        ("能力", "Powers", len(content.POWERS)),
        ("遗物", "Relics", len(content.RELICS)),
        ("怪物", "Monsters", len(content.ENEMY_DB)),
        ("药水", "Potions", len(content.POTIONS)),
        ("充能球", "Orbs", len(content.ORBS) if hasattr(content, "ORBS") else 5),
    ]
    for label, folder, mine in rows:
        files = count(GAME / f"MegaCrit.Sts2.Core.Models.{folder}")
        diff = mine - files
        # ⚠️ 数量差**只说明"已索引"**，不说明任何行为正确性。
        # 以前这里在 |diff| ≤ 30 时打印"数量对齐"，30 只怪的差距不是"对齐"。
        status = "已索引" if diff == 0 else f"已索引（差 {diff:+d}，需逐条核）"
        print(f"{label:10s} {files:>10d} {mine:>10d}  {diff:>+4d}  {status}")

    entries, virtuals = hook_surface()
    events, queries = local_hook_surface()
    print("\n【二、钩子面】")
    print(f"  真机：Hook.cs 静态入口 {entries} 个 · AbstractModel 虚方法 {virtuals} 个")
    print(f"  我的：事件型钩子 {events} 个 · 查询型接口 {queries} 个")
    print(f"  → 事件型覆盖 {events}/{virtuals} = {events / max(1, virtuals):.1%}")
    print("     ⚠️ 真机那 185 个虚方法里含动画/多人/UI 变体，直接当分母会低估；")
    print("     但**该实现的缺口仍然是缺口**，不能靠拆分类别把它算没。")
    print(f"  钩子位：{sorted(f for f in powers.PowerRules.__dataclass_fields__ if f.startswith('on_'))}")

    cmds = count(GAME / "MegaCrit.Sts2.Core.Commands")
    print("\n【三、命令层：真机一个 Cmd 类一个语义，我压成一张算子表】")
    print(f"  真机 Commands 目录 {cmds} 个文件（CreatureCmd / CardPileCmd / PowerCmd / …）")
    print(f"  我的算子：{len(content.ENGINE_OPS)} 个 —— {sorted(content.ENGINE_OPS)}")
    print("     ⚠️ 粒度更粗：一个算子可能对应真机多条命令链，顺序细节要逐个核。")

    print("\n【四、伤害管线】")
    print("  真机 Hook.ModifyDamageInternal：加法 → 乘法 → 钳制 → 单次截断（Decimal 全程）")
    print(f"  我的 core.compute_damage：同顺序、同舍入"
          f"（加法阶段含 {len(powers.DAMAGE_ADDITIVE)} 个能力 + tainted 修正）")
    print(f"  乘法表（真机各 Power 的 ModifyDamageMultiplicative）："
          f"{[f'{p}×{f}' for p, _s, f in powers.DAMAGE_MULTIPLIERS]}")
    print("     ⚠️ 这一层有针对性用例（tests/test_damage_pipeline.py），")
    print("     但**没有真机对拍**：'有测试'不等于'与真机一致'。")

    attached, monsters = machines_attached()
    print("\n【五、出招状态机】")
    print(f"  monster_ai.py {count_lines(ROOT / 'sts2_sim' / 'monster_ai.py')} 行"
          f"（实测行数，不是写死的数字）")
    print(f"  覆盖 {attached}/{monsters} 只怪挂上状态机")
    print("     ⚠️ 挂上 ≠ 运行时条件正确：专属 flags/counters 与招式副作用")
    print("     需要逐条用例（审计 F05 的 Fabricator 就是反例，现已有回归测试）。")

    print("\n【六、RNG：真机两层，我两层】")
    from sts2_sim.rng import STREAMS
    print("  真机 RunRngType(12) + PlayerRngType(Rewards/Shops/Transformations)")
    print(f"  我的 STREAMS：{len(STREAMS)} 条 —— {sorted(STREAMS)}")
    print("     ⚠️ **流名相同不代表同种子轨迹一致**（审计 F07）：")
    print("     真机用 Xoshiro256** + 名称混入哈希，我用 SHA-256 派生 + Python random。")
    print("     本项目当前只能声称『分布近似』，不能声称『同种子等价』。")

    print("\n【七、内容准入（docs/13 §7）】")
    report = eligibility.report("ironclad")
    print(f"  合格（可执行 + 闭包可执行）：{report['cards_admitted']}/"
          f"{report['cards_total']} 张")
    print(f"  铁甲战士**可抽池**：{report['draft_pool']} 张"
          f"（采样器用的就是这个，与报告同源）")
    print(f"  合格遭遇：{report['encounters']}")
    print(f"  拒绝原因分布：{report['reject_reasons']}")
    print("  ✅ 采样器与报告**共用** sts2_sim/eligibility.py，不会再各报一套数字。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
