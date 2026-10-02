"""从反编译源码抽取药水效果（``docs/09`` L5）。

药水与卡牌**结构完全同构**，所以直接复用卡牌抽取器的机械：

======================  ======================  ==============================
卡牌                    药水                    说明
======================  ======================  ==============================
``OnPlay(ctx, play)``   ``OnUse(ctx, target)``  效果方法
``CanonicalVars``       ``CanonicalVars``       同一套 ``DynamicVar`` / ``PowerVar<T>``
``CardType``            ``PotionUsage``         何时可用（战斗内 / 任意时候）
``TargetType``          ``TargetType``          目标
======================  ======================  ==============================

目标由**声明的** ``TargetType`` 决定，不去猜 ``target`` 形参到底指谁：
单人游戏里 ``AnyPlayer`` 就是自己、``AnyEnemy`` 就是敌人。

用法
----
    python tools/extract_potions.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.extract_cards import (  # noqa: E402
    apply_value_props, build_key_map, enum_tail, extract_effects, method_body,
    parse_vars, property_expr, snake,
)

SRC_ROOT = Path("data/decompiled/sts2/MegaCrit.Sts2.Core.Models.Potions")
OUT_DEFAULT = Path("data/content/repo/potions_source.json")
CODEX_DEFAULT = Path("data/content/repo/potions.json")


def property_block(source: str, name: str) -> str | None:
    """取**属性**的块体（``method_body`` 只认带括号的方法，属性匹配不到）。"""
    import re
    match = re.search(rf"\b{name}\s*\{{", source)
    if not match:
        return None
    start = match.end() - 1
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1:index]
    return None


def conditional_enum_values(source: str, prop: str) -> list[str]:
    """属性体是**条件分支**时，按出现顺序收集它引用的枚举值。

    ``FoulPotion.TargetType``（``FoulPotion.cs:37-46``）会按"是否在战斗中"
    分别返回 ``AllEnemies`` 与 ``TargetedNoCreature``。``property_expr`` 只认
    单个表达式，于是这瓶药被**静默丢掉** —— 实测 65 个药水类只抽出 64 个，
    而社区库有 65 个（缺的正是 ``foul_potion``）。

    这里不猜语义，只把**源码里写了哪几种值**如实列出来；选哪一个由调用方
    按"本引擎在什么语境下使用药水"决定，并把这件事记进缺口。
    """
    import re
    block = property_block(source, prop)
    if block is None:
        return []
    values: list[str] = []
    # 属性名可能出现在返回类型位置（`TargetType.TargetedNoCreature`），
    # 也可能只写枚举值（`return AllEnemies;`）—— 两种写法都要认。
    for match in re.finditer(rf"(?:{prop}\.)?(\w+)\s*;", block):
        token = match.group(1)
        if token in ("get", "set", "return", "break", "continue"):
            continue
        if token not in values:
            values.append(token)
    return values


def combat_branch(body: str | None) -> tuple[str | None, bool]:
    """把 ``OnUse`` 里的**战斗内分支**切出来，并报告是否还有别的分支。

    出处：``FoulPotion.OnUse``（``FoulPotion.cs:74-120``）有三个分支::

        if (CombatManager.Instance.IsInProgress) { 对全体敌人造成伤害 }
        else if (CurrentRoom is MerchantRoom)     { 获得金币（商人交互） }
        else (FakeMerchant 事件)                   { 同上 }

    ⚠️ 把整个方法体丢给效果抽取器会把三个分支**混在一起**：战斗内用这瓶药
    会一边打伤害一边白拿 100 金币 —— 静默的强度变化，而且日志一切正常。
    所以只抽战斗内那一段，其余分支记为未建模。

    返回 ``(战斗内分支文本 或 None, 是否还有其它分支)``。
    """
    if body is None:
        return None, False
    marker = "CombatManager.Instance.IsInProgress"
    start = body.find(marker)
    if start < 0:
        return body, False
    open_brace = body.find("{", start)
    if open_brace < 0:
        return body, False
    depth = 0
    for index in range(open_brace, len(body)):
        char = body[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return body[open_brace + 1:index], True
    return body, False


def parse_potion(path: Path, report: collections.Counter) -> dict | None:
    source = path.read_text(encoding="utf-8", errors="replace")
    class_name = path.stem

    rarity = property_expr(source, "Rarity")
    usage = property_expr(source, "Usage")
    target_type = property_expr(source, "TargetType")
    declaration_gaps: list[str] = []
    if target_type is None:
        # 条件属性：按"战斗内优先"取值（本引擎的"使用药水"是战斗内动作），
        # 并把"这是个条件声明"如实记进缺口 —— 不静默丢掉整瓶药。
        values = conditional_enum_values(source, "TargetType")
        if values:
            target_type = next((v for v in values if "Enemies" in v or "Enemy" in v),
                               values[0])
            declaration_gaps.append(
                "TargetType 是按战斗状态分支的条件属性，取值 " + "/".join(values)
                + f"；这里采用战斗内分支 {target_type}")
            report["conditional_target_type"] += 1
    if rarity is None or target_type is None:
        report["potion_without_declaration"] += 1
        return None

    variables = parse_vars(source)
    key_map = build_key_map(variables)
    for variable in variables:
        if variable["base"] is None:
            report[f"var_without_literal::{variable['name'][:24]}"] += 1

    # 目标：由**声明的** TargetType 决定，不解析形参
    declared = enum_tail(target_type) if target_type else ""
    target = "enemy" if "enemy" in declared.lower() else "self"

    on_use = method_body(source, "OnUse")
    combat_only, branched = combat_branch(on_use)
    if branched:
        declaration_gaps.append(
            "OnUse 含非战斗分支（商人/事件交互），本引擎只建模战斗内分支")
        report["combat_branch_only"] += 1
    effects, unsupported, choices = extract_effects(combat_only, key_map)
    # 药水与充能球的伤害/格挡多半带 `ValueProp.Unpowered`：把变量上的声明
    # 传染给效果，否则玩家的力量会把药水伤害加成（实测火焰药水 20 → 25）。
    apply_value_props(effects, variables)
    # 药水的目标由 TargetType 统一决定：把效果里判成 unknown/self 的一并纠正
    for effect in effects:
        if target == "enemy" and effect.get("target") in ("unknown", "self", "enemy"):
            effect["target"] = "enemy"
        elif target == "self" and effect.get("target") in ("unknown", "enemy"):
            effect["target"] = "self"
    for command in set(unsupported):
        report[f"unsupported_command::{command}"] += 1
    if on_use is None:
        report["potion_without_on_use"] += 1
    if not effects:
        report["potion_without_effects"] += 1

    # ⚠️ **申报缺口也要进 `unsupported`**：`content._load_potions` 就是按它
    # 决定 `effects_incomplete` 的。只记在报告里而没进记录，等于这瓶药
    # 带着"只建模了战斗内分支"的事实进入训练集 —— 静默的强度偏差。
    unsupported = list(unsupported) + declaration_gaps

    return {
        # ⚠️ 必须用 `snake()`：类名是驼峰（`AttackPotion`），而社区库的 id 是
        # 下划线（`attack_potion`）。直接 `.lower()` 会让 64 个药水**一个都对不上**
        # —— 而且不报错，只是全都变成"社区库没有的新药水"。
        "pid": snake(class_name),
        "class": class_name,
        "rarity": (enum_tail(rarity) if rarity else ""),
        "usage": (enum_tail(usage) if usage else ""),
        "target_type": declared,
        "effects": effects,
        "vars": variables,
        "unsupported": sorted(set(unsupported)),
        "choice_commands": choices,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从反编译源码抽取药水效果")
    parser.add_argument("--src", default=str(SRC_ROOT))
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--codex", default=str(CODEX_DEFAULT))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f"没有源码目录：{src}")
        return 3
    files = sorted(src.glob("*.cs"))
    report: collections.Counter = collections.Counter()
    potions = [p for p in (parse_potion(f, report) for f in files) if p]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(potions, ensure_ascii=False, indent=1), encoding="utf-8")

    with_effects = [p for p in potions if p["effects"]]
    clean = [p for p in with_effects if not p["unsupported"] and not p["choice_commands"]]
    print(f"扫描 {len(files)} 个药水类，抽取 {len(potions)} 个")
    print(f"  有可执行效果        {len(with_effects)}")
    print(f"  且无未支持/选牌     {len(clean)}")
    print(f"  稀有度：{collections.Counter(p['rarity'] for p in potions).most_common()}")
    print(f"  可用时机：{collections.Counter(p['usage'] for p in potions).most_common()}")
    print(f"  目标：{collections.Counter(p['target_type'] for p in potions).most_common()}")

    unsupported = {k.split("::", 1)[1]: v for k, v in report.items()
                   if k.startswith("unsupported_command::")}
    if unsupported and not args.quiet:
        print("\n未支持的命令（按出现次数）：")
        for name, count in sorted(unsupported.items(), key=lambda kv: -kv[1]):
            print(f"    {count:4d}  {name}")

    codex_path = Path(args.codex)
    if codex_path.exists():
        codex = {c["id"].lower() for c in
                 json.loads(codex_path.read_text(encoding="utf-8"))}
        ours = {p["pid"] for p in potions}
        print(f"\n与社区库对账：社区库 {len(codex)} / 源码 {len(ours)}")
        print(f"  源码有、社区库没有：{sorted(ours - codex)[:8]}")
        print(f"  社区库有、源码没有：{sorted(codex - ours)[:8]}")
    print(f"\n写出 {out}（{len(potions)} 个）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
