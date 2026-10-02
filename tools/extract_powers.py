"""从反编译源码抽取能力表（``docs/09`` L2）。

**为什么需要它**：社区库 ``powers.json`` 只有 257 个能力，而源码里有 **268 个
能力类** —— 少的 11 个是卡牌专属能力（``HyperbeamFocusDownPower`` 等），
于是 ``Hyperbeam`` 这类卡一抽取出来就会报"未知 power"，lint 直接失败。

叠加语义与正负性**沿继承链取**（与怪物 HP 同一手法）：
``HyperbeamFocusDownPower : TemporaryFocusPower`` 自己不声明 ``StackType``，
只看自己的文件会取不到。

用法
----
    python tools/extract_powers.py     # 抽取 + 与社区库对账
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

SRC_ROOT = Path("data/decompiled/sts2/MegaCrit.Sts2.Core.Models.Powers")
OUT_DEFAULT = Path("data/content/repo/powers_source.json")
CODEX_DEFAULT = Path("data/content/repo/powers.json")

#: 抽象类是别的能力的基类，不是独立能力，不该进能力表。
def snake(text: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()


def power_id_of(class_name: str) -> str:
    name = class_name[:-5] if class_name.endswith("Power") else class_name
    return snake(name)


def read_sources(root: Path) -> dict[str, str]:
    return {path.stem: path.read_text(encoding="utf-8", errors="replace")
            for path in sorted(root.glob("*.cs"))}


def base_class_of(source: str) -> str | None:
    match = re.search(r"\bclass\s+\w+\s*:\s*(\w+)", source)
    return match.group(1) if match else None


def is_abstract(source: str) -> bool:
    return bool(re.search(r"\babstract\s+class\b", source))


def resolve_property(sources: dict[str, str], class_name: str, prop: str,
                     pattern: str, depth: int = 0) -> str | None:
    """沿继承链找一个属性的取值，返回 ``pattern`` 的第一个捕获组。"""
    if depth > 6 or class_name not in sources:
        return None
    match = re.search(pattern, sources[class_name])
    if match:
        return match.group(1)
    base = base_class_of(sources[class_name])
    return resolve_property(sources, base, prop, pattern, depth + 1) if base else None


#: ``StackType => PowerStackType.Counter`` 的取值。
#: ⚠️ 少数能力的 StackType 是**条件式**的（``MonologuePower`` / ``ShrinkPower``
#: 在属性体里 if/return），静态取不到唯一值 —— 那时返回 None 并报出来，
#: 不能默认成 Counter：叠错语义会让层数计算整体偏掉。
STACK_PATTERN = r"StackType\s*=>\s*PowerStackType\.(\w+)"
STACK_BODY = r"StackType\s*\{\s*get\s*\{"

#: ⚠️ 正负性的真机制是 ``Type => PowerType.Debuff``，**不是** ``IsPositive``。
#: 我第一版抽的是 ``IsPositive``（只有 9 个能力显式声明它），结果 44 个能力和
#: 社区库对不上 —— 那是"抽不到就落默认值"的静默错误，比缺字段危险得多。
TYPE_PATTERN = r"\bType\s*=>\s*PowerType\.(\w+)"


def type_of(sources: dict[str, str], class_name: str) -> str | None:
    """``PowerType.Debuff`` / ``PowerType.Buff``，沿继承链取。"""
    return resolve_property(sources, class_name, "Type", TYPE_PATTERN)


def stack_type_of(sources: dict[str, str], class_name: str) -> str | None:
    if class_name not in sources:
        return None
    source = sources[class_name]
    match = re.search(STACK_PATTERN, source)
    if match:
        return match.group(1).lower()
    if re.search(STACK_BODY, source):
        return None                    # 条件式叠加：静态定不了
    base = base_class_of(source)
    return stack_type_of(sources, base) if base else None


#: ``InstanceType => PowerInstanceType.X``。
#:
#: 三种语义（``PowerInstanceType`` 枚举的文档注释 + ``PowerModel.cs:144`` 的默认实现）：
#:
#: * ``None`` —— **默认**：同类能力合成一个，再施加就叠层；
#: * ``Instanced`` —— 每次施加**新建一个实例**（``TheBombPower``：放第二个炸弹
#:   就该从 3 重新倒数，而不是把第一个炸弹变成 6）；
#: * ``InstancedPerApplier`` —— **每个施加者**一个实例，同一施加者再施加则叠到那一个
#:   （``OblivionPower``：两个玩家互不干扰，但同一个人连放两张要叠加）。
#:
#: ⚠️ 抽不到就落 ``"none"``（真机默认值），**不是**"没抽到"：95% 的能力都是默认值，
#: 把它们全报成"定不下来"只会淹没真正的少数派。
INSTANCE_PATTERN = r"InstanceType\s*=>\s*PowerInstanceType\.(\w+)"


def instance_type_of(sources: dict[str, str], class_name: str) -> str:
    """沿继承链取 ``InstanceType``；没声明就是真机的默认 ``None``。"""
    value = resolve_property(sources, class_name, "InstanceType", INSTANCE_PATTERN)
    return value.lower() if value else "none"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从反编译源码抽取能力表")
    parser.add_argument("--src", default=str(SRC_ROOT))
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--codex", default=str(CODEX_DEFAULT))
    args = parser.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f"没有源码目录：{src}")
        return 3
    sources = read_sources(src)
    report: collections.Counter = collections.Counter()
    powers: list[dict] = []

    for class_name, source in sources.items():
        if is_abstract(source):
            report["abstract_base_class"] += 1
            continue
        stack = stack_type_of(sources, class_name)
        if stack is None:
            report["stack_type_unresolved"] += 1
        power_type = type_of(sources, class_name)
        if power_type is None:
            # ⚠️ 这不是 bug：``TemporaryStrengthPower`` 这类基类的 ``Type`` 是
            # **条件式**的（按当前层数返回 Buff 或 Debuff），静态取不到唯一值。
            # 照实留 None，别猜 —— 猜错会让"力量被削成负数"显示成正 buff。
            report["type_unresolved"] += 1
        allow_negative = re.search(r"AllowNegative\s*=>\s*true", source) is not None
        if allow_negative:
            report["allow_negative"] += 1
        powers.append({
            "id": power_id_of(class_name),
            "class": class_name,
            "stack_type": stack,
            "type": power_type.lower() if power_type else None,
            "allow_negative": allow_negative,
            "instance_type": instance_type_of(sources, class_name),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(powers, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"扫描 {len(sources)} 个能力类，抽出 {len(powers)} 个能力"
          f"（跳过抽象基类 {report['abstract_base_class']} 个）")
    print(f"  stack_type 分布："
          f"{collections.Counter(p['stack_type'] for p in powers).most_common()}")
    print(f"  stack_type 定不下来的：{report['stack_type_unresolved']}"
          f"（条件式叠加，见 STACK_BODY 注释）")
    print(f"  正负性定不下来的：{report['type_unresolved']}"
          f"（条件式 Type，按层数取正负，静态无唯一值）")
    print(f"  负面能力：{sum(1 for p in powers if p['type'] == 'debuff')}")
    print(f"  允许负值（AllowNegative）：{report['allow_negative']}")
    print(f"  type 分布：{collections.Counter(p['type'] for p in powers).most_common()}")
    print(f"  实例类型分布："
          f"{collections.Counter(p['instance_type'] for p in powers).most_common()}")

    codex_path = Path(args.codex)
    if codex_path.exists():
        codex = {c["id"].lower() for c in json.loads(codex_path.read_text(encoding="utf-8"))}
        ours = {p["id"] for p in powers}
        missing = sorted(ours - codex)
        print(f"\n与社区库对账：社区库 {len(codex)} 个 / 源码 {len(ours)} 个")
        print(f"  源码有、社区库没有的 {len(missing)} 个：{missing}")
        print(f"  （社区库多出来的：{sorted(codex - ours)[:8]}）")
    print(f"\n写出 {out}（{len(powers)} 个）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
