"""从反编译源码抽取**卡池成员**（``docs/09`` L5）。

为什么必须抽它：真机的"获得一张随机攻击牌"这类效果从**卡池**里抽，
而社区库的 ``cards.json`` **没有 color 字段** —— 光看社区数据建不出卡池。

真机把成员写在卡池类里，非常直白：

    public sealed class SilentCardPool : CardPoolModel {
        protected override CardModel[] GenerateAllCards() {
            return new CardModel[91] { ModelDb.Card<Abrasive>(), … };
        }
    }

卡池不止用于战斗内的随机生成，**卡牌奖励、商店、事件**都用同一套，
所以这份数据是 Run 层的基础设施。

用法
----
    python tools/extract_pools.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.extract_cards import snake  # noqa: E402

SRC_ROOT = Path("data/decompiled/sts2/MegaCrit.Sts2.Core.Models.CardPools")
OUT_DEFAULT = Path("data/content/repo/card_pools.json")

#: ``ModelDb.Card<Abrasive>()`` → 卡牌类名
CARD_REF = re.compile(r"ModelDb\.Card<(\w+)>\s*\(\s*\)")
#: ``public override bool IsColorless => false;``
IS_COLORLESS = re.compile(r"IsColorless\s*=>\s*(true|false)")
TITLE = re.compile(r'Title\s*=>\s*"([^"]+)"')

#: 这些池不是"可抽取"的卡池：占位/测试/特殊用途。
SKIP_POOLS = frozenset({"MockCardPool", "DeprecatedCardPool", "DeprivedCardPool"})


def parse_pool(path: Path, report: collections.Counter) -> dict | None:
    source = path.read_text(encoding="utf-8", errors="replace")
    if path.stem in SKIP_POOLS:
        report["skipped_pool"] += 1
        return None
    cards = CARD_REF.findall(source)
    if not cards:
        report["pool_without_cards"] += 1
        return None
    colorless = IS_COLORLESS.search(source)
    title = TITLE.search(source)
    return {
        "pool": path.stem,
        # ⚠️ 类名是驼峰、卡 id 是下划线，必须用同一个 `snake()` 转换，
        # 否则池里的卡一张都对不上（而且不报错）。
        "cards": sorted({snake(name) for name in cards}),
        "is_colorless": bool(colorless and colorless.group(1) == "true"),
        "title": title.group(1) if title else "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从反编译源码抽取卡池成员")
    parser.add_argument("--src", default=str(SRC_ROOT))
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    args = parser.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f"没有源码目录：{src}")
        return 3
    report: collections.Counter = collections.Counter()
    pools = [p for p in (parse_pool(f, report) for f in sorted(src.glob("*.cs"))) if p]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pools, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"扫描 {len(list(src.glob('*.cs')))} 个卡池类，抽取 {len(pools)} 个"
          f"（跳过占位池 {report['skipped_pool']} 个）")
    for pool in pools:
        mark = "无色" if pool["is_colorless"] else "角色"
        print(f"    {pool['pool']:24s} {mark}  {len(pool['cards']):3d} 张")
    total = sum(len(p["cards"]) for p in pools)
    print(f"合计 {total} 条归属（同一张卡可属于多个池）")
    print(f"\n写出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
