"""把社区原始数据转换成我们的内容表（``docs/06`` 的 normalize / validate 阶段）。

三步走，**顺序不能反**：

    1. inspect   看清原始数据的真实 schema（不要猜字段名）
    2. convert   启发式映射 → data/content/<版本>/*.json + 一份报告
    3. load      模拟器通过 STS2_CONTENT_DIR 加载（见 sts2_sim.content.load_content_dir）

``convert`` 是**故意做得保守**的：映射不上的字段一律留空并记进报告，绝不猜。
猜错的数值比缺失的数值危险得多——前者会让模型学到错的东西且毫无察觉
（``docs/02`` §2.5）。

用法
----
    python tools/import_content.py inspect                 # 打印 schema 摘要
    python tools/import_content.py inspect --dataset cards # 只看某一类
    python tools/import_content.py convert --version 0.105.0
    python tools/import_content.py report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

RAW_ROOT = Path("data/raw/spire-codex")
CONTENT_ROOT = Path("data/content")

#: 我们把哪些端点当作哪类内容
DATASET_HINTS: dict[str, tuple[str, ...]] = {
    "cards": ("card",),
    "relics": ("relic",),
    "monsters": ("monster", "enemy"),
    "encounters": ("encounter",),
    "events": ("event",),
    "potions": ("potion",),
    "powers": ("power",),
    "keywords": ("keyword",),
    "characters": ("character",),
    "ancients": ("ancient",),
}
#: 只在导出 ZIP 里出现的结构性数据集（卡牌效果/词缀/充能球都要用）。
#: 它们没有对应的散装端点文件时，只能从 ZIP 取——注意那份是**中文**。
ZIP_ONLY_DATASETS: tuple[str, ...] = (
    "enchantments", "afflictions", "orbs", "intents",
    "modifiers", "achievements", "epochs",
)
for _name in ZIP_ONLY_DATASETS:
    DATASET_HINTS.setdefault(_name, (_name.rstrip("s"), _name))

#: 采集时要跳过的成员：这些是**项目自己的元数据**，不是游戏内容。
#: 不排除的话 `manifest.json` 会被当成一个只有 1 条记录的数据集混进报告。
IGNORED_DATASETS: frozenset[str] = frozenset({"manifest"})


# ==========================================================================
# 读取原始数据（JSON 或 ZIP 里的 JSON）
# ==========================================================================
def iter_raw_records(path: Path) -> Iterable[tuple[str, Any]]:
    """产出 ``(成员标识, 解析后的 JSON)``。支持 .json 与 .zip。

    ⚠️ 普通文件返回的是**完整路径**而不是文件名：语言信息藏在目录里
    （``<root>/zhs/cards.json``），只返回文件名会让多语言筛选失效。
    """
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".json"):
                    continue
                try:
                    yield name, json.loads(archive.read(name).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
    elif path.suffix == ".json":
        try:
            yield str(path), json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return


def normalize_records(payload: Any) -> list[dict]:
    """把各种可能的包裹形式统一成记录列表。"""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "items", "results", "records", "entries"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        # 形如 {"CARD.STRIKE": {...}} 的字典形式
        values = [v for v in payload.values() if isinstance(v, dict)]
        if values and len(values) == len(payload):
            return values
        return [payload]
    return []


def dataset_name(member: str) -> str:
    stem = Path(member).stem.lower()
    for dataset, hints in DATASET_HINTS.items():
        if any(hint in stem for hint in hints):
            return dataset
    return stem


#: 导出包里可能出现的语言目录名
KNOWN_LANGS: tuple[str, ...] = (
    "zhs", "zht", "eng", "jpn", "kor", "fra", "deu", "esp", "ptb", "rus", "tur", "pol",
)


def language_of(member: str) -> str:
    """从成员路径里识别语言目录，例如 ``zhs/cards.json`` → ``zhs``。

    ⚠️ 实测 ``export_zhs.zip`` 里**根本没有语言目录**，成员名就是
    ``cards.json``（内容却是中文）。光看成员路径会全部落到 ``unknown``，
    于是 ``export_zhs.zip`` 与 ``export_eng.zip`` 会被当成同一份语言而**合并**，
    重复导入又回来了。所以语言还要能从**归档文件名**推出来。
    """
    for part in Path(member).parts[:-1]:
        token = part.lower()
        if token in KNOWN_LANGS:
            return token
        for lang in KNOWN_LANGS:
            if token.endswith(f"_{lang}") or token.endswith(f"-{lang}"):
                return lang
    return ""


def language_of_archive(path: Path) -> str:
    """从归档文件名推语言：``export_zhs.zip`` → ``zhs``。"""
    stem = path.stem.lower()
    for lang in KNOWN_LANGS:
        if stem.endswith(f"_{lang}") or stem.endswith(f"-{lang}"):
            return lang
    return ""


def collect_buckets(files: list[Path], preferred_lang: str = "zhs",
                    ) -> tuple[dict[str, list[dict]], str, dict[str, int],
                               dict[str, dict[str, Any]]]:
    """选出要用的一份数据，并记下每个数据集最终取自哪个文件。

    真实下载会同时拿到**三种来源**，必须去重，否则记录数会成倍膨胀
    （实测踩过：encounters 87 条变成了 261 条 = 87×3）：

      1. 单端点 JSON（`data/raw/<v>/cards.json`）——**英文，首选**
      2. 全量导出 ZIP（`export_zhs.zip` 里的 `zhs/cards.json`）——**中文**
      3. 另一种语言的导出 ZIP（`export_eng.zip`）

    规则：**按「数据集」粒度回退**——某个数据集有单端点文件就用它，
    没有才去 ZIP 里取对应语言那一份。

    ⚠️ 这里踩过三次坑，方向各不相同，别只修一边：

    * **重复导入**：不加区分地把所有来源合并 → 同一份数据读 3 遍
      （encounters 87 条变 261 条）。
    * **静默丢失**：改成 `sources = plain if plain else archives` 的
      **全有或全无** → 只要散装文件存在，ZIP 就整个不读，于是 ZIP 独有的
      7 个数据集（enchantments / intents / orbs / afflictions / modifiers /
      achievements / epochs）被丢掉，报告里一个字都不提。
      实测 raw 115 个怪物、归一化后 114 —— 少的那个见 `convert_monster`。
    * **语言串味**：把散装文件一律当英文而不看它们的语言目录，会让
      `zhs/cards.json` 与 `eng/cards.json` 被**合到一起**。两个来源的语言
      都必须分组后再挑一份。

    语言也必须钉死：散装端点是**英文**、导出包是**中文**，而卡牌效果解析器
    吃的是英文描述（`Deal 6 damage.`）。让中文覆盖英文会直接废掉解析器，
    所以**散装的优先级高于导出包**，不是随手挑一份。

    返回 ``(buckets, chosen_lang, lang_counts, provenance)``，
    ``provenance[数据集] = {"source", "lang", "records"}``。
    """
    plain = [p for p in files if p.suffix != ".zip"]
    archives = [p for p in files if p.suffix == ".zip"]

    def group(paths: list[Path], lang_of) -> tuple[dict, dict]:
        """按 ``{语言: {数据集: [记录]}}`` 分组，并记下每个数据集的来源文件。"""
        by_lang: dict[str, dict[str, list[dict]]] = {}
        where: dict[str, dict[str, str]] = {}
        for path in paths:
            for member, payload in iter_raw_records(path):
                records = normalize_records(payload)
                if not records:
                    continue
                lang = lang_of(path, member)
                name = dataset_name(member)
                if name in IGNORED_DATASETS:
                    continue
                by_lang.setdefault(lang, {}).setdefault(name, []).extend(records)
                source = str(path) if path.suffix != ".zip" else f"{path}::{member}"
                where.setdefault(lang, {}).setdefault(name, source)
        return by_lang, where

    # 散装端点：优先看路径里的语言目录，没有目录的按**默认英文**处理
    # （spire-codex 的裸端点返回的就是英文）。
    plain_by_lang, plain_where = group(
        plain, lambda path, member: language_of(str(path)) or "eng")
    # 导出包：语言优先看成员路径，其次看归档文件名（zip 里通常没有语言目录，
    # 光看成员路径会全部落到 unknown，两个语言包就会被合并）。
    arch_by_lang, arch_where = group(
        archives, lambda path, member: language_of(member)
        or language_of_archive(path) or "unknown")

    def counts_of(by_lang: dict) -> dict[str, int]:
        return {lang: sum(len(v) for v in data.values()) for lang, data in by_lang.items()}

    plain_counts, arch_counts = counts_of(plain_by_lang), counts_of(arch_by_lang)

    def pick(by_lang: dict, counts: dict[str, int]) -> str:
        if preferred_lang in by_lang:
            return preferred_lang
        if len(by_lang) == 1:
            return next(iter(by_lang))          # 只有一份，不必比大小
        return max(counts, key=lambda k: counts[k]) if counts else ""

    plain_lang = pick(plain_by_lang, plain_counts)
    arch_lang = pick(arch_by_lang, arch_counts)
    chosen = plain_lang or arch_lang

    # 把两个来源的语言都算进 lang_counts：调用方靠它判断"有没有多语言"。
    lang_counts = {lang: plain_counts.get(lang, 0) + arch_counts.get(lang, 0)
                   for lang in set(plain_counts) | set(arch_counts)}

    # --- 归档打底，散装覆盖：每个数据集**只留一份** ---
    buckets = {name: list(recs)
               for name, recs in arch_by_lang.get(arch_lang, {}).items()}
    provenance: dict[str, dict[str, Any]] = {}
    for name, recs in buckets.items():
        provenance[name] = {"source": arch_where.get(arch_lang, {}).get(name, ""),
                            "lang": arch_lang, "records": len(recs)}
    for name, recs in plain_by_lang.get(plain_lang, {}).items():
        buckets[name] = recs                      # 英文散装覆盖中文归档
        provenance[name] = {"source": plain_where[plain_lang][name],
                            "lang": plain_lang, "records": len(recs)}

    # --- 谁被丢了？必须说出来，不能静默 ---
    # 只在**被选中的语言**内部比对：没被选中的语言本来就不该进来。
    seen = set(arch_by_lang.get(arch_lang, {})) | set(plain_by_lang.get(plain_lang, {}))
    lost = sorted(seen - set(buckets))
    if lost:
        raise AssertionError(f"这些数据集被静默丢掉了：{lost}")
    return buckets, chosen, lang_counts, provenance


# ==========================================================================
# inspect：schema 摘要
# ==========================================================================
def flatten_schema(node: Any, prefix: str = "", depth: int = 0,
                   max_depth: int = 3, out: dict | None = None) -> dict:
    out = {} if out is None else out
    if depth > max_depth:
        return out
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flatten_schema(value, path, depth + 1, max_depth, out)
    elif isinstance(node, list):
        path = f"{prefix}[]"
        if node:
            flatten_schema(node[0], path, depth + 1, max_depth, out)
        else:
            out.setdefault(path, []).append("<空数组>")
    else:
        text = repr(node)
        out.setdefault(prefix, []).append(text[:60] + ("…" if len(text) > 60 else ""))
    return out


def cmd_inspect(args) -> int:
    root = RAW_ROOT / args.version if args.version else RAW_ROOT
    files = sorted(p for p in root.rglob("*") if p.suffix in (".json", ".zip"))
    if not files:
        print(f"没有找到原始数据：{root}")
        print("先在**有网络的终端**里运行：python tools/fetch_content.py endpoints")
        return 3

    wanted = set(args.dataset.split(",")) if args.dataset else None
    _, chosen_lang, lang_counts, provenance = collect_buckets(files, args.lang)
    if lang_counts:
        print(f"检测到语言 {lang_counts} → 采用 {chosen_lang!r}")
    if provenance:
        print("各数据集来源（散装端点=英文优先，导出包=中文补缺）：")
        for name in sorted(provenance):
            info = provenance[name]
            print(f"    {name:14s} {info['lang']:4s} {info['records']:5d} 条  {info['source']}")

    for path in files:
        print(f"\n{'=' * 72}\n文件 {path}  ({path.stat().st_size:,} 字节)")
        for member, payload in iter_raw_records(path):
            name = dataset_name(member)
            if wanted and name not in wanted:
                continue
            lang = language_of(member)
            if lang_counts and lang and lang != chosen_lang:
                continue          # 只看将要使用的那一份，避免同一份 schema 打印好几遍
            records = normalize_records(payload)
            if not records:
                continue
            print(f"\n--- {member}  →  数据集 {name!r}：{len(records)} 条 ---")
            merged: dict[str, list[str]] = {}
            for record in records[:40]:
                for key, samples in flatten_schema(record).items():
                    merged.setdefault(key, [])
                    for sample in samples[:2]:
                        if sample not in merged[key]:
                            merged[key].append(sample)
            for key in sorted(merged):
                samples = ", ".join(merged[key][:2])
                print(f"    {key:44s} {samples}")
    print("\n把这份输出贴回来，我据此写精确的字段映射。")
    return 0


# ==========================================================================
# convert：启发式映射
# ==========================================================================
#: 目标字段 → 候选原始字段名（按优先级）
FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "cid": ("id", "key", "card_id", "model_id", "slug", "name_id"),
    "name": ("name", "title", "display_name", "localized_name"),
    "cost": ("cost", "energy", "energy_cost", "base_cost"),
    "card_type": ("type", "card_type", "cardType", "kind"),
    "rarity": ("rarity", "tier"),
    "target": ("target", "target_type", "targets"),
    "text": ("description", "text", "desc", "body", "raw_description"),
    "upgraded_text": ("upgraded_description", "upgrade_description", "description_upgraded"),
    "hp": ("hp", "health", "hit_points", "max_hp", "hp_range"),
    "moves": ("moves", "attacks", "abilities", "move_list"),
    "exhaust": ("exhaust", "exhausts", "is_exhaust"),
    "keywords": ("keywords", "tags", "properties"),
}

#: 效果文本 → 效果算子。**先覆盖高频句式，覆盖不了的一律进报告**（docs/06 §6.3）。
EFFECT_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"对所有敌人造成\s*(\d+)\s*点伤害"), "damage_all", "enemy"),
    (re.compile(r"deal\s+(\d+)\s+damage\s+to\s+ALL\s+enemies", re.I), "damage_all", "enemy"),
    (re.compile(r"造成\s*(\d+)\s*点伤害"), "damage", "enemy"),
    (re.compile(r"deal\s+(\d+)\s+damage", re.I), "damage", "enemy"),
    (re.compile(r"获得\s*(\d+)\s*点格挡"), "block", "self"),
    (re.compile(r"gain\s+(\d+)\s+block", re.I), "block", "self"),
    (re.compile(r"抽\s*(\d+)\s*张牌"), "draw", "self"),
    (re.compile(r"draw\s+(\d+)\s+cards?", re.I), "draw", "self"),
    (re.compile(r"获得\s*(\d+)\s*点能量"), "gain_energy", "self"),
    (re.compile(r"gain\s+(\d+)\s+energy", re.I), "gain_energy", "self"),
    (re.compile(r"失去\s*(\d+)\s*点生命"), "lose_hp", "self"),
    (re.compile(r"lose\s+(\d+)\s+HP", re.I), "lose_hp", "self"),
    # 真机常见的连写形式："Deal 6 damage to ALL enemies." / "Gain 5 Block."
    (re.compile(r"(\d+)\s+damage\s+to\s+all\s+enemies", re.I), "damage_all", "enemy"),
)

#: 能力文本 → (能力名, 目标)。**目标必须逐条给对**：力量是自增益，
#: 若被误判成"对敌施加"，模型会看到一个语义完全错误的卡牌定义。
POWER_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"施加\s*(\d+)\s*层易伤"), "vulnerable", "enemy"),
    (re.compile(r"apply\s+(\d+)\s+vulnerable", re.I), "vulnerable", "enemy"),
    (re.compile(r"施加\s*(\d+)\s*层虚弱"), "weak", "enemy"),
    (re.compile(r"apply\s+(\d+)\s+weak", re.I), "weak", "enemy"),
    (re.compile(r"获得\s*(\d+)\s*点力量"), "strength", "self"),
    (re.compile(r"gain\s+(\d+)\s+strength", re.I), "strength", "self"),
)

RARITY_MAP = {
    "basic": "basic", "starter": "basic", "common": "common",
    "uncommon": "uncommon", "rare": "rare", "curse": "curse",
    "special": "special", "status": "status",
    "普通": "common", "罕见": "uncommon", "稀有": "rare",
    "基础": "basic", "诅咒": "curse",
}

TYPE_MAP = {
    "attack": "attack", "skill": "skill", "power": "power",
    "status": "status", "curse": "curse",
    "攻击": "attack", "技能": "skill", "能力": "power",
}


def pick(record: dict, field: str) -> Any:
    for key in FIELD_CANDIDATES.get(field, ()):
        if key in record and record[key] not in (None, "", [], {}):
            return record[key]
    # 大小写不敏感回退
    lowered = {k.lower(): v for k, v in record.items()}
    for key in FIELD_CANDIDATES.get(field, ()):
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


#: 卡牌文本里的展示用标记：`[gold]Block[/gold]`、`[sine]...[/sine]`、`[energy:1]`…
#: ⚠️ **不剥掉这些标记，正则就一条都匹配不上**——实测覆盖率会从 57% 掉到 42%。
MARKUP_RE = re.compile(r"\[[^\[\]]{0,24}\]")

#: 真机文本里的占位符（角色名、数值等）
PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_]+\}")


def strip_markup(text: str) -> str:
    """去掉展示用标记，只留下可以解析的正文。"""
    cleaned = MARKUP_RE.sub("", text)
    cleaned = PLACEHOLDER_RE.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_effects(text: str) -> tuple[list[dict], list[str]]:
    """从描述文本里抽出可执行效果。返回 ``(效果列表, 无法识别的片段)``。"""
    if not text:
        return [], []
    text = strip_markup(text)
    effects: list[dict] = []
    consumed: list[tuple[int, int]] = []

    for pattern, op, target in EFFECT_PATTERNS:
        for match in pattern.finditer(text):
            effects.append({"op": op, "amount": int(match.group(1)), "target": target})
            consumed.append(match.span())
    for pattern, power, target in POWER_PATTERNS:
        for match in pattern.finditer(text):
            effects.append({"op": "apply_power", "amount": int(match.group(1)),
                            "power": power, "target": target})
            consumed.append(match.span())

    leftovers = _uncovered(text, consumed)
    return effects, leftovers


def _uncovered(text: str, spans: list[tuple[int, int]]) -> list[str]:
    if not spans:
        return [text.strip()] if text.strip() else []
    spans = sorted(spans)
    merged: list[list[int]] = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    pieces, cursor = [], 0
    for start, end in merged:
        if start > cursor:
            pieces.append(text[cursor:start])
        cursor = end
    if cursor < len(text):
        pieces.append(text[cursor:])
    return [p.strip(" 。.，,;；") for p in pieces if p.strip(" 。.，,;；")]


def infer_target(effects: list[dict]) -> str:
    """从效果推断目标类型。

    ⚠️ 不能简单地"有伤害就是 enemy"：纯格挡/抽牌的技能牌若被标成 ``enemy``，
    引擎会为它生成"选哪个敌人"的动作，而这张牌根本不需要目标——
    表现为动作空间里多出一堆无意义分支，策略会被这些噪声带偏。
    """
    if any(e["op"] == "damage_all" for e in effects):
        return "all_enemies"
    offends = any(e["op"] in ("damage",) or
                  (e["op"] == "apply_power" and e.get("target") == "enemy")
                  for e in effects)
    if offends:
        return "enemy"
    if any(e.get("target") == "self" or e["op"] in ("block", "draw", "gain_energy",
                                                    "lose_hp") for e in effects):
        return "none"
    return "none"


def convert_card(record: dict, report: Counter) -> dict | None:
    cid = pick(record, "cid")
    if not cid:
        report["card_without_id"] += 1
        return None
    cid = str(cid).split(".")[-1].lower()
    text = pick(record, "text") or ""
    effects, leftovers = parse_effects(text)
    upgraded_text = pick(record, "upgraded_text") or ""
    upgrade, upgrade_leftovers = parse_effects(upgraded_text)

    for piece in leftovers:
        report[f"card_effect_unparsed::{piece[:24]}"] += 1
    for piece in upgrade_leftovers:
        report[f"card_upgrade_unparsed::{piece[:24]}"] += 1
    if not effects:
        report["card_without_effects"] += 1

    rarity = str(pick(record, "rarity") or "").lower()
    card_type = str(pick(record, "card_type") or "").lower()
    # X 费：``is_x_cost`` 是**唯一**能区分"X 费"与"不可打出"的信号。
    # ⚠️ 两者的构造参数会撞车：普通 X 费卡写 ``base(0, …)``、诅咒写 ``base(-1, …)``，
    # 而 ``Cascade`` 是 X 费却写 ``base(-1, …)`` —— 只看费用数字必然搞错一边。
    x_cost = bool(record.get("is_x_cost")) or bool(record.get("is_x_star_cost"))
    return {
        "cid": cid,
        "name": pick(record, "name") or cid,
        # 真机 `CardEnergyCost.Canonical = (!CostsX) ? canonicalCost : 0` ——
        # X 费卡的规范费用被**强制为 0**（`Cascade` 的原始输入是 -1 也一样）。
        "cost": 0 if x_cost else int(pick(record, "cost") or 0),
        "is_x_cost": x_cost,
        "card_type": TYPE_MAP.get(card_type, card_type or "skill"),
        "rarity": RARITY_MAP.get(rarity, rarity or "common"),
        "target": infer_target(effects),
        "effects": effects,
        "upgrade": upgrade,
        "exhaust": bool(pick(record, "exhaust") or "消耗" in text or "Exhaust" in text),
        "text": text,
        "verified": False,          # ⭐ 导入的内容一律未对拍（docs/06 §6.5）
    }


# --------------------------------------------------------------------------
# 怪物（真实 schema：spire-codex /api/monsters）
# --------------------------------------------------------------------------
#: 真机的 intent 文案 → 我们的意图类型
INTENT_MAP: dict[str, str] = {
    "attack": "attack",
    "attack + defend": "attack",
    "attack + buff": "attack",
    "attack + summon": "attack",
    "attack + debuff": "attack_debuff",
    "defend": "block",
    "defend + buff": "block",
    "buff": "buff",
    "debuff": "debuff",
    "status": "debuff",
    "stun": "unknown",
    "sleep": "unknown",
    "summon": "buff",
    "unknown": "unknown",
    "attack + status": "attack_debuff",
}

#: 我们的引擎支持的能力。其余能力**必须报出来**，不能静默丢弃——
#: 丢掉一个能力等于改变了怪物的行为，而且没有任何信号。
SUPPORTED_POWERS = {"strength", "vulnerable", "weak"}

#: 真机能力 target → 我们（施法者视角）的目标
POWER_TARGET_MAP = {"player": "enemy", "self": "self", "all_players": "enemy"}


def convert_monster(record: dict, report: Counter) -> dict | None:
    mid = record.get("id")
    if not mid:
        report["monster_without_id"] += 1
        return None
    eid = str(mid).lower()

    lo = record.get("min_hp")
    hi = record.get("max_hp") or lo
    if lo is None:
        # ⚠️ 这里**不能** `return None`。旧代码这么干过，把 `TestSubject`
        # （真机上是一只三阶段复活 Boss）整条丢掉了，报告里只有一行
        # `monster_without_hp`，怪物表从 115 悄悄变成 114。
        # 缺 HP 就照实写 null，由 `sts2_sim.content` 用**源码**补齐，并在
        # 加载收尾时硬校验（`_require_enemy_hp`）。
        report["monster_without_hp_from_codex"] += 1

    moves = []
    for raw in record.get("moves") or []:
        move_id = str(raw.get("id", "")).lower()
        if not move_id:
            continue
        intent_text = str(raw.get("intent") or "Unknown").strip().lower()
        damage = raw.get("damage") or {}
        hits = damage.get("hit_count") or 1
        value = damage.get("normal") or 0
        effects: list[dict] = []
        if value:
            for _ in range(int(hits)):
                effects.append({"op": "damage", "amount": int(value)})
        if raw.get("block"):
            effects.append({"op": "block", "amount": int(raw["block"]), "target": "self"})
        for power in raw.get("powers") or []:
            name = str(power.get("power_id", "")).lower()
            if name not in SUPPORTED_POWERS:
                report[f"monster_power_unsupported::{name}"] += 1
                continue
            effects.append({
                "op": "apply_power", "amount": int(power.get("amount") or 0),
                "power": name,
                "target": POWER_TARGET_MAP.get(str(power.get("target", "")).lower(),
                                               "enemy"),
            })
        moves.append({
            "mid": move_id,
            "name": raw.get("name") or move_id,
            "intent": INTENT_MAP.get(intent_text, "unknown"),
            "intent_text": raw.get("intent") or "Unknown",
            "value": int(value),
            "times": int(hits),
            "effects": effects,
            "weight": 1.0,
        })
    if not moves:
        report["monster_without_moves"] += 1

    pattern = record.get("attack_pattern") or {}
    pattern_type = str(pattern.get("type") or "")
    cycle: list[str] = []
    if pattern_type == "cycle":
        # cycle 型可以**忠实**还原：states 的 next 指针已构成闭环
        states = {s.get("id"): s for s in pattern.get("states") or []}
        cursor = pattern.get("initial_move")
        seen: set[str] = set()
        while cursor:
            state = states.get(cursor) or states.get(f"{cursor}_MOVE")
            if state is None or cursor in seen:
                break
            seen.add(cursor)
            move_id = state.get("move_id")
            if move_id:
                cycle.append(str(move_id).lower())
            cursor = state.get("next")
    else:
        report[f"monster_pattern_not_cycle::{pattern_type or 'none'}"] += 1

    # 固有属性必须**连数值一起**留下。
    # ⚠️ 旧代码只留了一个"未建模能力的 id 列表"，**把数值丢了** ——
    # 于是 46 只怪（40%）开局根本拿不到自己的固有属性，
    # 而 `artifact 3` vs `artifact 1` 是完全不同的战斗。
    innate: list[dict] = []
    for power in record.get("innate_powers") or []:
        name = str(power.get("power_id", "")).lower()
        if not name:
            continue
        amount = power.get("amount")
        innate.append({
            "power": name,
            "amount": int(amount) if isinstance(amount, (int, float)) else 0,
            "amount_ascension": power.get("amount_ascension"),
            "modeled": name in SUPPORTED_POWERS,
        })
    unmodeled = sorted({p["power"] for p in innate if not p["modeled"]})
    if unmodeled:
        report["monster_innate_power_unmodeled"] += 1

    return {
        "eid": eid,
        "name": record.get("name") or eid,
        "hp": [int(lo), int(hi)] if lo is not None else None,
        "type": record.get("type") or "Normal",
        "moves": moves,
        "cycle": cycle,
        "pattern_type": pattern_type,
        "innate_powers": innate,
        "unmodeled_innate_powers": unmodeled,
        "verified": False,
    }


def convert_encounter(record: dict, report: Counter) -> dict | None:
    eid = record.get("id")
    if not eid:
        report["encounter_without_id"] += 1
        return None
    monsters = [str(m.get("id", "")).lower() for m in (record.get("monsters") or [])]
    return {
        "eid": str(eid).lower(),
        "name": record.get("name") or eid,
        "room_type": record.get("room_type") or "Monster",
        "act": record.get("act") or "",
        "is_weak": bool(record.get("is_weak")),
        "monsters": monsters,
        "verified": False,
    }


# --------------------------------------------------------------------------
# 转换主流程
# --------------------------------------------------------------------------
def _provisional_encounters(raw_monsters: list[dict], report: Counter) -> list[dict]:
    """从怪物记录反推遭遇表（缺 /api/encounters 时的兜底）。"""
    table: dict[str, dict] = {}
    for monster in raw_monsters:
        mid = str(monster.get("id", "")).lower()
        for encounter in monster.get("encounters") or []:
            eid = encounter.get("encounter_id")
            if not eid:
                continue
            entry = table.setdefault(str(eid).lower(), {
                "eid": str(eid).lower(),
                "name": encounter.get("encounter_name") or eid,
                "room_type": encounter.get("room_type") or "Monster",
                "act": encounter.get("act") or "",
                "is_weak": bool(encounter.get("is_weak")),
                "monsters": [],
                "provisional": True,      # ⚠️ 成员可能不全
                "verified": False,
            })
            if mid and mid not in entry["monsters"]:
                entry["monsters"].append(mid)
    return sorted(table.values(), key=lambda e: e["eid"])


def cmd_convert(args) -> int:
    root = RAW_ROOT / args.version if args.version else RAW_ROOT
    files = sorted(p for p in root.rglob("*") if p.suffix in (".json", ".zip"))
    if not files:
        print(f"没有找到原始数据：{root}")
        return 3

    buckets, chosen_lang, lang_counts, provenance = collect_buckets(files, args.lang)
    if not buckets:
        print(f"{root} 下没有解析出任何记录。")
        return 3
    if lang_counts:
        print(f"检测到多语言数据 {lang_counts} → 采用 {chosen_lang!r}")
    print("各数据集来源：")
    for name in sorted(provenance):
        info = provenance[name]
        print(f"    {name:14s} {info['lang']:4s} {info['records']:5d} 条  {info['source']}")

    report: Counter = Counter()
    out_dir = CONTENT_ROOT / (args.version or "unversioned")
    out_dir.mkdir(parents=True, exist_ok=True)

    cards = [c for c in (convert_card(r, report) for r in buckets.get("cards", [])) if c]
    monsters = [m for m in (convert_monster(r, report)
                            for r in buckets.get("monsters", [])) if m]
    encounter_records = buckets.get("encounters", [])
    encounters = [e for e in (convert_encounter(r, report) for r in encounter_records) if e]

    # 转换阶段**任何**一条被丢掉都要说出来：raw 多少条、出来多少条。
    for name, produced in (("cards", cards), ("monsters", monsters),
                           ("encounters", encounters)):
        raw_count = len(buckets.get(name, []))
        if raw_count != len(produced):
            report[f"dropped_during_convert::{name}"] += raw_count - len(produced)

    payloads: dict[str, list] = {}
    if cards:
        payloads["cards"] = cards
        (out_dir / "cards.json").write_text(
            json.dumps(cards, ensure_ascii=False, indent=1), encoding="utf-8")
    if monsters:
        payloads["monsters"] = monsters
        (out_dir / "monsters.json").write_text(
            json.dumps(monsters, ensure_ascii=False, indent=1), encoding="utf-8")
    if encounters:
        payloads["encounters"] = encounters
        (out_dir / "encounters.json").write_text(
            json.dumps(encounters, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        # 没有 /api/encounters 时，从怪物记录里的 encounters 字段**反推**遭遇表。
        # ⚠️ 反推的表可能缺成员（只有被抓到的怪物才会出现在名单里），所以标 provisional。
        provisional = _provisional_encounters(buckets.get("monsters", []), report)
        if provisional:
            payloads["encounters_provisional"] = provisional
            (out_dir / "encounters_provisional.json").write_text(
                json.dumps(provisional, ensure_ascii=False, indent=1), encoding="utf-8")
            report["encounters_provisional_from_monsters"] += 1

    for dataset in ("relics", "potions", "events", "keywords", "powers", "characters",
                    *ZIP_ONLY_DATASETS):
        records = buckets.get(dataset)
        if not records:
            continue
        payloads[dataset] = records
        (out_dir / f"{dataset}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    # 语言必须写进报告：这些表里混着**中文**（导出包）与英文（散装端点）。
    # 拿英文句式模板去解析中文描述会一条都解不出来，而且不报错。
    non_english = sorted(name for name in payloads
                         if provenance.get(name, {}).get("lang") not in ("eng", ""))
    if non_english:
        print(f"⚠️ 这些数据集取自导出包，是**中文**，不能用英文句式解析：{non_english}")

    parsed = sum(1 for c in cards if c["effects"])
    summary = {
        "version": args.version or "unversioned",
        "cards_total": len(cards),
        "cards_with_effects": parsed,
        "cards_without_effects": len(cards) - parsed,
        "coverage": round(parsed / max(1, len(cards)), 3) if cards else 0.0,
        "monsters_total": len(monsters),
        "monsters_with_cycle": sum(1 for m in monsters if m["cycle"]),
        "datasets": {k: len(v) for k, v in payloads.items()},
        "unmapped": {k: v for k, v in report.most_common(40)},
    }
    (out_dir / "import_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"输出目录 {out_dir}")
    print(f"卡牌 {len(cards)} 张，其中 {parsed} 张解析出可执行效果"
          f"（覆盖率 {summary['coverage']:.0%}）")
    print("各数据集记录数：" + ", ".join(f"{k}={v}" for k, v in summary["datasets"].items()))
    if report:
        print("\n未能映射的片段（前 15 条，需要补句式模板）：")
        for key, count in report.most_common(15):
            print(f"    {count:4d}  {key}")
    print(f"\n报告 → {out_dir / 'import_report.json'}")
    print("\n下一步：")
    print(f"    python tools/healthcheck.py --content {out_dir}")
    print("若报告里有未映射的片段，把那部分贴回来，我据此补全。")
    return 0


def cmd_report(args) -> int:
    path = CONTENT_ROOT / (args.version or "unversioned") / "import_report.json"
    if not path.exists():
        print(f"没有报告：{path}（先运行 convert）")
        return 3
    summary = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="社区数据 → 内容表")
    parser.add_argument("--version", default="")
    parser.add_argument("--lang", default="zhs",
                        help="导出包里的语言目录，默认 zhs（找不到时自动选记录最多的）")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="打印原始数据 schema")
    inspect.add_argument("--dataset", default="", help="只看的端点，逗号分隔")
    sub.add_parser("convert", help="转换成内容表 JSON")
    sub.add_parser("report", help="查看上一次转换的报告")
    args = parser.parse_args(argv)
    return {"inspect": cmd_inspect, "convert": cmd_convert,
            "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
