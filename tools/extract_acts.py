"""从反编译源码抽取**幕（Act）**数据（``data/content/repo/acts_source.json``）。

铁律（``docs/09`` §5.2）：结构、数值、公式全部来自 ``data/decompiled/sts2/``；
社区库（``data/raw/spire-codex/``）不参与，一个字都不抄。
本文件里每条数据都带 ``*_source``（``文件:行``），可逐条回源码核对。

真机的幕模型（``MegaCrit.Sts2.Core.Models/ActModel.cs``）::

    ActsByIndex 把幕按 Index 分组：一局只走 3 个幕索引，
    但 index 0 有两个候选 —— Overgrowth（IsDefault=true）与
    Underdocks（IsDefault=false，IsUnlocked 要 UnderdocksEpoch）。

    GenerateRooms(act)：
      前 NumberOfWeakEncounters 个普通战斗 ← AllWeakEncounters
      其余 (GetNumberOfRooms - NumberOfWeakEncounters) 个 ← AllRegularEncounters
      精英 ← AllEliteEncounters
      Boss = rng.NextItem(AllBossEncounters)

    GetMapPointTypes(mapRng) → MapPointTypeCounts(unknown, rest)
      elites / shops / point_types_that_ignore_rules **不由子类给**，
      而是 ``MapPointTypeCounts`` 的初始化式（见该文件 12-16 行）。

⚠️ 三处最容易抄错的地方，都在源码里写得很绕：

1. **遭遇分类是 ``Where`` 谓词，不是命名后缀**。``ActModel.cs:149-164`` 按
   ``RoomType`` + ``IsWeak`` 分四类（``FogmogNormal`` 不带后缀也照样进
   「普通」）。命名只是巧合，不许按名字猜。
2. **``rest`` 的两种写法语义不同**：``NextGaussianInt(mean,stdDev,min,max)``
   的上下界是**闭区间**（``Rng.cs:251-279`` 的循环条件 ``num3 < min || num3 > max``），
   而 ``NextInt(minInclusive,maxExclusive)`` 的上界**取不到**
   （``Rng.cs:89-103``）。所以 Glory 的 ``NextInt(5,7)`` 其实是 5 或 6。
3. **``unknown`` 是叠加的**：Hive / Glory 在 ``StandardRandomUnknownCount``
   的返回值上再 ``-1``（``Hive.cs:120`` / ``Glory.cs:109``），不是另一个分布参数。
4. **``elites`` / ``shops`` / ``point_types_that_ignore_rules`` 不在幕文件里**。
   它们是 ``MapPointTypeCounts`` 上带初始化式的属性
   （``MapPointTypeCounts.cs:12-16``），幕文件只调
   ``new MapPointTypeCounts(unknown, rest)``。所以那三项四张幕**完全相同**，
   这不是"抄漏了"。

用法
----
    python tools/extract_acts.py
    python tools/extract_acts.py --self-check     # 抽完再验一遍不变量
    python tools/extract_acts.py --out data/content/repo/acts_source.json

⚠️ 不要用 PowerShell 的字符串替换来改本文件：``Get-Content``/``Set-Content``
往返会按系统 ANSI 重新编码，把中文注释写成非法字节。改代码用编辑工具。

⚠️ 本文件**只读**源码、**只写**上面那一个 JSON。不要在这里加载 ``sts2_sim``
（那会把"抽取"和"引擎"耦合起来，引擎改一次抽取就得跟着跑）。

⚠️ 输出是**裸 JSON 数组**，与 ``cards_source.json`` / ``events_source.json``
（``extract_cards.py:2174`` / ``extract_events.py`` 都写 ``json.dumps(cards, …)``）
和 ``tools/audit_data.py`` 的 ``as_records`` 口径一致；``indent=1`` +
``ensure_ascii=False`` 也与它们逐字相同。顶层不要加元信息字典 ——
``audit_data.audit_content`` 会按主键查重，把元信息当记录会报"缺主键"。
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

#: 反编译源码树根（**唯一真相**）。
DECOMPILED = Path("data/decompiled/sts2")
ACTS_DIR = DECOMPILED / "MegaCrit.Sts2.Core.Models.Acts"
ENCOUNTERS_DIR = DECOMPILED / "MegaCrit.Sts2.Core.Models.Encounters"
#: 事件与远古（``Neow : AncientEventModel : EventModel``）都在这一层。
ELDER_EVENTS_DIR = DECOMPILED / "MegaCrit.Sts2.Core.Models.Events"
ACT_MODEL = DECOMPILED / "MegaCrit.Sts2.Core.Models" / "ActModel.cs"
POINT_COUNTS = DECOMPILED / "MegaCrit.Sts2.Core.Map" / "MapPointTypeCounts.cs"
OUT_DEFAULT = Path("data/content/repo/acts_source.json")

# ==========================================================================
# 源码里的模型引用
# ==========================================================================
#: ``ModelDb.Encounter<VantomBoss>()`` / ``ModelDb.Event<AromaOfChaos>()`` /
#: ``ModelDb.AncientEvent<Neow>()``。
#:
#: 泛型实参里**不许有空白**：反编译输出是 ``<VantomBoss>``，写成 ``[\w.]+``
#: 会跨过换行把后面别的调用一起吃进来。这里刻意用 ``[A-Za-z0-9_.]+`` 且不
#: 允许空白 —— 宁可漏（漏了会因"数量与数组长度不符"被报出来），不可串。
MODEL_REF = re.compile(r"ModelDb\.(Encounter|Event|AncientEvent)<([A-Za-z0-9_.]+)>")

#: ``new global::_003C_003Ez__ReadOnlyArray<EncounterModel>(new EncounterModel[22]``
#: 里的数组长度。用来与"真的数出来多少个"对账 —— 对不上就是解析漏了。
ARRAY_LENGTH = re.compile(r"new\s+[A-Za-z0-9_.<>]+\s*\[\s*(\d+)\s*\]")

#: ``RoomSet.SwapToOrCreateAtIndex<EncounterModel, NibbitsWeak>(_rooms.normalEncounters, 3)``
SWAP = re.compile(
    r"RoomSet\.SwapToOrCreateAtIndex\s*<\s*([A-Za-z0-9_.]+)\s*,\s*"
    r"([A-Za-z0-9_.]+)\s*>\s*\(\s*[\w.]*?(\w+)\s*,\s*(\d+)\s*\)")

#: ``MapPointTypeCounts.StandardRandomUnknownCount(mapRng)``
STANDARD_UNKNOWN = re.compile(
    r"MapPointTypeCounts\.StandardRandomUnknownCount\s*\(\s*\w+\s*\)")

#: ``mapRng.NextGaussianInt(12, 1, 10, 14)``
GAUSSIAN_INT = re.compile(
    r"NextGaussianInt\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")

#: ``mapRng.NextInt(5, 7)``
NEXT_INT = re.compile(r"NextInt\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")

#: 类型名 → 角色（从 ``class X : Y`` 里取基类）。泛型基类要连参数一起取。
CLASS_DECL = re.compile(r"\bclass\s+([A-Za-z_]\w*)\s*:\s*([A-Za-z0-9_.<>,\s]+?)\s*(?:\{|$)")

#: ``RoomType.Monster`` / ``EncounterTag.Nibbit`` / ``CardKeyword.X``
ROOM_TYPE = re.compile(r"RoomType\.(\w+)")
ENCOUNTER_TAG = re.compile(r"EncounterTag\.(\w+)")

#: ``IsEpochRevealed<UnderdocksEpoch>()``
EPOCH_REVEALED = re.compile(r"IsEpochRevealed\s*<\s*([A-Za-z0-9_.]+)\s*>")

#: 幕的三个 ``_rooms`` 列表（``RoomSet.cs:16-24``）。**只认这三个**，
#: 其它集合（比如 ``_rooms.events`` 之外的）不记，免得把不相干的替换当遭遇。
ROOM_LISTS = {
    "normalEncounters": "normal_encounters",
    "eliteEncounters": "elite_encounters",
    "events": "events",
}

#: ``ActModel.AllWeakEncounters`` 等四个谓词（``ActModel.cs:149-164``）。
#: 抽出来是为了让"分类规则"本身也有出处，而不是把规则硬编码在 Python 里。
PREDICATE_PROPERTIES = {
    "AllWeakEncounters": ("weak_encounters", "monster + is_weak"),
    "AllRegularEncounters": ("regular_encounters", "monster + not is_weak"),
    "AllEliteEncounters": ("elite_encounters", "elite"),
    "AllBossEncounters": ("boss_encounters", "boss"),
}

#: 社区数据**明确不采用**（铁律 1）。写在这里是让读者一眼看到边界。
CODEX_NOT_USED = "data/raw/spire-codex/repo/encounters.json"


# ==========================================================================
# C# 文本小工具
# ==========================================================================
#: 换行符常量。**不要**在 f-string 里直接写 ``"\n"``：那在 Python 3.11 及更早
#: 版本里是语法错误（f-string 的表达式部分不许出现反斜杠）。用常量既绕开限制，
#: 也不影响可读性。
NEWLINE = "\n"


def line_of(text: str, position: int) -> int:
    """``position`` 在 ``text`` 里的**1 起**行号。"""
    return text.count(NEWLINE, 0, position) + 1


def site(path: Path, text: str, position: int) -> str:
    """``文件:行`` —— 仓库相对路径，跨平台一致（反斜杠统一成 ``/``）。"""
    return f"{path.as_posix()}:{line_of(text, position)}"


def declaration_end(source: str, start: int) -> int | None:
    """从 ``source[start]`` 起找**声明体**的结尾，返回分号下标。

    ``property_expr`` 与 ``declaration_body`` 共用这一趟扫描，避免两处
    各写一份"什么时候算结束"的判断（口径不一致过一次：``property_expr``
    只认 ``=>``，于是 ``NumOfElites { get; init; } = …`` 这类**块体 + 初始化式**
    的属性一律抽不到，而且报出来的理由是"公式与预期不符"—— 理由错，误导人）。
    """
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == ";" and depth == 0:
            return index
    return None


def property_expr(source: str, name: str) -> tuple[str, int, int] | None:
    """取 ``Name => <表达式>;`` 的表达式，返回 ``(表达式, 起, 止)``。

    用**圆括号/方括号配平**找结尾，不用 ``(.+?);`` —— 表达式里可能带
    ``new ...(...)`` 这类含分号的构造，非贪婪到第一个分号会截断。
    """
    match = re.search(rf"\b{name}\s*=>", source)
    if not match:
        return None
    start = match.end()
    end = declaration_end(source, start)
    return None if end is None else (source[start:end], start, end)


def declaration_body(source: str, name: str) -> tuple[str, int, int] | None:
    """取**整个属性声明** ``Name … ;`` 的文本，返回 ``(声明, 起, 止)``。

    ``MapPointTypeCounts`` 的三个常量都是 ``{ get; init; } = <值>`` 形式
    （``MapPointTypeCounts.cs:12-16``），**没有 ``=>``** —— 用
    :func:`property_expr` 一个都取不到。
    """
    match = re.search(rf"\b{name}\b", source)
    if not match:
        return None
    start = match.start()
    end = declaration_end(source, match.end())
    return None if end is None else (source[start:end], start, end)


def initializer_expr(declaration: str) -> str | None:
    """``… { get; init; } = 12;`` → ``12``（取**顶层**等号右边）。"""
    depth = 0
    for index, char in enumerate(declaration):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "=" and depth == 0:
            if declaration[index:index + 2] == "=>":
                continue
            return declaration[index + 1:].rstrip(";").strip()
    return None


def int_property(source: str, name: str) -> tuple[int | None, str | None, int]:
    """取 ``=> <整数>`` 的属性，返回 ``(值, 原始文本, 行号)``。

    值解析不出时返回 ``None`` 且**保留原始文本** —— 调用方要么据此报缺口，
    要么生成一条带 ``raw`` 的 ``unsupported``。绝不猜一个数字。
    """
    found = property_expr(source, name)
    if found is None:
        return None, None, 0
    expr, start, _end = found
    raw = " ".join(expr.split())
    number = re.fullmatch(r"(-?\d+)", raw)
    return (int(number.group(1)) if number else None), raw, line_of(source, start)


def method_body(source: str, name: str) -> tuple[str, int] | None:
    """取**方法**体 ``Name(...) { … }`` 的内容与** ``{`` 在源文件里的偏移**。

    ⚠️ 返回偏移而**不是行号**：拿到方法体之后再往里面定位（``body[:k]``）
    时，只有偏移能与 :func:`site` 直接相加；混用行号会把行数算成两遍
    （实测把 ``GenerateAllEncounters`` 的出处写成了文件第 4 行）。
    """
    match = re.search(rf"\b{name}\s*\([^)]*\)\s*\{{", source)
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
                return source[start + 1:index], start
    return None


def model_refs(expr: str, kind: str, path: Path, text: str,
               offset: int = 0) -> list[tuple[str, str]]:
    """``[(类名, 出处)]`` —— 表达式里 ``ModelDb.<kind><T>()`` 的 T，按出现顺序。"""
    return [(m.group(2), site(path, text, offset + m.start()))
            for m in MODEL_REF.finditer(expr) if m.group(1) == kind]


def declared_length(expr: str) -> int | None:
    """``new EncounterModel[22]`` → ``22``。用来对账"数出来几个"。"""
    match = ARRAY_LENGTH.search(expr)
    return int(match.group(1)) if match else None


# ==========================================================================
# 反编译树索引
# ==========================================================================
def class_index() -> dict[str, list[Path]]:
    """``类名 → 文件``（同名多个文件时全都留着，调用方要判歧义）。"""
    index: dict[str, list[Path]] = collections.defaultdict(list)
    for path in sorted(DECOMPILED.rglob("*.cs")):
        index[path.stem].append(path)
    return index


def base_of(text: str) -> str | None:
    """``class X : Y`` 的 ``Y``（泛型参数去掉）。"""
    match = CLASS_DECL.search(text)
    if not match:
        return None
    return match.group(2).split("<")[0].strip()


def single_source(index: dict[str, list[Path]], name: str,
                  prefer_dir: Path | None = None) -> Path | None:
    """唯一文件才返回；找不到或歧义都返回 ``None``。

    歧义**必须**当缺口报出来：真机里 ``DecimillipedeSegment`` 这类"多文件同名"
    是存在的，随便挑一个会安静地读错文件。

    ``prefer_dir`` 是**合法的消歧依据**，不是猜：``LostWisp`` 在
    ``Models.Events``（事件）与 ``Models.Relics``（遗物）下各有一个，
    而调用方要的是 ``EventModel`` 的子类 —— 类型所在的命名空间就是答案。
    只有在先验目录里仍然找不到，或仍然有多个时才算歧义。
    """
    found = index.get(name) or []
    if prefer_dir is not None:
        scoped = [p for p in found
                  if p.parent.as_posix() == prefer_dir.as_posix()]
        if len(scoped) == 1:
            return scoped[0]
        if scoped:
            return None
    return found[0] if len(found) == 1 else None


# ==========================================================================
# 遭遇 / 事件 元数据
# ==========================================================================
#: ``RoomType`` 枚举（``MegaCrit.Sts2.Core.Rooms/RoomType.cs:32-43``）。
#: 只认这几个；认不出的值照实记，由调用方报缺口。
KNOWN_ROOM_TYPES = frozenset(
    {"Unassigned", "Monster", "Elite", "Boss", "Treasure", "Shop", "Event",
     "RestSite", "Map"})


def encounter_metadata(index: dict[str, list[Path]], name: str,
                       gaps: ActGaps) -> dict:
    """一个遭遇类的 ``RoomType`` / ``IsWeak`` / ``Tags``。

    ``IsWeak`` 与 ``Tags`` 都是 ``virtual``，**没重写就是继承来的默认值**
    （``EncounterModel.cs:56`` 的 ``IsWeak => false``、
    ``:109`` 的 ``Tags => Array.Empty<EncounterTag>()``）。所以"文件里没有"
    等于"false / 空集"，不是"抽不出来" —— 这一点必须写清楚，
    否则每一只精英怪都会被标成缺口。
    """
    path = single_source(index, name, ENCOUNTERS_DIR)
    if path is None:
        gaps.model(name, f"遭遇类 {name} 在反编译树里找不到唯一文件"
                         f"（层 {ENCOUNTERS_DIR.as_posix()}）")
        return {"id": snake(name), "class": name, "room_type": None,
                "is_weak": None, "tags": [], "csharp_source": None}
    text = path.read_text(encoding="utf-8", errors="replace")
    room = property_expr(text, "RoomType")
    room_value = None
    room_site = None
    if room is not None:
        match = ROOM_TYPE.search(room[0])
        if match:
            room_value = match.group(1)
            room_site = site(path, text, room[1] + match.start())
        else:
            gaps.model(name,
                       f"遭遇 {name} 的 RoomType 不是 ``RoomType.X`` 形式："
                       f"{room[0].strip()[:60]!r}（{site(path, text, room[1])}）")
    else:
        gaps.model(name, f"遭遇 {name} 没有 RoomType 声明（{path.as_posix()}）")
    if room_value is not None and room_value not in KNOWN_ROOM_TYPES:
        gaps.model(name,
                   f"遭遇 {name} 的 RoomType={room_value} 不在 RoomType 枚举里"
                   f"（{room_site}）")

    weak = re.search(r"override\s+bool\s+IsWeak\s*=>\s*(true|false)", text)
    is_weak = bool(weak and weak.group(1) == "true")
    weak_site = site(path, text, weak.start()) if weak else None
    # 没重写 → 用基类默认 false。出处写基类那一行，别写空。
    if weak is None:
        base_path = DECOMPILED / "MegaCrit.Sts2.Core.Models" / "EncounterModel.cs"
        weak_site = f"{base_path.as_posix()}:56"

    tags_prop = property_expr(text, "Tags")
    tags = sorted(set(ENCOUNTER_TAG.findall(tags_prop[0]))) if tags_prop else []
    # ⚠️ ``EncounterTag.None`` 是**枚举的哨兵值**，不是标签。把它当标签留下的后果
    # 不是"多一个字段"，而是 ``SharesTagsWith`` 会把**两只都没标签**的遭遇
    # 判成"共享标签"（``EncounterModel.cs:274-281`` 取交集），于是
    # ``AddWithoutRepeatingTags``（``ActModel.cs:419-430``）额外拒绝它们相邻 ——
    # 整幕的遭遇顺序分布都会偏。所以必须丢掉。
    tags = [tag for tag in tags if tag != "None"]
    tags_site = site(path, text, tags_prop[1]) if tags_prop else \
        f"{(DECOMPILED / 'MegaCrit.Sts2.Core.Models' / 'EncounterModel.cs').as_posix()}:109"

    return {
        "id": snake(name),
        "class": name,
        "room_type": room_value,
        "is_weak": is_weak,
        "tags": tags,
        "csharp_source": path.as_posix(),
        "room_type_source": room_site,
        "is_weak_source": weak_site,
        "tags_source": tags_site,
    }


def model_metadata(index: dict[str, list[Path]], name: str, expected_base: str,
                   gaps: ActGaps) -> dict:
    """事件 / 远古类的 ``class`` 与文件出处。

    先验目录由**基类名**决定：``AncientEventModel`` 与 ``EventModel`` 都在
    ``MegaCrit.Sts2.Core.Models.Events`` 下（``AncientEventModel : EventModel``，
    见 ``AncientEventModel.cs``），所以两者用同一个目录消歧。
    """
    prefer = ELDER_EVENTS_DIR if expected_base in ("EventModel",
                                                   "AncientEventModel") else None
    path = single_source(index, name, prefer)
    if path is None:
        gaps.model(name, f"模型类 {name} 在反编译树里找不到唯一文件"
                         f"（应为 {expected_base} 的子类）")
        return {"id": snake(name), "class": name, "csharp_source": None}
    text = path.read_text(encoding="utf-8", errors="replace")
    actual = base_of(text)
    if actual != expected_base:
        gaps.model(name,
                   f"模型类 {name} 的基类是 {actual!r}，不是预期的 {expected_base}"
                   f"（{path.as_posix()}）")
    return {"id": snake(name), "class": name, "csharp_source": path.as_posix()}


# ==========================================================================
# 地图点类型计数
# ==========================================================================
def map_point_counts(act_source: str, act_path: Path,
                     point_counts_source: str,
                     gaps: ActGaps) -> dict:
    """逐字抄 ``GetMapPointTypes``，并把 ``MapPointTypeCounts`` 的常量一并记下。

    ``elites`` / ``shops`` / ``point_types_that_ignore_rules`` **不在幕文件里**：
    它们写在 ``MapPointTypeCounts`` 的初始化式上（``MapPointTypeCounts.cs:12-16``），
    幕文件只是 ``new MapPointTypeCounts(unknown, rest)`` 用了那个构造器
    （``:35-39``，参数顺序是 ``(unknownCount, restCount)``）。所以这里同时读两个文件。
    """
    body = method_body(act_source, "GetMapPointTypes")
    result: dict = {}
    if body is None:
        gaps.add(
            f"幕文件里找不到 GetMapPointTypes（{act_path.as_posix()}）")
        return {"rest": None, "unknown": None, "rest_source": None,
                "unknown_source": None}

    text, body_at = body

    # ---- rest ----------------------------------------------------------
    gauss = GAUSSIAN_INT.search(text)
    next_int = NEXT_INT.search(text)
    if gauss:
        result["rest"] = {
            "kind": "gaussian_int",
            "mean": int(gauss.group(1)), "stddev": int(gauss.group(2)),
            "min": int(gauss.group(3)), "max": int(gauss.group(4)),
            "bounds": "inclusive",
            "raw": " ".join(gauss.group(0).split()),
        }
        result["rest_source"] = site(act_path, act_source, body_at + gauss.start())
    elif next_int:
        result["rest"] = {
            "kind": "next_int",
            "min": int(next_int.group(1)), "max": int(next_int.group(2)),
            # ⚠️ ``NextInt(minInclusive, maxExclusive)``：上界取不到
            # （``Rng.cs:89-103``）。这里把语义写进数据，读的人不用回源码。
            "max_exclusive": True,
            "bounds": "min_inclusive_max_exclusive",
            "raw": " ".join(next_int.group(0).split()),
        }
        result["rest_source"] = site(act_path, act_source, body_at + next_int.start())
    else:
        result["rest"] = None
        result["rest_source"] = None
        gaps.add(
            f"GetMapPointTypes 里的 rest 计数不是 NextGaussianInt/NextInt 字面量"
            f"（{site(act_path, act_source, body_at)} 起）")

    # ---- unknown -------------------------------------------------------
    standard = STANDARD_UNKNOWN.search(text)
    if standard is None:
        result["unknown"] = None
        result["unknown_source"] = None
        gaps.add(
            f"GetMapPointTypes 没有调用 MapPointTypeCounts.StandardRandomUnknownCount"
            f"（{site(act_path, act_source, body_at)} 起）：unknown 的分布参数在"
            f"{POINT_COUNTS.as_posix()}:30-33，必须显式引用才能确定是标准分布")
    else:
        # 标准分布的参数**逐字**来自 MapPointTypeCounts.StandardRandomUnknownCount。
        std = GAUSSIAN_INT.search(point_counts_source)
        base = {
            "kind": "gaussian_int",
            "mean": int(std.group(1)), "stddev": int(std.group(2)),
            "min": int(std.group(3)), "max": int(std.group(4)),
            "bounds": "inclusive",
            "raw": " ".join(std.group(0).split()),
            "call": "MapPointTypeCounts.StandardRandomUnknownCount",
            "call_source": f"{POINT_COUNTS.as_posix()}:30-33",
        } if std else None
        if base is None:
            gaps.add(
                f"StandardRandomUnknownCount 的参数不是 NextGaussianInt 字面量"
                f"（{POINT_COUNTS.as_posix()}）")
        # 叠加的偏移量（Hive/Glory 是 -1）：直接把源码里的常量抄下来。
        tail = text[standard.end():]
        offset_match = re.match(r"\s*([+-])\s*(\d+)", tail)
        offset = 0
        if offset_match:
            offset = int(offset_match.group(2))
            if offset_match.group(1) == "-":
                offset = -offset
        elif not re.match(r"\s*[;,)]", tail):
            gaps.add(
                f"StandardRandomUnknownCount 之后接的不是常量偏移，是 "
                f"{tail.strip()[:40]!r}（{act_path.as_posix()}）：unknown 的最终取值"
                f"抽不出来，照实报缺口")
        result["unknown"] = {
            "base": base,
            "offset": offset,
            "base_source": site(act_path, act_source, body_at + standard.start()),
            "min": (base["min"] + offset) if base else None,
            "max": (base["max"] + offset) if base else None,
        }
        result["unknown_source"] = result["unknown"]["base_source"]

    # ---- elites / shops / point_types_that_ignore_rules ----------------
    # ⚠️ 这三个字段**没有 ``=>``**，是 ``{ get; init; } = <值>`` 形式
    # （``MapPointTypeCounts.cs:12-16``）。所以必须用 ``declaration_body``
    # 连初始化式一起取，用 ``property_expr`` 会一个都取不到 —— 而且报出来的
    # 理由是"公式与预期不符"，把人引向错误方向（实测踩过一次）。
    elites_raw, elites_line, elites_init = None, None, None
    elites_decl = declaration_body(point_counts_source, "NumOfElites")
    if elites_decl:
        declaration, start, _end = elites_decl
        elites_raw = " ".join(declaration.split())
        elites_line = line_of(point_counts_source, start)
        elites_init = initializer_expr(declaration)
    shops_decl = declaration_body(point_counts_source, "NumOfShops")
    ignore_decl = declaration_body(point_counts_source, "PointTypesThatIgnoreRules")
    ignore_raw = None
    if ignore_decl:
        ignore_raw = " ".join((initializer_expr(ignore_decl[0]) or "").split()) or None

    # 公式里的两个常量都从源码读，不写死。
    formula_text = elites_init or elites_raw or ""
    factor = re.search(r"\*\s*(\d+(?:\.\d+)?)f", formula_text)
    base_elites = re.search(r"Math\.Round\(\s*(\d+(?:\.\d+)?)f", formula_text)
    ascension = re.search(r"AscensionLevel\.(\w+)", formula_text)
    boost = re.search(r"\?\s*(\d+(?:\.\d+)?)f\s*:\s*(\d+(?:\.\d+)?)f",
                      formula_text)
    elites: dict = {
        "raw": elites_raw,
        "initializer": elites_init,
        "source": f"{POINT_COUNTS.as_posix()}:{elites_line}",
    }
    if base_elites and boost:
        normal = float(base_elites.group(1)) * float(boost.group(2))
        boosted = float(base_elites.group(1)) * float(boost.group(1))
        # C# ``Math.Round(double)`` 用银行家舍入（ToEven），Python 的 ``round``
        # 同样如此 —— 但这里**不隐藏**这一步，把公式和两个取值都写出来。
        elites.update({
            "base": float(base_elites.group(1)),
            "ascension_level": ascension.group(1) if ascension else None,
            "ascension_multiplier": float(boost.group(1)),
            "default_multiplier": float(boost.group(2)),
            "value_default": int(round(normal)),
            "value_with_ascension": int(round(boosted)),
            "formula": "round(base * (SwarmingElites ? ascension_multiplier : default_multiplier))",
        })
    else:
        gaps.add(
            f"NumOfElites 的公式与预期不符：initializer={elites_init!r}"
            f" declaration={elites_raw!r}"
            f"（{POINT_COUNTS.as_posix()}:{elites_line}）")
    if factor and not base_elites:
        gaps.add(
            f"NumOfElites 认出了倍数 {factor.group(1)} 但认不出基数"
            f"（{POINT_COUNTS.as_posix()}:{elites_line}）")

    shops: dict = {"raw": None, "source": None, "value": None}
    if shops_decl:
        declaration, start, _end = shops_decl
        raw = " ".join(declaration.split())
        value_raw = initializer_expr(declaration)
        number = re.fullmatch(r"(\d+)", (value_raw or "").strip())
        shops = {
            "raw": raw,
            "initializer": value_raw,
            "source": f"{POINT_COUNTS.as_posix()}:{line_of(point_counts_source, start)}",
            "value": int(number.group(1)) if number else None,
        }
        if number is None:
            gaps.add(
                f"NumOfShops 不是字面量常量：initializer={value_raw!r}"
                f" declaration={raw!r}（{shops['source']}）")
    else:
        gaps.add(f"找不到 NumOfShops（{POINT_COUNTS.as_posix()}）")

    if ignore_decl is None:
        ignore_types: list[str] = []
        ignore_source = None
        gaps.add(
            f"找不到 PointTypesThatIgnoreRules（{POINT_COUNTS.as_posix()}）")
    else:
        ignore_source = \
            f"{POINT_COUNTS.as_posix()}:{line_of(point_counts_source, ignore_decl[1])}"
        # 空集就是 ``new HashSet<MapPointType>()``（``MapPointTypeCounts.cs:12``）
        # 的默认值 —— 四张幕都用这个构造器，没有一处改它。这就是"默认空集"。
        ignore_types = sorted(set(ROOM_TYPE.findall(ignore_raw or "")))
        if ignore_types:
            gaps.add(
                f"PointTypesThatIgnoreRules 在基类里非空：{ignore_types}"
                f"（{ignore_source}）")
        if ignore_raw is None:
            gaps.add(
                f"PointTypesThatIgnoreRules 的声明里没有 ``= <值>``："
                f"{' '.join(ignore_decl[0].split())[:80]!r}（{ignore_source}）")

    result["elites"] = elites
    result["shops"] = shops
    result["point_types_that_ignore_rules"] = ignore_types
    result["point_types_that_ignore_rules_source"] = ignore_source
    # ``elites`` / ``shops`` **不由幕文件给**：幕文件只调
    # ``new MapPointTypeCounts(unknown, rest)``（``MapPointTypeCounts.cs:35-39``），
    # 其余三个字段走 ``MapPointTypeCounts`` 自己的初始化式。把这件事写进数据，
    # 免得读的人去幕文件里找 NumOfElites 而找不到。
    result["inherited_from_base_source"] = {
        "elites": f"{POINT_COUNTS.as_posix()}:14",
        "shops": f"{POINT_COUNTS.as_posix()}:16",
        "point_types_that_ignore_rules": f"{POINT_COUNTS.as_posix()}:12",
        "constructor": f"{POINT_COUNTS.as_posix()}:35-39",
    }
    # 幕文件的 ``new MapPointTypeCounts(unknown, rest)``：参数顺序也要有出处，
    # 反了会让 rest 与 unknown 互换（而且不会报错）。
    ctor = re.search(r"new\s+MapPointTypeCounts\s*\(([^)]*)\)", text)
    if ctor:
        args = " ".join(ctor.group(1).split())
        result["constructor_args"] = args
        result["constructor_source"] = site(act_path, act_source, body_at + ctor.start())
        if "unknown" not in args.split(",")[0] or "rest" not in args.split(",")[-1]:
            gaps.add(
                f"new MapPointTypeCounts 的实参顺序与预期 (unknown, rest) 不符："
                f"{args!r}（{result['constructor_source']}）")
    else:
        gaps.add(
            f"GetMapPointTypes 里没有 new MapPointTypeCounts(...)"
            f"（{site(act_path, act_source, body_at)} 起）")
    return result


def classification_rules(act_model_source: str, path: Path) -> dict:
    """抄 ``ActModel`` 的四个 ``Where`` 谓词 —— 分类规则本身也要有出处。"""
    rules: dict[str, dict] = {}
    for prop, (key, human) in PREDICATE_PROPERTIES.items():
        found = property_expr(act_model_source, prop)
        if found is None:
            rules[key] = {"predicate": None, "source": None}
            continue
        expr, start, _end = found
        rules[key] = {
            "predicate": " ".join(expr.split()),
            "human": human,
            "source": site(path, act_model_source, start),
            "is_null_checked": "!= null" in expr or "is not null" in expr,
        }
    return rules


# ==========================================================================
# 单张幕
# ==========================================================================
class ActGaps:
    """一张幕自己的缺口表。

    两本账分开记，是为了让 JSON 里读得懂：
      * :meth:`add` —— **这张幕**的问题（源文件、行号齐全），进该幕记录的
        ``unsupported``；
      * :meth:`top` —— **跨幕**的问题（某个遭遇类在反编译树里找不到唯一文件、
        某个类的基类不对…）。它不属于任何一张幕，硬塞进幕记录会让
        "这张幕缺了什么"变得不可读；由 :func:`build` 汇总后单独打印。

    ``kind="model"`` 的按**类名去重**：同一个遭遇类被两张幕引用时，
    "这个类找不到文件"会在两张幕里各报一次 —— 那是重复噪声，不是两个问题。
    其它按整条文本去重。
    """

    def __init__(self) -> None:
        self.entries: list[str] = []
        self.top_entries: list[str] = []
        self._seen_models: set[str] = set()

    def add(self, message: str) -> None:
        self.entries.append(message)

    def model(self, name: str, message: str) -> None:
        marker = f"{name}\0{message}"
        if marker in self._seen_models:
            return
        self._seen_models.add(marker)
        self.top_entries.append(message)


def parse_act(path: Path, index: dict[str, list[Path]],
              point_counts_source: str,
              rules: dict, gaps: ActGaps) -> dict:
    source = path.read_text(encoding="utf-8", errors="replace")
    class_name = path.stem
    act_id = snake(class_name)

    # ---- 声明 ---------------------------------------------------------
    act_index, act_index_raw, act_index_line = int_property(source, "Index")
    is_default_prop = property_expr(source, "IsDefault")
    is_default = bool(is_default_prop and "true" in is_default_prop[0].lower())
    weak_count, weak_raw, weak_line = int_property(source, "NumberOfWeakEncounters")
    base_rooms, base_rooms_raw, base_rooms_line = int_property(
        source, "BaseNumberOfRooms")
    if weak_count is None:
        gaps.add(
            f"{class_name}.NumberOfWeakEncounters 不是字面量：{weak_raw!r}"
            f"（{path.as_posix()}:{weak_line}）")
    if base_rooms is None:
        gaps.add(
            f"{class_name}.BaseNumberOfRooms 不是字面量：{base_rooms_raw!r}"
            f"（{path.as_posix()}:{base_rooms_line}）")
    if act_index is None:
        gaps.add(
            f"{class_name}.Index 不是字面量：{act_index_raw!r}"
            f"（{path.as_posix()}:{act_index_line}）")

    # ---- is_unlocked --------------------------------------------------
    unlocked = method_body(source, "IsUnlocked")
    is_unlocked: object = None
    is_unlocked_source = None
    if unlocked is None:
        gaps.add(f"{class_name} 没有 IsUnlocked 实现（{path.as_posix()}）")
    else:
        body, body_at = unlocked
        epoch = EPOCH_REVEALED.search(body)
        if re.search(r"\breturn\s+true\s*;", body):
            is_unlocked = True
            is_unlocked_source = site(path, source, body_at)
        elif epoch:
            is_unlocked = epoch.group(1)
            is_unlocked_source = site(path, source, body_at + epoch.start())
        else:
            gaps.add(
                f"{class_name}.IsUnlocked 既不是 ``return true`` 也不是"
                f" ``IsEpochRevealed<T>``（{site(path, source, body_at)} 起）")

    # ---- 地图点类型 ---------------------------------------------------
    counts = map_point_counts(source, path, point_counts_source, gaps)

    # ---- 池 -----------------------------------------------------------
    boss_order_prop = property_expr(source, "BossDiscoveryOrder")
    boss_order: list[dict] = []
    boss_order_source = None
    if boss_order_prop:
        boss_order_source = site(path, source, boss_order_prop[1])
        boss_order = [model_metadata(index, name, "EncounterModel", gaps)
                      for name, _ in model_refs(boss_order_prop[0], "Encounter",
                                                path, source, boss_order_prop[1])]
        length = declared_length(boss_order_prop[0])
        if length is not None and length != len(boss_order):
            gaps.add(
                f"{class_name}.BossDiscoveryOrder 声明 {length} 个但抽到 "
                f"{len(boss_order)} 个（{boss_order_source}）")
    else:
        gaps.add(f"{class_name} 没有 BossDiscoveryOrder（{path.as_posix()}）")

    ancients_prop = property_expr(source, "AllAncients")
    ancients: list[dict] = []
    ancients_source = None
    if ancients_prop:
        ancients_source = site(path, source, ancients_prop[1])
        ancients = [model_metadata(index, name, "AncientEventModel", gaps)
                    for name, _ in model_refs(ancients_prop[0], "AncientEvent",
                                              path, source, ancients_prop[1])]
        length = declared_length(ancients_prop[0])
        if length is not None and length != len(ancients):
            gaps.add(
                f"{class_name}.AllAncients 声明 {length} 个但抽到 {len(ancients)} 个"
                f"（{ancients_source}）")
    else:
        gaps.add(f"{class_name} 没有 AllAncients（{path.as_posix()}）")

    events_prop = property_expr(source, "AllEvents")
    events: list[dict] = []
    events_source = None
    if events_prop:
        events_source = site(path, source, events_prop[1])
        events = [model_metadata(index, name, "EventModel", gaps)
                  for name, _ in model_refs(events_prop[0], "Event",
                                            path, source, events_prop[1])]
        length = declared_length(events_prop[0])
        if length is not None and length != len(events):
            gaps.add(
                f"{class_name}.AllEvents 声明 {length} 个但抽到 {len(events)} 个"
                f"（{events_source}）")
    else:
        gaps.add(f"{class_name} 没有 AllEvents（{path.as_posix()}）")

    # ---- GenerateAllEncounters ---------------------------------------
    encounter_list: list[dict] = []
    encounters_source = None
    generated = method_body(source, "GenerateAllEncounters")
    if generated is None:
        gaps.add(
            f"{class_name} 没有 GenerateAllEncounters（{path.as_posix()}）")
    else:
        body, body_at = generated
        refs = model_refs(body, "Encounter", path, source, body_at)
        encounters_source = site(path, source, body_at)
        length = declared_length(body)
        if length is not None and length != len(refs):
            gaps.add(
                f"{class_name}.GenerateAllEncounters 声明 {length} 个但抽到 "
                f"{len(refs)} 个（{encounters_source}）")
        for name, ref_site in refs:
            meta = encounter_metadata(index, name, gaps)
            meta["source"] = ref_site
            encounter_list.append(meta)

    # ---- SwapToOrCreateAtIndex（首局固定顺序）------------------------
    swaps: list[dict] = []
    discover = method_body(source, "ApplyActDiscoveryOrderModifications")
    discover_source = None
    if discover is None:
        gaps.add(
            f"{class_name} 没有 ApplyActDiscoveryOrderModifications（{path.as_posix()}）")
    else:
        body, body_at = discover
        discover_source = site(path, source, body_at)
        for match in SWAP.finditer(body):
            base_type, specific, collection, desired = match.groups()
            key = ROOM_LISTS.get(collection)
            if key is None:
                gaps.add(
                    f"{class_name} 的 SwapToOrCreateAtIndex 指向未知集合 "
                    f"{collection!r}（{site(path, source, body_at + match.start())}）")
                continue
            entry = {
                "collection": key,
                "index": int(desired),
                "model": snake(specific),
                "class": specific,
                "base_type": base_type,
                "source": site(path, source, body_at + match.start()),
            }
            if key == "events":
                entry["kind"] = "event"
            else:
                entry["kind"] = "encounter"
            entry["semantics"] = (
                "RoomSet.SwapToOrCreateAtIndex：列表里已有该模型 → 与 index 处的元素"
                "互换；没有 → 把 index 处替换成该模型的新实例"
                "（RoomSet.cs:177-192）")
            swaps.append(entry)
        # 只在「首局」分支里生效 —— 条件本身也是数据。
        guard = re.search(r"if\s*\(\s*unlockState\.(\w+)\s*([<>=!]+)\s*(\d+)\s*\)", body)
        if guard:
            discover_guard = {
                "condition": f"unlockState.{guard.group(1)} {guard.group(2)} {guard.group(3)}",
                "source": site(path, source, body_at + guard.start()),
            }
        else:
            discover_guard = None
    if discover is not None and not swaps:
        # 空实现是**正常**的（Underdocks / Hive / Glory 都是空的）：
        # 它们没有首局固定顺序。但要留下出处，别和"没抽到"混淆。
        swaps_note = "该方法为空实现：本幕没有首局固定顺序"
    else:
        swaps_note = ""

    # ---- 遭遇分类（照抄 ActModel 的谓词）-----------------------------
    weak_pool: list[str] = []
    regular_pool: list[str] = []
    elite_pool: list[str] = []
    boss_pool: list[str] = []
    for entry in encounter_list:
        room, is_weak = entry["room_type"], entry["is_weak"]
        if room is None or is_weak is None:
            # 抽不出就只能**不分类**。绝不能默认成 False 落进普通池 ——
            # 那会把一场弱遭遇悄悄当成普通遭遇（分布变了，还不报错）。
            gaps.add(
                f"遭遇 {entry['class']} 的 RoomType={room!r} / IsWeak={is_weak!r} "
                f"抽不出，按 ActModel.AllWeakEncounters 等谓词无法分类"
                f"（{entry.get('source')}）")
        elif room == "Monster" and is_weak is True:
            weak_pool.append(entry["id"])
        elif room == "Monster" and is_weak is False:
            regular_pool.append(entry["id"])
        elif room == "Elite":
            elite_pool.append(entry["id"])
        elif room == "Boss":
            boss_pool.append(entry["id"])
        else:
            gaps.add(
                f"遭遇 {entry['class']} 的 RoomType={room} 不属于 "
                f"Monster/Elite/Boss 三类，GenerateAllEncounters 里为什么会有它"
                f"？（{entry.get('source')}）")

    # ⚠️ ``normal_encounters`` 按 **GenerateAllEncounters 的源码顺序**排，
    # 不是"弱池 + 普通池"拼接 —— 真机的 ``_rooms.normalEncounters`` 是在
    # ``GenerateRooms`` 里按**抽取顺序**追加的（``ActModel.cs:358``/``:370``），
    # 与源码顺序无关；但"哪些遭遇算普通"由 ``AllEncounters`` 的
    # ``RoomType == Monster`` 决定。拼接会把两个池子的组内顺序强加给这张表，
    # 读的人无从判断顺序是哪来的。
    normal_pool = [entry["id"] for entry in encounter_list
                   if entry["room_type"] == "Monster"]
    # 与 ``ActModel.GenerateRooms``（``ActModel.cs:349-371``）核对抽取逻辑。
    # ⚠️ 这里**不是**在报缺口：普通遭遇槽位数大于池子大小时，真机的
    # ``GrabBag`` 会**重新装满**再抽（``ActModel.cs:351-357`` /
    # ``:363-369`` 的 ``if (!grabBag.Any()) { foreach (var e in AllXxx) Add(e) }``），
    # 于是同一场遭遇会在**同一幕里出现两次**。这是真机行为，抽取没有错，
    # 但"池子不够长"这件事必须**显式记下来** —— 静默按"池子大小=槽位数"
    # 建模会让每一局的普通遭遇分布都偏（不是少一条数据，是整幕的分布错）。
    pool_refill: dict = {}
    if weak_count is not None and base_rooms is not None:
        normal_slots = base_rooms
        pool_refill = {
            "normal_slots": normal_slots,
            "weak_slots": weak_count,
            "regular_slots": normal_slots - weak_count,
            "weak_pool_size": len(weak_pool),
            "regular_pool_size": len(regular_pool),
            "weak_pool_refills": weak_count > len(weak_pool),
            "regular_pool_refills": normal_slots - weak_count > len(regular_pool),
            "note": ("GrabBag 每次被取空就按 AllWeakEncounters / AllRegularEncounters "
                     "重新装满（ActModel.cs:351-357 / :363-369），因此"
                     "``*_pool_refills=true`` 时同一场遭遇会在同一幕里重复出现"),
            "source": "data/decompiled/sts2/MegaCrit.Sts2.Core.Models/ActModel.cs:349-371",
        }

    # 取用方式：三张池子都是"**取完一轮从头再来**"，用访问计数取模
    # （``RoomSet.cs:70-86``）。记下来是因为它决定"池子大小 = 一轮里最多
    # 出现几种"，与 :data:`pool_refill` 是两件事（一个管同一幕内重复，
    # 一个管整局内循环）。静默按"每局只出现一次"建模会让遭遇分布整体偏。
    pool_cycles = {
        "normal": "RoomSet.NextNormalEncounter = normalEncounters[visited % Count]"
                  "（RoomSet.cs:72）",
        "elite": "RoomSet.NextEliteEncounter = eliteEncounters[visited % Count]"
                 "（RoomSet.cs:74）",
        "event": "RoomSet.NextEvent = events[visited % Count]（RoomSet.cs:70）",
        "boss": "RoomSet.NextBossEncounter：第一次给 Boss，第二次起给 SecondBoss"
                "（若设置）（RoomSet.cs:76-86）",
        "source": "data/decompiled/sts2/MegaCrit.Sts2.Core.Rooms/RoomSet.cs:70-86",
    }

    result = {
        "id": act_id,
        "csharp_class": class_name,
        "csharp_source": path.as_posix(),
        "index": act_index,
        "index_source": f"{path.as_posix()}:{act_index_line}",
        "is_default": is_default,
        "is_default_source": site(path, source, is_default_prop[1]) if is_default_prop else None,
        "base_number_of_rooms": base_rooms,
        "base_number_of_rooms_raw": base_rooms_raw,
        "base_number_of_rooms_source": f"{path.as_posix()}:{base_rooms_line}",
        "number_of_weak_encounters": weak_count,
        "number_of_weak_encounters_raw": weak_raw,
        "number_of_weak_encounters_source": f"{path.as_posix()}:{weak_line}",
        "is_unlocked": is_unlocked,
        "is_unlocked_source": is_unlocked_source,
        "map_point_counts": counts,
        "ancients": ancients,
        "ancients_source": ancients_source,
        "events": events,
        "events_source": events_source,
        "all_encounters": encounter_list,
        "encounters_source": encounters_source,
        "normal_encounters": normal_pool,
        "weak_encounters": weak_pool,
        "regular_encounters": regular_pool,
        "elite_encounters": elite_pool,
        "boss_encounters": boss_pool,
        "pool_refill": pool_refill,
        "pool_cycles": pool_cycles,
        "boss_discovery_order": boss_order,
        "boss_discovery_order_source": boss_order_source,
        "discovery_order_swaps": swaps,
        "discovery_order_swaps_source": discover_source,
        "discovery_order_swaps_note": swaps_note,
        "discovery_order_guard": discover_guard,
        "encounter_classification": rules,
    }
    return result


# ==========================================================================
# 主流程
# ==========================================================================
def build(src_root: Path) -> tuple[list[dict], list[str], dict]:
    """抽取全部幕。返回 ``(记录, 缺口, 统计)``。"""
    unsupported: list[str] = []
    index = class_index()
    act_model_path = src_root / ACT_MODEL.relative_to(DECOMPILED)
    point_counts_path = src_root / POINT_COUNTS.relative_to(DECOMPILED)
    acts_dir = src_root / ACTS_DIR.relative_to(DECOMPILED)

    if not act_model_path.exists():
        raise SystemExit(f"找不到 {act_model_path}（反编译源码是唯一真相，缺了不能猜）")
    if not point_counts_path.exists():
        raise SystemExit(f"找不到 {point_counts_path}")
    if not acts_dir.exists():
        raise SystemExit(f"找不到 {acts_dir}")

    act_model_source = act_model_path.read_text(encoding="utf-8", errors="replace")
    point_counts_source = point_counts_path.read_text(
        encoding="utf-8", errors="replace")
    rules = classification_rules(act_model_source, act_model_path)

    # 幕后选：``ActModel`` 的具体子类，且 ``Index >= 0``
    # （``ModelDb.ActsByIndex`` 就是按这个条件分组的，``ModelDb.cs:323-345``）。
    # ``DeprecatedAct`` 的 Index 是 -1，必须排除 —— 它不是"第 4 张幕"。
    candidates: list[Path] = []
    for path in sorted(acts_dir.glob("*.cs")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if base_of(text) != "ActModel":
            continue
        if path.stem == "ActModel":
            continue
        value, _raw, _line = int_property(text, "Index")
        if value is None or value < 0:
            unsupported.append(
                f"{path.stem} 是 ActModel 的子类但 Index={_raw!r}（<0 或非字面量），"
                f"按 ModelDb.ActsByIndex 的规则（ModelDb.cs:334）它不进幕列表"
                f"（{path.as_posix()}:{_line}）")
            continue
        candidates.append(path)

    if len(candidates) != 4:
        unsupported.append(
            f"反编译树里有 {len(candidates)} 个 Index>=0 的 ActModel 子类，"
            f"预期 4（Overgrowth / Underdocks / Hive / Glory）："
            f"{[p.stem for p in candidates]}")

    records = []
    top_gaps: list[str] = []
    for path in candidates:
        # 每张幕一份**自己的**缺口表：``parse_act`` 认得出的缺口记到这张幕上
        # （带源文件与行号），认不出的（比如"某个遭遇类在树里找不到唯一文件"）
        # 进 ``gaps.top_entries``，最后汇总成顶层缺口。两条路都不丢东西。
        gaps = ActGaps()
        record = parse_act(path, index, point_counts_source, rules, gaps)
        record["unsupported"] = gaps.entries
        top_gaps.extend(gaps.top_entries)
        records.append(record)
    unsupported.extend(top_gaps)
    # 排序：先按幕索引，再按 id —— ``ModelDb.Acts`` 的注释就要求这个顺序
    # （``ModelDb.cs:296``："callers depend on this list being sorted by act
    # index, then by default/non-default"；default 排在前面正是 id 顺序）。
    records.sort(key=lambda r: (r["index"] if r["index"] is not None else 99,
                                r["id"]))

    # 幕索引重复（同一 index 有两个候选）是**真机设计**，不是错误：
    # index 0 有 Overgrowth 与 Underdocks 两个。这里只统计，不报缺口。
    stats = {
        "acts": len(records),
        "acts_by_index": dict(sorted(collections.Counter(
            r["index"] for r in records).items())),
        "index_0_candidates": [r["id"] for r in records if r["index"] == 0],
        "per_act": {
            r["id"]: {
                "index": r["index"],
                "is_default": r["is_default"],
                "is_unlocked": r["is_unlocked"],
                "base_number_of_rooms": r["base_number_of_rooms"],
                "number_of_weak_encounters": r["number_of_weak_encounters"],
                "weak_encounters": len(r["weak_encounters"]),
                "regular_encounters": len(r["regular_encounters"]),
                "elite_encounters": len(r["elite_encounters"]),
                "boss_encounters": len(r["boss_encounters"]),
                "all_encounters": len(r["all_encounters"]),
                "ancients": len(r["ancients"]),
                "events": len(r["events"]),
                "discovery_order_swaps": len(r["discovery_order_swaps"]),
            }
            for r in records
        },
    }
    return records, unsupported, stats


def self_check(records: list[dict]) -> list[str]:
    """几条**机器可判定**的不变量。返回问题清单（空 = 通过）。

    ⚠️ 为什么工具要自带自检：抽取器的失败模式是**安静地少抽**或**安静地抽错**
    （``property_expr`` 取不到块体属性那一次，报出来的理由是"公式与预期不符"，
    把人引向完全错误的方向）。自检不依赖"看起来对不对"，而是回到源码验：

      * 每条 ``文件:行`` 引用的文件存在、行号在文件范围内；
      * 池子里的 id 互不重复，且**每个 id 都来自这一
        张幕自己的** ``all_encounters`` / ``events``；
      * ``weak + regular + elite + boss`` 恰好等于 ``all_encounters``；
      * ``boss_discovery_order`` 里的每个 boss 也在 ``boss_encounters`` 里；
      * 分类谓词确实抄到了 ``ActModel.cs`` 的四个属性。
    """
    problems: list[str] = []
    line_cache: dict[str, list[str]] = {}

    def check_site(act_id: str, field: str, value: object) -> None:
        if not isinstance(value, str) or ":" not in value:
            # ``None`` 是**允许**的（比如本幕没有某个可选项）；但字段名要能对上。
            if value is None:
                return
            problems.append(f"{act_id}.{field} 的出处不是 ``文件:行``：{value!r}")
            return
        path_text, _, line_text = value.rpartition(":")
        if not line_text.isdigit():
            # ``…/File.cs:177-192`` 这类区间出处是合法的说明性引用。
            if "-" in line_text and all(p.isdigit() for p in line_text.split("-")):
                return
            problems.append(f"{act_id}.{field} 的出处行号不是数字：{value!r}")
            return
        path = Path(path_text)
        if not path.exists():
            problems.append(f"{act_id}.{field} 指向不存在的文件：{value!r}")
            return
        if path_text not in line_cache:
            line_cache[path_text] = path.read_text(
                encoding="utf-8", errors="replace").split(NEWLINE)
        lines = line_cache[path_text]
        if not (1 <= int(line_text) <= len(lines)):
            problems.append(
                f"{act_id}.{field} 的行号 {line_text} 超出 {path_text}"
                f"（共 {len(lines)} 行）：{value!r}")

    for record in records:
        act_id = record["id"]
        for field in ("index_source", "is_default_source",
                      "base_number_of_rooms_source",
                      "number_of_weak_encounters_source", "is_unlocked_source",
                      "ancients_source", "events_source", "encounters_source",
                      "boss_discovery_order_source",
                      "discovery_order_swaps_source"):
            check_site(act_id, field, record.get(field))
        for field in ("rest_source", "unknown_source"):
            check_site(act_id, "map_point_counts." + field,
                       record["map_point_counts"].get(field))

        # 池子大小：四类相加必须等于总表。
        pools = [record["weak_encounters"], record["regular_encounters"],
                 record["elite_encounters"], record["boss_encounters"]]
        total = sum(len(pool) for pool in pools)
        if total != len(record["all_encounters"]):
            problems.append(
                f"{act_id}: 弱+普+精+Boss = {total}，但 all_encounters 有 "
                f"{len(record['all_encounters'])} 条（有遭遇没被分类）")
        flat = [cid for pool in pools for cid in pool]
        if len(set(flat)) != len(flat):
            problems.append(f"{act_id}: 遭遇池里有重复 id（同一场遭遇进了两个池）")
        ids = {entry["id"] for entry in record["all_encounters"]}
        for pool_name, pool in zip(
                ("weak_encounters", "regular_encounters", "elite_encounters",
                 "boss_encounters"), pools):
            stray = [cid for cid in pool if cid not in ids]
            if stray:
                problems.append(
                    f"{act_id}.{pool_name} 里有 all_encounters 之外的 id：{stray}")

        # 归一化后的池必须就是源码顺序（分类不许重排）。
        normal_from_source = [entry["id"] for entry in record["all_encounters"]
                              if entry["room_type"] == "Monster"]
        if record["normal_encounters"] != normal_from_source:
            problems.append(
                f"{act_id}.normal_encounters 与 GenerateAllEncounters 的"
                f"源码顺序不一致")

        # 弱遭遇的条数必须与 NumberOfWeakEncounters 相符（GrabBag 的池容量）。
        weak_count = record["number_of_weak_encounters"]
        if weak_count is None:
            problems.append(f"{act_id}: number_of_weak_encounters 抽不出")
        elif weak_count > 0 and not record["weak_encounters"]:
            problems.append(
                f"{act_id}: NumberOfWeakEncounters={weak_count} 但弱遭遇池为空"
                f"（GenerateRooms 会抽出 Nothing）")

        # Boss 的发现顺序必须是 Boss 池的子集（真机从 AllBossEncounters 里挑）。
        for boss in ([b["id"] for b in record["boss_discovery_order"]]):
            if boss not in record["boss_encounters"]:
                problems.append(
                    f"{act_id}.boss_discovery_order 里的 {boss!r} 不在 "
                    f"boss_encounters 里")
        if len(record["boss_discovery_order"]) != len(record["boss_encounters"]):
            problems.append(
                f"{act_id}: boss_discovery_order 有 "
                f"{len(record['boss_discovery_order'])} 个，boss_encounters 有 "
                f"{len(record['boss_encounters'])} 个（真机两者是同一个集合的"
                f"两种顺序）")

        # 事件 / 远古的 id 必须唯一。
        for field in ("events", "ancients"):
            got = [entry["id"] for entry in record[field]]
            if len(set(got)) != len(got):
                problems.append(f"{act_id}.{field} 有重复 id：{got}")

        # 分类谓词必须真的来自 ActModel.cs（不是硬编码在 Python 里的）。
        for key, rule in record["encounter_classification"].items():
            if not rule.get("predicate") or not rule.get("source"):
                problems.append(f"{act_id}.encounter_classification.{key} 没有谓词或出处")
            elif ACT_MODEL.name not in rule["source"]:
                problems.append(
                    f"{act_id}.encounter_classification.{key} 的出处不是 "
                    f"{ACT_MODEL.name}：{rule['source']}")

        # 地图点计数的三项常量必须抽到。
        counts = record["map_point_counts"]
        if not counts.get("rest"):
            problems.append(f"{act_id}: map_point_counts.rest 抽不出")
        if not (counts.get("unknown") or {}).get("base"):
            problems.append(f"{act_id}: map_point_counts.unknown.base 抽不出")
        if counts["elites"].get("value_default") is None:
            problems.append(f"{act_id}: map_point_counts.elites 抽不出")
        if counts["shops"].get("value") is None:
            problems.append(f"{act_id}: map_point_counts.shops 抽不出")

    # 幕的集合必须真的按 ModelDb.ActsByIndex 的规则选出来。
    index_0 = sorted(r["id"] for r in records if r["index"] == 0)
    if index_0 != ["overgrowth", "underdocks"]:
        problems.append(f"幕索引 0 的候选不是密林+暗港：{index_0}")
    index_1 = sorted(r["id"] for r in records if r["index"] == 1)
    if index_1 != ["hive"]:
        problems.append(f"幕索引 1 的候选不是蜂巢：{index_1}")
    index_2 = sorted(r["id"] for r in records if r["index"] == 2)
    if index_2 != ["glory"]:
        problems.append(f"幕索引 2 的候选不是荣耀：{index_2}")
    defaults = [r["id"] for r in sorted(records, key=lambda r: r["index"])
                if r["is_default"]]
    if defaults != ["overgrowth", "hive", "glory"]:
        problems.append(f"默认幕列表不是密林→蜂巢→荣耀：{defaults}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从反编译源码抽取幕（Act）数据")
    parser.add_argument("--src", default=str(DECOMPILED))
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--quiet", action="store_true")
    #: ``--self-check``：不信任"跑通了就是对的"，把几条**机器可判定**的
    #: 不变量再验一遍（每张幕的字段能回到源码某一行、池大小与声明数一致、
    #: 分类谓词确实来自 ActModel、4 张幕按 modeldb 的规则选出来）。
    parser.add_argument("--self-check", action="store_true",
                        help="抽完之后再验一遍不变量，失败退出码 2")
    args = parser.parse_args(argv)

    src_root = Path(args.src)
    if not src_root.exists():
        print(f"没有反编译源码目录：{src_root}")
        return 3

    # 缺口统一**排序 + 去重**：集合迭代顺序会随哈希种子变，直接塞进 JSON
    # 会让同一份源码产出不同字节（确定性验收会挂）。
    records, unsupported, stats = build(src_root)
    unsupported = sorted(set(unsupported))
    for record in records:
        record["unsupported"] = sorted(set(record["unsupported"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ``indent=1`` + ``ensure_ascii=False`` 与 ``extract_cards.py`` /
    # ``extract_events.py`` 完全一致：同一套 JSON 外壳与命名风格。
    out.write_text(json.dumps(records, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    print(f"扫描 {src_root.as_posix()}，抽取 {len(records)} 张幕")
    for record in records:
        print(f"  index={record['index']}  {record['id']:12s} "
              f"is_default={str(record['is_default']):5s} "
              f"is_unlocked={str(record['is_unlocked']):20s} "
              f"rooms={record['base_number_of_rooms']} "
              f"weak={record['number_of_weak_encounters']}  遭遇 "
              f"弱{len(record['weak_encounters'])}/"
              f"普{len(record['regular_encounters'])}/"
              f"精{len(record['elite_encounters'])}/"
              f"Boss{len(record['boss_encounters'])}"
              f"（共 {len(record['all_encounters'])}）"
              f"  远古 {len(record['ancients'])}  事件 {len(record['events'])}"
              f"  首局替换 {len(record['discovery_order_swaps'])}")
    print("\n地图点计数（逐字抄 GetMapPointTypes 的公式参数）：")
    for record in records:
        counts = record["map_point_counts"]
        rest = counts["rest"]
        unknown = counts["unknown"]
        base = (unknown or {}).get("base") or {}
        print(f"  {record['id']:12s} "
              f"rest={rest['raw'] if rest else None}"
              f"  unknown={base.get('raw')}"
              f" offset={(unknown or {}).get('offset')}"
              f" → [{(unknown or {}).get('min')},{(unknown or {}).get('max')}]"
              f"  elites={counts['elites'].get('value_default')}"
              f"（升腾 {counts['elites'].get('value_with_ascension')}）"
              f"  shops={counts['shops'].get('value')}"
              f"  ignore_rules={counts['point_types_that_ignore_rules']}")

    total_gaps = sum(len(r["unsupported"]) for r in records)
    print(f"\n幕记录内的 unsupported 共 {total_gaps} 条"
          f"（{len(records)} 张幕，逐条带源文件与行号）")
    if not args.quiet:
        for record in records:
            for entry in record["unsupported"]:
                print(f"    [{record['id']}] {entry}")
    if unsupported:
        print(f"\n顶层 unsupported（跨幕，不属于任何一张幕）共 {len(unsupported)} 条：")
        for entry in unsupported:
            print(f"    {entry}")
    print(f"\n写出 {out}（{len(records)} 张幕）")
    print(f"社区数据 {CODEX_NOT_USED} **未被读取**（铁律 1：源码是唯一真相）")

    if args.self_check:
        problems = self_check(records)
        if problems:
            print(f"\n❌ 自检发现 {len(problems)} 个问题：")
            for problem in problems:
                print(f"    {problem}")
            return 2
        print("\n✅ 自检通过：每张幕的每个字段都能回到源码里的那一行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
