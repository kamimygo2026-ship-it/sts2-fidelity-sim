"""幕（Act）定义。

真机一局走 **3 个幕索引**、共 **4 张幕地图**（``docs/12`` F08）：

===== ===================== ================================================
索引   地图                  说明
===== ===================== ================================================
0     ``Overgrowth``        密林，``IsDefault = true``
0     ``Underdocks``        暗港，第一章的另一张图，``IsDefault = false``（需解锁）
1     ``Hive``              蜂巢
2     ``Glory``             荣耀
===== ===================== ================================================

来源（唯一真相，全部来自 ``data/decompiled/sts2/``）：

  * ``MegaCrit.Sts2.Core.Models.ActModel``：``Index`` / ``IsDefault`` /
    ``BaseNumberOfRooms`` / ``NumberOfWeakEncounters`` / ``GetNumberOfRooms`` /
    ``GetMapPointTypes`` / ``GetRandomList``（按索引分组随机选备选幕）
  * ``MegaCrit.Sts2.Core.Models.Acts.{Overgrowth,Underdocks,Hive,Glory}``
  * ``MegaCrit.Sts2.Core.Map.MapPointTypeCounts``：精英/商店/未知/营火的配额
  * ``MegaCrit.Sts2.Core.Entities.Ascension.AscensionManager``：
    ``HasLevel(level) => _level >= (int)level``，所以 ``SwarmingElites`` 从 A1 起
    生效（精英 5 → ``round(5 * 1.6) = 8``），``DoubleBoss`` 从 A10 起生效

⚠️ **数据来源的优先级**：``data/content/repo/acts_source.json``（``tools/extract_acts.py``
机械抽取）**优先**；本文件里的 :data:`ACT_FALLBACK` 只是抽取物尚未生成时的兜底，
数值同样是逐字抄源码并标了行号。两者由 ``tests/test_acts.py`` 交叉校验，防止漂移。
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: ``AscensionLevel`` 的枚举值（``MegaCrit.Sts2.Core.Entities.Ascension.AscensionLevel``）。
ASCENSION_SWARMING_ELITES = 1
ASCENSION_DOUBLE_BOSS = 10

#: 真机的网格列数（``StandardActMap._mapWidth = 7``，``StandardActMap.cs:19``）。
GRID_COLUMNS = 7


@dataclass(frozen=True, slots=True)
class MapPointCounts:
    """``MapPointTypeCounts``（``MapPointTypeCounts.cs:12-46``）。

    ``NumOfElites`` / ``NumOfShops`` 是常量公式；``NumOfUnknowns`` / ``NumOfRests``
    由每张幕的 ``GetMapPointTypes(mapRng)`` 从**地图专用流**抽出 —— 所以它们必须在
    地图生成的第一步消费，晚一步整张图的随机数就错位了。
    """

    num_elites: int
    num_shops: int
    num_unknowns: int
    num_rests: int
    #: ``PointTypesThatIgnoreRules``：这些点类型不受放置规则约束（默认空）。
    ignore_rules: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ActDef:
    act_id: str
    csharp_class: str
    index: int
    is_default: bool
    base_number_of_rooms: int
    number_of_weak_encounters: int
    #: 营火配额的抽取方式，逐字抄 ``GetMapPointTypes``。
    #: ``("gaussian_int", mean, stddev, min, max)`` 或 ``("next_int", min, max)``。
    rest_spec: tuple
    #: 未知点配额在 ``StandardRandomUnknownCount`` 之上的偏移（Hive/Glory 是 -1）。
    unknown_offset: int
    #: ``("gaussian_int", 12, 1, 10, 14)``：``StandardRandomUnknownCount``。
    unknown_spec: tuple = ("gaussian_int", 12, 1, 10, 14)
    #: 解锁所需 Epoch 名（``IsUnlocked`` 的实现）；``None`` 表示恒解锁。
    unlock_epoch: str | None = None
    events: tuple[str, ...] = field(default_factory=tuple)
    ancients: tuple[str, ...] = field(default_factory=tuple)
    weak_encounters: tuple[str, ...] = field(default_factory=tuple)
    regular_encounters: tuple[str, ...] = field(default_factory=tuple)
    elite_encounters: tuple[str, ...] = field(default_factory=tuple)
    boss_encounters: tuple[str, ...] = field(default_factory=tuple)
    boss_discovery_order: tuple[str, ...] = field(default_factory=tuple)

    # ---- 房间数 ------------------------------------------------------
    def number_of_rooms(self, is_multiplayer: bool = False) -> int:
        """``GetNumberOfRooms``：**不含** Boss 房与远古房（``ActModel.cs:261-269``）。"""
        return self.base_number_of_rooms - (1 if is_multiplayer else 0)

    def number_of_floors(self, is_multiplayer: bool = False) -> int:
        """``GetNumberOfFloors`` = 房间数 + 2（Boss + 远古，``ActModel.cs:276-279``）。"""
        return self.number_of_rooms(is_multiplayer) + 2

    # ---- 点类型配额 --------------------------------------------------
    def map_point_counts(self, rng, ascension: int = 0) -> MapPointCounts:
        """按源码顺序从 ``mapRng`` 抽出本幕的点类型配额。

        ⚠️ **顺序是语义的一部分**：真机先算 ``restCount`` 再算 ``unknownCount``
        （例如 ``Overgrowth.cs:136-137``）。反过来会让地图与真机分叉。
        """
        kind = self.rest_spec[0]
        if kind == "gaussian_int":
            _, mean, stddev, low, high = self.rest_spec
            rests = rng.next_gaussian_int(mean, stddev, low, high)
        elif kind == "next_int":
            _, low, high = self.rest_spec
            rests = rng.next_int(low, high)          # 上界开区间
        else:                                        # pragma: no cover
            raise ValueError(f"未知的营火配额抽取方式：{self.rest_spec!r}")

        _, mean, stddev, low, high = self.unknown_spec
        unknowns = rng.next_gaussian_int(mean, stddev, low, high) + self.unknown_offset

        # ``MapPointTypeCounts.cs:14``：round(5 * (SwarmingElites ? 1.6 : 1))
        elites = round(5 * (1.6 if ascension >= ASCENSION_SWARMING_ELITES else 1.0))
        return MapPointCounts(num_elites=int(elites), num_shops=3,
                              num_unknowns=unknowns, num_rests=rests)


#: 抽取物尚未生成时的兜底表（数值逐字抄源码，行号见各条注释）。
ACT_FALLBACK: dict[str, ActDef] = {
    # Overgrowth.cs:47 BaseNumberOfRooms=15、:45 weak=3、:49 Index=0、:51 IsDefault=true
    # Overgrowth.cs:134-138 GetMapPointTypes：rest = NextGaussianInt(7,1,6,7)
    "overgrowth": ActDef(
        act_id="overgrowth", csharp_class="Overgrowth", index=0, is_default=True,
        base_number_of_rooms=15, number_of_weak_encounters=3,
        rest_spec=("gaussian_int", 7, 1, 6, 7), unknown_offset=0,
    ),
    # Underdocks.cs:42 rooms=15、:40 weak=3、:44 Index=0、:46 IsDefault=false
    # Underdocks.cs:107-110 IsUnlocked => unlockState.IsEpochRevealed<UnderdocksEpoch>()
    # Underdocks.cs:112-116 GetMapPointTypes：rest = NextGaussianInt(7,1,6,7)
    "underdocks": ActDef(
        act_id="underdocks", csharp_class="Underdocks", index=0, is_default=False,
        base_number_of_rooms=15, number_of_weak_encounters=3,
        rest_spec=("gaussian_int", 7, 1, 6, 7), unknown_offset=0,
        unlock_epoch="UnderdocksEpoch",
    ),
    # Hive.cs:47 rooms=14、:45 weak=2、:49 Index=1、:51 IsDefault=true
    # Hive.cs:117-121 GetMapPointTypes：rest = NextGaussianInt(6,1,6,7)、unknown - 1
    "hive": ActDef(
        act_id="hive", csharp_class="Hive", index=1, is_default=True,
        base_number_of_rooms=14, number_of_weak_encounters=2,
        rest_spec=("gaussian_int", 6, 1, 6, 7), unknown_offset=-1,
    ),
    # Glory.cs:43 rooms=13、:41 weak=2、:45 Index=2、:47 IsDefault=true
    # Glory.cs:106-110 GetMapPointTypes：rest = NextInt(5, 7)、unknown - 1
    "glory": ActDef(
        act_id="glory", csharp_class="Glory", index=2, is_default=True,
        base_number_of_rooms=13, number_of_weak_encounters=2,
        rest_spec=("next_int", 5, 7), unknown_offset=-1,
    ),
}


def act_def(act_id: str) -> ActDef:
    """取一张幕的定义（先查抽取物，再退回兜底表）。

    抽取物（``acts_source.json``）目前由 ``tools/extract_acts.py`` 产出；装载逻辑在
    ``content.ACTS``。抽取物里没有的字段用兜底表补齐，两边都缺就报错 —— 绝不猜。
    """
    extracted = _extracted_acts().get(act_id)
    fallback = ACT_FALLBACK.get(act_id)
    if extracted is None:
        if fallback is None:
            raise KeyError(f"未知的幕：{act_id!r}")
        return fallback
    return extracted if fallback is None else _merge(fallback, extracted)


def _merge(fallback: ActDef, extracted: ActDef) -> ActDef:
    """抽取物优先；抽取物为空的字段用兜底（例如遭遇池还没抽到的阶段）。"""
    data = {
        name: getattr(extracted, name) if getattr(extracted, name) else getattr(fallback, name)
        for name in ActDef.__dataclass_fields__
    }
    return ActDef(**data)


def _extracted_acts() -> dict[str, ActDef]:
    """``content.ACTS``（若有）。延迟导入，避免 ``mapgen`` ←→ ``content`` 循环。"""
    try:
        from . import content
    except Exception:                                # pragma: no cover
        return {}
    return getattr(content, "ACTS", {}) or {}


def acts_by_index() -> dict[int, tuple[ActDef, ...]]:
    """``ModelDb.ActsByIndex``：同一幕索引下的候选幕。

    真机 ``ActModel.GetRandomList``（``ActModel.cs:538-565``）在这上面随机：

      * 未解锁的候选**直接排除**（``IsUnlocked``）；
      * 抽取阶段若某张非默认幕**从未被发现过**，就一定先给它（保底体验），
        否则在候选里等概率抽；
      * ``GetDefaultList`` 取每个索引下 ``IsDefault`` 的那张。
    """
    table: dict[int, list[ActDef]] = {}
    for definition in ACT_FALLBACK.values():
        table.setdefault(definition.index, []).append(definition)
    return {index: tuple(items) for index, items in table.items()}


def default_act_list() -> tuple[ActDef, ...]:
    """``GetDefaultList``：每个索引取 ``IsDefault`` 的那张（密林 → 蜂巢 → 荣耀）。"""
    result: list[ActDef] = []
    for index in sorted(acts_by_index()):
        candidates = [d for d in acts_by_index()[index] if d.is_default]
        if not candidates:                           # pragma: no cover
            raise ValueError(f"幕索引 {index} 没有默认幕")
        result.append(candidates[0])
    return tuple(result)


def random_act_list(rng, unlocked_epochs: frozenset[str] = frozenset(),
                    is_multiplayer: bool = False,
                    discovered: frozenset[str] = frozenset()) -> tuple[ActDef, ...]:
    """``GetRandomList``：按索引逐个抽（密林/暗港二选一，再蜂巢、荣耀）。

    ⚠️ 真机的"发现保底"是**按存档**记账的（``SaveManager.Progress.DiscoveredActs``）；
    引擎没有存档层，所以把 ``discovered`` 显式当参数传进来（默认空 =
    把未发现的非默认幕当作"还没拿到过"，于是第一次一定给暗港 —— 与真机新档一致）。
    """
    result: list[ActDef] = []
    for index in sorted(acts_by_index()):
        pool = [d for d in acts_by_index()[index]
                if d.unlock_epoch is None or d.unlock_epoch in unlocked_epochs]
        if not pool:                                 # pragma: no cover
            raise ValueError(f"幕索引 {index} 没有已解锁的幕")
        forced = None
        if not is_multiplayer:
            for candidate in pool:
                if not candidate.is_default and candidate.act_id not in discovered:
                    forced = candidate
                    break
        result.append(forced if forced is not None else rng.next_item(pool))
    return tuple(result)
