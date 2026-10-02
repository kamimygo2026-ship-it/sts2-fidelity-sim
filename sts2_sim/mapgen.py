"""地图生成：忠实复刻 ``StandardActMap``（含剪枝与后处理）。

来源（唯一真相，全部位于 ``data/decompiled/sts2/``）：

  * ``MegaCrit.Sts2.Core.Map.StandardActMap`` —— 7 列网格、路径生成、
    点类型分配、剪枝、后处理（本文件的骨架）
  * ``MegaCrit.Sts2.Core.Map.MapPoint`` / ``ActMap`` —— 点、父子关系、网格访问
  * ``MegaCrit.Sts2.Core.Map.MapPointType`` —— 点类型枚举（**含** ``Unknown`` 与
    ``Ancient``）
  * ``MegaCrit.Sts2.Core.Map.MapPathPruning`` —— 重复线段剪枝 + 配额修复
  * ``MegaCrit.Sts2.Core.Map.MapPostProcessing`` —— ``CenterGrid`` /
    ``SpreadAdjacentMapPoints`` / ``StraightenPaths``
  * ``MegaCrit.Sts2.Core.Models.ActModel`` —— ``GetNumberOfRooms`` / 点类型配额
  * ``MegaCrit.Sts2.Core.Odds.UnknownMapPointOdds`` —— 未知点掷房间（在 ``run.py``）

**信息边界**（``docs/01`` §1.2）：地图的**结构**（有哪些点、什么类型、怎么连）是
**L1 公开** —— 真机整幕地图一开始就全画出来。点的**内容**（哪只怪、哪个事件、
什么掉落）是 **L3 隐藏**，放在 ``run.RunHidden.node_contents``，入口是
``RunEnv._enter_node()``。

⚠️ **本文件更正的一处早期建模错误**：旧版把地图点类型写成 ``event``，也就是把
"未知点"直接当成了事件房。真机的类型是 ``Unknown``，**进房间那一刻**才由
``UnknownMapPointOdds.Roll`` 掷成 事件/怪/精英/宝箱/商店；而且掷出来的类型会
**改变后续概率**（越掷不出事件，事件概率越高 —— ``AbstractOdds``）。
旧建模会让玩家提前知道"这里是事件"，凭空多出一层信息。

⚠️ **保真度边界（不许含糊）**：

  1. 真机 PRNG 是 ``MegaRandom``（实现未反编译），引擎用 Mersenne Twister
     （``rng.Rng``）。所以**同 seed 不会得到与真机逐点相同的地图**；本文件复刻的是
     **算法、判定条件与 RNG 消费顺序**。
  2. 两处**已知近似**（真机依赖 ``HashSet`` 枚举顺序，无法从反编译结果还原）：
     行内 ``SpreadAdjacentMapPoints`` 的并列候选、剪枝后 ``Children`` 的遍历顺序。
     都只在对称/并列情形下影响观感，不影响"能不能走通"与点类型配额。
  3. 本文件只到「可执行 / 闭包可执行」两层，**真机对拍 = 0**（``docs/13`` §7）。
     不得用本文件声称地图与真机一致。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .acts import (
    ASCENSION_DOUBLE_BOSS,
    GRID_COLUMNS,
    ActDef,
    MapPointCounts,
    act_def,
)
from .rng import MAP_POINT_TYPE_ORDER, Rng, RngSet

# ==========================================================================
# 点类型（与 ``MapPointType`` 枚举同名同序）
# ==========================================================================
UNASSIGNED = "unassigned"
UNKNOWN = "unknown"
SHOP = "shop"
TREASURE = "treasure"
REST_SITE = "rest_site"
MONSTER = "monster"
ELITE = "elite"
BOSS = "boss"
ANCIENT = "ancient"

#: 旧名兼容：``EVENT`` 现在语义正确了（未知点，不是事件房）。
EVENT = UNKNOWN
REST = REST_SITE
MERCHANT = SHOP

#: ``StandardActMap._lowerMapPointRestrictions``：不能出现在地图**下半段**（``StandardActMap.cs:27-31``）。
_LOWER_RESTRICTIONS = frozenset({REST_SITE, ELITE})
#: ``_upperMapPointRestrictions``：不能出现在地图**上半段**（``StandardActMap.cs:33``）。
_UPPER_RESTRICTIONS = frozenset({REST_SITE})
#: ``_parentMapPointRestrictions``：不能与相邻（父或子）同类型（``StandardActMap.cs:36-42``）。
_PARENT_RESTRICTIONS = frozenset({ELITE, REST_SITE, TREASURE, SHOP})
#: ``_childMapPointRestrictions``：不能与子节点同类型（``StandardActMap.cs:45-51``）。
_CHILD_RESTRICTIONS = frozenset({ELITE, REST_SITE, TREASURE, SHOP})
#: ``_siblingPointTypeRestrictions``：只有这些类型允许做兄弟（``StandardActMap.cs:54-61``）。
_SIBLING_RESTRICTIONS = frozenset({REST_SITE, MONSTER, UNKNOWN, ELITE, SHOP})

#: ``int.MaxValue``： ``MapPostProcessing.ComputeGap`` 在没有同行邻居时返回它。
_INT_MAX = 2 ** 31 - 1


@dataclass(frozen=True, slots=True)
class MapNode:
    """地图上的一个点。``edges`` 是**通向下一行**的子节点 id。"""

    node_id: int
    row: int
    col: int
    kind: str
    edges: tuple[int, ...]

    @property
    def is_boss(self) -> bool:
        return self.kind == BOSS


@dataclass(frozen=True, slots=True)
class MapGraph:
    """一幕地图的结构（L1：可以整体交给观测）。"""

    nodes: tuple[MapNode, ...]
    #: 总层数 = ``ActModel.GetNumberOfFloors``（含远古层与 Boss 层）。
    rows: int
    columns: int
    #: 站在远古点时可以走的第一批节点（真机 ``StartingMapPoint`` 的子节点）。
    start: tuple[int, ...]
    #: 起点（``MapPointType.Ancient``）的节点 id。
    ancient: int
    boss: int
    act_id: str
    act_index: int
    #: 双 Boss 模式（A10 ``DoubleBoss``）下的第二个 Boss；否则 ``None``。
    second_boss: int | None = None

    def by_id(self, node_id: int) -> MapNode:
        return self.nodes[node_id]

    def row_nodes(self, row: int) -> tuple[int, ...]:
        return tuple(node.node_id for node in self.nodes if node.row == row)

    def __len__(self) -> int:
        return len(self.nodes)


# ==========================================================================
# 生成期可变点（对应真机的 ``MapPoint``）
# ==========================================================================
class _Point:
    """可变地图点。

    ⚠️ 相等性是**引用相等**：真机 ``MapPoint`` 只有一个没 override 的
    ``private bool Equals(MapPoint)``（``MapPoint.cs:32-35``），所以 ``HashSet<MapPoint>``
    与 ``List.Contains`` 实际走的是默认引用语义。这里一律用 ``is`` 比较，
    与真机一致（换成按坐标比较会让"同一个坐标上的两个对象"被误判为同一个点，
    而剪枝恰好会制造这种情形）。
    """

    __slots__ = ("uid", "col", "row", "kind", "can_be_modified", "_children", "_parents")

    #: 进程内自增序号。只用于剪枝时的集合运算（真机用对象引用本身），
    #: 构造顺序是确定性的，所以它也是确定性的。
    _next_uid = 0

    def __init__(self, col: int, row: int) -> None:
        _Point._next_uid += 1
        self.uid = _Point._next_uid
        self.col = col
        self.row = row
        self.kind = UNASSIGNED
        #: ``CanBeModified``：固定行（怪/宝箱/营火）与 Boss/远古点不允许被改写。
        self.can_be_modified = True
        self._children: list["_Point"] = []
        self._parents: list["_Point"] = []

    @property
    def coord(self) -> tuple[int, int]:
        return (self.col, self.row)

    @property
    def children(self) -> tuple["_Point", ...]:
        return tuple(self._children)

    @property
    def parents(self) -> tuple["_Point", ...]:
        return tuple(self._parents)

    @property
    def child_count(self) -> int:
        """``Children.Count``。热门路径直接读计数，别去构造元组。"""
        return len(self._children)

    @property
    def parent_count(self) -> int:
        """``parents.Count``。"""
        return len(self._parents)

    def add_child(self, child: "_Point") -> None:
        """``MapPoint.AddChildPoint``：集合语义，重复添加是空操作。"""
        if any(existing is child for existing in self._children):
            return
        self._children.append(child)
        child._parents.append(self)

    def remove_child(self, child: "_Point") -> None:
        """``MapPoint.RemoveChildPoint``：不存在时是空操作。"""
        self._children = [item for item in self._children if item is not child]
        child._parents = [item for item in child._parents if item is not self]


def _point_sort_key(point: _Point) -> tuple[int, int]:
    """``MapPoint.CompareTo`` → ``(col, row)``（``MapCoord.cs:70-73``）。"""
    return (point.col, point.row)


# ==========================================================================
# 生成器
# ==========================================================================
class _Builder:
    """``StandardActMap`` 的生成态。

    坐标语义（与真机一致，别改动）：

      * 远古点固定在 ``(3, 0)``，**不在**网格数组里（``ActMap.GetPoint`` 特判）；
      * 网格是 ``7 × _map_length``，但第 0 行永远是空的；路径占据第 1..
        ``_map_length - 1`` 行；
      * Boss 在 ``(3, _map_length)`` —— **在网格之外**（``GetRowCount()`` 那一行）。
    """

    def __init__(self, rng: Rng, act: ActDef, counts: MapPointCounts, *,
                 is_multiplayer: bool = False,
                 replace_treasure_with_elites: bool = False,
                 has_second_boss: bool = False) -> None:
        self._rng = rng
        self._act = act
        self._counts = counts
        self._is_multiplayer = is_multiplayer
        self._rooms = act.number_of_rooms(is_multiplayer)
        #: ``_mapLength = GetNumberOfRooms(isMultiplayer) + 1``（``StandardActMap.cs:89``）。
        self._map_length = self._rooms + 1
        self._replace_treasure = replace_treasure_with_elites
        self._grid: dict[tuple[int, int], _Point] = {}
        self._start_points: list[_Point] = []
        self.boss = _Point(GRID_COLUMNS // 2, self._map_length)
        self.boss.can_be_modified = False
        self.ancient = _Point(GRID_COLUMNS // 2, 0)
        self.ancient.can_be_modified = False
        self.second_boss: _Point | None = (
            _Point(GRID_COLUMNS // 2, self._map_length + 1) if has_second_boss else None)
        if self.second_boss is not None:
            self.second_boss.can_be_modified = False

    # ---- 网格访问（``ActMap``） ---------------------------------------
    def get_point(self, col: int, row: int) -> _Point | None:
        """``ActMap.GetPoint``：Boss/双 Boss/远古特判优先，再查网格。"""
        if (col, row) == self.boss.coord:
            return self.boss
        if self.second_boss is not None and (col, row) == self.second_boss.coord:
            return self.second_boss
        if (col, row) == self.ancient.coord:
            return self.ancient
        if 0 <= col < GRID_COLUMNS and 0 <= row < self._map_length:
            return self._grid.get((col, row))
        return None

    def _get_or_create(self, col: int, row: int) -> _Point:
        point = self.get_point(col, row)
        if point is not None:
            return point
        point = _Point(col, row)
        self._grid[(col, row)] = point
        return point

    def all_points(self):
        """``ActMap.GetAllMapPoints``：**列优先**（``c`` 外层、``r`` 内层）。

        这个顺序是语义的一部分：它决定了两处 ``StableShuffle`` 的输入顺序，
        而输入顺序在"排序键相同"时才影响结果 —— 但真机就是这个顺序，照抄。
        注意：**不含**远古点与 Boss 点（它们不在网格数组里）。
        """
        for col in range(GRID_COLUMNS):
            for row in range(self._map_length):
                point = self._grid.get((col, row))
                if point is not None:
                    yield point

    def row_points(self, row: int) -> list[_Point]:
        """``ForEachInRow(Grid, row, ...)``：按列升序，只遍历网格。"""
        return [self._grid[(col, row)] for col in range(GRID_COLUMNS)
                if (col, row) in self._grid]

    # ---- 路径生成 -----------------------------------------------------
    def generate(self) -> None:
        """``GenerateMap``（``StandardActMap.cs:207-234``）：7 条路径 → Boss → 远古。"""
        for index in range(GRID_COLUMNS):
            point = self._get_or_create(self._rng.next_int(GRID_COLUMNS), 1)
            if index == 1:
                # 真机只对**第二条**路径去重（``StandardActMap.cs:212-218``）：
                # 保证至少有两个不同起点。这个"只查一次"的行为照抄。
                while any(existing is point for existing in self._start_points):
                    point = self._get_or_create(self._rng.next_int(GRID_COLUMNS), 1)
            if not any(existing is point for existing in self._start_points):
                self._start_points.append(point)
            self._path_generate(point)

        for point in self.row_points(self._map_length - 1):
            point.add_child(self.boss)
        if self.second_boss is not None:
            self.boss.add_child(self.second_boss)
        for point in self.row_points(1):
            self.ancient.add_child(point)

    def _path_generate(self, start: _Point) -> None:
        point = start
        while point.row < self._map_length - 1:
            coord = self._next_coord(point)
            child = self._get_or_create(*coord)
            point.add_child(child)
            point = child

    def _next_coord(self, current: _Point) -> tuple[int, int]:
        """``GenerateNextCoord``：三个方向**先洗牌**再挑第一个合法的。"""
        col = current.col
        low = max(0, col - 1)
        high = min(col + 1, GRID_COLUMNS - 1)
        options = [-1, 0, 1]
        self._rng.shuffle(options)                    # UnstableShuffle
        target_for = {-1: low, 0: col, 1: high}
        for delta in options:
            target = target_for[delta]
            if not self._has_invalid_crossover(current, target):
                return (target, current.row + 1)
        raise RuntimeError("Cannot find next node")   # 真机同（理论上到不了）

    def _has_invalid_crossover(self, current: _Point, target_col: int) -> bool:
        """``HasInvalidCrossover``：禁止两条路径"交叉"（``StandardActMap.cs:184-205``）。"""
        delta = target_col - current.col
        if delta in (0, GRID_COLUMNS):
            return False
        same_row = self._grid.get((target_col, current.row))
        if same_row is None:
            return False
        for child in same_row.children:
            if child.col - same_row.col == -delta:
                return True
        return False

    # ---- 点类型分配 ---------------------------------------------------
    def assign_point_types(self) -> None:
        """``AssignPointTypes``（``StandardActMap.cs:248-307``）。"""
        for point in self.row_points(self._map_length - 1):
            point.kind = REST_SITE
            point.can_be_modified = False
        treasure_kind = ELITE if self._replace_treasure else TREASURE
        for point in self.row_points(self._map_length - 7):
            point.kind = treasure_kind
            point.can_be_modified = False
        for point in self.row_points(1):
            point.kind = MONSTER
            point.can_be_modified = False

        queue: deque[str] = deque()
        for _ in range(self._counts.num_rests):
            queue.append(REST_SITE)
        for _ in range(self._counts.num_shops):
            queue.append(SHOP)
        for _ in range(self._counts.num_elites):
            queue.append(ELITE)
        for _ in range(self._counts.num_unknowns):
            queue.append(UNKNOWN)
        self._assign_remaining(queue)

        for point in self.all_points():
            if point.kind == UNASSIGNED:
                point.kind = MONSTER
        self.boss.kind = BOSS
        self.ancient.kind = ANCIENT
        if self.second_boss is not None:
            self.second_boss.kind = BOSS

    def _assign_remaining(self, queue: deque[str]) -> None:
        """``AssignRemainingTypesToRandomPoints``：最多 3 轮，每轮重洗未定点。"""
        for _ in range(3):
            if not queue:
                break
            pending = [point for point in self.all_points() if point.kind == UNASSIGNED]
            self._rng.stable_shuffle(pending, key=_point_sort_key)
            for point in pending:
                if not queue:
                    break
                point.kind = self._next_valid_type(queue, point)

    def _next_valid_type(self, queue: deque[str], point: _Point) -> str:
        """``GetNextValidPointType``：转圈找一个放得下的类型，找不到就留 Unassigned。"""
        for _ in range(len(queue)):
            kind = queue.popleft()
            if kind in self._counts.ignore_rules or self._is_valid(kind, point):
                return kind
            queue.append(kind)
        return UNASSIGNED

    def _is_valid(self, kind: str, point: _Point) -> bool:
        """``IsValidPointType``：五条规则全过才合法（``StandardActMap.cs:405-485``）。"""
        return (self._valid_lower(kind, point)
                and self._valid_upper(kind, point)
                and self._valid_with_relatives(kind, point)
                and self._valid_with_children(kind, point)
                and self._valid_with_siblings(kind, point))

    @staticmethod
    def _valid_lower(kind: str, point: _Point) -> bool:
        return not (point.row < 6 and kind in _LOWER_RESTRICTIONS)

    def _valid_upper(self, kind: str, point: _Point) -> bool:
        return not (point.row >= self._map_length - 3 and kind in _UPPER_RESTRICTIONS)

    @staticmethod
    def _valid_with_relatives(kind: str, point: _Point) -> bool:
        """``IsValidWithParents``（名字有误导性：它把**父与子**一起看）。

        ``StandardActMap.cs:451-458``：``mapPoint.parents.Concat(mapPoint.Children)``。
        """
        if kind in _PARENT_RESTRICTIONS:
            return not any(other.kind == kind
                           for other in (*point.parents, *point.children))
        return True

    @staticmethod
    def _valid_with_children(kind: str, point: _Point) -> bool:
        if kind in _CHILD_RESTRICTIONS:
            return not any(child.kind == kind for child in point.children)
        return True

    @staticmethod
    def _valid_with_siblings(kind: str, point: _Point) -> bool:
        if kind in _SIBLING_RESTRICTIONS:
            siblings = [other for parent in point.parents
                        for other in parent.children if other is not point]
            return not any(sibling.kind == kind for sibling in siblings)
        return True


# ==========================================================================
# 剪枝与修复（``MapPathPruning``）
# ==========================================================================
def _prune_and_repair(builder: _Builder, counts: MapPointCounts, rng: Rng) -> None:
    """``PruneAndRepair``：剪重复线段 → 补配额，最多 3 轮（``MapPathPruning.cs:22-32``）。"""
    for _ in range(3):
        _prune_duplicate_segments(builder, rng)
        if not _repair_pruned_point_types(builder, counts, rng):
            break


def _prune_duplicate_segments(builder: _Builder, rng: Rng) -> None:
    """``PruneDuplicateSegments``：剪到没有重复线段为止（上限 50 轮保护）。"""
    iterations = 0
    matching = _find_matching_segments(builder)
    while _prune_paths(builder, matching, rng):
        iterations += 1
        if iterations > 50:
            raise RuntimeError(
                f"Unable to prune matching segments in {iterations} iterations")
        matching = _find_matching_segments(builder)


def _find_all_paths(point: _Point) -> list[list[_Point]]:
    """``FindAllPaths``：从远古点走到 Boss 的所有路径（含首尾）。

    实现用**后缀记忆化**：同一个子节点的后缀只算一次，再各自接上前缀。
    结果与逐个递归完全一致（顺序也一致），但少建大量重复列表 —— 密集地图上
    ``FindAllPaths`` 会被调用十几次，这里是主要耗时点之一。
    """
    memo: dict[int, list[list[_Point]]] = {}

    def suffixes(node: _Point) -> list[list[_Point]]:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.kind == BOSS:
            result: list[list[_Point]] = [[node]]
        else:
            result = []
            for child in node._children:
                for tail in suffixes(child):
                    result.append([node, *tail])
        memo[id(node)] = result
        return result

    return suffixes(point)


def _find_matching_segments(builder: _Builder) -> list[list[list[_Point]]]:
    """``FindMatchingSegments``：返回"同起讫、同类型序列"的重复线段组。"""
    segments: dict[str, list[list[_Point]]] = {}
    #: ``key`` → 已收录线段的**内部点 uid 集合**。真机的去重判据
    #: ``OverlappingSegment``（``MapPathPruning.cs:233-247``）是"两个线段在
    #: 1..len-2 上有一个点是同一个对象"，而同一 key 下所有线段长度相同，
    #: 所以"与**任意**已收录线段重叠" ⇔ "内部点与已有内部点集相交"。
    #: 这样判等是精确等价的，且把 O(组内线段数) 的扫描降成一次集合求交。
    interiors: dict[str, set[int]] = {}
    type_strs = {point.uid: str(MAP_POINT_TYPE_ORDER[point.kind])
                 for point in _all_nodes(builder)}
    for path in _find_all_paths(builder.ancient):
        _add_segments(path, segments, interiors, type_strs)
    # 真机是 SortedDictionary<string, ...>(StringComparer.Ordinal)：**按键排序**。
    # 组的先后会决定 rng.shuffle 的分配，所以这个排序必须照抄。
    return [items for _key, items in sorted(segments.items()) if len(items) > 1]


def _all_nodes(builder: _Builder) -> list[_Point]:
    """网格点 + 远古点 + Boss（点类型的字符串表要覆盖所有会进线段的位置）。"""
    nodes = [builder.ancient]
    nodes.extend(builder.all_points())
    nodes.append(builder.boss)
    if builder.second_boss is not None:
        nodes.append(builder.second_boss)
    return nodes


def _add_segments(path: list[_Point], segments: dict[str, list[list[_Point]]],
                  interiors: dict[str, set[int]],
                  type_strs: dict[int, str]) -> None:
    """把一条路径上的所有合法线段登记进 ``segments``。

    这里为了速度做了一件等价改写：真机是"每条线段各自拼一次 key 字符串"
    （``GenerateSegmentKey`` 里 ``string.Join(",", 类型枚举值)``）。本实现先把
    整条路径的类型值拼成 ``joined``，再按偏移量**切片**取子串 —— 拼出来的字符串
    逐字符相同（枚举值都是个位数、逗号分隔，切出来的正好是同样的序列），
    但省掉了 38 万次 ``join``。
    """
    total = len(path)
    uids = [point.uid for point in path]
    types = [type_strs[uid] for uid in uids]
    joined = ",".join(types)
    #: ``offsets[k]`` = 第 k 个元素在 ``joined`` 里的起点；``+ len(types[k])`` 得到终点。
    offsets = [0] * (total + 1)
    for index in range(total):
        offsets[index + 1] = offsets[index] + len(types[index]) + 1
    for i in range(total - 1):
        start = path[i]
        # ``IsValidSegmentStartMapPoint``：子节点 ≤1 时只有第 0 行算合法起点。
        if start.child_count <= 1 and start.row != 0:
            continue
        first_row = start.row
        head = (f"0-" if first_row == 0
                else f"{start.col},{first_row}-")
        for j in range(2, total - i):
            e = i + j
            end = path[e]
            if end.parent_count < 2:              # ``IsValidSegmentEndMapPoint``
                continue
            key = f"{head}{end.col},{end.row}-{joined[offsets[i]:offsets[e + 1]]}"
            existing = segments.get(key)
            if existing is None:
                segments[key] = [path[i:e + 1]]
                interiors[key] = set(uids[i + 1:e])
                continue
            inner = uids[i + 1:e]
            if interiors[key].isdisjoint(inner):
                existing.append(path[i:e + 1])
                interiors[key].update(inner)


def _prune_paths(builder: _Builder, matching: list[list[list[_Point]]],
                 rng: Rng) -> bool:
    for segment_list in matching:
        rng.shuffle(segment_list)                     # UnstableShuffle
        if _prune_all_but_last(builder, segment_list) != 0:
            return True
        if _break_relationship(segment_list):
            return True
    return False


def _prune_all_but_last(builder: _Builder, matches: list[list[_Point]]) -> int:
    removed = 0
    for index, segment in enumerate(matches):
        if index == len(matches) - 1:                 # 最后一组留着（真机语义）
            return removed
        if _prune_segment(builder, segment):
            removed += 1
    return removed


def _prune_segment(builder: _Builder, segment: list[_Point]) -> bool:
    result = False
    for i in range(len(segment) - 1):
        point = segment[i]
        if not _is_in_map(builder, point):
            return True
        if (point.child_count > 1 or point.parent_count > 1
                or any(parent.child_count == 1 and not _is_removed(builder, parent)
                       for parent in point._parents)):
            continue
        tail = segment[i:]
        if not any(node.child_count > 1 and node.parent_count == 1 for node in tail):
            if segment[-1].parent_count == 1:
                return False
            if not any(child.parent_count == 1
                       for child in point._children if child not in segment):
                _remove_point(builder, point)
                result = True
    return result


def _remove_point(builder: _Builder, point: _Point) -> None:
    """``MapPathPruning.RemovePoint``：摘掉点并断开所有父子边。"""
    builder._grid.pop(point.coord, None)
    builder._start_points = [item for item in builder._start_points if item is not point]
    for child in list(point.children):
        point.remove_child(child)
    for parent in list(point.parents):
        parent.remove_child(point)


def _break_relationship(matches: list[list[_Point]]) -> bool:
    for segment in matches:
        if _break_in_segment(segment):
            return True
    return False


def _break_in_segment(segment: list[_Point]) -> bool:
    result = False
    for i in range(len(segment) - 1):
        point = segment[i]
        if point.child_count >= 2:
            following = segment[i + 1]
            if following.parent_count != 1:
                point.remove_child(following)
                result = True
    return result


def _is_in_map(builder: _Builder, point: _Point) -> bool:
    """``MapPathPruning.IsInMap``：网格里没有且不是远古点 → 只有 Boss 算在图上。"""
    if builder._grid.get(point.coord) is None and point.kind != ANCIENT:
        return point.kind == BOSS
    return True


def _is_removed(builder: _Builder, point: _Point) -> bool:
    return builder._grid.get(point.coord) is None


def _repair_pruned_point_types(builder: _Builder, counts: MapPointCounts,
                               rng: Rng) -> bool:
    """``RepairPrunedPointTypes``：把缺口补回成怪点（顺序：商店→精英→营火→未知）。"""
    repaired = False
    repaired |= _repair_point_type(builder, SHOP, counts.num_shops, rng)
    repaired |= _repair_point_type(builder, ELITE, counts.num_elites, rng)
    repaired |= _repair_point_type(builder, REST_SITE, counts.num_rests, rng)
    return repaired | _repair_point_type(builder, UNKNOWN, counts.num_unknowns, rng)


def _repair_point_type(builder: _Builder, kind: str, target: int, rng: Rng) -> bool:
    current = sum(1 for point in builder.all_points() if point.kind == kind)
    deficit = target - current
    if deficit <= 0:
        return False
    result = False
    candidates = [point for point in builder.all_points()
                  if point.kind == MONSTER and point.can_be_modified]
    rng.stable_shuffle(candidates, key=_point_sort_key)
    for point in candidates:
        if deficit == 0:
            break
        if builder._is_valid(kind, point):
            point.kind = kind
            deficit -= 1
            result = True
    return result


# ==========================================================================
# 后处理（``MapPostProcessing``）
# ==========================================================================
def _is_column_empty(builder: _Builder, col: int) -> bool:
    return all((col, row) not in builder._grid for row in range(builder._map_length))


def _center_grid(builder: _Builder) -> None:
    """``CenterGrid``：左右各空两列时整体挪一列。

    ⚠️ 真机只挪**网格数组**里的点，远古点与 Boss 点不在数组里、因此**不挪**
    （``MapPostProcessing.cs:15-27``）。于是挪完之后远古点的列可能与第一行不齐 ——
    这是真机自带的行为，照抄，不要"顺手修好"。
    """
    left_empty = _is_column_empty(builder, 0) and _is_column_empty(builder, 1)
    right_empty = (_is_column_empty(builder, GRID_COLUMNS - 1)
                   and _is_column_empty(builder, GRID_COLUMNS - 2))
    if left_empty and not right_empty:
        shift = -1
    elif not left_empty and right_empty:
        shift = 1
    else:
        return
    for row in range(builder._map_length):
        columns = (range(GRID_COLUMNS - 1, -1, -1) if shift > 0
                   else range(GRID_COLUMNS))
        for col in columns:
            point = builder._grid.pop((col, row), None)
            new_col = col + shift
            if not 0 <= new_col < GRID_COLUMNS:
                continue                              # 挤出边界 = 丢掉（真机同）
            builder._grid.pop((new_col, row), None)   # 真机无条件写入该格
            if point is not None:
                point.col = new_col
                builder._grid[(new_col, row)] = point


def _neighbour_columns(col: int) -> set[int]:
    return {candidate for candidate in (col - 1, col, col + 1)
            if 0 <= candidate < GRID_COLUMNS}


def _allowed_columns(point: _Point) -> list[int]:
    """``GetAllowedPositions``：父与子的"同列或左右一列"取交集。

    ⚠️ 真机遍历 ``HashSet<int>``，并列时的先后不可复刻；这里**按列升序**枚举，
    并已在模块 docstring 里登记为已知近似。
    """
    allowed = set(range(GRID_COLUMNS))
    for other in (*point.parents, *point.children):
        allowed &= _neighbour_columns(other.col)
    return sorted(allowed)


def _compute_gap(candidate_col: int, row_nodes: list[_Point],
                 current: _Point) -> int:
    best = _INT_MAX
    for other in row_nodes:
        if other is not current:
            best = min(best, abs(candidate_col - other.col))
    return best


def _spread_adjacent_points(builder: _Builder) -> None:
    """``SpreadAdjacentMapPoints``：同行挤在一起的点尽量拉开（``MapPostProcessing.cs:158``）。"""
    for row in range(builder._map_length):
        nodes = [builder._grid[(col, row)] for col in range(GRID_COLUMNS)
                 if (col, row) in builder._grid]
        moved = True
        while moved:
            moved = False
            for node in nodes:
                col = node.col
                best_col, best_gap = col, _compute_gap(col, nodes, node)
                for candidate in _allowed_columns(node):
                    if candidate == col:
                        continue
                    occupant = builder._grid.get((candidate, row))
                    if occupant is not None and occupant is not node:
                        continue
                    candidate_gap = _compute_gap(candidate, nodes, node)
                    if candidate_gap > best_gap:
                        best_col, best_gap = candidate, candidate_gap
                if best_col != col:
                    builder._grid.pop((col, row), None)
                    node.col = best_col
                    builder._grid[(best_col, row)] = node
                    moved = True


def _straighten_paths(builder: _Builder) -> None:
    """``StraightenPaths``：把"拐出去又拐回来"的单线节点拉直（``MapPostProcessing.cs:74``）。"""
    for row in range(builder._map_length):
        for col in range(GRID_COLUMNS):
            point = builder._grid.get((col, row))
            if point is None or len(point.parents) != 1 or len(point.children) != 1:
                continue
            parent = point.parents[0]
            child = point.children[0]
            spikes_left = point.col < child.col and point.col < parent.col
            spikes_right = point.col > child.col and point.col > parent.col
            if spikes_left and col < GRID_COLUMNS - 1:
                target = col + 1
                if (target, row) in builder._grid:
                    continue
                point.col = target
                builder._grid.pop((col, row), None)
                builder._grid[(target, row)] = point
            if spikes_right and col > 0:
                target = col - 1
                if (target, row) in builder._grid:
                    continue
                point.col = target
                builder._grid.pop((col, row), None)
                builder._grid[(target, row)] = point


# ==========================================================================
# 对外入口
# ==========================================================================
#: 生成结果缓存。地图是 ``(seed, 幕, 进阶, 人数)`` 的纯函数，而生成一张图要跑
#: 真机的剪枝+后处理（密集 seed 上可达几百毫秒）。训练里每个 episode 都会
#: ``reset``，同一个 seed 反复生成没有意义 —— 缓存让"同一局重开"免费。
#: :class:`MapGraph` 是冻结 dataclass（字段全是 tuple），共享安全。
_MAP_CACHE: dict[tuple, MapGraph] = {}
_MAP_CACHE_LIMIT = 512


def clear_map_cache() -> None:
    """清空地图缓存（测试用：验证"生成确实是纯函数"时必须绕开缓存）。"""
    _MAP_CACHE.clear()


def map_cache_size() -> int:
    return len(_MAP_CACHE)


def generate_map(seed: int, act: str | ActDef = "overgrowth", *,
                 act_index: int | None = None,
                 is_multiplayer: bool = False,
                 ascension: int = 0,
                 has_second_boss: bool | None = None,
                 replace_treasure_with_elites: bool = False,
                 point_counts: MapPointCounts | None = None) -> MapGraph:
    """生成一幕地图。

    :param seed: run 主种子。地图走**独立派生流** ``act_{index+1}_map``
        （``StandardActMap.cs:113``），不消费 ``up_front`` —— 地图长什么样与
        开局掷出的内容无关。
    :param act: 幕 id（``"overgrowth"`` / ``"underdocks"`` / ``"hive"`` / ``"glory"``）
        或 :class:`sts2_sim.acts.ActDef`。
    :param ascension: 进阶等级。A1 起 ``SwarmingElites`` 生效（精英 5 → 8），
        A10 起 ``DoubleBoss`` 生效（多一个 Boss 层）。
    """
    definition = act if isinstance(act, ActDef) else act_def(act)
    stream_index = definition.index if act_index is None else act_index
    if has_second_boss is None:
        has_second_boss = ascension >= ASCENSION_DOUBLE_BOSS
    cache_key = (seed, definition.act_id, stream_index, is_multiplayer, ascension,
                 has_second_boss, replace_treasure_with_elites,
                 None if point_counts is None else repr(point_counts))
    cached = _MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rng = RngSet(seed).act_map(stream_index)
    # ⚠️ 配额必须**第一个**从地图流里抽（真机在构造函数里就这么做，
    # ``StandardActMap.cs:93``）：晚一步整张图的随机数序列就错位。
    counts = point_counts if point_counts is not None else definition.map_point_counts(
        rng, ascension)

    builder = _Builder(rng, definition, counts, is_multiplayer=is_multiplayer,
                       replace_treasure_with_elites=replace_treasure_with_elites,
                       has_second_boss=has_second_boss)
    builder.generate()
    builder.assign_point_types()
    _prune_and_repair(builder, counts, rng)
    _center_grid(builder)
    _spread_adjacent_points(builder)
    _straighten_paths(builder)
    graph = _flatten(builder)

    if len(_MAP_CACHE) >= _MAP_CACHE_LIMIT:
        _MAP_CACHE.pop(next(iter(_MAP_CACHE)))
    _MAP_CACHE[cache_key] = graph
    return graph


def _flatten(builder: _Builder) -> MapGraph:
    """把生成态摊平成不可变图：远古点 → 各行（列升序）→ Boss → 双 Boss。"""
    order: list[_Point] = [builder.ancient]
    for row in range(1, builder._map_length):
        order.extend(builder.row_points(row))
    order.append(builder.boss)
    if builder.second_boss is not None:
        order.append(builder.second_boss)

    ids = {id(point): index for index, point in enumerate(order)}
    nodes = tuple(
        MapNode(node_id=index, row=point.row, col=point.col, kind=point.kind,
                edges=tuple(ids[id(child)] for child in point.children
                            if id(child) in ids))
        for index, point in enumerate(order)
    )
    return MapGraph(
        nodes=nodes,
        rows=builder._act.number_of_floors(builder._is_multiplayer),
        columns=GRID_COLUMNS,
        start=tuple(ids[id(point)] for point in builder.row_points(1)),
        ancient=ids[id(builder.ancient)],
        boss=ids[id(builder.boss)],
        act_id=builder._act.act_id,
        act_index=builder._act.index,
        second_boss=(None if builder.second_boss is None
                     else ids[id(builder.second_boss)]),
    )
