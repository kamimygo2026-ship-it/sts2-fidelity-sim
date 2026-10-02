"""命名 RNG 流。

对齐真机的关键：随机数按**用途分流**，且分流结构必须与真机一致。若"洗牌"与
"掉落"共用一条流，就会产生真机不存在的相关性，模型会学到虚假规律。

本骨架用标准库 ``random.Random``（Mersenne Twister）。Rust 核心阶段会换成
xoshiro256** / PCG64，但**流的分流结构与派生方式不变**。

⚠️ 本模块的所有对象只能存在于 ``core.Hidden`` 内部，绝不可进入观测
（见 ``docs/01`` §1.2 的 L3 级别）。
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")

#: 与真机**完全一致**的命名流。
#:
#: 来源：``MegaCrit.Sts2.Core.Entities.Rngs.RunRngType``（12 个枚举值），
#: 流名由 ``RunRngSet.CreateRng`` 用 ``SnakeCase(枚举名)`` 生成。
#: ``RunRngSet.cs`` 为每条流都写了用途说明，这里是逐条对应：
#:
#:   up_front                  开局一次性掷出**所有**内容：遭遇、事件、遗物
#:   shuffle                   抽牌堆洗牌（战斗开始与弃牌堆回洗）
#:   unknown_map_point         未知地图点的房间类型
#:   combat_card_generation    战斗内生成卡（攻击药水）。**与卡牌奖励分开**，
#:                             免得用药水影响到奖励
#:   combat_potion_generation  战斗内生成药水（炼金术）
#:   combat_card_selection     战斗内随机选牌（真劲）
#:   combat_energy_costs       战斗内随机费用（混乱 / 蛇眼）
#:   combat_targets            战斗内随机目标（弹跳药瓶 / 回旋镖）
#:   monster_ai                怪物出招的随机
#:   niche                     一次性杂项（诅咒之路等）
#:   combat_orbs               战斗内随机充能球（混沌）
#:   treasure_room_relics      多人宝箱的遗物分配
#:
#: ⚠️ **分流结构必须与真机一致**：合成或拆错流，会让"抽到好牌"与"下场遇弱怪"
#: 产生真机不存在的相关性，模型会学到虚假规律（``docs/02`` §2.4）。
#:
#: ⚠️ 真机的**内容预掷**（遭遇/事件/遗物）全在 ``up_front`` 里。这印证了
#: ``docs/01`` §1.8 的问题 3：内容在开局就定好了，所以重开节点面对同一批内容。
#:
#: ⚠️ **更正一处早先的错误结论**：本文件曾写着"真机没有独立的卡牌奖励流"。
#: 那是查漏了 —— 真机有**两层** RNG：
#:
#:   * ``RunRngType``（12 条，本列表的 run 级部分）
#:   * ``PlayerRngType``（3 条，**玩家级**）：``Rewards`` / ``Shops`` /
#:     ``Transformations``（``MegaCrit.Sts2.Core.Entities.Rngs.PlayerRngType``）
#:
#: 所以卡牌奖励走 ``rewards``、商店走 ``shops``、变形走 ``transformations``，
#: **不是**借 ``up_front``。借错流会让"抽到好牌"与"下场遇弱怪"产生真机
#: 不存在的相关性，模型会学到虚假规律。
STREAMS: tuple[str, ...] = (
    "up_front",
    # --- 玩家级（PlayerRngType） ---
    "rewards",
    "shops",
    "transformations",
    # --- run 级（RunRngType） ---
    "shuffle",
    "unknown_map_point",
    "combat_card_generation",
    "combat_potion_generation",
    "combat_card_selection",
    "combat_energy_costs",
    "combat_targets",
    "monster_ai",
    "niche",
    "combat_orbs",
    "treasure_room_relics",
)


def derive(master_seed: int, purpose: str) -> int:
    """从主种子派生某条流的种子。

    用哈希而不是加法/异或，是为了让不同 purpose 之间**统计独立**——
    否则 ``seed+1`` 这类派生的流之间会存在可被模型捕捉的相关性。
    """
    digest = hashlib.sha256(f"{master_seed}:{purpose}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


#: ``MapPointType`` 的枚举值顺序（``MegaCrit.Sts2.Core.Map.MapPointType``）。
#:
#: 地图剪枝把线段编码成字符串时用的是 ``(int)PointType``
#: （``MapPathPruning.GenerateSegmentKey``），而 ``SortedDictionary`` 按字符串
#: 序比较 —— 所以**枚举值本身**参与了剪枝顺序。这些数字必须与源码一致。
MAP_POINT_TYPE_ORDER: dict[str, int] = {
    "unassigned": 0,
    "unknown": 1,
    "shop": 2,
    "treasure": 3,
    "rest_site": 4,
    "monster": 5,
    "elite": 6,
    "boss": 7,
    "ancient": 8,
}


class Rng:
    """``MegaCrit.Sts2.Core.Random.Rng`` 的引擎侧门面。

    ⚠️ **PRNG 本身不同，接口语义与消费顺序相同**。真机用的 ``MegaRandom`` 实现
    没有反编译出来（``Rng.cs`` 里只有一层包装），这里换成 ``random.Random``
    （Mersenne Twister）。所以**同一 seed 不会得到与真机逐点相同的地图**——
    本类复刻的是"哪一步抽了几次、上界开还是闭、怎么排序"，不是位级一致。
    想声称位级一致，必须先拿到 ``MegaRandom`` 的实现并做真机对拍（``docs/13`` §7）。

    逐条对齐的语义（来源 ``MegaCrit.Sts2.Core.Random.Rng``）：

      * ``NextInt(maxExclusive)`` / ``NextInt(minInclusive, maxExclusive)``
        —— 上界**开**区间（``Rng.cs:83-103``）。``NextInt(5, 7)`` 只出 5 或 6。
      * ``NextFloat()`` —— ``[0, 1)`` 的**一次**抽样（``Rng.cs:167-186``）。
        不能用 ``random.uniform`` 代替，那会消耗两次。
      * ``NextGaussianInt`` —— Box-Muller，取 ``sin`` 分支，``Math.Round``
        是**银行家舍入**（``Rng.cs:260-281``）。注意 ``NextGaussianDouble``
        用的是 ``cos``（``Rng.cs:227-249``）——这个差异是真机自带的，不是笔误。
      * ``Shuffle`` / ``UnstableShuffle`` —— Fisher-Yates 从尾部往前
        （``Rng.cs:342-354``、``ListExtensions.cs:45-60``）。
      * ``StableShuffle`` —— **先按元素自然序排序**，再 Fisher-Yates
        （``ListExtensions.cs:22-31``）。"stable" 指结果与输入顺序无关。
    """

    __slots__ = ("_rng", "calls")

    def __init__(self, seed: int, name: str | None = None) -> None:
        #: 抽了多少次 —— 用于测试"消费顺序"（真机 ``Rng._counter`` 同义）。
        self.calls = 0
        self._rng = random.Random(derive(seed, name) if name else seed)

    # ---- 整数 --------------------------------------------------------
    def next_int(self, *args: int) -> int:
        """``NextInt``。一个参数 = 上界（开）；两个参数 = 左闭右开。"""
        if len(args) == 1:
            low, high = 0, args[0]
        elif len(args) == 2:
            low, high = args
        else:                                            # pragma: no cover
            raise TypeError("next_int 接受 1 或 2 个参数")
        if low >= high:
            raise ValueError("next_int: 上界必须大于下界（真机直接抛异常）")
        self.calls += 1
        return self._rng.randrange(low, high)

    # ---- 浮点 --------------------------------------------------------
    def next_float(self) -> float:
        self.calls += 1
        return self._rng.random()

    def next_double(self) -> float:
        return self.next_float()

    def next_gaussian_double(self, mean: float = 0.0, stddev: float = 1.0,
                             low: float = 0.0, high: float = 1.0) -> float:
        """``NextGaussianDouble``：``cos`` 分支 + 拒绝采样到 ``[0, 1]``。"""
        if low > high:
            raise ValueError("next_gaussian_double: low 不得大于 high")
        if not 0.0 <= mean <= 1.0:
            raise ValueError("next_gaussian_double: mean 必须落在 [0, 1]（真机同）")
        while True:
            d = 1.0 - self.next_double()
            u = 1.0 - self.next_double()
            radius = math.sqrt(-2.0 * math.log(d))
            value = mean + radius * math.cos(2.0 * math.pi * u) * stddev
            if 0.0 <= value <= 1.0:
                return value * (high - low) + low

    def next_gaussian_int(self, mean: int, stddev: int, low: int, high: int) -> int:
        """``NextGaussianInt``：``sin`` 分支 + 银行家舍入 + 拒绝采样。"""
        if low > high:
            raise ValueError("next_gaussian_int: low 不得大于 high")
        if not low <= mean <= high:
            raise ValueError("next_gaussian_int: mean 必须落在 [low, high]（真机同）")
        while True:
            d = 1.0 - self.next_double()
            u = 1.0 - self.next_double()
            value = mean + stddev * math.sqrt(-2.0 * math.log(d)) * math.sin(
                2.0 * math.pi * u)
            rounded = int(round(value))       # Python round = 银行家舍入，与 Math.Round 同
            if low <= rounded <= high:
                return rounded

    # ---- 序列 --------------------------------------------------------
    def shuffle(self, items: list) -> list:
        """``UnstableShuffle``：结果与输入顺序有关。"""
        for index in range(len(items) - 1, 0, -1):
            other = self.next_int(index + 1)
            items[index], items[other] = items[other], items[index]
        return items

    def stable_shuffle(self, items: list, key=None) -> list:
        """``StableShuffle``：先按自然序排序再洗，结果与输入顺序**无关**。"""
        items[:] = sorted(items, key=key)
        return self.shuffle(items)

    def next_item(self, items):
        """``NextItem``：``NextInt(0, count)`` 取一个。"""
        if not items:
            return None
        return items[self.next_int(len(items))]


class RngSet:
    """一组命名随机流。整体可快照/恢复，这是 SL 重开语义的基础。"""

    __slots__ = ("master_seed", "streams")

    def __init__(self, master_seed: int) -> None:
        self.master_seed = master_seed
        self.streams: dict[str, random.Random] = {
            name: random.Random(derive(master_seed, name)) for name in STREAMS
        }

    def __getitem__(self, name: str) -> random.Random:
        return self.streams[name]

    def named(self, purpose: str) -> "Rng":
        """按用途从主种子派生一条**独立**的 ``Rng``。

        对应真机 ``new Rng(runState.Rng.Seed, name)``，例如
        ``StandardActMap.CreateFor`` 里的 ``$"act_{CurrentActIndex + 1}_map"``
        （``StandardActMap.cs:113``）。

        ⚠️ 刻意**不**复用 ``STREAMS`` 里那 12 条：真机的地图流是从 run 种子另起的，
        与 ``up_front`` 无关。挂到 ``up_front`` 上会让"地图长什么样"和"开局掷出的
        内容"产生真机不存在的相关性（``docs/02`` §2.4）。
        """
        return Rng(self.master_seed, purpose)

    def act_map(self, act_index: int) -> "Rng":
        """本幕地图的专用流（``StandardActMap.CreateFor``，索引从 0 起）。"""
        return self.named(f"act_{act_index + 1}_map")

    # ---- 快照 / 恢复 -------------------------------------------------
    def get_state(self) -> dict[str, object]:
        return {name: rng.getstate() for name, rng in self.streams.items()}

    def set_state(self, state: dict[str, object]) -> None:
        for name, value in state.items():
            self.streams[name].setstate(value)  # type: ignore[arg-type]

    def clone(self) -> "RngSet":
        other = RngSet.__new__(RngSet)
        other.master_seed = self.master_seed
        other.streams = {}
        for name, rng in self.streams.items():
            twin = random.Random()
            twin.setstate(rng.getstate())
            other.streams[name] = twin
        return other

    # ---- 常用操作 ----------------------------------------------------
    def shuffle(self, seq: list[T], stream: str = "shuffle") -> None:
        self.streams[stream].shuffle(seq)

    def choice(self, seq: Sequence[T], stream: str = "misc") -> T:
        return self.streams[stream].choice(seq)

    def randint(self, a: int, b: int, stream: str = "misc") -> int:
        return self.streams[stream].randint(a, b)

    def uniform(self, low: float, high: float, stream: str = "misc") -> float:
        """``Rng.NextFloat(low, high)``：左闭右开的浮点区间。

        商店价格用它（``Shops.NextFloat(0.95f, 1.05f)``）。
        """
        return self.streams[stream].uniform(low, high)

    def next_index(self, stream: str, length: int) -> int:
        """``RngStream.NextItem`` 的下标形式：从 ``[0, length)`` 里取一个。

        充能球（闪电打随机敌人）与"弹跳药瓶"这类效果走的是 ``combat_targets``
        这条**独立流**，不是主随机 —— 混用会让同一种子下的洗牌顺序跟着变，
        破坏"重开后面对同一副牌序"。
        """
        if length <= 0:
            raise ValueError("next_index: length 必须为正")
        return self.streams[stream].randrange(length)

    def next_float(self, stream: str = "misc") -> float:
        """``Rng.NextFloat()``：``[0, 1)`` 的一次抽样。

        ⚠️ **不能**用 :meth:`uniform`：那个底层是 ``random.uniform``，
        会消耗**两次**随机数，于是同一条流上后续的值与真机分叉
        （遗物稀有度、商店价格都吃这个亏，而且不报错）。
        """
        return self.streams[stream].random()

    def weighted_choice(
        self, items: Iterable[tuple[T, float]], stream: str = "monster"
    ) -> T:
        pool = list(items)
        total = sum(w for _, w in pool)
        if total <= 0:
            raise ValueError("weighted_choice: 权重之和必须为正")
        roll = self.streams[stream].random() * total
        acc = 0.0
        for item, weight in pool:
            acc += weight
            if roll < acc:
                return item
        return pool[-1][0]
