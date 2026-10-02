"""遗物抓包（``MegaCrit.Sts2.Core.Runs/RelicGrabBag.cs``）。

真机把"本局能出的遗物"按**稀有度分桶**、开局洗一次，之后所有遗物来源
（精英奖励、事件、商店）都从桶的**前端**取，取走的永远不再出现。::

    Populate(player, rng)                       RelicGrabBag.cs:69-92
      池 = SharedRelicPool ∪ 角色遗物池，**只保留 4 种稀有度**
      （Common / Uncommon / Rare / Shop，`_rarities` 就这四个）
      每个桶 `UnstableShuffle(rng)`
    PullFromFront(rarity, filter)               RelicGrabBag.cs:129-146
      桶里第一个满足 filter 的，取走后从桶里删除
      桶空 → null，调用方用 `RelicFactory.FallbackRelic`（``Circlet``）兜底

洗牌用的是 ``Rng.UpFront``（``RunManager.cs:522-526``），
稀有度掷点用的是调用方给的 RNG —— 事件里写的是
``PullNextRelicFromFront(base.Owner)``，那个重载走 ``PlayerRng.Rewards``
（``RelicFactory.cs:80-83``），也就是本引擎的 ``rewards`` 流。

⚠️ 为什么必须建这个而不是"随机抽一个同稀有度的遗物"：
抓包的核心语义是**取走就不再出现**。少了它，同一局会重复拿到同一个遗物，
而"重开之后还能再抽到它"这件事会让重开策略学到真机不存在的套利。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .rng import RngSet

#: 会进抓包的稀有度 —— 逐字照抄 ``RelicGrabBag._rarities``（RelicGrabBag.cs:16-22）。
#: ``Ancient`` / ``Event`` / ``Starter`` **不进包**（它们是剧情与起始给的）。
BAG_RARITIES: tuple[str, ...] = ("common", "uncommon", "rare", "shop")

#: 桶空时的兜底遗物：``RelicFactory.FallbackRelic => ModelDb.Relic<Circlet>()``
#: （``RelicFactory.cs:13``）。
FALLBACK_RELIC = "circlet"


def unstable_shuffle(items: list[str], rng: RngSet,
                     stream: str = "up_front") -> None:
    """``ListExtensions.UnstableShuffle``（``ListExtensions.cs:45-60``）。

    ⚠️ 是**从尾部往前的 Fisher-Yates**：``n`` 从 ``count-1`` 递减到 1，
    每次 ``j = rng.NextInt(n + 1)``（**闭区间 0..n**）后交换 ``list[n]`` 与 ``list[j]``。
    写成"从头往后"或"用 shuffle()"都会得到另一个排列 —— 而遗物顺序是玩家看得见的。
    """
    for n in range(len(items) - 1, 0, -1):
        j = rng.next_index(stream, n + 1)
        items[n], items[j] = items[j], items[n]


def roll_rarity(rng: RngSet, stream: str = "rewards") -> str:
    """``RelicFactory.RollRarity``（``RelicFactory.cs:85-93``）。

    .. code-block:: csharp

       float num = rng.NextFloat();
       return (num < 0.5f) ? Common : ((!(num < 0.83f)) ? Rare : Uncommon);

    即 **50% 普通 / 33% 罕见 / 17% 稀有**（阈值为 0.5 与 0.83）。
    """
    roll = rng.next_float(stream)
    if roll < 0.5:
        return "common"
    if roll < 0.83:
        return "uncommon"
    return "rare"


class RelicGrabBag:
    """按稀有度分桶、从前端取走的遗物包。"""

    __slots__ = ("_deques", "pulled")

    def __init__(self) -> None:
        self._deques: dict[str, list[str]] = {}
        #: 取走过的遗物（本局不再出现）。用于日志与测试。
        self.pulled: list[str] = []

    @property
    def is_populated(self) -> bool:
        return bool(self._deques)

    def populate(self, entries: Iterable[tuple[str, str]], rng: RngSet,
                 stream: str = "up_front") -> None:
        """``Populate``：按稀有度分桶 + 逐桶洗牌。``entries`` 是 ``(rid, rarity)``。"""
        if self.is_populated:
            raise RuntimeError("RelicGrabBag 已经填过了（真机也会抛）")
        for rid, rarity in entries:
            if rarity in BAG_RARITIES:
                self._deques.setdefault(rarity, []).append(rid)
        for bucket in self._deques.values():
            unstable_shuffle(bucket, rng, stream)

    def counts(self) -> dict[str, int]:
        return {rarity: len(bucket) for rarity, bucket in self._deques.items()}

    def has_available(self) -> bool:
        """``HasAvailableRelics``：还有没有能取的（事件用它决定要不要出现）。"""
        return any(self._deques.get(rarity) for rarity in BAG_RARITIES)

    def pull_from_front(self, rarity: str) -> str | None:
        """``PullFromFront``：该稀有度桶的第一个，取走后删除。桶空返回 ``None``。"""
        bucket = self._deques.get(rarity)
        if not bucket:
            return None
        rid = bucket.pop(0)
        self.pulled.append(rid)
        return rid

    def remove(self, rid: str) -> None:
        """``Remove``：把某个遗物从桶里拿掉（已获得 / 被其他来源取走时）。"""
        for bucket in self._deques.values():
            if rid in bucket:
                bucket.remove(rid)
                return
