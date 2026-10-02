"""未知地图点的房间掷骰（``UnknownMapPointOdds``）。

真机的地图点类型里有 ``Unknown``（问号），它**进房间那一刻**才决定是什么房：
事件 / 怪 / 精英 / 宝箱 / 商店。而且掷出来的结果会**改变后续概率** ——
一直没掷出事件，事件概率就越来越高（``AbstractOdds`` 的"概率递增"）。

来源：``MegaCrit.Sts2.Core.Odds.UnknownMapPointOdds`` +
``MegaCrit.Sts2.Core.Runs.RunManager.BuildRoomTypeBlacklist`` /
``RollRoomTypeFor`` / ``TryGetRoomTypeForTutorial``。

⚠️ 三个**必须显式建模**的行为，漏掉任何一个都会让分布偏，而且不报错：

  1. 概率是**跨房间累计**的，不是每次重置。掷中某个类型时它**回落**到基础值，
     其余类型各自**加上**自己的基础值（``UnknownMapPointOdds.cs:159-175``）。
     精英的基础值是 ``-1``（负数 = 永不出现），所以精英只会通过遗物/事件
     修改概率后出现 —— 在基础状态下**永远掷不出精英**。
  2. 商店会被**拉黑**：上一个点走过商店，或者下一步所有点都是商店
     （``RunManager.cs:660-668``）。不建模就会连续两个商店。
  3. **首次存档**（``NumberOfRuns == 0``）有教学硬规则：前两个未知点固定是
     事件，第三个固定是怪（``UnknownMapPointOdds.cs:125-136`` 与
     ``RunManager.cs:1000-1026``）。引擎没有存档层，所以这行为**参数化**：
     :class:`UnknownMapPointOdds` 的 ``number_of_runs`` 默认 1（普通玩家，
     也就是训练要的分布），设为 0 才复刻"全新存档第一次开局"。

⚠️ 未建模：``Hook.ModifyUnknownMapPointRoomTypes`` /
``Hook.ModifyOddsIncreaseForUnrolledRoomType``（"致命敌人"这类修正器的钩子）。
修正器系统整体未实现，所以这里不假装有 —— 见 ``docs/12`` §2.9 与 ``docs/13`` §17。
"""

from __future__ import annotations

#: 房间类型（与 ``RoomType`` 的用法一致，用字符串是为了和 ``mapgen`` 的点类型对齐）。
MONSTER = "monster"
ELITE = "elite"
TREASURE = "treasure"
SHOP = "shop"
EVENT = "event"
REST_SITE = "rest_site"

#: ``UnknownMapPointOdds.cs:29-35`` 的 ``_baseOdds``。
#: **顺序也是语义**：掷骰与更新都按这个顺序累加。
BASE_ODDS: tuple[tuple[str, float], ...] = (
    (MONSTER, 0.1),
    (ELITE, -1.0),
    (TREASURE, 0.02),
    (SHOP, 0.03),
)


class UnknownMapPointOdds:
    """未知点的房间概率状态（真机是 run 级常驻对象，跨房间累计）。"""

    __slots__ = ("_rng", "_odds", "number_of_runs")

    def __init__(self, rng, number_of_runs: int = 1) -> None:
        #: 真机走 ``RunRngSet.UnknownMapPoint`` 这条**专用流**
        #: （``RunOddsSet.cs:19``），不是 ``up_front``。
        self._rng = rng
        self._odds: dict[str, float] = dict(BASE_ODDS)
        #: 存档里已完成的开局数。``0`` = 全新存档（触发教学硬规则）。
        self.number_of_runs = number_of_runs

    # ---- 只读视图（可存读档，真机也是这么序列化的） --------------------
    @property
    def monster_odds(self) -> float:
        return self._odds[MONSTER]

    @property
    def elite_odds(self) -> float:
        return self._odds[ELITE]

    @property
    def treasure_odds(self) -> float:
        return self._odds[TREASURE]

    @property
    def shop_odds(self) -> float:
        return self._odds[SHOP]

    @property
    def event_odds(self) -> float:
        """``EventOdds``：**剩下的**都是事件（``UnknownMapPointOdds.cs:97``）。"""
        return max(0.0, 1.0 - sum(value for value in self._odds.values() if value > 0))

    def snapshot(self) -> dict[str, float]:
        return dict(self._odds)

    def restore(self, values: dict[str, float]) -> None:
        """读档恢复：真机存的是四个概率值（``SerializableRunOddsSet``）。"""
        for kind, _base in BASE_ODDS:
            if kind in values:
                self._odds[kind] = float(values[kind])

    def reset_to_base(self) -> None:
        """``ResetToBase``：**幕与幕之间**调用（``UnknownMapPointOdds.cs:183``）。"""
        self._odds = dict(BASE_ODDS)

    # ---- 掷骰 --------------------------------------------------------
    def roll(self, blacklist: frozenset[str] = frozenset(),
             unknown_visited: int = 0) -> str:
        """掷一个房间类型。

        :param blacklist: 本次不允许掷出的类型（见 :func:`build_blacklist`）。
        :param unknown_visited: 本局**已经走过的未知点数**（教学硬规则要用）。
            真机是从 ``MapPointHistory`` 里数出来的。
        """
        # ⚠️ 教学覆盖先于常规掷骰（``RunManager.TryGetRoomTypeForTutorial``）：
        # 全新存档的**第一个**未知点一定是事件。
        if self.number_of_runs == 0 and unknown_visited == 0:
            return EVENT
        # ⚠️ 常规掷骰里还有一条同类硬规则（``UnknownMapPointOdds.cs:125-136``）：
        # 第 2 个未知点仍是事件、第 3 个必是怪。两条合起来才是真机行为。
        if self.number_of_runs == 0:
            if unknown_visited < 2:
                return EVENT
            if unknown_visited == 2:
                return MONSTER

        candidates = [kind for kind, _base in BASE_ODDS if kind not in blacklist]
        if EVENT not in blacklist:
            candidates.append(EVENT)
        # 默认值：有事件就是事件，否则取候选里最小的那个（真机 ``roomTypes.Order().First()``）。
        chosen = EVENT if EVENT not in blacklist else min(candidates)
        roll = self._rng.next_float()
        cumulative = 0.0
        for kind, _base in BASE_ODDS:
            current = self._odds[kind]
            if kind in blacklist or current < 0:
                continue
            cumulative += current
            if roll <= cumulative:
                chosen = kind
                break
        self._apply_roll(chosen, blacklist)
        return chosen

    def _apply_roll(self, chosen: str, blacklist: frozenset[str]) -> None:
        """掷中 → 回落基础值；没掷中的（且在候选里）→ 各自加上基础值。"""
        for kind, base in BASE_ODDS:
            if kind == chosen:
                self._odds[kind] = base
            elif kind not in blacklist:
                self._odds[kind] += base


def build_blacklist(previous_room_kinds: tuple[str, ...],
                    next_kinds: tuple[str, ...]) -> frozenset[str]:
    """``RunManager.BuildRoomTypeBlacklist``（``RunManager.cs:660-668``）。

    只有一条规则：上一个点进过商店，**或者**下一步所有点都是商店 → 拉黑商店。
    """
    if SHOP in previous_room_kinds:
        return frozenset({SHOP})
    if next_kinds and all(kind == SHOP for kind in next_kinds):
        return frozenset({SHOP})
    return frozenset()
