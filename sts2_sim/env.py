"""SL 战斗环境：meta-episode + 重开预算 + 跨试次知识累积。

``docs/01`` §1.4 的实现：

    MetaEpisode(节点 s₀, 隐藏状态 h, 预算 K):
        for i = 1…K:
            τᵢ ← rollout(从 s₀ 开始, 隐藏状态固定为 h, 策略 π(·|可见状态, knowledge))
            knowledge ← knowledge ∪ reveal(τᵢ)
            if 接受: return τᵢ

**SL 在这里是环境的一等公民，不是外部脚本里的一个方法。** 阶段机：

```
      ┌──────────── restart（扣预算、惩罚）────────────┐
      ▼                                              │
  [combat] ──战斗结束──► [meta] ──accept──► [done] ───┘
      │                     │
      └──restart────────────┘
```

  * ``combat``：正常出牌动作 **+（预算未耗尽时）restart**
  * ``meta``  ：战斗已结束，但预算还在 → 选 **accept**（认了这个结果）或 **restart**
  * ``done``  ：meta-episode 结束，返回**被接受那次**的终局奖励

关键语义（``docs/01`` §1.6 测试 2）：``restart`` 通过 ``restore`` 回到节点开头，
**保留隐藏 RNG 状态**——同一副牌序。若实现成重掷，agent 会学到真机上不存在的策略。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .content import DEFAULT_CHARACTER, STARTING_DECK
from .core import (
    Action, BASE_ENERGY, PLAYER_MAX_HP, CombatInit, CombatState, Snapshot,
    legal_actions as combat_legal_actions,
    render_text, restore, snapshot, start_combat, step,
)
from .observe import AttemptView, Observation, observe

STAGE_COMBAT = "combat"
STAGE_META = "meta"
STAGE_DONE = "done"

#: SL 重开预算上限（用户规格）。
#: * 模拟器内：50 —— 训练时可以开得大，但**必须有上限**，否则"一直重开到赢"
#:   是退化解，学不到牌理。
#: * 真机桥接：10 —— 真机每次存档/读档要几十秒，重开 50 次不现实。
MAX_ATTEMPT_BUDGET_SIM = 50
MAX_ATTEMPT_BUDGET_REAL = 10

#: 到顶后如何选定结果：
#:   "best" —— 取历次试次里**评价最好**的那次（默认，对应"到次数之后选择最优解"）
#:   "last" —— 只认最后一次（消融用：更接近"不能回退"的严格语义）
SELECTION_BEST = "best"
SELECTION_LAST = "last"


@dataclass
class AttemptRecord:
    index: int
    hand_by_turn: list[tuple[str, ...]] = field(default_factory=list)
    outcome: str = "running"

    def view(self) -> AttemptView:
        return AttemptView(index=self.index,
                           hand_by_turn=tuple(self.hand_by_turn),
                           outcome=self.outcome)


def attempt_score(state: CombatState) -> float:
    """一次试次的评价分。赢了当然好；同样是赢，剩血越多越好。

    这是"最优解"的判据。它**不进入观测**（属于环境内部评价），
    只在 meta-episode 结束时决定保留哪次结果、以及给出终局回报。
    """
    hp_ratio = state.player.hp / max(1, state.player.max_hp)
    if state.phase == "won":
        return 1.0 + 0.5 * hp_ratio
    return -1.0 + 0.1 * hp_ratio


@dataclass
class AttemptOutcome:
    index: int
    won: bool
    score: float
    reward: float
    snapshot: Snapshot


class SpireEnv:
    """一个"节点"= 一场战斗，可带 K 次重开预算。"""

    def __init__(self, seed: int = 0,
                 encounter: Sequence[str] = ("jaw_worm",),
                 deck: Sequence[str] | None = None,
                 player_hp: int = 80,
                 attempt_budget: int = 0,
                 restart_penalty: float = -0.01,
                 selection: str = SELECTION_BEST,
                 character: str = DEFAULT_CHARACTER,
                 ascension: int = 0,
                 room: str = "monster",
                 relics: Sequence[str] = (),
                 potions: Sequence[str | None] = (),
                 max_energy: int | None = None,
                 player_max_hp: int | None = None) -> None:
        """``attempt_budget`` 是**允许的重开次数**（不是总试次数）。

        * ``0`` → 不允许重开，即 **NoSL**（``docs/01`` §1.3）
        * 总试次数上限 = ``attempt_budget + 1``
        * 上限 50（模拟器）/ 10（真机桥接）——用户规格

        ⭐ ``character`` / ``ascension`` / ``room`` / ``relics`` / ``potions`` /
        ``max_energy`` 是**战斗上下文**，收进 :class:`sts2_sim.core.CombatInit`
        一并传给 ``start_combat``（见该类型的说明）。

        ⚠️ 以前这里**一个都不传**，于是 ``CombatPool(character="defect")``
        训练出来的 ``CombatState.character`` 是 ``ironclad``：
        "生成一张能力牌"按铁甲战士的卡池取牌，而报告里一切正常
        （外部复核 R2 的实测：请求 defect / 实际 ironclad、最大生命 80、遗物为空）。
        """
        if attempt_budget < 0:
            raise ValueError("attempt_budget 必须 ≥ 0（0 表示不允许重开，即 NoSL）")
        if attempt_budget > MAX_ATTEMPT_BUDGET_SIM:
            raise ValueError(
                f"attempt_budget={attempt_budget} 超过模拟器上限 "
                f"{MAX_ATTEMPT_BUDGET_SIM}；真机请用 ≤{MAX_ATTEMPT_BUDGET_REAL}")
        if selection not in (SELECTION_BEST, SELECTION_LAST):
            raise ValueError(f"selection ∈ {{{SELECTION_BEST}, {SELECTION_LAST}}}")
        self._seed = seed
        self._encounter = tuple(encounter)
        self._deck = tuple(deck) if deck is not None else tuple(STARTING_DECK)
        self._player_hp = player_hp
        self._attempt_budget = attempt_budget
        self._restart_penalty = restart_penalty
        self._selection = selection
        #: 战斗上下文：**一份**定义，``reset`` / ``replay`` 都从这里取。
        self._character = str(character)
        self._ascension = int(ascension)
        self._room = str(room)
        self._relics = tuple(str(r) for r in relics)
        self._potions = tuple(potions)
        #: ``None`` = 按角色表取（``CHARACTERS[character].energy``），
        #: 取不到才退回 ``BASE_ENERGY``。写死 3 会让"能量不是 3 的角色"静默错。
        self._max_energy = max_energy
        self._player_max_hp = player_max_hp

        self._node_snapshot: Snapshot | None = None
        self._state: CombatState | None = None
        self._records: list[AttemptRecord] = []
        self._attempts_used = 0
        self._current: AttemptRecord | None = None
        self._last_turn_seen = 0
        self._stage = STAGE_DONE
        self._outcome_reward = 0.0
        self._best: AttemptOutcome | None = None
        self._finished_attempts: list[AttemptOutcome] = []

    # ---- 战斗上下文 ---------------------------------------------------
    def combat_init(self) -> CombatInit:
        """这一环境开局用的**显式上下文**（测试与训练清单都读它）。

        能量与最大生命按角色表补齐（取不到才退回常量）—— 于是"角色声明的资源"
        与"战斗里实际拿到的资源"不可能对不上。实测 ``defect`` 的永久上限是
        **75**，而训练池里给的是 60–80：不按角色表补上限就会造出
        "defect 当前生命 78 / 上限 80"这种真机不存在的状态。
        """
        from .content import CHARACTERS
        definition = CHARACTERS.get(self._character)
        energy = self._max_energy
        if energy is None:
            energy = (definition.energy if definition is not None
                      else BASE_ENERGY)
        max_hp = self._player_max_hp
        if max_hp is None:
            max_hp = (definition.hp if definition is not None
                      else PLAYER_MAX_HP)
        return CombatInit(character=self._character, ascension=self._ascension,
                          room=self._room, relics=self._relics,
                          potions=self._potions, max_energy=int(energy),
                          player_max_hp=int(max_hp))

    def _starting_hp(self, init: CombatInit) -> int:
        """开局当前生命。

        ⚠️ 起始生命随机 60–80 是**有意为之的训练课程变量**
        （``CombatPool`` 每局重掷，见 ``docs/13``），但**不能超过该角色的永久上限**：
        defect 的上限是 75，直接给 80 会造出真机不存在的状态。这里只做
        "不超过上限"的裁剪，不改变课程本身。
        """
        upper = int(init.player_max_hp or PLAYER_MAX_HP)
        return max(1, min(int(self._player_hp), upper))

    # ---- 只读访问 ----------------------------------------------------
    @property
    def raw_state(self) -> CombatState:
        """⚠️ **仅供测试与调试**。策略路径不得使用——那会绕过信息隔离。"""
        assert self._state is not None
        return self._state

    @property
    def attempts_used(self) -> int:
        return self._attempts_used

    @property
    def attempts_left(self) -> int:
        return max(0, self._attempt_budget - self._attempts_used)

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def attempt_budget(self) -> int:
        return self._attempt_budget

    # ---- 生命周期 ----------------------------------------------------
    def reset(self, seed: int | None = None) -> tuple[Observation, dict]:
        if seed is not None:
            self._seed = seed
        # ⚠️ 上下文**每次 reset 现算**：角色表是内容加载之后才有的，
        # 在 ``__init__`` 里算会把"内容还没加载"时的常量固化下来。
        init = self.combat_init()
        self._state = start_combat(self._deck, self._encounter, self._seed,
                                   self._starting_hp(init),
                                   **init.as_kwargs())
        self._node_snapshot = snapshot(self._state)   # 含隐藏 RNG
        self._records = []
        self._attempts_used = 0
        self._current = AttemptRecord(index=0)
        self._last_turn_seen = 0
        self._stage = STAGE_COMBAT
        self._outcome_reward = 0.0
        self._best = None
        self._finished_attempts = []
        self._record_turn()
        return self._observe(), self._info()

    def legal_actions(self) -> list[Action]:
        """当前阶段的合法动作。**mask 由环境给出**，策略不得绕过（docs/01 §1.6 测试 6）。"""
        if self._state is None:
            return []
        if self._stage == STAGE_COMBAT:
            actions = combat_legal_actions(self._state)
            if self.attempts_left > 0:
                actions.append(Action("restart"))
            return actions
        if self._stage == STAGE_META:
            actions = []
            if self.attempts_left > 0:
                actions.append(Action("restart"))
            actions.append(Action("accept"))
            return actions
        return []

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        assert self._state is not None, "先调用 reset()"

        if action.kind == "restart":
            return self._do_restart()
        if action.kind == "accept":
            return self._do_accept()
        if self._stage != STAGE_COMBAT:
            raise ValueError(f"阶段 {self._stage} 不接受战斗动作 {action!r}")

        previous_turn = self._state.turn
        result = step(self._state, action)
        if self._state.turn != previous_turn:
            self._record_turn()

        if not result.done:
            return self._observe(), 0.0, False, False, self._info()

        # 战斗结束：先记录，再决定是否需要 meta 决策
        self._finish_current("won" if result.won else "lost")
        self._outcome_reward = result.reward
        self._record_outcome()
        if self.attempts_left > 0:
            self._stage = STAGE_META
            return self._observe(), 0.0, False, False, self._info()
        return self._finalize()

    # ---- SL：重开 / 接受 ---------------------------------------------
    def _do_restart(self) -> tuple[Observation, float, bool, bool, dict]:
        assert self._node_snapshot is not None
        if self.attempts_left <= 0:
            raise ValueError("非法动作：重开预算已耗尽")
        if self._stage == STAGE_COMBAT:
            self._finish_current("aborted")
        self._attempts_used += 1
        # ⭐ restore 保留隐藏 RNG —— 重开面对同一副牌序
        self._state = restore(self._node_snapshot)
        self._current = AttemptRecord(index=len(self._records))
        self._last_turn_seen = 0
        self._stage = STAGE_COMBAT
        self._outcome_reward = 0.0
        self._record_turn()
        return self._observe(), self._restart_penalty, False, False, self._info()

    def _do_accept(self) -> tuple[Observation, float, bool, bool, dict]:
        if self._stage != STAGE_META:
            raise ValueError(f"阶段 {self._stage} 不接受 accept")
        return self._finalize()

    def _record_outcome(self) -> None:
        """记录本次试次的评价，并更新"当前最优"。"""
        assert self._state is not None
        score = attempt_score(self._state)
        outcome = AttemptOutcome(
            index=self._attempts_used,
            won=self._state.phase == "won",
            score=score,
            reward=score,
            snapshot=snapshot(self._state),
        )
        self._finished_attempts.append(outcome)
        if self._best is None or score > self._best.score:
            self._best = outcome

    def _chosen_outcome(self) -> AttemptOutcome | None:
        if not self._finished_attempts:
            return None
        if self._selection == SELECTION_LAST:
            return self._finished_attempts[-1]
        return self._best

    def _finalize(self) -> tuple[Observation, float, bool, bool, dict]:
        """结束 meta-episode：**保留最优（或最后一次）试次的结果**。

        这是"到次数之后选择最优解"的实现：状态与回报都来自被保留的那次试次，
        而不是最后一次乱试。
        """
        chosen = self._chosen_outcome()
        self._stage = STAGE_DONE
        chosen_index = None
        if chosen is not None:
            self._state = restore(chosen.snapshot)
            self._outcome_reward = chosen.reward
            chosen_index = chosen.index
        self._kept_index = chosen_index
        return self._observe(), self._outcome_reward, True, False, self._info()

    def restart_attempt(self) -> bool:
        """编程式重开（脚本 / 测试用）。等价于执行 ``restart`` 动作。"""
        if self._node_snapshot is None or self.attempts_left <= 0:
            return False
        self._do_restart()
        return True

    def accept_attempt(self) -> None:
        if self._stage == STAGE_META:
            self._finalize()
        elif self._current is not None and self._current.outcome == "running":
            self._current.outcome = "accepted"

    @property
    def best_score(self) -> float | None:
        return self._best.score if self._best is not None else None

    @property
    def kept_attempt(self) -> int | None:
        return getattr(self, "_kept_index", None)

    @property
    def attempts_finished(self) -> int:
        return len(self._finished_attempts)

    def _finish_current(self, outcome: str) -> None:
        if self._current is not None:
            self._current.outcome = outcome
            self._records.append(self._current)
            self._current = None

    # ---- 观测 ---------------------------------------------------------
    def _record_turn(self) -> None:
        """记录本回合开局手牌。

        手牌顺序 = 抽牌顺序，所以这份记录**就是**被观察到的牌序——
        SL 玩家重开后能利用的正是它。
        """
        assert self._state is not None
        if (self._state.turn == self._last_turn_seen and self._current
                and self._current.hand_by_turn):
            return
        self._last_turn_seen = self._state.turn
        if self._current is not None:
            self._current.hand_by_turn.append(
                tuple(card.cid for card in self._state.hand)
            )

    def _observe(self) -> Observation:
        assert self._state is not None
        prior = [r.view() for r in self._records]
        return observe(self._state, prior, self._attempts_used, self.attempts_left)

    def _info(self) -> dict:
        assert self._state is not None
        return {
            "legal_actions": self.legal_actions(),
            "stage": self._stage,
            "phase": self._state.phase,
            "attempts_used": self._attempts_used,
            "attempts_left": self.attempts_left,
            "attempts_finished": len(self._finished_attempts),
            "best_score": self._best.score if self._best is not None else None,
            "kept_attempt": getattr(self, "_kept_index", None),
            "selection": self._selection,
        }

    # ---- 调试 ---------------------------------------------------------
    def render_text(self) -> str:
        assert self._state is not None
        header = (f"[试次 {self._attempts_used + 1}/{self._attempt_budget}"
                  f" | 阶段 {self._stage} | 剩余重开 {self.attempts_left}]")
        if self._stage == STAGE_META:
            return header + "\n战斗已结束，请选择 accept 或 restart"
        return header + "\n" + render_text(self._state)

    def replay(self, actions: Sequence[Action], seed: int | None = None) -> CombatState:
        """复现任意一局（bug 报告与对拍的入口）。"""
        init = self.combat_init()
        state = start_combat(self._deck, self._encounter,
                             self._seed if seed is None else seed,
                             self._starting_hp(init), **init.as_kwargs())
        for action in actions:
            if state.finished() or action.kind not in ("play_card", "end_turn"):
                break
            step(state, action)
        return state
