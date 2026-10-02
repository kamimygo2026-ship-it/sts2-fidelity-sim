"""观测张量化：``Observation`` → 定长数组。

这是策略**真正的输入路径**。所以在 ``observe.py`` 之外，这里必须**再验一遍**
隐藏信息防线：``tests/test_featurize.py`` 会断言"重掷隐藏随机状态后编码逐位不变"、
"抽牌堆顺序反转后编码逐位不变"。投影对了但特征化漏了，等于没防。

设计对齐 ``docs/03`` §3.2：

  * 每个游戏对象一个 token：``token_type`` / ``token_entity``(id) / ``token_num``(连续特征)
  * 变长用 ``token_mask`` 屏蔽
  * 牌堆（抽/弃/消耗）用**多重集计数向量** ``pile_counts``，交给网络做 bag 池化
    —— 计数向量在数学上不含顺序信息，这是"结构上不可能泄漏顺序"的实现

纯 numpy，不依赖 torch，便于单独测试与复用。
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .content import CARD_DB, ENEMY_DB, POTIONS, RELICS
from .core import Action
from .observe import Observation

# --------------------------------------------------------------------------
# 词表（从内容表确定性构建；加新内容时索引会变，所以训练前要冻结并记录版本）
# --------------------------------------------------------------------------
CARD_IDS: list[str] = sorted(CARD_DB)
ENEMY_IDS: list[str] = sorted(ENEMY_DB)
RELIC_IDS: list[str] = sorted(RELICS)
POTION_IDS: list[str] = sorted(POTIONS)
CARD_VOCAB: dict[str, int] = {cid: i for i, cid in enumerate(CARD_IDS)}
ENEMY_VOCAB: dict[str, int] = {eid: i for i, eid in enumerate(ENEMY_IDS)}
RELIC_VOCAB: dict[str, int] = {rid: i for i, rid in enumerate(RELIC_IDS)}
POTION_VOCAB: dict[str, int] = {pid: i for i, pid in enumerate(POTION_IDS)}
#: 能力 id 词表：**先占位**，由 :func:`rebuild_vocab` 现算填入
#: （能力的来源是引擎注册表 + 内容表，不能在 import 时定死 —— 那时内容还没加载）。
POWER_VOCAB: dict[str, int] = {}

#: 充能球 id 词表。球是**角色机制**，不在内容目录里，所以来自引擎注册表。
from .orbs import ORB_DEFS  # noqa: E402  （放在这里避免循环导入）

ORB_IDS: list[str] = sorted(ORB_DEFS)
ORB_VOCAB: dict[str, int] = {oid: i for i, oid in enumerate(ORB_IDS)}

#: 已知能力（骨架版只做这三种；真实内容需要扩展，见 docs/03 §3.8 的坑）
POWER_NAMES: tuple[str, ...] = ("strength", "vulnerable", "weak")

#: ⭐ **能力实体的词表**（外部复核 R4）。
#:
#: 原先张量化只把 ``strength`` / ``vulnerable`` / ``weak`` 三个量写进固定列，
#: 于是引擎已经实现的 197 个能力里，其余 190 来个对模型**完全不存在** ——
#: 实测"给玩家加 9 层中毒"前后编码数组一字不差，而中毒是公开信息、
#: 也是玩家决策的核心依据之一（球槽 +2 同理）。
#:
#: 词表取**能力注册表**（``powers.RULES``，引擎真正实现了行为的那些）
#: 与内容表 ``content.POWERS`` 的并集：前者保证"引擎在结算的量"一定查得到，
#: 后者兼容"内容里有、引擎当纯数据"的那些。用 ``clear/update`` 原地更新
#: （理由见 :func:`rebuild_vocab`）。
POWER_IDS: list[str] = []


def _known_power_ids() -> list[str]:
    from .content import POWERS, POWER_NAMES as _NAMES
    from .powers import RULES
    ids = set(POWERS) | set(RULES) | set(_NAMES)
    return sorted(ids)

#: ⭐ 动作类型词表。审计 F09：药水一直是"可调用的效果函数"而不是动作，
#: 于策略而言不存在。补 `use_potion` 之后策略才可能学会留药/交药。
ACTION_KINDS: tuple[str, ...] = ("play_card", "end_turn", "restart", "accept",
                                 "select_card", "use_potion")
ACTION_KIND_VOCAB: dict[str, int] = {k: i for i, k in enumerate(ACTION_KINDS)}

# --------------------------------------------------------------------------
# 布局常量
# --------------------------------------------------------------------------
MAX_HAND = 10
MAX_ENEMIES = 5
MAX_RELICS = 25
MAX_POTIONS = 3
MAX_ORBS = 10
#: 能力 token 的容量：玩家 + 全部敌人身上的能力**合计**。
#: 取 40 而不是"猜一个"：一只 Boss 可以同时挂十来种能力，玩家这边
#: （姿态/护盾/中毒/诅咒…）也能上两位数；容量不足**报错**而不是截断
#: （见 :func:`_check_capacity`）。
MAX_POWERS = 40
#: SL：跨试次已揭示的牌（"上一遍我第 1 回合抽到了什么"）
MAX_REVEALED = 128
NUM_TOKEN_TYPES = 8

(TOKEN_GLOBAL, TOKEN_HAND, TOKEN_ENEMY, TOKEN_RELIC, TOKEN_POTION,
 TOKEN_REVEALED, TOKEN_ORB, TOKEN_POWER) = range(8)
#: 已揭示 token 在 token 序列里的**起始下标**（按 type id 切片是错的——那是类型不是位置）
REVEALED_OFFSET = 1 + MAX_HAND + MAX_ENEMIES + MAX_RELICS + MAX_POTIONS + MAX_ORBS
#: 能力 token 排在**最后**：新增 token 一律追加在末尾，
#: 这样 `REVEALED_OFFSET` 这类已算好的偏移不会变（与"新增列追加在末尾"同一理由）。
POWER_OFFSET = REVEALED_OFFSET + MAX_REVEALED
MAX_TOKENS = POWER_OFFSET + MAX_POWERS

MAX_ACTIONS = 64
NUM_PILES = 3                        # draw / discard / exhaust
#: 选牌候选的**硬上限**。选择界面的候选数不会超过"能同时给出的动作数"：
#: ``legal_actions`` 对每个候选**恰好**产生一个 ``select_card`` 动作，
#: 而动作总数已由 :data:`MAX_ACTIONS` 约束。所以候选下标必然 < ``MAX_ACTIONS``。
#: 取这个上界而不是"猜一个经验值"（实测真实内容下弃牌堆可以有 17+ 张，
#: 猜小了会让合法局面在训练中途报错）。
MAX_CANDIDATES = MAX_ACTIONS
#: 动作槽位/下标的编码上限。取各来源里的最大值。
MAX_ACTION_SLOTS = max(MAX_HAND, MAX_CANDIDATES)

#: ``token_num`` 的列含义。**同一列在不同 token 类型下含义不同**（紧凑布局），
#: 具体分配见下方 ``FEATURE_USAGE``。
#:
#: ⚠️ 新增列一律**追加在末尾**（审计 F04 补的是 23 起那几列）：
#: 中间插列会让所有按数字下标写的位置（``numbers[i, 5]`` 之类）静默错位。
FEATURE_LAYOUT: tuple[str, ...] = (
    "hp_ratio", "hp_x100", "block_x50", "energy_x3", "turn_x15",
    "cost_x3", "upgraded", "playable",
    "intent_kind", "intent_value_x30", "intent_times_x3",
    "strength", "vulnerable", "weak",
    "slot_pos", "alive",
    "attempts_left_x4", "attempts_used_x4", "has_knowledge", "n_revealed_x32",
    # 选牌状态（人类屏幕上写着"选择一张牌弃置"，模型也必须看得到）
    "selecting", "select_purpose", "select_remaining_x4",
    # ⭐ 审计 F04：这些机制引擎在结算，而模型完全看不到
    "stars_x10",             # 23 星资源（摄政王）
    "orb_count_x5",          # 24 充能球数量
    "max_energy_x3",         # 25 每回合能量上限
    "select_source",         # 26 选牌来源牌堆（hand/discard/draw）
    "n_candidates_x8",       # 27 候选张数
    "relic_count_x16",       # 28 遗物数量
    "potion_count_x4",       # 29 药水数量
    # ⭐ 外部复核 R4：以下三列补的是"引擎在结算、而模型看不见"的公开状态
    "orb_slots_x5",          # 30 **空球槽**（充能球的容量，屏幕上画着空槽）
    "power_amount_x10",      # 31 能力 token 的层数
    "power_on_player",       # 32 该能力挂在玩家(1)还是敌人(0)身上
)
#: ⚠️ 由**布局推导**，不要再手写一个数字：两处独立维护必然对不上
#: （实测加了三列选牌特征后 `NUM_FEATURES` 仍是 20，整个模块 import 就炸了）。
NUM_FEATURES = len(FEATURE_LAYOUT)

#: 每种 token 实际用到哪些列（写出来是为了避免"某列到底谁在用"的混乱）
FEATURE_USAGE: dict[str, tuple[str, ...]] = {
    "global": ("hp_ratio", "hp_x100", "block_x50", "energy_x3", "turn_x15",
               "strength", "vulnerable", "weak",
               "attempts_left_x4", "attempts_used_x4", "has_knowledge", "n_revealed_x32",
               "selecting", "select_purpose", "select_remaining_x4",
               "stars_x10", "orb_count_x5", "max_energy_x3",
               "select_source", "n_candidates_x8", "relic_count_x16",
               "potion_count_x4", "orb_slots_x5"),
    "hand": ("cost_x3", "upgraded", "playable", "slot_pos", "alive"),
    "enemy": ("hp_ratio", "hp_x100", "block_x50", "intent_kind", "intent_value_x30",
              "intent_times_x3", "strength", "vulnerable", "weak", "slot_pos", "alive"),
    # 遗物 / 药水：身份走 token_entity 的各自词表，这里只补一个槽位与"存在"标志
    "relic": ("slot_pos", "alive"),
    "potion": ("slot_pos", "cost_x3", "alive"),
    "orb": ("intent_value_x30", "slot_pos", "alive"),
    # 已揭示的牌：用 attempt_index 与槽位编码"这是第几遍、第几张"
    "revealed": ("cost_x3", "slot_pos", "alive"),
    # 能力实体：层数 + 挂在谁身上（`slot_pos` 对敌人是它的槽位，对玩家恒 0）
    "power": ("power_amount_x10", "slot_pos", "power_on_player"),
}

INTENT_KINDS: tuple[str, ...] = ("attack", "attack_debuff", "block", "buff", "debuff",
                                 "unknown")
INTENT_VOCAB: dict[str, int] = {k: i for i, k in enumerate(INTENT_KINDS)}

#: 选牌用途词表（与 ``content.ENGINE_SELECTION_PURPOSES`` 对齐）。
SELECT_PURPOSES: tuple[str, ...] = ("", "discard", "exhaust", "to_hand", "to_draw")
#: 选牌来源牌堆词表（审计 F09：候选来自哪个牌堆是公开信息，必须能区分）。
SELECT_SOURCES: tuple[str, ...] = ("", "hand", "discard", "draw")


def _check_capacity(name: str, count: int, capacity: int) -> None:
    """容量溢出**必须报错**（``docs/13`` §8.2）。

    静默截断会让策略看到的世界**比真实世界小**，而且是系统性的：
    手牌 12 张时它只看到前 10 张，于是"少打了一张牌"这件事没有任何信号。
    报错才能逼出显式扩容或明确的课程上限。
    """
    if count > capacity:
        raise ValueError(
            f"{name} 数量 {count} 超过编码容量 {capacity}；"
            f"请显式扩容或限制课程范围（不允许静默截断）")


# --------------------------------------------------------------------------
# 编码
# --------------------------------------------------------------------------
def _blank_tokens() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(MAX_TOKENS, dtype=np.int64),      # type
        np.full(MAX_TOKENS, -1, dtype=np.int64),   # entity id
        np.zeros((MAX_TOKENS, NUM_FEATURES), dtype=np.float32),
        np.zeros(MAX_TOKENS, dtype=np.float32),    # mask
    )


def _power_amount(powers: tuple[tuple[str, int], ...], name: str) -> float:
    for key, value in powers:
        if key == name:
            return float(value)
    return 0.0


def encode_observation(obs: Observation,
                       legal: Sequence[Action]) -> dict[str, np.ndarray]:
    """把一次决策编码成网络输入。

    只读取 ``Observation`` 的字段——**不接触任何 state / hidden**。
    """
    # ⭐ 容量检查放在最前：溢出必须**报错**，不能静默截断（docs/13 §8.2）。
    _check_capacity("手牌", len(obs.hand), MAX_HAND)
    _check_capacity("敌人", len(obs.enemies), MAX_ENEMIES)
    _check_capacity("遗物", len(obs.relics), MAX_RELICS)
    _check_capacity("药水", len(obs.potions), MAX_POTIONS)
    _check_capacity("充能球", len(obs.orbs), MAX_ORBS)
    n_revealed = sum(len(h) for a in obs.prior_attempts for h in a.hand_by_turn)
    _check_capacity("SL 已揭示牌", n_revealed, MAX_REVEALED)
    _check_capacity("能力",
                    len(obs.player_powers)
                    + sum(len(e.powers) for e in obs.enemies), MAX_POWERS)

    types, entities, numbers, mask = _blank_tokens()
    cursor = 0

    # --- 全局 token ---------------------------------------------------
    types[cursor] = TOKEN_GLOBAL
    numbers[cursor] = _global_features(obs)
    mask[cursor] = 1.0
    cursor += 1

    # --- 手牌 ---------------------------------------------------------
    for slot, card in enumerate(obs.hand):
        index = cursor + slot
        types[index] = TOKEN_HAND
        entities[index] = CARD_VOCAB.get(card.cid, -1)
        numbers[index, 5] = card.cost / 3.0
        numbers[index, 6] = float(card.upgraded)
        numbers[index, 7] = float(card.playable)
        numbers[index, 14] = slot / 10.0
        numbers[index, 15] = 1.0
        mask[index] = 1.0
    cursor += MAX_HAND

    # --- 敌人 ---------------------------------------------------------
    for slot, enemy in enumerate(obs.enemies):
        index = cursor + slot
        types[index] = TOKEN_ENEMY
        entities[index] = ENEMY_VOCAB.get(enemy.eid, -1)
        numbers[index, 0] = enemy.hp / max(1, enemy.max_hp)
        numbers[index, 1] = enemy.hp / 100.0
        numbers[index, 2] = enemy.block / 50.0
        numbers[index, 8] = INTENT_VOCAB.get(enemy.intent_kind, len(INTENT_KINDS) - 1) / 8.0
        numbers[index, 9] = enemy.intent_value / 30.0
        numbers[index, 10] = enemy.intent_times / 3.0
        numbers[index, 11] = _power_amount(enemy.powers, "strength") / 10.0
        numbers[index, 12] = _power_amount(enemy.powers, "vulnerable") / 10.0
        numbers[index, 13] = _power_amount(enemy.powers, "weak") / 10.0
        numbers[index, 14] = slot / 5.0
        numbers[index, 15] = float(enemy.hp > 0)
        mask[index] = 1.0
    cursor += MAX_ENEMIES

    # --- ⭐ 遗物（审计 F04：以前是**空占位**，遗物对模型不存在）--------
    for slot, rid in enumerate(obs.relics):
        index = cursor + slot
        types[index] = TOKEN_RELIC
        entities[index] = RELIC_VOCAB.get(rid, -1)
        numbers[index, 14] = slot / 25.0
        numbers[index, 15] = 1.0
        mask[index] = 1.0
    cursor += MAX_RELICS

    # --- ⭐ 药水（同上；药水的**槽位**也是公开信息）--------------------
    for slot, pid in obs.potions:
        if not 0 <= slot < MAX_POTIONS:
            continue
        index = cursor + slot
        types[index] = TOKEN_POTION
        entities[index] = POTION_VOCAB.get(pid, -1)
        numbers[index, 5] = slot / 3.0
        numbers[index, 14] = slot / 3.0
        numbers[index, 15] = 1.0
        mask[index] = 1.0
    cursor += MAX_POTIONS

    # --- ⭐ 充能球（审计 F04：球在结算，模型却看不到）------------------
    for slot, orb in enumerate(obs.orbs):
        index = cursor + slot
        types[index] = TOKEN_ORB
        entities[index] = ORB_VOCAB.get(orb.oid, -1)
        numbers[index, 9] = orb.value / 30.0
        numbers[index, 14] = slot / 5.0
        numbers[index, 15] = 1.0
        mask[index] = 1.0
    cursor += MAX_ORBS

    # --- ⭐ SL：跨试次已揭示的牌 --------------------------------------
    # 这是"重开"能带来收益的**唯一信息通道**（docs/01 §1.4）。
    # 手牌顺序 = 抽牌顺序，所以历次试次每回合的开局手牌拼起来，就是被观察到的牌序。
    revealed_slot = 0
    for attempt in obs.prior_attempts:
        for turn_index, hand in enumerate(attempt.hand_by_turn):
            for cid in hand:
                index = cursor + revealed_slot
                types[index] = TOKEN_REVEALED
                entities[index] = CARD_VOCAB.get(cid, -1)
                numbers[index, 5] = attempt.index / 4.0
                numbers[index, 14] = (turn_index * 10 + revealed_slot % 10) / 100.0
                numbers[index, 15] = 1.0
                mask[index] = 1.0
                revealed_slot += 1
    cursor += MAX_REVEALED

    # --- ⭐ 能力实体（外部复核 R4）-----------------------------------
    #
    # 为什么要有这一段：能力量原先只写进 `strength`/`vulnerable`/`weak` 三列，
    # 于是"给玩家加 9 层中毒"前后**编码数组一字不差** —— 而中毒是公开信息，
    # 也是玩家该不该继续出牌的核心依据。引擎已经实现了 197 个能力，
    # 张量化却只看得见 3 个。
    #
    # 编码方式与遗物/球一致：**一种能力一个 token**，身份走 `POWER_VOCAB`
    # 的各自下标空间，层数进 `power_amount_x10`，`power_on_player` 区分挂在谁身上
    # （敌人的 `slot_pos` 填它的槽位；玩家的恒 0）。
    power_tokens = [(name, amount, 1.0, 0.0)
                    for name, amount in obs.player_powers]
    for slot, enemy in enumerate(obs.enemies):
        power_tokens.extend((name, amount, 0.0, slot / 5.0)
                            for name, amount in enemy.powers)
    _check_capacity("能力", len(power_tokens), MAX_POWERS)
    for offset, (name, amount, on_player, slot_pos) in enumerate(power_tokens):
        index = cursor + offset
        types[index] = TOKEN_POWER
        entities[index] = POWER_VOCAB.get(name, -1)
        numbers[index, 31] = float(amount) / 10.0
        numbers[index, 32] = on_player
        numbers[index, 14] = slot_pos
        mask[index] = 1.0
    cursor += MAX_POWERS

    # --- 牌堆（多重集计数，无顺序）------------------------------------
    # ⚠️ **升级状态必须分开编码**（审计 F04）：牌堆编码以前忽略 `upgraded`，
    # 于是"牌堆里那张是升级过的打击"和"没升级的打击"在特征里完全一样 ——
    # 而它们打出来差 3 点伤害。
    pile_counts = np.zeros((NUM_PILES, len(CARD_IDS)), dtype=np.float32)
    pile_upgraded = np.zeros((NUM_PILES, len(CARD_IDS)), dtype=np.float32)
    for row, bag in enumerate((obs.draw_bag, obs.discard_bag, obs.exhaust_bag)):
        _fill_pile(pile_counts[row], pile_upgraded[row], bag.items)

    # --- 动作 ---------------------------------------------------------
    act_kind = np.zeros(MAX_ACTIONS, dtype=np.int64)
    act_card = np.full(MAX_ACTIONS, -1, dtype=np.int64)
    act_slot = np.full(MAX_ACTIONS, -1, dtype=np.int64)
    act_target = np.full(MAX_ACTIONS, -1, dtype=np.int64)
    act_item = np.full(MAX_ACTIONS, -1, dtype=np.int64)
    act_mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
    if len(legal) > MAX_ACTIONS:
        raise ValueError(f"合法动作数 {len(legal)} 超过 MAX_ACTIONS={MAX_ACTIONS}")
    for i, action in enumerate(legal):
        act_kind[i] = ACTION_KIND_VOCAB.get(action.kind, 0)
        act_slot[i] = action.hand_index
        act_target[i] = action.target
        # 出牌指向手牌里的一张：编码卡牌身份。
        if action.kind == "play_card" and 0 <= action.hand_index < len(obs.hand):
            act_card[i] = CARD_VOCAB.get(obs.hand[action.hand_index].cid, -1)
        # ⭐ 选牌指向**候选列表**里的第 N 个，而不是手牌的第 N 张（审计 F09）。
        # 来源可能是弃牌堆/抽牌堆，用 `obs.hand[hand_index]` 会张冠李戴。
        elif action.kind == "select_card" and obs.selection is not None:
            candidates = obs.selection.candidates
            if 0 <= action.hand_index < len(candidates):
                act_card[i] = CARD_VOCAB.get(candidates[action.hand_index].cid, -1)
        # 药水：**身份**走 `act_item`（自己的词表），槽位走 `act_slot`。
        # ⚠️ 曾经把药水身份塞进 `act_card` —— 那会让药水下标去查卡牌 embedding 表，
        # 两张表大小不同，越界时直接 IndexError。
        elif action.kind == "use_potion":
            act_slot[i] = action.slot
            if 0 <= action.slot < len(obs.potions):
                act_item[i] = POTION_VOCAB.get(obs.potions[action.slot][1], -1)
        # 容量守卫：动作侧的越界同样必须**报错**，不能留给 embedding 抛 IndexError
        # （那条错误信息不会告诉你是哪个动作、哪个字段）。
        if act_slot[i] >= MAX_ACTION_SLOTS:
            raise ValueError(
                f"动作下标 {act_slot[i]} 超过编码容量 {MAX_ACTION_SLOTS}"
                f"（动作 {action!r}，不允许静默截断）")
        act_mask[i] = 1.0

    return {
        "token_type": types,
        "token_entity": entities,
        "token_num": numbers,
        "token_mask": mask,
        "pile_counts": pile_counts,
        "pile_upgraded": pile_upgraded,
        "revealed_counts": _revealed_counts(obs),
        "act_kind": act_kind,
        "act_card": act_card,
        "act_slot": act_slot,
        "act_target": act_target,
        "act_item": act_item,
        "act_mask": act_mask,
        "n_actions": np.int64(len(legal)),
    }


def _revealed_counts(obs: Observation) -> np.ndarray:
    """已揭示牌的**按卡牌汇总**计数（供 bag 池化，与 revealed token 互补）。

    token 形式保留"第几遍、第几张"的顺序；计数形式给网络一个置换不变的概览。
    两者都不含未揭示信息。
    """
    counts = np.zeros(len(CARD_IDS), dtype=np.float32)
    for attempt in obs.prior_attempts:
        for hand in attempt.hand_by_turn:
            for cid in hand:
                index = CARD_VOCAB.get(cid)
                if index is not None:
                    counts[index] += 1.0
    return counts


def _global_features(obs: Observation) -> np.ndarray:
    out = np.zeros(NUM_FEATURES, dtype=np.float32)
    out[0] = obs.player_hp / max(1, obs.player_max_hp)
    out[1] = obs.player_hp / 100.0
    out[2] = obs.player_block / 50.0
    out[3] = obs.energy / 3.0
    out[4] = obs.turn / 15.0
    out[11] = _power_amount(obs.player_powers, "strength") / 10.0
    out[12] = _power_amount(obs.player_powers, "vulnerable") / 10.0
    out[13] = _power_amount(obs.player_powers, "weak") / 10.0
    out[14] = len(obs.hand) / 10.0
    out[15] = 1.0
    # ⭐ SL 预算与知识量：策略必须知道"还能重开几次""我已经知道了多少"
    out[16] = obs.attempts_left / 4.0
    out[17] = obs.attempts_used / 4.0
    out[18] = 1.0 if obs.prior_attempts else 0.0
    out[19] = sum(len(h) for a in obs.prior_attempts for h in a.hand_by_turn) / 32.0
    # 选牌状态：不编码的话，模型在"选牌"与"出牌"两种局面下拿到的特征几乎一样，
    # 只能靠动作掩码的形状去猜，而掩码形状随候选数量变化 —— 那是让模型猜一个
    # 本可以直接告诉它的东西。
    out[20] = 1.0 if obs.selecting_purpose else 0.0
    out[21] = SELECT_PURPOSES.index(obs.selecting_purpose) / max(1, len(SELECT_PURPOSES) - 1) \
        if obs.selecting_purpose in SELECT_PURPOSES else 0.0
    out[22] = obs.selecting_remaining / 4.0
    # ⭐ 审计 F04 补的公开状态：星、球、能量上限、选牌来源与候选数、遗物/药水数量
    out[23] = obs.stars / 10.0
    out[24] = len(obs.orbs) / 5.0
    out[25] = obs.player_max_energy / 3.0
    source = obs.selection.source_pile if obs.selection else ""
    out[26] = (SELECT_SOURCES.index(source) / max(1, len(SELECT_SOURCES) - 1)
               if source in SELECT_SOURCES else 0.0)
    out[27] = (len(obs.selection.candidates) if obs.selection else 0) / 8.0
    out[28] = len(obs.relics) / 16.0
    out[29] = len(obs.potions) / 4.0
    # ⭐ 外部复核 R4：**空球槽**也是公开的（屏幕上画着几个空槽），
    # 而 `orb_count_x5` 只数了已充能的球。少了这一列，
    # "球槽 +2 但都是空的"与"没加槽"对模型完全一样。
    out[30] = obs.orb_slots / 5.0
    return out


def _fill_pile(row: np.ndarray, upgraded_row: np.ndarray,
               items: tuple[tuple[str, bool, int], ...]) -> None:
    """填一行牌堆计数：``row`` 收全部，``upgraded_row`` 只收升级过的。

    ⚠️ 两者是"总数"与"其中升级过的"的关系，不是互斥拆分 ——
    这样 ``pile_counts.sum()`` 仍然是牌堆总张数（既有测试依赖这条不变量）。
    """
    for cid, upgraded, count in items:
        index = CARD_VOCAB.get(cid)
        if index is None:
            continue
        row[index] += count
        if upgraded:
            upgraded_row[index] += count


# --------------------------------------------------------------------------
# 词表重建（内容表变化时必须调用，否则 embedding 下标会张冠李戴）
# --------------------------------------------------------------------------
def rebuild_vocab() -> dict[str, int]:
    """按当前 ``content.CARD_DB`` / ``ENEMY_DB`` 重建词表。

    ⚠️ **原地更新，不重新绑定名字。** 别处写的是 ``from .featurize import CARD_VOCAB``，
    那是把**对象**绑进了对方的命名空间；这里若 ``CARD_VOCAB = {...}`` 重新绑定，
    对方手里的仍是旧字典——于是内容换了、词表没换，模型静默张冠李戴。
    这是本项目里最隐蔽的一类 bug，所以统一用 clear/update。

    ⚠️ 换内容表之后**必须重新训练**：卡牌 embedding 按词表下标查表，下标变了而权重
    没变，A 牌就被当成了 B 牌。``vocab_fingerprint()`` 用来把这个版本钉进 checkpoint。
    """
    CARD_IDS.clear()
    CARD_IDS.extend(sorted(CARD_DB))
    ENEMY_IDS.clear()
    ENEMY_IDS.extend(sorted(ENEMY_DB))
    RELIC_IDS.clear()
    RELIC_IDS.extend(sorted(RELICS))
    POTION_IDS.clear()
    POTION_IDS.extend(sorted(POTIONS))
    CARD_VOCAB.clear()
    CARD_VOCAB.update({cid: i for i, cid in enumerate(CARD_IDS)})
    ENEMY_VOCAB.clear()
    ENEMY_VOCAB.update({eid: i for i, eid in enumerate(ENEMY_IDS)})
    RELIC_VOCAB.clear()
    RELIC_VOCAB.update({rid: i for i, rid in enumerate(RELIC_IDS)})
    POTION_VOCAB.clear()
    POTION_VOCAB.update({pid: i for i, pid in enumerate(POTION_IDS)})
    POWER_IDS.clear()
    POWER_IDS.extend(_known_power_ids())
    POWER_VOCAB.clear()
    POWER_VOCAB.update({pid: i for i, pid in enumerate(POWER_IDS)})
    return vocab_fingerprint()


def vocab_fingerprint() -> dict[str, int]:
    """内容版本指纹：存进 checkpoint，加载时比对，防止词表错位。"""
    digest = hashlib.sha256(
        "\n".join([*CARD_IDS, "|", *ENEMY_IDS, "|", *RELIC_IDS, "|",
                   *POTION_IDS, "|", *ORB_IDS, "|", *POWER_IDS]).encode("utf-8"))
    return {
        "n_cards": len(CARD_IDS),
        "n_enemies": len(ENEMY_IDS),
        "n_relics": len(RELIC_IDS),
        "n_potions": len(POTION_IDS),
        "n_powers": len(POWER_IDS),
        "vocab_hash": int.from_bytes(digest.digest()[:8], "big"),
    }


def configure_content(path) -> dict:
    """加载 JSON 内容表并重建词表。**必须在构建模型之前调用。**"""
    from .content import load_content_dir
    info = load_content_dir(path)
    vocab = rebuild_vocab()          # ← 别漏：内容换了词表必须跟着换
    info.update(vocab)
    return info


def batch_encode(pairs: Sequence[tuple[Observation, Sequence[Action]]]
                 ) -> dict[str, np.ndarray]:
    """把一批观测堆成 batch（训练采样用）。"""
    encoded = [encode_observation(obs, legal) for obs, legal in pairs]
    keys = encoded[0].keys()
    out: dict[str, np.ndarray] = {}
    for key in keys:
        if key == "n_actions":
            out[key] = np.array([e[key] for e in encoded], dtype=np.int64)
        else:
            out[key] = np.stack([e[key] for e in encoded])
    return out


def shapes() -> dict[str, tuple[int, ...]]:
    """当前张量形状。**用函数而不是常量**——内容表变了形状就变，
    常量会在导入时被固化，导致它悄悄描述的是旧内容。"""
    return {
        "token_type": (MAX_TOKENS,),
        "token_entity": (MAX_TOKENS,),
        "token_num": (MAX_TOKENS, NUM_FEATURES),
        "token_mask": (MAX_TOKENS,),
        "pile_counts": (NUM_PILES, len(CARD_IDS)),
        "pile_upgraded": (NUM_PILES, len(CARD_IDS)),
        "revealed_counts": (len(CARD_IDS),),
        "act_kind": (MAX_ACTIONS,),
        "act_card": (MAX_ACTIONS,),
        "act_slot": (MAX_ACTIONS,),
        "act_target": (MAX_ACTIONS,),
        "act_item": (MAX_ACTIONS,),
        "act_mask": (MAX_ACTIONS,),
    }
