"""内容准入与依赖闭包（``docs/13`` §7 / 工单 T01）。

审计 F03 的结论是：``draft_pool()`` 只筛"稀有度 + 有没有效果"，于是 418 张里
混进 223 张 ``effects_incomplete``；``random_deck()`` 又会跨角色混牌。这等于
**采样器没有执行内容门禁**，而 ``content_status.py`` 报告的"可训练"却是按门禁口径
统计的 —— 两份数字对不上，训练拿到的是另一个分布。

本模块把"哪些内容能进训练"变成**唯一一处、可查询、可测试、可复现**的定义。
``docs/13`` §7 要求区分六个状态，而不是一个 ``verified`` 布尔值：

============ ==========================================================
状态          证明的事情
============ ==========================================================
已索引        找到类、ID、静态定义和来源
已解析        必需表达式与效果都被识别，无未解释片段
可执行        所需算子、钩子、子系统有实现
闭包可执行    可能生成、召唤、转化到的对象也满足准入
源码用例通过  针对性输入及交互与源码依据一致
参考验证通过  独立执行/真机轨迹在声明范围内通过
============ ==========================================================

当前实现覆盖前四项（**已索引 → 闭包可执行**）。后两项需要真机对拍，
``data/reference_traces/`` 到位之前一律为 ``False``，不得用前四项冒充。

⚠️ 执行资格是**内容与场景共同决定**的（``docs/13`` §7）：一张卡支持基础版本
不代表升级版本也支持；怪物状态机正常不代表召唤物正常。所以这里按
**卡（含升级）+ 场景（角色/房间）**分别给结论。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

from .content import (
    CARD_DB, CARD_POOLS, ENCOUNTERS, ENEMY_DB, ENGINE_CALC_KINDS, ENGINE_OPS,
    ENGINE_SELECTION_PURPOSES, ENGINE_TARGETS, ENGINE_TRIGGER_HOOKS,
    POTIONS, RELICS, character_pool,
)

# ==========================================================================
# 状态
# ==========================================================================
S_INDEXED = "indexed"
S_PARSED = "parsed"
S_EXECUTABLE = "executable"
S_CLOSURE = "closure"
S_SOURCE_CASE = "source_case"
S_REFERENCE = "reference"

#: 上一阶段成立才有下一阶段成立（``docs/13`` §7 的六层）
STATUS_ORDER: tuple[str, ...] = (
    S_INDEXED, S_PARSED, S_EXECUTABLE, S_CLOSURE, S_SOURCE_CASE, S_REFERENCE,
)

#: 能进奖励/商店/选牌池的稀有度。诅咒/状态/任务牌**不可抽**，
#: 它们靠"不可打出 + 占手牌"起作用，不是课程分布的一部分。
DRAFTABLE_RARITIES: frozenset[str] = frozenset({"common", "uncommon", "rare"})

# --------------------------------------------------------------------------
# 拒绝原因：**可数、可排序、可报告**，不许出现"其它"
# --------------------------------------------------------------------------
R_UNKNOWN = "unknown_card"                     # 引用了不存在的卡
R_INCOMPLETE = "effects_incomplete"            # 源码效果没解析全（审计 F03 的主因）
R_TRIGGERS = "triggers_incomplete"             # 手牌触发器缺钩子
R_NO_EFFECT = "no_effects"                     # **可打出**却什么都不做（解析缺口）
R_CLOG = "clog"                                # 不可打出的诅咒/状态牌（**不是缺口**）
R_TEXT_ONLY = "effects_from_community_text"    # 效果只来自社区文本（非权威来源）
R_BAD_OP = "unsupported_op"                    # 引擎没有这个算子
R_BAD_POWER = "unimplemented_power"            # 引用了引擎没实现的能力
R_BAD_SELECT = "unsupported_selection"         # 未实现的选牌用途/来源
R_BAD_TARGET = "unsupported_target"
R_BAD_TRIGGER = "unsupported_trigger"          # 未实现的手牌触发器钩子
R_MISSING_GEN = "generates_unknown_card"       # 生成/引用了不存在的卡
R_MISSING_RELIC = "references_unknown_relic"
R_CLOSURE = "closure_gap"                      # 依赖闭包里有不合格对象
R_EMPTY_UPGRADE = "upgrade_not_supported"      # 升级版与基础版效果不一致
#: **单人 profile 遇不到**：``CardMultiplayerConstraint.MultiplayerOnly``。
#: 不是"引擎做不到"，而是"这个 profile 里不存在这张牌"（`docs/12` §2.32）。
R_MULTIPLAYER = "multiplayer_only"
#: **运行期公式**认得出来、但引擎算不对（`docs/12` §2.33）。
#: 与 ``R_BAD_OP`` 同理：认出来却算不了，必须拒绝而不是算成 0。
R_BAD_CALC = "unsupported_calc"


# ==========================================================================
# 卡牌
# ==========================================================================
def _effect_reasons(effects: Iterable, where: str) -> list[str]:
    """逐条效果检查**算子级 / 能力级**可执行性。"""
    from .powers import IMPLEMENTED

    reasons: list[str] = []
    for effect in effects:
        op = getattr(effect, "op", "")
        # ⭐ 运行期公式：认出来但引擎算不了 → 拒绝（**不能**算成 0）。
        calc_kind = getattr(effect, "calc_kind", "")
        if calc_kind and calc_kind not in ENGINE_CALC_KINDS:
            reasons.append(f"{R_BAD_CALC}:{calc_kind}@{where}")
        if op not in ENGINE_OPS:
            reasons.append(f"{R_BAD_OP}:{op}@{where}")
            continue
        if op == "select_card":
            if effect.purpose not in ENGINE_SELECTION_PURPOSES:
                reasons.append(f"{R_BAD_SELECT}:{effect.purpose}@{where}")
            if effect.select_from not in ("hand", "discard", "draw"):
                reasons.append(f"{R_BAD_SELECT}:{effect.select_from}@{where}")
        if op == "apply_power":
            power = effect.power or ""
            if not power:
                reasons.append(f"{R_BAD_POWER}:<empty>@{where}")
            elif power not in IMPLEMENTED:
                reasons.append(f"{R_BAD_POWER}:{power}@{where}")
        if op == "add_card":
            card = effect.card or ""
            if not card:
                reasons.append(f"{R_MISSING_GEN}:<empty>@{where}")
            elif card not in CARD_DB:
                reasons.append(f"{R_MISSING_GEN}:{card}@{where}")
        if op == "channel" and not effect.orb:
            reasons.append(f"{R_BAD_OP}:channel-no-orb@{where}")
        if getattr(effect, "target", "") not in ENGINE_TARGETS | {"all_enemies"}:
            reasons.append(f"{R_BAD_TARGET}:{effect.target}@{where}")
    return reasons


def is_clog(card) -> bool:
    """"占手牌的废物牌"：真机靠**不可打出**起作用，本来就该没有效果。

    ⚠️ 这与"可打出的牌却什么都不做"是**两件事**（``docs/13`` §7）：
    前者是设计，后者是解析缺口。混在一起会让 97 张诅咒/状态牌被当成
    引擎缺陷，或者反过来让真正缺效果的牌静默过关。
    """
    from . import keywords as keyword_rules
    if keyword_rules.is_unplayable(card):
        return True
    return card.cost < 0 and not card.is_x_cost


def card_reasons(cid: str) -> tuple[str, ...]:
    """这张卡**自身**（含升级版）为什么不能进训练。空元组 = 合格。

    不含闭包检查 —— 闭包是 :func:`closure_reasons` 的事，
    两者分开才能回答"是这张卡不全，还是它生成的牌不全"。
    """
    card = CARD_DB.get(cid)
    if card is None:
        return (R_UNKNOWN,)

    reasons: list[str] = []
    # ⭐ **单人 profile 遇不到这张牌**：多人专用。这条要排在最前面 ——
    # 它解释的是"为什么不判它"，而不是"引擎哪里没做到"
    # （`docs/12` §2.31.3 测出 16 张多人专用卡混进了单人池）。
    if card.multiplayer_only:
        reasons.append(R_MULTIPLAYER)
    if card.effects_incomplete:
        reasons.append(R_INCOMPLETE)
    if card.triggers_incomplete:
        reasons.append(R_TRIGGERS)

    # ⭐ **效果必须来自反编译源码**（docs/09 铁律 1：源码是机制的唯一真相，
    # 社区库只提供 id / 类名 / 结构，**不提供数值**）。
    #
    # 社区文本解析出来的效果只有两个来源：那张卡在源码里**根本不存在**
    # （社区库超前于本地构建），或者源码记录被拒绝却没标 `effects_incomplete`。
    # 两种都不该进训练集 —— 它们的数字没有任何权威依据。
    #
    # 实测抓到 `scare`（惊吓）：本地反编译源码里没有 `Scare.cs`，
    # 它的"施加 1 层虚弱"完全来自社区文本，却被判成可训练。
    #
    # ⚠️ 诅咒/状态牌（clog）不走这条：它们本来就该没有效果，
    # "从文本解析"对它们没有意义（实测 12 张）。
    #
    # ⚠️ 内置占位内容（`CONTENT_SOURCE == "builtin"`）也不走这条：那些卡是引擎
    # 自己手写的占位数据，`effect_source` 默认就是 `text`，与"社区库"无关。
    # 这条规则针对的是**真实内容**。
    from .content import CONTENT_SOURCE
    if (CONTENT_SOURCE != "builtin" and card.effect_source != "source"
            and not is_clog(card)):
        reasons.append(R_TEXT_ONLY)

    if not card.effects and not card.is_x_cost and not is_clog(card):
        reasons.append(R_NO_EFFECT)

    reasons.extend(_effect_reasons(card.effects, "base"))
    reasons.extend(_effect_reasons(card.upgrade, "upgrade"))
    for hook, trigger_effects in card.triggers:
        if hook not in ENGINE_TRIGGER_HOOKS:
            reasons.append(f"{R_BAD_TRIGGER}:{hook}")
        reasons.extend(_effect_reasons(trigger_effects, f"trigger:{hook}"))
    return tuple(dict.fromkeys(reasons))          # 去重且保序


def draftable(cid: str) -> bool:
    """能不能作为**奖励候选**出现（稀有度 + 可打出）。

    与"合格性"分开：合格说的是"引擎能不能忠实执行"，可抽说的是
    "这张牌在真机的奖励界面上会不会出现"。诅咒/状态牌合格但不可抽。
    """
    card = CARD_DB.get(cid)
    if card is None or card.rarity not in DRAFTABLE_RARITIES:
        return False
    return not is_clog(card)


def generated_cards(cid: str) -> tuple[str, ...]:
    """这张卡可能生成/引用到的其它卡 id（``add_card`` / ``select_card`` 目标）。"""
    card = CARD_DB.get(cid)
    if card is None:
        return ()
    out: list[str] = []
    for effect in (*card.effects, *card.upgrade,
                   *(e for _h, effs in card.triggers for e in effs)):
        if effect.op == "add_card" and effect.card:
            out.append(effect.card)
        if effect.op == "move_all_matching" and effect.card:
            out.append(effect.card)
    return tuple(dict.fromkeys(out))


def closure_gaps(cid: str, seen: frozenset[str] = frozenset()) -> tuple[str, ...]:
    """依赖闭包缺口：这张卡能生成、而生成物本身不合格或不存在。

    ``docs/13`` §7：**闭包可执行**要求"可能生成、召唤、转化到的对象也满足准入"。
    少了这一步，`anger` 合格但 `anger` 生成的牌不合格时就会静默放过。

    同时返回**生成物自身的缺口**（前缀 ``closure:``），便于一眼看出根因。
    """
    gaps: list[str] = []
    for generated in generated_cards(cid):
        if generated in seen:
            continue
        if generated not in CARD_DB:
            gaps.append(f"{R_MISSING_GEN}:{generated}")
            continue
        inner = card_reasons(generated)
        if inner:
            gaps.append(f"{R_CLOSURE}:{generated}←{inner[0]}")
            continue
        gaps.extend(f"{R_CLOSURE}:{g}" for g in closure_gaps(
            generated, seen | {cid, generated}))
    return tuple(dict.fromkeys(gaps))


@dataclass(frozen=True)
class CardAdmission:
    """一张卡的准入结论：**自身**与**闭包**分别可查。"""

    cid: str
    own: tuple[str, ...] = ()
    closure: tuple[str, ...] = ()

    @property
    def admitted(self) -> bool:
        return not self.own and not self.closure

    @property
    def status(self) -> str:
        """已到达的最深状态（``docs/13`` §7 的六层里前四层可自动判定）。

        后两层（源码用例、参考验证）需要真机证据，由 ``manifest`` 提供，
        这里**不假装**已经通过。
        """
        if self.cid not in CARD_DB:
            return "missing"
        if R_INCOMPLETE in self.own or R_TRIGGERS in self.own:
            return S_INDEXED
        if self.own:
            return S_PARSED
        if self.closure:
            return S_EXECUTABLE
        return S_CLOSURE

    def reasons(self) -> tuple[str, ...]:
        return self.own + self.closure


def card_admission(cid: str) -> CardAdmission:
    return CardAdmission(cid, card_reasons(cid), closure_gaps(cid))


# ==========================================================================
# 汇总：训练可用的内容
# ==========================================================================
_CACHE: dict[str, object] = {}
_CACHE_KEY: tuple = ()


def _content_key() -> tuple:
    """内容表指纹 —— 变了就作废缓存（内容可在运行期被替换）。"""
    return (len(CARD_DB), len(ENEMY_DB), len(CARD_POOLS), len(ENCOUNTERS),
            len(POTIONS), len(RELICS),
            sum(1 for c in CARD_DB.values() if c.effects_incomplete))


def _cached(name: str, factory):
    global _CACHE_KEY
    key = _content_key()
    if key != _CACHE_KEY:
        _CACHE.clear()
        _CACHE_KEY = key
    if name not in _CACHE:
        _CACHE[name] = factory()
    return _CACHE[name]


def admitted_cards() -> frozenset[str]:
    """全部合格卡（含不可抽的诅咒/状态 —— "合格"说的是可执行性）。"""
    return _cached("admitted", lambda: frozenset(
        cid for cid in CARD_DB if card_admission(cid).admitted))


def card_pool(character: str = "ironclad") -> tuple[str, ...]:
    """某个角色的**可抽**卡池（合格 + 可抽稀有度 + 在该角色池里）。

    ⚠️ 这是训练采样该用的池子。审计 F03：旧 ``draft_pool()`` 从**全体**卡里抽，
    铁甲战士会被给到静默 / 缺陷 / 摄政王的牌 —— 池子错了等于在另一个游戏里训练。
    """
    name = character_pool(character)
    members = CARD_POOLS.get(name) or ()
    return tuple(cid for cid in members if draftable(cid) and cid in admitted_cards())


def reward_pool(pool_name: str | None = None,
                character: str = "ironclad") -> tuple[str, ...]:
    """某个卡池里**可以发放**的卡（合格 + 可抽稀有度）。

    ⚠️ 这是发奖 / 商店 / 事件造牌**唯一**该用的池子。``run.draw_card_reward``
    原先直接从 ``CARD_POOLS`` 抽，**不过门禁** —— 实测 ``RngSet(0)`` 会发出
    ``demonic_shield``（引擎执行不了的牌）。那等于"模拟器给玩家一张它自己
    算不对的牌"，而且没有任何提示（外部复核 R3）。

    受限课程会**预先缩池**，因此抽到的稀有度分布与真机不同 —— 这是**声明过的**
    课程选择（训练清单里 ``draft_pool_coverage`` 记着覆盖率），不是静默近似。
    完整范围模式要的是"遇到缺失内容就记 ``unsupported``"，那由内容侧补齐，
    不能靠在发放之后重抽来掩盖。
    """
    name = pool_name or character_pool(character)
    members = CARD_POOLS.get(name) or ()
    return tuple(cid for cid in members if draftable(cid) and cid in admitted_cards())


def random_deck_cards(character: str = "ironclad") -> tuple[str, ...]:
    """某个角色的**基础牌组**（来自 characters.json，不是跨角色拼接）。"""
    from .content import CHARACTERS
    definition = CHARACTERS.get(character)
    if definition is not None:
        return tuple(definition.deck)
    from .content import STARTING_DECK
    return tuple(STARTING_DECK)


# --------------------------------------------------------------------------
# 遭遇 / 敌人
# --------------------------------------------------------------------------
def encounter_reasons(encounter: Sequence[str]) -> tuple[str, ...]:
    reasons: list[str] = []
    for eid in encounter:
        enemy = ENEMY_DB.get(eid)
        if enemy is None:
            reasons.append(f"unknown_enemy:{eid}")
            continue
        if enemy.hp is None:
            reasons.append(f"enemy_without_hp:{eid}")
        from .powers import IMPLEMENTED
        for pid, _amount in enemy.innate_powers:
            if pid not in IMPLEMENTED:
                reasons.append(f"innate_unimplemented:{eid}:{pid}")
    return tuple(dict.fromkeys(reasons))


def encounter_pool(kind: str = "monster",
                   character: str = "ironclad") -> tuple[tuple[str, ...], ...]:
    """合格遭遇。**必须全成员合格** —— 少一员的战斗不是"难度低一点"，

    而是完全不同的局面（``docs/13`` §5）。
    """
    table = ENCOUNTERS.get(kind) or ()
    return tuple(enc for enc in table if not encounter_reasons(enc))


def training_encounters() -> dict[str, tuple[tuple[str, ...], ...]]:
    return _cached("encounters", lambda: {
        kind: encounter_pool(kind) for kind in ("monster", "elite", "boss")})


# ==========================================================================
# 指纹与报告
# ==========================================================================
def content_fingerprint() -> dict:
    """内容版本指纹 —— **必须进 checkpoint**（审计 F12）。

    内容变了而词表下标没变，同名 embedding 会指向另一张牌，而且不报错。
    """
    digest = hashlib.sha256()
    for cid in sorted(CARD_DB):
        card = CARD_DB[cid]
        digest.update(
            f"{cid}|{card.cost}|{card.card_type}|{card.rarity}|"
            f"{int(card.effects_incomplete)}|{len(card.effects)}|"
            f"{len(card.upgrade)}\n".encode("utf-8"))
    for eid in sorted(ENEMY_DB):
        enemy = ENEMY_DB[eid]
        digest.update(f"{eid}|{enemy.hp}|{len(enemy.moves)}\n".encode("utf-8"))
    admitted = admitted_cards()
    return {
        "cards": len(CARD_DB),
        "cards_admitted": len(admitted),
        "enemies": len(ENEMY_DB),
        "content_hash": int.from_bytes(digest.digest()[:8], "big"),
        "admission_hash": int.from_bytes(
            hashlib.sha256("\n".join(sorted(admitted)).encode("utf-8")
                           ).digest()[:8], "big"),
    }


def report(character: str = "ironclad") -> dict:
    """准入报告：**每个拒绝原因各有多少张**，以及训练池实际有多大。"""
    counts: dict[str, int] = {}
    closure_only = 0
    for cid in CARD_DB:
        admission = card_admission(cid)
        if admission.admitted:
            continue
        if not admission.own and admission.closure:
            closure_only += 1
        for reason in admission.reasons():
            head = reason.split(":", 1)[0]
            counts[head] = counts.get(head, 0) + 1
    pool = card_pool(character)
    return {
        "character": character,
        "cards_total": len(CARD_DB),
        "cards_admitted": len(admitted_cards()),
        "cards_closure_only_failures": closure_only,
        "reject_reasons": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "draft_pool": len(pool),
        "starting_deck": len(random_deck_cards(character)),
        "encounters": {k: len(v) for k, v in training_encounters().items()},
        "fingerprint": content_fingerprint(),
    }


def assert_trainable(character: str = "ironclad",
                     allow_builtin: bool = False) -> None:
    """训练启动门禁：池子或遭遇为空时**立刻失败**，不静默退化成 builtin。

    审计 F03：``train_combat.py`` 没有显式内容加载，默认跑的是 12 张卡、
    3 只怪的内置占位内容 —— 训练"跑起来了"但训练的是另一个游戏。

    ``allow_builtin`` 只给**单元测试与冒烟**用，且必须由调用方显式打开；
    默认关闭，避免"忘了传 --content"时静默跑内置内容。
    """
    from .content import CONTENT_SOURCE

    if CONTENT_SOURCE == "builtin" and not allow_builtin:
        raise RuntimeError(
            "训练内容未加载：当前是内置占位内容（12 张卡 / 3 只怪）。"
            "请用 --content 指定真实内容目录，或设置 STS2_CONTENT_DIR。")
    pool = card_pool(character)
    if not pool:
        raise RuntimeError(f"角色 {character!r} 的可训卡池为空（准入全被拒）")
    encounters = training_encounters().get("monster") or ()
    if not encounters:
        raise RuntimeError("合格遭遇池为空（准入全被拒）")
