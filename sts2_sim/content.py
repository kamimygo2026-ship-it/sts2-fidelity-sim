"""内容数据表。

**数据驱动**：卡牌/怪物是数据，不是类（``docs/02`` §2.3）。

⚠️ 本文件中的所有条目 ``verified=False`` —— 数值与 AI 权重都是**按记忆写的近似值**，
必须经过真机对拍才能进入训练集（``docs/06`` §6.3 的 ``verified`` 标记）。
骨架阶段的目的是把**架构机制**跑通，不是复刻数值。

真实内容将从社区数据管线导入（``docs/06``）：
    spire-codex  GET /api/exports/{lang}  →  全量 JSON  →  normalize  →  RON/JSON
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as _replace
from typing import Literal

Target = Literal["enemy", "all_enemies", "self", "none"]


# --------------------------------------------------------------------------
# 效果 DSL
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Effect:
    """一条可执行的效果。真实内容里 90% 的卡由这些算子覆盖，剩余走 ``op="custom"``。"""

    op: str                       # damage | damage_all | block | apply_power | draw | lose_hp | gain_energy | add_card | heal | gain_stars
    amount: int = 0
    power: str | None = None
    target: Target = "enemy"
    card: str | None = None
    #: ``add_card`` 的目标牌堆：``hand`` / ``draw`` / ``discard``。
    #: ⚠️ **不能默认成弃牌堆就当没事**：真机"加一张 Soul 进抽牌堆"与"进手牌"
    #: 是不同效果，一律塞弃牌堆是静默的行为错误。
    pile: str = "discard"
    #: ``add_card`` 的位置（``top`` / ``random`` / 空 = 默认）。
    position: str = ""
    #: ``select_card`` 的**来源牌堆**（``hand`` / ``discard`` / ``draw``）。
    #: 字段名不能叫 ``from``（Python 关键字），JSON 里仍是 ``from``。
    select_from: str = "hand"
    #: ``select_card`` 的用途：``discard`` / ``exhaust`` / ``upgrade`` / ``transform``。
    purpose: str = ""
    #: ``channel`` 的充能球 id（``lightning`` / ``frost`` / ``dark`` / ``glass`` /
    #: ``plasma``）。与 ``power`` 分开，避免"球"和"能力"两个概念混在一个字段里。
    orb: str = ""
    #: ``evoke_next`` 是否**移除**被激发的球。
    #: 真机 ``OrbCmd.EvokeNext(ctx, player, dequeue)`` 的第 3 个参数：
    #: `DualCast` 先 ``dequeue: false`` 激发一次（球留着），再正常激发一次（球移走）。
    dequeue: bool = True
    #: 次数依赖 X（``ResolveEnergyXValue``）：打出时才知道，由引擎按 X 展开。
    times_x: bool = False
    #: 量依赖 X：打出时才知道，由引擎按 X 代入。
    amount_x: bool = False
    #: ``ValueProp.Unpowered``：这个量**不经过**力量/虚弱/易伤（伤害）
    #: 或敏捷/脆弱（格挡）的修正，但**仍然可以被格挡**。
    #: 药水与充能球都走这条 —— 不认它会让药水伤害被玩家的力量加成。
    unpowered: bool = False
    #: ``ValueProp.Unblockable``：完全不经过格挡。
    unblockable: bool = False
    #: ``move_all_matching`` 的过滤条件（``(键, 值)`` 元组）。
    #: 键的词汇表见 ``tools/extract_cards.py`` 的 ``FILTER_PATTERNS``。
    card_filter: tuple[tuple[str, object], ...] = ()
    #: ``move_all_matching`` 的"随机取 N 张"（空 = 全部）。
    take_random: str = ""
    #: ``obtain_relic`` 要给哪个遗物。**空 = 随机**（走遗物抓包按稀有度取，
    #: 见 ``sts2_sim/relicbag.py`` 与 ``RelicFactory.PullNextRelicFromFront``）。
    relic: str = ""
    #: ``set_power_var`` 要写哪个**能力实例变量**（``Damage`` / ``Block`` …）。
    #: 真机写法是 ``PowerCmd.Apply<X>(…)?.SetDamage(值)`` —— 把卡牌的动态变量
    #: 搬进能力自己的 ``DynamicVars``（``TheBombPower.SetDamage`` /
    #: ``ToricToughnessPower.SetBlock``）。以前这句被抽取器**静默丢掉**，
    #: 于是卡看起来可用、数值却少一块。
    var: str = ""
    #: ``set_power_var`` 的**取值来源**（空 = 用 ``amount``）。
    #: ``last_block_gain`` = 最近一次 ``CreatureCmd.GainBlock`` 的**返回值**
    #: （``ToricToughness`` 的 ``SetBlock(blockAmount)``）—— 注意它**不是**卡面写的
    #: 那个数：真机拿到的是已经过敏捷/脆弱/倍率的实际获得量。
    set_from: str = ""
    #: ⭐ ``generate_card`` / ``add_card`` 生成的牌**本回合免费**
    #: （``CardModel.SetToFreeThisTurn()``，CardModel.cs:1267）。
    #: 真机写法是"先生成、再对那个卡实例调一次"，所以这是**实例级**修饰符，
    #: 由 :attr:`sts2_sim.core.CardInstance.free_this_turn` 承载。
    #: 丢掉它 = 本回合凭空多花能量（比真机弱），而且日志一切正常。
    free: bool = False
    #: ⭐ ``generate_card`` 要**排除**的卡 id（``where !(c is JackOfAllTrades)``）：
    #: 真机用这种写法防"把这张牌自己生成出来"。丢掉它会让 `jack_of_all_trades`
    #: 有概率生成它自己（真机永远不会），是**静默**的强度偏差。
    exclude: tuple[str, ...] = ()
    #: ⭐ ``add_card`` / ``generate_card`` 造出来的牌**直接是升级版**
    #: （``CardCmd.Upgrade(刚生成的牌)``）。真机有两处这种写法：::
    #:
    #:     CunningPotion:  foreach (item in await Shiv.CreateInHand(…)) CardCmd.Upgrade(item);
    #:     CosmicConcoction: foreach (item in distinctForCombat) { CardCmd.Upgrade(item); AddGeneratedCardToCombat(item, …); }
    #:
    #: 丢掉它 = 给出的牌比真机**弱一档**（未升级的 Shiv / 无色牌），
    #: 而且报告完全正常 —— 所以要么置这个标志，要么整条拒绝。
    upgrade: bool = False
    #: ``autoplay_from_draw`` 打出后是否**强制消耗**
    #: （``CardPileCmd.AutoPlayFromDrawPile(…, forceExhaust: true)``，``Havoc``）。
    #: 丢掉它 = 那张牌打完进弃牌堆、还能再抽到 —— 比真机强。
    force_exhaust: bool = False
    #: ⭐ ``random_deck_cards`` 用哪条**随机流**（引擎流名：``niche`` /
    #: ``transformations`` / ``shuffle`` …）。真机把流名写在调用里
    #: （``StableShuffle(base.Owner.RunState.Rng.Niche)``），**必须照着抽** ——
    #: 换成别的流，分布一样但**序列**与真机分叉（`docs/13` §6.1 的两层口径）。
    rng: str = ""
    #: ⭐ ``random_deck_cards`` 的**取法**：``shuffle`` = 先 ``StableShuffle``
    #: 再 ``Take(N)``；``item`` = 循环 ``NextItem`` + ``Remove``（取一张删一张）。
    #: 两种写法**消费的随机数不同**，所以不能合并成一种实现。
    pick: str = ""
    #: ⭐ **``choose_one`` 的候选**（S02）：``((卡 id, 该卡的效果组), …)``。
    #:
    #: 真机出处：``KnowledgeDemon.CurseOfKnowledgeMove``（``KnowledgeDemon.cs:157-176``）
    #: 先造出两张卡（`CombatState.CreateCard`），让玩家二选一，再执行选中那张的
    #: ``IChoosable.OnChosen``。候选卡**不在任何牌堆里**，所以不能靠 cid 回查效果 ——
    #: 同一个 cid 在不同轮次的载荷不同（`Disintegration` 的三轮是 6 / 7 / 8）。
    #: 第一个字段只用于**显示与日志**（界面上显示的就是那张卡）。
    choices: tuple[tuple[str, tuple["Effect", ...]], ...] = ()
    #: ⭐ **运行期公式**（`docs/12` §2.33）：这一条的量不是常量，而是
    #: ``CalculationBase + ExtraDamage × multiplier(lambda)`` —— 真机的
    #: ``CalculatedVar.WithMultiplier``（docstring 写明"乘的是 ExtraVar 的值"，
    #: 例子就是 ``PerfectedStrike`` 的"6 + 每张打击牌 2"）。
    #:
    #: ``calc_kind`` 空 = 不是计算型（走 :attr:`amount`）；
    #: 非空时 :attr:`amount` 恒为 0，真正的量在**结算那一刻**由
    #: :func:`sts2_sim.powers.calc_amount` 按 ``calc_kind`` 算。
    #: 认不出的形状**不会**走到这里 —— 抽取器直接报"量由运行期公式决定"
    #: 把整张卡排除（宁缺勿猜）。
    calc_kind: str = ""
    #: 公式的参数（``target_power`` 的能力 id / ``all_cards_tag`` 的标签）。
    calc_arg: str = ""
    #: 公式的基准项（``CalculationBase``，升级增量已并入）。
    calc_base: int = 0
    #: 公式的增量项（``ExtraDamage`` / ``CalculationExtra``，升级增量已并入）。
    calc_extra: int = 0


@dataclass(frozen=True)
class CardDef:
    cid: str
    name: str
    cost: int
    card_type: str                # attack | skill | power
    rarity: str
    target: Target
    effects: tuple[Effect, ...]
    upgrade: tuple[Effect, ...] = ()
    exhaust: bool = False
    verified: bool = False        # ⚠️ 未对拍
    #: 真机关键词（``Exhaust`` / ``Ethereal`` / ``Innate`` / ``Retain`` /
    #: ``Unplayable`` / ``Sly`` / ``Eternal``）。引擎还没全部实现，但**必须存下来**
    #: —— 少了 ``Ethereal`` 会让"回合结束烧掉"的卡永久留在手里。
    keywords: tuple[str, ...] = ()
    #: 真机标签（``Strike`` / ``Defend`` / ``Shiv`` …），供"打击牌"类效果引用。
    tags: tuple[str, ...] = ()
    #: ⭐ **星费**（摄政王的资源，``CardModel.CanonicalStarCost``）。0 = 不要星。
    #: ⚠️ 少了它，带星费的卡在引擎里**白嫖**（`comet` 要 5 星却 0 星可打），
    #: 而门禁只看"效果能不能落地"，根本发现不了 —— 这是"静默变强"。
    star_cost: int = 0
    #: ``HasStarCostX``：花掉**当前全部星**（`GetStarCostWithModifiers` 的第一条分支）。
    is_x_star_cost: bool = False
    #: 动态变量 ``(名字, 基础值, 升级增量)``。效果的量本来是**变量引用**，
    #: 载入时已解析成具体数值，这里保留原值供审计与再解析。
    vars: tuple[tuple[str, int, int], ...] = ()
    #: 效果数值的来源：``text`` = 解析描述文本，``source`` = 反编译源码。

    #: 两者的可信度完全不同，必须能区分。
    effect_source: str = "text"
    #: **手牌里的触发式效果**（``OnTurnEndInHand`` 等），与"打出的效果"分开存。
    #: 真机有 12 张卡是这样 —— 其中 10 张是敌人塞给你的状态/诅咒牌
    #: （`burn` / `decay` / `infection` / `toxic` …）。
    #: ⚠️ 不接这些触发器，模拟器就**比真机简单**：手里的燃烧牌毫无代价，
    #: 模型会学会"囤着不管"，而这在真机里是要掉血的。
    triggers: tuple[tuple[str, tuple[Effect, ...]], ...] = ()
    #: ⚠️ 这张卡**有触发器，但引擎没实现对应的钩子**（如 `before_hand_draw`）。
    #: 与 `effects_incomplete` 同理：必须能被排除出训练集，不能静默当成"没有触发"。
    triggers_incomplete: bool = False
    #: X 费卡：打出时消耗**全部**能量，效果强度随 X 变化。
    #:
    #: ⚠️ 不能靠费用数字判断，两个方向都会错：
    #: 普通 X 费卡（`whirlwind` / `skewer` …）的构造参数是 ``0`` ——
    #: 不标记就会被当成**免费**，白嫖 9 张牌；而 `Cascade` 的构造参数是 ``-1``，
    #: 不标记就会被当成**不可打出**。``is_x_cost`` 是唯一可靠的信号。
    is_x_cost: bool = False
    #: ⭐ **多人专用**（``CardMultiplayerConstraint.MultiplayerOnly``）：
    #: 单人局里**根本不会出现**这张牌，所以它不该进单人训练池。
    #:
    #: ⚠️ 这条与"效果不全"是**两件事**：多人专用卡的效果在源码里可能是完整的、
    #: 引擎也实现得没问题 —— 只是这个 profile（单人）遇不到它。
    #: 不判它，"卡牌可执行"这个数字里就会混进单人局不存在的牌
    #: （实测 16 张，`docs/12` §2.32）。
    multiplayer_only: bool = False
    #: ⚠️ **效果已知不完整**：源码里有这张卡的效果，但含我们还没实现的命令
    #: （选牌、造牌、锻造…），于是退回了文本解析的结果。
    #:
    #: 这不是"小瑕疵"：``acrobatics`` 真机是"抽 3 再弃 1"，文本版只剩"抽 3" ——
    #: 模型会拿到一张**被强化**的卡并学会滥用它。所以必须能被排除出训练集
    #: （见 ``docs/06`` §6.5 与 :func:`unverified_report`）。
    effects_incomplete: bool = False
    #: ⭐ **战斗内能否被生成 / 转化到**（``CardModel.CanBeGeneratedInCombat``，默认 true）。
    #: 本地构建里 19 张卡覆写成 ``false``（知识恶魔那几张专属牌、部分状态牌）。
    #:
    #: 用途是**转化与随机生成**的候选池：
    #: ``CardFactory.GetFilteredTransformationOptions`` 在 ``isInCombat`` 时按它过滤
    #: （``CardFactory.cs:201-204``）。不判它，转化会发出真机永远发不出的牌 ——
    #: 而且卡面看起来完全正常。
    can_be_generated_in_combat: bool = True

#: 卡牌 ``CanonicalVars`` 里代表**一种机制**、而引擎还没实现的变量名。
#:
#: ⚠️ 为什么必须单独列出来、而不是"任何没被消费的变量都算缺口"：
#: `Tank` 声明了 `DamageIncrease` / `DamageDecrease`，但那是**能力自身的量纲**
#: （由 `TankPower` 消费，不是卡的效果），一刀切会把它误判成缺口。
#: 所以这里只收**语义无歧义**的那些：出现即代表卡上少了一整条机制。
MECHANIC_VARS: dict[str, str] = {
    "Repeat": "效果重复 N 次",
    "Increase": "本场成长（每次打出递增）",
    "Cards": "附加 / 抽取 N 张",
    "Shivs": "生成 N 张 Shiv",
    "PutBack": "放回抽牌堆 N 张",
    "PlayMax": "本回合最多再打 N 张",
    "CalculationBase": "运行期公式的基数",
    "CalculationExtra": "运行期公式的增量",
    "CalculatedCards": "运行期算出的张数",
    "CalculatedChannels": "运行期算出的引导次数",
    "Energy": "能量收支（引擎没有对应效果）",
}

#: 明确**不算**缺口的变量：它们是能力的量纲，由 `PowerCmd.Apply` 消费。
POWER_MAGNITUDE_VARS: frozenset[str] = frozenset({
    "DamageIncrease", "DamageDecrease",
})


def _unconsumed_mechanics(record: dict) -> list[str]:
    """返回"声明了但没有任何效果引用它"的机制变量说明（空 = 完整）。

    判据是**变量名是否出现在效果/触发器的 JSON 里** —— 因为解析层会把
    ``amount_var`` 原样保留在记录里，所以"被消费过"必然留下痕迹。
    这是保守方向：宁可疑报（卡被排除），不可漏报（卡静默变弱）。
    """
    import json as _json
    import re as _re

    blob = _json.dumps(record.get("effects") or [], ensure_ascii=False)
    blob += _json.dumps(record.get("triggers") or {}, ensure_ascii=False)
    out: list[str] = []
    for var in record.get("vars") or ():
        name = str(var.get("name") or "")
        if name not in MECHANIC_VARS or name in POWER_MAGNITUDE_VARS:
            continue
        if _re.search(rf'"{name}"', blob) or _re.search(rf"\.{name}\b", blob):
            continue                 # 效果在用这个变量 → 已建模
        out.append(f"未建模的机制变量 {name}（{MECHANIC_VARS[name]}）")
    return out


@dataclass(frozen=True)
class MoveDef:
    mid: str
    intent: str                   # attack | attack_debuff | block | buff | debuff | unknown
    value: int
    times: int = 1
    effects: tuple[Effect, ...] = ()
    weight: float = 1.0
    first_turn_only: bool = False
    after_first_turn: bool = True
    name: str = ""
    #: 自爆招式：真机在招式处理函数结尾写 ``await CreatureCmd.Kill(base.Creature)``，
    #: 打完自己就死，因此**永远不会请求后继状态**。只有 ``gas_bomb`` 与
    #: ``waterfall_giant`` 的 ``EXPLODE_MOVE`` 是这样。不建模的话，它们下一回合
    #: 会去 roll 后继并抛"没有后继状态"（实测 gas_bomb 直接炸掉 Run 层）。
    kills_self: bool = False
    #: ⭐ **按怪物实例计数器选载荷**：``(计数器名, ((值, 效果组), …))``。
    #:
    #: 真机出处：``KnowledgeDemon.CurseOfKnowledgeMove``（``KnowledgeDemon.cs:157-176``）
    #: 读 ``_curseOfKnowledgeSets[CurseOfKnowledgeCounter]`` —— **同一个招式状态
    #: 跑三次**，每次给的两张卡不同、``Disintegration`` 的 6/7/8 也不同。
    #: 非空时引擎按 ``enemy.ai_counters[计数器名]`` 找匹配的组，**优先于** :attr:`effects`；
    #: 找不到匹配组就返回**空**（不是退回 `effects` —— 那会把第 4 轮错算成第 1 轮）。
    effects_by_counter: tuple[str, tuple[tuple[int, tuple[Effect, ...]], ...]] = ()
    #: ⭐ 招式**走完**之后要自增的实例计数器名。真机写在招式函数末尾::
    #:
    #:     if (base.CombatState.IsLiveCombat()) CurseOfKnowledgeCounter++;
    #:
    #: 正常跑完与"挂起后续跑"**两条路径都必须递增**（`core._finish_enemy_move`），
    #: 漏一处这个计数器就永远停在 0，条件分支也就永远走同一条。
    counter_increment: str = ""
    #: 递增是否带 ``if (base.CombatState.IsLiveCombat())`` 守卫。
    #: ``KnowledgeDemon.cs:172`` 有，``TestSubject.cs:256`` 没有（裸的 ``Respawns++``）。
    #: 战斗已结束还记账，会让"第 3 次复活"这类分支被多推一格。
    counter_increment_live_only: bool = False


@dataclass(frozen=True)
class EnemyDef:
    eid: str
    name: str
    #: 血量区间。``None`` = **还不知道**，在 ``_load_monster_ai`` 用源码补齐后
    #: 由 ``load_content_dir`` 硬校验。故意不留默认值 ``(1, 1)``：那会让一只
    #: Boss 变成一血木桩而且毫无提示（``TestSubject`` 就差点这样）。
    hp: tuple[int, int] | None
    ai: str                       # 引用 engine 里的 AI 函数名
    moves: tuple[MoveDef, ...]
    verified: bool = False
    #: 忠实的循环出招顺序（来自真机的 attack_pattern.type == "cycle"）
    cycle: tuple[str, ...] = ()
    #: 真机招式模式类型（cycle / random / conditional / ...）。非 cycle 时我们的
    #: 复现是近似，必须能看出哪些怪属于这一类。
    pattern_type: str = ""
    #: 真机自带但我们**没有建模**的能力 —— 不能静默丢弃，要能报出来
    unmodeled_innate_powers: tuple[str, ...] = ()
    #: 固有属性 ``(能力, 层数)``：真机在这只怪入场时施加（``AfterAddedToRoom``）。
    #: ⚠️ 少了它，46 只怪（40%）开局就拿不到自己的固有属性 ——
    #: `artifact 3` 与 `artifact 1` 是完全不同的战斗。
    innate_powers: tuple[tuple[str, int], ...] = ()
    #: 出招状态机的原始记录（来自 ``tools/extract_monsters.py``）。
    #: 有它就用真机状态机，没有才退回 ``cycle`` / 权重近似的旧逻辑。
    ai_record: dict | None = None


@dataclass(frozen=True)
class CharacterDef:
    """角色定义（来自真机 ``characters.json``）。

    起始牌组**必须**从这里取，不能靠猜——真机的基础牌按角色分开命名
    （``strike_ironclad`` / ``defend_silent``…），写死 ``strike`` 会一张都找不到。
    """

    cid: str
    name: str
    hp: int
    gold: int
    energy: int
    deck: tuple[str, ...]
    relics: tuple[str, ...] = ()
    verified: bool = False


# --------------------------------------------------------------------------
# 卡牌（近似值，未对拍）
# --------------------------------------------------------------------------
_CARDS: tuple[CardDef, ...] = (
    CardDef("strike", "打击", 1, "attack", "basic", "enemy",
            (Effect("damage", 6),), (Effect("damage", 9),)),
    CardDef("defend", "防御", 1, "skill", "basic", "none",
            (Effect("block", 5, target="self"),), (Effect("block", 8, target="self"),)),
    CardDef("bash", "重击", 2, "attack", "basic", "enemy",
            (Effect("damage", 8), Effect("apply_power", 2, "vulnerable")),
            (Effect("damage", 10), Effect("apply_power", 3, "vulnerable"))),
    CardDef("anger", "愤怒", 0, "attack", "common", "enemy",
            (Effect("damage", 6), Effect("add_card", 1, card="anger", target="self")),
            (Effect("damage", 8), Effect("add_card", 1, card="anger", target="self"))),
    CardDef("cleave", "顺劈斩", 1, "attack", "common", "all_enemies",
            (Effect("damage_all", 8),), (Effect("damage_all", 11),)),
    CardDef("iron_wave", "铁斩波", 1, "attack", "common", "enemy",
            (Effect("damage", 5), Effect("block", 5, target="self")),
            (Effect("damage", 7), Effect("block", 7, target="self"))),
    CardDef("shrug_it_off", "耸肩无视", 1, "skill", "common", "none",
            (Effect("block", 8, target="self"), Effect("draw", 1, target="self")),
            (Effect("block", 11, target="self"), Effect("draw", 1, target="self"))),
    CardDef("pommel_strike", "柄击", 1, "attack", "common", "enemy",
            (Effect("damage", 9), Effect("draw", 1, target="self")),
            (Effect("damage", 10), Effect("draw", 2, target="self"))),
    CardDef("thunderclap", "雷霆一击", 1, "attack", "common", "all_enemies",
            (Effect("damage_all", 4), Effect("apply_power", 1, "vulnerable", "all_enemies")),
            (Effect("damage_all", 7), Effect("apply_power", 1, "vulnerable", "all_enemies"))),
    CardDef("bloodletting", "放血", 0, "skill", "uncommon", "none",
            (Effect("lose_hp", 3, target="self"), Effect("gain_energy", 2, target="self")),
            (Effect("lose_hp", 3, target="self"), Effect("gain_energy", 3, target="self")),
            exhaust=True),
    CardDef("flex", "屈伸", 0, "skill", "common", "none",
            (Effect("apply_power", 2, "strength", "self"),),
            (Effect("apply_power", 4, "strength", "self"),)),
    CardDef("injury", "创伤", 0, "skill", "curse", "none", ()),   # 诅咒：不可打出
)

# --------------------------------------------------------------------------
# 怪物（近似值，未对拍）
# --------------------------------------------------------------------------
_ENEMIES: tuple[EnemyDef, ...] = (
    EnemyDef("jaw_worm", "颚虫", (40, 44), "ai_jaw_worm", (
        MoveDef("chomp", "attack", 11, 1, (Effect("damage", 11),), weight=0.25),
        MoveDef("thrash", "attack", 7, 1,
                (Effect("damage", 7), Effect("block", 5, target="self")), weight=0.30),
        MoveDef("bellow", "buff", 3, 1,
                (Effect("apply_power", 3, "strength", "self"),
                 Effect("block", 6, target="self")), weight=0.45),
    )),
    EnemyDef("cultist", "邪教徒", (48, 54), "ai_cultist", (
        MoveDef("incantation", "buff", 3, 1,
                (Effect("apply_power", 3, "strength", "self"),),
                weight=1.0, first_turn_only=True),
        MoveDef("dark_strike", "attack", 6, 1, (Effect("damage", 6),),
                weight=1.0, after_first_turn=True),
    )),
    EnemyDef("red_louse", "红虱", (10, 15), "ai_louse", (
        MoveDef("bite", "attack", 6, 1, (Effect("damage", 6),), weight=0.5),
        MoveDef("spit", "debuff", 2, 1,
                (Effect("apply_power", 2, "weak"),), weight=0.5),
    )),
)

CARD_DB: dict[str, CardDef] = {c.cid: c for c in _CARDS}
ENEMY_DB: dict[str, EnemyDef] = {e.eid: e for e in _ENEMIES}

#: 起始牌组（内置占位版）。真机内容加载后会被**原地替换**成 characters.json 的权威数据。
STARTING_DECK: list[str] = ["strike"] * 5 + ["defend"] * 4 + ["bash"]

#: 内置模式的卡池。⚠️ **必须在模块级就建好**：Run 层的卡牌奖励只从池里抽，
#: 池是空的会直接抛错。而 `load_builtin()` 只负责"恢复"，模块导入时并不走它 ——
#: 只在 `load_builtin()` 里建池，会让 `--content ""`（内置）模式一进 Run 层就崩。
_BUILTIN_CARD_POOL: str = "IroncladCardPool"

#: 默认角色（真机有 5 个）。Ironclad 机制最直白，适合先跑通端到端。
DEFAULT_CHARACTER = "ironclad"
CHARACTERS: dict[str, CharacterDef] = {}

_BUILTIN_STARTING_DECK: tuple[str, ...] = tuple(STARTING_DECK)

POWER_NAMES = ("strength", "vulnerable", "weak")

# --------------------------------------------------------------------------
# 遭遇 / 遗物 / 事件（全部未对拍，占位性质）
# --------------------------------------------------------------------------
#: ⚠️ 真机的遭遇表、精英与 Boss 配置必须从社区数据导入（docs/06）。这里是占位。
#: 难度刻意调到"基线 bot 不是稳赢"的量级，好让基线有区分度——
#: 稳赢的基线无法用来检测回归（docs/04 §4.1）。
ENCOUNTERS: dict[str, tuple[tuple[str, ...], ...]] = {
    "monster": (
        ("jaw_worm",),
        ("cultist",),
        ("red_louse", "red_louse"),
        ("jaw_worm", "red_louse"),
        ("cultist", "red_louse"),
        ("jaw_worm", "jaw_worm"),
        ("cultist", "red_louse", "red_louse"),
    ),
    "elite": (
        ("cultist", "jaw_worm"),
        ("jaw_worm", "jaw_worm", "red_louse"),
    ),
    "boss": (
        ("jaw_worm", "cultist", "red_louse"),
        ("cultist", "cultist", "jaw_worm"),
    ),
}

#: 内置遭遇表快照（``load_builtin`` 用它还原）——**必须放在 ENCOUNTERS 定义之后**
_BUILTIN_ENCOUNTERS: dict[str, tuple[tuple[str, ...], ...]] = {
    key: tuple(value) for key, value in ENCOUNTERS.items()
}

#: 遗物（骨架版：只做"改变开局"这一类，规则型遗物待补）
RELICS: dict[str, str] = {
    "burning_blood": "战斗胜利后回复 6 点生命",
    "vajra": "战斗开始时获得 1 点力量",
    "anchor": "战斗开始时获得 10 点格挡",
    "bag_of_preparation": "战斗开始时多抽 2 张牌",
}
RELIC_POOL: tuple[str, ...] = tuple(RELICS)

#: **源码抽取的真实事件表**（``tools/extract_events.py`` → ``events_source.json``）。
#:
#: 键是 cid 风格的事件 id（``this_or_that``），值是 :class:`sts2_sim.events.EventDef`。
#: 骨架期这里曾放过两个占位事件（``bonfire`` / ``shrine``）—— 已删除：
#: 占位事件与真机毫无关系，留在代码里只会让人误以为"事件在做"。
EVENT_DB: dict[str, object] = {}

#: 真机一场战斗的**敌人槽位上限**（``EncounterModel.Slots`` 最多到 "fifth"）。
#: 用途：遭遇表里成员数超过它的记录必须被排除 —— 真机不可能同时站 6 只怪，
#: 那是"把可选成员列表当成了实际生成结果"，照单生成会得到一场不存在的战斗。
MAX_ENEMY_SLOTS = 5


#: 稀有度权重（卡牌奖励抽取用）
RARITY_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("common", 0.60),
    ("uncommon", 0.37),
    ("rare", 0.03),
)

#: 进阶难度表（0–10）。⚠️ 真机数值需对拍；这里只演示"数据驱动"。
ASCENSION_TABLE: dict[int, dict[str, float]] = {
    level: {
        "enemy_hp_mult": 1.0 + level * 0.03,
        "start_damage": 0.0 if level < 6 else 5.0,
        "elite_hp_mult": 1.0 + level * 0.04,
    }
    for level in range(11)
}


# --------------------------------------------------------------------------
# 静态检查（docs/06 §6.3 的 validate 阶段，骨架版）
# --------------------------------------------------------------------------
def lint() -> list[str]:
    """返回问题列表（空 = 通过）。

    ⚠️ **负费用是合法的**：真机用 ``cost = -1`` 表示"不可打出"（burn / dazed / decay
    这类状态与诅咒牌，实测 31 张）。早期版本一律报"费用为负"，把 31 张合法卡当成错误。

    ⚠️ **这里查的是"数据有效性"，不是"引擎覆盖率"**，两者必须分开：

    * 效果引用了**真机里不存在**的能力 id → 是 bug（拼错、抽取错），这里报出来。
    * 效果引用了真机存在、但**引擎还没实现**的能力 → 是已知缺口，由
      ``CardDef.effects_incomplete`` 标记并用 ``engine_coverage()`` 统计，
      **不在这里报错** —— 否则一旦启动源码抽取，lint 会被几百条
      "引擎还没做"淹掉，真正的数据错误反而看不见了。
    """
    problems: list[str] = []
    known_powers = set(POWERS) or set(POWER_NAMES)
    for card in CARD_DB.values():
        for group in (card.effects, card.upgrade):
            for eff in group:
                if eff.op == "apply_power" and eff.power not in known_powers:
                    problems.append(f"{card.cid}: 未知 power {eff.power!r}")
                if eff.op == "add_card" and eff.card not in CARD_DB:
                    problems.append(f"{card.cid}: 引用了不存在的卡 {eff.card!r}")
                # ⚠️ 用 `ENGINE_OPS` 这张**唯一**的算子表，别在这里另抄一份：
                # 抄两份的结果是加算子时漏改一处，lint 会把合法算子报成"未知"
                # （实测 `select_card` 就被漏报过一轮）。
                if eff.op not in ENGINE_OPS:
                    problems.append(f"{card.cid}: 未知算子 {eff.op!r}")
                if eff.op == "select_card" and eff.purpose not in ENGINE_SELECTION_PURPOSES:
                    problems.append(f"{card.cid}: 未知选牌用途 {eff.purpose!r}")
        if card.cost < -1:
            problems.append(f"{card.cid}: 费用 {card.cost} 不合法（-1 才表示不可打出）")
    for enemy in ENEMY_DB.values():
        if enemy.hp is None:
            problems.append(f"{enemy.eid}: 血量未知（codex 与源码都没给出）")
            continue
        lo, hi = enemy.hp
        if lo > hi:
            problems.append(f"{enemy.eid}: 血量区间非法")
        if not enemy.moves:
            problems.append(f"{enemy.eid}: 没有出招")
    return problems


def engine_coverage() -> dict:
    """**引擎覆盖率**：多少张卡在引擎里能忠实执行（与 :func:`lint` 分工不同）。

    ``effects_incomplete`` 的卡**不能进训练集**：它们的真机效果比引擎实现的多
    （``acrobatics`` 少一个弃牌、``abrasive`` 少两个能力），拿来训练等于给模型
    一张被强化或被削弱的卡，模型会学会滥用/回避它。
    """
    total = len(CARD_DB)
    by_source = sum(1 for c in CARD_DB.values() if c.effect_source == "source")
    incomplete = [c.cid for c in CARD_DB.values() if c.effects_incomplete]

    # 怪物固有属性的缺口：真机 46 只怪有固有属性，引擎没实现的那些
    # 等于"换了一只怪"（`aeonglass` 的 `artifact 3` 是它整场战斗的核心）。
    # 必须能被看见，而不是静默生效一半。
    from .powers import IMPLEMENTED
    with_innate = [e for e in ENEMY_DB.values() if e.innate_powers]
    missing_innate: dict[str, int] = {}
    monsters_missing: list[str] = []
    for enemy in with_innate:
        absent = sorted({pid for pid, _amount in enemy.innate_powers
                         if pid not in IMPLEMENTED})
        if absent:
            monsters_missing.append(enemy.eid)
            for pid in absent:
                missing_innate[pid] = missing_innate.get(pid, 0) + 1

    # 遗物：**未实现的钩子要排序报出来**。真机有 100+ 种遗物钩子，
    # 引擎只认 :data:`HOOK_TIMING` 里那几个 —— "还差多少"必须是可数的，
    # 否则覆盖率就是自我安慰。
    hook_gaps: dict[str, int] = {}
    for definition in RELICS.values():
        for name in definition.unmodeled_hooks:
            hook_gaps[name] = hook_gaps.get(name, 0) + 1

    # ⭐ 缺口**按类别**拆开（审计 F11）：`commands` / `query` 是真的缺行为；
    # `bookkeeping` 只写遗物自己的私有字段与状态显示（`CardsPlayedThisTurn = 0`），
    # **本身不改变任何游戏状态** —— 把它们算进"遗物缺行为"会高估缺口。
    hook_kinds: dict[str, int] = {}
    effectful_hooks = 0
    bookkeeping_hooks = 0
    for definition in RELICS.values():
        for _name, kind in definition.unmodeled_hook_kinds:
            hook_kinds[kind] = hook_kinds.get(kind, 0) + 1
            if kind == "bookkeeping":
                bookkeeping_hooks += 1
            else:
                effectful_hooks += 1

    # ⭐ **实现了但打不到**的能力。这一项不能省：只报"引擎已实现 N 个"
    # 会造成错觉 —— 能力写完了，但没有任何东西施加它（招式效果缺失、
    # 施加它的那张卡被判残缺、或者只在特定条件下才施加）。
    # 实测这一步抓出过 19 个"实现了却永远不触发"的能力。
    reachable: set[str] = set()
    for card in CARD_DB.values():
        if card.effects_incomplete:
            continue
        for effect in card.effects:
            if effect.op == "apply_power" and effect.power:
                reachable.add(effect.power)
    for potion in POTIONS.values():
        if potion.effects_incomplete:
            continue
        for effect in potion.effects:
            if effect.op == "apply_power" and effect.power:
                reachable.add(effect.power)
    for enemy in ENEMY_DB.values():
        for pid, _amount in enemy.innate_powers:
            reachable.add(pid)
        for move in enemy.moves:
            for effect in move.effects:
                if effect.op == "apply_power" and effect.power:
                    reachable.add(effect.power)

    return {
        "cards_total": total,
        "cards_effect_from_source": by_source,
        "cards_effect_from_text": total - by_source,
        "cards_incomplete": len(incomplete),
        "cards_trainable": total - len(incomplete),
        "trainable_ratio": round((total - len(incomplete)) / max(1, total), 4),
        "incomplete_examples": sorted(incomplete)[:20],
        "monsters_total": len(ENEMY_DB),
        "monsters_with_innate_powers": len(with_innate),
        "monsters_with_unimplemented_innate": len(monsters_missing),
        "unimplemented_innate_ranked": sorted(missing_innate.items(),
                                              key=lambda kv: -kv[1])[:20],
        "powers_implemented": len(IMPLEMENTED),
        "powers_reachable": len(IMPLEMENTED & reachable),
        "powers_implemented_but_unreachable": sorted(IMPLEMENTED - reachable),
        "relics_total": len(RELICS),
        "relics_with_behavior": sum(1 for r in RELICS.values() if r.has_behavior),
        "relics_fully_modeled": sum(1 for r in RELICS.values() if r.fully_modeled),
        "relics_with_combat_gaps": sum(1 for r in RELICS.values() if r.combat_gaps),
        "unimplemented_relic_hook_kinds": len(hook_gaps),
        "unimplemented_relic_hooks": sum(hook_gaps.values()),
        "unimplemented_relic_hook_ranked": sorted(hook_gaps.items(),
                                                  key=lambda kv: -kv[1])[:20],
        # ⭐ 按类别拆开 —— **只有 effectful 才是真的缺行为**（审计 F11）。
        "relic_hook_gap_kinds": hook_kinds,
        "relic_hook_gaps_effectful": effectful_hooks,
        "relic_hook_gaps_bookkeeping": bookkeeping_hooks,
        "relics_with_effectful_gaps": sum(
            1 for r in RELICS.values() if r.effectful_gaps),
        "relics_bookkeeping_only_gaps": sum(
            1 for r in RELICS.values() if r.bookkeeping_only_gaps),
    }


def unverified_report() -> dict[str, list[str]]:
    """列出尚未对拍的内容（训练时应被排除，见 docs/06 §6.5）。"""
    return {
        "cards": [c.cid for c in CARD_DB.values() if not c.verified],
        "cards_effects_incomplete": [c.cid for c in CARD_DB.values()
                                     if c.effects_incomplete],
        "enemies": [e.eid for e in ENEMY_DB.values() if not e.verified],
    }


# --------------------------------------------------------------------------
# 数据驱动加载（tools/import_content.py 的产出）
# --------------------------------------------------------------------------
#: 当前生效的内容来源："builtin"（占位）或 JSON 目录路径
CONTENT_SOURCE = "builtin"
CONTENT_DIR_ENV = "STS2_CONTENT_DIR"


def load_builtin() -> None:
    """恢复内置占位内容（测试用：内容加载会改全局状态，必须能还原）。"""
    global CONTENT_SOURCE
    CARD_DB.clear()
    CARD_DB.update({card.cid: card for card in _CARDS})
    ENEMY_DB.clear()
    ENEMY_DB.update({enemy.eid: enemy for enemy in _ENEMIES})
    ENCOUNTERS.clear()
    ENCOUNTERS.update({k: tuple(v) for k, v in _BUILTIN_ENCOUNTERS.items()})
    CHARACTERS.clear()
    POWERS.clear()
    STARTING_DECK.clear()
    STARTING_DECK.extend(_BUILTIN_STARTING_DECK)
    # 占位内容也要有卡池：Run 层的卡牌奖励**必须**从池里抽，
    # 池不存在就直接报错（而不是退回"从全部卡里抽"那条错的路）。
    # 占位内容只有一个池，就是它自己那几张卡。
    CARD_POOLS.clear()
    CARD_POOLS[_BUILTIN_CARD_POOL] = tuple(card.cid for card in _CARDS)
    POTIONS.clear()
    # ⚠️ 遗物也必须清空：否则从真机内容切回内置时会**残留 299 个真遗物**，
    # 内置模式下的战斗会莫名其妙带上"金刚杵"，而且不报错。
    RELICS.clear()
    # ⚠️ 事件同理：不清空的话，从真机内容切回内置时会**残留 68 个真事件**。
    EVENT_DB.clear()
    CONTENT_SOURCE = "builtin"


def _normalize_card_id(name: str) -> str:
    """``StrikeIronclad`` → ``strike_ironclad``（与卡牌表的 cid 对齐）。

    ⚠️ 同一个归一化必须用在**所有**按类名给出 id 的地方（起始牌组、起始遗物、
    药水）。只对卡牌做而遗物只 ``.lower()``，会让 ``RingOfTheSnake`` 变成
    ``ringofthesnake`` —— 与遗物表的 ``ring_of_the_snake`` 对不上，
    于是角色的初始遗物**静默失效**（审计 F02 的"初始遗物"那一半）。
    """
    import re
    spaced = re.sub(r"(?<!^)(?=[A-Z])", "_", str(name))
    return spaced.replace(" ", "_").lower()


def _load_monster_ai(directory, summary: dict) -> None:
    """把出招状态机挂到怪物定义上。

    数据由 ``tools/extract_monsters.py`` 从**反编译源码**提取（``docs/09`` L1）——
    不能用 codex 的 ``attack_pattern``，那里会把链式赋值的转移记成 ``next: null``。
    """
    import json
    path = directory / "monster_ai.json"
    if not path.exists():
        return
    entries = json.loads(path.read_text(encoding="utf-8"))
    attached = 0
    hp_from_source = 0
    hp_mismatch: list[str] = []
    for entry in entries:
        eid = str(entry.get("eid", "")).lower()
        enemy = ENEMY_DB.get(eid)
        if enemy is None:
            continue
        # HP **以源码为准**（``docs/09`` L1 的通用规则：源码是机制的唯一真相）。
        # codex 的 hp 降级成对账用：两边不一致要能看见，而不是静默取一边。
        hp = entry.get("hp")
        if hp and hp[0] is not None:
            lo = int(hp[0])
            hi = int(hp[1]) if hp[1] is not None else lo
            if enemy.hp is not None and (lo, hi) != enemy.hp:
                hp_mismatch.append(f"{eid}: 源码={(lo, hi)} codex={enemy.hp}")
            if (lo, hi) != enemy.hp:
                enemy = _replace(enemy, hp=(lo, hi))
                hp_from_source += 1
        # 自爆招式：真机把 `CreatureCmd.Kill(base.Creature)` 写在招式**处理函数体**里，
        # 提取结果只存在于 `monster_ai.json` 的 `moves[处理函数名]`；而 `MoveDef`
        # 来自 `monsters.json` 的招式表 —— 两边从来没对上，标记会静默失效。
        # 这里按 **状态 id 与处理函数名** 双向匹配后合并。
        killers: set[str] = set()
        ai_moves = entry.get("moves") or {}
        for state_id, state in (entry.get("states") or {}).items():
            handler = state.get("move")
            if handler and ai_moves.get(handler, {}).get("kills_self"):
                killers.update({state_id.lower(), state_id.lower().removesuffix("_move"),
                                str(handler).lower()})
        if killers and any(
                m.mid.lower() in killers
                or m.mid.lower().removesuffix("_move") in killers
                for m in enemy.moves):
            enemy = _replace(enemy, moves=tuple(
                _replace(m, kills_self=True)
                if m.mid.lower() in killers
                or m.mid.lower().removesuffix("_move") in killers
                else m
                for m in enemy.moves))
            summary["self_destruct_moves"] = summary.get("self_destruct_moves", 0) + 1

        ENEMY_DB[eid] = _replace(enemy, ai_record=entry)
        attached += 1
    summary["monster_ai"] = attached
    if hp_from_source:
        summary["hp_from_source"] = hp_from_source
    if hp_mismatch:
        summary["hp_mismatch"] = hp_mismatch[:20]
        print(f"⚠️ {len(hp_mismatch)} 只怪物的源码 HP 与 codex 不一致（取源码）：")
        for line in hp_mismatch[:5]:
            print(f"    {line}")


def _require_enemy_hp() -> None:
    """加载收尾硬校验：**任何一只怪都不许没有血量**。

    codex 里 ``TestSubject`` 的 HP 是 null，旧代码在 ``_load_monsters`` 直接
    ``return None`` 把整条记录丢掉 —— 一只真 Boss 就这么消失了，而且报告里
    只有一行 ``monster_without_hp``。宁可在这里炸掉。
    """
    missing = sorted(e.eid for e in ENEMY_DB.values() if e.hp is None)
    if missing:
        raise ValueError(
            f"{len(missing)} 只怪物没有血量：{missing[:10]}。"
            f"先用 tools/extract_monsters.py 从源码取 HP（docs/09 L1）。")


@dataclass(frozen=True)
class PowerDef:
    """能力定义（来自真机 ``powers.json``，257 个）。

    ``stack_type`` 决定叠加语义，直接来自源码的 ``PowerStackType``：
    ``Counter`` 按层数叠加、``Single`` 只记有无、``None`` 不叠加。
    ``allow_negative`` 为真时**允许负值**（只有 5 个）——写成"≤0 就删掉"
    会让"力量被削成负数"悄悄变成"力量为 0"。
    """

    pid: str
    name: str
    kind: str            # buff | debuff
    stack_type: str      # counter | single | none
    allow_negative: bool
    description: str
    #: ``InstanceType``（来自源码的 ``PowerModel.InstanceType``）：
    #:
    #: * ``none`` —— **真机默认**：同类能力合成一个，再施加就叠层（242 个）；
    #: * ``instanced`` —— 每次施加**新建一个实例**（21 个）。`TheBombPower` 的
    #:   注释就是最好的例子："放第二个炸弹，你要的是另一个从 3 开始倒数的炸弹，
    #:   而不是把第一个变成 6"；
    #: * ``instancedperapplier`` —— **每个施加者**一个实例，同一施加者再施加
    #:   则叠到那一个（2 个：`oblivion` / `strangle`）。
    #:
    #: ⚠️ 少了它，两个 `the_bomb` 会共用一份倒计时与一份伤害载荷 —— 而日志一切正常。
    instance_type: str = "none"


#: **分实例**的能力 id（``instanced`` + ``instancedperapplier``）。
#: 由 :func:`_load_powers` 从源码的 ``InstanceType`` 现算，引擎按它决定
#: "再施加一次是叠层还是新建实例"（`core.Combatant.add_power`）。
INSTANCED_POWERS: set[str] = set()


#: 能力注册表。空 = 还没加载真机内容（只有内置占位）。
POWERS: dict[str, PowerDef] = {}


def _load_powers(directory, summary: dict) -> None:
    import json
    path = directory / "powers.json"
    if not path.exists():
        return
    entries = json.loads(path.read_text(encoding="utf-8"))
    # 源码优先：**源码是机制的唯一真相**。社区库只补它独有的字段（可读描述）。
    # 实测两边在 `stack_type` 上有 1 处分歧（`well_laid_plans`）、在正负性上
    # 0 处分歧 —— 分歧本身要能看见，所以记进 summary。
    from_source: dict[str, dict] = {}
    source_path = directory / "powers_source.json"
    if source_path.exists():
        from_source = {str(e.get("id", "")).lower(): e
                       for e in json.loads(source_path.read_text(encoding="utf-8"))}

    parsed: dict[str, PowerDef] = {}
    stack_disagreements: list[str] = []
    for entry in entries:
        pid = str(entry.get("id", "")).lower()
        if not pid:
            continue
        source = from_source.get(pid) or {}
        stack_type = str(source.get("stack_type")
                         or entry.get("stack_type") or "").lower()
        codex_stack = str(entry.get("stack_type") or "").lower()
        if source.get("stack_type") and codex_stack and stack_type != codex_stack:
            stack_disagreements.append(f"{pid}: 源码={stack_type} 社区库={codex_stack}")
        parsed[pid] = PowerDef(
            pid=pid,
            name=entry.get("name") or pid,
            kind=str(source.get("type") or entry.get("type") or "").lower(),
            stack_type=stack_type,
            allow_negative=bool(source.get("allow_negative")
                                or entry.get("allow_negative")),
            description=entry.get("description") or "",
            instance_type=str(source.get("instance_type") or "none").lower(),
        )

    # 源码补缺：社区库只有 257 个能力，源码里有 265 个 —— 少的是卡牌专属能力
    # （`hyperbeam_focus_down` / `cacophony` …）。不补的话这些卡一抽取出来就会
    # 报"未知 power"，把真正的数据错误淹掉。
    added_from_source = 0
    for pid, entry in from_source.items():
        if not pid or pid in parsed:
            continue
        parsed[pid] = PowerDef(
            pid=pid,
            name=entry.get("class") or pid,
            kind=str(entry.get("type") or "").lower() or "unknown",
            stack_type=str(entry.get("stack_type") or "").lower() or "counter",
            allow_negative=bool(entry.get("allow_negative")),
            description="",            # 源码里没有可读描述
            instance_type=str(entry.get("instance_type") or "none").lower(),
        )
        added_from_source += 1
    if parsed:
        POWERS.clear()
        POWERS.update(parsed)
        # ⭐ `InstanceType` 是**引擎行为**的输入（叠层 vs 新实例），所以在这里现算一次。
        INSTANCED_POWERS.clear()
        INSTANCED_POWERS.update(pid for pid, p in parsed.items()
                                if p.instance_type != "none")
        summary["powers"] = len(parsed)
        summary["powers_instanced"] = len(INSTANCED_POWERS)
        if added_from_source:
            summary["powers_added_from_source"] = added_from_source
        if stack_disagreements:
            summary["powers_stack_type_mismatch"] = stack_disagreements
        summary["negative_powers"] = sorted(p.pid for p in parsed.values()
                                            if p.allow_negative)


def _is_pure_clog(record: dict, card: "CardDef | None") -> bool:
    """这张卡是不是"纯占位"的诅咒 / 状态牌？

    ⭐ 这类牌**没有效果、也打不出**，它们的作用就是**占住手牌位、卡住你的攻防**
    （社区数据里 `type` 是 ``curse`` / ``status``）。把"什么都不做"当成"效果残缺"
    是错的 —— 它们**本来就该什么都不做**，而"打不出 + 占手牌"这个机制
    引擎已经天然具备（``CardInstance.playable()`` 对负费用返回 False）。

    真正的残缺只有一种：**源码里确实有东西**（效果、触发器、或我们抽不出的命令），
    只是我们建不出来。所以判据要连 ``unsupported`` 一起看 ——
    `alchemize` 的效果全是 `PotionCmd` 这类抽不出的命令，
    它虽然 ``effects`` 为空，却是**真残缺**，不能当诅咒放过去。
    """
    if card is None or card.card_type not in ("curse", "status"):
        return False
    return not any(record.get(field) for field in
                   ("effects", "triggers", "unsupported", "choice_commands"))


def _merge_source_keywords(card: "CardDef", record: dict) -> "CardDef":
    """把**源码里的关键字**合进一张（因效果抽不出而）退回文本版的卡。

    为什么必须合：关键字是**独立于效果**的一套机制，源码抽不出效果**不代表**
    抽不出关键字（``CanonicalKeywords`` 是一个独立的属性表达式）。
    不合的后果很具体：``Dazed`` 失去 ``Ethereal`` → 回合结束不被消耗、
    永远在手里循环，模拟器比真机简单；``Debt`` 失去 ``Unplayable`` → 一张
    本该打不出的诅咒变成了能打出的牌。

    只在文本版**没有**该关键字时才补，避免把 codex 的写法覆盖掉；
    未知关键字由 :func:`_keyword_defects` 标记，不在这里判。
    """
    from . import keywords as keyword_rules
    merged = tuple(sorted(set(card.keywords) | set(record.get("keywords") or ())))
    if merged == tuple(card.keywords):
        return card
    return _replace(card, keywords=merged)


def _keyword_defects(keywords) -> list[str]:
    """卡牌带了引擎没实现的关键字 → 必须标残缺（``docs/09`` §5.4）。

    未知关键字**静默失效**是最隐蔽的一类"模拟器比真机简单"：
    卡能打出、日志照打，只是少了那条自动行为。宁可不训练这张卡。
    """
    from . import keywords as keyword_rules
    unknown = sorted(k for k in (keywords or ())
                     if k not in keyword_rules.ENGINE_KEYWORDS)
    return [f"引擎未实现的关键字 {unknown}"] if unknown else []


def _source_stub(cid: str, record: dict) -> "CardDef":
    """源码里有、文本库里没有的卡 → 建一张**明确残缺**的占位卡。

    为什么不是"干脆不建"：不建的话这张卡在引擎里**根本不存在**，
    报告里也看不到 —— 于是"内容缺失"变成一个不可见的洞
    （实测 ``abundance`` / ``dowsing`` / ``imitation_learning`` /
    ``blade_symphony`` / ``deprecated_card`` 五张）。建出来并标
    ``effects_incomplete``，它就会出现在残缺清单里，且**进不了训练集**。
    费用等结构信息仍来自源码，不猜。
    """
    return CardDef(
        cid=cid,
        name=record.get("class", cid),
        cost=int(record["cost"]) if record.get("cost") is not None else -1,
        is_x_cost=bool(record.get("is_x_cost")),
        star_cost=int(record.get("star_cost") or 0),
        is_x_star_cost=bool(record.get("is_x_star_cost")),
        card_type=record.get("card_type") or "skill",
        rarity=record.get("rarity") or "common",
        target="self" if record.get("target") == "self" else "enemy",
        effects=(),
        upgrade=(),
        exhaust="Exhaust" in (record.get("keywords") or ()),
        verified=False,
        keywords=tuple(sorted(set(record.get("keywords") or ()))),
        tags=tuple(record.get("tags") or ()),
        effect_source="source",
        effects_incomplete=True,
    )


def _event_playability_gap(definition) -> str:
    """事件图**能不能从首页走到结束**？有问题就返回一条理由，否则返回 ``""``。

    判据（都是结构性的，不需要猜）：

    * 从 ``INITIAL`` 出发沿 ``goto`` 能到达的**每一页**都至少有一个
      **没锁住**的选项 —— 真机 ``EventOption.IsLocked`` 是"显示但点不动"
      （``EventOption.cs:62``），一页全是锁住的就是死页。古事件（Ancient）
      大量用 ``RelicOption<T>()``（``onChosen`` 省略 → 锁住）声明选项池，
      抽取器只看到池子里的声明就会以为"有选项"；
    * 不能有"整页的选项都**既没有效果、也没有去向**"的页 —— 点下去什么都不
      发生，而那个选项随后被 ``WasChosen`` 关掉（``EventOption.cs:176``），
      玩家就卡在这一页了。实测 ``DollRoom`` 的 ``ChooseRandom``（委派给
      没被抽到的私有方法）与 ``TheFutureOfPotions`` 的内联 delegate 都栽在这里；
    * 至少要有一条能走到 ``finished`` 的路。

    ⚠️ 这不是"更严的闸门"，而是**抽取缺口的照妖镜**：以上三种形态在真机里
    都不可能出现（真机每个选项要么改状态、要么翻页），出现就说明抽取器漏了东西。
    """
    seen: set[str] = set()
    todo = [definition.initial_page]
    finished_reachable = False
    while todo:
        page_id = todo.pop()
        if page_id in seen:
            continue
        seen.add(page_id)
        page = definition.pages.get(page_id)
        if page is None:
            return f"页 {page_id} 不存在"
        live = [option for option in page.options if not option.locked]
        if not live:
            return f"页 {page_id} 没有可点的选项"
        if all(option.outcome == "none" and not option.effects_raw
               for option in live):
            return f"页 {page_id} 所有选项都没有效果也没有去向（点了会卡住）"
        for option in live:
            if option.outcome == "finished":
                finished_reachable = True
            elif option.outcome == "goto" and option.outcome_page:
                todo.append(option.outcome_page)
    if not finished_reachable:
        return "没有任何一条路能结束事件"
    return ""


def _load_events_source(directory, summary: dict) -> None:
    """加载**源码抽取**的事件图（``tools/extract_events.py`` → ``events_source.json``）。

    与卡牌/遗物同一套诚实性规则（``docs/09`` §5.2）：

    * 选项效果里出现引擎没有的命令、或需要玩家选牌 → **整个事件不可用**
      （``EventDef.reasons`` 非空 → 排除出事件池），而不是"少结算一半效果"；
    * ``IsAllowed`` 的条件只认白名单里的几种写法（``events.parse_gate``），
      认不出的一律不可用 —— 判错的后果是"本不该出现的事件出现了"，比少出现一个严重；
    * 多阶段事件里，选项指向的页必须真实存在，否则状态机会卡住（当缺口报出来）。
    """
    import json

    from . import events as event_rules

    path = directory / "events_source.json"
    if not path.exists():
        return
    records = json.loads(path.read_text(encoding="utf-8"))
    parsed: dict[str, event_rules.EventDef] = {}
    blocked: list[str] = []
    act_blocked: list[str] = []

    for record in records:
        eid = record.get("eid") or ""
        if not eid:
            continue
        reasons: list[str] = list(record.get("reasons") or ())
        vars_ = {v["name"] for v in record.get("vars") or ()}
        rolls = {r["var"] for r in record.get("rng_rolls") or ()}
        #: ``CanonicalVars`` 的字面量（``{名: 值}``）。``IsAllowed`` 里
        #: ``base.DynamicVars.X.BaseValue`` 这种"阈值写成变量"的写法要用它解析
        #: 成具体数字（``events.parse_gate`` 的 ``vars`` 参数）。
        var_values = {v["name"]: int(v["base"])
                      for v in record.get("vars") or ()
                      if v.get("base") is not None}

        pages: dict[str, event_rules.EventPageDef] = {}
        for page_id, page in (record.get("pages") or {}).items():
            options: list[event_rules.EventOptionDef] = []
            for option in page.get("options") or ():
                option_reasons = list(option.get("reasons") or ())
                if option.get("unsupported"):
                    option_reasons += [f"效果缺口：{u}" for u in option["unsupported"]]
                if option.get("choices"):
                    option_reasons.append("需要玩家选牌")
                for effect in option.get("effects") or ():
                    amount, var = effect.get("amount"), effect.get("amount_var")
                    if effect.get("needs_runtime"):
                        option_reasons.append("量需要运行期取值")
                    elif (amount is None and not var
                          and not effect.get("amount_x")):
                        option_reasons.append("效果的量抽不出")
                    elif var and var not in vars_ and var not in rolls:
                        option_reasons.append(f"未知变量 {var}")
                    # ⭐ 奖励（``RewardsCmd.OfferCustom``）：**逐项校验候选来源**。
                    # 指定了引擎里不存在的药水 / 遗物 / 卡池就必须报缺口 ——
                    # 否则运行时只会打一条 ⚠️ 然后什么都不给，
                    # 事件看起来"跑通了"而收益凭空消失。
                    if effect.get("op") == "offer_rewards":
                        spec = dict(effect.get("filter") or ())
                        kind = spec.get("kind")
                        if kind not in ("relic", "potion", "card"):
                            option_reasons.append(
                                f"奖励类型 {kind or '未定'} 引擎没实现")
                        elif kind == "relic" and spec.get("relic") \
                                and spec["relic"] not in RELICS:
                            option_reasons.append(
                                f"奖励遗物 {spec['relic']} 不在遗物表里")
                        elif kind == "potion" and spec.get("potion") \
                                and spec["potion"] not in POTIONS:
                            option_reasons.append(
                                f"奖励药水 {spec['potion']} 不在药水表里")
                        elif kind == "card" and spec.get("pool") \
                                and spec["pool"] != "__character__" \
                                and spec["pool"] not in CARD_POOLS:
                            option_reasons.append(
                                f"奖励卡池 {spec['pool']} 不在卡池表里")
                    # ⭐ 选牌**按用途**判，不是"一出现就报缺口"：
                    # 引擎实现了"从牌组选一张 → 移除 / 升级"，
                    # 其余用途（附魔 / 转化 / 复制 / 从生成候选里挑）如实报出来。
                    if effect.get("op") == "select_card":
                        from .runeffects import RUN_SELECTION_PURPOSES

                        purpose = effect.get("purpose") or ""
                        if purpose not in RUN_SELECTION_PURPOSES:
                            option_reasons.append(
                                f"选牌用途 {purpose or '未定'} 引擎没实现")
                    # ⭐ Run 层"必须指哪张牌"的算子（升级 / 降级 / 转化）：
                    # 没有明确目标卡时运行期只会静默跳过（见
                    # `RUN_OPS_NEEDING_CARD` 的说明），必须在这里拒绝。
                    defect = _run_effect_defect(effect)
                    if defect:
                        option_reasons.append(defect)
                key = option.get("text_key") or ""
                options.append(event_rules.EventOptionDef(
                    key=key,
                    name=key.rsplit(".", 1)[-1] if key else "",
                    effects_raw=tuple(option.get("effects") or ()),
                    outcome=(option.get("outcome") or {}).get("kind", "none"),
                    outcome_page=(option.get("outcome") or {}).get("page", ""),
                    locked=bool(option.get("locked")),
                    reasons=tuple(sorted(set(option_reasons))),
                    unsupported=tuple(option.get("unsupported") or ()),
                ))
            pages[page_id] = event_rules.EventPageDef(page_id=page_id,
                                                      options=tuple(options))

        # 选项指向的页必须存在，否则事件会卡在一页上
        for page in pages.values():
            for option in page.options:
                if option.outcome == "goto" and option.outcome_page \
                        and option.outcome_page not in pages:
                    reasons.append(f"选项 {option.name} 指向不存在的页 "
                                   f"{option.outcome_page}")

        gate_kind, gate_args = event_rules.parse_gate(record.get("gate") or {},
                                                      var_values)
        if record.get("gate") and gate_kind == "unknown":
            reasons.append("进入条件算不出（IsAllowed 写法未支持）")

        option_reasons = sorted({reason for page in pages.values()
                                 for option in page.options
                                 for reason in option.reasons})
        definition = event_rules.EventDef(
            eid=eid,
            name=str(record.get("name") or eid),
            pages=pages,
            initial_page="INITIAL" if "INITIAL" in pages else next(iter(pages), ""),
            gate_body=str((record.get("gate") or {}).get("body") or ""),
            gate_kind=gate_kind,
            gate_args=tuple(gate_args),
            combat_encounter=str(record.get("combat_encounter") or ""),
            literal_vars=tuple((v["name"], int(v["base"]))
                               for v in record.get("vars") or ()
                               if v.get("base") is not None),
            rng_rolls=tuple((r["var"], r["raw"])
                            for r in record.get("rng_rolls") or ()),
            reasons=tuple(sorted(set(reasons) | set(option_reasons))),
            act=str(record.get("act") or ""),
        )
        if not definition.pages or not any(p.options for p in pages.values()):
            definition = _replace(definition, reasons=tuple(
                sorted(set(definition.reasons) | {"选项没抽出来"})))
        else:
            gap = _event_playability_gap(definition)
            if gap:
                definition = _replace(definition, reasons=tuple(
                    sorted(set(definition.reasons) | {gap})))
        parsed[eid] = definition
        if definition.reasons:
            blocked.append(eid)
        if not _event_act_allowed(definition.act):
            act_blocked.append(eid)

    if parsed:
        EVENT_DB.clear()
        EVENT_DB.update(parsed)
        summary["events"] = len(parsed)
        summary["events_usable"] = sum(1 for e in parsed.values() if e.usable)
        summary["events_blocked"] = len(blocked)
        summary["events_blocked_examples"] = sorted(blocked)[:10]
        summary["events_act_blocked"] = len(act_blocked)
        pool = event_pool()
        summary["events_pool"] = len(pool)
        summary["events_pool_ids"] = list(pool)


#: 本模拟器只跑**第 1 幕**，所以事件池按幕筛。真机的幕索引（``Acts/*.cs``）::
#:
#:     Overgrowth   Index => 0
#:     Underdocks   Index => 0     ← 与 Overgrowth 同为第 1 幕的两种主题
#:     Hive         Index => 1
#:     Glory        Index => 2
EVENT_ACT1_TAGS: tuple[str, ...] = ("Act 1 - Overgrowth", "Underdocks")


def _event_act_allowed(act: str) -> bool:
    """这个事件能不能出现在**第 1 幕**。

    社区库的 ``act`` 字段有四种取值（实测）：``Act 1 - Overgrowth`` / ``Underdocks`` /
    ``Act 2 - Hive`` / ``Act 3 - Glory``，另有 22 个事件是 ``None``（跨幕 / 起始事件）。
    判据按**保守**来：只有明确写着第 1 幕标签或没写幕的才放行；
    ``Act 2 - Hive`` / ``Act 3 - Glory`` 一律排除 —— 让第 3 幕的事件出现在第 1 幕，
    会给出真机不可能有的收益。
    """
    if not act:
        return True
    if "Act 2" in act or "Act 3" in act:
        return False
    return any(tag in act for tag in EVENT_ACT1_TAGS)


def relic_bag_entries(character: str = "ironclad") -> tuple[tuple[str, str], ...]:
    """抓包的池子构成：**共享池 ∪ 角色池**，只留 4 种会进包的稀有度。

    真机（``RelicGrabBag.Populate``，``RelicGrabBag.cs:69-92``）::

        list = ModelDb.RelicPool<SharedRelicPool>().GetUnlockedRelics(unlockState)
        list.AddRange(player.Character.RelicPool.GetUnlockedRelics(unlockState))
        list.RemoveAll(r => !_rarities.Contains(r.Rarity));   // Common/Uncommon/Rare/Shop

    社区库的 ``pool`` 字段就是这两类（``shared`` / 各角色名），``rarity`` 已归一化。
    不属于这两类的（Ancient / Event / Starter）**不进包**。
    """
    pool_name = character if character in _RELIC_CHARACTER_POOLS else "ironclad"
    entries = [(rid, definition.rarity) for rid, definition in RELICS.items()
               if getattr(definition, "pool", "shared") in ("shared", pool_name)]
    return tuple(entries)


#: 角色遗物池的名字（社区库 ``pool`` 字段的取值，除 ``shared`` 外）。
_RELIC_CHARACTER_POOLS: frozenset[str] = frozenset(
    {"ironclad", "silent", "defect", "regent", "necrobinder"})


def event_pool(act_id: str | None = None) -> tuple[str, ...]:
    """事件池。

    :param act_id: 幕 id。给了就**只**用那张幕的事件表（``acts_source.json``
        ``events``）当依据 —— 真机的事件池是 ``ActModel.AllEvents +
        ModelDb.AllSharedEvents``（``ActModel.cs:334``），逐幕不同。
        不给就退回旧口径（第 1 幕的那种），只为兼容还没接幕表的调用点。

    ⚠️ 静态条件判成 ``never`` 的（``WarHistorianRepy.IsAllowed => return false``）
    **不进池**：它永远不会出现，留在池里只会让"第 1 幕事件池 N 个"这个数字
    虚高 —— 而抽取的实际分布才是训练真正吃到的。
    """
    allowed: frozenset[str] | None = None
    if act_id is not None:
        definition = ACTS.get(act_id)
        events = getattr(definition, "events", ())
        if events:
            allowed = frozenset(events)
    return tuple(sorted(eid for eid, definition in EVENT_DB.items()
                        if definition.usable
                        and definition.gate_kind != "never"
                        and (allowed is None
                             or eid in allowed)
                        and (allowed is not None
                             or _event_act_allowed(definition.act))))


def _load_cards_source(directory, summary: dict) -> None:
    """用**反编译源码**抽取的效果覆盖卡牌表（``tools/extract_cards.py``）。

    为什么是覆盖而不是补充：描述文本只能覆盖一半句式（``At the start of your
    turn…`` 这类触发器永远解不准），而源码是可执行代码。

    这里的取舍原则（``docs/02`` §2.5「猜错的数值比缺失的数值危险得多」）：

    * **全部效果都能落地**才采用源码版本；有一处算不出来就**整张卡不覆盖**，
      保留文本解析的结果，并把卡号记进 ``cards_source_skipped``。
    * 效果的"量"在源码里是**变量引用**（``DynamicVars.Damage.BaseValue``），
      在这里解析成具体数值：升级版 = ``基础值 + 升级增量``。这样升级前后
      不可能对不上 —— 而抄两份数字一定会对不上。
    * ``needs_runtime``（造成的伤害合计、当前护甲…）与未支持的算子一律拒绝，
      绝不当成 0。
    """
    import json
    path = directory / "cards_source.json"
    if not path.exists():
        return
    records = json.loads(path.read_text(encoding="utf-8"))
    adopted = 0
    skipped: list[str] = []
    added = 0
    defective: list[str] = []
    missing_power_counts: dict[str, int] = {}

    for record in records:
        cid = record.get("cid")
        if not cid:
            continue
        existing = CARD_DB.get(cid)

        # 算子级缺口 → **不采纳**，退回文本版本并标记（文本版只有引擎认识的算子）。
        op_reason = _source_reject_reason(record)
        if op_reason:
            skipped.append(f"{cid}: {op_reason}")
            # ⚠️ **不管文本版有没有效果都要标记**：源码版建不出来 = 这张卡在引擎里
            # 的行为与真机不同。早先这里加了 `and existing.effects` 的条件，
            # 于是"源码被拒 + 文本也没效果"的卡（`chaos` 引导随机球）既没有效果
            # 也没被标记 —— 静默变成一张空卡。
            #
            # ⭐ 例外：**纯占位的诅咒 / 状态牌**。"什么都不做"就是它们的真机行为
            # （作用是占手牌位、卡住攻防），标成残缺是误报。
            #
            # ⚠️ 但这个豁免**只针对"是否算残缺"**，不针对关键字：
            # 关键字是独立的一套机制（`Dazed` 的 Ethereal、`Debt` 的 Unplayable），
            # 早先把豁免套在整段上，于是 clog 卡的关键字一起没了 ——
            # 表现是"虚无的牌回合结束不消耗、不可打出的诅咒能打出"。
            if existing is not None:
                merged = _merge_source_keywords(existing, record)
                if not _is_pure_clog(record, existing):
                    merged = _replace(merged, effects_incomplete=True)
                CARD_DB[cid] = merged
            else:
                CARD_DB[cid] = _source_stub(cid, record)
                added += 1
            continue

        base = _resolve_source_effects(record, upgraded=False)
        upgraded = _resolve_source_effects(record, upgraded=True)
        has_triggers = bool(record.get("triggers"))
        # ⚠️ **只带触发器、没有 OnPlay 效果的卡不能被跳过**：
        # `burn` / `decay` / `infection` 这些状态牌的正常效果就在触发器里，
        # 按"没有效果"丢掉它们，等于让敌人塞给你的负面牌完全失效。
        if base is None or upgraded is None or (
                not base and not upgraded and not has_triggers):
            skipped.append(f"{cid}: 源码效果解析不出（或为空）")
            # ⚠️ **不能只在"文本版有效果"时才标记**。早先这里写的是
            # ``if existing is not None and existing.effects``，于是"源码抽不出 +
            # 文本解析也没效果"的卡**两个标记都不沾** —— 直接以"零效果"的身份
            # 混进训练集。实测 `BodySlam`（伤害 = 当前护甲）就是这样变成
            # "打出去什么都不发生"的空卡，而它是一张核心攻击卡。
            # 唯一豁免是**纯占位的诅咒/状态牌**：什么都不做就是它们的真机行为。
            # （同样：豁免只针对"是否算残缺"，关键字照样要合，见上。）
            if existing is not None:
                merged = _merge_source_keywords(existing, record)
                if not _is_pure_clog(record, existing):
                    merged = _replace(merged, effects_incomplete=True)
                CARD_DB[cid] = merged
            else:
                CARD_DB[cid] = _source_stub(cid, record)
                added += 1
            continue

        # 能力级缺口 → **照样采纳**（比文本版更接近真机），但标记为不完整。
        defects: list[str] = []
        missing = _missing_powers(record)
        if missing:
            defects.append(f"引擎未实现的能力 {missing}")
            for name in missing:
                missing_power_counts[name] = missing_power_counts.get(name, 0) + 1
        # 关键字缺口同理：未知关键字静默失效 = 少了那条自动行为。
        merged = _merge_source_keywords(existing, record) if existing else None
        defects.extend(_keyword_defects(
            merged.keywords if merged else record.get("keywords")))

        # 手牌触发器（`OnTurnEndInHand`）。**引擎没实现的钩子要标记**，
        # 不能静默当成"这张卡没有触发" —— 那会让状态牌毫无代价，
        # 模拟器比真机简单，模型会学着囤牌。
        values = _var_values(record, False)
        triggers: dict[str, tuple[Effect, ...]] = {}
        trigger_defects: list[str] = []
        for hook, hook_effects in (record.get("triggers") or {}).items():
            if hook not in ENGINE_TRIGGER_HOOKS:
                trigger_defects.append(f"未实现的触发钩子 {hook}")
                continue
            resolved_trigger = _resolve_effects(hook_effects, values)
            if resolved_trigger is None:
                trigger_defects.append(f"{hook} 的量解析不出")
                continue
            triggers[hook] = resolved_trigger
        if trigger_defects:
            defects.extend(trigger_defects)

        # ⭐⭐ **声明了却没有任何效果消费它的"机制变量"** → 这张卡在引擎里缺一块。
        #
        # 抽取器只抓**直接命令**，而下面这些 `CanonicalVars` 表达的是"次数 / 成长 /
        # 附加张数 / 运行期公式"：既没变成效果，也没记进 `unsupported`，
        # 于是卡被当成"完全复刻"通过准入 —— 而它在模拟器里**比真机弱**。
        # 实测确认过的例子（每个都回 C# 源码核过）：
        #
        #   Quadcast        for i < Repeat(4): EvokeNext      → 引擎只激发 1 次
        #   BouncingFlask   for i < Repeat(3): 每次上 3 层毒   → 引擎只上 3 层
        #   IceLance        for i < Repeat(2): Channel<Frost> → 引擎只引导 1 个
        #   CloakAndDagger  for i < Cards(1): 加 Shiv         → 引擎根本没加
        #   Claw / Rampage / TheBall / Maul / KinglyPunch     → "本场成长"整条缺失
        #
        # 处理：**标记 `effects_incomplete`**，让它按既有门禁离开训练集
        # （`docs/09` 铁律 5：能排除的内容就排除，好过让模型学错）。
        # 实现一个机制就自动放行对应的卡 —— 门禁与实现是同一套判据。
        defects.extend(_unconsumed_mechanics(record))

        # 关键字：源码是真相，但**文本版独有的**关键字（codex 补的）不丢 ——
        # 两边取并集，缺失的那个方向都会造成"少了自动行为"。
        keywords = tuple(sorted(set(record.get("keywords") or ())
                                | set(existing.keywords if existing else ())))
        card = CardDef(
            cid=cid,
            name=(existing.name if existing else record.get("class", cid)),
            cost=int(record["cost"]) if record.get("cost") is not None else -1,
            card_type=record.get("card_type") or "skill",
            rarity=record.get("rarity") or "common",
            target="self" if record.get("target") == "self" else "enemy",
            effects=base,
            upgrade=upgraded,
            exhaust=bool(record.get("exhaust")) or "Exhaust" in keywords,
            verified=False,
            keywords=keywords,
            tags=tuple(record.get("tags") or ()),
            vars=tuple((v["name"], int(v["base"]), int(v["upgrade_delta"]))
                       for v in record.get("vars") or [] if v.get("base") is not None),
            effect_source="source",
            effects_incomplete=bool(defects),
            # ⭐ 转化 / 随机生成的候选池按它过滤（`CardFactory.cs:201-204`）。
            can_be_generated_in_combat=bool(
                record.get("can_be_generated_in_combat", True)),
            is_x_cost=bool(record.get("is_x_cost")),
            # ⭐ 星费（摄政王）：源码是唯一真相 —— 引擎以前不抽它，
            # 于是 23 张带星费的卡（`comet` 要 5 星）在引擎里**白嫖**。
            star_cost=int(record.get("star_cost") or 0),
            is_x_star_cost=bool(record.get("is_x_star_cost")),
            triggers=tuple(sorted(triggers.items())),
            triggers_incomplete=any("触发钩子" in d or "量解析不出" in d
                                    for d in trigger_defects),
        )
        if existing is None:
            added += 1
        adopted += 1
        CARD_DB[cid] = card
        if defects:
            defective.append(f"{cid}: {'；'.join(defects)}")

    summary["cards_from_source"] = adopted
    if added:
        summary["cards_added_from_source"] = added
    # ⭐ **多人专用标记必须在所有分支之后统一落一次**。
    # `_load_cards_source` 有三条提前 `continue` 的分支（算子缺口 / 效果解析不出 /
    # 纯占位豁免），逐条去写这个字段一定会漏 —— 而漏掉的那条正好是
    # "算子缺口 + 多人专用"，那类卡会**绕过**这条门禁。
    multiplayer = 0
    for record in records:
        cid = record.get("cid")
        if not cid or not record.get("multiplayer_only"):
            continue
        card = CARD_DB.get(cid)
        if card is None or card.multiplayer_only:
            continue
        CARD_DB[cid] = _replace(card, multiplayer_only=True)
        multiplayer += 1
    if multiplayer:
        summary["cards_multiplayer_only"] = multiplayer
    incomplete = sorted(c.cid for c in CARD_DB.values() if c.effects_incomplete)
    if incomplete:
        summary["cards_effects_incomplete"] = len(incomplete)
        summary["cards_effects_incomplete_examples"] = incomplete[:20]
    if defective:
        summary["cards_with_defects_examples"] = defective[:10]
    if missing_power_counts:
        # ⭐ 这就是 L2 的**数据驱动工单**：按"有多少张卡受影响"排序，
        # 而不是凭感觉挑能力去实现。
        summary["missing_powers_ranked"] = sorted(
            missing_power_counts.items(), key=lambda kv: -kv[1])[:30]
    if skipped:
        summary["cards_source_skipped"] = len(skipped)
        summary["cards_source_skipped_examples"] = skipped[:20]


#: 源码效果里引擎**已经实现**的算子（``core.py`` 的效果分发）。
#:
#: 每个算子都标出它对应的**真机命令**（``docs/09`` §5.2 铁律 R1：
#: 机制声明必须带源码出处）。这份对照来自 ``tools/extract_cards.py`` 的分发表 ——
#: 抽取器就是把下面这些命令翻译成算子的，所以出处与实现是同一处真相。
#:
#: ==================== ==========================================
#: 算子                  真机命令
#: ==================== ==========================================
#: ``damage``           ``CreatureCmd.Damage``
#: ``damage_all``       ``CreatureCmd.Damage`` 遍历 ``HittableEnemies``
#: ``block``            ``CreatureCmd.GainBlock``
#: ``apply_power``      ``PowerCmd.Apply<T>``
#: ``draw``             ``CardPileCmd.Draw``
#: ``lose_hp``          ``CreatureCmd.Damage`` + ``ValueProp.Unblockable``（自伤）
#: ``gain_energy``      ``PlayerCmd.GainEnergy``
#: ``heal``             ``CreatureCmd.Heal``
#: ``gain_stars``       ``PlayerCmd.GainStars``
#: ``add_card``         ``CardPileCmd.AddGeneratedCardToCombat`` / ``AddGeneratedCard``
#: ``move_self_to_pile````CardPileCmd.Add(this, PileType.X)``
#: ``move_all_matching````CardPileCmd.Add`` 配 ``TakeRandom`` 筛选
#: ``select_card``      ``CardSelectCmd.FromHand`` / ``FromCombatPile``
#: ``channel``          ``OrbCmd.Channel<T>``
#: ``evoke_next``       ``OrbCmd.EvokeNext``
#: ``add_orb_slots``    ``OrbCmd.AddSlots``
#: ``gain_max_hp``      ``CreatureCmd.GainMaxHp``（**同时回血**，见 docs/09 §5.4）
#: ``lose_max_hp``      ``CreatureCmd.LoseMaxHp``（只缩上限，超出部分扣血，下限 1）
#: ``set_power_var``    ``PowerCmd.Apply<T>(…)?.SetXxx(…)``（把卡牌动态变量写进
#:                      能力实例：``TheBombPower.SetDamage`` / ``ToricToughnessPower.SetBlock``）
#: ``generate_card``    ``CardFactory.GetDistinctForCombat`` + ``CardPileCmd.AddGeneratedCard(s)ToCombat``
#:                      （从卡池随机生成 N 张**互不重复**的牌进手牌）
#: ``upgrade_hand``     ``CardCmd.Upgrade(PileType.Hand.GetPile(owner).Cards)``
#:                      （把**整手牌**升级，`Bellows`）
#: ``random_draw_cards````StableShuffle(Rng.CombatCardSelection).Take(N)``
#:                      （战斗内**抽牌堆**随机取 N 张升级，`StoneCracker`）
#: ``shuffle``          ``CardPileCmd.Shuffle``（弃牌堆并入抽牌堆后整体 ``StableShuffle``）
#: ``autoplay_from_draw````CardPileCmd.AutoPlayFromDrawPile``（从抽牌堆顶取出 N 张后逐张
#:                      ``CardCmd.AutoPlay``；``forceExhaust`` 决定打完是否消耗）
#: **不能由修饰符随机生成**的卡（``CardModel.CanBeGeneratedByModifiers``，
#: ``CardModel.cs``：默认 ``true``，全库只有 8 张覆写成 ``false``）。
#:
#: 用处：`DistinguishedCape`（"从诅咒池随机取 N 张**不重复**的诅咒加入牌组"）
#: 的候选就是 ``CurseCardPool`` 里过 ``c.CanBeGeneratedByModifiers`` 的那批。
#: 不排掉这 8 张，真机发不出的牌会出现在玩家牌组里。
#:
#: ⚠️ 这张表由 `tests/test_run_gates.py::test_not_generated_table_is_current`
#: **扫描源码**钉住 —— 上游新增覆写时测试会红，而不是静默漂移。
NOT_GENERATED_BY_MODIFIERS: frozenset[str] = frozenset({
    "ascenders_bane", "bad_luck", "curse_of_the_bell", "enthralled",
    "folly", "greed", "poor_sleep", "spore_mind",
})

#: ``procure_random_potion````PotionCmd.TryToProcure`` +
#:                      ``PotionFactory.CreateRandomPotionOutOfCombat``（随机获得药水）
#: ``random_deck_cards````StableShuffle`` + ``Take``（Run 层：从牌组随机取 N 张不重复）
#: ``add_random_cards`` ``Rng.NextItem(<卡池列表>)`` + ``Remove`` + ``CreateCard`` +
#:                      ``CardPileCmd.Add(…, Deck)``（从**卡池**随机取 N 张加入牌组）
#: ``choose_one``        ``CardSelectCmd.FromChooseACardScreen`` + ``IChoosable.OnChosen``
#: ==================== ==========================================
ENGINE_OPS = frozenset({
    "damage", "damage_all", "block", "apply_power", "draw", "lose_hp",
    "gain_energy", "add_card", "heal", "gain_stars", "select_card",
    "move_self_to_pile", "channel", "evoke_next", "add_orb_slots",
    "move_all_matching",
    # ``set_power_var``：``PowerCmd.Apply<X>(…)?.SetDamage(…)`` —— 把卡牌的动态变量
    # 写进**能力实例**（`TheBombPower` 的爆炸伤害 / `ToricToughnessPower` 的格挡）。
    "set_power_var",
    # 战斗内也合法：果汁药水加永久上限；LoseMaxHp 是卡牌/药水的削上限
    "gain_max_hp", "lose_max_hp",
    # ⭐ ``CardFactory.GetDistinctForCombat`` + ``AddGeneratedCard(s)ToCombat``：
    # **从卡池随机生成 N 张互不重复的牌**加入手牌（`white_noise` / `infernal_blade`
    # / `toolbox` / `orobic_acid` …）。实现落在 `powers.generate_cards_to_hand`
    # （与能力侧共用同一份）。
    "generate_card",
    # ``upgrade_hand``：``CardCmd.Upgrade(PileType.Hand.GetPile(owner).Cards)``
    # （**整手牌**升级，`Bellows`）。
    "upgrade_hand",
    # ``random_draw_cards``：``StableShuffle(Rng.CombatCardSelection).Take(N)``
    # （战斗内**抽牌堆**随机取 N 张升级，`StoneCracker`）。
    "random_draw_cards",
    # ⭐ 三个"整堆 / 自动打出 / 采购"小算子（`docs/12` §2.16）：
    # ``shuffle``                ``CardPileCmd.Shuffle``（弃牌堆并入抽牌堆后整体洗）
    # ``autoplay_from_draw``     ``CardPileCmd.AutoPlayFromDrawPile``（从抽牌堆顶自动打出 N 张）
    # ``procure_random_potion``  ``PotionCmd.TryToProcure`` +
    #                            ``PotionFactory.CreateRandomPotionOutOfCombat``
    "shuffle", "autoplay_from_draw", "procure_random_potion",
    # ⭐ Run 层：**从卡池随机取 N 张**（不重复）加入牌组
    # （`DistinguishedCape`：先 Curses 张诅咒、再 Cards 张 Apparition）。
    "add_random_cards",
    # ⭐ Run 层：**从牌组随机取 N 张**（不重复）再对它们做一件事。
    # 使用者：`Whetstone` / `WarPaint` / `WarHammer` / `SandCastle` /
    # `FragrantMushroom`（都是 `StableShuffle(Rng.Niche).Take(Cards)` →
    # `foreach CardCmd.Upgrade`）。
    "random_deck_cards",
    # ⭐ **S02 · 显式二选一**：`KnowledgeDemon.CurseOfKnowledgeMove` 造两张临时卡
    # （不在任何牌堆里），玩家选一张后执行它的 `IChoosable.OnChosen`。
    # 实现落在 `core._apply_effects` 的挂起分支 + `core._resolve_selection`
    # 的 `purpose == "choose"`，续跑由 `core._resume_enemy_turn` 负责。
    "choose_one",
})
#: 引擎已实现的**选牌用途**。``transform`` 等还没做，对应卡会被标记残缺。
#: ⭐ ``remember`` 是"**记住这张卡**"：既不移动也不复制，只把克隆快照交给
#: 刚施加的能力实例（`NightmarePower.SetSelectedCard` → 下回合复制 Amount 份入手）。
#: ⭐ ``transform`` 是"**把这张牌原位换成随机的另一张**"
#: （`CardCmd.TransformToRandom` → `CardFactory.GetDefaultTransformationOptions`）。
ENGINE_SELECTION_PURPOSES = frozenset({"discard", "exhaust", "to_hand", "to_draw",
                                       "remember", "transform"})
#: 引擎已实现的**手牌触发器**。``before_hand_draw`` 还没做（Bolas / ThrummingHatchet
#: 的"打过后回到手牌"），对应卡会被标记 `triggers_incomplete`。
ENGINE_TRIGGER_HOOKS = frozenset({"on_turn_end_in_hand"})
#: 目标只能是这两者；``all_enemies`` 由 ``damage_all`` 算子本身表达。
ENGINE_TARGETS = frozenset({"enemy", "self"})

#: 引擎**能算对**的**运行期公式**种类（`docs/12` §2.33）。
#:
#: 抽取器只对"认得出来的 lambda 形状"写 ``calc_kind``；这张表决定引擎算不算得对。
#: 不在表里的 ``calc_kind`` → 卡被拒（理由 ``unsupported_calc``），
#: **不会**静默算成 0。与 ``powers.CALC_KINDS`` 必须一致
#: （`tests/test_calc_formulas.py` 钉住）。
ENGINE_CALC_KINDS = frozenset({
    "own_block", "target_block", "draw_pile_count", "exhaust_count",
    "target_power", "self_power", "all_cards_tag", "card_plays_total",
    "distinct_orb_types", "target_power_floor",
})


#: Run 层里**必须有明确目标卡**的算子。
#:
#: 真机这三个命令的参数都是"某一张牌"，而那张牌往往来自
#: ``Rng.NextItem(候选)`` + ``Remove``（**随机且不重复**）或 ``CardSelectCmd``::
#:
#:     Reflections.TouchAMirror:  c = base.Rng.NextItem(upgradableCards);
#:                                upgradableCards.Remove(c);
#:                                CardCmd.Upgrade(c);          // 升级 4 张，不重复
#:     Pomander:                  foreach (item in upgradableCards) CardCmd.Upgrade(item);
#:
#: 引擎的 Run 层实现只在"候选**恰好 1 张**"或"给了明确卡 id"时才动手，
#: 其余情况只打一句 ``⚠️ …需要玩家选择，跳过`` 然后**继续**（`runeffects.py`）。
#: 于是事件照样被算成"可跑"、遗物照样算"完全复刻"，而收益凭空消失 ——
#: 实测 `reflections`（升级 4 张 / 降级 2 张）就落在这一类。
#: 按 `docs/09` 铁律 5：宁可整条拒绝，也不静默少结算。
RUN_OPS_NEEDING_CARD = frozenset({"upgrade_card", "downgrade_card",
                                   "transform_card"})


def _run_effect_defect(effect: dict) -> str | None:
    """Run 层效果的**额外**门禁（算子在表里还不够）。"""
    if effect.get("op") in RUN_OPS_NEEDING_CARD and not effect.get("card"):
        return (f"{effect['op']} 没有明确目标卡"
                f"（真机是随机/玩家选牌，引擎未实现）")
    if effect.get("op") == "random_deck_cards":
        from .rng import STREAMS
        from .runeffects import RUN_SELECTION_PURPOSES
        if effect.get("purpose") not in RUN_SELECTION_PURPOSES:
            return f"随机取牌的用途 {effect.get('purpose')} 引擎没实现"
        if effect.get("rng") not in set(STREAMS) | {"event"}:
            return f"随机流 {effect.get('rng') or '未定'} 引擎没实现"
        if effect.get("pick") not in ("shuffle", "item"):
            return f"随机取牌的取法 {effect.get('pick') or '未定'} 引擎没实现"
    return None


def _source_reject_reason(record: dict, *, require_cost: bool = True) -> str | None:
    """**算子级**缺口：能否采用源码效果？返回拒绝理由（``None`` = 可以采用）。

    ⚠️ 必须与"**能力级**缺口"分开处理（见 :func:`_missing_powers`）：

    * **算子级**（``discard`` / ``channel`` / 需要选牌…）—— 引擎的分发器会撞上
      不认识的算子。**不能采纳**，退回文本版本，否则 lint 与运行期都会炸。
    * **能力级**（真机有 ``dexterity``，引擎只实现了 3 个能力）—— 算子
      ``apply_power`` 是合法的，只是这个能力不被模拟。**可以采纳**（比文本版
      更接近真机），但要标记 ``effects_incomplete`` 并排除出训练集。

    ``require_cost=False`` 给**药水**用：药水的 record 里根本没有 ``Cost`` 字段，
    照卡牌判据会把 65 瓶**全部**判成"X 费"而拒绝。这是药水**唯一**能复用这张
    算子门禁的入口 —— 缺了它，``_load_potions`` 就只能自己看 ``unsupported``，
    于是"算子引擎没实现、但 ``unsupported`` 是空的"那几瓶会被算成**可用**：
    实测 ``blessing_of_the_forge``（``upgrade_card``）、``cunning_potion``
    （``upgrade_card``）、``glowwater_potion``（``exhaust``）都在"45 瓶可用"里，
    而 ``core._apply_one`` 没有这些算子 —— 一用就 ``ValueError: 未知算子``，
    崩在训练循环里，而且**只在抽到那瓶药时才崩**。
    """
    if record.get("unsupported"):
        return f"未支持的命令 {record['unsupported'][:3]}"
    if record.get("choice_commands"):
        return "需要玩家选牌动作（引擎未实现）"
    # ⭐ **卡级目标**也要门禁，不只是效果级。`TargetType.AllAllies` = 这张牌
    # 作用于**队友**（`EnergySurge` 给队友能量 / `Rally` 给队友格挡 /
    # `BladeSymphony` 给队友造 Shiv）。而抽取器把效果的目标一律落成
    # `self` / `enemy` 两档 —— 于是一张"给队友"的牌会变成"给自己"：
    # 实测 `energy_surge` 在引擎里就是**白给自己能量**，比真机强。
    # 多人专用内容按 `docs/09` 铁律 5 **排除**，不做近似。
    if record.get("target") == "all_allies":
        return "多人专用目标 all_allies（单人局没有队友）"
    if require_cost and record.get("cost") is None:
        return "费用不是字面量（X 费）"
    for effect in record.get("effects") or []:
        if effect.get("needs_runtime"):
            return f"量依赖运行期状态：{effect.get('amount_raw', '')[:40]}"
        if effect["op"] not in ENGINE_OPS:
            return f"引擎未实现的算子 {effect['op']}"
        if effect.get("op") == "select_card":
            if effect.get("purpose") not in ENGINE_SELECTION_PURPOSES:
                return f"引擎未实现的选牌用途 {effect.get('purpose')}"
            if effect.get("from") not in ("hand", "discard", "draw"):
                return f"引擎未实现的选牌来源 {effect.get('from')}"
            # ⭐ **可变张数**必须拒绝。真机写作
            # ``CardSelectorPrefs(prompt, min, max)``，``min = 0`` 时玩家可以选
            # 0..max 任意张（`gamblers_brew` 是 0..∞、`neows_fury` 是 0..手牌空位）。
            # 引擎的 `PendingSelection` 只有"选够 N 张"这一条路 ——
            # `legal_actions` 在挂起时**只给候选下标**，没有"结束选择"动作。
            # 于是 `remaining = max(1, amount)` 会把它们变成：
            #   `gamblers_brew` → "只弃 1 张、且不抽牌"（真机是弃 N 抽 N）
            #   `neows_fury`    → "必须捞 1 张"（真机可以不捞）
            # 两个都不会报错，属于静默的机制替换。
            if not effect.get("amount") and not effect.get("amount_var"):
                return "可变张数选牌（CardSelectorPrefs min=0）：引擎没有『结束选择』动作"
        if effect.get("op") == "channel" and not effect.get("orb"):
            # 泛型实参没抽出来（如 `chaos` 引导**随机**球）—— 不能猜成某种球
            return "充能球 id 为空（随机球？）"
        if effect.get("op") == "move_all_matching":
            if effect.get("from") not in ("hand", "draw", "discard", "exhaust"):
                return f"未实现的集合来源 {effect.get('from')}"
            taken = str(effect.get("take_random") or "")
            if taken and not (taken.isdigit() or "MaxCardsInHand" in taken):
                # 只认"取 N 张"与"补满手牌"两种 —— 运行时猜一个会**多拿牌**，
                # 而多拿牌是明显的强度提升（静默变强）。
                return f"取牌数量认不出：{taken[:24]}"
        if effect.get("op") == "generate_card":
            # ``CardFactory.GetDistinctForCombat`` 的"池 + 牌型"必须**明确**：
            # 认不出就拒绝整张卡。默认猜成"角色池 / 任意牌型"会把
            # `white_noise`（**能力**牌）变成"任意一张牌"，强度差一个数量级，
            # 而且报告里看起来完全正常。
            spec = dict(effect.get("filter") or ())
            # ⚠️ 缺字段时是 **None**，不是 ""。拿 `spec.get(...)` 直接进白名单
            # 会把"没有牌型过滤"（`bundle_of_joy` 的无色任意牌）误判成
            # "牌型认不出" —— 实测就这样把两张本该采纳的卡退回文本版。
            if (spec.get("pool") or "") not in ("", "colorless", "character"):
                return f"生成卡的卡池认不出：{spec.get('pool')}"
            if (spec.get("card_type") or "") not in ("", "attack", "skill", "power"):
                return f"生成卡的牌型认不出：{spec.get('card_type')}"
        if effect.get("op") == "autoplay_from_draw":
            # 位置只有 ``top`` / ``bottom`` 两种实现；`random` 走的是
            # `Rng.CombatCardSelection`，引擎没接（认不出就拒绝，不近似成 top）。
            if effect.get("position") not in ("top", "bottom"):
                return f"自动打出的取牌位置未实现：{effect.get('position')}"
        if effect.get("op") == "procure_random_potion":
            # `amount > 0` = 单次采购；`0` = "填满空槽"（真机的 `while HasOpenPotionSlots`）。
            if int(effect.get("amount") or 0) < 0:
                return "药水采购次数为负"
        if effect.get("target") not in ENGINE_TARGETS | {"all_enemies"}:
            if effect.get("target") == "all_allies":
                # `TargetType.AllAllies` = **给所有队友**（`OneForAll` 那类），
                # 而单人局里"队友"恒为空集。写成"未知目标"会让人以为只是抽取器没认出，
                # 其实是**多人专用内容**（按 `docs/09` 铁律 5 排除，不做近似）。
                return "多人专用目标 all_allies（单人局没有队友）"
            return f"未知目标 {effect.get('target')}"
        if effect.get("op") == "apply_power" and not effect.get("power"):
            return "能力名为空（泛型实参没解析出来）"
    return None


def _missing_powers(record: dict) -> list[str]:
    """**能力级**缺口：这张卡引用了哪些引擎还没实现的能力。

    ⚠️ 判定必须用 ``powers.IMPLEMENTED``（引擎**真正实现了行为**的那张白名单），
    不是 ``POWER_NAMES`` —— 后者是早期硬编码的 3 个能力，只是一个编码词表。
    用错的话，已经实现的能力仍然会被算成缺失，卡牌白白被排除出训练集
    （实测漏算 26 处引用）。
    """
    from .powers import IMPLEMENTED
    implemented = set(IMPLEMENTED) | set(POWER_NAMES)
    return sorted({e.get("power") for e in record.get("effects") or []
                   if e["op"] == "apply_power"
                   and e.get("power") and e["power"] not in implemented})


def _var_values(record: dict, upgraded: bool) -> dict[str, int | None]:
    values: dict[str, int | None] = {}
    for variable in record.get("vars") or []:
        if variable.get("base") is None:
            values[variable["name"]] = None
        else:
            values[variable["name"]] = int(variable["base"]) + (
                int(variable["upgrade_delta"]) if upgraded else 0)
    return values


def _resolve_effects(effects: list[dict],
                     values: dict[str, int | None]) -> tuple[Effect, ...] | None:
    """把源码效果解析成具体数值；有任何一个量算不出来就返回 ``None``。

    ``times > 1`` 展开成多条同名效果 —— 与 ``convert_monster`` 的处理一致，
    引擎的多段伤害就是这么表达的。
    """
    resolved: list[Effect] = []
    for effect in effects:
        # ⚠️ `needs_runtime` 的量**不能**当成 0：那是"看起来正常但一定算错"。
        # 交给调用方判成"解析不出"，进而把整张卡标记残缺。
        if effect.get("needs_runtime"):
            return None
        # X 依赖不是"解析不出"：真机 `ResolveEnergyXValue()` 取的是**打出时消耗的
        # 能量**，由引擎在出牌那一步代入。这里照实标记，量先置 0。
        times_x = bool(effect.get("times_x"))
        amount_x = bool(effect.get("amount_x"))
        amount = effect.get("amount")
        # ⭐ ``set_power_var`` 允许**没有量**：``from`` 说明值取自运行期
        # （``last_block_gain`` = 最近一次实际获得的格挡）。当成 0 就错了。
        set_from = str(effect.get("from") or "")
        if amount is None and effect.get("amount_var"):
            amount = values.get(effect["amount_var"])
            if amount is None and not effect.get("calc_kind"):
                return None
        # ⚠️ 量与变量**都没有**时不能默认成 0：那会把"每次掉 13 点血"变成
        # "掉 0 点"，卡看起来还在、效果却没了（实测 `regret` / `shame` 中过）。
        # 宁可整张卡标残缺，也不能静默变弱。
        #
        # ⭐ 例外：**运行期公式**（`calc_kind`）本来就没有字面量 ——
        # 它的量在结算那一刻算（见 :func:`sts2_sim.powers.calc_amount`）。
        # ⭐ 例外之二：**``choose_one`` 本来就没有量** —— 它是一个选择枢纽，
        # 载荷全在 ``choices`` 里（每个候选各带一组效果）。当成"算不出量"
        # 会把整条二选一判成残缺，KnowledgeDemon 的入口永远接不上。
        is_choice = effect["op"] == "choose_one"
        if (amount is None and not amount_x and not set_from
                and not effect.get("calc_kind") and not is_choice):
            return None
        amount = 0 if amount is None else int(amount)
        target = effect.get("target", "enemy")
        # ⚠️ ``all_enemies`` **不能塌缩成单个目标**：`_apply_one` 对
        # `apply_power` 与 `damage` 都有专门分支能展开到全体（`_opponents_of`）。
        # 塌缩掉之后 `philosophers_stone`（给**所有**敌人 +1 力量）与弹珠袋
        # （给**所有**敌人上易伤）都只会命中第一个目标，日志却照常打印。
        # `damage_all` 只是"全体伤害"的另一种算子写法，目标由引擎展开。
        if effect["op"] == "damage_all":
            target = "enemy"
        # `times_x` 不在这里展开（打出时才知道 X），保留标记交给引擎
        repeat = 1 if times_x else max(1, int(effect.get("times") or 1))
        # ⭐ 次数也可能是个**常量变量**：真机常写
        # `for (int i = 0; i < base.DynamicVars.Repeat.IntValue; i++)` 来把效果重复 N 次
        # （`IceLance` 引导 2 个冰球、`Quadcast` 激发 4 次）。与 `amount_var` 同一套规则 ——
        # 变量查不到就**整张卡标残缺**，绝不默认成 1（那会让卡静默变弱）。
        times_var = effect.get("times_var")
        if not times_x and times_var:
            loop_count = values.get(times_var)
            if loop_count is None:
                return None
            repeat = max(0, int(loop_count))
        # ⭐ `Quadcast`：`EvokeNext(…, i == Repeat - 1)` —— **只有最后一次移除球**。
        # 展开成"前 N-1 次 `dequeue=False`、最后一次 `True`"，与源码的循环逐次对应。
        dequeue_tail = bool(effect.get("dequeue_tail")) and not times_x
        # ⭐ 运行期公式：基准与增量都按**变量名**取（升级增量已由
        # ``_var_values(record, upgraded)`` 加进来），量本身在结算时才算。
        calc_kind = str(effect.get("calc_kind") or "")
        calc_base = calc_extra = 0
        if calc_kind:
            calc_base = int(values.get(effect.get("calc_base_var") or "") or 0)
            extra_var = str(effect.get("calc_extra_var") or "")
            if extra_var:
                calc_extra = int(values.get(extra_var) or 0)
            else:
                # ⭐ **写在 OnPlay 里的表达式公式**（`Hang` 的
                # ``Math.Max(2, powerAmount)``、`Dominate` 的
                # ``target.GetPower<VulnerablePower>()?.Amount ?? 0``）：
                # 真机没有 ``CalculationBase`` / ``ExtraDamage`` 这两个变量，
                # 整个量就是那个表达式 —— 所以基准 0、增量 **1**。
                # 留成 0 会让公式恒等于 0（静默无效，实测踩过）。
                calc_extra = 1
        for index in range(repeat):
            # ⭐ ``choose_one``：候选是**嵌套的效果组**，必须先递归解析完再构造 ——
            # 候选里只要有任意一条落不进 DSL（`_resolve_effects` 返回 None），
            # 整条 `choose_one` 就返回 None（调用方把这张卡/这招标残缺）。
            # 不能"跳过解析不了的那个候选"：那等于**偷偷改掉玩家的选项**。
            choices: tuple[tuple[str, tuple[Effect, ...]], ...] = ()
            if effect["op"] == "choose_one":
                resolved_choices: list[tuple[str, tuple[Effect, ...]]] = []
                for choice in effect.get("choices") or ():
                    sub = _resolve_effects(choice.get("effects") or [], values)
                    if sub is None:
                        return None
                    resolved_choices.append((str(choice.get("card") or ""), sub))
                if not resolved_choices:
                    # 没有候选的二选一是数据错误：真机一定会给出卡片列表。
                    return None
                choices = tuple(resolved_choices)
            resolved.append(Effect(op=effect["op"], amount=amount,
                                   power=effect.get("power") or None,
                                   target=target, card=effect.get("card") or None,
                                   pile=effect.get("pile") or "discard",
                                   position=effect.get("position") or "",
                                   select_from=effect.get("from") or "hand",
                                   purpose=effect.get("purpose") or "",
                                   orb=effect.get("orb") or "",
                                   dequeue=(index == repeat - 1) if dequeue_tail
                                   else bool(effect.get("dequeue", True)),
                                   times_x=times_x, amount_x=amount_x,
                                   unpowered=bool(effect.get("unpowered")),
                                   unblockable=bool(effect.get("unblockable")),
                                   calc_kind=calc_kind,
                                   calc_arg=str(effect.get("calc_arg") or ""),
                                   calc_base=calc_base, calc_extra=calc_extra,
                                   card_filter=tuple(
                                       (k, v) for k, v in (effect.get("filter") or ())),
                                   take_random=effect.get("take_random") or "",
                                   relic=effect.get("relic") or "",
                                   var=effect.get("var") or "",
                                   set_from=set_from,
                                   free=bool(effect.get("free")),
                                   exclude=tuple(str(item) for item
                                                 in (effect.get("exclude") or ())),
                                   upgrade=bool(effect.get("upgrade")),
                                   force_exhaust=bool(effect.get("force_exhaust")),
                                   rng=str(effect.get("rng") or ""),
                                   pick=str(effect.get("pick") or ""),
                                   choices=choices))
    return tuple(resolved)


def _resolve_source_effects(record: dict, upgraded: bool) -> tuple[Effect, ...] | None:
    return _resolve_effects(record.get("effects") or [], _var_values(record, upgraded))


@dataclass(frozen=True)
class PotionDef:
    """药水定义（来自 ``tools/extract_potions.py``，64 个）。

    药水与卡牌**同构**，所以复用同一套 ``Effect`` 与变量解析；
    区别只有三点：**何时可用**（``usage``）、**目标**（``target_type``）、
    以及它属于 Run 层资源而不是牌组。
    """

    pid: str
    name: str
    rarity: str
    #: ``combat_only`` / ``any_time`` / ``automatic``
    usage: str
    #: ``any_player`` / ``any_enemy`` / ``all_enemies`` / ``self``
    target_type: str
    effects: tuple[Effect, ...]
    verified: bool = False
    #: ⚠️ 效果**已知不完整**（含引擎还没实现的命令）→ 必须排除出训练/评测。
    effects_incomplete: bool = False
    #: 所属药水池：``shared`` 或角色名（``ironclad`` …），另有 ``event``。
    #: **随机药水按它筛池**（``PotionFactory.GetPotionOptions``：
    #: 角色药水池 ∪ 共享药水池），来自社区库的结构元数据。
    pool: str = "shared"


#: 药水注册表。
POTIONS: dict[str, PotionDef] = {}

#: **卡池成员**：``{池名: (卡 id, ...)}``。来自 ``tools/extract_pools.py``。
#:
#: ⚠️ 社区库的 ``cards.json`` **没有 color 字段**，光看它建不出卡池。
#: 而卡池是 Run 层的基础设施：**卡牌奖励、商店、事件、战斗内随机生成**
#: 全都从池里抽。没有它，奖励会从"全部卡牌"里抽 —— 铁甲战士能被给到
#: 静默/缺陷的牌（实测就是这样）。
CARD_POOLS: dict[str, tuple[str, ...]] = {
    _BUILTIN_CARD_POOL: tuple(card.cid for card in _CARDS),
}

#: 角色 id → 它的卡池名。真机卡池类名是 ``<Character>CardPool``。
def character_pool(character_id: str) -> str:
    return f"{character_id.title().replace('_', '')}CardPool"


def _load_card_pools(directory, summary: dict) -> None:
    import json
    path = directory / "card_pools.json"
    if not path.exists():
        return
    entries = json.loads(path.read_text(encoding="utf-8"))
    pools = {str(e["pool"]): tuple(str(c) for c in e.get("cards", ()))
             for e in entries if e.get("cards")}
    if pools:
        CARD_POOLS.clear()
        CARD_POOLS.update(pools)
        summary["card_pools"] = len(pools)
        summary["card_pool_members"] = sum(len(v) for v in pools.values())


def _load_potions(directory, summary: dict) -> None:
    """加载药水表：社区库给名字/描述，**效果以源码为准**。"""
    import json
    from .powers import IMPLEMENTED
    codex_path = directory / "potions.json"
    source_path = directory / "potions_source.json"
    if not (codex_path.exists() or source_path.exists()):
        return
    codex = {}
    if codex_path.exists():
        codex = {str(p.get("id", "")).lower(): p
                 for p in json.loads(codex_path.read_text(encoding="utf-8"))}
    source = {}
    if source_path.exists():
        source = {str(p.get("pid", "")).lower(): p
                  for p in json.loads(source_path.read_text(encoding="utf-8"))}

    parsed: dict[str, PotionDef] = {}
    incomplete: list[str] = []
    deprecated: list[str] = []
    for pid in sorted(set(codex) | set(source)):
        record = source.get(pid)
        entry = codex.get(pid) or {}
        if record is None:
            continue                      # 社区库独有但源码没有 → 不猜
        # ⭐ **已从游戏移除的占位类**：``DeprecatedPotion`` 的
        # ``Rarity => PotionRarity.None``，源码注释写着
        # *"Represents a potion that has been removed from the game.
        # Mostly used for the run history."*
        # 它**不是缺口**（没有 `OnUse`、也不该出现在任何池里），
        # 所以既不加载、也不计入"残缺" —— 否则它会一直占着一个名额，
        # 让人以为"还差一瓶没做"。
        if str(record.get("rarity") or "") == "none":
            deprecated.append(pid)
            continue
        effects = _resolve_effects(record.get("effects") or [],
                                   _var_values(record, False))
        name = entry.get("name") or record.get("class") or pid
        # ⭐ **算子门禁必须走和卡牌同一张表**（`_source_reject_reason`）。
        #
        # 以前这里只看 `unsupported` / `choice_commands` 两个字段，于是
        # "抽取器认得出来、但引擎没实现那个算子" 的药水会被算成**可用**：
        #   blessing_of_the_forge → `upgrade_card`、cunning_potion → `upgrade_card`、
        #   glowwater_potion → `exhaust`
        # 它们会进 `legal_actions` 的动作空间，然后 `use_potion` 在
        # `core._apply_one` 里撞上 `ValueError: 未知算子` —— 训练中途崩，
        # 而且只在抽到那瓶药的时候崩（最难复现的一类）。
        # `require_cost=False`：药水没有 `Cost` 字段，用卡牌判据会全员误杀。
        reason = _source_reject_reason(record, require_cost=False)
        # PowderedDemise.OnUse 只施加 DemisePower，伤害在能力回调里。
        # 仅解析出 apply_power 不能证明药水可执行，缺行为时必须整瓶拒绝。
        missing_powers = any(effect.op == "apply_power"
                             and effect.power not in IMPLEMENTED
                             for effect in effects or ())
        if reason or effects is None or not effects or missing_powers:
            # 建不出来 / 引擎执行不了 → 保留条目但标记不完整，别静默当成"没有效果"
            parsed[pid] = PotionDef(
                pid=pid, name=name, rarity=str(record.get("rarity") or ""),
                usage=str(record.get("usage") or ""),
                target_type=str(record.get("target_type") or ""),
                effects=effects or (), verified=False, effects_incomplete=True,
                pool=str(entry.get("pool") or "shared"))
            incomplete.append(pid)
            continue
        parsed[pid] = PotionDef(
            pid=pid, name=name, rarity=str(record.get("rarity") or ""),
            usage=str(record.get("usage") or ""),
            target_type=str(record.get("target_type") or ""),
            effects=effects, verified=False, effects_incomplete=False,
            pool=str(entry.get("pool") or "shared"))
    if parsed:
        POTIONS.clear()
        POTIONS.update(parsed)
        summary["potions"] = len(parsed)
        summary["potions_usable"] = len(parsed) - len(incomplete)
        summary["potions_incomplete"] = len(incomplete)
        # 已从游戏移除的占位类（`rarity == "none"`）：**不是缺口**，
        # 单独计数以便核对（`DeprecatedPotion.cs` 的类注释）。
        summary["potions_deprecated"] = sorted(deprecated)


#: 遗物钩子 → **战斗内时机**。键是引擎认识的时机名，值是数据里的钩子名。
#:
#: 时机名与 :func:`sts2_sim.relics.apply_hook` 的取值一一对应，
#: 顺序照抄真机 ``CombatManager.StartTurn`` + ``SetupPlayerTurn``。
HOOK_TIMING: dict[str, str] = {
    "combat_start": "BeforeCombatStart",
    "before_side_turn_start": "BeforeSideTurnStart",
    "after_energy_reset": "AfterEnergyReset",
    "after_player_turn_start": "AfterPlayerTurnStart",
    "after_side_turn_start": "AfterSideTurnStart",
    # ⭐ 玩家**打出任意一张牌之后**。这条时机只有在"牌型条件"能建模之后才敢开：
    # `GamePiece`（仅能力牌）/ `LostWisp` / `Permafrost` / `RainbowRing` /
    # `DaughterOfTheWind` 的行为就是"只在某类牌上触发" —— 条件丢掉就是**每张牌都触发**。
    # 现在抽取器会把 `CardType.X` 抽成 `card_type` 守卫，由 `relics._applies` 求值。
    #
    # ⚠️ 仍然**不能**开的是那些带"取模计数"的（`Kunai` 每 3 张攻击牌 +1 敏捷、
    # `IronClub` 每 N 张抽 1 张…）—— 它们会被 `STATEFUL_CONDITIONS` 的
    # `modulo_counter` 判成 stateful 而拒绝采纳，如实算作缺口。
    # 好消息是**拒绝是安全的**：不会无条件生效。
    "after_card_played": "AfterCardPlayed",
    # ---- Run 层（``docs/09`` §19）----------------------------------------
    # ⚠️ 这三个是**牌组 / 金币 / 最大生命**层面的效果，不走战斗内分发，
    # 由 `runeffects.apply_run_effects` 结算。
    "obtained": "AfterObtained",
    "combat_end": "AfterCombatEnd",
    "combat_victory": "AfterCombatVictory",
    #
    # ⚠️ **`AfterCardPlayed` 与 `AfterSideTurnEnd` 暂时不能加进来**（虽然总线
    # 已经支持这两个时机）。原因：这两个时机上的遗物钩子几乎都带**条件**，
    # 而抽取器目前建不出这些条件 —— 一旦采纳就是**无条件生效**，静默变强。
    # 实测证据（全部来自反编译源码）：
    #
    #   Kunai.AfterCardPlayed         仅攻击牌 且 AttacksPlayedThisTurn % N == 0 → +1 敏捷
    #   OrnamentalFan.AfterCardPlayed 同上 → +4 格挡
    #   LetterOpener.AfterCardPlayed  仅技能牌 且每 N 张 → 5 点伤害
    #   IronClub.AfterCardPlayed      CardsPlayed % N == 0 → 抽 1 张
    #   GamePiece.AfterCardPlayed     仅**能力牌** → 抽牌
    #   ParryingShield.AfterSideTurnEnd  格挡 ≥ X 时对**随机**敌人造成 6 点伤害
    #   LunarPastry.AfterSideTurnEnd  participants.Contains(自己) 守卫
    #
    # 抽取器现在只认"回合数守卫"（``TurnNumber``）与少数跨回合计数器，
    # 认不出"牌型条件 / 取模计数 / 随机目标" —— 于是 guards 变成空元组，
    # 效果被当成**每张牌都触发**。这不是小偏差：苦无会从"每 3 张攻击牌 +1 敏捷"
    # 变成"每张牌 +1 敏捷"。
    #
    # 要启用这两个时机，先做**两件事**（见 docs/12 §2.6「遗物口径」）：
    #   1. `tools/extract_relics.py` 把牌型条件 / 取模计数 / 随机目标识别成
    #      可求值的守卫（认不出就标 `stateful` 拒绝采纳，绝不默认无条件）；
    #   2. `sts2_sim/relics.py` 在求值层真的实现这些守卫。
    # 在那之前，这 18 个遗物**照实算作缺口**。
}

#: 哪些时机属于 **Run 层**（效果作用于牌组/角色属性，而不是战斗内的单位）。
RUN_TIMINGS: frozenset[str] = frozenset({"obtained", "combat_end", "combat_victory"})

#: ``AfterRoomEntered`` 单列：它在**所有**房间类型都触发，只有战斗范围才算开局效果。
ROOM_ENTERED_HOOK = "AfterRoomEntered"

#: 反查：数据里的钩子名 → 引擎的时机名。
#: ⚠️ 别用 ``HOOK_TIMING.get(hook_name)`` —— 那是反的，会一条都匹配不上，
#: 而且只会表现为"遗物数量变少"，不报错。
HOOK_TIMING_BY_NAME: dict[str, str] = {
    hook: timing for timing, hook in HOOK_TIMING.items()}

#: 遗物钩子里**引擎已经实现**的部分。
#:
#: 其余钩子照实记录进 ``RelicDef.unmodeled_hooks`` —— 报告里能看到还差多少，
#: 而不是"看起来已经一致"。
MODELED_RELIC_HOOKS = frozenset({
    "ModifyMaxEnergy", "ModifyHandDraw",          # 每回合能量 / 抽牌
    ROOM_ENTERED_HOOK,                            # 战斗开局（按房间范围筛）
} | set(HOOK_TIMING.values()))

#: **战斗阶段**的钩子。只要有任何一个没复刻出来，这个遗物的数值部分就**不能采纳**：
#: 那会让它比真机更强或更弱。``bread`` 就是例子 —— ``ModifyMaxEnergy`` 抽得出
#: "+1 能量（第 2 回合起）"，但它的另一半在第 1 回合 ``PlayerCmd.LoseEnergy(2)``
#: 没实现，只采纳前一半等于把一个负面遗物变成纯收益。
COMBAT_PHASE_HOOKS = frozenset({
    "BeforeCombatStart", ROOM_ENTERED_HOOK, "AfterSideTurnStart",
    "BeforeSideTurnStart", "AfterPlayerTurnStart", "AfterEnergyReset",
    "AfterSideTurnEnd", "BeforeSideTurnEnd", "AfterCardPlayed",
    "BeforeCardPlayed", "BeforeHandDraw", "AfterBlockCleared",
    "AfterCombatEnd", "AfterCombatVictory", "AfterDamageReceived",
    # 查询型：靠**返回值**改变战斗行为（没有命令，照样是缺口）。
    # `velvet_choker` 的"每回合最多 6 张"就在这里。
    "ShouldPlay",
})

#: ``BeforeCombatStart`` / ``AfterRoomEntered`` 里能算成"战斗开局效果"的房间范围。
COMBAT_ROOM_SCOPES = frozenset({"combat", "elite", "boss"})

#: **Run 层**的遗物时机：效果作用于牌组/金币/最大生命，不在战斗里结算。
#: 它们能用的选牌用途是 ``runeffects.RUN_SELECTION_PURPOSES``
#: （``removal`` / ``upgrade`` / ``transform``），与战斗内那套
#: （``discard`` / ``exhaust`` / ``to_hand`` / ``to_draw``）**不是同一批**。
#: 搞混的后果见 :func:`_relic_hook` 里的注释。
RUN_LEVEL_TIMINGS = frozenset({"obtained", "combat_end", "combat_victory"})


@dataclass(frozen=True)
class RelicHook:
    """遗物的一个**已复刻**的钩子：什么时机、带什么条件、做什么。"""

    timing: str
    effects: tuple[Effect, ...]
    #: 守卫（"命中则不生效"）。带回合条件的效果只在满足时触发。
    guards: tuple[tuple[str, object], ...] = ()
    #: 只在哪种战斗生效（``elite`` / ``boss``），``None`` = 任意战斗
    room: str | None = None
    note: str = ""


@dataclass(frozen=True)
class RelicDef:
    """遗物定义（来自 ``tools/extract_relics.py``，299 个）。

    遗物分三层产出，**每层都如实标注覆盖**：

    * 身份层（全量）：名字 / 稀有度 / 变量表 / 重写了哪些钩子
    * **数值层**：``max_energy`` / ``hand_draw`` —— 直接改变每回合的能量与手牌
    * 效果层：开局与回合开始钩子里能落进效果 DSL 的部分

    剩余钩子进 ``unmodeled_hooks``。**不做的部分必须可见**，否则覆盖率会
    自己骗自己 —— 102 种钩子、212 处还没实现，这是现状。
    """

    rid: str
    name: str
    rarity: str
    #: 变量名 → 值（算不出的留 ``None``）
    values: dict[str, int | None]
    #: 每回合能量增量：``(变量名, 符号, 守卫)``。守卫是 ``(种类, 值, 极性)``，
    #: 极性决定"命中则生效"还是"命中则不生效"（见 :mod:`sts2_sim.relics`）。
    max_energy: tuple[str, int, tuple[tuple[str, object], ...]] | None = None
    #: 每回合抽牌增量，结构同上。可以是负数（``big_mushroom`` 第 1 回合少抽 2 张）。
    hand_draw: tuple[str, int, tuple[tuple[str, object], ...]] | None = None
    #: 已复刻的钩子（按 :data:`HOOK_TIMING` 归类）
    hooks: tuple[RelicHook, ...] = ()
    #: 引擎**没实现**的钩子（照实记录，供覆盖率报告）
    unmodeled_hooks: tuple[str, ...] = ()
    #: 上面那些钩子的**类别**：``(钩子名, 类别)``，类别 ∈
    #: ``commands``（体里有命令 → 真的缺行为）/ ``query``（靠返回值改行为）/
    #: ``bookkeeping``（只写自己的私有字段与状态显示 → **本身不改变游戏**）。
    #: 分开的理由见 ``tools/extract_relics.classify_hook``：把 bookkeeping 也算成
    #: "遗物缺行为"会**高估缺口**（审计 F11）。
    unmodeled_hook_kinds: tuple[tuple[str, str], ...] = ()
    #: 没实现的钩子里**属于战斗阶段**的那些 —— 它们会让数值部分失真
    combat_gaps: tuple[str, ...] = ()
    #: 所属池：``shared`` 或角色名（``ironclad`` …）。**遗物抓包按它筛**
    #: （``RelicGrabBag.Populate``：共享池 ∪ 角色池）。来自社区库的结构元数据。
    pool: str = "shared"

    @property
    def fully_modeled(self) -> bool:
        return not self.unmodeled_hooks

    @property
    def has_behavior(self) -> bool:
        return bool(self.hooks or self.max_energy or self.hand_draw)

    @property
    def effectful_gaps(self) -> tuple[str, ...]:
        """**真正缺失行为**的钩子：``commands`` / ``query`` / 认不出类别的。

        ``bookkeeping`` 不算 —— 它们只是给同类里别的钩子记数，
        单独实现出来不改变任何数值。
        """
        return tuple(name for name, kind in self.unmodeled_hook_kinds
                     if kind != "bookkeeping") or (
            # 老数据没有类别字段时退回保守口径：全部算缺口（宁可高估，不可漏报）
            self.unmodeled_hooks if not self.unmodeled_hook_kinds else ())

    @property
    def bookkeeping_only_gaps(self) -> bool:
        """有缺口，但**全部**是记账型 —— 没有玩家可见的行为缺失。"""
        return (bool(self.unmodeled_hooks)
                and bool(self.unmodeled_hook_kinds)
                and not self.effectful_gaps)


#: 遗物注册表。
RELICS: dict[str, RelicDef] = {}


def _relic_numeric(record: dict, field: str, values: dict[str, int | None],
                   blocked: bool,
                   ) -> tuple[str, int, tuple[tuple[str, object], ...]] | None:
    """数值修饰 → ``(变量, 符号, 守卫)``；条件认不出或值算不出就返回 ``None``。

    守卫来自源码的 **early-return**：``if (TurnNumber < 3) return amount;``
    意思是"第 3 回合**起**才生效"。语义照原样带着，交给 :mod:`sts2_sim.relics`
    在具体回合上求值 —— 这里不提前化简，免得把条件丢掉。
    """
    raw = record.get(field)
    if not raw or raw.get("unresolved_guard") or blocked:
        return None
    var = str(raw.get("var") or "")
    if not var or values.get(var) is None:
        return None
    # 守卫带**极性**：数值钩子几乎都是 `if (cond) return amount;`（排除式），
    # 但源码里也可能出现包含式，照原样带着交给求值层。
    guards = tuple((str(g["kind"]), g["value"], str(g.get("polarity") or "skip_when"))
                   for g in raw.get("guards") or ())
    return var, int(raw.get("sign", 1)), guards


def _relic_hook(timing: str, hook_name: str, body: dict,
                values: dict[str, int | None]) -> RelicHook | None:
    """把一个钩子体转成 :class:`RelicHook`；转化不了就返回 ``None``。

    ⚠️ **抽不干净就不采纳**。三种情况会返回 ``None``：效果里有引擎没有的算子、
    钩子里还有没识别的语句、以及条件算不出（跨回合计数器 / 地图点类型）。
    房间范围只对 ``AfterRoomEntered`` 有意义 —— 它在所有房间都会触发。
    """
    if not body or not body.get("effects"):
        return None
    if body.get("unsupported") or body.get("choices"):
        return None
    scope = body.get("scope") or {}
    if scope.get("stateful") or scope.get("map_point"):
        return None
    room = scope.get("room")
    if hook_name == ROOM_ENTERED_HOOK:
        if room not in COMBAT_ROOM_SCOPES:
            return None                 # 商店回血 / 休息点回血 —— 不是战斗效果
    elif room in COMBAT_ROOM_SCOPES:
        pass                            # `BeforeCombatStart` 带 Boss 判断（万用表）
    else:
        room = None                     # `BeforeCombatStart` 只在战斗里跑，any 即任意
    resolved = _resolve_effects(body["effects"], values)
    if resolved is None or not resolved:
        return None
    # ⚠️ **两套算子表**：战斗内时机用 `ENGINE_OPS`，Run 层时机用
    # `runeffects.RUN_OPS`（作用于牌组/金币/最大生命，不在战斗里结算）。
    # 用错表的结果是"这一层的效果整批被拒"，而且只表现为"遗物没有行为"。
    from .runeffects import RUN_OPS, RUN_SELECTION_PURPOSES
    allowed = ENGINE_OPS | RUN_OPS
    if any(e.op not in allowed for e in resolved):
        return None                     # 引擎没这个算子 → 不猜
    # ⭐ **选牌用途也必须门禁**。少了这一步，一个"从牌组里选一张附魔"的钩子
    # 会被判成"完全复刻" —— 而运行期 `runeffects` 遇到不支持的用途只是
    # 打一行日志跳过（`⚠️ 选牌用途 'enchant' 引擎没实现，跳过`）。
    # 于是报告说"这个遗物做好了"，实际玩起来它什么也不做。
    # 实测 `royal_stamp`（皇家印章）就是这样被算成"完全复刻"的。
    # 战斗内时机用战斗侧的用途表，Run 层时机用 Run 层的 —— 两套本来就不同。
    allowed_purposes = (RUN_SELECTION_PURPOSES if timing in RUN_LEVEL_TIMINGS
                        else ENGINE_SELECTION_PURPOSES)
    if any(e.op == "select_card" and e.purpose not in allowed_purposes
           for e in resolved):
        return None
    # ⭐ Run 层里"必须指哪张牌"的算子（升级 / 降级 / 转化）：没有明确目标卡时，
    # 运行期只会打一句 ⚠️ 然后跳过 —— 遗物会被算成"完全复刻"却什么都不做。
    if any(e.op in RUN_OPS_NEEDING_CARD and not e.card for e in resolved):
        return None
    # ⭐ `random_deck_cards`：用途 / 随机流 / 取法三样都得是引擎认识的那种，
    # 否则 `runeffects` 只会打一行 ⚠️ 跳过，而遗物照样算"完全复刻"。
    # ⚠️ import 必须在使用**之前**：函数内的 `import … as X` 会让 `X` 成为
    # **局部名**，在赋值前使用就是 `NameError`（不会退回模块级）。
    from .rng import STREAMS as RUN_STREAMS
    #: ``event`` = 用**调用方传进来的事件流**（`events.apply_option` 传
    #: `run.rng`），不是某条命名流 —— 事件里的 `StableShuffle(base.Rng)`
    #: 抽出来就是它。
    allowed_streams = set(RUN_STREAMS) | {"event"}
    if any(e.op == "random_deck_cards"
           and (e.purpose not in RUN_SELECTION_PURPOSES
                or e.rng not in allowed_streams
                or e.pick not in ("shuffle", "item")) for e in resolved):
        return None
    # ⭐ **守卫必须是引擎算得出的种类**。算不出的守卫不会报错 ——
    # `relics.hooks_at` 看到 `_applies(...) is not True` 就**静默跳过**，
    # 于是报告说这个遗物"完全复刻"，实际玩起来它什么也不做。
    # 名单是 `relics.EVALUABLE_GUARDS`（与求值实现放在一起，改一处即可）。
    from .relics import EVALUABLE_GUARDS
    if any(str(g["kind"]) not in EVALUABLE_GUARDS
           for g in scope.get("turn_guards") or ()):
        return None
    guards = tuple((str(g["kind"]), g["value"], str(g.get("polarity") or "skip_when"))
                   for g in scope.get("turn_guards") or ())
    return RelicHook(timing=timing, effects=tuple(resolved), guards=guards,
                     room=room, note=_describe_effects(resolved))


def _parse_effects_by_counter(source: dict, eid: str, move,
                              rejected: list[str]) -> dict:
    """解析招式的 ``effects_by_counter``（**按实例计数器选载荷**）。

    真机出处：``KnowledgeDemon.CurseOfKnowledgeMove`` 用
    ``_curseOfKnowledgeSets[CurseOfKnowledgeCounter]`` 决定这一轮给哪两张卡、
    ``Disintegration`` 是 6 / 7 / 8 哪一档（``KnowledgeDemon.cs:157-176``）。

    ⚠️ 任一组的候选解析不出（``_resolve_effects`` 返回 ``None``）就**整条不采纳**
    并记账：宁可这一招保持残缺、被准入挡下，也不能少给玩家一个候选 ——
    那等于偷偷改掉选项，而报告是干净的。
    """
    spec = source.get("effects_by_counter") or {}
    name = str(spec.get("counter") or "")
    if not name:
        return {}
    groups: list[tuple[int, tuple[Effect, ...]]] = []
    for group in spec.get("groups") or ():
        sub = _resolve_effects(group.get("effects") or [], {})
        if sub is None:
            rejected.append(f"{eid}.{move.mid}")
            return {}
        groups.append((int(group.get("value", 0)), sub))
    if not groups:
        return {}
    return {"effects_by_counter": (name, tuple(groups))}


def _with_intent_damage(move, effects: tuple[Effect, ...]) -> tuple[Effect, ...]:
    """把**意图上的伤害**补成效果。

    ⚠️ 这一步不能省，否则是灾难性的静默回归：真机的怪物攻击伤害来自
    **意图**（``SingleAttackIntent(Damage)`` → ``DamageCmd.Attack(Damage).FromMonster(this)``），
    招式处理函数体里**没有**独立的伤害命令。抽取器把 ``DamageCmd.Attack`` 当成
    装饰（它只带数值、不带独立语义）之后，攻击招式的效果列表会变成空的 ——
    而 ``core._effective_effects`` 只在**已有** ``damage`` 效果时才用意图数值覆盖，
    于是敌人**永远打不出伤害**。实测：221 个攻击招式里 197 个丢了伤害。

    数值仍然来自源码：``move_values``（``GetValueIfAscension`` 的进阶/普通两档）
    由出招状态机那一步提供，这里只负责把"这一招是攻击"这件事表达成效果。
    """
    if not move.value or move.value <= 0:
        return effects
    if any(e.op in ("damage", "damage_all") for e in effects):
        return effects                     # 源码里已经显式抽到了伤害，别叠加
    times = max(1, int(move.times or 1))
    attack = tuple(Effect(op="damage", amount=int(move.value), target="enemy")
                   for _ in range(times))
    return attack + effects


#: 真机**意图类名** → 本引擎的 ``MoveDef.intent``。
#:
#: ⚠️ 只映射**语义等价**的两种攻击意图：``SingleAttackIntent`` 与
#: ``MultiAttackIntent`` 都是"打伤害"，段数由 ``times`` 表达。
#: 其余类名（``StatusIntent`` / ``DeathBlowIntent`` …）**不在这里猜** ——
#: 猜错的后果是"这一招的意图显示错误"，而意图是玩家据以决策的公开信息。
#: 映射不了的会落到 :func:`_enemy_move_gaps` 里被如实报出来。
MONSTER_INTENT_KINDS: dict[str, str] = {
    "SingleAttackIntent": "attack",
    "MultiAttackIntent": "attack",
}


def _state_mids(state_id: str, handler: str) -> set[str]:
    """一个 move 状态在招式表里可能对应的 mid 写法。

    两种命名体系并存：状态机里的状态 id 是 ``CLAW_MOVE``，而招式表的 mid
    来自社区库（``claw``），处理函数名又是 ``ClawMove``。三种都要能对上。
    """
    return {state_id.lower(), state_id.lower().removesuffix("_move"),
            str(handler or "").lower()}


def _move_state_gaps(enemy) -> list[tuple[str, str]]:
    """这只怪**状态机里指向了、但招式表里没有**的 move 状态。

    返回 ``[(状态 id, 处理函数名)]``。空 = 每个状态都能落到一个 ``MoveDef``。

    ⚠️ 为什么必须有这个检查：``_choose_intent`` 找不到对应招式时是**抛错**的
    （那是对的，静默退化更糟）。但如果内容表里就有这个洞，错误会等到
    "这只怪真的出场"才炸 —— 实测 ``torch_head_amalgam`` 在整局 fuzz 的
    seed=134 直接把 Run 层打断。所以缺口必须在**加载期**被发现。
    """
    ai = getattr(enemy, "ai_record", None) or {}
    known: set[str] = set()
    for move in enemy.moves:
        known |= _state_mids(move.mid, move.mid)
    gaps: list[tuple[str, str]] = []
    for state_id, state in (ai.get("states") or {}).items():
        if state.get("kind") != "move":
            continue
        handler = str(state.get("move") or "")
        if not (_state_mids(state_id, handler) & known):
            gaps.append((state_id, handler))
    return gaps


def _drop_encounters_with_gaps(broken: set[str]) -> int:
    """把**含有出招缺口怪**的遭遇从遭遇表里摘掉，返回摘掉的组数。

    ``docs/13`` §7 的"闭包可执行"：一个遭遇里只要有一只怪的招式表不全，
    整场战斗就不可信 —— 而且它会在运行时抛错打断整局。
    宁可少一组遭遇（明确标成受限范围），也不要静默带洞跑。
    """
    removed = 0
    for kind, table in list(ENCOUNTERS.items()):
        kept = tuple(members for members in table
                     if not (set(members) & broken))
        removed += len(table) - len(kept)
        ENCOUNTERS[kind] = kept
    return removed


def _enemy_move_gaps() -> dict[str, list[str]]:
    """全部怪物里"状态机指向了招式表却没有"的缺口。空 = 闭包完整。"""
    gaps: dict[str, list[str]] = {}
    for eid, enemy in sorted(ENEMY_DB.items()):
        missing = _move_state_gaps(enemy)
        if missing:
            gaps[eid] = [f"{state_id}({handler})" for state_id, handler in missing]
    return gaps


def _load_monster_moves(directory, summary: dict) -> None:
    """用**源码抽取的招式效果**覆盖社区库版本（``docs/09`` §15 的 3.4）。

    为什么必须覆盖：招式表来自社区库，而社区库对很多招式**只记了意图、没记效果** ——
    实测 39 条 buff/debuff 招式是空的（`frail` 在源码里有 14 处，社区库 0 处）。
    于是这些怪在模拟器里"那一回合什么也不做"，而意图还显示着 buff/debuff。

    规则（与其它内容层一致）：

    * 源码抽干净了（没有未支持命令、没有需要玩家选择、量都解析得出）→ **采用源码**
    * 抽不干净 → 保留社区库版本，并把这条**记进报告**，绝不半信半疑地混用
    """
    import json
    from dataclasses import replace as _replace

    path = directory / "monster_moves.json"
    if not path.exists():
        return
    records = {str(r.get("eid", "")).lower(): r
               for r in json.loads(path.read_text(encoding="utf-8"))}

    adopted = kept = 0
    rejected: list[str] = []
    kills_self_patched: list[str] = []
    created: list[str] = []
    for eid, enemy in list(ENEMY_DB.items()):
        record = records.get(eid)
        if not record:
            continue
        ai = getattr(enemy, "ai_record", None) or {}
        # 状态 id → 处理函数名；招式表里的 mid 要能对上状态
        mid_to_handler: dict[str, str] = {}
        for state_id, state in (ai.get("states") or {}).items():
            if state.get("kind") != "move":
                continue
            handler = str(state.get("move") or "")
            for key in (state_id.lower(), state_id.lower().removesuffix("_move"),
                        handler.lower()):
                mid_to_handler[key] = handler

        new_moves = []
        changed = False
        for move in enemy.moves:
            handler = mid_to_handler.get(move.mid.lower())
            source = (record.get("moves") or {}).get(handler or "")
            # ⭐ **按计数器选载荷**的招式：载荷不在 `effects` 里，而在
            # `effects_by_counter`（KnowledgeDemon 的三组候选 + 6/7/8）。
            counter_field: dict = {}
            if source and source.get("effects_by_counter"):
                counter_field = _parse_effects_by_counter(source, eid, move, rejected)
            if source and source.get("counter_increment"):
                counter_field["counter_increment"] = str(source["counter_increment"])
                if source.get("counter_increment_live_only"):
                    counter_field["counter_increment_live_only"] = True
            has_counter_payload = bool(counter_field.get("effects_by_counter"))
            effects = None
            if (source and not source.get("unsupported") and not source.get("choices")
                    and not has_counter_payload):
                effects = _resolve_effects(source.get("effects") or [], {})
                if effects is not None:
                    effects = _with_intent_damage(move, effects)
            if effects is None and source and source.get("effects"):
                # 抽出来了但落不进 DSL（量依赖运行期等）→ 如实记账
                rejected.append(f"{eid}.{move.mid}")
            flags = {}
            if source:
                # 自爆 / 逃跑以**源码**为准（社区库在这一项上也不全）
                if source.get("kills_self") and not move.kills_self:
                    flags["kills_self"] = True
                    kills_self_patched.append(f"{eid}.{move.mid}")
                elif source.get("kills_self") is False and move.kills_self:
                    flags["kills_self"] = False
            if effects is not None and effects != move.effects:
                new_moves.append(_replace(move, effects=effects, **flags,
                                          **counter_field))
                changed = True
            elif flags or counter_field:
                new_moves.append(_replace(move, **flags, **counter_field))
                changed = True
            else:
                new_moves.append(move)
            # ⚠️ 计数**不能**只在"值变了"时加：那样同一份内容加载两次会得到
            # 两个不同的数字（第一次 239、第二次 75），统计就没法当台账用。
            # 判据是"这条招式**采纳了源码**"，与当前值是否相同无关。
            if effects is not None:
                adopted += 1
            else:
                kept += 1
        if changed:
            ENEMY_DB[eid] = _replace(enemy, moves=tuple(new_moves))

        # ⭐ **补出状态机需要、而社区库没有的招式**。
        #
        # 社区库的招式列表有时缺项或命名对不上：实测 ``torch_head_amalgam``
        # 的状态机有 5 个 move 状态（首个是 ``STRONG_TACKLE_MOVE``，伤害 26），
        # 而社区库给的是 ``tackle / tackle_2 / beam / tackle_3 / tackle_4`` ——
        # **根本没有 26 点那一下**，于是运行到第 1 招就抛"状态机指向未知招式"，
        # 整局被打断。
        #
        # 补的依据**全部来自源码**（不是猜）：``move_values`` 里那条意图与数值、
        # 以及 ``monster_moves.json`` 的 kills_self。意图类名映射不到的**不补**，
        # 交给 :func:`_enemy_move_gaps` 报出来并在加载期挡住。
        enemy = ENEMY_DB[eid]
        gaps = _move_state_gaps(enemy)
        if gaps:
            extra: list = []
            unresolved: list[str] = []
            values = ai.get("move_values") or {}
            for state_id, handler in gaps:
                info = values.get(handler) or {}
                kind = MONSTER_INTENT_KINDS.get(str(info.get("intent") or ""))
                value = info.get("normal")
                if kind is None or not value:
                    unresolved.append(f"{state_id}({handler})")
                    continue
                times = max(1, int(info.get("times") or 1))
                effects = tuple(Effect(op="damage", amount=int(value), target="enemy")
                                for _ in range(times))
                body = (record.get("moves") or {}).get(handler) or {}
                extra.append(MoveDef(
                    mid=state_id.lower(), name=state_id, intent=kind,
                    value=int(value), times=times, effects=effects,
                    kills_self=bool(body.get("kills_self"))))
                created.append(f"{eid}.{state_id}")
            if extra:
                ENEMY_DB[eid] = _replace(enemy, moves=tuple(enemy.moves) + tuple(extra))
            if unresolved:
                rejected.extend(f"{eid}.{state_id}" for state_id, _h in gaps
                                if f"{state_id}({_h})" in unresolved)

    summary["monster_move_effects_from_source"] = adopted
    summary["monster_move_effects_from_codex"] = kept
    summary["monster_move_effects_rejected"] = len(rejected)
    if rejected:
        summary["monster_move_effects_rejected_examples"] = rejected[:10]
    if kills_self_patched:
        summary["monster_kills_self_from_source"] = kills_self_patched
    if created:
        summary["monster_moves_created_from_source"] = created


def _load_relics(directory, summary: dict) -> None:
    import json
    source_path = directory / "relics_source.json"
    if not source_path.exists():
        return
    codex_path = directory / "relics.json"
    codex: dict[str, dict] = {}
    if codex_path.exists():
        codex = {str(r.get("id", "")).lower(): r
                 for r in json.loads(codex_path.read_text(encoding="utf-8"))}
    entries = json.loads(source_path.read_text(encoding="utf-8"))

    parsed: dict[str, RelicDef] = {}
    unresolved_numeric: list[str] = []
    for record in entries:
        rid = str(record.get("rid", "")).lower()
        if not rid:
            continue
        values = _var_values(record, False)
        body_hooks: dict = record.get("hooks") or {}
        declared = set(record.get("declared_hooks") or ())
        declared_kinds: dict = record.get("declared_hook_kinds") or {}

        # 没复刻出来的钩子：重写了但不在已实现名单里的 + 抽不干净的
        gaps: set[str] = {name for name in declared
                          if name not in MODELED_RELIC_HOOKS}
        adopted: list[RelicHook] = []
        for hook_name, body in body_hooks.items():
            timing = HOOK_TIMING_BY_NAME.get(hook_name)
            if hook_name == ROOM_ENTERED_HOOK:
                timing = "combat_start"
            if timing is None:
                gaps.add(hook_name)
                continue
            converted = _relic_hook(timing, hook_name, body, values)
            if converted is None:
                # 只有"本来该有东西却抽不出来"才算缺口，空钩子不算
                if body.get("effects") or body.get("unsupported") or body.get("choices"):
                    gaps.add(hook_name)
                continue
            adopted.append(converted)

        # ⭐ 缺口**按类别**记录下来（审计 F11 的同类问题）：
        # `bookkeeping` 型钩子（只写自己的私有字段 / 状态显示）**本身不改变游戏**，
        # 把它们算成"遗物缺行为"会高估缺口 —— 实测 109 处"体是空的"里绝大多数是这一类。
        # 类别来自抽取器（`tools/extract_relics.classify_hook`），这里只做汇总；
        # 认不出类别的（老数据 / 抽不出体）按 `unknown` 处理，**算作真缺口**。
        kinds: dict[str, str] = {}
        for name in gaps:
            kind = (body_hooks.get(name) or {}).get("kind") or declared_kinds.get(name)
            kinds[name] = str(kind) if kind else "unknown"

        # ⚠️ 只要还有**战斗阶段**的钩子没复刻，数值半边就不采纳（否则会把
        # "有代价的收益"变成纯收益）。
        #
        # ⭐ `ShouldPlay` 必须在这个集合里：`velvet_choker`（天鹅绒项圈）=
        # "+1 能量，但每回合最多打出 6 张牌"。它的 `AfterCardPlayed` 只是计数
        # （记账型），真正的限制在 `ShouldPlay`。少了这条，"+1 能量"会被单独采纳，
        # 而 6 张上限消失 —— 实测就是这个原因把它放出去的。
        #
        # TODO（需要单独评审）：更彻底的口径是"任何 `commands`/`query` 类缺口都挡
        # 数值半边"。那会连带挡住 `sozu`（`ShouldProcurePotion`）/ `ectoplasm`，
        # 改变它们现有的强度，所以不在本批次里顺手改。
        blocked = bool(gaps & COMBAT_PHASE_HOOKS)
        max_energy = _relic_numeric(record, "max_energy", values, blocked)
        hand_draw = _relic_numeric(record, "hand_draw", values, blocked)
        for field, mod in (("max_energy", max_energy), ("hand_draw", hand_draw)):
            if record.get(field) and mod is None:
                unresolved_numeric.append(f"{rid}.{field}")

        parsed[rid] = RelicDef(
            rid=rid,
            name=(codex.get(rid) or {}).get("name") or record.get("class") or rid,
            rarity=str(record.get("rarity") or ""),
            # `pool` 是**结构元数据**（shared / 角色名），不是机制数值，
            # 所以按规则取社区库那一份（与 name 同理）；缺省 shared。
            pool=str((codex.get(rid) or {}).get("pool") or "shared").lower(),
            values=values,
            max_energy=max_energy,
            hand_draw=hand_draw,
            hooks=tuple(adopted),
            unmodeled_hooks=tuple(sorted(gaps)),
            unmodeled_hook_kinds=tuple(sorted(kinds.items())),
            combat_gaps=tuple(sorted(gaps & COMBAT_PHASE_HOOKS)),
        )
    if not parsed:
        return
    RELICS.clear()
    RELICS.update(parsed)
    summary["relics"] = len(parsed)
    summary["relics_energy"] = sum(1 for r in parsed.values() if r.max_energy)
    summary["relics_hand_draw"] = sum(1 for r in parsed.values() if r.hand_draw)
    summary["relics_with_hooks"] = sum(1 for r in parsed.values() if r.hooks)
    summary["relics_with_behavior"] = sum(1 for r in parsed.values() if r.has_behavior)
    summary["relics_fully_modeled"] = sum(1 for r in parsed.values() if r.fully_modeled)
    # ⭐ 真正的剩余工作量：只有 effectful（commands / query）缺口才算"缺行为"。
    summary["relics_with_effectful_gaps"] = sorted(
        r for r, d in parsed.items() if d.effectful_gaps)
    summary["relics_bookkeeping_only_gaps"] = sorted(
        r for r, d in parsed.items() if d.bookkeeping_only_gaps)
    summary["relics_unresolved_numeric"] = unresolved_numeric
    summary["relics_combat_gaps"] = sorted(
        r for r, d in parsed.items() if d.combat_gaps and d.has_behavior)
    timing_counts: dict[str, int] = {}
    for definition in parsed.values():
        for hook in definition.hooks:
            timing_counts[hook.timing] = timing_counts.get(hook.timing, 0) + 1
    summary["relic_hooks_by_timing"] = dict(sorted(timing_counts.items()))


def _describe_effects(effects: tuple[Effect, ...]) -> str:
    """把效果序列写成一句短说明（战斗日志用）。"""
    parts = []
    for effect in effects:
        if effect.op == "apply_power":
            scope = "全体敌人" if effect.target == "all_enemies" else ""
            parts.append(f"{scope}{effect.power} +{effect.amount}")
        elif effect.op == "block":
            parts.append(f"格挡 +{effect.amount}")
        elif effect.op == "heal":
            parts.append(f"回复 {effect.amount} 点生命")
        elif effect.op == "gain_stars":
            parts.append(f"星辰 +{effect.amount}")
        elif effect.op == "gain_energy":
            parts.append(f"能量 +{effect.amount}")
        elif effect.op == "channel":
            parts.append(f"引导 {effect.orb or '?'}")
        elif effect.op == "draw":
            parts.append(f"抽 {effect.amount} 张")
        else:
            parts.append(f"{effect.op} {effect.amount}")
    return "、".join(parts)


def _load_characters(directory, summary: dict) -> None:
    """加载角色定义。**起始牌组以它为准**——真机基础牌是按角色命名的。"""
    import json
    path = directory / "characters.json"
    if not path.exists():
        return
    entries = json.loads(path.read_text(encoding="utf-8"))
    parsed: dict[str, CharacterDef] = {}
    for entry in entries:
        raw_deck = entry.get("starting_deck") or entry.get("deck") or []
        deck = tuple(_normalize_card_id(x) for x in raw_deck)
        if not deck:
            continue
        cid = str(entry.get("id", "")).lower()
        parsed[cid] = CharacterDef(
            cid=cid,
            name=entry.get("name") or cid,
            hp=int(entry.get("starting_hp") or entry.get("max_hp") or 80),
            gold=int(entry.get("starting_gold") or 99),
            energy=int(entry.get("max_energy") or 3),
            deck=deck,
            relics=tuple(_normalize_card_id(r)
                         for r in (entry.get("starting_relics") or [])),
        )
    if not parsed:
        return
    CHARACTERS.clear()
    CHARACTERS.update(parsed)
    summary["characters"] = sorted(parsed)
    chosen = parsed.get(DEFAULT_CHARACTER) or next(iter(parsed.values()))
    STARTING_DECK.clear()
    STARTING_DECK.extend(chosen.deck)
    summary["starting_character"] = chosen.cid
    summary["missing_starting_cards"] = [c for c in chosen.deck if c not in CARD_DB]


#: 幕定义（``tools/extract_acts.py`` → ``acts_source.json``）。
#:
#: 真机一局走 **3 个幕索引**、共 **4 张幕地图**：索引 0 是密林 ``Overgrowth`` /
#: 暗港 ``Underdocks`` 二选一，索引 1 是蜂巢 ``Hive``，索引 2 是荣耀 ``Glory``。
#: 地图生成（``mapgen``）与按幕的内容池（``run``）都读这里。
#:
#: ⚠️ 值为 :class:`sts2_sim.acts.ActDef`。**不要**在这里放"近似值"：
#: 抽不到就让它缺，缺了由 :func:`acts.act_def` 退回带行号的兜底表。
ACTS: dict[str, object] = {}


def _load_acts(directory, summary: dict) -> None:
    """加载幕定义（``tools/extract_acts.py`` → ``acts_source.json``）。

    ⚠️ 必须排在**事件之后**：``events`` 池要回查 ``EVENT_DB``，早于事件加载会把
    "还没加载"误判成"事件不存在"。

    本函数只装载**能对上的**东西，并把对不上的记为缺口：
    遭遇池目前存的是**源码 id**（``fuzzy_wurm_crawler_weak`` 这种），而社区
    ``encounters.json`` 用的是另一套 ``eid``（且只覆盖 87 条）——两套 id 的桥
    还没搭（需要从 ``EncounterModel`` 子类抽成员），所以这里**只登记、不假装能跑**。
    """
    import json

    from .acts import ActDef

    path = directory / "acts_source.json"
    if not path.exists():
        return
    records = json.loads(path.read_text(encoding="utf-8"))
    loaded: dict[str, ActDef] = {}
    event_gaps: dict[str, list[str]] = {}
    unresolved_encounters: dict[str, int] = {}
    for record in records:
        counts = record.get("map_point_counts") or {}
        rest = counts.get("rest") or {}
        if rest.get("kind") == "next_int":
            rest_spec = ("next_int", int(rest["min"]), int(rest["max"]))
        else:
            rest_spec = ("gaussian_int", int(rest["mean"]), int(rest["stddev"]),
                         int(rest["min"]), int(rest["max"]))
        unknown = counts.get("unknown") or {}
        base = unknown.get("base") or {}
        unknown_spec = ("gaussian_int", int(base["mean"]), int(base["stddev"]),
                        int(base["min"]), int(base["max"]))
        unlocked = record.get("is_unlocked")
        definition = ActDef(
            act_id=str(record["id"]),
            csharp_class=str(record.get("csharp_class", record["id"])),
            index=int(record["index"]),
            is_default=bool(record["is_default"]),
            base_number_of_rooms=int(record["base_number_of_rooms"]),
            number_of_weak_encounters=int(record["number_of_weak_encounters"]),
            rest_spec=rest_spec,
            unknown_offset=int(unknown.get("offset", 0)),
            unknown_spec=unknown_spec,
            unlock_epoch=None if unlocked is True else str(unlocked),
            events=tuple(str(item["id"]) for item in record.get("events", ())),
            ancients=tuple(str(item["id"]) for item in record.get("ancients", ())),
            weak_encounters=tuple(str(x) for x in record.get("weak_encounters", ())),
            regular_encounters=tuple(
                str(x) for x in record.get("regular_encounters", ())),
            elite_encounters=tuple(
                str(x) for x in record.get("elite_encounters", ())),
            boss_encounters=tuple(str(x) for x in record.get("boss_encounters", ())),
            boss_discovery_order=tuple(
                str(item["id"]) for item in record.get("boss_discovery_order", ())),
        )
        loaded[definition.act_id] = definition

        missing_events = [eid for eid in definition.events if eid not in EVENT_DB]
        if missing_events:
            event_gaps[definition.act_id] = missing_events
        unresolved_encounters[definition.act_id] = sum(
            len(pool) for pool in (definition.weak_encounters,
                                   definition.regular_encounters,
                                   definition.elite_encounters,
                                   definition.boss_encounters))

    ACTS.clear()
    ACTS.update(loaded)
    summary["acts"] = len(loaded)
    if event_gaps:
        summary["act_event_gaps"] = event_gaps
    if unresolved_encounters:
        # 不是"缺口"，是**尚未接通的那一层**：池子里存的是源码 id，社区遭遇表
        # 用的是另一套 eid。写清楚数量，免得下游以为这些池已经能用。
        summary["act_encounter_ids_pending_bridge"] = unresolved_encounters


def load_content_dir(path) -> dict:
    """从 JSON 内容表加载卡牌与怪物（``docs/06`` 的 verify/freeze 产出）。

    ⚠️ **调用顺序有硬约束**：必须先加载内容、再构建词表与模型。
    卡牌 embedding 是按词表下标索引的，内容换了而下标没换，会让模型"张冠李戴"
    且**不会报错**——最危险的一类 bug。配合 ``featurize.configure_content()``
    使用，它会顺带重建词表。
    """
    import json
    from pathlib import Path

    global CONTENT_SOURCE
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} 不存在")
    loaded_summary: dict[str, object] = {"source": str(directory)}

    cards_path = directory / "cards.json"
    if cards_path.exists():
        entries = json.loads(cards_path.read_text(encoding="utf-8"))
        loaded: dict[str, CardDef] = {}
        for entry in entries:
            effects = tuple(_effect_from_json(item) for item in entry.get("effects", []))
            upgrade = tuple(_effect_from_json(item) for item in entry.get("upgrade", []))
            card = CardDef(
                cid=entry["cid"],
                name=entry.get("name", entry["cid"]),
                cost=int(entry.get("cost", 0)),
                is_x_cost=bool(entry.get("is_x_cost")),
                card_type=entry.get("card_type", "skill"),
                rarity=entry.get("rarity", "common"),
                target=entry.get("target", "enemy"),
                effects=effects,
                upgrade=upgrade,
                exhaust=bool(entry.get("exhaust", False)),
                verified=bool(entry.get("verified", False)),
            )
            loaded[card.cid] = card
        if loaded:
            CARD_DB.clear()
            CARD_DB.update(loaded)
            loaded_summary["cards"] = len(loaded)

    # 源码抽取的效果**覆盖**文本解析的结果（docs/09 铁律：源码是机制的唯一真相）。
    # 必须排在 `_load_characters` 之前：起始牌组要按 cid 校验，而覆盖会改变
    # 卡牌是否可用。
    _load_cards_source(directory, loaded_summary)

    # ⚠️ 角色必须在卡牌之后加载：起始牌组要按 cid 校验是否存在
    _load_characters(directory, loaded_summary)
    if "missing_starting_cards" not in loaded_summary:
        loaded_summary["missing_starting_cards"] = [
            cid for cid in STARTING_DECK if cid not in CARD_DB]

    monsters_path = directory / "monsters.json"
    if monsters_path.exists():
        entries = json.loads(monsters_path.read_text(encoding="utf-8"))
        enemies: dict[str, EnemyDef] = {}
        for entry in entries:
            moves = tuple(
                MoveDef(
                    mid=move["mid"],
                    name=move.get("name", move["mid"]),
                    intent=move.get("intent", "unknown"),
                    value=int(move.get("value", 0)),
                    times=int(move.get("times", 1)),
                    effects=tuple(_effect_from_json(e) for e in move.get("effects", [])),
                    weight=float(move.get("weight", 1.0)),
                    kills_self=bool(move.get("kills_self", False)),
                )
                for move in entry.get("moves", [])
            )
            hp = entry.get("hp")
            if hp and hp[0] is not None:
                hp = (int(hp[0]), int(hp[1]) if hp[1] is not None else int(hp[0]))
            else:
                hp = None          # codex 缺 HP → 等源码补，末尾统一校验
            enemy = EnemyDef(
                eid=entry["eid"], name=entry.get("name", entry["eid"]),
                hp=hp, ai=entry.get("pattern_type", ""),
                moves=moves, verified=bool(entry.get("verified", False)),
                cycle=tuple(entry.get("cycle", ())),
                pattern_type=entry.get("pattern_type", ""),
                unmodeled_innate_powers=tuple(entry.get("unmodeled_innate_powers", ())),
                innate_powers=tuple(
                    (str(p["power"]).lower(), int(p.get("amount") or 0))
                    for p in entry.get("innate_powers", ()) if p.get("power")),
            )
            enemies[enemy.eid] = enemy
        if enemies:
            ENEMY_DB.clear()
            ENEMY_DB.update(enemies)
            loaded_summary["monsters"] = len(enemies)

    loaded_summary["encounters"] = _load_encounters(directory)
    # ⚠️ 出招状态机必须在**怪物之后**加载：它要按 eid 挂到 ENEMY_DB 上。
    # 放在怪物之前会一条都匹配不上（实测 `monster_ai: 0`，而且不报错）。
    _load_monster_ai(directory, loaded_summary)
    # ⚠️ 招式效果的覆盖必须排在**状态机之后**：它要按 `ai_record.states`
    # 把招式表的 mid 对应到源码里的处理函数名（`IncantationMove`）。
    # 放在前面会一条都匹配不上（实测 adopted=0，而且不报错）。
    _load_monster_moves(directory, loaded_summary)
    # ⭐ **闭包闸门**：招式表的洞必须在**加载期**发现。
    # `_choose_intent` 在找不到招式时抛错（这是对的 —— 静默退化成"什么都不做"
    # 更糟），但如果内容表里本来就有这个洞，错误会等到"这只怪真的出场"才炸，
    # 而那时可能已经训练了几十万步。所以这里收尾校验，并把**含有缺口怪的遭遇**
    # 排除出遭遇表（受限范围是允许的，静默带洞执行不是）。
    loaded_summary["monster_move_gaps"] = _enemy_move_gaps()
    if loaded_summary["monster_move_gaps"]:
        loaded_summary["encounters_dropped_for_move_gaps"] = _drop_encounters_with_gaps(
            set(loaded_summary["monster_move_gaps"]))
    _require_enemy_hp()
    _load_powers(directory, loaded_summary)
    _load_potions(directory, loaded_summary)
    _load_card_pools(directory, loaded_summary)
    _load_relics(directory, loaded_summary)
    # 事件必须排在最后：它的选项效果引用卡牌 / 遗物 / 药水的 id，
    # 前面没加载完就会把"卡 id 不存在"误判成缺口。
    _load_events_source(directory, loaded_summary)
    # ⭐ 幕定义排在事件之后：`events` 池要回查 EVENT_DB。
    _load_acts(directory, loaded_summary)
    # ⭐ 收尾校验：**角色定义的每一项都必须能对上加载后的内容表**。
    # 审计 F02 的另一半：`RunEnv(character="silent")` 只影响奖励池，
    # 起始牌组与初始遗物都不是该角色的。牌组那半已由 `_load_characters` 修好，
    # 遗物那半要靠这里 —— 名字对不上时**必须报出来**，否则初始遗物静默失效。
    loaded_summary["character_gaps"] = _character_gaps()
    CONTENT_SOURCE = str(directory)
    return loaded_summary


def _character_gaps() -> dict[str, list[str]]:
    """每个角色的起始牌组 / 初始遗物里**加载后找不到的 id**。

    空 = 全部对得上。非空必须报出来：一个静默失效的初始遗物与"没有遗物"
    在战斗里长得一模一样，只有数值会偏。
    """
    gaps: dict[str, list[str]] = {}
    for cid, definition in CHARACTERS.items():
        missing: list[str] = []
        missing.extend(f"card:{c}" for c in definition.deck if c not in CARD_DB)
        missing.extend(f"relic:{r}" for r in definition.relics if r not in RELICS)
        if missing:
            gaps[cid] = missing
    return gaps


def _load_encounters(directory) -> int:
    """加载遭遇表；优先真实的 encounters.json，退回反推的 provisional 版本。

    反推版本可能**缺成员**（只有被抓到的怪物才会出现在名单里），所以只保留
    "所有成员都在怪物表里"的遭遇——不完整的一组怪比没有更危险。
    """
    import json
    for name in ("encounters.json", "encounters_provisional.json"):
        path = directory / name
        if not path.exists():
            continue
        entries = json.loads(path.read_text(encoding="utf-8"))
        grouped: dict[str, list[tuple[str, ...]]] = {"monster": [], "elite": [], "boss": []}
        skipped_unknown = 0
        skipped_event = 0
        skipped_slots = 0
        for entry in entries:
            room = str(entry.get("room_type", "")).lower()
            key = {"monster": "monster", "elite": "elite", "boss": "boss"}.get(room)
            if key is None:
                continue
            # ⚠️ **事件遭遇必须排除**：它们的 act 为空，怪也是不可战斗的占位实体
            # （例如 9999 血、只会用 NOTHING 的假人）。混进地图池会让战斗永远打不完。
            if not entry.get("act"):
                skipped_event += 1
                continue
            members = tuple(entry.get("monsters", ()))
            if not members or any(m not in ENEMY_DB for m in members):
                skipped_unknown += 1
                continue
            # ⚠️ **超过槽位上限的遭遇必须排除**：真机最多 5 只怪（``EncounterModel``
            # 的 Slots 到 "fifth"）。社区数据把"可选成员"写成了"实际生成结果"，
            # 照单全收会生成一场真机不存在的战斗，而且第 6 只怪连槽位名都没有。
            if len(members) > MAX_ENEMY_SLOTS:
                skipped_slots += 1
                continue
            grouped[key].append(members)
        total = sum(len(v) for v in grouped.values())
        if total:
            ENCOUNTERS.clear()
            ENCOUNTERS.update({k: tuple(v) for k, v in grouped.items()})
        _LAST_ENCOUNTER_LOAD["skipped_event"] = skipped_event
        _LAST_ENCOUNTER_LOAD["skipped_missing_monster"] = skipped_unknown
        _LAST_ENCOUNTER_LOAD["skipped_exceeds_slots"] = skipped_slots
        return total
    return 0


#: 上一次遭遇表加载的统计（健康检查会读它）
_LAST_ENCOUNTER_LOAD: dict[str, int] = {
    "skipped_event": 0, "skipped_missing_monster": 0, "skipped_exceeds_slots": 0}


def _effect_from_json(item: dict) -> Effect:
    """只取 DSL 认识的键，多余的忽略（宽容解析，严格检查交给 lint）。"""
    allowed = {"op", "amount", "power", "target", "card"}
    return Effect(**{k: v for k, v in item.items() if k in allowed})


def _autoload_from_env() -> None:
    import os
    path = os.environ.get(CONTENT_DIR_ENV)
    if path:
        try:
            load_content_dir(path)
        except (FileNotFoundError, ValueError) as exc:  # pragma: no cover
            print(f"[content] 环境变量 {CONTENT_DIR_ENV}={path} 加载失败：{exc}")
