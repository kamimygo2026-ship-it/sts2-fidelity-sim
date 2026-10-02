"""Run 层的效果结算（``docs/09`` §19）。

战斗内的效果由 ``core._apply_effects`` 跑；**Run 层**（遗物拾取、战后、事件、
营火、商店）用的是一批**作用于牌组与角色属性**的命令，两边的 DSL 相同、
作用对象不同：

====================== ====================================================
算子                    真机命令
====================== ====================================================
``gain_max_hp``        ``CreatureCmd.GainMaxHp``
``lose_max_hp``        ``CreatureCmd.LoseMaxHp``
``heal``               ``CreatureCmd.Heal``
``lose_hp``            ``CreatureCmd.LoseHp``（可直接扣到 0）
``add_card``           ``CardPileCmd.AddCurseToDeck<T>`` / ``…ToDeck``
``remove_card_from_deck`` ``CardPileCmd.RemoveFromDeck``
``upgrade_card``       ``CardCmd.Upgrade``
``transform_card``     ``CardCmd.Transform`` / ``TransformTo<T>``
``enchant_card``       ``CardCmd.Enchant<T>``
====================== ====================================================

⭐ **最大生命的两个方向语义不同，别想当然**（源码 ``CreatureCmd.cs:841-882``）::

    // GainMaxHp：加完上限**再回等量的血**
    decimal num = await SetMaxHp(creature, MaxHp + amount);
    ...
    await Heal(creature, num);            // ← 这一行在方法体**最后**

    // LoseMaxHp：只砍血槽；当前血量**超出新上限**的那部分才掉，且走 Damage
    decimal newMaxHp = MaxHp - amount;
    if (newMaxHp < CurrentHp)
        await Damage(context, creature, CurrentHp - newMaxHp, Unblockable|Unpowered, …);
    await SetMaxHp(creature, Math.Max(1m, newMaxHp));

⚠️ 我一开始把 ``gain_max_hp`` 实现成"只加上限、不回血" —— 因为读源码时
**只看了方法的前十行**（``SetMaxHp`` 那一句），而 ``Heal`` 在方法**末尾**。
症状是"所有加最大生命的遗物都少回一次血"。教训写进 ``docs/09`` §5.4：
**读完整方法体，不要用截断的 grep 窗口下结论**。

⚠️ **需要玩家选牌的算子不能在这里静默近似**：`Whetstone`（升级一张）、
`Astrolabe`（转化）、`PandorasBox`（移除）在真机里都是"**让玩家从牌组里挑**"，
抽取器已经把它们识别成选牌（``CardSelectCmd.FromDeck*``）并排除出这一层。
在这里偷偷升级一张随机牌，等于把"玩家的选择"换成了"随机" —— 机制被换掉了，
而且日志看起来完全正常。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .content import CARD_DB, Effect
from .rng import STREAMS

#: Run 层**已实现**的算子。``lint`` 与覆盖率报告用它判断哪些遗物/事件可用。
RUN_OPS: frozenset[str] = frozenset({
    "gain_max_hp", "lose_max_hp", "heal", "lose_hp", "add_card",
    "remove_card_from_deck",
    "upgrade_card", "transform_card", "enchant_card", "gain_gold", "lose_gold",
    "lose_all_gold", "offer_rewards",
    # ⭐ "从牌组随机取 N 张不重复再做事"（`Whetstone` / `WarPaint` / `WarHammer`
    # / `SandCastle` / `FragrantMushroom`）—— 见 `docs/12` §2.19。
    "random_deck_cards",
    # ⭐ "把整副牌组复制一份"（`Reflections.Shatter`）—— 见 `docs/12` §2.21。
    "clone_deck",
    # ⭐ "从**卡池**随机取 N 张不重复加入牌组"（`DistinguishedCape`）—— §2.22。
    "add_random_cards",
})

#: 需要玩家选牌的算子 —— 在 Run 层**不能自动结算**（要由环境给出选择动作）。
CHOICE_OPS: frozenset[str] = frozenset({"select_card"})

#: Run 层**已实现**的选牌用途。
#:
#: 真机的用途写在方法名里（``CardSelectCmd.FromDeckForRemoval`` /
#: ``FromDeckForUpgrade`` / ``FromDeckForTransformation`` / ``FromDeckForEnchantment``）。
#: 引擎实现了"从牌组选一张 → **移除** / **升级** / **转化**"三种；
#: 其余（附魔整套系统、复制、从生成候选里挑）**如实拒绝**，让事件标残缺 ——
#: 随机替你选一张等于把机制换掉了，而"看起来实现了"比"明确没实现"危险得多。
RUN_SELECTION_PURPOSES: frozenset[str] = frozenset({"removal", "upgrade",
                                                    "transform", "downgrade",
                                                    "duplicate"})


@dataclass
class RunSelection:
    """一次**等待玩家选择**的 Run 层选牌。

    字段大部分是人类可见信息（牌组、用途、还要选几张），可以进观测。
    ``candidates`` 是**内部下标**；对外用 ``observe.deck_slot_order`` 的槽位编号。

    ⚠️ ``rng`` 是**隐藏状态**（事件自己的 RNG），**绝不可进观测** ——
    它只用来在选完之后掷"转化成哪张牌"。
    """

    purpose: str
    count: int
    source: str
    candidates: tuple[int, ...]
    #: 选完之后要继续跑的效果（本次没跑到的其余效果）
    rest: tuple[Effect, ...] = ()
    #: 已经选中的内部下标
    chosen: tuple[int, ...] = ()
    #: ``transform`` 用途要用的 RNG（事件 RNG；见 :func:`transform_replacement`）
    rng: object | None = None


def _deck_candidates(player, purpose: str, source: str) -> tuple[int, ...]:
    """这次选牌的候选（**内部下标**）。

    资格判据（都以源码为准）：

    * ``removal`` / ``transform`` / ``enchant`` —— 要过 ``CardModel.IsRemovable``
      （``CardModel.cs:738``：永恒牌不可移除、牌组里也不可转化）。
      不筛的话 `AscendersBane` 这类"设计上就动不了"的牌会出现在候选里；
    * ``transform`` 再多一条 ``c.Type != CardType.Quest``
      （``CardSelectCmd.cs:591``：``Cards.Where(c => c.Type != Quest && c.IsTransformable)``）
      —— 任务牌不能被转化，少了这条就与真机的候选列表不一样；
    * ``upgrade`` —— 已经升级过的牌不能再升（``CardCmd.Upgrade`` 对升级牌是空操作，
      真机的候选列表也会把它们排掉）。
    """
    if source not in ("deck", "deck_pile"):
        return ()
    candidates = []
    for index, card in enumerate(player.deck):
        if not _removable(card):
            continue
        if purpose == "transform" and _card_type(card) == "quest":
            continue
        if purpose == "duplicate" and _card_type(card) == "quest":
            # `DollysMirror.Filter`：``c.Type != CardType.Quest`` —— 任务牌不能被复制。
            continue
        if purpose == "upgrade" and card.upgraded:
            continue
        if purpose == "downgrade" and not card.upgraded:
            # 真机 `Reflections.TouchAMirror` 的候选就是
            # `Deck.Cards.Where(c => c.IsUpgraded)`（**已升级**的牌）。
            continue
        candidates.append(index)
    return tuple(candidates)


def _stream_handle(state, name: str, provided):
    """按流名取随机流句柄，返回 ``(句柄, 拿得到吗)``。

    * ``event`` —— **调用方传进来的事件流**（`events.apply_option` 传 `run.rng`）；
    * 其余命名流 —— ``state.hidden.rng``（`RngSet`，用流名分派）。

    拿不到就如实返回 ``False``（调用方跳过），**绝不**按别的流近似 ——
    换一条流，分布一样但序列与真机分叉。
    """
    if name == "event":
        return provided, provided is not None
    if name in STREAMS:
        return state.hidden.rng, True
    return None, False


def _random_pick(candidates: tuple[int, ...], count: int, stream: str,
                 rng, pick: str) -> tuple[int, ...]:
    """从候选（**内部下标**）里随机取 ``count`` 个**不重复**的。

    两种取法**消费的随机数不同**，所以要分开实现（合并成一种会让同种子下的
    序列与真机分叉，而分布看起来一样）：

    * ``shuffle`` —— 真机 ``StableShuffle(rng)`` 后 ``Take(N)``；
    * ``item``    —— 真机循环 ``rng.NextItem(pool)`` + ``pool.Remove(它)``。

    ``stream`` 是**源码里写的那条流**（`Whetstone` 一众用 ``niche``；
    事件里的 ``base.Rng`` 用特殊值 ``event``）。

    ⚠️ 两种 RNG 对象的接口**不同**，必须分开调用（用 ``hasattr`` 判）：

    * :class:`sts2_sim.rng.RngSet` —— ``next_index(stream, n)`` / ``shuffle(list, stream)``；
    * :class:`sts2_sim.events.EventRng` —— ``next_index(n)`` / ``stable_shuffle(list)``。
    """
    if count <= 0 or not candidates:
        return ()
    event_rng = hasattr(rng, "stable_shuffle")
    order = list(candidates)
    if pick == "shuffle":
        if event_rng:
            rng.stable_shuffle(order)
        else:
            rng.shuffle(order, stream)
        return tuple(order[:count])
    picked: list[int] = []
    pool = list(candidates)
    for _ in range(count):
        if not pool:
            break
        index = (rng.next_index(len(pool)) if event_rng
                 else rng.next_index(stream, len(pool)))
        picked.append(pool.pop(index))
    return tuple(picked)


def _card_type(card) -> str:
    definition = CARD_DB.get(card.cid)
    return definition.card_type if definition is not None else ""


def _removable(card) -> bool:
    """``CardModel.IsRemovable``（``CardModel.cs:738``）：永恒牌不能被移除。

    同一处还定义 ``IsTransformable``：``!IsRemovable`` 的牌**在牌组里**不可转化。
    所以 Run 层的移除/转化候选都要过这一关。
    """
    from . import keywords as keyword_rules
    return not keyword_rules.is_eternal(CARD_DB[card.cid])


def _random_card(state, predicate) -> str | None:
    """从**卡池**里随机抽一张满足条件的卡（用于转化 / 随机升级）。"""
    pool = sorted(cid for cid, card in CARD_DB.items() if predicate(card))
    if not pool:
        return None
    return pool[state.hidden.rng.next_index("transformations", len(pool))]


def _card_pool_of(cid: str) -> str | None:
    """这张牌属于哪个卡池（``CardModel.Pool`` = 包含它的那个池）。

    实测 ``card_pools.json`` 里**没有一张牌同时属于两个池**，所以这是无歧义的；
    真机 ``RelicModel.Pool`` 也是"第一个包含它的池"（``RelicModel.cs:174``）。
    """
    from .content import CARD_POOLS

    for name, ids in CARD_POOLS.items():
        if cid in ids:
            return name
    return None


def transform_options(cid: str) -> tuple[str, ...]:
    """``CardFactory.GetDefaultTransformationOptions``（``CardFactory.cs:170-212``）。

    真机算法逐条复刻::

        // 池子：任务牌 / 事件牌 / 古牌 / Token 牌 → 无色池；否则用这张牌自己的池
        pool = (original.Type != Quest && Rarity not in {Event, Ancient, Token})
               ? original.Pool : ModelDb.CardPool<ColorlessCardPool>();
        cards = pool.GetUnlockedCards(...);
        // 过滤（GetFilteredTransformationOptions）：
        //   原牌稀有度**不是** Status / Curse 时，候选只留 Common / Uncommon / Rare
        //   （`(uint)(rarity - 2) <= 2u` ↔ enum 值 2/3/4）
        //   isInCombat → 再加 `CanBeGeneratedInCombat`（事件都不是战斗内，故不加）
        //   去掉与原牌同 id 的
        //   FilterForPlayerCount（单人时是 no-op）

    ⚠️ ``GetUnlockedCards`` 会过 ``FilterThroughEpochs(unlockState, …)`` ——
    本引擎按"内容全解锁"处理（与 ``draw_card_reward`` / 遗物抓包同一取舍），
    这一点是**已知的近似**，不是"已经一致"。
    """
    from .content import CARD_POOLS

    card = CARD_DB.get(cid)
    if card is None:
        return ()
    pool_name = _card_pool_of(cid) or ""
    if card.card_type == "quest" or card.rarity in ("event", "ancient", "token"):
        pool_name = "ColorlessCardPool"
    options = list(CARD_POOLS.get(pool_name, ()))
    if card.rarity not in ("status", "curse"):
        options = [other for other in options
                   if CARD_DB.get(other) is not None
                   and CARD_DB[other].rarity in ("common", "uncommon", "rare")]
    return tuple(other for other in options if other != cid)


def transform_replacement(cid: str, rng) -> str | None:
    """``CardFactory.CreateRandomCardForTransform``：``rng.NextItem(候选)``。

    ``rng`` **必须是**调用方给的那一条 —— 事件里写的是 ``base.Rng``
    （``AromaOfChaos.cs:31`` / ``MorphicGrove.cs:50`` / ``WhisperingHollow.cs:63``
    / ``Symbiote.cs:79``），也就是**每个事件自己的 RNG**；
    遗物则各写各的（``Astrolabe.cs:24`` 用 ``RunState.Rng.Niche``、
    ``Claws.cs:29`` 用 ``PlayerRng.Transformations``）。传错流就等于换了个随机源。
    """
    options = transform_options(cid)
    if not options:
        return None
    return options[rng.next_index(len(options))]


def offer_rewards(state, effect: Effect, events: list[str]) -> bool:
    """``RewardsCmd.OfferCustom``：把一组奖励放进**待选队列**，交给环境去选。

    真机（``RewardsCmd.OfferCustom(owner, list)``）把列表里的 ``Reward`` **逐条**
    摆到奖励界面：每一条各自"拿 / 跳过"（``RelicReward.OnSelect`` /
    ``PotionReward.OnSelect``）。所以这里**一条奖励 = 一个 ``RewardGroup``**：
    ``PotionReward`` / ``RelicReward`` 的组里只有一个候选，
    ``CardReward`` 的组里是它的 N 张候选（挑一张）。

    ⚠️ 早先的实现把 ``amount`` 当成"一个组里放几个候选"，于是
    "给 5 瓶药水"变成了"五选一拿一瓶" —— 机制被换掉，而日志看起来完全正常。

    ⚠️ 这里**不替玩家做决定** —— 真机是弹一个奖励界面让玩家挑。
    引擎把它表达成 `RunState.pending_rewards` + `take_reward` 动作，
    与"卡牌三选一"走同一套交互。
    """
    from .run import RewardGroup

    spec = dict(effect.card_filter)
    kind = spec.get("kind")
    if not kind:
        events.append("⚠️ 奖励组缺少类型（抽取器没解析出 kind），跳过")
        return False
    count = max(1, int(effect.amount or 1))
    note = _reward_note(spec)
    added = 0
    for _ in range(count):
        options = _reward_options(state, spec)
        if not options:
            events.append(f"⚠️ 找不到可用的{kind}奖励候选，跳过（第 {added + 1} 份）")
            break
        state.pending_rewards.append(
            RewardGroup(kind=kind, options=options, note=note))
        added += 1
    if not added:
        return False
    events.append(f"提供奖励：{note} × {added}")
    return True


def _reward_note(spec: dict) -> str:
    """奖励的一句话说明（写进日志与 ``RewardGroup.note``）。"""
    kind = str(spec.get("kind"))
    if spec.get("potion"):
        return f"药水 {spec['potion']}"
    if kind == "relic":
        return f"遗物（{spec['rarity']}）" if spec.get("rarity") else "随机遗物"
    if kind == "potion":
        if spec.get("mode") == "pool":
            rarity = f"{spec['rarity']} " if spec.get("rarity") else ""
            return f"{rarity}随机药水（角色池 ∪ 共享池）"
        return "随机药水（稀有度掷点）"
    if kind == "card":
        return f"卡牌奖励 {spec.get('choices')} 选 1（{spec.get('pool')}）"
    return kind


def _potion_pool(state, rarity: str = "") -> list[str]:
    """随机药水的候选池（``PotionFactory.GetPotionOptions``，``PotionFactory.cs:88-91``）。

    ``player.Character.PotionPool ∪ ModelDb.PotionPool<SharedPotionPool>()`` ——
    也就是**角色池 ∪ 共享池**（``event`` / 别的角色的药水都不在内）。
    """
    from .content import POTIONS

    character = str(getattr(state.player, "character", "ironclad"))
    return sorted(pid for pid, definition in POTIONS.items()
                  if getattr(definition, "pool", "shared") in ("shared", character)
                  and (not rarity or definition.rarity == rarity))


def _roll_potion_rarity(rng) -> str:
    """``PotionFactory.CreateRandomPotions`` 的稀有度掷点（``PotionFactory.cs:74-77``）::

        float num = rng.NextFloat();
        num <= 0.1  → Rare
        num <= 0.35 → Uncommon
        否则         → Common
    """
    roll = rng.next_float("rewards")
    if roll <= 0.1:
        return "rare"
    if roll <= 0.35:
        return "uncommon"
    return "common"


def _reward_options(state, spec: dict) -> tuple[str, ...]:
    """按**奖励规格**生成候选（``kind`` / ``rarity`` / ``potion`` / ``relic`` …）。

    规格来自源码（``tools/extract_events.py`` 的 ``extract_event_rewards``）：

    * ``relic`` —— ``RelicReward.Populate``（``RelicReward.cs:75-94``）：
      指定遗物就直接用，否则 ``RelicFactory.PullNextRelicFromFront``
      （掷稀有度走 ``rewards`` 流 → 从抓包前端取走）；桶空 → ``Circlet`` 兜底。
    * ``potion``：
      - 有 ``potion`` → **指定的那一瓶**（真机 ``ModelDb.Potion<X>()``）；
      - ``mode="pool"`` → ``PlayerRng.Rewards.NextItem(items)``，
        ``items`` 是角色池 ∪ 共享池（可再按稀有度过滤）；
      - ``mode="factory"`` → ``PotionFactory.CreateRandomPotionOutOfCombat``。
    * ``card`` —— 角色池（或源码指定的卡池）按奖励稀有度概率抽 ``choices`` 张。
    """
    from .content import POTIONS, RELICS
    from .run import draw_card_reward

    rng = state.hidden.rng
    kind = spec.get("kind")
    if kind == "relic":
        specific = str(spec.get("relic") or "")
        if specific:
            return (specific,) if specific in RELICS else ()
        from .relicbag import FALLBACK_RELIC, roll_rarity

        bag = getattr(state, "relic_bag", None)
        if bag is None:
            # 没有抓包就不能"随机挑一个同稀有度的遗物"：那会让"取走不再出现"
            # 这条语义落空，同一局会重复拿到同一个遗物。
            return ()
        rarity = str(spec.get("rarity") or "") or roll_rarity(rng, "rewards")
        pulled = bag.pull_from_front(rarity)
        return (FALLBACK_RELIC,) if pulled is None else (pulled,)
    if kind == "potion":
        specific = str(spec.get("potion") or "")
        if specific:
            return (specific,) if specific in POTIONS else ()
        pool = _potion_pool(state, str(spec.get("rarity") or ""))
        if not pool:
            return ()
        if spec.get("mode") == "factory":
            rarity = _roll_potion_rarity(rng)
            candidates = [pid for pid in pool if POTIONS[pid].rarity == rarity] or pool
            return (candidates[rng.next_index("rewards", len(candidates))],)
        return (pool[rng.next_index("rewards", len(pool))],)
    if kind == "card":
        character = str(getattr(state.player, "character", "ironclad"))
        choices = max(1, int(spec.get("choices") or 1))
        pool_name = str(spec.get("pool") or "")
        if pool_name == "__character__":
            pool_name = ""
        return draw_card_reward(rng, "monster", choices, character,
                                pool_name or None)
    return ()


def _pull_random_relic(state, effect: Effect, events: list[str]) -> str:
    """``RelicFactory.PullNextRelicFromFront``：掷稀有度 → 抓包前端取。

    随机源：``rewards`` 流（真机 ``RollRarity(player)`` → ``player.PlayerRng.Rewards``，
    ``RelicFactory.cs:80-83``）。桶空 → ``FallbackRelic``（``Circlet``）。
    """
    from .relicbag import FALLBACK_RELIC, roll_rarity

    bag = getattr(state, "relic_bag", None)
    if bag is None:
        events.append("⚠️ 随机遗物需要遗物抓包，但本局没有建包，跳过")
        return ""
    rarity = dict(effect.card_filter).get("rarity") or ""
    if not rarity:
        rarity = roll_rarity(state.hidden.rng, "rewards")
    pulled = bag.pull_from_front(str(rarity))
    if pulled is None:
        # 桶空：真机用 `FallbackRelic` 兜底（`RelicFactory.cs:47` 的 `?? FallbackRelic`）
        events.append(f"⚠️ {rarity} 遗物桶空了 → 兜底 {FALLBACK_RELIC}")
        return FALLBACK_RELIC
    return pulled


def apply_run_effects(state, effects: Iterable[Effect],
                      events: list[str], rolled: dict | None = None,
                      obtain_relic=None, rng=None) -> int:
    """把一批效果作用到 **Run 层状态**（``state.player``：血 / 金币 / 牌组）。

    返回成功结算的效果条数。遇到不认识的算子会**记进 events 并跳过**，
    不静默当作"做完了"。

    ``rolled`` 是**事件掷出来的量**（``CalculateVars``）。事件选项里的
    ``Gold`` 这类变量只有在事件开始时才有值，所以事件层会把掷点传进来。
    ``obtain_relic`` 是拾取遗物的回调（要触发 ``AfterObtained``；
    见 :func:`sts2_sim.events.apply_option` 的说明）。

    ⭐ **遇到需要玩家选牌的算子就挂起**：把这次选牌写进
    ``state.pending_selection``，剩下的效果一起存进去，**立刻返回**。
    由环境给出选择动作后调 :func:`resolve_run_selection` 继续 ——
    绝不"替玩家随机选一张"（那是换了个机制，而且日志看不出差别）。
    """
    player = state.player
    rolled = rolled or {}
    done = 0
    for index, effect in enumerate(effects):
        op = effect.op
        if op == "select_card":
            # 需要玩家选牌 → 挂起，把**剩下的**效果一起存下来
            if effect.purpose not in RUN_SELECTION_PURPOSES:
                events.append(f"⚠️ 选牌用途 {effect.purpose!r} 引擎没实现，跳过")
                continue
            # `transform` 要"用哪条随机流掷替换牌"由**调用方**决定
            # （事件是 `base.Rng`、遗物各写各的），所以 rng 由调用方显式传进来
            # （``events.apply_option`` 传事件 RNG）。没传就没有可信的流 ——
            # 那时**如实拒绝**，绝不随便挑一条
            # （挑错会让同一种子下的结果与真机分叉）。
            selected_rng = None
            if effect.purpose == "transform":
                selected_rng = rng
                if selected_rng is None:
                    events.append("⚠️ 转化需要调用方指定的随机流"
                                  "（事件 RNG / Niche / Transformations），"
                                  "此处没有，跳过")
                    continue
            candidates = _deck_candidates(player, effect.purpose,
                                          effect.select_from)
            if not candidates:
                events.append("⚠️ 没有可选的牌（候选为空），跳过")
                continue
            state.pending_selection = RunSelection(
                purpose=effect.purpose,
                count=max(1, int(effect.amount or 1)),
                source=effect.select_from,
                candidates=candidates,
                rest=tuple(effects)[index + 1:],
                rng=selected_rng,
            )
            events.append(f"等待玩家选牌：{effect.purpose}（{len(candidates)} 个候选）")
            return done
        # 事件掷出来的量：`Effect.amount` 已经是解析后的值（在
        # `events.resolve_option_effects` 里用 `rolled` 解过），所以这里只需要
        # 处理"本来就是变量、但解析时没值"的情况。
        amount = effect.amount
        if amount is None and getattr(effect, "amount_var", None):
            amount = rolled.get(effect.amount_var)
        amount = int(amount or 0)
        if op == "obtain_relic":
            # `RelicCmd.Obtain(relic, owner)` → 入列 + 触发 `AfterObtained`。
            # ⚠️ 必须走回调（`RunEnv._obtain_relic`）：直接 append 会让
            # **拾取类遗物的效果全部不生效**，而且不报错。
            relic_id = getattr(effect, "relic", "") or effect.card or ""
            if not relic_id:
                # 空 = **随机遗物**：`RelicFactory.PullNextRelicFromFront(...)`
                # → 掷稀有度（`rewards` 流）→ 抓包前端取（取走不再出现）
                relic_id = _pull_random_relic(state, effect, events)
                if not relic_id:
                    continue
            if obtain_relic is not None:
                obtain_relic(relic_id)
            else:
                player.relics.append(relic_id)
                events.append(f"获得遗物 {relic_id}（⚠️ 未触发拾取效果）")
            events.append(f"获得遗物 {relic_id}")
        elif op == "downgrade_card":
            # `CardCmd.Downgrade(card)`：降回基础形态（附魔与苦痛保留）。
            # 引擎的升级是布尔标记，所以"降级"= 去掉升级标记。
            targets = [c for c in player.deck if c.upgraded
                       and effect.card in (None, "", c.cid)]
            if len(targets) == 1:
                targets[0].upgraded = False
                events.append(f"降级 {CARD_DB[targets[0].cid].name}")
            elif not targets:
                events.append("⚠️ 没有可降级的牌（都已是最基础形态）")
            else:
                events.append(f"⚠️ 降级需要玩家选择（{len(targets)} 张候选），跳过")
                continue
        elif op == "gain_max_hp":
            # `GainMaxHp` = 加上限 **+ 回等量的血**（源码最后一行 `await Heal(num)`）。
            player.max_hp += amount
            healed = min(amount, player.max_hp - player.hp)
            player.hp += healed
            events.append(f"最大生命 +{amount}（现 {player.max_hp}），回复 {healed} 点生命")
        elif op == "lose_max_hp":
            # `LoseMaxHp`：**只砍血槽**。当前血量超出新上限的那部分才掉，
            # 而且是走 `Damage`（Unblockable | Unpowered）而不是直接减。
            # ⚠️ `Math.Max(1m, newMaxHp)`：上限最低到 **1**，不会砍成 0 或负数。
            new_max = max(1, player.max_hp - amount)
            if player.hp > new_max:
                lost = player.hp - new_max
                player.hp = new_max
                player.max_hp = new_max
                events.append(f"最大生命 −{amount}（现 {new_max}），生命随之失去 {lost} 点")
            else:
                player.max_hp = new_max
                events.append(f"最大生命 −{amount}（现 {new_max}），当前生命不变")
        elif op == "heal":
            healed = min(amount, player.max_hp - player.hp)
            player.hp += healed
            events.append(f"回复 {healed} 点生命")
        elif op == "lose_hp":
            lost = min(amount, player.hp)
            player.hp -= lost
            events.append(f"失去 {lost} 点生命")
        elif op == "gain_gold":
            player.gold += amount
            events.append(f"获得 {amount} 金币")
        elif op == "lose_gold":
            player.gold = max(0, player.gold - amount)
            events.append(f"失去 {amount} 金币")
        elif op == "lose_all_gold":
            # `PlayerCmd.LoseGold(base.Owner.Gold, …)`（``MorphicGrove.cs:46``）：
            # 量就是**当前金币**，`LoseGold` 夹到 0（``PlayerCmd.cs:198``）。
            lost = player.gold
            player.gold = 0
            events.append(f"失去全部 {lost} 金币")
        elif op == "add_card":
            from .core import CardInstance
            cid = effect.card
            if cid not in CARD_DB:
                events.append(f"⚠️ 要加的卡 {cid!r} 不在卡表里，跳过")
                continue
            for _ in range(max(1, amount)):
                player.deck.append(CardInstance(cid))
            events.append(f"牌组加入 {amount} 张 {CARD_DB[cid].name}")
        elif op == "remove_card_from_deck":
            # ⚠️ 真机是**玩家选**一张移除。这里只在"恰好只有一种可选"时自动做，
            # 否则如实拒绝 —— 随机移除等于换了个机制。
            #
            # ⭐ 候选必须过 ``CardModel.IsRemovable``（``CardModel.cs:738``）：
            # ``IsRemovable => !Keywords.Contains(CardKeyword.Eternal)``。
            # 少了这一条，``AscendersBane`` / ``CurseOfTheBell`` / ``Greed`` /
            # ``Folly`` 这些"永恒"牌会被移除掉 —— 而它们的整个设计目的
            # 就是**移除不掉**（真机连事件里的移除选项都会把它们排除）。
            candidates = [c for c in player.deck
                          if effect.card in (None, c.cid) and _removable(c)]
            if len(candidates) == 1:
                player.deck.remove(candidates[0])
                events.append(f"牌组移除 {CARD_DB[candidates[0].cid].name}")
            elif not candidates:
                events.append("⚠️ 没有可移除的牌（永恒牌不能被移除），跳过")
                continue
            else:
                events.append(f"⚠️ 移除牌需要玩家选择（{len(candidates)} 张候选），跳过")
                continue
        elif op == "random_deck_cards":
            # ⭐ 真机形态（`Whetstone` / `WarPaint` / `WarHammer` / `SandCastle` /
            # `FragrantMushroom`，逐字相同）::
            #
            #     Deck.Cards.Where(c => c.IsUpgradable).ToList()
            #         .StableShuffle(base.Owner.RunState.Rng.Niche)
            #         .Take(base.DynamicVars.Cards.IntValue);
            #     foreach (item in enumerable) CardCmd.Upgrade(item);
            #
            # 即"**随机取 N 张不重复**的牌，再对它们做一件事"。以前它落成
            # "没有目标卡的 `upgrade_card`"，运行期只打一句 ⚠️ 就跳过 ——
            # 遗物算"完全复刻"却什么都不做（`docs/12` §2.18.3）。
            if effect.purpose not in RUN_SELECTION_PURPOSES:
                events.append(f"⚠️ 随机取牌的用途 {effect.purpose!r} 引擎没实现，跳过")
                continue
            stream_rng, ok = _stream_handle(state, effect.rng, rng)
            if not ok:
                events.append(f"⚠️ 随机流 {effect.rng!r} 拿不到，跳过")
                continue
            candidates = _deck_candidates(player, effect.purpose, "deck")
            # ⚠️ 键名是 `type`（`extract_cards.FILTER_PATTERNS` 的 key），
            # 不是 `card_type`（那是 `generate_card` 的写法）—— 两个都读。
            spec = dict(effect.card_filter)
            wanted = str(spec.get("type") or spec.get("card_type") or "")
            if wanted:
                candidates = tuple(index for index in candidates
                                   if _card_type(player.deck[index]).lower()
                                   == wanted.lower())
            picked = _random_pick(candidates, max(0, int(effect.amount or 0)),
                                  effect.rng, stream_rng, effect.pick)
            chosen = [player.deck[index] for index in picked]
            for card in chosen:
                if effect.purpose == "upgrade":
                    card.upgraded = True
                    events.append(f"随机升级 {CARD_DB[card.cid].name}")
                elif effect.purpose == "removal":
                    # ⚠️ 按**实例**移除：`CardInstance` 的 `==` 按字段比，
                    # 两张同名同级的牌会被判成相等 —— 用 `is` 才准。
                    player.deck = [held for held in player.deck if held is not card]
                    events.append(f"随机移除 {CARD_DB[card.cid].name}")
                elif effect.purpose == "downgrade":
                    # 真机 `Reflections.TouchAMirror`：随机降级 N 张。
                    card.upgraded = False
                    events.append(f"随机降级 {CARD_DB[card.cid].name}")
                else:                       # pragma: no cover - 上面已挡
                    events.append(f"⚠️ 随机取牌的用途 {effect.purpose!r} 没有实现动作")
            if not chosen:
                events.append("⚠️ 没有符合条件的牌（候选为空），跳过")
        elif op == "add_random_cards":
            # ⭐ 真机 `DistinguishedCape.AfterObtained`：从**卡池**（不是牌组）
            # 随机取 N 张**不重复**的牌造出来加进牌组::
            #
            #     List<CardModel> pool = CardPool<CurseCardPool>().GetUnlockedCards(…)
            #         .Where(c => c.CanBeGeneratedByModifiers).ToList();
            #     for (i < Curses) {
            #         c = RunState.Rng.Niche.NextItem(pool);  pool.Remove(c);
            #         await CardPileCmd.Add(RunState.CreateCard(c, owner), PileType.Deck);
            #     }
            spec = dict(effect.card_filter)
            pool_name = str(spec.get("pool") or "")
            members = [cid for cid in CARD_POOLS.get(pool_name, ())
                       if cid in CARD_DB]
            if spec.get("generatable"):
                # `c.CanBeGeneratedByModifiers`（`CardModel.cs` 默认 true，
                # 全库 8 张覆写成 false —— 见 `NOT_GENERATED_BY_MODIFIERS`）。
                from .content import NOT_GENERATED_BY_MODIFIERS
                members = [cid for cid in members
                           if cid not in NOT_GENERATED_BY_MODIFIERS]
            if not members:
                events.append(f"⚠️ 卡池 {pool_name!r} 没有可用成员，跳过")
                continue
            stream_rng, ok = _stream_handle(state, effect.rng, rng)
            if not ok:
                events.append(f"⚠️ 随机流 {effect.rng!r} 拿不到，跳过")
                continue
            picked = _random_pick(tuple(range(len(members))),
                                  max(0, int(effect.amount or 0)),
                                  effect.rng, stream_rng, effect.pick)
            from .core import CardInstance
            for index in picked:
                cid = members[index]
                player.deck.append(CardInstance(cid))
                events.append(f"牌组获得 {CARD_DB[cid].name}")
        elif op == "clone_deck":
            # ⭐ 真机 `Reflections.Shatter`：把牌组里**每一张**克隆一份再加进牌组
            # （牌组翻倍），随后另加一张诅咒。
            # ``RunState.CloneCard`` 复制的是"同一张牌的另一个实例"——
            # 引擎侧就是同 ``cid``、同升级状态、**新 uid** 的副本。
            # ⚠️ 必须先把原牌**快照**下来再往牌组里追加：直接迭代
            # `player.deck` 会边加边遍历，牌组无限增长（而且是死循环）。
            from .core import CardInstance

            originals = list(player.deck)
            for card in originals:
                player.deck.append(CardInstance(card.cid, card.upgraded))
            events.append(f"牌组复制一份（{len(originals)} → {len(player.deck)} 张）")
        elif op == "upgrade_card":
            targets = [c for c in player.deck if not c.upgraded]
            if len(targets) == 1 or (effect.card and effect.card in CARD_DB):
                chosen = (targets[0] if len(targets) == 1
                          else next(c for c in player.deck if c.cid == effect.card))
                chosen.upgraded = True
                events.append(f"升级 {CARD_DB[chosen.cid].name}")
            else:
                events.append("⚠️ 升级牌需要玩家选择，跳过")
                continue
        elif op == "transform_card":
            from .core import CardInstance
            # ``CardModel.IsTransformable``（``CardModel.cs:740-751``）：
            # ``!IsRemovable`` 时**只有当这张牌不在牌组里**才可转化 ——
            # 也就是说牌组里的永恒牌不能被转化（战斗内则可以）。
            if effect.card and effect.card in CARD_DB:
                player.deck = [CardInstance(effect.card)
                               if c.cid == effect.card and _removable(c) else c
                               for c in player.deck]
                events.append(f"转化为 {CARD_DB[effect.card].name}")
            else:
                events.append("⚠️ 随机转化需要玩家选择，跳过")
                continue
        elif op == "enchant_card":
            events.append(f"⚠️ 附魔 {effect.card!r} 尚未实现（附魔系统未建），跳过")
            continue
        elif op == "offer_rewards":
            # `RewardsCmd.OfferCustom`：**交给环境去选**，不在这里替玩家决定。
            if not offer_rewards(state, effect, events):
                continue
        else:
            events.append(f"⚠️ Run 层不认识的算子 {op!r}，跳过")
            continue
        done += 1
    return done


def resolve_run_selection(state, internal_index: int, events: list[str],
                          obtain_relic=None) -> bool:
    """玩家选中牌组里的第 ``internal_index`` 张牌，结算这次选牌。

    返回 ``True`` 表示**这次选牌结算完了**（可能还有下一张要选，
    由 ``state.pending_selection`` 是否仍存在判断）。

    用途语义（都以源码为准）：

    * ``removal`` —— ``CardPileCmd.RemoveFromDeck``：把牌移出牌组（真机进 Limbo）；
    * ``upgrade`` —— ``CardCmd.Upgrade``：升到升级版；
    * ``transform`` —— ``CardCmd.TransformToRandom(card, rng, …)``
      （``CardCmd.cs:325-328`` → ``CardTransformation.GetReplacement``）：
      换成 ``CardFactory.CreateRandomCardForTransform`` 掷出的另一张牌。
      ⚠️ 随机源是**调用方给的**那条（事件用 ``base.Rng``），
      所以挂起时已经把它存在 ``selection.rng`` 上。
    """
    selection = getattr(state, "pending_selection", None)
    if selection is None:
        raise ValueError("非法动作：当前没有等待中的选牌")
    if internal_index not in selection.candidates:
        raise ValueError(f"非法动作：{internal_index} 不在候选里")
    player = state.player
    card = player.deck[internal_index]
    name = CARD_DB[card.cid].name if card.cid in CARD_DB else card.cid

    if selection.purpose == "removal":
        # ⚠️ 再确认一次资格：永恒牌不可移除（`IsRemovable`）
        if not _removable(card):
            raise ValueError(f"非法动作：{name} 是永恒牌，不能被移除")
        player.deck.pop(internal_index)
        events.append(f"牌组移除 {name}")
    elif selection.purpose == "upgrade":
        if card.upgraded:
            raise ValueError(f"非法动作：{name} 已经升级过了")
        card.upgraded = True
        events.append(f"升级 {name}")
    elif selection.purpose == "transform":
        _transform_deck_card(state, internal_index, selection, events)
    elif selection.purpose == "duplicate":
        # ⭐ `DollysMirror`：`RunState.CloneCard(cardModel)` 之后把副本加回牌组。
        # 与 `clone_deck` 同一语义（同一张牌的另一个实例：同 cid、同升级状态、
        # **新 uid**），只是这里"哪一张"是**玩家选的**。
        from .core import CardInstance

        player.deck.append(CardInstance(card.cid, card.upgraded))
        events.append(f"复制 {name}（牌组现有 {len(player.deck)} 张）")
    else:                                   # pragma: no cover - 入口已过滤
        raise ValueError(f"未实现的选牌用途 {selection.purpose!r}")

    chosen = selection.chosen + (internal_index,)
    remaining_count = selection.count - 1
    if remaining_count > 0:
        # 下标会因 `pop` 偏移 → **重算**候选，不能复用旧的下标集合
        candidates = _deck_candidates(player, selection.purpose,
                                      selection.source)
        # 已经选过的牌不能再选（升级过的会自然被 `upgrade` 的资格判据排掉）
        candidates = tuple(i for i in candidates if i not in chosen)
        if not candidates:
            events.append("⚠️ 还要再选，但没有候选了（提前结束这次选牌）")
            remaining_count = 0
        else:
            state.pending_selection = RunSelection(
                purpose=selection.purpose, count=remaining_count,
                source=selection.source, candidates=candidates,
                rest=selection.rest, chosen=chosen, rng=selection.rng)
            events.append(f"还要再选 {remaining_count} 张")
            return False

    state.pending_selection = None
    events.append("选牌结束")
    if selection.rest:
        apply_run_effects(state, selection.rest, events,
                          obtain_relic=obtain_relic, rng=selection.rng)
    return True


def _transform_deck_card(state, internal_index: int, selection, events: list[str]) -> None:
    """把牌组里第 ``internal_index`` 张换成随机的另一张（``CardCmd.Transform``）。

    ⚠️ **原地替换**（真机 ``CardCmd.Transform`` 把新牌放回原牌堆的**同一个位置**，
    ``CardCmd.cs:398-420``）：不是"先删后加" —— 那会把牌组顺序打乱，
    而牌组顺序在真机里是玩家看得见、并且会被 ``Headbutt`` 这类牌引用的。
    """
    from .core import CardInstance

    player = state.player
    card = player.deck[internal_index]
    name = CARD_DB[card.cid].name if card.cid in CARD_DB else card.cid
    replacement = transform_replacement(card.cid, selection.rng)
    if replacement is None:
        events.append(f"⚠️ {name} 没有可转化的候选，跳过")
        return
    # 真机 `CardModel.IsUpgraded` 的转化结果**不保留升级**：
    # `CreateRandomCardForTransform` 造的是基础形态的新牌（`CardFactory.cs:177-181`）。
    player.deck[internal_index] = CardInstance(replacement)
    new_name = CARD_DB[replacement].name if replacement in CARD_DB else replacement
    events.append(f"转化 {name} → {new_name}")
