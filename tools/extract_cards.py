"""从反编译源码抽取卡牌效果（``docs/09`` L3）。

**为什么不解析描述文本**：英文描述是给人读的，`"Deal 8 damage. Apply 2 Vulnerable."`
这种句式能覆盖一半，剩下的 `"At the start of your turn…"`、`"Whenever you play a card…"`
永远解不准。而 C# 里写的是**可执行代码**：

    await DamageCmd.Attack(base.DynamicVars.Damage.BaseValue)
        .FromCard(this, cardPlay).Targeting(cardPlay.Target).Execute(choiceContext);
    await PowerCmd.Apply<VulnerablePower>(choiceContext, cardPlay.Target,
        base.DynamicVars.Vulnerable.BaseValue, base.Owner.Creature, this);

所以效果从源码抽，描述文本只留作交叉验证与人工核对（``docs/06`` §6.4）。

设计要点
--------
1. **量用变量引用，不用死数字**。源码的升级写的是
   ``base.DynamicVars.Damage.UpgradeValueBy(3m)`` —— 改的是**基础值**，
   不是"升级后再加一段效果"。所以效果里存 ``amount_var="Damage"``，
   升级版由 ``vars[name].base + upgrade_delta`` 算出来。这样升级前后
   **不可能对不上**，而抄两份数字一定会对不上。
2. **认不出的命令必须报出来**，不能静默丢弃（``docs/02`` §2.5）。
3. 目标由命令参数决定，不猜：``PowerCmd.Apply`` 的第 3 个参数是
   ``cardPlay.Target``（敌人）还是 ``base.Owner.Creature``（自己）。
4. ``amount_raw`` 保留原始表达式文本。量解析不出时，它是唯一能看出
   "到底少认了什么写法"的线索。

用法
----
    python tools/extract_cards.py                    # 抽取 + 打印覆盖率
    python tools/extract_cards.py --out data/content/repo/cards_source.json

⚠️ **不要用 PowerShell 的字符串替换来改本文件**：`Get-Content`/`Set-Content`
往返会按系统 ANSI 重新编码，把中文注释写成非法字节（实测把本文件写坏过一次，
第 59 字节起 UTF-8 解不开，只能整份重写）。改代码用编辑工具，别走 shell 文本替换。
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

SRC_ROOT = Path("data/decompiled/sts2/MegaCrit.Sts2.Core.Models.Cards")
OUT_DEFAULT = Path("data/content/repo/cards_source.json")

#: 纯表现层命令，对机制没有任何影响 —— 不算"未支持"。
COSMETIC = {
    ("CreatureCmd", "TriggerAnim"), ("CreatureCmd", "TriggerAnimIfAlive"),
    ("VfxCmd", "PlayOnCreatureCenter"), ("VfxCmd", "Play"), ("SfxCmd", "Play"),
    ("VfxCmd", "PlayOnCard"), ("CardCmd", "PreviewCardPileAdd"), ("CardCmd", "Preview"),
    ("CardCmd", "SetPortrait"), ("CardCmd", "PlayParticle"),
    ("VfxCmd", "PlayFullScreenInCombat"), ("VfxCmd", "PlayOnCreature"),
    ("VfxCmd", "PlayVfx"), ("VfxCmd", "GetSideCenterFloor"),
    # ⚠️ 下面这些是**怪物招式**里的装饰/纯表现命令。不忽略它们，
    # 每一招都会被判成"含未支持命令" —— 而它们一个数值都不影响，
    # 结果是整批招式白白排除在外（实测 30+ 条）。
    ("VfxCmd", "PlayOnCreatureCenters"), ("VfxCmd", "GetSideCenter"),
    ("SfxCmd", "PlayLoop"), ("SfxCmd", "StopLoop"), ("SfxCmd", "SetParam"),
    ("TalkCmd", "Play"), ("ThinkCmd", "Play"),
    ("CreatureCmd", "TriggerAnimIfAlive"),
    # `DamageCmd.Attack(...)` 的链式配置：这些只是修饰符，不是独立效果。
    #
    # ⚠️ ``("DamageCmd", "Attack")` **不在这里** —— 它本身是"造成伤害"这条效果的唯一
    # 出处。早先把它当装饰（理由：怪物招式的数值由意图给出）的后果是：
    # 下面的 `DamageCmd.Attack` 专用分支变成**死代码**，卡牌的伤害整条消失，
    # 而且**既不报未支持、也没有效果** —— `Bludgeon` / `BodySlam` / `Clash`
    # 这类纯攻击卡于是静默变成空卡（实测 12 张 attack 卡零效果）。
    # 怪物侧的"数值由意图给出"改由 `damage_from_intent=True` 显式表达。
    ("DamageCmd", "WithHitCount"),
    ("DamageCmd", "FromMonster"), ("DamageCmd", "Execute"),
    ("DamageCmd", "WithAttackerAnim"), ("DamageCmd", "WithAttackerFx"),
    ("DamageCmd", "WithHitFx"), ("DamageCmd", "WithHitVfxNode"),
    ("DamageCmd", "OnlyPlayAnimOnce"), ("DamageCmd", "BeforeDamage"),
    ("DamageCmd", "WithDamageProps"), ("DamageCmd", "Spawning"),
    ("AttackCommand", "Execute"),
    # 非战斗表现层：事件房间的立绘/UI 管线（`NEventRoom.Instance.SetPortrait`、
    # `Instance.Layout.RemoveNodesOnPortrait`）与文本拼装（`locString.Add`）。
    # 它们看着像"改状态的动词"（Set / Add / Remove），实测在事件里各出现 1-2 次，
    # 不忽略就会被报成"未识别的效果写法"，把真缺口淹没在噪声里。
    ("VfxCmd", "PlayNonCombatVfx"), ("NEventRoom", "SetPortrait"),
    ("Layout", "RemoveNodesOnPortrait"), ("locString", "Add"),
    ("CollectionsMarshal", "SetCount"), ("StringHelper", "GetDeterministicHashCode"),
}

#: 表现层的**接收者**：特效/音效节点与容器（``NFanOfKnivesVfx.Create``、
#: ``CombatVfxContainer.AddChildSafely``）。它们的调用看着像"改状态的动词"
#: （``Create`` / ``Add``），于是一批卡和怪物招式被判成"未识别的效果写法"。
#: 只忽略**接收者名字带表现后缀**的，不碰 ``CardModel.AddKeyword`` 这类真效果。
COSMETIC_RECEIVER = re.compile(r"^(?:CombatVfxContainer|.*(?:Vfx|Sfx|Anim|Particle))$")

#: 命令调用的识别式（``PowerCmd.Apply`` / ``CardPileCmd.Add`` …）。
#:
#: ⚠️ **不能要求方法名后面紧跟 ``(``**：真机大量写泛型实参
#: （``PowerCmd.Apply<VulnerablePower>(…)`` / ``CardPileCmd.AddCurseToDeck<CurseOfTheBell>(…)``），
#: 用 ``\w*Cmd\.\w+\s*\(`` 会**整批漏掉** —— 症状是"扫描说这里没有命令"，
#: 于是兜底告警（"函数体非平凡却没抽出效果"）与静默空卡检查双双失灵，
#: 而表面上一切都正常。实测 ``Disintegration.OnChosen`` 就因此被判成"没有命令"。
COMMAND_CALL = re.compile(r"\b(\w+Cmd)\.(\w+)")

#: 引擎的算子词汇表（``sts2_sim.content.Effect.op``）。
#: 抽取阶段就按这张表判"能不能落地"，报告里能直接看出还差什么。
SUPPORTED_OPS = {
    "damage", "damage_all", "block", "apply_power", "draw", "lose_hp",
    "gain_energy", "add_card", "set_power_var",
    # Run 层（遗物 / 事件）
    "gain_max_hp", "lose_max_hp", "transform_card", "enchant_card",
    "remove_card_from_deck", "offer_rewards",
}

#: "询问玩家选牌"的交互命令 → 可执行的 ``select_card`` 算子。
#:
#: ⚠️ 这些**不是"认不出"**，而是需要引擎支持"选一张牌"这个动作。
#: 真机的效果序列是 `抽牌 → 选牌 → 弃牌`，所以选牌必须**落在序列的正确位置**上，
#: 不能在最后补一刀 —— 顺序错了会让"先抽后弃"变成"先弃后抽"。
CHOICE_COMMANDS: dict[tuple[str, str], str] = {
    ("CardSelectCmd", "FromHandForDiscard"): "discard",
    ("CardSelectCmd", "FromHandForUpgrade"): "upgrade",
    ("CardSelectCmd", "FromHandForExhaust"): "exhaust",
    ("CardSelectCmd", "FromHand"): "",          # 用途由 prompt 或外层包装决定
    # 从**指定牌堆**选（Headbutt / Hologram / Dredge…）。牌堆在参数里
    # （`PileType.Discard.GetPile(...)`），用途同样由外层 `CardPileCmd.Add` 决定。
    ("CardSelectCmd", "FromCombatPile"): "",
    # ⭐ **牌组**（非战斗）里的选牌：遗物的 `AfterObtained` 与事件的选项大量用它们。
    # 用途**写在方法名里**（`FromDeckForRemoval` → removal），比 prompt 可靠。
    # ⚠️ 早先这几个只在 `SELECT_METHODS` 里登记、没进这张表，于是它们走到
    # `effect_of_command` 的兜底分支被报成"未支持命令" —— 事件侧 24 个选项
    # 全部卡在这里，而其实只需要"引擎支持一次选牌动作"。
    ("CardSelectCmd", "FromDeck"): "",
    ("CardSelectCmd", "FromDeckForRemoval"): "removal",
    ("CardSelectCmd", "FromDeckForUpgrade"): "upgrade",
    ("CardSelectCmd", "FromDeckForTransformation"): "transform",
    ("CardSelectCmd", "FromDeckForTransform"): "transform",
    ("CardSelectCmd", "FromDeckForEnchantment"): "enchant",
    ("CardSelectCmd", "FromDeckForDuplication"): "duplicate",
    # ⭐ `FromDeckGeneric`（`DollysMirror`）：**通用 prompt**，用途判不出 ——
    # 先按空用途登记，再由 `CloneCard` 反推成 `duplicate`（见 `extract_effects`
    # 收尾处）。不登记它就会走到兜底分支被报成"未支持命令"。
    ("CardSelectCmd", "FromDeckGeneric"): "",
    ("CardSelectCmd", "FromChooseACardScreen"): "choose",
}
#: ``CardSelectCmd`` 的**全部**方法名。用来判断"某个 CardPileCmd.Add 是不是
#: 只是包着一次选牌" —— 只统计已登记的会让 `FromCombatPile` 被误判成集合移动。
SELECT_METHODS: frozenset[str] = frozenset({
    "FromHand", "FromHandForDiscard", "FromHandForUpgrade", "FromHandForExhaust",
    "FromCombatPile", "FromChooseACardScreen",
    # ⚠️ **牌组**（非战斗）里的选牌。遗物的 `AfterObtained` 大量用它们
    # （`Whetstone` 升级一张、`Astrolabe` 转化、`PandorasBox` 移除…）。
    # 不登记的话，`CardSelectCmd.FromDeckForUpgrade(...)` 后面那句
    # `CardCmd.Upgrade(card)` 会被抽成"升级一张**随机**牌" ——
    # 而真机是**让玩家选**。这是"看起来实现了、其实换了个机制"。
    "FromDeckForUpgrade", "FromDeckForEnchantment", "FromDeckForRemoval",
    "FromDeckForTransform", "FromDeck", "FromDeckForDuplication",
    # ⭐ `FromDeckGeneric`（`DollysMirror`）：prompt 是**通用的**
    # （`base.SelectionScreenPrompt`），用途判不出来 —— 靠**后续动作**反推
    # （它后面写的是 `RunState.CloneCard(选中那张)` → `duplicate`）。
    # 不登记它，整条选牌会以 `CardSelectCmd.FromDeckGeneric` 收场（当成缺口），
    # 而真机明明是"让玩家选一张复制"。
    "FromDeckGeneric",
})

#: "选牌的**执行**命令"：参数是选牌结果时，它只是把选中的牌落地，
#: **不是独立效果** —— 效果已经由那条 ``select_card``（带用途）表达了。
#:
#: ⚠️ 不去重就会出现"选一张移除"变成**移除两次**：先是 `select_card(removal)`
#: 的挂起流程移掉一张，紧接着这条 `RemoveFromDeck` 又想移一张
#: （候选已经变了，行为与真机不同，而且日志看不出异常）。
#: 真机是 `选牌 → 用结果执行动作`，效果只在**动作**里发生一次。
SELECTION_ACTIONS: dict[tuple[str, str], str] = {
    ("CardPileCmd", "RemoveFromDeck"): "removal",
    ("CardCmd", "Upgrade"): "upgrade",
    ("CardCmd", "Enchant"): "enchant",
    ("CardCmd", "TransformToRandom"): "transform",
    ("CardCmd", "Transform"): "transform",
    ("CardCmd", "Downgrade"): "downgrade",
}

#: ``CardSelectorPrefs.<X>Prompt`` → 用途。``FromHand`` 走这条判定。
PROMPT_PURPOSES: tuple[tuple[str, str], ...] = (
    ("DiscardSelectionPrompt", "discard"),
    ("ExhaustSelectionPrompt", "exhaust"),
    ("UpgradeSelectionPrompt", "upgrade"),
    ("TransformSelectionPrompt", "transform"),
    ("RetainSelectionPrompt", "retain"),
    ("SlySelectionPrompt", "sly"),
)

#: ``CardSelectCmd.<方法名>`` → 用途。**方法名里就写着用途**，比 prompt 可靠。
#:
#: ⚠️ 不认这张表时，`CardSelectCmd.FromDeckForRemoval(...)` 会被抽成
#: "用途为空的选牌"，最终以 `CardSelectCmd(用途不明)` 收场 —— 而真机里
#: 它的用途明明白白写在方法名上。实测 24 个事件选项卡在这里。
METHOD_PURPOSES: tuple[tuple[str, str], ...] = (
    ("FromDeckForRemoval", "removal"),
    ("FromDeckForUpgrade", "upgrade"),
    ("FromDeckForTransformation", "transform"),
    ("FromDeckForTransform", "transform"),
    ("FromDeckForEnchantment", "enchant"),
    ("FromDeckForDuplication", "duplicate"),
    ("FromHandForDiscard", "discard"),
    ("FromHandForExhaust", "exhaust"),
    ("FromHandForUpgrade", "upgrade"),
    # 从"生成出来的候选"里选（`FromChooseACardScreen`）：候选不在任何牌堆里，
    # 而是上游 `CardFactory.CreateForReward` / `GetDistinctForCombat` 造出来的。
    ("FromChooseACardScreen", "choose"),
    # 通用选牌（`FromDeck` / `FromHand` / `FromCombatPile`）：用途只能靠 prompt
    # 或外层 `CardPileCmd.Add` 判，判不出就如实报"用途不明"。
)


def resolve_selection(body: str, match: re.Match,
                      wrapped: tuple[str, str] | None = None,
                      key_map: dict[str, str] | None = None,
                      locals_: dict[str, str] | None = None) -> dict | None:
    """把一条选牌命令解析成 ``select_card`` 效果；解析不出返回 ``None``。

    ``wrapped`` 是**包在外面的** ``CardPileCmd.Add(X, PileType.Y, [Position])``
    给出的 ``(目标牌堆, 位置)``。有它的时候 **用途由外层决定**，不看 prompt ——
    真机常写 ``CardSelectCmd.FromHand(prefs: new CardSelectorPrefs(base.SelectionScreenPrompt, N), …)``，
    那个 ``SelectionScreenPrompt`` 是个通用"选一张牌"提示，**不带用途信息**
    （实测 4 张卡因此被判成"用途不明"）。

    选几张从 ``new CardSelectorPrefs(prompt, N)`` 的第 2 个参数取；
    从哪个牌堆选则看 ``PileType.X.GetPile(...)``（``FromCombatPile`` 用）。
    """
    args, _end, _open = call_args(body, match.end())
    method_name = match.group(2)
    prompt = ""
    prompt_match = re.search(r"(\w+SelectionPrompt)", args)
    if prompt_match:
        prompt = prompt_match.group(1)

    # 来源牌堆：`FromDeck*` 一律是牌组（方法名里写着），否则看 `PileType.X.GetPile(...)`
    source = "deck" if method_name.startswith("FromDeck") else "hand"
    pile_match = re.search(r"PileType\.(\w+)", args)
    if pile_match:
        source = pile_match.group(1).lower()

    purpose = ""
    if wrapped is not None:
        purpose = f"to_{wrapped[0]}"
    else:
        # ① 用途**写在方法名里**（`FromDeckForRemoval`）——最可靠，先看它
        for token, name in METHOD_PURPOSES:
            if token == method_name:
                purpose = name
                break
        # ② 否则看 prompt（`FromHand` + `DiscardSelectionPrompt`）
        if not purpose:
            for token, name in PROMPT_PURPOSES:
                if token == prompt:
                    purpose = name
                    break

    count: int | None = 1
    count_var: str | None = None
    prefs = re.search(r"new\s+CardSelectorPrefs\s*\(", args)
    if prefs:
        pref_args = [part.strip()
                     for part in split_top(balanced(args, prefs.end() - 1)[0])]
        if len(pref_args) >= 3:
            # 三参构造 = ``CardSelectorPrefs(prompt, min, max)`` → **可变张数**
            # （`gamblers_brew` 的 0..∞）。照实写 0 交给门禁拒绝；
            # ⚠️ 绝不能把 `max` 当成"要选这么多"——那是把"任意张"变成"最多张全选"。
            count, count_var = 0, None
        elif len(pref_args) == 2:
            amount, var, _raw = resolve_amount(pref_args[1], key_map or {},
                                               locals_ or {})
            if amount is not None:
                count = int(amount)
            elif var:
                # 张数来自**声明的变量**（`DynamicVars.Cards.IntValue`）——
                # 带上变量名，由 `content._resolve_effects` 填值。
                # ⚠️ 以前这里只认字面量、其余**兜底成 1**：`YummyCookie`
                # （真机升 4 张）在引擎里只升 1 张，静默变弱一档。
                count, count_var = None, var
            else:
                # 认不出就整条拒绝，不兜底。
                return None
    # ⚠️ 用途可能**后置**（分句写法：先选牌赋给局部变量，后面 `CardPileCmd.Add`
    # 才说明去哪）。这时**照实返回空用途**，由调用方等外层来补 ——
    # 但不能连 `from` 和 `count` 一起丢掉：把它们硬编码成 hand/1 会让
    # "从弃牌堆选"变成"从手牌选"（实测 `hologram` 的候选数因此变成 0）。
    effect = {"op": "select_card", "from": source, "purpose": purpose,
              "count": count, "target": "self", "amount": count}
    if count_var:
        effect["amount_var"] = count_var
    if wrapped is not None and wrapped[1]:
        effect["position"] = wrapped[1]
    return effect


def extract_effects(
    body: str,
    key_map: dict[str, str],
    allow_bookkeeping: bool = False,
    damage_from_intent: bool = False,
    self_card_id: str | None = None,
) -> tuple[list[dict], list[str], int]:
    """按**源码顺序**抽效果，返回 ``(效果, 未支持命令, 选牌命令数)``。

    ⚠️ "源码顺序"要用**括号包含关系**校正，不能只看出现位置：

    * 分成两句写（``Acrobatics``）：``… Draw(…); await CardSelectCmd.FromHandForDiscard(…);
      CardCmd.Discard(choiceContext, cardModel);`` → 位置顺序就是正确顺序。
    * 嵌套写（``Prepared``）：``CardCmd.Discard(choiceContext, await
      CardSelectCmd.FromHandForDiscard(…))`` → ``CardCmd.Discard`` 在文本里**更靠前**，
      按位置排会变成"先弃后选"，正好把顺序搞反。

    所以：若某条 ``Discard``/``Exhaust`` 的**参数括号里**含一次选牌，那它只是外层
    包装，真正的动作是选牌，这条要丢掉。

    ``allow_bookkeeping`` 只给**遗物**路径打开：遗物里"仅记账"的钩子确实等于不做事，
    但**卡牌**里同样的判定会误杀真效果（例：``card.Replay = 2`` 是纯赋值，
    对遗物是记账、对卡牌却是"这张牌打两次"）。所以卡牌**默认关闭**，宁可报残缺，
    也不静默产出空卡。

    ``damage_from_intent`` 只给**怪物招式**路径打开：招式的攻击力来自意图，不来自
    招式函数体（见 :func:`sts2_sim.content._with_intent_damage` 的说明）。
    卡牌没有意图可依，所以默认关闭 —— 此时 ``DamageCmd.Attack`` 的数值抽不出
    就算硬缺口，宁可不采纳这张卡，也不让它"打出去什么都不发生"。
    """
    if not body:
        return [], [], 0
    locals_ = collect_locals(body)
    x_loops = x_loop_spans(body, locals_)
    # ⭐ "上界是常量变量"的循环 → 落在里面的效果重复 N 次（`IceLance` 引导 2 个冰球）。
    count_loops = count_loop_spans(body, key_map, locals_)
    # ⭐ "次数取决于运行期状态"的循环（球数 / 牌数）→ 落在里面的命令**报缺口**。
    runtime_loops = runtime_loop_spans(body, key_map, locals_, x_loops, count_loops)
    # ⭐ "只有升级版才有"的结构性效果（`if (base.IsUpgraded)`）→ 同样报缺口。
    upgraded_blocks = upgraded_only_spans(body)
    #: 走到收尾才统一并入 unsupported 的理由。**不在这里 continue** ——
    #: 那样会把这条命令本来能报的**更具体**的缺口吞掉
    #: （实测 `Darkness` 的 `OrbCmd.Passive` 就是这样消失的）。
    pending_reasons: list[str] = []

    def in_x_loop(position: int) -> bool:
        return any(start < position < end for start, end in x_loops)

    def in_runtime_loop(position: int) -> bool:
        return any(start < position < end for start, end in runtime_loops)

    def in_upgraded_block(position: int) -> bool:
        return any(start < position < end for start, end in upgraded_blocks)

    def count_loop_at(position: int) -> tuple[int | None, str | None, str] | None:
        for start, end, amount, var, loop_var in count_loops:
            if start < position < end:
                return amount, var, loop_var
        return None

    def apply_loop_count(effect: dict, position: int) -> dict:
        """把"外层 ``for`` 循环"的次数套到效果上。

        ⚠️ **`CardPileCmd.Add` 分支走的是另一条路径**（它 `continue` 掉了主循环
        末尾那段套用逻辑），所以那里产出的 `add_card` 必须显式调这个 ——
        不调就是**静默变弱**：`DistinguishedCape` 的第二段真机是
        ``for (i < Cards) CreateCard<Apparition>()``（加 **Cards** 张），
        引擎只会加 **1** 张。
        """
        if effect.get("times_var") or int(effect.get("times") or 1) != 1:
            return effect
        loop = count_loop_at(position)
        if loop is None:
            return effect
        amount, var, _loop_var = loop
        if var:
            effect["times_var"] = var
        elif amount and amount > 1:
            effect["times"] = amount
        return effect

    effects: list[dict] = []
    unsupported: list[str] = []
    #: 已经被别的识别器**消费掉**的位置区间（如生成牌自带的
    #: ``SetToFreeThisTurn``）—— 交给 `silent_gap_markers` 排除。
    consumed_spans: list[tuple[int, int]] = []
    choices = 0
    matches = sorted(COMMAND_CALL.finditer(body), key=lambda m: m.start())

    # ⭐ ``<Power>.Steal(...)`` —— **不是** ``*Cmd.*`` 调用，命令扫描整条看不见它：
    # `ThievingHopper.ThieveryMove` 的"偷走玩家牌组里的一张牌"因此**静默消失**
    # （抽取结果只剩一个 damage，而且**没有** unsupported 标记 —— 这正是最难发现的一类）。
    #
    # ⚠️ 现在**只记缺口、不产算子**：偷牌需要"选哪张（`_stealPriorities` 四级优先）+
    # 从战斗里移走 + 存进能力实例 + 死亡时经 Run 层奖励归还"一整套，引擎还没实现。
    # 直接产一个引擎不认识的算子会让这些招式**整条被拒**（覆盖倒退）；
    # 记成 unsupported 则是"如实标残缺但招式照跑" —— 与门禁的口径一致。
    pending_ops: list[tuple[int, dict | str]] = []
    for steal in re.finditer(r"\.\s*Steal\s*\(", body):
        if "SwipePower" in body:
            pending_ops.append((steal.start(),
                                "SwipePower.Steal(牌)：偷牌 + 死亡归还引擎未实现"))
        else:
            # `ThieveryPower.Steal()`：偷**金币** —— 那是 Run 层的资源，
            # 战斗引擎拿不到（`PlayerCmd.LoseGold`）。
            pending_ops.append((steal.start(),
                                "ThieveryPower.Steal()（Run 侧金币）引擎未实现"))

    # ⭐ "造出来就是升级版"（`CunningPotion` / `CosmicConcoction`）：
    # 这些牌被逐张 `CardCmd.Upgrade`，所以造牌效果要带上 `upgrade` 标志。
    upgraded_generated, upgraded_starts = generated_upgrades(body, locals_)

    # ⭐ "从牌组随机取 N 张不重复再升级/降级"（`Whetstone` / `WarPaint` …）：
    # 产出一条 `random_deck_cards`，并把 foreach 里那条 `CardCmd.Upgrade`
    # **消费**掉（它的动作已经由 `purpose` 表达）。
    random_pick_vars: set[str] = set()
    for pick_effect, pick_var, pick_at in random_deck_picks(body, locals_, key_map):
        pending_ops.append((pick_at, pick_effect))
        random_pick_vars.add(pick_var)
    # ⭐ `CardCmd.Upgrade(<流>.NextItem(<集合>))`：随机取**一张**同一件事，
    # 但写法是"命令直接套在 NextItem 外面"（`EndlessConveyor.ObserveChef`）。
    random_item_spans: list[tuple[int, int]] = []
    for item_effect, item_start, item_end in random_item_picks(body, locals_, key_map):
        pending_ops.append((item_start, item_effect))
        random_item_spans.append((item_start, item_end))
    # ⭐ 循环形态：`for (i < N) { c = <流>.NextItem(集合); 集合.Remove(c); Upgrade(c) }`
    # （`Reflections.TouchAMirror`）—— 取 N 张**不重复**。
    for loop_effect, loop_start, loop_end in random_remove_loop_picks(
            body, locals_, key_map):
        pending_ops.append((loop_start, loop_effect))
        random_item_spans.append((loop_start, loop_end))
    #: 已被"循环级"识别器**消费**掉的区间（`clone_deck` / `add_random_cards`）——
    #: 落在里面的 `CardPileCmd.Add` 不再单独产效果。
    clone_deck_spans: list[tuple[int, int]] = []
    # ⭐ "从**卡池**随机取 N 张不重复 → 加入牌组"（`DistinguishedCape`）。
    for pool_effect, pool_start, pool_end in add_random_pool_picks(
            body, locals_, key_map):
        pending_ops.append((pool_start, pool_effect))
        clone_deck_spans.append((pool_start, pool_end))
    # ⭐ "整副牌组复制一份"（`Reflections.Shatter`）：循环体里的 `CardPileCmd.Add`
    # 与 `CloneCard` 都由这一条效果表达，要一并**消费**掉。
    for clone_effect, clone_start, clone_end in clone_deck_picks(body):
        pending_ops.append((clone_start, clone_effect))
        random_item_spans.append((clone_start, clone_end))
        clone_deck_spans.append((clone_start, clone_end))
        # 这个循环的次数由 `clone_deck` 自己表达（真机是"牌组的张数"，而
        # `for (i < originalDeckSize)` 的上界是运行期局部变量）→
        # 不再算"运行期次数"缺口。**只对认出来的那个循环**生效。
        runtime_loops[:] = [span for span in runtime_loops
                            if not (clone_start <= span[0] <= clone_end)]

    # ⭐ **静态造牌工厂**：``Shiv.CreateInHand(owner, state)`` /
    # ``Soul.CreateInHand(owner, n, state)``。这类**不是** ``*Cmd.*`` 调用，
    # 命令扫描整条看不见它 —— 而它真的往手牌里加牌。
    #
    # 后果正是本项目最难发现的一类错误："报告干净、卡却比真机弱"：
    # 实测 ``CloakAndDagger`` 只剩 6 点格挡、``LeadingStrike`` 只剩 3 点伤害，
    # 那张 / 那两张 Shiv 静默消失（`_unconsumed_mechanics` 事后能标残缺，
    # 但救不回数值，也说不清缺的是哪一句）。
    #
    # 出处（语义都写在**工厂方法自己**里，不靠卡面文本）：
    #   ``Shiv.CreateInHand(Player, ICombatState, Player?)``        Shiv.cs:83
    #       → ``CreateInHand(owner, 1, …)``，即**一张**。
    #   ``Shiv.CreateInHand(Player, int count, …)``                 Shiv.cs:88
    #       → ``CardPileCmd.AddGeneratedCardsToCombat(shivs, PileType.Hand, …)``
    #   ``Soul.CreateInHand(Player, int amount, …)``                Soul.cs:31
    #       → 同样进 ``PileType.Hand``。
    # 所以两者都是"造 N 张、放进**手牌**"，N 来自第 2 个实参；
    # 两参重载的 N 恒为 1，真正的张数由**包住它的 for 循环**给出
    # （``for (int i = 0; i < base.DynamicVars.Cards.IntValue; i++)``）。
    for factory in re.finditer(r"\b(\w+)\.CreateInHand\s*\(", body):
        card_id = snake(factory.group(1))
        if card_id not in known_card_ids():
            continue                     # 不是卡牌类 → 不是造牌工厂
        args, _end, _open = call_args(body, factory.end() - 1)
        pieces = [p.strip() for p in split_top(args)]
        # 形如 `(owner, state)` = 一张；`(owner, count, state)` = count 张。
        explicit = pieces[1] if len(pieces) >= 3 else None
        loop = count_loop_at(factory.start())
        if in_x_loop(factory.start()):
            # X 张（`for i < ResolveEnergyXValue()`）—— 静态抽不出来，
            # 猜成 1 就是强度差 X 倍。如实记缺口，由上游拒绝这张卡。
            pending_ops.append((factory.start(),
                                f"{factory.group(1)}.CreateInHand(X 张，运行期取值)"))
            continue
        if explicit is not None and loop is not None:
            # 循环 N 次 × 每次 M 张 —— 引擎的 `add_card` 只有一个 `amount`，
            # 表达不了乘积。宁可报缺口也不把 N×M 写成其中一个。
            pending_ops.append((factory.start(),
                                f"{factory.group(1)}.CreateInHand(循环 × 张数，乘积算不出)"))
            continue
        if loop is not None:
            amount, var = loop[0], loop[1]
            raw = f"循环上界 {var or amount}"
        else:
            amount, var, raw = resolve_amount(explicit, key_map, locals_) \
                if explicit else (1, None, "1")
        effect = _effect("add_card", amount, var, None, "self", card=card_id,
                         pile="hand", position="", raw=raw)
        if factory.start() in upgraded_starts:
            # `CunningPotion`：这批 Shiv 是**已升级**的。
            effect["upgrade"] = True
        if amount is None and var is None:
            pending_ops.append((factory.start(),
                                f"{factory.group(1)}.CreateInHand(张数抽不出：{(explicit or '')[:32]})"))
            continue
        pending_ops.append((factory.start(), effect))

    pending_ops.sort(key=lambda item: item[0])

    def flush_pending(before: int) -> None:
        while pending_ops and pending_ops[0][0] < before:
            _pos, item = pending_ops.pop(0)
            if isinstance(item, dict):
                effects.append(item)
            else:
                unsupported.append(item)

    select_spans = [call_span(body, m) for m in matches
                    if m.group(1) == "CardSelectCmd" and m.group(2) in SELECT_METHODS]

    # 外层包装：`CardPileCmd.Add(X, PileType.Y, [Position])` 决定"选完之后去哪"。
    # 用**括号包含关系**认它，而不是看参数文本 —— 参数可能是个多层嵌套的调用。
    wraps: list[tuple[int, int, str, str]] = []
    for match in matches:
        if (match.group(1), match.group(2)) != ("CardPileCmd", "Add"):
            continue
        start, end = call_span(body, match)
        args, _e, _o = call_args(body, match.end())
        pile, position = resolve_pile(args)
        wraps.append((start, end, pile, position))

    def enclosing_wrap(span: tuple[int, int]) -> tuple[str, str] | None:
        for start, end, pile, position in wraps:
            if start < span[0] and span[1] < end:
                return pile, position
        return None

    def wrap_of_local(name: str) -> tuple[str, str] | None:
        """分句写法：``cardModel = await CardSelectCmd.FromCombatPile(…);`` 之后
        ``CardPileCmd.Add(cardModel, PileType.Draw, …)`` —— 参数是**局部变量**，
        包含关系不成立，必须跟着变量回到那次选牌。
        """
        definition = locals_.get(name, "")
        if "CardSelectCmd" not in definition:
            return None
        for start, end, pile, position in wraps:
            if name in body[start:end]:
                return pile, position
        return None

    # 选牌可能**先出现、用途后出现**（分句写法），所以要先把"哪次选牌赋给了哪个
    # 局部变量"记下来，等看到 `CardPileCmd.Add(那个变量, PileType.Y)` 再补用途。
    # 补不上用途的选牌最后统一报成"用途不明"，不能带着空用途进效果表。
    selection_of_local: dict[str, int] = {}      # 局部变量名 → effects 下标
    unresolved_selections: list[int] = []        # 用途待定的 effects 下标

    for match in matches:
        cmd_class, method = match.group(1), match.group(2)
        # 先把"位置在这条命令之前"的偷牌 / 造牌动作按序放进来。
        flush_pending(match.start())
        # ⭐ 落在"次数取决于运行期状态"的循环里（`Shatter` 按球数激发、
        # `EvilEye` 按条件 1/2 次）：引擎没有"按集合长度/条件重复"的算子，
        # 把循环体当成跑一次就是**静默的机制替换**（报告里这张卡还是"可执行"）。
        # 如实报缺口 → 整张卡标残缺、排除出训练集。
        if in_runtime_loop(match.start()):
            marker = "循环次数取决于运行期状态（球数/牌数/条件）引擎未实现"
            if marker not in pending_reasons:
                pending_reasons.append(marker)
        # ⭐ 落在 `if (base.IsUpgraded)` 块里：引擎只建模数值升级，
        # 无条件抽出来就是"基础版比真机强"（`Spinner` 白得一个玻璃球）。
        if in_upgraded_block(match.start()):
            marker = "升级版专属效果（if (base.IsUpgraded)）引擎未实现"
            if marker not in pending_reasons:
                pending_reasons.append(marker)
        if (cmd_class, method) in CHOICE_COMMANDS:
            selection = resolve_selection(body, match,
                                          enclosing_wrap(call_span(body, match)),
                                          key_map, locals_)
            if selection is None:
                choices += 1
                unsupported.append(f"{cmd_class}.{method}(来源不明)")
                continue
            if not selection.get("purpose"):
                # 用途还没定：先占位，等后面的 `CardPileCmd.Add` 来补。
                # `from` / `count` 已经正确填好了。
                start, end = call_span(body, match)
                selection["_span"] = (start, end)
                unresolved_selections.append(len(effects))
            effects.append(selection)
            continue
        # "选牌的**执行**命令"：参数是选牌结果时它只是落地那条选牌，
        # 不能再产出一条效果（否则"选一张移除"会移除两次）。
        if (cmd_class, method) in SELECTION_ACTIONS:
            args, _e, _o = call_args(body, match.end())
            first = (split_top(args) or [""])[0].strip()
            # 参数是"选牌结果"的两种写法都算：
            #   * 直接是那个局部变量（`CardCmd.Upgrade(cardModel)`）；
            #   * 是"遍历选牌结果"的 foreach 变量（`foreach (item in list)
            #     CardCmd.Upgrade(item)`，`Pomander` / `YummyCookie`）。
            is_selection_result = (
                (first in locals_ and "CardSelectCmd" in locals_.get(first, ""))
                or foreach_over_selection(body, locals_, first))
            if is_selection_result:
                # 往回找**最近那次**选牌：用途若还没定，就用动作反推
                for index in range(len(effects) - 1, -1, -1):
                    if effects[index].get("op") == "select_card":
                        if not effects[index].get("purpose"):
                            effects[index]["purpose"] = SELECTION_ACTIONS[(cmd_class, method)]
                            if index in unresolved_selections:
                                unresolved_selections.remove(index)
                                effects[index].pop("_span", None)
                        break
                continue
        if (cmd_class, method) == ("CardPileCmd", "Add"):
            args, _e, _o = call_args(body, match.end())
            first = (split_top(args) or [""])[0].strip()
            # `clone_deck`（`Reflections.Shatter`）已经把"克隆每一张 + 加回牌组"
            # 整段表达了 → 循环体里的这条 `Add` 要**消费**掉，
            # 否则它会去解析 `CloneCard(...)` 并报"卡 id 解析不出"。
            if any(start <= match.start() < stop
                   for start, stop in clone_deck_spans):
                continue
            # `DollysMirror`：`Add(CloneCard(选中的那张), Deck)` —— "复制"这件事
            # 已经由 `select_card(purpose="duplicate")` 表达了（见收尾处的
            # `CloneCard` 反推），这条 `Add` 不能再单独产效果。
            if "CloneCard" in locals_.get(first, ""):
                continue
            pile, position = resolve_pile(args)
            add_start, add_end = call_span(body, match)
            # ⚠️ 方向别搞反：要问的是"**选牌是否在这个 Add 的参数里**"
            # （`Add(await CardSelectCmd.…(…), PileType.Hand)`），
            # 而不是"这个 Add 是否在某次选牌里"。反了的话选牌包装会被当成集合，
            # `Dredge` / `NeowsFury` 这类卡会报"谓词认不出"。
            wraps_select = any(add_start < s and e < add_end
                               for s, e in select_spans)
            if first == "this":
                # 把**打出的这张牌本身**移进目标牌堆（Bolas / MakeItSo …）
                effects.append({"op": "move_self_to_pile", "pile": pile,
                                "position": position, "target": "self", "amount": 0})
            elif wraps_select:
                pass          # 选牌那条已经带上 purpose 了
            elif first in locals_ and "CardSelectCmd" in locals_.get(first, ""):
                # 分句写法：参数是"刚才那次选牌"的结果 → 把用途补回那次选牌
                wrapped = wrap_of_local(first)
                target_index = unresolved_selections[0] if unresolved_selections else None
                if wrapped is not None and target_index is not None:
                    effects[target_index]["purpose"] = f"to_{wrapped[0]}"
                    if wrapped[1]:
                        effects[target_index]["position"] = wrapped[1]
                    selection_of_local[first] = target_index
                    unresolved_selections.remove(target_index)
                else:
                    unsupported.append("CardPileCmd.Add(找不到对应的选牌)")
            else:
                # ⭐ 参数是"**造出来的一张具体的卡**"：真机写
                #
                #     CardModel card = base.Owner.RunState.CreateCard<Apotheosis>(base.Owner);
                #     await CardPileCmd.Add(card, PileType.Deck);
                #
                # 即"往牌组加一张指定卡"（`JewelryBox` 加 Apotheosis、
                # `LargeCapsule` / `NeowsBones` 同理）。这类**不是**集合移动，
                # 但以前会掉进"集合，谓词认不出"那个桶里被整条拒掉 ——
                # 19 个遗物的拾取效果就是这么丢的。
                made = resolve_card_id(first, body, locals_,
                                       self_card_id=self_card_id)
                if made is not None:
                    effects.append(apply_loop_count(
                        {"op": "add_card", "card": made, "pile": pile,
                         "position": position, "target": "self", "amount": 1},
                        match.start()))
                    continue
                # 参数是 `item` / `cards` / `array` 这类**局部集合**：
                # 由 `.Where(<谓词>)` 选出，走结构化解析（认不出就拒绝）。
                collection = extract_collection_move(locals_, body, args)
                if collection is None:
                    # `foreach (item in PileType.Hand.GetPile(p).Cards) Add(item, Draw)`
                    # —— 逐张搬 ≡ **整堆**搬（`Reboot`）。
                    source_pile = foreach_source_pile(body, first.strip())
                    if source_pile:
                        effects.append({
                            "op": "move_all_matching", "from": source_pile,
                            "filter": (), "take_random": "", "pile": pile,
                            "position": position, "target": "self",
                            "amount": 0})
                        continue
                    unsupported.append("CardPileCmd.Add(集合，谓词认不出)")
                else:
                    collection["pile"] = pile
                    if position:
                        collection["position"] = position
                    collection["target"] = "self"
                    collection["amount"] = 0
                    effects.append(collection)
            continue
        parsed = effect_of_command(body, match, key_map, locals_, x_loops,
                                   damage_from_intent=damage_from_intent,
                                   self_card_id=self_card_id,
                                   upgraded_generated=upgraded_generated,
                                   random_pick_vars=random_pick_vars,
                                   random_item_spans=tuple(random_item_spans))
        if parsed is None:
            unsupported.append(f"{cmd_class}.{method}")
            continue
        if parsed.get("op") == "_cosmetic":
            continue
        # 落在"上界依赖 X 的 for 循环"里的效果，次数就是 X（`Tempest` 引导 X 个球）
        if in_x_loop(match.start()) and "times_x" not in parsed:
            parsed["times_x"] = True
        # 落在"上界是常量变量的 for 循环"里的效果，次数就是那个变量
        # （`IceLance` 的 `for i < Repeat(2)` → 引导 2 个冰球）。
        # ⚠️ 只在效果**没有**自己的次数时套用：`DamageCmd.Attack(...).WithHitCount(3)`
        # 的次数来自链式调用，不能被外层循环覆盖。
        if "times_x" not in parsed and not parsed.get("times_var") \
                and int(parsed.get("times") or 1) == 1:
            loop = count_loop_at(match.start())
            if loop is not None:
                amount, var, loop_var = loop
                if var:
                    parsed["times_var"] = var
                    # ⭐ `Quadcast`：`EvokeNext(…, i == Repeat - 1)` —— 只有**最后一次**
                    # 移除球，前几次是"激发但不移除"。次数抽出来了还不够，
                    # `dequeue` 也得跟着循环变量走，否则第一次就把球移走，
                    # 后面几次等于激发空气（比"只激发 1 次"更错）。
                    raw = str(parsed.get("dequeue_raw") or "")
                    if loop_var and re.search(rf"\b{re.escape(loop_var)}\b", raw):
                        parsed.pop("dequeue", None)
                        parsed["dequeue_tail"] = True
                elif amount and amount > 1:
                    parsed["times"] = amount
        if parsed.get("op") == "_generated_cards":
            # ⭐ ``GetDistinctForCombat`` 造牌：一条 C# 语句可能对应**多条**效果
            # （`OrobicAcid` 的三次 ``AddRange``），所以单独走一个展开分支。
            # ⚠️ 必须排在下面"任何 ``_`` 开头的伪算子都变缺口"之前 ——
            # 否则它会被自己的下划线前缀当成缺口拒掉。
            effects.extend(parsed["effects"])
            consumed_spans.extend(parsed.get("consumed") or ())
            continue
        if parsed.get("op") == "_card_unresolved":
            unsupported.append(parsed["detail"])
            continue
        if parsed.get("op") == "_times_unresolved":
            unsupported.append(parsed["detail"])
            continue
        if parsed.get("op") == "_damage_unresolved":
            unsupported.append(parsed["detail"])
            continue
        if parsed.get("op") == "_amount_unresolved":
            unsupported.append(parsed["detail"])
            continue
        # ⭐ 兜底：**任何 ``_`` 开头的伪算子都必须变成显式缺口**。
        # 伪算子是"抽取器认出来了但落不了地"的中间态（`_card_unresolved` /
        # `_relic_unresolved` / `_damage_unresolved` …）。少写一个分支的后果是
        # 它**留在效果表里**冒充真效果 —— 上游只会看到"有个 op 叫 _relic_unresolved"，
        # 而下游（content）会把它当成一条普通效果去解析，于是报出
        # "效果的量抽不出"这种**理由错**的缺口。实测 `RelicCmd.Obtain` 走的就是这条路。
        if str(parsed.get("op", "")).startswith("_"):
            unsupported.append(parsed.get("detail") or str(parsed.get("op")))
            continue
        # 选牌的结果已经由 `select_card` 表达，包装层的 Discard/Exhaust 要丢掉
        # （两种写法都要覆盖：分句写是"紧跟其后"，嵌套写是"把选牌包在括号里"）。
        if parsed.get("op") in ("discard", "exhaust"):
            start, end = call_span(body, match)
            wraps_select = any(start < s and e < end for s, e in select_spans)
            follows_select = bool(effects) and effects[-1].get("op") == "select_card" \
                and effects[-1].get("purpose") == parsed.get("op")
            if wraps_select or follows_select:
                continue
        effects.append(parsed)
        # ⭐ 链式 setter（``PowerCmd.Apply<X>(…).SetDamage(…)``）：把卡牌动态变量
        # 搬进**能力实例**。看 :func:`chained_setter` 的说明 —— 这一句以前是
        # 静默丢失的，属于"报告说干净、数值其实少一块"。
        if parsed.get("op") == "apply_power" and parsed.get("power"):
            setter, why = chained_setter(body, match, str(parsed["power"]),
                                         key_map, locals_)
            if setter is not None:
                effects.append(setter)
            elif why:
                unsupported.append(why)

    # 收尾：位置在最后一条命令之后的偷牌 / 造牌动作（例如整段只有它们）。
    flush_pending(len(body))
    # 收尾：把"运行期次数循环"与"升级版专属效果"两条理由并进 unsupported。
    # 放在这里（而不是命令循环里 continue）是为了**不吞掉**这条命令本来
    # 能报的更具体缺口 —— `Darkness` 的 `OrbCmd.Passive` 就是这样保住的。
    for marker in pending_reasons:
        if marker not in unsupported:
            unsupported.append(marker)
    # ⭐ 兜底：body 里确实有"生成的牌被逐张升级"，但**没有任何**造牌效果带上
    # ``upgrade`` 标志 —— 那说明造牌那一步本身就没抽出来（张数/来源认不出）。
    # 不报的话这条 `CardCmd.Upgrade` 已经被消费掉，卡就变成"少一档强度"
    # 却查不出缺口：实测 `StormOfSteel` 会只剩"弃掉手牌"。
    if upgraded_generated and not any(
            effect.get("upgrade") for effect in effects
            if effect.get("op") in ("add_card", "generate_card")):
        unsupported.append("生成的牌被逐张升级，但造牌效果抽不出（张数/来源认不出）")
    # ⭐ `RunState.CloneCard(<选牌结果>)` → 这次选牌的用途是"**复制**"。
    # `FromDeckGeneric` 的 prompt 是通用的（`base.SelectionScreenPrompt`），
    # 判不出用途；**只能从后续动作反推**（`DollysMirror.AfterObtained`：
    # 选一张 → `CloneCard` → `CardPileCmd.Add(card, Deck)`）。
    if re.search(r"\bCloneCard\s*\(", body):
        for index in list(unresolved_selections):
            effect = effects[index]
            if effect.get("op") == "select_card" and not effect.get("purpose"):
                effect["purpose"] = "duplicate"
                effect.pop("_span", None)
                unresolved_selections.remove(index)
    # ⭐ `.SetSelectedCard(<选牌结果>)`（`Nightmare`）→ 这次选牌的用途是
    # "**记住这张卡**"：既不移动也不复制，只是把克隆快照交给刚施加的那个能力实例
    # （下回合由 `NightmarePower.BeforeHandDraw` 复制 Amount 份入手）。
    if any(e.get("op") == "set_power_var" and e.get("from") == "remembered_card"
           for e in effects):
        for index in list(unresolved_selections):
            effect = effects[index]
            if effect.get("op") == "select_card" and not effect.get("purpose"):
                effect["purpose"] = "remember"
                effect.pop("_span", None)
                unresolved_selections.remove(index)
                break
    # 收尾：用途仍未定的选牌不能带着空用途进效果表（引擎会拒绝，但报告里
    # 会出现"未知选牌用途 ''"这种看不懂的东西）。改成可读的"用途不明"。
    for index in unresolved_selections:
        effect = effects[index] if index < len(effects) else None
        if effect is not None and effect.get("op") == "select_card" \
                and not effect.get("purpose"):
            effect["purpose"] = ""
            effect["_unresolved"] = True
    cleaned: list[dict] = []
    for effect in effects:
        effect.pop("_span", None)
        if effect.pop("_unresolved", False):
            choices += 1
            unsupported.append("CardSelectCmd(用途不明)")
            continue
        cleaned.append(effect)

    # ⭐ **无条件**的静默缺口扫描（见 :data:`SILENT_GAP_PATTERNS`）：
    # 不能只在"一条效果都没抽到"时才报 —— `UpMySleeve` 的降费就是在补上
    # Shiv 造牌之后变成静默丢失的。
    for marker in silent_gap_markers(body, consumed_spans):
        if marker not in unsupported:
            unsupported.append(marker)

    # ⭐ 最后一道防线：**一个效果都没抽出来、却也没报任何"未支持"** 时，
    # 说明这张卡的效果用的是我们没见过的写法（静态工厂 / 直接改状态 / 循环里包着）。
    # 不报的话它就会静默变成"打出去什么都不发生"的空卡 ——
    # 比标残缺危险得多（标残缺至少会被排除出训练集）。
    if not cleaned and not unsupported:
        tokens = unrecognized_calls(body)
        if tokens:
            unsupported.extend(f"未识别的效果写法：{token}" for token in tokens)
        elif allow_bookkeeping and is_bookkeeping_only(body):
            # 只做记账/显示（重置内部计数器、刷图标状态）→ **正确地什么都不做**。
            # 无头模拟器没有图标、也没有跨战斗的"上一回合"计数，
            # 这类如实归零，而不是混进"未识别"里虚报缺口。
            pass
        elif body_looks_substantial(body):
            unsupported.append("未识别的效果写法（函数体非平凡，效果藏在 switch/delegate 里）")
    return cleaned, unsupported, choices

#: 奖励类型 → 引擎里的 ``kind``。
REWARD_KINDS: dict[str, str] = {
    "RelicReward": "relic",
    "PotionReward": "potion",
    "CardReward": "card",
    "GoldReward": "gold",
}

#: `RewardsCmd.OfferCustom(owner, X)` 里 X 常常是个**私有方法**的名字
#: （`GenerateRewards()` / `GenerateXxxRewards()`）。只抽钩子体会一条奖励都看不到。
REWARD_METHOD_CALL = re.compile(r"\b(Generate\w*Rewards?)\s*\(\s*\)")


def extract_rewards(body: str, source: str = "",
                    method_lookup=None) -> dict | None:
    """解析"提供一组奖励"（``RewardsCmd.OfferCustom``）。

    奖励列表通常写在**另一个私有方法**里，所以要跟着调用走一层::

        public override async Task AfterObtained()
        {
            await CardPileCmd.AddCurseToDeck<CurseOfTheBell>(base.Owner);
            await RewardsCmd.OfferCustom(base.Owner, GenerateRewards());   // ← 跟过去
        }

        private List<Reward> GenerateRewards()
        {
            … new RelicReward(RelicRarity.Uncommon, base.Owner) …
        }

    真机的 ``TestMode.IsOn`` 分支是**测试专用**（固定奖励），必须跳过 ——
    它的分支里写的是 ``new RelicReward(ModelDb.Relic<Anchor>()…)`` 这种写死的怪东西。

    返回 ``{"groups": [{"kind":…, "count":…, "rarity":…}, …]}``；
    解析不出任何一组就返回 ``None``（调用方照实记缺口，不猜）。
    """
    from tools.extract_monsters import method_body

    called = REWARD_METHOD_CALL.search(body)
    target = None
    if called:
        # 优先从源码里按方法名取（与其它抽取路径一致）
        if source:
            target = method_body(source, called.group(1))
        if target is None and method_lookup:
            target = method_lookup.get(called.group(1))
    if target is None:
        target = body
    # 丢掉 TestMode 分支（把它变成永不成立的条件，后续扫描自然跳过）
    cleaned = re.sub(r"if\s*\(\s*TestMode\.IsOn\s*\)\s*\{", "if (false) {", target)

    groups: list[dict] = []
    for match in re.finditer(r"new\s+(\w+Reward)\s*\(([^)]*)\)", cleaned):
        kind = REWARD_KINDS.get(match.group(1))
        if kind is None:
            continue
        args = match.group(2)
        # `new RelicReward(ModelDb.Relic<Anchor>()…` 是 TestMode 残留 → 跳过
        if "ModelDb." in args:
            continue
        group: dict = {"kind": kind}
        rarity = re.search(r"RelicRarity\.(\w+)", args)
        if rarity:
            group["rarity"] = rarity.group(1).lower()
        # ⚠️ 奖励常常写在 `for (…; j < N; …) list.Add(new PotionReward(…))` 里，
        # 出现次数是 1 而**实际份数是 N**（Cauldron 是大锅：N=5 瓶药水）。
        # 只数"出现几次"会把 5 瓶记成 1 瓶。
        loop = _enclosing_loop_bound(cleaned, match.start())
        if loop:
            # 循环上界常是**局部变量**（`int intValue = base.DynamicVars["Potions"].IntValue;`），
            # 要跟着解析回 DynamicVars，否则份数无从取值。
            for name, expr in collect_locals(cleaned).items():
                if loop == name:
                    loop = expr
                    break
            group["count_var"] = loop
        groups.append(group)
    if not groups:
        return None
    # 只对"没在循环里"的重复做合并；循环里的份数由 `count_var` 表达
    summary: list[dict] = []
    for group in groups:
        if (summary and summary[-1] == group
                and "count_var" not in group):
            summary[-1] = {**group, "count": summary[-1].get("count", 1) + 1}
        else:
            summary.append({**group, "count": group.get("count", 1)})
    return {"groups": summary}


#: `for (…; <ident> < <上界>; …)` —— 取循环上界
LOOP_BOUND = re.compile(r"for\s*\([^;{}]*;\s*\w+\s*<\s*([\w\[\]\"\.]+)\s*;")


def _enclosing_loop_bound(text: str, position: int) -> str | None:
    """这个位置外层有没有 ``for`` 循环？有就返回它的上界表达式。"""
    best: str | None = None
    for match in LOOP_BOUND.finditer(text):
        if match.end() > position:
            continue
        # 粗略判断"这个 for 包住了 position"：中间没有出现同级的 `return`
        between = text[match.end():position]
        if between.count("{") > between.count("}"):
            best = match.group(1)
    return best


#: 手牌里被动生效的钩子 —— 不是"打出的效果"，要分开存。
TRIGGER_HOOKS = ("OnTurnEndInHand", "AfterCardDrawn", "AfterCardPlayed",                 "AfterCardEnteredCombat", "BeforeHandDraw")


# ==========================================================================
# C# 解析小工具
# ==========================================================================
def balanced(text: str, open_index: int) -> tuple[str, int]:
    """从 ``text[open_index]``（必须是 ``(``）起取配平的内容，返回 (内容, 右括号下标)。"""
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:index], index
    return text[open_index + 1:], len(text) - 1


def resolve_relic_id(expr: str, body: str,
                     locals_: dict[str, str]) -> str | None:
    """解析"给出的是哪个遗物"（``RelicCmd.Obtain(relic, owner)``）。

    * ``ModelDb.Relic<LavaRock>().ToMutable()`` → ``lava_rock``（明确）
    * ``RelicFactory.PullNextRelicFromFront(owner)`` → ``""``（**随机**：
      由遗物抓包按稀有度取，见 ``sts2_sim/relicbag.py``）
    """
    texts = [expr]
    token = expr.strip()
    if re.fullmatch(r"\w+", token) and token in locals_:
        texts.append(locals_[token])
    for text in texts:
        match = re.search(r"Relic\s*<\s*(\w+)\s*>", text)
        if match:
            return snake(match.group(1))
        if "RelicFactory.PullNextRelic" in text:
            return ""                       # 随机遗物（走抓包）
    return None


def relic_rarity_hint(expr: str, body: str,
                      locals_: dict[str, str]) -> str:
    """``PullNextRelicFromFront(owner, RelicRarity.Rare, …)`` → ``rare``。

    显式指定稀有度的重载要把稀有度带上：抓包按它选桶
    （``RelicGrabBag.PullFromFront(rarity, …)``），丢了就变成"随机稀有度"。
    """
    texts = [expr]
    token = expr.strip()
    if re.fullmatch(r"\w+", token) and token in locals_:
        texts.append(locals_[token])
    for text in texts:
        match = re.search(r"RelicRarity\.(\w+)", text)
        if match:
            return snake(match.group(1))
    return ""


def relic_failure_reason(expr: str, body: str, locals_: dict[str, str]) -> str:
    """遗物 id 解析不出时给出具体原因。"""
    texts = [expr]
    token = expr.strip()
    if re.fullmatch(r"\w+", token) and token in locals_:
        texts.append(locals_[token])
    joined = " ".join(texts)
    if "RelicFactory." in joined or "PullNextRelic" in joined:
        return "需要遗物抓包（RelicGrabBag）"
    return "遗物 id 解析不出"


def balanced_braces(text: str, open_index: int) -> tuple[str, int]:
    """从 ``text[open_index]``（必须是 ``{``）起取配平的方法体内容。

    ⚠️ 不要拿 :func:`balanced` 当方法体用：那个配平的是**圆括号**
    （用于取命令参数）。用途搞混过一次 —— 拿 ``balanced`` 取
    ``GenerateInitialOptions`` 的方法体，结果在方法体里**第一个右括号**
    处就截断了（``…GetPile(base.Owner)…``），整段选项一个都没抽到，
    而工具**不报任何错**，只是安静地少了一整类内容。
    """
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:index], index
    return text[open_index + 1:], len(text) - 1


def call_args(text: str, after: int) -> tuple[str, int, int]:
    """取 ``after`` 之后**第一个**调用的参数，返回 ``(参数, 右括号下标, 左括号下标)``。

    ⚠️ 不要传 ``match.end() - 1`` 给 :func:`balanced`：那是方法名的**最后一个
    字母**，不是 ``(``。传错时 ``text[open_index + 1:]`` 会从 ``(`` 开始切，
    于是返回的内容**带着括号自己**，``split_top`` 只能切出一整块 ——
    症状是 ``PowerCmd.Apply`` 的能力名恒为空、目标恒为 unknown、
    量恒为 0，而**不报任何错**（实测就是这么静默错了一整轮）。

    泛型调用 ``PowerCmd.Apply<VulnerablePower>(...)`` 里 ``match.end()`` 指向
    ``<``，所以必须自己找 ``(``。返回值里的左括号下标用来取泛型实参。
    """
    open_index = text.find("(", after)
    if open_index == -1:
        return "", len(text) - 1, -1
    args, end = balanced(text, open_index)
    return args, end, open_index


def split_top(text: str) -> list[str]:
    """按顶层逗号切参数（忽略括号与泛型里的逗号）。"""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "([{<":
            depth += 1
        elif char in ")]}>":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def method_body(source: str, name: str) -> str | None:
    """按大括号配平取方法体。"""
    match = re.search(rf"\b{name}\s*\([^)]*\)\s*\{{", source)
    if not match:
        return None
    start = match.end() - 1
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1:index]
    return None


def method_spans(source: str) -> list[tuple[str, int, int]]:
    """``[(方法名, 体起点, 体终点)]`` —— 源文件里所有带大括号方法体。

    事件抽取要用**位置**判断"这句 ``SetEventState`` 写在哪个方法里"
    （从而知道"选了这个选项之后去哪一页"），所以这里返回区间而不只是文本。
    ``override_method_bodies`` 也复用同一套正则，避免两处口径不一致。
    """
    spans: list[tuple[str, int, int]] = []
    for match in re.finditer(
            r"\b(?:public|protected|private|internal)\s+(?:override\s+)?"
            r"(?:async\s+)?[\w<>\[\]\.\?,\s]+?\s+(\w+)\s*\([^)]*\)\s*\{", source):
        name = match.group(1)
        start = match.end() - 1
        depth = 0
        for index in range(start, len(source)):
            char = source[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    spans.append((name, start + 1, index))
                    break
    return spans


def enclosing_method(spans: list[tuple[str, int, int]],
                     position: int) -> str | None:
    """``position`` 落在哪个方法里（取**最内层**的那个）。"""
    best: tuple[str, int] | None = None
    for name, start, end in spans:
        if start <= position < end and (best is None or start > best[1]):
            best = (name, start)
    return best[0] if best else None


def override_method_bodies(source: str) -> dict[str, str]:
    """``{方法名: 方法体}`` —— 卡牌类里**所有**带方法体的方法。

    为什么要全扫一遍：卡牌的机制不一定写在 ``OnPlay`` 里。真机有一批卡把效果放在
    **钩子**里（``Void.AfterCardDrawn`` 抽到就扣 1 点能量、``QuestCard.OnQuestComplete``），
    而早先只扫 ``OnPlay`` + ``TRIGGER_HOOKS`` 这 5 个钩子，**没被扫到的钩子连
    "未支持"都不会报** —— 卡于是静默变成"什么都不做"（实测 ``Void`` 就是这样
    混进了可训练集）。全扫 + 报未扫描钩子，把"我们没看"变成一条显式缺口。
    """
    found: dict[str, str] = {}
    for name, start, end in method_spans(source):
        found.setdefault(name, source[start:end])
    # 属性 getter / 表达式体成员不算方法
    for name in ("OnUpgrade",):
        found.pop(name, None)
    return found


def property_expr(source: str, name: str) -> str | None:
    """取 ``X => <表达式>;`` 的表达式部分。"""
    match = re.search(rf"\b{name}\s*=>\s*(.+?);", source, re.S)
    return match.group(1).strip() if match else None


def snake(text: str) -> str:
    """``StrikeIronclad`` → ``strike_ironclad``（与 codex 的 cid 对齐）。"""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()


def enum_tail(expr: str) -> str:
    """``CardType.Attack`` → ``attack``；``TargetType.AnyEnemy`` → ``any_enemy``。"""
    return snake(expr.strip().split(".")[-1])


def power_id_of(class_name: str) -> str:
    """``VulnerablePower`` → ``vulnerable``；``EnergyNextTurnPower`` → ``energy_next_turn``。"""
    name = class_name[:-5] if class_name.endswith("Power") else class_name
    return snake(name)


def orb_id_of(class_name: str) -> str:
    name = class_name[:-3] if class_name.endswith("Orb") else class_name
    return snake(name)


# ==========================================================================
# 动态变量
# ==========================================================================
#: **运行期公式**：``CalculatedVar.WithMultiplier((card, target) => …)`` 的 lambda 体
#: → ``(calc_kind, arg)``（`docs/12` §2.33）。
#:
#: 真机算式统一是 ``CalculationBase + ExtraDamage × multiplier(lambda)``
#: （``CalculatedVar.WithMultiplier`` 的 docstring 写得很清楚：
#: "This will be multiplied by the ``GetExtraVar`` value"，
#: 例子就是 ``PerfectedStrike`` 的"6 点基础 + 每张打击牌 2 点"）。
#:
#: ⚠️ **只认这张表里的形状**。认不出的（历史记录、Osty、球队列、集合求和…）
#: 照旧报"量由运行期公式决定"并把卡排除出训练集 —— 宁缺勿猜
#: （``docs/09`` 铁律：猜错的数值比缺失的数值危险得多）。
CALC_FORMULAS: tuple[tuple[str, str, str], ...] = (
    # (lambda 体正则, calc_kind, 取参数的分组名)
    (r"card\.Owner\.Creature\.Block", "own_block", ""),
    (r"target\?\.Block \?\? 0", "target_block", ""),
    (r"PileType\.Draw\.GetPile\(card\.Owner\)\.Cards\.Count", "draw_pile_count", ""),
    (r"PileType\.Exhaust\.GetPile\(card\.Owner\)\.Cards\.Count", "exhaust_count", ""),
    (r"target\?\.GetPowerAmount<(\w+)>\(\) \?\? 0", "target_power", "power"),
    (r"Math\.Max\(0, card\.Owner\.Creature\.GetPowerAmount<(\w+)>\(\)\)",
     "self_power", "power"),
    (r"card\.Owner\.PlayerCombatState\.AllCards\.Count\(\(CardModel c\) => "
     r"c\.Tags\.Contains\(CardTag\.(\w+)\)\)", "all_cards_tag", "tag"),
    (r"CombatManager\.Instance\.History\.CardPlaysFinished\.Count\(\)",
     "card_plays_total", ""),
    # `Synchronize`：`(from orb in OrbQueue.Orbs group orb by orb.Id).Count()`
    # —— **不同种类的球数**（不是球总数）。
    (r"\(from orb in card\.Owner\.PlayerCombatState\.OrbQueue\.Orbs group orb by "
     r"orb\.Id\)\.Count\(\)", "distinct_orb_types", ""),
)

#: **写在 OnPlay 里**（不是 ``WithMultiplier``）的运行期公式：``amount_raw`` → 形状。
#:
#: 这类卡的量是当场算出来的（`Hang` 的 ``int num = Math.Max(2, powerAmount)``），
#: 抽取器原先只能报"量由运行期公式决定"。参数来源记在第三列：
#:
#: * ``local_power`` —— 局部变量 ``powerAmount = cardPlay.Target.GetPowerAmount<X>()``
#:   给出的**能力 id**；正则捕获的是**下限**，引擎里编码成 ``"<pid>:<下限>"``。
#: * ``power`` —— 正则直接捕获 C# 类名（过一遍 ``power_id_of``）。
RAW_CALC_FORMULAS: tuple[tuple[str, str, str], ...] = (
    (r"Math\.Max\((\d+), powerAmount\)", "target_power_floor", "local_power"),
    (r"cardPlay\.Target\.GetPower<(\w+)>\(\)\?\.Amount \?\? 0",
     "target_power", "power"),
)


def _calc_lambda_body(args_body: str) -> str:
    """从 ``WithMultiplier(...)`` 的参数里取出 lambda 的**表达式体**（归一化空白）。

    ``(CardModel card, Creature? _) => card.Owner.Creature.Block`` → 右侧表达式；
    ``delegate(CardModel card, Creature? _) { … }``（块体）→ 返回 ``""``（不认）。

    ⚠️ 取出来之后必须用 ``fullmatch`` 匹配形状表：``KnifeTrap`` 的
    "消耗堆里**飞刀**的张数" 与 `AshenStrike` 的"消耗堆总张数"前半段一模一样，
    用 ``search`` 会把两张卡都算成 `exhaust_count` —— 而前者会多算牌
    （静默变强），正是最危险的那种错。
    """
    text = re.sub(r"\s+", " ", args_body).strip()
    arrow = text.find("=>")
    if arrow == -1:
        return ""
    body = text[arrow + 2:].strip()
    if body.startswith("{"):
        return ""
    return body


def calc_formula_of(source: str, var_name_: str) -> tuple[str, str]:
    """这个计算变量的 lambda 是哪种**认得出来的**形状？认不出返回 ``("", "")``。

    ⚠️ 正则要在**整份源码**上匹配，不能拿 ``property_expr`` 的子串位置去
    ``balanced()`` —— 下标不同源，切出来的是文件里另一段（实测 `Mimic`
    就这样被判成"认不出"，而 `BodySlam` 侥幸能过）。
    """
    match = re.search(
        r"new\s+Calculated(?:Damage|Block)?Var\s*\([^)]*\)\s*"
        r"(?:\.FromOsty\(\)\s*)?\.WithMultiplier\(", source)
    if not match:
        match = re.search(r"new\s+CalculatedVar\s*\(\"" + re.escape(var_name_) +
                          r"\"\)\s*\.WithMultiplier\(", source)
    if not match:
        return ("", "")
    args, _end = balanced(source, match.end() - 1)
    body = _calc_lambda_body(args)
    if not body:
        return ("", "")
    for pattern, kind, group in CALC_FORMULAS:
        found = re.fullmatch(pattern, body)
        if found:
            arg = found.group(1) if group else ""
            # ⚠️ 能力参数在源码里是**类名**（``VulnerablePower``），引擎的 id 是
            # ``vulnerable`` —— 不转换的话 ``target.power("VulnerablePower")``
            # 恒为 0层，公式静默算成基数（实测 `Bully` 打 6 而不是 15）。
            if group == "power":
                arg = power_id_of(arg)
            return (kind, arg)
    # ⭐ **lambda 体是一个方法调用**时，能力藏在那个方法里。
    # 出处：`SovereignBlade.CanonicalVars` 写的是
    # ``new CalculatedBlockVar(ValueProp.Move).WithMultiplier((card, _) => GetOwnerParryAmount(card))``，
    # 而 `GetOwnerParryAmount` 的实现是
    # ``card.Owner.Creature.GetPowerAmount<ParryPower>()``。
    # 不跟这一层，"格挡 = 招架层数"会被整条判成"量由运行期公式决定"（卡被排除）。
    helper = re.fullmatch(r"(\w+)\(card\)", body)
    if helper is not None:
        power = _power_from_helper(source, helper.group(1))
        if power:
            return ("self_power", power)
    return ("", "")


def _power_from_helper(source: str, helper_name: str) -> str:
    """``<helper>(card)`` 的实现里读了哪个能力？找不到返回空串。

    只认 ``GetPowerAmount<XPower>()`` 这一种读法 —— 认不出就返回空，
    由调用方照旧报"量由运行期公式决定"（宁缺勿猜）。
    """
    body = method_body(source, helper_name)
    if not body:
        return ""
    found = re.search(r"GetPowerAmount<(\w+)>\(\)", body)
    return power_id_of(found.group(1)) if found else ""


def var_name(var_type: str, type_param: str | None) -> str:
    """变量类型 → **规范名**。

    真机按名字索引：``DynamicVars.Damage``、``DynamicVars["ThornsPower"]``，
    而 ``PowerVar<ThornsPower>`` 的规范名取的是**能力名**（``Thorns``）。
    引用时两种写法都会出现，靠 :func:`var_keys` 全部登记。
    """
    if var_type == "PowerVar" and type_param:
        return type_param[:-5] if type_param.endswith("Power") else type_param
    return var_type[:-3] if var_type.endswith("Var") else var_type


def var_keys(variable: dict) -> set[str]:
    """一个变量能被引用的**所有**写法。

    ⚠️ 真机的命名不一致，必须同时登记两种：
    ``new PowerVar<ThornsPower>(4m)`` 之后，源码里
    ``DynamicVars["ThornsPower"]``（保留 ``Power``）与
    ``DynamicVars.Dexterity``（去掉 ``Power``）**都会出现** —— 实测只登记
    一种会让 99 条效果的量解析不出来。
    """
    keys = {variable["name"]}
    type_param = variable.get("type_param")
    if type_param and not variable.get("named"):
        keys.add(type_param)
        if type_param.endswith("Power"):
            keys.add(type_param[:-5])
    return keys


def build_key_map(variables: list[dict]) -> dict[str, str]:
    """``引用写法 → 变量规范名``。"""
    mapping: dict[str, str] = {}
    for variable in variables:
        for key in var_keys(variable):
            mapping.setdefault(key, variable["name"])
    return mapping


def parse_vars(source: str) -> list[dict]:
    """解析 ``CanonicalVars`` 里的动态变量（保持源码顺序）。

    同时抽出 ``ValueProp``：它决定这个量是否走"受力量/易伤影响"的管线。
    真机写 ``new DamageVar(20m, ValueProp.Unpowered)``（药水、充能球都是这样），
    **忽略它会让药水伤害被玩家的力量加成** —— 实测火焰药水打出 25 而不是 20。

    ⚠️ 变量有**两种**写法，第二种坑过一次：
    ``new CardsVar(2)``（位置式，名字从类型推）与
    ``new EnergyVar("GainEnergy", 1)``（**具名式**，名字和数值都在参数里）。
    只把 ``parts[0]`` 当数字，会让具名式的名字与数值**双双丢失** ——
    变量表里多出一条名叫 ``Energy``、值为 ``None`` 的记录，于是所有引用它的
    效果都解析不出（实测 ``Bread`` 就是这样被打包成"没有效果"的）。
    77 处具名变量分布在 66 张卡上，影响面不小。
    """
    expr = property_expr(source, "CanonicalVars")
    if not expr:
        return []
    variables: list[dict] = []
    for match in re.finditer(r"new\s+(\w+Var)\s*(?:<\s*(\w+)\s*>)?\s*\(", expr):
        var_type, type_param = match.group(1), match.group(2)
        args, _end = balanced(expr, match.end() - 1)
        parts = split_top(args)
        if not parts:
            continue
        named = re.fullmatch(r'"(\w+)"', parts[0].strip())
        if named:
            # 具名式：("名字", 值[, ValueProp.X])
            name = named.group(1)
            value_part = parts[1].strip() if len(parts) > 1 else ""
            rest = parts[2:]
        else:
            name = var_name(var_type, type_param)
            value_part = parts[0].strip()
            rest = parts[1:]
        number = re.fullmatch(r"(-?\d+(?:\.\d+)?)m?", value_part)
        prop = ""
        for argument in rest:
            found = re.search(r"ValueProp\.(\w+)", argument)
            if found:
                prop = found.group(1)
        variables.append({
            "name": name,
            "var_type": var_type,
            "type_param": type_param,
            # 具名变量的规范名是**显式名字**，此时不能再拿类型名当别名 ——
            # 否则 `DynamicVars.Energy` 会错误地指到 `GainEnergy` 上。
            "named": bool(named),
            "base": int(float(number.group(1))) if number else None,
            # 计算类变量（CalculationBaseVar / CalculatedVar）的真机取值依赖
            # 运行时状态，这里照实留 None 并由报告暴露，**不猜**。
            "raw": value_part,
            "value_prop": prop,
            "upgrade_delta": 0,
        })
    return variables


def parse_upgrade(source: str, variables: list[dict]) -> None:
    """解析 ``OnUpgrade``：``DynamicVars.X.UpgradeValueBy(n)`` 改动的是**基础值**。"""
    body = method_body(source, "OnUpgrade")
    if not body:
        return
    by_name = {v["name"]: v for v in variables}
    for match in re.finditer(
            r'DynamicVars(?:\[\s*"(\w+)"\s*\]|\.(\w+))\s*\.\s*UpgradeValueBy\(\s*'
            r'(-?\d+(?:\.\d+)?)m?\s*\)', body):
        name = match.group(1) or match.group(2)
        if name in by_name:
            by_name[name]["upgrade_delta"] += int(float(match.group(3)))


# ==========================================================================
# 效果抽取
# ==========================================================================
def resolve_amount(expr: str, key_map: dict[str, str],
                   locals_: dict[str, str] | None = None) -> tuple[int | None, str | None, str]:
    """把参数表达式解析成 ``(字面量, 变量名, 原始文本)``。

    支持：
      * 字面量 ``3m``、``-3m``
      * ``base.DynamicVars.Damage.BaseValue`` / ``DynamicVars["ThornsPower"].IntValue``
      * 整份变量 ``CreatureCmd.GainBlock(..., base.DynamicVars.Block, ...)``
      * ``-base.DynamicVars["StrengthPower"].BaseValue``（取负）
      * **局部变量**：``int num = base.DynamicVars.Vulnerable.IntValue;`` 之后的
        ``num`` —— 真机大量先取到局部变量再传参，不跟这一层会漏掉一批。
      * 变量名后的 ``.IntValue`` / ``.BaseValue`` 等取值器一律忽略（同一个量）。
    """
    text = expr.strip()
    sign = 1
    negative = False
    if text.startswith("-"):
        sign, negative = -1, True
        text = text[1:].strip()

    number = re.fullmatch(r"(-?\d+(?:\.\d+)?)m?", text)
    if number:
        return sign * int(float(number.group(1))), None, expr.strip()

    if locals_ and re.fullmatch(r"\w+", text) and text in locals_:
        return resolve_amount(locals_[text], key_map, None)

    # ⚠️ **局部变量的成员访问**：真机常写
    # ``var damage = base.DynamicVars.Damage;`` 之后用 ``damage.BaseValue``。
    # 只支持"整串是变量名"会漏掉这一大类写法（实测药水 `fire_potion` /
    # `explosive_ampoule` 的伤害量因此解析不出、整瓶药被标残缺）。
    if locals_:
        head, _, tail = text.partition(".")
        if head in locals_:
            return resolve_amount(locals_[head] + ("." + tail if tail else ""),
                                  key_map, None)

    match = re.search(r'DynamicVars(?:\[\s*"(\w+)"\s*\]|\.(\w+))', text)
    if match:
        name = match.group(1) or match.group(2)
        canonical = key_map.get(name)
        if canonical:
            # 取负的量（`-base.DynamicVars["StrengthPower"].BaseValue`）用
            # 负数变量名表达不了，退回"运行期取值"，由引擎显式拒绝而不是当 0。
            if negative:
                return None, None, expr.strip()
            return None, canonical, expr.strip()
    return None, None, expr.strip()


#: 这些写法说明数值**依赖运行期状态**，静态抽不出来。
#: 必须显式标记，否则会被当成 0 —— 那是"看起来正常但一定算错"的静默错误。
RUNTIME_MARKERS = ("Results", "PlayerCombatState", "Creature.Block",
                   "CalculatedVar", "HittableEnemies")


def needs_runtime(text: str) -> bool:
    return any(marker in text for marker in RUNTIME_MARKERS)


def collect_locals(body: str) -> dict[str, str]:
    """收集 ``类型 x = <表达式>;`` 形式的局部变量赋值。

    两类都要：小写基元（``int n = …``）与**大写类型**（``CardModel cardModel = …``）。
    后者是"加的卡是哪张"的唯一线索，漏了会让 ``add_card`` 解析不出卡 id。
    """
    found = {match.group(1): match.group(2).strip()
             for match in re.finditer(
                 r"\b(?:int|decimal|var|float)\??\s+(\w+)\s*=\s*([^;]+);", body)}
    # ⚠️ 类型的正则要允许**数组与泛型**：真机写 `CardModel[] array =
    # (await CardSelectCmd.FromHand(…)).ToArray();` —— 不允许 `[]` 就抓不到这个
    # 局部变量，`Glimmer` 那样的卡会被误判成"集合移动"而不是"选牌后移牌堆"。
    for match in re.finditer(
            r"\b([A-Z]\w*(?:<[^>]*>)?(?:\[\])?)\s+(\w+)\s*=\s*([^;]+);", body):
        found.setdefault(match.group(2), match.group(3).strip())
    return found


def call_span(body: str, match: re.Match) -> tuple[int, int]:
    """一条命令调用的 ``(起, 止)`` 下标（含参数括号）。"""
    _args, end, open_index = call_args(body, match.end())
    return match.start(), (end if open_index != -1 else match.end())


#: ``(await PowerCmd.Apply<X>(…)).SetDamage(…)`` —— **链式 setter**。
#:
#: ⚠️ 这一句以前被**静默丢掉**：命令扫描只认 ``\w+Cmd\.\w+``，``SetDamage`` 不匹配，
#: 于是卡在报告里是"干净"的，效果表里却少了"爆炸造成多少伤害"（``TheBomb``）
#: 或"下次清格挡时给多少格挡"（``ToricToughness``）。真机这两种写法都是把
#: **卡牌的动态变量搬进能力实例**（``SetDamage`` / ``SetBlock`` 改的是
#: 能力自己的 ``DynamicVars``）。现在识别成 ``set_power_var`` 算子；
#: 参数认不出时**显式**记 unsupported，不再静默放过。
#: ⚠️ 前导的 ``)`` 要允许**多个**：真机写的是
#: ``(await PowerCmd.Apply<X>(…)).SetDamage(…)`` —— ``Apply`` 自己的右括号之后
#: 还有一层"await 表达式的括号"，只允许一个 ``)`` 会**一个都匹配不到**。
CHAINED_SETTER = re.compile(r"^\s*\)*\s*\??\.\s*Set(\w+)\s*\(")


def chained_setter(body: str, match: re.Match, power: str,
                   key_map: dict[str, str],
                   locals_: dict[str, str]) -> tuple[dict | None, str | None]:
    """识别紧跟 ``PowerCmd.Apply<X>(…)`` 的 ``.SetXxx(值)``。

    返回 ``(效果, 未支持原因)``；两者都是 ``None`` = 后面没有 setter。

    见过的两种参数写法：

    * ``SetDamage(base.DynamicVars["BombDamage"].BaseValue)`` —— 卡牌动态变量；
    * ``SetBlock(blockAmount)``，而 ``blockAmount`` 是**上一条命令的返回值**
      （``decimal blockAmount = await CreatureCmd.GainBlock(…)``）——
      标记成 ``from="last_block_gain"``，由引擎取"最近一次**实际**获得的格挡"
      （不是卡面写的那个数：真机拿到的是 ``GainBlock`` 的返回值，已经过了
      敏捷/脆弱/倍率）。
    """
    _start, end = call_span(body, match)
    found = CHAINED_SETTER.match(body[end:end + 200])
    if not found:
        return None, None
    name = found.group(1)
    args, _close, _open = call_args(body, end + found.end() - 1)
    expr = (split_top(args) or [""])[0].strip()
    if name == "SelectedCard":
        # ⭐ ``Apply<X>(…).SetSelectedCard(<选牌结果>)``：把**选中的那张卡**存进
        # 刚施加的能力实例（`NightmarePower.SetSelectedCard` 内部是
        # ``CreateClone()`` + ``ClearAffliction``）。它不是数值变量，
        # 所以走"记住选中的卡"这条取值来源 —— 引擎在 `set_power_var` 里取
        # `state.remembered_card`。出处：`Nightmare.cs:57-77`、
        # `NightmarePower.cs:73-79`。
        # ⚠️ 少了它，`Nightmare` 会被记成"用途不明 + 量认不出"，整张卡排除出训练集
        # （实测就是这么回事）。
        return ({"op": "set_power_var", "power": power, "var": "selected_card",
                 "amount": 0, "from": "remembered_card",
                 "times": 1, "target": "self"}, None)
    # ⚠️ **先判"上一条命令的返回值"**，再谈数值解析：`blockAmount` 的局部变量定义里
    # 含有 `base.DynamicVars.Block`，直接走 `resolve_amount` 会解析成**卡面的 5 点**
    # —— 而真机拿到的是 ``GainBlock`` 的返回值（已经过了敏捷/脆弱/倍率）。
    # 这是"看起来能解析、其实换了语义"的那类错，必须挡在解析之前。
    source = locals_.get(expr, "")
    if "GainBlock" in source and "await" in source:
        return ({"op": "set_power_var", "power": power, "var": name,
                 "amount": 0, "from": "last_block_gain",
                 "times": 1, "target": "self"}, None)
    amount, var, raw = resolve_amount(expr, key_map, locals_)
    if amount is None and var is None:
        return None, f"PowerCmd.Apply<…>.Set{name}({expr}) 的量认不出"
    return ({"op": "set_power_var", "power": power, "var": name, "amount": amount,
             "amount_var": var, "amount_raw": raw,
             "times": 1, "target": "self"}, None)


def target_of(expr: str, locals_: dict[str, str] | None = None,
              depth: int = 0) -> str:
    """从参数表达式判目标：``cardPlay.Target`` = 敌人，``base.Owner.Creature`` = 自己。

    ⚠️ 目标经常是**局部变量**（``PowerCmd.Apply<PoisonPower>(ctx, hittableEnemy, …)``），
    只看实参文本会判成 ``unknown``，那些卡就只能被标记残缺。
    所以要**跟一层局部变量**看它的来源。
    """
    text = expr.strip()
    # ⚠️ ``base.Creature`` 是**怪物**写法（卡牌写 ``base.Owner.Creature``）：
    # 怪物的所有招式处理函数都用它指自己。不认它，"这只怪给自己上 buff"
    # 会全部落成 ``target: unknown``，于是整批招式被判残缺 —— 实测 60+ 处。
    if "base.Creature" in text and "Target" not in text:
        return "self"
    # ⚠️ ``HittableEnemies``（复数）是**集合**，和 ``cardPlay.Target``（单体）
    # 不是一回事。两者归进同一个桶会让"给**全体**敌人上易伤"退化成"只给第一个" ——
    # 实测 `BagOfMarbles`（弹珠袋）就是这样：日志照常打印，只有一个敌人中招。
    collection = any(token in text for token in (
        "HittableEnemies", "hittableEnemies", "GetOpponentsOf", "Opponents",
        "combatState.Enemies", "CombatState.Enemies", "Enemies.ToList"))
    # 从集合里**取单个**的写法不是全体（`.First()` / `.Take(n)` / 随机挑一个）
    single_out = any(token in text for token in (
        ".First(", ".FirstOrDefault(", ".Take(", ".ElementAt", ".Random",
        ".OrderBy", ".MinBy", ".MaxBy", "GetRandom"))
    if collection and not single_out:
        return "all_enemies"
    if "cardPlay.Target" in text or "hittableEnemy" in text:
        return "enemy"
    if "base.Owner.Osty" in text:
        return "pet"
    if "base.Owner" in text:
        return "self"
    if locals_ and depth < 3 and re.fullmatch(r"\w+", text) and text in locals_:
        return target_of(locals_[text], locals_, depth + 1)
    return "unknown"


def resolve_pile(args: str) -> tuple[str, str]:
    """解析 ``PileType.Hand`` / ``CardPilePosition.Top`` → ``(牌堆, 位置)``。

    ⚠️ 只在**顶层参数**里找，不能拿整串参数去正则搜：``CardPileCmd.Add(await
    CardSelectCmd.FromCombatPile(ctx, PileType.Discard.GetPile(...), …), PileType.Hand)``
    里**有两个** ``PileType`` —— 搜到的是嵌套选牌的**来源**牌堆（Discard），
    而真正的目标是 ``Hand``。实测 `Dredge` 因此被判成"选完放进弃牌堆"，
    正好相反（它是"从弃牌堆挑回手牌"）。

    牌堆**必须**抽对：真机"加一张 Soul 进**抽牌堆**"与"加进**手牌**"是不同的效果，
    一律塞弃牌堆是静默的行为错误。
    """
    pile = "discard"
    position = ""
    for argument in split_top(args):
        token = argument.strip()
        match = re.fullmatch(r"PileType\.(\w+)", token)
        if match:
            pile = match.group(1).lower()
            continue
        match = re.fullmatch(r"CardPilePosition\.(\w+)", token)
        if match:
            position = match.group(1).lower()
    return pile, position


#: `.Where(<谓词>)` 里认得的条件。**认不出就整条拒绝，不猜** ——
#: 猜错一个过滤条件会让"把所有 0 费牌拿回手"变成"把所有牌拿回手"，
#: 强度差出好几倍，而且看起来完全正常。
#:
#: ⚠️ 参数名不能写死成 `c`：真机既有 `c => c.Rarity == …` 也有
#: `Filter(CardModel card) { … card.EnergyCost … }`，用 `\w+\.` 匹配任意名字。
FILTER_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\w+\.Rarity\s*==\s*CardRarity\.(\w+)", "rarity"),
    (r"\w+\.Type\s*==\s*CardType\.(\w+)", "type"),
    (r"\w+\.Keywords\.Contains\(CardKeyword\.(\w+)\)", "has_keyword"),
    (r"\w+\.IsUpgradable", "upgradable"),
    (r"\w+\.IsTransformable", "transformable"),
    (r"\w+\.EnergyCost\.GetWithModifiers\([^)]*\)\s*==\s*(\d+)", "cost"),
    (r"!?\w+\.EnergyCost\.CostsX", "not_x_cost"),
    # `c.IsUpgraded`（**已升级**，`Reflections.TouchAMirror` 的降级候选）——
    # ⚠️ 与 `upgradable`（`IsUpgradable`，可升级）只差一个字母，别合并。
    (r"\w+\.IsUpgraded", "upgraded"),
)


def parse_card_filter(expr: str) -> dict[str, object] | None:
    """把 ``.Where(<谓词>)`` 的条件解析成声明式过滤条件。

    只认 ``FILTER_PATTERNS`` 里的写法，**任一子条件认不出就整条返回 None**：
    宁可这张卡标残缺，也不能用一个猜出来的过滤条件。
    """
    text = expr.strip()
    if text.endswith(")") and "(" in text:
        text = text[:text.rindex("(")]          # 去掉最外层的调用括号
    # ⚠️ **lambda 头不是过滤条件**：`(CardModel c) => c.IsUpgradable` 里的
    # `(CardModel c) =>` 必须剥掉，否则第一个子条件会带着它去匹配
    # （`Whetstone` 的 `c != null` 就是这样判不出"空值守卫"的）。
    text = re.sub(r"^\s*(?:\([^)]*\)|\w+)\s*=>\s*", "", text)
    conditions = [part.strip() for part in text.split("&&")]
    result: dict[str, object] = {}
    for condition in conditions:
        condition = condition.strip()
        negated = condition.startswith("!")
        body = condition[1:].strip() if negated else condition
        # ⚠️ **空值守卫不是过滤条件**：`c != null` / `c is not null` 只是防 NPE。
        # 把它当成一条"认不出的条件"会让整条谓词作废 ——
        # 实测 `Whetstone`（"随机升级 2 张**攻击**牌"）就因为 `c != null`
        # 与 `c.Type == CardType.Attack` 并列而整条被拒。
        if re.fullmatch(r"[\w\.\?]+(\[\d+\])?\s*(!=\s*null|is\s+not\s+null)", body):
            continue
        # `c?.IsUpgradable ?? false` ≡ `c.IsUpgradable`（可空调用 + 默认值）：
        # `SandCastle` / `FragrantMushroom` 写的就是这一种。
        body = re.sub(r"\?\.", ".", body)
        body = re.sub(r"\s*\?\?\s*(?:true|false)\s*$", "", body)
        matched = False
        for pattern, key in FILTER_PATTERNS:
            found = re.search(pattern, body)
            if not found:
                continue
            matched = True
            value: object = found.group(1) if found.groups() else True
            if key == "cost":
                value = int(str(value))
            # ⚠️ key 已经带 `not_` 前缀的（`not_x_cost`）**不要**再加一次 ——
            # 它的模式自己就能匹配 `!` 的有无（`!?\w+…`），所以无论
            # lambda 头剥没剥掉（`!` 在不在 body 里）都该落到同一个 key。
            name = (f"not_{key}" if negated and not key.startswith("not_")
                    else key)
            result[name] = value
            break
        if not matched:
            return None                          # 认不出的条件 → 整条作废
    return result


def filter_of_where(definition: str, body: str) -> dict[str, object] | None:
    """从"集合的构造表达式"里取出 ``.Where(...)`` 的过滤条件。

    支持两种写法：内联 lambda（``Where(c => …)``）与**方法引用**
    （``Where(Filter)``，条件在另一个方法体里）。
    """
    where = re.search(r"\.Where\s*\(", definition)
    if not where:
        return None
    predicate = balanced(definition, where.end() - 1)[0].strip()
    if re.fullmatch(r"\w+", predicate):
        # 方法引用：条件写在那个方法体里（真机 `AllForOne.Filter`）
        method = method_body(body, predicate)
        if method is None:
            return None
        predicate = method
    return parse_card_filter(predicate)


def foreach_source_pile(body: str, name: str) -> str | None:
    """``foreach (<T> <name> in PileType.X.GetPile(…).Cards…)`` → ``"x"``。

    真机有一批卡写成"逐张搬"，语义等于"**整堆**搬"：

        // Reboot.cs:24-27：手牌全部进抽牌堆（与 BottledPotential 同一件事）
        foreach (CardModel item in PileType.Hand.GetPile(base.Owner).Cards.ToList())
            await CardPileCmd.Add(item, PileType.Draw);

    只认集合表达式的话，这种写法会掉进 ``CardPileCmd.Add(集合，谓词认不出)``
    被整条拒掉 —— 而它一点都不含糊。
    """
    for loop in re.finditer(r"\bforeach\s*\(", body):
        args, _end = balanced(body, loop.end() - 1)
        head = re.match(rf"\s*\w+\s+{re.escape(name)}\s+in\s+(.+)$", args, re.S)
        if not head:
            continue
        found = re.search(r"PileType\.(\w+)\.GetPile", head.group(1))
        if found:
            return found.group(1).lower()
    return None


def foreach_over_selection(body: str, locals_: dict[str, str], name: str) -> bool:
    """``name`` 是"遍历**选牌结果**"的 foreach 变量吗？

    真机写法（``Pomander`` / ``YummyCookie``）::

        List<CardModel> list = (await CardSelectCmd.FromDeckForUpgrade(…)).ToList();
        foreach (CardModel item in list) CardCmd.Upgrade(item);

    它与"选牌结果的局部变量"是同一件事：升级已经由那条 ``select_card``
    的**用途**表达了，后面这条 ``CardCmd.Upgrade`` 不该再产出一条效果。
    不认它，`CardCmd.Upgrade(item)` 会变成一条孤立的 ``upgrade_card``
    （没有目标卡）—— 而引擎的 Run 层在执行它时只会打一句 ⚠️ 然后跳过。
    """
    for loop in re.finditer(r"\bforeach\s*\(", body):
        args, _end = balanced(body, loop.end() - 1)
        head = re.match(rf"\s*\w+\s+{re.escape(name)}\s+in\s+(.+)$", args, re.S)
        if not head:
            continue
        expr = head.group(1).strip()
        token = expr.split(".")[0].strip()
        source = expr
        if re.fullmatch(r"\w+", token):
            source += " " + locals_.get(token, "")
        if "CardSelectCmd" in source:
            return True
    return False


def extract_collection_move(locals_: dict[str, str], body: str,
                            args: str) -> dict | None:
    """``CardPileCmd.Add(<局部集合>, PileType.Y, [Pos])`` → ``move_all_matching``。

    集合的构造形态是真机代码，这里只做**结构化识别**：

        PileType.Discard.GetPile(base.Owner).Cards.Where(c => c.Cost == 0).ToList()
        PileType.Draw.GetPile(base.Owner).Cards.Where(c => c.Rarity == Rare)
            .TakeRandom(count, base.Owner.RunState.Rng.CombatCardSelection)
    """
    first = (split_top(args) or [""])[0].strip()
    definition = locals_.get(first)
    if definition is None:
        # ⭐ 实参**本身就是那个集合表达式**（不经局部变量）：
        #
        #     BottledPotential: await CardPileCmd.Add(PileType.Hand.GetPile(p).Cards, PileType.Draw);
        #
        # 以前只认"裸变量名"，这种写法整条掉进"谓词认不出"被拒 ——
        # 而它一点都不含糊：**整个手牌**移进抽牌堆。
        definition = first if "PileType." in first else None
    if definition is None or "PileType." not in definition:
        return None
    if ".Where(" in definition:
        conditions = filter_of_where(definition, body)
        if conditions is None:
            return None
    else:
        # ⚠️ **没有 `.Where(...)` = 移动全部**，不是"认不出"。
        # 把"无谓词"也判成缺口会把 `BottledPotential`（手牌全部进抽牌堆）
        # 这类效果整段丢掉。
        conditions = {}
    pile_match = re.search(r"PileType\.(\w+)\.GetPile", definition)
    if not pile_match:
        return None
    taken = re.search(r"TakeRandom\s*\(\s*([^,]+),", definition)
    return {
        "op": "move_all_matching",
        "from": pile_match.group(1).lower(),
        "filter": tuple(sorted(conditions.items())),
        "take_random": (taken.group(1).strip() if taken else ""),
    }


#: 认不出的"效果写法"必须报出来。⚠️ 这条防线很关键：
#: 真机有些卡的效果**不是** `*Cmd.*` 调用，而是
#:
#:   * **静态工厂方法**：``Shiv.CreateInHand(owner, state)``、
#:     ``PotionFactory.CreateRandomPotionInCombat(...)``
#:   * **直接改状态**：``card.EnergyCost.SetThisCombat(1, reduceOnly: true)``
#:   * 包在 ``for`` / ``switch`` 里的效果（``MadScience``）
#:
#: 这些写法在旧实现里**既没有效果、也没有"未支持"标记** ——
#: 于是卡牌静默变成"打出去什么都不发生"，而这正是最难发现的一类错误。
#: 只有**改状态的动词**才算"效果写法"。不这样收口的话，
#: 参数守卫（``ArgumentNullException.ThrowIfNull``）与条件读取
#: （``e.HappenedThisTurn`` / ``Card.Keywords.Contains``）都会被报成"未识别的效果"，
#: 于是诊断信息变得不可信 —— 而"结论对、理由错"的告警比没有告警更糟。
EFFECT_VERBS: tuple[str, ...] = (
    "Create", "Add", "Set", "Apply", "Gain", "Deal", "Play", "Draw", "Exhaust",
    "Discard", "Upgrade", "Transform", "Remove", "Reduce", "Modify", "Move",
    "Summon", "Channel", "Forge", "Procure", "Kill", "Heal", "Shuffle",
    "AutoPlay", "Enchant", "Duplicate", "Copy", "Lose", "Decrement", "Increment",
)
#: 看着像动词、其实是查询的（``GetPile`` / ``Contains`` / ``ToList`` …）。
QUERY_METHODS: frozenset[str] = frozenset({
    "Contains", "Count", "Any", "All", "Where", "Select", "First",
    "FirstOrDefault", "ToList", "ToArray", "GetPile", "Sum", "Min", "Max",
    "ThrowIfNull", "Wait", "IsAlive", "GetPower", "HasPower", "GetTeammatesOf",
    "HappenedThisTurn", "HappenedLastPlayerTurn", "IsUpgradable",
})


def body_looks_substantial(body: str) -> bool:
    """函数体"看着有东西"却什么都没抽出来 —— 也是危险信号。

    覆盖动词扫描抓不到的两类写法：

    * 效果包在 ``switch`` 的分支里（``MadScience``）
    * 效果包在 ``.Where(delegate(CardModel c) { … })`` 里（``HiddenGem``）

    正常"什么都不做"的卡（诅咒、quest 卡）函数体是空的，不会命中这里。
    """
    text = re.sub(r"//[^\n]*", "", body)
    if re.search(r"\bswitch\s*\(", text):
        return True
    if "delegate" in text or re.search(r"=>\s*\{", text):
        return True
    # 成员赋值（``card.Replay = 2``）——排除声明局部变量的 `int x = …`
    return bool(re.search(r"\b\w+\.\w+\s*=(?!=)", text))


#: **纯记账 / 纯显示**的成员，改了它们对机制没有影响。
#: 遗物里大量出现：``base.Status = RelicStatus.Normal``（图标状态）、
#: ``AnyAttacksPlayedLastTurn = false``（内部计数器清零）、``RefreshStatus()``。
#: 无头模拟器没有图标也没有"上一回合"的跨战斗计数，这些**正确地什么都不做**。
BOOKKEEPING_MEMBERS: tuple[str, ...] = (
    "Status", "RelicStatus", "RefreshStatus", "Flash", "InvokeDisplayAmountChanged",
    "ShouldFlashOnPlayer", "IsUsedUp", "DisplayAmount", "ShowCounter",
)

#: **纯表现层**的命令：调了它们对战斗数值没有任何影响（只创建/更新 VFX 节点、播 SFX）。
#:
#: ⚠️ 判据是**回源码读它的实现**，不是"名字看着像特效"：
#: `ForgeCmd.PlayCombatRoomForgeVfx`（`ForgeCmd.cs:106-146`）返回 ``void``，
#: 全程只碰 ``NCreature`` / ``NSovereignBladeVfx`` 这些 Godot 节点。
#: 列进这里的命令在"未扫描的钩子"扫描里会被剥掉 —— 否则
#: `SovereignBlade.AfterCardChangedPiles`（整个方法都是这段特效）会让
#: 这张卡被报成缺口、排除出训练集。
PRESENTATION_COMMANDS: tuple[tuple[str, str], ...] = (
    ("ForgeCmd", "PlayCombatRoomForgeVfx"),
)


def is_bookkeeping_only(body: str) -> bool:
    """这个函数体是不是**只做记账/显示**、没有任何机制效果？

    ⚠️ 这个判断必须保守：只要出现**任何一个**效果命令（``Cmd`` 类的调用、
    ``PowerCmd`` / ``CardPileCmd`` / ``CreatureCmd`` …），就返回 ``False``。
    它的用途是把 `AfterCombatEnd` 那类"重置计数器 + 刷新图标"如实归到
    "正确地什么都不做"，而不是混进"未识别的效果写法"里 ——
    后者会让人以为有 22 个遗物的效果没做，其实一个机制都不缺。
    """
    text = re.sub(r"//[^\n]*", "", body)
    # 任何命令类调用都算"有机制"（泛型实参由 COMMAND_CALL 自己容忍）
    if COMMAND_CALL.search(text):
        return False
    if re.search(r"\bHook\.\w+\s*\(", text):
        return False
    # 剩下的赋值语句里，除了记账成员以外还有别的成员 → 不算纯记账
    for match in re.finditer(r"\b(\w+)\.(\w+)\s*=(?!=)", text):
        if match.group(2) not in BOOKKEEPING_MEMBERS:
            return False
    # 裸标识符赋值（`AnyAttacksPlayedLastTurn = false`）＝ 内部计数器
    return True


def unrecognized_calls(body: str) -> list[str]:
    """找出函数体里"看着像效果、但我们没认出来"的调用。

    ⚠️ 这条防线很关键：真机有些卡的效果**不是** `*Cmd.*` 调用，而是

      * **静态工厂方法**：``Shiv.CreateInHand(owner, state)``、
        ``PotionFactory.CreateRandomPotionInCombat(...)``
      * **直接改状态**：``card.EnergyCost.SetThisCombat(1, reduceOnly: true)``
      * 包在 ``for`` / ``switch`` 里的效果（``MadScience``）

    这些在旧实现里**既没有效果、也没有"未支持"标记** —— 卡牌静默变成
    "打出去什么都不发生"，而这是最难发现的一类错误。
    """
    found: list[str] = []
    for match in re.finditer(r"\b(\w+(?:\.\w+)?)\.(\w+)\s*\(", body):
        receiver, method = match.group(1), match.group(2)
        if receiver.endswith("Cmd") or receiver == "Cmd":
            continue                      # 已识别的命令类
        if COSMETIC_RECEIVER.match(receiver):
            continue                      # 表现层容器 / 特效节点（`NFanOfKnivesVfx.Create`）
        if (receiver, method) in COSMETIC:
            continue
        if method in QUERY_METHODS:
            continue                      # 查询，不是效果
        if not method.startswith(EFFECT_VERBS):
            continue                      # 不是改状态的动词
        token = f"{receiver}.{method}"
        if token not in found:
            found.append(token)
    return found


#: X 费的**权威**声明在源码里：``protected override bool HasEnergyCostX => true``。
#: ⚠️ 不能靠 codex 的 ``is_x_cost``，也不看构造参数：
#: 普通 X 费卡写 ``base(0, …)``、``Cascade`` 写 ``base(-1, …)`` —— 看数字必然搞错一边。
X_COST_DECL = re.compile(r"HasEnergyCostX\s*=>\s*true")
#: 效果读 X 的入口（``CardModel.ResolveEnergyXValue``）。
#: 真机把它传给 ``WithHitCount`` / 层数 / 球数，所以这些量是**打出时才知道**的。
#: ⚠️ 两种 X 都要认：``ResolveEnergyXValue``（能量）与 ``ResolveStarXValue``（星费，
#: 摄政王的资源）—— 只认前者会让 `stardust` 被当成"次数认不出"而不是"X 次"。
X_VALUE_CALL = re.compile(r"Resolve(?:Energy|Star)XValue\s*\(\s*\)")


#: **静态造牌工厂**的方法名（``X.CreateInHand(…)`` / ``X.Create(…)``）。
#: 它们由**别的卡**调用，不是自己这张卡被"打出"或"触发"时的钩子，
#: 所以在"未扫描钩子"那一遍里必须排掉（详见 ``parse_card`` 里的说明）。
CARD_FACTORY_METHODS: frozenset[str] = frozenset(
    {"CreateInHand", "Create", "CreateClone"})


def silent_gap_markers(body: str, consumed: list[tuple[int, int]] | None = None
                       ) -> list[str]:
    """见 :data:`SILENT_GAP_PATTERNS` 的说明。返回去重后按声明顺序的理由。

    ``consumed`` 是**已经被别的识别器消费掉**的位置区间（``[起, 止)``）——
    例如 ``cardModel.SetToFreeThisTurn()`` 已经被 :func:`generated_card_effects`
    认成 ``generate_card`` 的 ``free`` 标记。落进这些区间的命中**不算缺口**：
    不然"认出来了"反而会被静默缺口扫描再报一次，行为是对的、报告却是错的。

    ⚠️ 去注释必须**等长替换**（原来直接 ``sub`` 成空串）：位置一变，
    ``consumed`` 里的区间就对不上了 —— 表现为"有时报、有时不报"，最难查的一类。
    """
    consumed = consumed or []
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), body)
    out: list[str] = []
    for pattern, reason in SILENT_GAP_PATTERNS:
        for hit in re.finditer(pattern, text):
            if any(start <= hit.start() and hit.end() <= stop
                   for start, stop in consumed):
                continue
            if reason not in out:
                out.append(reason)
            break
    return out


#: **静默缺口**：这些写法**确实改机制**，但引擎没建模；麻烦的是它们出现时
#: 往往已经有别的效果被抽出来了 —— 于是"一条效果都没抽到才报"的那个兜底扫描
#: （见 :func:`extract_effects` 收尾处）**根本不会触发**，卡在报告里是干净的。
#:
#: 实测（补上 ``Shiv.CreateInHand`` 造牌之后）：``UpMySleeve`` 的
#: 「每次打出费用 -1」（``base.EnergyCost.AddThisCombat(-1)``）就没有任何标记了。
#: 所以这几条要**无条件**报出来。
#:
#: 名单是**白名单**：只收"改状态"的成员调用，不收查询、不收集合操作
#: （``list.Add`` / ``AddChildSafely`` 这类表现层与容器操作照旧不报 ——
#: 理由与 :data:`EFFECT_VERBS` 的注释一致：结论对、理由错的告警比没有告警更糟）。
SILENT_GAP_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\.EnergyCost\.(?:Add|Set)This(?:Combat|Turn)",
     "战斗内改费用（EnergyCost.Add/SetThisCombat）引擎未实现"),
    (r"\.EnergyCost\.SetThisTurnOrUntilPlayed",
     "本回合内改费用（SetThisTurnOrUntilPlayed）引擎未实现"),
    # ⚠️ 这两条**都要在名单里**。``SetToFreeThisTurn`` 只在
    # `generated_card_effects` 认得出来的那条路径上被消费掉（由 `consumed` 排除），
    # **其余路径一律算缺口**：
    #
    #   `BulletTime`     foreach (手牌) if (!CostsX) card.SetToFreeThisTurn()   ← 手牌全体免费
    #   `MummifiedHand`  抽到能力牌时 cardModel?.SetToFreeThisTurn()            ← 遗物侧
    #   `LiquidMemories` 药水：选一张弃牌堆的牌 → 回手 + 免费                    ← 要走选牌
    #
    # 这条名单曾经被我按"已经实现了 SetToFreeThisTurn"整条删掉过一次 ——
    # 后果正是本项目最忌讳的那一类：**报告从"残缺"变成"干净"，而行为比真机弱**。
    (r"\.SetToFreeThisTurn",
     "本回合免费（SetToFreeThisTurn）用在生成牌之外的路径，引擎未实现"),
    (r"\.SetToFreeThisCombat",
     "本场免费（SetToFreeThisCombat）引擎未实现"),
    (r"\.CreateClone\s*\(",
     "克隆一张牌（CreateClone）引擎未实现"),
    (r"PotionFactory\.CreateRandomPotionInCombat",
     "随机药水（CreateRandomPotionInCombat）引擎未实现"),
)


def count_loop_spans(body: str, key_map: dict[str, str],
                     locals_: dict[str, str] | None = None
                     ) -> list[tuple[int, int, int | None, str | None, str]]:
    """找出"上界是一个**可解析的常量变量**的 ``for`` 循环"。

    返回 ``[(循环体起点, 终点, 字面量次数, 变量名, 循环变量名)]``。

    出处：真机大量卡写成"把效果重复 N 次"，N 来自 ``CanonicalVars``::

        IceLance:      for (int i = 0; i < base.DynamicVars.Repeat.IntValue; i++)
                           await OrbCmd.Channel<FrostOrb>(choiceContext, base.Owner);
        CloakAndDagger: for (int i = 0; i < base.DynamicVars.Cards.IntValue; i++)
                           await Shiv.CreateInHand(base.Owner, base.CombatState);

    抽取器原来只认"上界依赖 X"的循环（:func:`x_loop_spans`），
    这类**字面量/变量上界**的循环被整个忽略 —— 循环体里的命令只算**一次**，
    于是卡明显变弱（``IceLance`` 该引导 2 个冰球，引擎只引导 1 个）。
    与 X 循环一样，用**区间包含**判断某条命令是否落在循环里。
    """
    locals_ = locals_ or {}
    spans: list[tuple[int, int, int | None, str | None, str]] = []
    for match in re.finditer(r"\bfor\s*\(", body):
        args, end = balanced(body, match.end() - 1)
        if is_x_dependent_expr(args, locals_):
            continue                     # X 上界交给 `x_loop_spans`
        bound = loop_bound(args)
        if bound is None:
            continue
        amount, var, _raw = resolve_amount(bound, key_map, locals_)
        if amount is None and var is None:
            continue                     # 上界认不出 → 不当成"重复 N 次"
        brace = body.find("{", end)
        if brace == -1:
            continue
        # ⚠️ **必须用 `balanced_braces`**（配平花括号）。这里原来写的是
        # `balanced`（配平**圆括号**）—— 循环体在内部第一个 `)` 处就被截断，
        # 落在截断点之后的命令**一个都进不了循环**：
        #
        #   `FightThrough`  for (…) { await CreateCard<Wound>() ×2 }  → 只加 1 张 Wound
        #   `Shatter`       for (…) { await OrbCmd.EvokeNext(…) }      → 只激发 1 个球
        #   `BouncingFlask` for (…) { await PowerCmd.Apply<Poison>(…) }→ 次数丢成 1 次
        #
        # 这三张里前两张**已经在训练集里**，属于"报告干净、比真机弱"的一类。
        _inner, close = balanced_braces(body, brace)
        # 循环变量名（`for (int i = 0; …)`）—— `Quadcast` 的
        # `dequeue: i == Repeat - 1` 要靠它判定。
        # ⚠️ `args` 是**括号里面**的内容（不含 `for (`），所以正则从行首开始匹配。
        loop_var_match = re.match(r"\s*(?:int\s+)?(\w+)\s*=", args)
        loop_var = loop_var_match.group(1) if loop_var_match else ""
        spans.append((brace, close, amount, var, loop_var))
    return spans


def runtime_loop_spans(body: str, key_map: dict[str, str],
                       locals_: dict[str, str],
                       x_loops: list[tuple[int, int]],
                       count_loops: list[tuple[int, int, int | None,
                                                    str | None, str]]
                       ) -> list[tuple[int, int]]:
    """找出"**次数取决于运行期状态**"的 ``for`` 循环区间。

    这一类既不是 X 循环（:func:`x_loop_spans`），也不是"上界是可解析常量变量"的
    循环（:func:`count_loop_spans`）：上界是**球数 / 手牌数 / 牌堆数**这类运行期读数::

        Shatter:   int orbCount = …OrbQueue.Orbs.Count;
                   for (int i = 0; i < orbCount; i++) { EvokeNext(false); EvokeNext(); }
        EvilEye:   …（按牌数循环）

    ⚠️ 引擎**没有**"按集合长度重复"的算子。把循环体当成跑一次是**静默的机制替换**：
    `Shatter` 在球数 ≠ 1 时会激发错的次数，而报告里这张卡是"可执行"，
    没有任何标记。所以这类循环体里的命令要**如实报缺口**（整张卡标残缺、
    排除出训练集），而不是按"展开一次"糊过去。

    认不准时**宁可漏报也不误伤**：上界表达式里已经带 X（由 ``x_loops`` 覆盖）
    或已由 ``count_loops`` 认下的循环，这里一律跳过。
    """
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\bfor\s*\(", body):
        args, end = balanced(body, match.end() - 1)
        if is_x_dependent_expr(args, locals_):
            continue
        brace = body.find("{", end)
        if brace == -1:
            continue
        if any(start <= brace <= stop for start, stop in x_loops):
            continue
        if any(start <= brace <= stop for start, stop, *_rest in count_loops):
            continue
        bound = loop_bound(args)
        if bound is None:
            continue
        amount, var, _raw = resolve_amount(bound, key_map, locals_)
        if amount is not None or var is not None:
            continue
        _inner, close = balanced_braces(body, brace)
        spans.append((brace, close))
    return spans


def generated_upgrades(body: str, locals_: dict[str, str] | None = None
                       ) -> tuple[set[str], set[int]]:
    """``foreach`` 遍历**刚生成的牌**并逐张 ``CardCmd.Upgrade`` → 那些牌是升级版。

    真机只有两处这种写法（都是"造出来就是升级版"）::

        CunningPotion:    foreach (CardModel item in await Shiv.CreateInHand(player, Cards, …))
                              CardCmd.Upgrade(item);
        CosmicConcoction: foreach (CardModel item in distinctForCombat)
                          {   CardCmd.Upgrade(item);
                              await CardPileCmd.AddGeneratedCardToCombat(item, PileType.Hand, owner); }

    返回 ``(被升级的 foreach 变量名, 生成调用的起点位置)``。

    ⚠️ 不认这条的后果是**给出未升级的牌** —— 比真机弱一档，而报告完全正常
    （`cunning_potion` 的 3 张 Shiv、`cosmic_concoction` 的 3 张无色牌）。
    这类"静默变弱"正是本项目最忌讳的，所以要么把 ``upgrade`` 标志带上，
    要么整条拒绝。
    """
    names: set[str] = set()
    starts: set[int] = set()
    for loop in re.finditer(r"\bforeach\s*\(", body):
        args, end = balanced(body, loop.end() - 1)
        head = re.match(r"\s*\w+\s+(\w+)\s+in\s+", args, re.S)
        if not head:
            continue
        name = head.group(1)
        brace = body.find("{", end)
        if brace == -1:
            continue
        inner, _close = balanced_braces(body, brace)
        if not re.search(rf"CardCmd\s*\.\s*Upgrade\s*\(\s*{re.escape(name)}\b", inner):
            continue
        # ⚠️ **必须确认遍历的是"刚生成的牌"**，光看"循环变量被 Upgrade"不够：
        # `Apotheosis` 写的是 `foreach (card in AllCards) CardCmd.Upgrade(card)`
        # —— 那是"升级**牌组**里所有牌"，与造牌毫无关系。
        # 判据放宽的后果是这条 `CardCmd.Upgrade` 被当成"造牌自带的升级"消费掉，
        # 于是 `apotheosis` / `dirge` / `knife_trap` / `jackpot` / `drain_power` /
        # `storm_of_steel` 的升级效果**整段消失**（实测一次踩中 11 张）。
        expr_start = loop.end() + head.end()
        expr_text = body[expr_start:end]
        generated_here = list(re.finditer(
            r"\b\w+\.(?:CreateInHand|Create)\s*\("
            r"|CardFactory\.GetDistinctForCombat\s*\(",
            expr_text))
        if not generated_here:
            # ``in distinctForCombat`` / ``in enumerable`` —— 集合是上面某一句的
            # 工厂调用赋的局部变量。⚠️ 少了这一层跟随，`CosmicConcoction` 会被判成
            # "不是遍历生成的牌"，那条 `CardCmd.Upgrade(item)` 于是变成引擎没有的
            # `upgrade_card`（整瓶药被拒）—— 而它本该是"3 张已升级的无色牌"。
            token = expr_text.strip().split(".")[0].strip()
            definition = (locals_ or {}).get(token, "")
            if not definition or not re.search(
                    r"\b\w+\.(?:CreateInHand|Create)\s*\("
                    r"|CardFactory\.GetDistinctForCombat\s*\(", definition):
                continue
            names.add(name)
            # ⚠️ 位置要指向**赋值那一句里的生成调用**（下面静态工厂扫描是按
            # `factory.start()` 认的）：`HiddenDaggers` 的
            # `IEnumerable<CardModel> enumerable = await Shiv.CreateInHand(…)`
            # 就靠这一步才拿得到 `upgrade` 标志。
            assign = re.search(rf"\b{re.escape(token)}\s*=\s*(.+?);", body, re.S)
            if assign:
                for call in re.finditer(
                        r"\b\w+\.(?:CreateInHand|Create)\s*\("
                        r"|CardFactory\.GetDistinctForCombat\s*\(",
                        assign.group(1)):
                    starts.add(assign.start(1) + call.start())
            continue
        names.add(name)
        # ``in`` 之后的表达式就是"生成这批牌"的那次调用，落在 body 的这段区间里。
        for call in generated_here:
            starts.add(expr_start + call.start())
    return names, starts


def while_loop_spans(body: str) -> list[tuple[int, int]]:
    """``while (<条件>) { … }`` 的块区间 ``(体起点, 终点)``。

    用途：区分"**填满空槽**"（``while (player.HasOpenPotionSlots) { …采购… }``，
    `EntropicBrew`）与"来一瓶"（单次 ``PotionCmd.TryToProcure``，`Alchemize`）。
    两者差一个数量级，混了就静默变强。
    """
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\bwhile\s*\(", body):
        _args, end = balanced(body, match.end() - 1)
        brace = body.find("{", end)
        if brace == -1:
            continue
        _inner, close = balanced_braces(body, brace)
        spans.append((brace, close))
    return spans


#: 真机随机流名 → 引擎流名（``rng.STREAMS``）。
#: ⚠️ **必须照着抽**：换成别的流，分布一样但**序列**与真机分叉。
RNG_STREAMS: tuple[tuple[str, str], ...] = (
    ("Rng.Niche", "niche"),
    ("Rng.Shuffle", "shuffle"),
    ("Rng.CombatCardSelection", "combat_card_selection"),
    ("Rng.CombatCardGeneration", "combat_card_generation"),
    ("Rng.Transformations", "transformations"),
    ("Rng.Rewards", "rewards"),
    ("Rng.Shops", "shops"),
    ("Rng.UpFront", "up_front"),
)


def random_deck_picks(body: str, locals_: dict[str, str],
                      key_map: dict[str, str]
                      ) -> list[tuple[dict, str, int]]:
    """识别"**从牌组随机取 N 张不重复**，再对它们做一件事"。

    真机形态（`Whetstone` / `WarPaint` / `WarHammer` / `SandCastle` /
    `FragrantMushroom` 逐字相同）::

        IEnumerable<CardModel> enumerable = PileType.Deck.GetPile(base.Owner).Cards
            .Where((CardModel c) => c != null && c.Type == CardType.Attack && c.IsUpgradable)
            .ToList().StableShuffle(base.Owner.RunState.Rng.Niche)
            .Take(base.DynamicVars.Cards.IntValue);
        foreach (CardModel item in enumerable) { CardCmd.Upgrade(item); }

    返回 ``[(效果, 被消费的 foreach 变量名, 语句位置)]``。

    认不出的**一律不产出**（调用方照旧报缺口）—— 尤其是
    ``base.Rng``（**事件自己的流**）：引擎的 Run 层拿不到事件流的句柄，
    硬按 ``niche`` 近似会让同种子下的结果与真机分叉。
    """
    picks: list[tuple[dict, str, int]] = []
    for match in re.finditer(r"\.StableShuffle\s*\(", body):
        stmt_start = body.rfind(";", 0, match.start()) + 1
        stmt_end = body.find(";", match.end())
        if stmt_end == -1:
            continue
        statement = body[stmt_start:stmt_end]
        pile = re.search(r"PileType\.(\w+)\.GetPile", statement)
        taken = re.search(r"\.Take\s*\(([^)]*)\)", statement)
        if not pile or not taken:
            continue
        # 牌组（Run 层）与**抽牌堆**（战斗内）是两条不同的路：
        # `StoneCracker` 是"战斗开始时从抽牌堆随机取 Cards 张升级"。
        pile_name = pile.group(1).lower()
        if pile_name not in ("deck", "draw"):
            continue
        operation = ("random_deck_cards" if pile_name == "deck"
                     else "random_draw_cards")
        stream_args, _end = balanced(body, match.end() - 1)
        stream = ""
        for token, name in RNG_STREAMS:
            if token in stream_args:
                stream = name
                break
        if not stream and re.search(r"\bbase\.Rng\b", stream_args):
            # ⭐ **事件自己的流**（真机 `base.Rng` = `EventModel.Rng`）。
            # 引擎由调用方把它传下来（`events.apply_option` → `apply_run_effects(rng=run.rng)`），
            # 所以这里用特殊值 ``event`` 而不是某条命名流 —— 按 `niche` 近似
            # 会让同种子下的结果与真机分叉（分布一样，序列不同）。
            stream = "event"
        if not stream:
            continue
        amount, var, raw = resolve_amount(taken.group(1).strip(), key_map, locals_)
        if amount is None and var is None:
            continue
        conditions: dict[str, object] = {}
        where = re.search(r"\.Where\s*\(", statement)
        if where:
            parsed = parse_card_filter(balanced(statement, where.end() - 1)[0])
            if parsed is None:
                continue                   # 谓词认不出 → 不猜
            conditions = parsed
        assign = re.match(r"\s*[\w<>\[\],\s]*?\b(\w+)\s*=", statement)
        holder = assign.group(1) if assign else ""
        if not holder:
            continue
        loop_var, purpose = "", ""
        for loop in re.finditer(r"\bforeach\s*\(", body):
            loop_args, loop_end = balanced(body, loop.end() - 1)
            head = re.match(rf"\s*\w+\s+(\w+)\s+in\s+{re.escape(holder)}\s*$",
                            loop_args.strip())
            if not head:
                continue
            brace = body.find("{", loop_end)
            if brace == -1:
                continue
            inner, _close = balanced_braces(body, brace)
            for action, name in (("Upgrade", "upgrade"), ("Downgrade", "downgrade")):
                if re.search(rf"CardCmd\s*\.\s*{action}\s*\(\s*{re.escape(head.group(1))}\b",
                             inner):
                    loop_var, purpose = head.group(1), name
                    break
            if purpose:
                break
        if not purpose:
            # ⭐ **整批**升级（没有 `foreach`）：`CardCmd.Upgrade(cards, style)`
            # （`StoneCracker`：把那 N 张一次性全升了）。判据是"命令的参数
            # 就是那个结果变量"。
            tail = body[stmt_start:stmt_end + 240]
            if re.search(rf"CardCmd\s*\.\s*Upgrade\s*\(\s*{re.escape(holder)}\b", tail):
                purpose, loop_var = "upgrade", holder
        if not purpose:
            continue                       # 没有"对它们做什么" → 不认
        effect = {"op": operation, "amount": amount, "amount_var": var,
                  "power": None, "target": "self", "times": 1,
                  "amount_raw": raw, "purpose": purpose, "rng": stream,
                  "pick": "shuffle",
                  "filter": tuple(sorted(conditions.items()))}
        picks.append((effect, loop_var, stmt_start))
    return picks


def random_item_picks(body: str, locals_: dict[str, str],
                      key_map: dict[str, str]) -> list[tuple[dict, int, int]]:
    """识别 ``CardCmd.<动作>(<流>.NextItem(<集合>))`` —— **随机取一张**。

    真机（``EndlessConveyor.ObserveChef``）::

        IEnumerable<CardModel> enumerable = base.Owner.Deck.Cards.Where(c => c.IsUpgradable);
        CardModel[] array = enumerable.ToArray();
        if (array.Any()) { CardCmd.Upgrade(base.Rng.NextItem(array)); }

    返回 ``[(效果, 命令起点, 命令终点)]``：调用方要按**位置**把它插进效果表，
    并把那条 ``CardCmd.Upgrade`` **消费**掉（动作已由 `purpose` 表达）。

    认不出的一律不产出（尤其是集合的谓词解析不出时）—— 宁可报缺口。
    """
    picks: list[tuple[dict, int, int]] = []
    for match in re.finditer(
            r"CardCmd\s*\.\s*(Upgrade|Downgrade)\s*\(\s*([\w\.]+)\s*\.\s*NextItem\s*\(",
            body):
        purpose = {"Upgrade": "upgrade", "Downgrade": "downgrade"}[match.group(1)]
        stream_args = match.group(2)
        stream = ""
        for token, name in RNG_STREAMS:
            if token in stream_args:
                stream = name
                break
        if not stream and re.search(r"\bbase\.Rng\b", stream_args):
            stream = "event"
        if not stream:
            continue
        args, _end = balanced(body, match.end() - 1)
        token = args.strip().split(".")[0].strip()
        definition = locals_.get(token, "") if re.fullmatch(r"\w+", token) else ""
        where = re.search(r"\.Where\s*\(", definition)
        if where is None:
            # ⚠️ 集合常被**再转一手**：`EndlessConveyor` 写的是
            #     `enumerable2 = (enumerable as CardModel[]) ?? enumerable.ToArray();`
            # 谓词在上一层的 `enumerable` 里 —— 不再跟一层就整条认不出。
            for inner in dict.fromkeys(re.findall(r"\b(\w+)\b", definition)):
                candidate = locals_.get(inner, "")
                if ".Where(" in candidate:
                    definition = candidate
                    where = re.search(r"\.Where\s*\(", candidate)
                    break
        if where is None:
            continue                       # 没有谓词 = "任意一张" → 不认（可能不是牌组）
        conditions = parse_card_filter(balanced(definition, where.end() - 1)[0])
        if conditions is None:
            continue
        effect = {"op": "random_deck_cards", "amount": 1, "amount_var": None,
                  "power": None, "target": "self", "times": 1, "amount_raw": "1",
                  "purpose": purpose, "rng": stream, "pick": "item",
                  "filter": tuple(sorted(conditions.items()))}
        picks.append((effect, match.start(), match.end()))
    return picks


def random_remove_loop_picks(body: str, locals_: dict[str, str],
                             key_map: dict[str, str]
                             ) -> list[tuple[dict, int, int]]:
    """识别 ``for (i < N) { c = <流>.NextItem(集合); 集合.Remove(c); CardCmd.<动作>(c); }``。

    真机（``Reflections.TouchAMirror``）::

        List<CardModel> upgradableCards = Deck.Cards.Where(c => c.IsUpgradable).ToList();
        for (int i = 0; i < 4; i++) {
            if (upgradableCards.Count <= 0) break;
            CardModel card = base.Rng.NextItem(upgradableCards);
            upgradableCards.Remove(card);              // ← "取一张删一张" = 不重复
            CardCmd.Upgrade(card, CardPreviewStyle.MessyLayout);
        }

    与 :func:`random_item_picks` 的区别：那个是"命令直接套在 `NextItem` 外面"
    （只取一张），这个是**循环**取 N 张。取法都是 ``item``（`NextItem` + `Remove`）。

    返回 ``[(效果, 循环体起点, 循环体终点)]``：调用方按位置插入效果，
    并把循环体里那条动作命令**消费**掉。
    """
    picks: list[tuple[dict, int, int]] = []
    for loop in re.finditer(r"\bfor\s*\(", body):
        loop_args, loop_end = balanced(body, loop.end() - 1)
        bound = loop_bound(loop_args)
        if bound is None:
            continue
        amount, var, raw = resolve_amount(bound.strip(), key_map, locals_)
        if amount is None and var is None:
            continue
        brace = body.find("{", loop_end)
        if brace == -1:
            continue
        inner, close = balanced_braces(body, brace)
        taken = re.search(r"\b(\w+)\s*=\s*([\w\.]+)\s*\.\s*NextItem\s*\(\s*(\w+)\s*\)",
                           inner)
        if not taken:
            continue
        picked_var, stream_owner, collection = (taken.group(1), taken.group(2),
                                                taken.group(3))
        # ⭐ 必须"取一张、从候选里删一张" —— 没有 `Remove` 就是**有放回**抽样，
        # 会重复升级同一张，而日志完全正常。
        if not re.search(rf"{re.escape(collection)}\s*\.\s*Remove\s*\(\s*"
                         rf"{re.escape(picked_var)}\b", inner):
            continue
        purpose = ""
        for action, name in (("Upgrade", "upgrade"), ("Downgrade", "downgrade")):
            if re.search(rf"CardCmd\s*\.\s*{action}\s*\(\s*{re.escape(picked_var)}\b",
                         inner):
                purpose = name
                break
        if not purpose:
            continue
        stream = ""
        for token, name in RNG_STREAMS:
            if token in stream_owner:
                stream = name
                break
        if not stream and re.search(r"\bbase\.Rng\b", stream_owner):
            stream = "event"
        if not stream:
            continue
        definition = locals_.get(collection, "")
        where = re.search(r"\.Where\s*\(", definition)
        if where is None:
            continue
        conditions = parse_card_filter(balanced(definition, where.end() - 1)[0])
        if conditions is None:
            continue
        effect = {"op": "random_deck_cards", "amount": amount, "amount_var": var,
                  "power": None, "target": "self", "times": 1, "amount_raw": raw,
                  "purpose": purpose, "rng": stream, "pick": "item",
                  "filter": tuple(sorted(conditions.items()))}
        picks.append((effect, brace, close))
    return picks


def add_random_pool_picks(body: str, locals_: dict[str, str],
                          key_map: dict[str, str]
                          ) -> list[tuple[dict, int, int]]:
    """识别"从**卡池**随机取 N 张不重复 → 加入牌组"（`DistinguishedCape`）。

    真机（``DistinguishedCape.AfterObtained``）::

        List<CardModel> pool = (from c in ModelDb.CardPool<CurseCardPool>()
                                    .GetUnlockedCards(…)
                                where c.CanBeGeneratedByModifiers
                                orderby c.Id select c).ToList();
        for (int i = 0; i < base.DynamicVars["Curses"].IntValue; i++) {
            CardModel c = base.Owner.RunState.Rng.Niche.NextItem(pool);
            pool.Remove(c);
            curseResults.Add(await CardPileCmd.Add(
                base.Owner.RunState.CreateCard(c, base.Owner), PileType.Deck));
        }

    与 :func:`random_remove_loop_picks` 的区别：**源是卡池**（按 `CardPool<X>` 认），
    动作是"造一张加进牌组"。返回 ``[(效果, 循环体起点, 循环体终点)]``。
    """
    picks: list[tuple[dict, int, int]] = []
    for loop in re.finditer(r"\bfor\s*\(", body):
        loop_args, loop_end = balanced(body, loop.end() - 1)
        bound = loop_bound(loop_args)
        if bound is None:
            continue
        amount, var, raw = resolve_amount(bound.strip(), key_map, locals_)
        if amount is None and var is None:
            continue
        brace = body.find("{", loop_end)
        if brace == -1:
            continue
        inner, close = balanced_braces(body, brace)
        if "PileType.Deck" not in inner:
            continue
        taken = re.search(r"\b(\w+)\s*=\s*([\w\.]+)\s*\.\s*NextItem\s*\(\s*(\w+)\s*\)",
                           inner)
        if not taken:
            continue
        picked_var, stream_owner, collection = (taken.group(1), taken.group(2),
                                                taken.group(3))
        if not re.search(rf"{re.escape(collection)}\s*\.\s*Remove\s*\(\s*"
                         rf"{re.escape(picked_var)}\b", inner):
            continue                       # 没有 Remove = 有放回抽样 → 不认
        definition = locals_.get(collection, "")
        pool = re.search(r"CardPool\s*<\s*(\w+)\s*>", definition)
        if not pool:
            continue                       # 不是卡池（可能是牌组）→ 交给别的识别器
        conditions: list[tuple[str, object]] = [("pool", pool.group(1))]
        if "CanBeGeneratedByModifiers" in definition:
            conditions.append(("generatable", True))
        stream = ""
        for token, name in RNG_STREAMS:
            if token in stream_owner:
                stream = name
                break
        if not stream and re.search(r"\bbase\.Rng\b", stream_owner):
            stream = "event"
        if not stream:
            continue
        effect = {"op": "add_random_cards", "amount": amount, "amount_var": var,
                  "power": None, "target": "self", "times": 1, "amount_raw": raw,
                  "rng": stream, "pick": "item",
                  "filter": tuple(sorted(conditions))}
        picks.append((effect, brace, close))
    return picks


def clone_deck_picks(body: str) -> list[tuple[dict, int, int]]:
    """识别"**把整副牌组复制一份**"（``Reflections.Shatter``）。

    真机（``Reflections.cs``）::

        int originalDeckSize = base.Owner.Deck.Cards.Count;
        for (int i = 0; i < originalDeckSize; i++) {
            CardModel card = base.Owner.RunState.CloneCard(base.Owner.Deck.Cards[i]);
            await CardPileCmd.Add(card, PileType.Deck);
        }

    返回 ``[(效果, 循环体起点, 循环体终点)]``。判据要求**三件事同时在循环体里**：
    ``CloneCard``、目标牌堆是 ``PileType.Deck``、以及引用 ``Deck.Cards``
    （否则可能是"克隆别的集合"）—— 认不出就不产出。
    """
    picks: list[tuple[dict, int, int]] = []
    for loop in re.finditer(r"\bfor\s*\(", body):
        _args, loop_end = balanced(body, loop.end() - 1)
        brace = body.find("{", loop_end)
        if brace == -1:
            continue
        inner, close = balanced_braces(body, brace)
        if "CloneCard" not in inner or "PileType.Deck" not in inner:
            continue
        if not re.search(r"Deck\.Cards|PileType\.Deck\.GetPile", inner):
            continue
        effect = {"op": "clone_deck", "amount": 0, "amount_var": None,
                  "power": None, "target": "self", "times": 1, "amount_raw": ""}
        picks.append((effect, brace, close))
    return picks


def upgraded_only_spans(body: str) -> list[tuple[int, int]]:
    """找出 ``if (base.IsUpgraded) { … }`` 的块区间。

    真机有一批卡把"升级版才有的**结构性**差异"写在这样的分支里
    （``Spinner``+ 多引导一个玻璃球、``TrueGrit``+ 多一次选牌、
    ``Largesse``+ 把生成的牌升级…）。**18 处**在 ``OnPlay`` 里带命令。

    ⚠️ 引擎的升级只建模**数值**（``DynamicVars.UpgradeValueBy`` →
    ``vars[].upgrade_delta``），**没有**"升级版专属效果"这一层。于是这些命令
    会被无条件抽出来 —— 那意味着**基础版比真机强**：
    ``Spinner`` 不升级也白得一个玻璃球、``TrueGrit`` 不升级也多选一次。
    这类偏差不会报错、报告里也是"可执行"，所以必须在这里**如实报缺口**。

    两种写法都要认：

    * ``if (base.IsUpgraded) { … }`` —— 块内即升级版专属；
    * ``if (!base.IsUpgraded) return;`` —— **早退守卫**，从这一句起到方法末尾
      全是升级版专属（``StormOfSteel`` / ``HiddenDaggers`` 写的就是这个）。
      只认前一种会让这两张卡的"升级版才升级生成的牌"被当成无条件升级。
    """
    spans: list[tuple[int, int]] = []
    # 早退守卫：`if (!base.IsUpgraded) return;`（含 `{ return; }` 形态）
    for guard in re.finditer(
            r"if\s*\(\s*!\s*base\.IsUpgraded\s*\)\s*(?:\{\s*return\s*;\s*\}|return\s*;)",
            body):
        spans.append((guard.end(), len(body)))
    for match in re.finditer(r"if\s*\(\s*base\.IsUpgraded\s*\)", body):
        brace = body.find("{", match.end())
        if brace != -1 and not body[match.end():brace].strip(" \t\r\n)"):
            # `if (base.IsUpgraded)\n{ … }` 形态
            _inner, close = balanced_braces(body, brace)
            spans.append((brace, close))
            continue
        # 无花括号的单语句：`if (base.IsUpgraded) await …;`
        stop = body.find(";", match.end())
        if stop != -1:
            spans.append((match.end(), stop))
    return spans


def loop_bound(header: str) -> str | None:
    """从 ``for (int i = 0; i < <上界>; i++)`` 的头部取出上界表达式。"""
    parts = header.split(";")
    if len(parts) < 2:
        return None
    condition = parts[1]
    for operator in ("<=", "<"):
        if operator in condition:
            return condition.split(operator, 1)[1].strip()
    return None


def x_loop_spans(body: str, locals_: dict[str, str] | None = None) -> list[tuple[int, int]]:
    """找出"上界依赖 X 的 ``for`` 循环"的区间。

    ``Tempest`` 的 X 不在参数里，而在循环上界：
    ``int numOfOrbs = ResolveEnergyXValue(); for (int i = 0; i < numOfOrbs; i++) { … }``
    —— 落在这种循环里的效果，次数就是 X。只看参数会让它变成"引导 1 个球"。

    ⚠️ 上界常是**局部变量**（``numOfOrbs``），所以还要跟着变量回到 X 调用。
    """
    locals_ = locals_ or {}
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\bfor\s*\(", body):
        args, end = balanced(body, match.end() - 1)
        if not is_x_dependent_expr(args, locals_):
            continue
        brace = body.find("{", end)
        if brace == -1:
            continue
        # 同上：花括号体必须用 `balanced_braces`，用 `balanced` 会在第一个
        # 内层右括号处截断（X 循环体里几乎一定有函数调用带括号）。
        _inner, close = balanced_braces(body, brace)
        spans.append((brace, close))
    return spans


def is_x_dependent_expr(expr: str, locals_: dict[str, str]) -> bool:
    """表达式里（含它引用的局部变量）是否出现 X 调用。"""
    if X_VALUE_CALL.search(expr):
        return True
    for token in re.findall(r"\w+", expr):
        definition = locals_.get(token)
        if definition and X_VALUE_CALL.search(definition):
            return True
    return False


def is_x_dependent(expr: str, locals_: dict[str, str]) -> bool:
    """这个表达式是不是"打出时才知道的 X"（直接调或经局部变量转手）。"""
    if X_VALUE_CALL.search(expr):
        return True
    token = expr.strip()
    return bool(re.fullmatch(r"\w+", token) and token in locals_
                and X_VALUE_CALL.search(locals_[token]))


def resolve_times(expr: str, locals_: dict[str, str],
                  key_map: dict[str, str] | None = None
                  ) -> tuple[int | None, bool, str | None]:
    """解析 ``WithHitCount(<expr>)`` → ``(次数, 是否依赖 X, 次数变量)``。

    ⚠️ **认不出时必须返回 ``None``，不能默认成 1**：`whirlwind` 这类卡的次数是 X，
    默认成 1 会让"打 X 次"变成"打 1 次" —— 强度差 X 倍，而且看起来完全正常。
    实测 6 张 X 费卡就这样静默错了。

    ⭐ **常量变量**（``base.DynamicVars.Repeat.IntValue``）不是"认不出"：
    ``Repeat`` 是卡牌自己声明的动态变量，引擎会按具体数值展开
    （``_resolve_effects`` 的 ``times_var`` 分支）。实测 10 张卡的
    "打 Repeat 次"卡在这一条上被整批拒掉。

    真正的运行期表达式（``((!target.HasPower<Vulnerable>()) ? 1 : 2)``、
    ``(int)((CalculatedVar)…).Calculate(…)``）仍然返回"认不出" ——
    它们取决于打出时的目标/手牌状态，静态抽出来一定是错的。
    """
    text = expr.strip()
    if is_x_dependent(text, locals_):
        return None, True, None
    if re.fullmatch(r"\d+", text):
        return int(text), False, None
    if re.fullmatch(r"\w+", text) and text in locals_:
        return resolve_times(locals_[text].strip(), {}, key_map)
    # ⚠️ ``CalculatedVar`` 是**运行期公式**：``DynamicVars["CalculatedHits"]`` 只是它的壳，
    # 真正的值由 ``.Calculate(target)`` 按局面算出来（"本回合已打出的攻击牌数"之类）。
    # 绝不能落进下面那条"常量变量"分支 —— 那会把"打 N 次"变成"按 base 值打"，
    # 而卡在报告里是**干净**的。实测 `finisher` / `flechettes` / `radiate` 等
    # 9 张卡在这种写法上，判据必须先于 DynamicVars 分支。
    if "CalculatedVar" in text or ".Calculate(" in text:
        return None, False, None
    # 常量变量：与 `resolve_amount` 同一套映射（`DynamicVars.Repeat` / `["Repeat"]`）
    if key_map and "DynamicVars" in text and "?" not in text and ":" not in text:
        _amount, var, _raw = resolve_amount(text, key_map, None)
        if var:
            return None, False, var
    return None, False, None


def apply_value_props(effects: list[dict], variables: list[dict]) -> None:
    """把变量上声明的 ``ValueProp`` 传染给引用它的效果。

    'Unpowered' 表示这个量**不经过**力量 / 虚弱 / 易伤（伤害）或
    敏捷 / 脆弱（格挡）的修正，但**仍然可以被格挡**。
    药水与充能球都走这条 —— 不认它会让药水伤害被力量加成。
    """
    by_name = {v["name"]: v.get("value_prop") or "" for v in variables}
    for effect in effects:
        var = effect.get("amount_var")
        prop = by_name.get(var or "", "")
        if not prop:
            continue
        if "Unpowered" in prop:
            effect["unpowered"] = True
        if "Unblockable" in prop:
            effect["unblockable"] = True


def _effect(op: str, amount: int | None, var: str | None, power: str | None,
            target: str, times: int = 1, raw: str = "", card: str | None = None,
            pile: str = "", **extra: object) -> dict:
    effect: dict = {"op": op, "amount": amount, "amount_var": var, "power": power,
                    "target": target, "times": times, "amount_raw": raw}
    if card is not None:
        effect["card"] = card
    if pile:
        effect["pile"] = pile
    effect.update(extra)
    # 数值依赖运行期状态（造成的伤害合计、当前护甲、计算变量…）时显式标记。
    # 不标记就会被当成 0 —— "看起来正常但一定算错"，是本项目最忌讳的一类错误。
    if amount is None and var is None:
        effect["needs_runtime"] = needs_runtime(raw)
    return effect


def resolve_card_id(expr: str, body: str, locals_: dict[str, str],
                    self_card_id: str | None = None) -> str | None:
    """解析"造出/加入的是哪张卡"。

    真机把卡 id 放在泛型里：``base.CombatState.CreateCard<Debris>(base.Owner)``
    → ``debris``。但更多时候是先存进局部变量再传参
    （``CardPileCmd.AddGeneratedCardToCombat(cardModel, …)``），所以要**跟一层局部变量**。

    另外两种写法也要认：

    * ``CardModel card = CreateClone();``（``Anger``）—— **克隆正在打出的这张牌**，
      所以 id 就是这张卡自己（``self_card_id``）。不认它的话，``Anger`` 这张
      经典牌会永远停在"卡 id 解析不出"。
    * ``Soul.Create(...)`` 这类**卡牌类的静态工厂**（``GlimpseBeyond``）。

    解析不出**不能**当成"加了某张卡" —— 那样引擎会加一张空卡，
    而 lint 只会报一句"引用了不存在的卡 None"。
    """
    texts = [expr]
    token = expr.strip()
    # 集合的**下标**：``souls[0]``（``Severance`` 把 3 张 Soul 分别派到
    # 抽牌堆 / 弃牌堆 / 手牌）。元素是哪张卡要从**定义那个集合的工厂**读
    # （``List<Soul> souls = Soul.Create(…)``），不能靠下标猜。
    indexed = re.fullmatch(r"(\w+)\s*\[\s*\d+\s*\]", token)
    if indexed:
        definition = locals_.get(indexed.group(1), "")
        factory = re.search(r"\b(\w+)\.Create(?:InHand)?\s*\(", definition)
        if factory and snake(factory.group(1)) in known_card_ids():
            return snake(factory.group(1))
        return None
    if re.fullmatch(r"\w+", token) and token in locals_:
        texts.append(locals_[token])
    for text in texts:
        match = re.search(r"CreateCard\s*<\s*(\w+)\s*>", text)
        if match:
            return snake(_strip_card_suffix(match.group(1)))
        # 克隆自己：裸 `CreateClone()`（无接收者）＝ this.CreateClone()
        if self_card_id and re.search(r"(?<![\w.])CreateClone\s*\(\s*\)", text):
            return self_card_id
        # 卡牌类的静态工厂：`Soul.Create(...)`（必须确实是 Cards 目录里的类）
        for factory in re.finditer(r"\b(\w+)\.Create(?:InHand)?\s*\(", text):
            candidate = snake(factory.group(1))
            if candidate in known_card_ids():
                return candidate
    return None


def known_card_ids() -> frozenset[str]:
    """``Cards`` 源码目录里所有卡牌的 cid（``snake(类名)``）。

    用来判断 ``X.Create(...)`` 里的 ``X`` 是不是一张牌（``Soul.Create`` →
    ``soul``）。**按文件名取**，不猜语义：文件名就是卡牌类名，而 cid 由
    ``snake()`` 唯一决定（与 codex 对齐的规则见 ``parse_card``）。
    """
    global _KNOWN_CARD_IDS
    if _KNOWN_CARD_IDS is None:
        if SRC_ROOT.is_dir():
            _KNOWN_CARD_IDS = frozenset(snake(path.stem)
                                       for path in SRC_ROOT.glob("*.cs"))
        else:
            _KNOWN_CARD_IDS = frozenset()
    return _KNOWN_CARD_IDS


_KNOWN_CARD_IDS: frozenset[str] | None = None


def resolve_collection_card_id(expr: str, body: str,
                               locals_: dict[str, str]
                               ) -> tuple[str | None, str | None]:
    """复数版 ``AddGeneratedCardsToCombat(<集合>, …)``。

    返回 ``(卡 id, 张数的表达式)``。真机这个重载**不传张数** —— 张数就是集合长度，
    所以张数要么来自"填集合的那次调用"（``Soul.Create(player, DynamicVars.Cards…)``），
    要么来自包住它的 ``for`` 循环上界（``CrashLanding``）。

    ⚠️ 只认**一个**明确的卡 id 才返回；集合里混了多种卡（随机生成）就返回
    ``(None, None)``，让上层报"随机生成卡"而不是随便挑一张 ——
    猜错会变成"加错牌"，比不加更难发现。
    """
    name = expr.strip()
    # ⭐ 实参**直接就是工厂调用**：``AddGeneratedCardsToCombat(Soul.Create(owner, 3, state), …)``
    # （``GraveWarden``）。以前只认"裸变量名"，于是这种写法整条报成
    # "卡 id 解析不出" —— 而它其实一点都不含糊。
    direct = re.search(r"\b(\w+)\.Create(?:InHand)?\s*\(", name)
    if direct and snake(direct.group(1)) in known_card_ids():
        args, _end, _open = call_args(name, direct.end() - 1)
        pieces = split_top(args)
        # ``Create(owner, amount, state)`` 的张数在第 2 个实参；
        # ``CreateInHand(owner, state)`` 是"一张"（见 Shiv.cs:83 的重载）。
        if "CreateInHand" in direct.group(0):
            count = pieces[1] if len(pieces) >= 3 else "1"
        else:
            count = pieces[1] if len(pieces) >= 2 else "1"
        return snake(direct.group(1)), count
    if not re.fullmatch(r"\w+", name):
        return None, None
    windows: list[str] = []
    assign = re.search(rf"\b{name}\s*=\s*([^;]+);", body)
    if assign:
        windows.append(assign.group(1))
    addexpr = re.search(rf"\b{name}\.Add(?:Range)?\s*\(", body)
    if addexpr:
        windows.append(balanced(body, addexpr.end() - 1)[0])

    found: set[str] = set()
    amount_expr: str | None = None
    for text in windows:
        found.update(snake(m.group(1))
                     for m in re.finditer(r"CreateCard\s*<\s*(\w+)\s*>", text))
        found.update(snake(m.group(1)) for m in re.finditer(
            r"Add(?:GeneratedCard|ToCombat\w*)\s*<\s*(\w+)\s*>", text))
        for match in re.finditer(r"\b(\w+)\.Create(?:InHand)?\s*\(", text):
            candidate = snake(match.group(1))
            if candidate not in known_card_ids():
                continue
            found.add(candidate)
            args = balanced(text, match.end() - 1)[0]
            pieces = split_top(args)
            # `Soul.Create(player, 张数, state)` → 第 2 个参数是张数
            if len(pieces) >= 2 and amount_expr is None:
                amount_expr = pieces[1]
    found = {cid for cid in found if cid}
    if len(found) != 1:
        return None, None
    if amount_expr is None and addexpr is not None:
        # 集合是 `for (…) list.Add(…)` 填的 → 张数取**循环上界**
        amount_expr = _enclosing_loop_bound(body, addexpr.start())
    return next(iter(found)), amount_expr


#: ``CardFactory.GetDistinctForCombat(<player>, <查询>, <张数>, <rng>)`` ——
#: 真机的"**从卡池随机生成 N 张互不重复的牌**"。
#:
#: 引擎这边早就有一份逐条复刻的实现（``powers.generate_cards_to_hand``：
#: 互不重复 / 走 ``combat_card_generation`` 流 / 过准入门禁），能力侧
#: （``CreativeAiPower`` / ``HelloWorldPower`` / ``SpectrumShiftPower``）一直在用。
#: 缺的只是**卡牌与药水**这条入口的桥 —— 于是 20 多张卡与 5 瓶药水停在
#: "卡 id 解析不出 / 随机生成卡（卡池过滤 + combat_card_generation 随机流）"上。
GENERATE_CALL = re.compile(r"CardFactory\.GetDistinctForCombat\s*\(")


def _generate_call_texts(text: str) -> list[str]:
    """取出文本里每一个 ``GetDistinctForCombat(...)`` 调用的**完整文本**（含实参）。

    必须按括号配平取，不能用 ``[^)]*``：实参里就有带括号的调用
    （``…GetUnlockedCards(player.UnlockState, player.RunState.CardMultiplayerConstraint)``），
    截断之后连池子都认不出来。
    """
    out: list[str] = []
    for match in GENERATE_CALL.finditer(text):
        args, _end = balanced(text, match.end() - 1)
        out.append(f"CardFactory.GetDistinctForCombat({args})")
    return out


def generate_effect_of_call(call_text: str, key_map: dict[str, str],
                            locals_: dict[str, str]) -> dict | None:
    """把一个 ``GetDistinctForCombat`` 调用转成 ``generate_card`` 效果；认不出返回 ``None``。

    真机签名是 ``GetDistinctForCombat(Player player, IEnumerable<CardModel> cards,
    int count, Rng rng)``（``CardFactory.cs:119``），其中 ``cards`` 有两种固定来法：

    * ``player.Character.CardPool.GetUnlockedCards(...)`` —— **角色池**；
    * ``ModelDb.CardPool<ColorlessCardPool>().GetUnlockedCards(...)`` —— **无色池**。

    ``cards`` 后面常跟一段 LINQ：``from c in … where c.Type == CardType.Attack select c``。
    ⚠️ **有 ``where`` 却认不出条件时必须拒绝**，不能当成"任意牌"：
    `white_noise` 是"能力牌"、`infernal_blade` 是"攻击牌"，
    丢掉过滤条件等于把"随机一张攻击牌"变成"随机一张任意牌" —— 强度差一个数量级，
    而报告里这张卡看起来完全正常。
    """
    match = GENERATE_CALL.search(call_text)
    if match is None:
        return None
    args, _end = balanced(call_text, match.end() - 1)
    pieces = split_top(args)
    if len(pieces) < 3:
        return None
    # ⚠️ 签名是 ``(player, cards, count, rng)`` —— **池与 LINQ 都在第 2 个实参里**。
    # 读第 1 个（``player``）会永远找不到 ``CardPool``，于是整批卡被判成
    # "认不出池子"（第一版就踩了这个）。
    query, count_expr = pieces[1], pieces[2]
    if "ColorlessCardPool" in query:
        pool = "colorless"
    elif "CardPool" in query:
        pool = "character"
    else:
        # ``readOnlyList`` 这类局部变量（`BigHat` / `Crossbow`）—— 跟一层局部变量。
        name = query.strip()
        definition = locals_.get(name, "") if re.fullmatch(r"\w+", name) else ""
        if "ColorlessCardPool" in definition:
            pool = "colorless"
        elif "CardPool" in definition:
            pool = "character"
            query = definition      # 过滤条件也可能写在那一行里
        else:
            return None
    # 过滤条件写在 ``where … select`` 之间（LINQ 查询语法）。
    where_clause = ""
    clause = re.search(r"\bwhere\b(.*?)\bselect\b", query, re.S)
    if clause:
        where_clause = clause.group(1)
    card_type = ""
    found = re.search(r"\.Type\s*==\s*CardType\.(\w+)", where_clause)
    if found:
        card_type = found.group(1).lower()
    rarity = ""
    found = re.search(r"\.Rarity\s*==\s*CardRarity\.(\w+)", where_clause)
    if found:
        rarity = found.group(1).lower()
    # 排除自身：`where !(c is JackOfAllTrades)` —— 真机"别把这张牌自己生成出来"。
    excluded: list[str] = []
    for skip in re.finditer(r"!\s*\(\s*\w+\s+is\s+(\w+)\s*\)", where_clause):
        cid = snake(skip.group(1))
        if cid in known_card_ids():
            excluded.append(cid)
    # ⚠️ 把**已经认出来**的那几条条件从 where 里剥掉，剩下的必须为空 ——
    # 还剩条件就意味着"我们把一个不认识的过滤当成了没有过滤"，
    # 那时必须整条拒绝：`white_noise`（只给能力牌）丢掉过滤就变成
    # "随机一张任意牌"，强度差一个数量级，而报告里一切正常。
    #
    # 剥条件这一步**必须**做：第一版只剥 `!(c is X)`，于是
    # `white_noise` 的 `c.Type == CardType.Power` 留着 → 整批卡被误判成
    # "认不出过滤"（连已经写好的能力牌过滤都白费了）。
    remaining = where_clause
    for pattern in (r"(?:\w+\.)?Type\s*==\s*CardType\.\w+",
                    r"(?:\w+\.)?Rarity\s*==\s*CardRarity\.\w+",
                    r"!\s*\(\s*\w+\s+is\s+\w+\s*\)"):
        # ⚠️ 接收者要一起剥（`c.Type == …` 而不是只剥 `.Type == …`）：
        # 只剥后半截会在 remaining 里留下一个裸 `c`，于是**认得出来的条件
        # 反而把整张卡判成"认不出过滤"**（实测 `white_noise` 就这样被误杀）。
        for hit in re.finditer(pattern, remaining):
            remaining = remaining.replace(hit.group(0), "")
    if remaining.strip():
        return None
    count, var, raw = resolve_amount(count_expr, key_map, locals_)
    if count is None and var is None:
        return None                 # 张数认不出（`DynamicVars.Cards` 之外的写法）→ 不认
    effect = _effect("generate_card", count, var, None, "self", raw=raw)
    card_filter: list[tuple[str, str]] = [("pool", pool)]
    if card_type:
        card_filter.append(("card_type", card_type))
    if rarity:
        card_filter.append(("rarity", rarity))
    effect["filter"] = card_filter
    if excluded:
        effect["exclude"] = excluded
    # 真机的落点：`AddGeneratedCard(s)ToCombat(…, PileType.Hand, owner)`。
    effect["pile"] = "hand"
    return effect


def generated_card_effects(expr: str, body: str, locals_: dict[str, str],
                           key_map: dict[str, str], plural: bool
                           ) -> tuple[list[dict], list[tuple[int, int]]]:
    """解析 ``AddGeneratedCard(s)ToCombat`` 的实参 → ``generate_card`` 效果列表。

    返回 ``(效果列表, 已被消费的 SetToFreeThisTurn 位置区间)``。
    效果列表为空 = 认不出，调用方照旧报"随机生成卡"缺口。

    两种写法（都来自真实源码）::

        // 单数：WhiteNoise / InfernalBlade / Discovery …
        CardModel cardModel = CardFactory.GetDistinctForCombat(…, 1, rng).FirstOrDefault();
        if (cardModel != null) { cardModel.SetToFreeThisTurn(); await AddGeneratedCardToCombat(cardModel, …); }

        // 复数：OrobicAcid —— 三次 AddRange 各生成一张，再统一设免费
        List<CardModel> list = new List<CardModel>();
        list.AddRange(CardFactory.GetDistinctForCombat(… Attack …, 1, rng));
        list.AddRange(CardFactory.GetDistinctForCombat(… Skill  …, 1, rng));
        list.AddRange(CardFactory.GetDistinctForCombat(… Power  …, 1, rng));
        foreach (CardModel item in list) { item.SetToFreeThisTurn(); }
        await CardPileCmd.AddGeneratedCardsToCombat(list, PileType.Hand, base.Owner);

    ``SetToFreeThisTurn`` 的作用对象必须是**刚生成的那个局部变量**才算数 ——
    认不出作用对象就不置 ``free``（宁可少一个标记，也不能把别人的免费安到这张牌上）。
    """
    token = expr.strip()
    texts: list[str] = []
    free_vars: set[str] = set()
    if not re.fullmatch(r"\w+", token):
        return [], []
    definition = locals_.get(token, "")
    if not definition:
        # ⭐ 实参是 **foreach 的循环变量**（`JackOfAllTrades` / `BundleOfJoy`）::
        #
        #     IEnumerable<CardModel> distinctForCombat = CardFactory.GetDistinctForCombat(…);
        #     foreach (CardModel item in distinctForCombat.ToList())
        #         await CardPileCmd.AddGeneratedCardToCombat(item, PileType.Hand, base.Owner);
        #
        # 循环变量本身当然不在 ``locals_`` 里 —— 要顺着 ``in <集合>`` 找到
        # 定义那个集合的工厂调用。
        loop = re.search(
            rf"foreach\s*\(\s*\w+\s+{re.escape(token)}\s+in\s+([\w\.]+)", body)
        if loop:
            definition = locals_.get(loop.group(1).split(".")[0].strip(), "")
    if plural:
        # 集合的填充写在别处：找"以这个集合为接收者的 Add/AddRange"。
        for fill in re.finditer(rf"\b{re.escape(token)}\s*\.\s*Add(?:Range)?\s*\(", body):
            inner, _end = balanced(body, fill.end() - 1)
            texts.extend(_generate_call_texts(inner))
        # 免费：`foreach (CardModel item in list) { item.SetToFreeThisTurn(); }`
        for loop in re.finditer(
                rf"foreach\s*\(\s*\w+\s+(\w+)\s+in\s+{re.escape(token)}\s*\)", body):
            free_vars.add(loop.group(1))
    elif definition:
        texts.extend(_generate_call_texts(definition))
        free_vars.add(token)
    if not texts:
        return [], []
    effects: list[dict] = []
    for text in texts:
        effect = generate_effect_of_call(text, key_map, locals_)
        if effect is None:
            # 有一组认不出就整条不认：`OrobicAcid` 是三组，少一组就是"少给一张牌"。
            return [], []
        effects.append(effect)
    # 被消费掉的 `SetToFreeThisTurn` 位置（相对本 body）—— 交给
    # `silent_gap_markers` 排除，否则"认出来了"反而会被静默缺口扫描再报一次。
    consumed: list[tuple[int, int]] = []
    for call in re.finditer(r"\b(\w+)\s*\.\s*SetToFreeThisTurn\s*\(\s*\)", body):
        if call.group(1) not in free_vars:
            continue
        consumed.append((call.start(), call.end()))
        for effect in effects:
            effect["free"] = True
    return effects, consumed


def _strip_card_suffix(name: str) -> str:
    """``Debris`` 保持原样；``StrikeCard`` → ``Strike``。"""
    return name[:-4] if name.endswith("Card") and len(name) > 4 else name


def card_id_failure_reason(expr: str, body: str,
                           locals_: dict[str, str]) -> str:
    """解析不出卡 id 时给出**具体原因**（报告里要能直接看出该做什么）。

    三种情况要分开，因为要做的事完全不同：

    * ``CardFactory.GetDistinctForCombat(...)`` / ``GetRandomCard`` → **随机生成卡**：
      要的是"按卡池过滤 + 走 ``combat_card_generation`` 随机流"这套子系统；
    * ``CardSelectCmd.*`` → **玩家选牌**：要的是选择界面那套交互；
    * 其余 → 真的没认出来（这才是抽取器的锅）。
    """
    texts = [expr]
    token = expr.strip()
    if re.fullmatch(r"\w+", token) and token in locals_:
        texts.append(locals_[token])
    joined = " ".join(texts)
    if "CardFactory." in joined or "GetDistinctForCombat" in joined \
            or "GetRandomCard" in joined:
        return "随机生成卡（卡池过滤 + combat_card_generation 随机流）"
    if "CardSelectCmd." in joined:
        return "选牌结果（需要玩家选择）"
    return "卡 id 解析不出"


def effect_of_command(body: str, match: re.Match, key_map: dict[str, str],
                      locals_: dict[str, str],
                      x_loops: list[tuple[int, int]] | None = None,
                      damage_from_intent: bool = False,
                      self_card_id: str | None = None,
                      upgraded_generated: set[str] | frozenset[str] = frozenset(),
                      random_pick_vars: set[str] | frozenset[str] = frozenset(),
                      random_item_spans: tuple[tuple[int, int], ...] = (),
                      ) -> dict | None:
    """把一条命令调用转成一个效果 dict；认不出返回 ``None``。

    ``upgraded_generated`` 是"刚生成的牌被逐张升级"的循环变量名
    （见 :func:`generated_upgrades`）：``CardCmd.Upgrade(item)`` 落在里面时
    这条命令要**消费**掉（由造牌效果的 ``upgrade`` 标志表达），
    不能另产一条引擎没有的 ``upgrade_card``。
    """
    x_loops = x_loops or []

    def in_x_loop(position: int) -> bool:
        return any(start < position < end for start, end in x_loops)

    cmd_class, method = match.group(1), match.group(2)
    if (cmd_class, method) in COSMETIC:
        return {"op": "_cosmetic"}

    args, end, open_index = call_args(body, match.end())
    parts = split_top(args)
    # 链式构建器：命令之后到 `;` 之前还有 `.WithHitCount(2)` / `.Targeting*` 等
    tail_end = body.find(";", end)
    tail = body[end:tail_end if tail_end != -1 else len(body)]
    # head 必须**含泛型实参**：`PowerCmd.Apply<VulnerablePower>` 的能力名在
    # `<...>` 里，只用 `match.start():match.end()` 会拿不到。
    head = body[match.start():open_index] if open_index != -1 else match.group(0)

    def amount_at(index: int) -> tuple[int | None, str | None, str]:
        expr = parts[index] if len(parts) > index else ""
        return resolve_amount(expr, key_map, locals_)

    def x_amount_at(index: int) -> dict:
        """量是 X 时返回 ``{"amount_x": True}``，否则空 dict。"""
        expr = parts[index] if len(parts) > index else ""
        return {"amount_x": True} if is_x_dependent(expr, locals_) else {}

    if (cmd_class, method) == ("DamageCmd", "Attack"):
        if damage_from_intent:
            # **怪物招式**：数值由意图给出（``SingleAttackIntent(Damage)``）。
            # 意图数值另有一处来源（``GetValueIfAscension``，含进阶两档），
            # 这里再抽一份只会多出一个可能与进阶档位对不上的数，
            # 所以整条交给 `content._with_intent_damage`。
            return {"op": "_cosmetic"}
        amount, var, raw = amount_at(0)
        # ⚠️ 卡牌的伤害**没有**别处可依（怪物有意图，卡牌只有这一条）。
        # 数值抽不出时必须是**硬缺口**，绝不能悄悄产出一条 needs_runtime 的
        # "伤害量待定" 效果 —— 那会让 `BodySlam`（伤害 = 当前护甲）在引擎里
        # 变成"打 0 点"或"什么都不做"，而它是一张核心攻击卡。
        if amount is None and var is None and not is_x_dependent(raw, locals_):
            return {"op": "_damage_unresolved",
                    "detail": f"DamageCmd.Attack 伤害量抽不出：{raw[:40]}"}
        # ⚠️ 必须按**括号配平**取参数：`WithHitCount(ResolveEnergyXValue())`
        # 用 `[^)]+?` 会被内层括号截断成 `ResolveEnergyXValue(`，
        # 于是 X 认不出来、又被当成"次数认不出"。实测 3 张卡卡在这。
        hits_expr = None
        hits_call = re.search(r"WithHitCount\s*\(", tail)
        if hits_call:
            hits_expr = balanced(tail, hits_call.end() - 1)[0].strip()
        times, times_x, times_var = (resolve_times(hits_expr, locals_, key_map)
                                     if hits_expr else (1, False, None))
        if times is None and not times_x and not times_var:
            # 次数认不出（既不是字面量、也不是 X、也不是常量变量）→ 不能默认成 1
            return {"op": "_times_unresolved",
                    "detail": f"WithHitCount({hits_expr[:24]}) 认不出"}
        # ⚠️ 随机目标目前也按"打全体"处理，这是**近似**，必须能从数据里看出来
        op = "damage_all" if ("TargetingAllOpponents" in tail
                              or "TargetingRandomOpponents" in tail) else "damage"
        effect = _effect(op, amount, var, None, "enemy",
                         times if times is not None else 1, raw)
        if amount is None and var is None and is_x_dependent(raw, locals_):
            effect["amount_x"] = True
            effect.pop("needs_runtime", None)
        if times_x:
            effect["times_x"] = True
        if times_var:
            # 常量变量次数：交给引擎按卡牌动态变量的具体数值展开
            effect["times_var"] = times_var
        return effect

    if (cmd_class, method) == ("CreatureCmd", "GainBlock"):
        amount, var, raw = amount_at(1)
        return _effect("block", amount, var, None, "self", raw=raw)

    if (cmd_class, method) == ("PowerCmd", "Apply"):
        generic = re.search(r"Apply\s*<\s*(\w+)\s*>", head)
        power = power_id_of(generic.group(1)) if generic else ""
        # 参数：choiceContext, 目标, 层数, 施加者, 来源
        target = target_of(parts[1] if len(parts) > 1 else "", locals_)
        amount, var, raw = amount_at(2)
        effect = _effect("apply_power", amount, var, power, target, raw=raw)
        if amount is None and var is None and is_x_dependent(raw, locals_):
            effect["amount_x"] = True
            effect.pop("needs_runtime", None)
        return effect

    if (cmd_class, method) == ("CardPileCmd", "Draw"):
        amount, var, raw = amount_at(1)
        return _effect("draw", amount, var, None, "self", raw=raw)

    if (cmd_class, method) == ("PlayerCmd", "GainEnergy"):
        amount, var, raw = amount_at(0)
        return _effect("gain_energy", amount, var, None, "self", raw=raw)

    if (cmd_class, method) == ("PlayerCmd", "GainStars"):
        amount, var, raw = amount_at(0)
        return _effect("gain_stars", amount, var, None, "self", raw=raw)

    if (cmd_class, method) == ("CreatureCmd", "Damage"):
        target = target_of(parts[1] if len(parts) > 1 else "", locals_)
        amount, var, raw = amount_at(2)
        # ValueProp.Unblockable → 直接掉血（自伤类）；否则是普通伤害
        if "Unblockable" in args and target == "self":
            return _effect("lose_hp", amount, var, None, "self", raw=raw)
        return _effect("damage", amount, var, None, target, raw=raw)

    if (cmd_class, method) == ("CreatureCmd", "Heal"):
        amount, var, raw = amount_at(1)
        return _effect("heal", amount, var, None,
                       target_of(parts[0] if parts else "", locals_), raw=raw)

    if (cmd_class, method) in (("CardCmd", "Exhaust"), ("CardCmd", "ExhaustAndDraw")):
        return _effect("exhaust", 1, None, None, "self")

    if (cmd_class, method) in (("CardCmd", "Discard"), ("CardCmd", "DiscardAndDraw")):
        return _effect("discard", 1, None, None, "self")

    if (cmd_class, method) == ("OrbCmd", "Channel"):
        generic = re.search(r"Channel\s*<\s*(\w+)\s*>", head)
        # `Tempest` 是"引导 X 个球"：次数依赖 X，标记出来由引擎按 X 展开
        extra = {"times_x": True} if is_x_dependent(parts[1] if len(parts) > 1 else "",
                                                   locals_) else {}
        return _effect("channel", 1, None, None, "self",
                       orb=orb_id_of(generic.group(1)) if generic else "", **extra)

    if (cmd_class, method) == ("OrbCmd", "EvokeNext"):
        # ⚠️ 签名是 ``EvokeNext(choiceContext, player, dequeue)`` ——
        # 第 2 个参数是**玩家**、第 3 个是 ``dequeue``，**都不是次数**。
        # `DualCast` 是调用**两次**（第一次 ``dequeue: false`` = 激发但不移除）；
        # `MultiCast` 是在循环里调用 X 次。把参数当次数会让"激发两次"变成"激发一次"。
        dequeue_arg = args.split(",")[-1]
        dequeue = "false" not in dequeue_arg.lower()
        extra = {"times_x": True} if in_x_loop(match.start()) else {}
        return _effect("evoke_next", 1, None, None, "self",
                       dequeue=dequeue, **extra,
                       # ⭐ 把 ``dequeue`` 的**原始实参**带上：`Quadcast` 写的是
                       # ``EvokeNext(…, i == Repeat - 1)`` —— **只有最后一次移除球**。
                       # 光看"参数里没有 false"会得出"每次都移除"，那是错的
                       # （第一次就把球移走，后面三次激发空气）。
                       # 由 `extract_effects` 结合循环变量判定，见 `dequeue_tail`。
                       dequeue_raw=dequeue_arg.strip())

    if (cmd_class, method) == ("OrbCmd", "AddSlots"):
        # ⚠️ 签名是 ``AddSlots(Player owner, int count)`` —— **次数在第 2 个参数**。
        # 旧实现读的是第 1 个参数（``base.Owner``），于是永远抽不出量、退回 1。
        # 实测 ``Capacitor``（电容器）：源码 ``AddSlots(owner, Repeat=2)``、
        # 升级 ``Repeat.UpgradeValueBy(1)`` → 应当是 **2 槽 / 升级 3 槽**，
        # 引擎却恒为 1（基础值就错，升级完全无效）。
        amount, var, raw = amount_at(1)
        if amount is None and var is None:
            amount, var, raw = amount_at(0)      # 兜底：只有单个参数的写法
        # ⚠️ **量与变量只能给一个**：解析层是"`amount` 为 `None` 时才去查
        # `amount_var`"（见 `content._resolve_effects`）。这里若同时给出
        # `amount=1` 与 `amount_var="Repeat"`，那个 1 会**盖住变量** ——
        # 实测就是这样让 Capacitor 停在 1 的。
        # 两个都抽不出时**不再兜底成 1**：那会让"抽不出量"变成静默的 1，
        # 交给解析层判成缺口（整张卡标残缺）才是诚实的。
        return _effect("add_orb_slots", amount, var, None, "self", raw=raw)

    # 怪物招式里的"往玩家牌堆塞 N 张状态牌"：
    # `CardPileCmd.AddToCombatAndPreview<Dazed>(targets, PileType.Discard, 3, null)`
    # —— 与卡牌的 `AddGeneratedCard*` 是同一件事，只是泛型参数写在方法上。
    # ⚠️ 不认它，`Chomper.ScreechMove`（尖啸塞 3 张眩晕）这类招式会整条丢失。
    if cmd_class == "CardPileCmd" and method.startswith("AddToCombatAndPreview"):
        generic = re.search(r"AddToCombatAndPreview\s*<\s*(\w+)\s*>", head)
        card_id = snake(generic.group(1)) if generic else None
        if card_id is None:
            return {"op": "_card_unresolved", "detail": f"{method}(卡 id 解析不出)"}
        pile, position = resolve_pile(args)
        amount, var, raw = amount_at(2)
        return _effect("add_card", amount if amount is not None else 1, var, None,
                       "self", card=card_id, pile=pile, position=position, raw=raw)

    # ---- Run 层命令（遗物 / 事件用）----------------------------------------
    # ⚠️ 这些命令**不在战斗内**，但同属效果 DSL：遗物的 `AfterObtained`
    # 大量用它们（加最大生命 8 处、加诅咒 4 处、转化 5 处、附魔 10 处）；
    # 事件的选项体更是**全靠它们**（金币 25 处、给遗物 18 处、移牌 7 处）。
    # 不认它们，那 84 个遗物与 66 个事件的拾取/选项效果会被整批判成"未支持"。
    if (cmd_class, method) == ("PlayerCmd", "GainGold"):
        amount, var, raw = amount_at(0)
        return _effect("gain_gold", amount, var, None, "self", raw=raw)

    if (cmd_class, method) == ("PlayerCmd", "LoseGold"):
        # 签名 `LoseGold(decimal amount, Player player, GoldLossType type)`
        amount, var, raw = amount_at(0)
        return _effect("lose_gold", amount, var, None, "self", raw=raw)

    if (cmd_class, method) == ("CardPileCmd", "Shuffle"):
        # ``CardPileCmd.Shuffle(ctx, player)``（CardPileCmd.cs:1073-1097）：
        # 把**弃牌堆与抽牌堆合并**后整体 ``StableShuffle(Rng.Shuffle)`` 再放回。
        # 使用者：`BottledPotential`（药水）、`Reboot`（卡）。
        return _effect("shuffle", 1, None, None, "self")

    if (cmd_class, method) == ("CardPileCmd", "AutoPlayFromDrawPile"):
        # 签名 `(choiceContext, player, count, position, forceExhaust)`
        # （CardPileCmd.cs:1145）。使用者：`DistilledChaos`（药水）、
        # `Cascade` / `Havoc` / `IAmInvincible`（卡）、`MayhemPower`（能力）。
        amount, var, raw = amount_at(2)
        if amount is None and var is None:
            return {"op": "_amount_unresolved",
                    "detail": f"{method}(张数抽不出：{parts[2][:24] if len(parts) > 2 else ''})"}
        position = ""
        found = re.search(r"CardPilePosition\.(\w+)", args)
        if found:
            position = found.group(1).lower()
        if position not in ("top", "bottom"):
            # ``Random`` 走 `Rng.CombatCardSelection`，引擎没接 —— 认不出就拒绝，
            # 绝不近似成 Top（"随机打一张"变成"打顶张"是机制替换）。
            return {"op": "_card_unresolved",
                    "detail": f"{method}(取牌位置未实现：{position or '认不出'})"}
        force = re.search(r"forceExhaust\s*:\s*(true|false)", args)
        if force is None:
            # `forceExhaust` 决定这张牌打完是进消耗堆还是弃牌堆 —— 认不出就拒绝。
            return {"op": "_card_unresolved",
                    "detail": f"{method}(forceExhaust 认不出)"}
        effect = _effect("autoplay_from_draw", amount, var, None, "self", raw=raw)
        effect["position"] = position
        effect["force_exhaust"] = force.group(1) == "true"
        return effect

    if cmd_class == "PotionCmd" and method.startswith("TryToProcure"):
        # ⚠️ **两种药水池语义不同，不能混**：
        #   ``CreateRandomPotionOutOfCombat``（`EntropicBrew` / `DelicateFrond`）
        #       → 角色池 ∪ 共享池；
        #   ``CreateRandomPotionInCombat``（`Alchemize`）
        #       → 额外过滤 ``CanBeGeneratedInCombat``。
        # 后者引擎没实现 → 照旧报缺口，**不按前者近似**（池子不一样）。
        source = " ".join(parts)
        token = parts[0].strip() if parts else ""
        if re.fullmatch(r"\w+", token) and token in locals_:
            source += " " + locals_.get(token, "")
        if "CreateRandomPotionInCombat" in source:
            return {"op": "_card_unresolved",
                    "detail": f"{method}(战斗内药水池未实现)"}
        if "CreateRandomPotionOutOfCombat" not in source:
            return {"op": "_card_unresolved",
                    "detail": f"{method}(药水来源认不出)"}
        # ``while (player.HasOpenPotionSlots) { … }`` → **填满**（amount 记 0）；
        # 单次调用才是"来一瓶"（amount 记 1）。
        filled = any(start < match.start() < stop
                     for start, stop in while_loop_spans(body))
        effect = _effect("procure_random_potion", 0 if filled else 1, None, None,
                         "self")
        return effect

    if cmd_class == "RelicCmd" and method.startswith("Obtain"):
        generic = re.search(r"Obtain\s*<\s*(\w+)\s*>", head)
        if generic:
            # `RelicCmd.Obtain<LavaRock>(owner)` → 明确的遗物 id
            return _effect("obtain_relic", 1, None, None, "self",
                           relic=snake(generic.group(1)))
        # `RelicCmd.Obtain(relic, owner)`：遗物来自变量 —— 可能是
        # `RelicFactory.PullNextRelicFromFront(owner)`（**随机**，要走抓包），
        # 也可能是 `ModelDb.Relic<T>().ToMutable()`（明确）。两者要分开报。
        expr = parts[0] if parts else ""
        resolved = resolve_relic_id(expr, body, locals_)
        if resolved is None:
            return {"op": "_relic_unresolved",
                    "detail": f"Obtain({relic_failure_reason(expr, body, locals_)})"}
        # 随机遗物（``""``）可能带稀有度：`PullNextRelicFromFront(owner, RelicRarity.Rare, …)`
        # → 抓包按那个桶取（`RelicGrabBag.PullFromFront(rarity, …)`）。
        rarity = relic_rarity_hint(expr, body, locals_) if not resolved else ""
        return _effect("obtain_relic", 1, None, None, "self", relic=resolved,
                       filter=(("rarity", rarity),) if rarity else ())

    if cmd_class == "CardPileCmd" and method.startswith("RemoveFromDeck"):
        # 真机签名 `RemoveFromDeck(choiceContext, cardModel, player)`：目标是
        # **选牌结果**或一张具体牌。前者归 `select_card`（用途 removal），
        # 这里只管"具体某张牌"。
        target = parts[1] if len(parts) > 1 else ""
        card_id = resolve_card_id(target, body, locals_)
        return _effect("remove_card_from_deck", 1, None, None, "self",
                       card=card_id or "")

    if cmd_class == "CardCmd" and method.startswith("Downgrade"):
        # `Downgrade(card, style)`：把牌降回基础形态（**保留附魔与苦痛**）。
        # ⚠️ 与 `Upgrade` 同理：落在"随机取 N 张"区间里的那条已经被
        # `random_deck_cards(purpose="downgrade")` 表达了 → 消费掉。
        if any(start <= match.start() and match.end() <= stop
               for start, stop in random_item_spans):
            return {"op": "_cosmetic"}
        target = parts[0] if parts else ""
        card_id = resolve_card_id(target, body, locals_)
        return _effect("downgrade_card", 1, None, None, "self", card=card_id or "")

    if (cmd_class, method) == ("CreatureCmd", "GainMaxHp"):
        amount, var, raw = amount_at(1)
        return _effect("gain_max_hp", amount, var, None, "self", raw=raw)

    if (cmd_class, method) == ("CreatureCmd", "LoseMaxHp"):
        # 签名是 LoseMaxHp(choiceContext, creature, amount, isFromCard)，
        # 所以量在**第 3 个**参数（从 0 数）
        amount, var, raw = amount_at(2)
        return _effect("lose_max_hp", amount, var, None, "self", raw=raw)

    if cmd_class == "CardPileCmd" and method.startswith("AddCurseToDeck"):
        generic = re.search(r"AddCurseToDeck\s*<\s*(\w+)\s*>", head)
        card_id = snake(generic.group(1)) if generic else None
        if card_id is None:
            return {"op": "_card_unresolved", "detail": f"{method}(诅咒 id 解析不出)"}
        return _effect("add_card", 1, None, None, "self", card=card_id, pile="deck")

    if cmd_class == "CardCmd" and method.startswith("TransformTo"):
        generic = re.search(r"TransformTo\s*<\s*(\w+)\s*>", head)
        card_id = snake(generic.group(1)) if generic else None
        if card_id is None:
            return {"op": "_card_unresolved", "detail": f"{method}(卡 id 解析不出)"}
        return _effect("transform_card", 1, None, None, "self", card=card_id)

    if cmd_class == "CardCmd" and method.startswith("Enchant"):
        generic = re.search(r"Enchant\s*<\s*(\w+)\s*>", head)
        name = generic.group(1) if generic else None
        # `CardCmd.Enchant(enchantmentModel, cardModel, amount)`：附魔对象是变量
        if name is None and parts:
            found = re.search(r"Enchant\w*\s*<\s*(\w+)\s*>", args) or \
                re.search(r"(\w+)Model", parts[0])
            name = found.group(1) if found else None
        if name is None:
            return {"op": "_card_unresolved", "detail": f"{method}(附魔解析不出)"}
        amount, var, raw = amount_at(2 if "," in args else 1)
        return _effect("enchant_card", amount if amount is not None else 1,
                       var, None, "self", card=snake(name), raw=raw)

    if cmd_class == "CardPileCmd" and method.startswith("AddGeneratedCard"):
        first = parts[0] if parts else ""
        # ⚠️ 落点**按条件分叉**时必须拒绝：真机写
        #
        #     PileType newPileType = (i < 3) ? PileType.Draw : PileType.Discard;
        #     await CardPileCmd.AddGeneratedCardToCombat(card, newPileType, null, Random);
        #
        # （`TheInsatiable.LiquifyMove`：6 张 FranticEscape，**前 3 张进抽牌堆、
        #  后 3 张进弃牌堆**）。引擎的 `add_card` 只有一个 `pile`，
        # 取第一个会让 6 张**全进弃牌堆** —— 而报告里这条招式是"干净"的。
        # 认不出就整条拒绝，绝不近似。
        piles = set(re.findall(r"PileType\.(\w+)", args))
        if len(piles) > 1 and "?" in args:
            return {"op": "_card_unresolved",
                    "detail": f"{method}(落点按条件分叉，引擎未实现)"}
        # ⚠️ 落点更常见的写法是**先赋给一个局部变量**再传进来：
        #
        #     PileType newPileType = ((i < 3) ? PileType.Draw : PileType.Discard);
        #     await CardPileCmd.AddGeneratedCardToCombat(card, newPileType, null, Random);
        #
        # 只看实参里有没有 `?` 会**整条漏掉**（第一次修就是这么漏的）。
        # 这里跟一层局部变量，看到"三元 + 两个不同 PileType"就拒绝。
        arguments = split_top(args)
        if len(arguments) > 1:
            token = arguments[1].strip()
            if re.fullmatch(r"\w+", token):
                definition = locals_.get(token, "")
                branch = set(re.findall(r"PileType\.(\w+)", definition))
                if "?" in definition and len(branch) > 1:
                    return {"op": "_card_unresolved",
                            "detail": f"{method}(落点按条件分叉，引擎未实现)"}
        # ⚠️ 复数判定**不能**用 `endswith("s")`：方法名是
        # `AddGeneratedCardsToCombat`，结尾是 `t`（`AddGeneratedCardToCombat` 的
        # 复数是把 `Card` 变 `Cards`，不是加尾巴）。
        plural = method.startswith("AddGeneratedCards")
        card_id = resolve_card_id(first, body, locals_,
                                  self_card_id=self_card_id)
        amount_expr: str | None = None
        if plural:
            # 复数版：参数是**集合**，要跟到集合是怎么填的（`CrashLanding` 的
            # `list.Add(CreateCard<Debris>())`、`GlimpseBeyond` 的 `Soul.Create(...)`）。
            card_id, amount_expr = resolve_collection_card_id(first, body, locals_)
        if card_id is None:
            # ⭐ 先试"**随机生成 N 张**"这条（``CardFactory.GetDistinctForCombat``）：
            # 引擎已经有对应实现，缺的只是这座桥。认得出就不再是缺口。
            generated, consumed = generated_card_effects(
                first, body, locals_, key_map, plural)
            if generated:
                if first.strip() in upgraded_generated:
                    # `CosmicConcoction`：这批牌逐张 `CardCmd.Upgrade` → 升级版。
                    for item in generated:
                        item["upgrade"] = True
                return {"op": "_generated_cards", "effects": generated,
                        "consumed": consumed}
            # 解析不出加的是哪张卡 → 不能采纳（会变成"加一张空卡"）。
            # 原因要写具体：随机生成 / 选牌结果 / 真没认出来，三者要做的事不同。
            reason = card_id_failure_reason(first, body, locals_)
            return {"op": "_card_unresolved", "detail": f"{method}({reason})"}
        pile, position = resolve_pile(args)
        if not plural:
            return _effect("add_card", 1, None, None, "self", card=card_id,
                           pile=pile, position=position)
        # 复数版：张数来自集合长度。能解析成字面量/变量就用，否则**如实报缺口** ——
        # 猜成 1 会让 `CrashLanding`（应按手牌空位塞满）只塞一张，
        # 而模拟器里看不出任何异常。
        amount, var, raw = (resolve_amount(amount_expr, key_map, locals_)
                            if amount_expr else (None, None, ""))
        if amount is None and var is None:
            return {"op": "_amount_unresolved",
                    "detail": f"{method}(张数抽不出：{(amount_expr or '集合长度')[:32]})"}
        return _effect("add_card", amount, var, None, "self", card=card_id,
                       pile=pile, position=position, raw=raw)

    if (cmd_class, method) == ("CardCmd", "Upgrade"):
        # ``CardCmd.Upgrade(<刚生成的牌>)`` = "造出来就是升级版"，由造牌效果
        # 自己的 `upgrade` 标志表达（见 `generated_upgrades`）→ 这里**消费**掉，
        # 免得另产一条 `upgrade_card`（引擎没有这个算子，会把整张卡/整瓶药
        # 判成残缺 —— `cosmic_concoction` 之前就是这样被拒的）。
        # 同理：`foreach (item in <随机取来的 N 张>) CardCmd.Upgrade(item)` ——
        # 动作已由 `random_deck_cards` 的 `purpose` 表达（`docs/12` §2.19）。
        token = parts[0].strip() if parts else ""
        if token in upgraded_generated or token in random_pick_vars:
            return {"op": "_cosmetic"}
        # ⭐ `CardCmd.Upgrade(PileType.Hand.GetPile(owner).Cards, style)` ——
        # **整手牌**升级（`Bellows`："第 1 回合开始时把手牌全升级"）。
        # ⚠️ 判据必须同时是"手牌那一堆"**且没有 `.Where(...)`**：带谓词的是
        # "升级手牌里符合条件的那些"，语义不同。
        if re.search(r"PileType\.Hand\.GetPile\s*\(", args) and ".Where(" not in args:
            return _effect("upgrade_hand", 1, None, None, "self")
        # `CardCmd.Upgrade(base.Rng.NextItem(集合))` —— 整条命令已经被
        # `random_item_picks` 认成"随机取一张再升级"，这里消费掉它。
        if any(start <= match.start() and match.end() <= stop
               for start, stop in random_item_spans):
            return {"op": "_cosmetic"}
        return _effect("upgrade_card", 1, None, None, "self")

    return None


# ==========================================================================
# 单张卡
# ==========================================================================
def parse_card(path: Path, report: collections.Counter) -> dict | None:
    source = path.read_text(encoding="utf-8", errors="replace")
    class_name = path.stem

    ctor = re.search(r":\s*base\s*\(([^)]*)\)", source)
    if not ctor:
        report["card_without_ctor"] += 1
        return None
    ctor_args = split_top(ctor.group(1))
    if len(ctor_args) < 4:
        report["card_with_odd_ctor"] += 1
        return None

    cost_expr = ctor_args[0].strip()
    cost_number = re.fullmatch(r"(-?\d+)", cost_expr)
    cost = int(cost_number.group(1)) if cost_number else None
    # ⭐ X 费的**权威**声明在源码里，优先于构造参数与 codex。
    # 真机 `CardEnergyCost.Canonical = (!CostsX) ? canonicalCost : 0` ——
    # X 费卡的规范费用被强制为 0（所以 `Cascade` 构造参数写 -1 也只是原始输入）。
    is_x_cost = bool(X_COST_DECL.search(source))
    if is_x_cost:
        cost = 0
    if cost is None:
        # 真机有 X 费：构造参数不是字面量。照实留 None 并报告，不猜成 0。
        report[f"card_with_exotic_cost::{cost_expr[:24]}"] += 1

    # ⭐ **星费**（摄政王的资源；`CardModel.CanonicalStarCost`）。真机
    # `GetStarCostWithModifiers()`：X 星费 → 取**当前全部星**；否则用
    # ``Hook.ModifyStarCost`` 修正后的 `CurrentStarCost`。
    # ⚠️ 引擎以前完全不抽这个字段 —— 于是 23 张带星费的卡在引擎里**白嫖**
    # （`comet` 5 星照样 0 星打出），而门禁还把它们当成"源码版可用"。
    star = re.search(r"CanonicalStarCost\s*=>\s*(\d+)", source)
    star_cost = int(star.group(1)) if star else 0
    is_x_star_cost = bool(re.search(r"HasStarCostX\s*=>\s*true", source))
    if is_x_star_cost:
        star_cost = 0

    variables = parse_vars(source)
    parse_upgrade(source, variables)
    key_map = build_key_map(variables)
    for variable in variables:
        if variable["base"] is None:
            report[f"var_without_literal::{variable['name'][:24]}"] += 1

    on_play = method_body(source, "OnPlay")
    # `self_card_id` 让 `CreateClone()` 能解析成"克隆这张牌自己"（`Anger`）。
    self_card_id = snake(class_name)
    effects, unsupported, choices = extract_effects(
        on_play, key_map, self_card_id=self_card_id)
    apply_value_props(effects, variables)
    # ⭐ **运行期公式**（`docs/12` §2.33）：认得出来的形状在这里落地成
    # ``calc_kind`` + ``calc_base_var`` / ``calc_extra_var``（用**变量名**表达基准
    # 与增量，升级增量由 content 的 `_var_values` 统一算 —— 在这里烘死数字会
    # 让"升级版还是基础值"）。
    #
    # 认不出的照旧走下面的 `no_literal` → "量由运行期公式决定" → 整张卡排除
    # 出训练集。**宁缺勿猜**：多认一种形状就是一次"引擎能不能算对"的判断，
    # 判断错了就是静默数值错。
    has_calc_extra = any(v["name"] == "CalculationExtra" for v in variables)
    for effect in effects:
        if effect.get("amount") is not None:
            continue
        variable = effect.get("amount_var")
        if variable and variable.startswith("Calculated"):
            kind, arg = calc_formula_of(source, variable)
            if kind:
                effect["calc_kind"] = kind
                effect["calc_arg"] = arg
                effect["calc_base_var"] = "CalculationBase"
                effect["calc_extra_var"] = (
                    "CalculationExtra" if has_calc_extra else "ExtraDamage")
                continue
        # ⭐ **写在 OnPlay 里的**运行期公式（`Hang` / `Dominate`）：量不是
        # ``CalculatedVar``，而是当场算出来的表达式。认得出就落成 ``calc_kind``，
        # 基准 0 / 增量 1（整个量都来自那个表达式）。
        raw = str(effect.get("amount_raw") or "")
        for pattern, kind, source_of_arg in RAW_CALC_FORMULAS:
            found = re.fullmatch(pattern, raw)
            if not found:
                continue
            if source_of_arg == "local_power":
                # 局部变量给出的能力 id：`int powerAmount = …GetPowerAmount<HangPower>();`
                local = re.search(r"GetPowerAmount<(\w+)>\(\)", on_play or "")
                if not local:
                    break
                effect["calc_kind"] = kind
                effect["calc_arg"] = f"{power_id_of(local.group(1))}:{found.group(1)}"
            else:
                effect["calc_kind"] = kind
                effect["calc_arg"] = power_id_of(found.group(1))
            effect.setdefault("calc_base_var", "")
            effect.setdefault("calc_extra_var", "")
            break
    # ⚠️ **量的字面量缺失必须进 unsupported**。真机有一类"计算变量"
    # （`CalculatedDamage` / `CalculatedBlock` / `CalculatedDoom` …，公式写在
    # `CanonicalVars` 里，`BaseValue` 是运行期算的），抽取器抽得出"是伤害"、
    # 抽不出"打多少"。这类效果以前只在统计表里出现（`var_without_literal::*`），
    # **没进记录**；于是 `content` 那边整张卡不采纳、而文本版又恰好没效果时，
    # 卡片就静默变成"打出去什么都不发生"（实测 `BodySlam`/`GoldAxe`/`MindBlast`/
    # `Unleash`/`TimesUp`/`Squeeze`/`Protector` 共 7 张攻击卡）。
    # 宁可报残缺被排除出训练集，也不能留一张静默空卡。
    no_literal = {v["name"] for v in variables if v["base"] is None}
    for effect in effects:
        if (effect.get("amount") is not None or effect.get("amount_x")
                or effect.get("needs_runtime") or effect.get("calc_kind")):
            continue
        variable = effect.get("amount_var")
        if variable and variable not in no_literal:
            continue
        unsupported.append(
            f"量由运行期公式决定：{effect.get('op')} ← "
            f"{(effect.get('amount_raw') or '')[:32]}")
    for command in set(unsupported):
        report[f"unsupported_command::{command}"] += 1
    if choices:
        report["card_with_choice_command"] += 1
    if on_play is None:
        report["card_without_on_play"] += 1

    triggers: dict[str, list[dict]] = {}
    for hook in TRIGGER_HOOKS:
        body = method_body(source, hook)
        if not body:
            continue
        hook_effects, hook_unsupported, _choices = extract_effects(body, key_map)
        for command in set(hook_unsupported):
            report[f"unsupported_command::{command}"] += 1
        # ⚠️ 钩子里的未支持命令**必须进这张卡的 unsupported**。早先只加到统计表
        # （`report`）里，于是 `Void.AfterCardDrawn` 抽不出效果、也不报缺口，
        # 卡在引擎里就是"抽到什么都不发生" —— 而它本该扣 1 点能量。
        unsupported.extend(hook_unsupported)
        if hook_effects:
            triggers[snake(hook)] = hook_effects
        elif hook_unsupported:
            # 有钩子、有东西、但抽不出来 → 记一条**空触发**，让 content 那边
            # 能报"未实现的触发钩子"，而不是当作"这张卡没有触发"。
            triggers.setdefault(snake(hook), [])

    # 其余带命令的 override 方法（`OnQuestComplete` / `AfterCardPlayedLate` …）：
    # 我们**没有**把这些钩子接进引擎，所以它们的效果一定没落地 ——
    # 一律报"未扫描的钩子"，让这张卡被排除出训练集，而不是静默少一段机制。
    #
    # ⚠️ **静态造牌工厂不是钩子**，要排掉：``Soul.CreateInHand`` /
    # ``Shiv.CreateInHand`` 是**别的卡**调用它的入口（已经由
    # ``extract_effects`` 的静态工厂扫描建模），对**这张卡自己**来说
    # 不是"打出时会发生的事"。不排掉的话 ``Soul`` 这张牌会被报
    # "未扫描的钩子：CreateInHand" —— 理由错，而且把一张本来合格的牌挡在门外。
    scanned = set(TRIGGER_HOOKS) | {"OnPlay"} | CARD_FACTORY_METHODS
    for name, body in override_method_bodies(source).items():
        if name in scanned:
            continue
        # ⭐ 先剥掉**纯表现层**的命令（VFX / SFX）再判"有没有命令"：
        # `SovereignBlade.AfterCardChangedPiles` 整个方法只调用
        # `ForgeCmd.PlayCombatRoomForgeVfx`（创建/更新 Godot 节点 + 播 SFX，
        # `ForgeCmd.cs:106-146`，返回 void、不碰任何战斗状态）。
        # 不剥的话这张本来合格的牌会被"未扫描的钩子"挡在训练集门外 ——
        # 而它少的只是一段特效。
        stripped = body
        for cls, member in PRESENTATION_COMMANDS:
            stripped = stripped.replace(f"{cls}.{member}", "")
        if not COMMAND_CALL.search(stripped):
            continue                      # 只有表现/记账，没有命令
        extra_effects, extra_unsupported, _ = extract_effects(stripped, key_map)
        for command in set(extra_unsupported):
            report[f"unsupported_command::{command}"] += 1
        if extra_effects or extra_unsupported:
            unsupported.append(f"未扫描的钩子：{name}")
            report[f"unscanned_hook::{name}"] += 1

    keywords = sorted(set(re.findall(
        r"CardKeyword\.(\w+)", property_expr(source, "CanonicalKeywords") or "")))
    tags = sorted(set(re.findall(
        r"CardTag\.(\w+)", property_expr(source, "CanonicalTags") or "")))

    # ⭐ **多人专用**（``CardMultiplayerConstraint.MultiplayerOnly``）。
    # 枚举只有三个值（``CardMultiplayerConstraint.cs``）：``None`` /
    # ``MultiplayerOnly`` / ``SingleplayerOnly``；本地构建里
    # ``MultiplayerOnly`` 有 37 张、``SingleplayerOnly`` 为 0。
    #
    # ⚠️ 不抽这个字段，``eligibility`` 就无从判它 —— 实测有 16 张多人专用卡
    # 被当成"引擎可执行"进了**单人**训练池（`docs/12` §2.32），
    # 其中两张是 §2.28 那批刚放行的。
    multiplayer_only = bool(re.search(
        r"MultiplayerConstraint\s*=>\s*CardMultiplayerConstraint\.MultiplayerOnly",
        source))

    # ⭐ **能否在战斗内生成**（``CardModel.CanBeGeneratedInCombat``，默认 ``true``）。
    # 只有 19 张卡覆写成 ``false``（知识恶魔那类专属牌、部分状态牌）。
    #
    # 用途是**转化 / 随机生成**的候选池：``CardFactory.GetFilteredTransformationOptions``
    # 在 ``isInCombat`` 时按它过滤（``CardFactory.cs:201-204``）。
    # 不抽它，转化就会发出真机永远发不出的牌 —— 静默变强，而且卡面看着正常。
    can_be_generated_in_combat = not bool(re.search(
        r"CanBeGeneratedInCombat\s*=>\s*false", source))

    if not effects and not triggers:
        report["card_without_effects"] += 1

    return {
        "cid": snake(class_name),
        "class": class_name,
        "cost": cost,
        "is_x_cost": is_x_cost,
        # 星费（摄政王）：0 = 不要星；`is_x_star_cost` = 花掉**当前全部星**。
        "star_cost": star_cost,
        "is_x_star_cost": is_x_star_cost,
        "card_type": enum_tail(ctor_args[1]),
        "rarity": enum_tail(ctor_args[2]),
        "target": enum_tail(ctor_args[3]),
        "keywords": keywords,
        "tags": tags,
        "exhaust": "Exhaust" in keywords,
        # ⭐ 单人局里根本不会出现这张牌（``CardMultiplayerConstraint``）。
        "multiplayer_only": multiplayer_only,
        # ⭐ 战斗内能否被生成 / 转化到（默认 true；转化候选池按它过滤）。
        "can_be_generated_in_combat": can_be_generated_in_combat,
        "vars": variables,
        "effects": effects,
        "triggers": triggers,
        "unsupported": sorted(set(unsupported)),
        "choice_commands": choices,
    }


def summarize(cards: list[dict]) -> dict:
    total = len(cards)
    with_effects = [c for c in cards if c["effects"]]
    landing = [c for c in with_effects
               if all(e["op"] in SUPPORTED_OPS for e in c["effects"])]
    clean = [c for c in landing if not c["unsupported"] and not c["choice_commands"]]
    unresolved = [(c["cid"], e["op"], e["amount_raw"]) for c in cards
                  for e in c["effects"]
                  if e["amount"] is None and e["amount_var"] is None]
    runtime = sum(1 for c in cards for e in c["effects"] if e.get("needs_runtime"))
    return {
        "cards_total": total,
        "cards_with_effects": len(with_effects),
        "cards_landing": len(landing),
        "cards_clean": len(clean),
        "unresolved_amounts": len(unresolved),
        "runtime_amounts": runtime,
        "unresolved_examples": unresolved[:20],
        "keywords": collections.Counter(
            k for c in cards for k in c["keywords"]).most_common(),
    }


def cross_check(cards: list[dict], codex_path: Path) -> dict:
    """把源码抽出的数值与社区库的**描述文本**对拍。

    ⚠️ 这**不是**"谁对"的裁决，而是把分歧暴露出来。实测：
    263 条基础值与描述一致、其中 **226 条升级增量也对得上** —— 说明抽取逻辑
    是对的；剩下 18 张卡的数值两边**各自内部自洽**（例：``giant_rock``
    codex 16→20、源码 20→24），属于两个来源本身的数据分歧（版本差异或社区
    库笔误），只能靠人工定夺，不能静默取一边。

    按项目铁律"源码是机制的唯一真相"，引擎用源码值；分歧清单照样产出来，
    让这些卡不被当成"已验证"。
    """
    if not codex_path.exists():
        return {"checked": 0, "mismatches": [], "note": "没有社区库数据，跳过对拍"}
    codex = {c["cid"]: c for c in json.loads(codex_path.read_text(encoding="utf-8"))}

    def var_value(card: dict, name: str | None) -> int | None:
        variable = next((v for v in card["vars"] if v["name"] == name), None)
        return variable["base"] if variable else None

    checked = 0
    mismatches: list[dict] = []
    for card in cards:
        entry = codex.get(card["cid"])
        if not entry:
            continue
        text = re.sub(r"\[[^\]]*\]", "", entry.get("text") or "")
        numbers = {int(n) for n in re.findall(r"\b(\d+)\b", text)}
        if not numbers:
            continue
        for effect in card["effects"]:
            if effect["op"] not in ("damage", "damage_all", "block"):
                continue
            value = effect["amount"] if effect["amount"] is not None else var_value(
                card, effect["amount_var"])
            if value is None:
                continue
            checked += 1
            if value not in numbers:
                mismatches.append({
                    "cid": card["cid"], "op": effect["op"],
                    "source_value": value, "description_numbers": sorted(numbers),
                    "description": text[:120],
                })
            break               # 每张卡只对第一条伤害/格挡，避免多段效果重复计数
    return {"checked": checked, "mismatches": mismatches}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从反编译源码抽取卡牌效果")
    parser.add_argument("--src", default=str(SRC_ROOT))
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f"没有源码目录：{src}")
        return 3
    files = sorted(src.glob("*.cs"))
    report: collections.Counter = collections.Counter()
    cards = [c for c in (parse_card(p, report) for p in files) if c]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cards, ensure_ascii=False, indent=1), encoding="utf-8")

    stats = summarize(cards)
    total = stats["cards_total"]
    print(f"扫描 {len(files)} 个卡牌类，抽取 {total} 张")
    print(f"  有可执行效果          {stats['cards_with_effects']}")
    print(f"  效果全落在算子表内    {stats['cards_landing']}"
          f"  ({stats['cards_landing'] / max(1, total):.1%})")
    print(f"  且无未支持命令/选牌   {stats['cards_clean']}"
          f"  ({stats['cards_clean'] / max(1, total):.1%})")
    print(f"  量解析不出的效果条数  {stats['unresolved_amounts']}"
          f"（其中 {stats['runtime_amounts']} 条是运行期取值，已标记 needs_runtime）")
    print(f"  关键词：{stats['keywords']}")

    # 与社区库对拍：不是裁决，是把分歧摆出来
    codex_path = out.parent / "cards.json"
    audit = cross_check(cards, codex_path)
    if audit["checked"]:
        bad = len(audit["mismatches"])
        print(f"  与社区库描述对拍      {audit['checked']} 条，"
              f"分歧 {bad} 条（{bad / audit['checked']:.1%}）")
        audit_path = out.with_name("cards_source_audit.json")
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        print(f"  分歧清单 → {audit_path}")
        if not args.quiet:
            for item in audit["mismatches"][:12]:
                print(f"      {item['cid']:22s} 源码={item['source_value']:<4}"
                      f" 描述数字={item['description_numbers'][:5]}")

    if not args.quiet:
        unsupported = {k.split("::", 1)[1]: v for k, v in report.items()
                       if k.startswith("unsupported_command::")}
        if unsupported:
            print("\n未支持的命令（按出现次数）：")
            for name, count in sorted(unsupported.items(), key=lambda kv: -kv[1]):
                print(f"    {count:4d}  {name}")
        if stats["unresolved_examples"]:
            print("\n量解析不出的样例（卡 / 算子 / 原始表达式）：")
            for cid, op, raw in stats["unresolved_examples"]:
                print(f"    {cid:24s} {op:14s} {raw[:52]}")
        other = {k: v for k, v in report.items()
                 if not k.startswith("unsupported_command::")}
        if other:
            print("\n其它统计：")
            for key, count in sorted(other.items(), key=lambda kv: -kv[1])[:18]:
                print(f"    {count:4d}  {key}")
    print(f"\n写出 {out}（{total} 张）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
