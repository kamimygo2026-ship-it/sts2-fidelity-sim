"""从反编译源码抽取**事件**结构（``data/content/repo/events_source.json``）。

铁律（``docs/09`` §5.2）：结构、数值、条件全部来自源码；社区库只用来对账。

真机的事件模型（``MegaCrit.Sts2.Core.Models/EventModel.cs``）::

    BeginEvent(player, …)            Rng = new Rng(runSeed + slot + XXH64(eventId))
      CalculateVars()                把 DynamicVars 里的量掷出来（如 Gold = Rng.NextInt(41,69)）
      SetInitialEventState(…)        GenerateInitialOptions() → 第一页的选项
    选项被点 → EventOption.Chosen() → OnChosen()（一个方法或内联 lambda）
      里面可以 SetEventFinished(desc)        结束事件
      或     SetEventState(desc, options)    进入下一页

所以"一个事件 = 一页页选项 + 每个选项的效果与去向"。本工具就抽这个图。

⚠️ 与卡牌共用一套东西：``extract_effects``（算子翻译）、``parse_vars``（DynamicVars）、
``method_spans``（判断某句 ``SetEventState`` 写在哪个选项方法里）。
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.extract_cards import (  # noqa: E402
    REWARD_KINDS, balanced, balanced_braces, build_key_map, collect_locals,
    enclosing_method, extract_effects, method_spans, parse_vars, snake, split_top,
    _enclosing_loop_bound)
from tools.extract_monsters import method_body  # noqa: E402

SRC_DEFAULT = Path("data/decompiled/sts2/MegaCrit.Sts2.Core.Models.Events")
OUT_DEFAULT = Path("data/content/repo/events_source.json")
CODEX = Path("data/raw/spire-codex/repo/events.json")

#: 选项体里出现这些命令 → 需要引擎还没有的子系统（如实计一条 reason）。
#:
#: ⚠️ **选牌不在这里**：`CardSelectCmd` 本身不是子系统缺口，
#: 要按**用途**判（`removal` / `upgrade` 引擎已支持；`enchant` 等没支持）。
#: 早先写了一条"只要出现 CardSelectCmd 就报需要玩家选牌"的marker，
#: 结果 24 个本来能做的选项被一刀切掉 —— 标记过宽的后果和漏报一样糟：
#: 它把"能做"混进"不能做"，于是没人再去看那 24 个到底是什么。
#:
#: ⚠️ **`RelicFactory.PullNextRelicFromFront/Back` 的两条已经删掉了**：
#: 引擎**已经实现**了它（``runeffects._pull_random_relic`` → ``relicbag.RelicGrabBag``，
#: 走 ``rewards`` 流掷稀有度、从桶前端取走），这两条 marker 是**过期的误报** ——
#: 它们让 6 个本来能跑的事件（``this_or_that`` / ``luminous_choir`` / ``unrest_site`` …）
#: 被标成残缺。判"某功能有没有实现"要对着引擎代码看，不能只看抽取器里的名单。
SUBSYSTEM_MARKERS: tuple[tuple[str, str], ...] = (
    ("EnterCombatWithoutExitingEvent", "需要事件战斗交接"),
    ("CardCmd.Enchant", "需要附魔系统"),
    ("CardSelectCmd.FromDeckForEnchantment", "需要附魔系统"),
    ("PotionCmd.TryToProcure", "需要药水槽系统"),
    ("CardCmd.TransformToRandom", "需要随机转化（transformations 流）"),
    ("CardPileCmd.AddCursesToDeck", "需要随机诅咒"),
    ("PlayerCmd.MimicRestSiteHeal", "需要营火治疗量语义"),
    ("OstyCmd", "需要宠物子系统"),
    ("ForgeCmd", "需要锻造子系统"),
)


def codex_key(raw: str) -> str:
    """社区库的 id 是 SCREAMING_CASE，``snake()`` 会把它拆成 ``_w_o_o_d``。

    ⚠️ 踩过：拿 ``snake()`` 归一化社区 id，于是**每一个**查询都落空，
    而现象只是"社区库里的选项数都是 0" —— 看起来像数据缺失，其实是查错了键。
    社区 id 的归一化就是 ``lower()``（``WOOD_CARVINGS`` → ``wood_carvings``）。
    """
    return str(raw).strip().lower()


def page_of_key(key: str) -> str | None:
    """``X.pages.LINGER1.description`` → ``LINGER1``。"""
    match = re.search(r"\.pages\.(\w+)\.", key)
    return match.group(1) if match else None


# ==========================================================================
# 奖励（``RewardsCmd.OfferCustom``）
# ==========================================================================
#: ``new XxxReward(...)`` 的构造器。**参数必须按圆括号配平取**
#: （``balanced``），不能用 ``[^)]*`` —— 真机的参数里常常嵌着调用
#: （``ModelDb.Potion<GlowwaterPotion>()``），``[^)]*`` 会在第一个 ``)`` 处截断，
#: 于是参数看起来"含 ModelDb."而被当成 TestMode 残留丢掉。
REWARD_CTOR = re.compile(r"new\s+(\w+Reward)\s*\(")

#: `if (TestMode.IsOn) { … }` —— 测试专用分支（写死的假奖励），真机不会走。
TESTMODE_IF = re.compile(r"if\s*\(\s*TestMode\.IsOn\s*\)\s*\{")

#: `_ = base.Owner.PlayerRng.Rewards.NextItem(items)`：从**药水池**里均匀抽一个。
POTION_NEXT_ITEM = re.compile(r"Rewards\.NextItem\(\s*(\w+)\s*\)")

#: `base.DynamicVars["FoulPotions"].IntValue` / `base.DynamicVars.Potions.IntValue`
DYNAMIC_VAR = re.compile(
    r"(?:base\.)?DynamicVars(?:\[\"(\w+)\"\]|\.(\w+))\.(?:IntValue|BaseValue)")


def strip_testmode(text: str) -> str:
    """删掉 ``if (TestMode.IsOn) { … }`` **整块**。

    只把条件改成 ``if (false)`` 是不够的：块里的
    ``new RelicReward(ModelDb.Relic<Anchor>()…)`` 仍然会被扫到，然后为了排掉它
    就得给"参数里含 ModelDb."加一条一刀切的过滤 —— 那条过滤会误杀真奖励
    （``ModelDb.Potion<GlowwaterPotion>()`` 就是真奖励）。整块删掉才是干净的。
    """
    out: list[str] = []
    index = 0
    while True:
        match = TESTMODE_IF.search(text, index)
        if not match:
            out.append(text[index:])
            return "".join(out)
        out.append(text[index:match.start()])
        brace = text.find("{", match.end() - 1)
        if brace < 0:
            return "".join(out)
        _body, end = balanced_braces(text, brace)
        index = end + 1


def _var_value(expr: str, vars_: dict[str, int]) -> int | None:
    """把循环上界表达式解析成具体份数（``DynamicVars["X"].IntValue`` → 字面量）。"""
    text = expr.strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    match = DYNAMIC_VAR.fullmatch(text)
    if match:
        value = vars_.get(match.group(1) or match.group(2))
        return int(value) if value is not None else None
    return None


def _pool_of_options(options_expr: str, body: str,
                     locals_: dict[str, str]) -> str | None:
    """``new CardReward(options, …)`` 的卡池：跟着局部变量回到 ``CardPool<X>``。"""
    text = options_expr.strip()
    if re.fullmatch(r"\w+", text) and text in locals_:
        text = locals_[text]
    match = re.search(r"CardPool\s*<\s*(\w+)\s*>", text)
    if match:
        return match.group(1)
    if "Character.CardPool" in text:
        return "__character__"
    return None


def _potion_reward_spec(args: str, body: str,
                        locals_: dict[str, str]) -> dict | None:
    """``new PotionReward(<第 1 个参数>)`` → 药水规格。"""
    parts = split_top(args)
    first = parts[0].strip() if parts else ""
    # ``potionModel.ToMutable()`` → 变回那个局部变量名（真机到处包一层 ToMutable）
    first = re.sub(r"\.ToMutable\(\)\s*$", "", first).strip()
    # `new PotionReward(base.Owner)`：没有药水参数 → ``PotionReward.Populate``
    # 走 ``PotionFactory.CreateRandomPotionOutOfCombat``（稀有度掷点 0.1/0.35）。
    if not first or "Owner" in first:
        return {"kind": "potion", "mode": "factory"}
    match = re.search(r"ModelDb\.Potion\s*<\s*(\w+)\s*>", first)
    if match:
        return {"kind": "potion", "potion": snake(match.group(1))}
    # 参数是局部变量：跟着它找定义
    if re.fullmatch(r"\w+", first) and first in locals_:
        definition = locals_[first]
        match = re.search(r"ModelDb\.Potion\s*<\s*(\w+)\s*>", definition)
        if match:
            return {"kind": "potion", "potion": snake(match.group(1))}
        item = POTION_NEXT_ITEM.search(definition)
        if item:
            # `items` 是候选集合：角色药水池 ∪ 共享药水池，可能再按稀有度过滤
            spec: dict = {"kind": "potion", "mode": "pool"}
            source = locals_.get(item.group(1), "")
            rarity = re.search(r"PotionRarity\.(\w+)", source)
            if rarity:
                spec["rarity"] = rarity.group(1).lower()
            return spec
    return None


#: 事件类里的**私有辅助方法**声明：``private async Task AddGuilty(int amount)``
#: → ``{方法名: (泛型参数, 参数名, 方法体)}``。
#: ⚠️ **泛型参数也要收**：``AddAndPreview<T>(LocString loc)`` 靠 ``T`` 决定
#: "加的是哪张卡"（``Bugslayer.cs:39-44``），漏了它这条效果就抽不出卡 id。
HELPER_DECL = re.compile(
    r"\b(?:public|protected|private|internal)\s+(?:override\s+)?"
    r"(?:async\s+)?[\w<>\[\]\.\?,\s]+?\s+(\w+)\s*(?:<\s*([^<>]*?)\s*>)?\s*"
    r"\(([^)]*)\)\s*(?:where\b[^{]*)?\{")


def _helper_methods(source: str) -> dict[str, tuple[list[str], list[str], str]]:
    """这个事件类里所有**带形参表**的方法（名字 → (泛型参数, 参数名, 方法体)）。

    只用来摊平 ``await <方法名>(<实参>);`` 这种"选项体只写一句委派"的写法。
    """
    helpers: dict[str, tuple[list[str], list[str], str]] = {}
    for match in HELPER_DECL.finditer(source):
        generics = [name.strip() for name in split_top(match.group(2) or "")
                    if name.strip()]
        params: list[str] = []
        for part in split_top(match.group(3)):
            names = re.findall(r"\b(\w+)\b", part)
            if names:
                params.append(names[-1])          # 参数名是声明里最后一个标识符
        helpers.setdefault(match.group(1),
                           (generics, params,
                            balanced_braces(source, match.end() - 1)[0]))
    return helpers


#: ``await <私有方法>(<参数>);`` / ``await <私有方法><泛型>(<参数>);``
#: —— 只摊平**语句级**的委派。
#: 不做表达式级的（``IsValid(CardTag.Strike, c)`` 在 LINQ 里返回 bool，
#: 摊平它等于把判断体塞进查询表达式里，必错）。
HELPER_CALL = re.compile(r"await\s+(\w+)\s*(?:<\s*([^<>]*?)\s*>)?\s*\(")

#: 动态变量被**改写**（``BaseValue +=`` / ``BaseValue =``）。
#: ⚠️ 只认选项体里的；``CalculateVars`` 里的赋值是**正常**的初始化/掷点。
DYNAMIC_VAR_MUTATION = re.compile(
    r"DynamicVars(?:\[\"[^\"]+\"\]|\.\w+)\.BaseValue\s*[-+*/]?=")


def inline_helpers(body: str, helpers: dict[str, tuple[list[str], list[str], str]],
                   depth: int = 3) -> str:
    """把 ``await <私有方法>(…)`` 摊平成那个方法的**方法体**（形参按实参代换）。

    ⚠️ 只摊平**在本事件类里有定义**、且用 ``await`` 调用的方法 ——
    命令类（``CreatureCmd.*``）与基类方法（``SetEventFinished``）都不在里面，
    所以不会把别的东西误当辅助方法摊进来。
    ⚠️ **代换只发生在被调方法的体内**，不碰实参表达式 ——
    在整段调用文本上做代换会在实参里也替换同名标识符（``F(x + 1)`` 里把 ``x``
    也换掉），那是静默算错。
    ⚠️ 深度限制 3：真机的委派最多两层（``Decipher → LoseMaxHpAndUpgrade``），
    再深就不是"选项体一句委派"的形态了。
    """
    if depth <= 0:
        return body
    for match in list(HELPER_CALL.finditer(body)):
        name = match.group(1)
        if name not in helpers:
            continue
        generics, params, callee = helpers[name]
        args, end = balanced(body, match.end() - 1)
        if not re.match(r"\s*;", body[end + 1:]):
            continue                      # 不是语句级调用（后面还跟着别的东西）
        call_args = split_top(args)
        if len(call_args) != len(params):
            continue                      # 参数个数对不上 → 不猜
        type_args = [item.strip()
                     for item in split_top(match.group(2) or "") if item.strip()]
        if len(type_args) != len(generics):
            continue                      # 泛型实参个数对不上 → 不猜
        text = callee
        for param, argument in zip(generics + params, type_args + call_args):
            text = re.sub(rf"\b{param}\b", argument, text)
        text = inline_helpers(text, helpers, depth - 1)
        return inline_helpers(body[:match.start()] + text + body[end + 1:],
                              helpers, depth - 1)
    return body


def extract_event_rewards(body: str, source: str,
                          vars_: dict[str, int]) -> tuple[list[dict], str]:
    """把 ``RewardsCmd.OfferCustom(owner, list)`` 抽成**奖励规格**列表。

    真机（``RewardsCmd.OfferCustom``）把一组 ``Reward`` 摆到奖励界面，
    **每一条各自"拿/跳过"**（``RelicReward.OnSelect`` / ``PotionReward.OnSelect``）。
    所以引擎那边的表达是"每条奖励一个 ``RewardGroup``"，
    这里的规格里带 ``count``（这一条有几份，来自外层 ``for`` 循环）。

    三种来源都要跟上：

    * ``OfferCustom(owner, GenerateRewards())`` → 跟到那个**私有方法**；
    * ``OfferCustom(owner, list)`` → ``list`` 在本方法里累加（可能在 ``for`` 里）；
    * ``OfferCustom(owner, new List<Reward>(n) { … })`` → 就地扫描。

    返回 ``(规格列表, 失败原因)``。**解析不出任何一条就返回原因** ——
    调用方照实记缺口，绝不"少给一条奖励"。
    """
    call = re.search(r"RewardsCmd\.OfferCustom\s*\(", body)
    if not call:
        return [], "不是 OfferCustom"
    args, call_end = balanced(body, call.end() - 1)
    parts = split_top(args)
    list_expr = parts[1].strip() if len(parts) > 1 else ""
    target = strip_testmode(body)
    follow = re.fullmatch(r"(\w+)\s*\(\s*\)", list_expr or "")
    if follow:
        sub = method_body(source, follow.group(1))
        if sub is None:
            return [], f"奖励列表方法 {follow.group(1)}() 找不到"
        target = strip_testmode(sub)

    locals_ = collect_locals(target)
    specs: list[dict] = []
    for match in REWARD_CTOR.finditer(target):
        kind = REWARD_KINDS.get(match.group(1))
        if kind is None:
            continue
        ctor_args, _e = balanced(target, match.end() - 1)
        spec: dict | None
        if kind == "potion":
            spec = _potion_reward_spec(ctor_args, target, locals_)
        elif kind == "relic":
            spec = {"kind": "relic"}
            relic = re.search(r"ModelDb\.Relic\s*<\s*(\w+)\s*>", ctor_args)
            if relic:
                spec["relic"] = snake(relic.group(1))
            else:
                rarity = re.search(r"RelicRarity\.(\w+)", ctor_args)
                if rarity:
                    spec["rarity"] = rarity.group(1).lower()
        elif kind == "card":
            ctor_parts = split_top(ctor_args)
            # `new CardReward(options, 3, owner)`：第 2 个参数 = **候选张数**
            choices = None
            if len(ctor_parts) >= 2 and re.fullmatch(r"\d+", ctor_parts[1].strip()):
                choices = int(ctor_parts[1])
            options_expr = ctor_parts[0] if ctor_parts else ""
            pool = _pool_of_options(options_expr, target, locals_)
            if choices is None or pool is None:
                return [], (f"CardReward 的候选张数/卡池解析不出"
                            f"（{ctor_args.strip()[:48]}）")
            spec = {"kind": "card", "choices": choices, "pool": pool}
        else:
            return [], f"奖励类型 {kind} 引擎未实现"
        if spec is None:
            return [], f"{match.group(1)} 的参数解析不出（{ctor_args.strip()[:48]}）"
        count = 1
        loop = _enclosing_loop_bound(target, match.start())
        if loop:
            for name, expr in locals_.items():
                if loop == name:
                    loop = expr
                    break
            value = _var_value(loop, vars_)
            if value is None:
                return [], f"奖励份数解析不出（循环上界 {loop[:32]}）"
            count = value
        spec["count"] = count
        specs.append(spec)
    if not specs:
        return [], "OfferCustom(奖励列表里没有 new XxxReward)"
    # 位置：OfferCustom 之后还有没有**会产生效果**的命令？
    # 有的话"把奖励追加在最后"就不再等价于真机的顺序，如实拒绝。
    _tail_effects, _u, _c = extract_effects(body[call_end + 1:], {})
    if _tail_effects:
        return [], "OfferCustom 之后还有效果，顺序无法保证"
    return specs, ""


#: ``RelicOption<T>(…)`` / ``RelicOption(relic, …)`` ——
#: ``EventModel.cs:606-621`` 的封装：**标题/描述取遗物的**，
#: 本地化 key 由 ``OptionKey(pageName, relic.Id.Entry)`` 生成，
#: 也就是 ``{SLUG(类名)}.pages.{页}.options.{遗物 ENTRY}``（``EventModel.cs:637-640``）。
#: ``onChosen`` 省略时为 ``null`` → ``IsLocked``（``EventOption.cs:62``）。
RELIC_OPTION_CALL = re.compile(r"RelicOption\s*(?:<\s*(\w+)\s*>)?\s*\(")


def relic_option_records(body: str, class_name: str) -> list[tuple[int, dict]]:
    """抽出 ``RelicOption<T>(Method)`` 形式的选项（返回 ``(位置, 记录)``）。

    ⚠️ **只认泛型重载 + ``onChosen`` 是方法名的写法**：
    ``RelicOption<BigMushroom>(BigMushroom)``（``HungryForMushrooms.cs:15-16``）。
    非泛型重载（``RelicOption(seaGlass)``，参数是**遗物实例**）与
    ``RelicOption(r.ToMutable())`` 那种在 LINQ 里现算的，**这里不猜** ——
    古事件（Ancient）的选项池就是这么写的，它们走的本来也不是这条路。
    """
    found: list[tuple[int, dict]] = []
    for match in RELIC_OPTION_CALL.finditer(body):
        relic_class = match.group(1)
        if relic_class is None:
            continue
        args, _end = balanced(body, match.end() - 1)
        parts = [p.strip() for p in split_top(args)]
        on_chosen = parts[0] if parts else ""
        page_name = "INITIAL"
        if len(parts) >= 2 and parts[1].startswith('"'):
            page_name = parts[1].strip('"')
        if on_chosen and not re.fullmatch(r"\w+", on_chosen):
            continue
        entry = snake(relic_class).upper()
        key = f"{snake(class_name).upper()}.pages.{page_name}.options.{entry}"
        found.append((match.start(), {
            "key": key,
            "text_key": key,
            "page": page_name,
            "on_chosen": on_chosen,
            "locked": not on_chosen,
            "inline_body": None,
            "lambda_expr": None,
        }))
    return found


def option_records(body: str) -> list[tuple[int, dict]]:
    """从一段代码里抽出所有 ``new EventOption(...)``（返回 ``(位置, 记录)``）。

    三种 ``onChosen`` 写法都要认（真机都用）：

    * **方法引用**：``new EventOption(this, Plain, "…")``
    * **内联 lambda**：``new EventOption(this, async delegate { … }, "…")``
      —— ⚠️ 允许 ``delegate`` **不带参数表**（真机写 ``async delegate\\n{``）
    * **表达式 lambda**：``new EventOption(this, () => SomeMethod(x), "…")``

    ``onChosen`` 为 ``null`` 是**正常状态**，不是缺口：``EventOption.IsLocked``
    就是 ``OnChosen == null``（``EventOption.cs:62``）—— 那是"条件不满足所以锁住"
    的选项，界面上仍然显示。早先把它们记成"选项体抽不出"，把正常状态报成了缺口。

    返回位置是为了和 :func:`relic_option_records` **按源码顺序合并** ——
    选项在页面上的顺序是玩家看得见的（第一个选项就是默认高亮那个）。
    """
    found: list[tuple[int, dict]] = []
    for match in re.finditer(r"new EventOption\s*\(", body):
        args, _end = balanced(body, match.end() - 1)
        keys = re.findall(r'"([^"]+)"', args)
        parts = [p.strip() for p in re.split(r",(?![^()<>]*[>)]\s*$)", args)]
        on_chosen = parts[1] if len(parts) > 1 else ""
        inline = re.search(
            r"(?:async\s+)?(?:delegate\s*(?:\([^)]*\))?|\([^)]*\)\s*=>)\s*\{", args)
        expression = re.search(r"\([^)]*\)\s*=>\s*(?!\{)([^,]+)", args)
        entry = {
            "key": keys[-1] if keys else "",
            "text_key": keys[-1] if keys else "",
            "page": page_of_key(keys[-1]) if keys else None,
            "on_chosen": on_chosen,
            "locked": on_chosen.strip() == "null",
            "inline_body": (balanced_braces(args, inline.end() - 1)[0]
                            if inline else None),
            "lambda_expr": (expression.group(1).strip() if expression else None),
        }
        found.append((match.start(), entry))
    return found


def collect_pages(source: str, spans: list[tuple[str, int, int]],
                  key_map: dict[str, str], class_name: str = "",
                  vars_: dict[str, int] | None = None) -> tuple[dict, dict]:
    """返回 ``(pages, option_methods)``。

    * ``pages[page_id] = {"options": [...], "from": 方法名}``
    * ``option_methods[方法名] = {effects, unsupported, choices, outcome}``

    ``class_name`` / ``vars_`` 分别是事件类名与 ``CanonicalVars`` 的字面量：
    前者用于 ``RelicOption`` 的本地化 key（``EventModel.cs:637-640``），
    后者用于把奖励份数（``DynamicVars["X"].IntValue``）解析成具体数字。
    """
    vars_ = vars_ or {}
    pages: dict[str, dict] = {}

    def records(text: str) -> list[dict]:
        """``new EventOption`` 与 ``RelicOption<T>(…)`` 按**源码顺序**合并。"""
        merged = list(option_records(text))
        merged.extend(relic_option_records(text, class_name))
        merged.sort(key=lambda item: item[0])
        return [record for _index, record in merged]

    # 1) 第一页来自 `GenerateInitialOptions`
    initial = re.search(r"IReadOnlyList<EventOption>\s+GenerateInitialOptions\s*\([^)]*\)\s*\{",
                        source)
    if initial:
        body = balanced_braces(source, initial.end() - 1)[0]
        pages["INITIAL"] = {"from": "GenerateInitialOptions",
                            "options": records(body)}
    else:
        # 有些事件把第一页写在别处（例如 Ancient 事件）；如实记下来
        pages["INITIAL"] = {"from": "", "options": []}

    # 2) 其余页来自 `SetEventState(desc, options)` —— 每个这样的调用都在某个选项方法里
    called: dict[str, list[dict]] = {}
    for match in re.finditer(r"SetEventState\s*\(", source):
        args, _end = balanced(source, match.end() - 1)
        page = page_of_key(args) or ""
        if not page or page in pages:
            continue
        owner = enclosing_method(spans, match.start())
        pieces = [p.strip() for p in re.split(r",(?![^()<>]*[>)]\s*$)", args)]
        options_expr = pieces[1] if len(pieces) > 1 else ""
        # 选项来自另一个方法（`LingerPage()` / `new List<EventOption>{…}`）
        call = re.fullmatch(r"(\w+)\s*\(\s*\)", options_expr or "")
        if call:
            called[call.group(1)] = []
            pages[page] = {"from": call.group(1), "options": []}
        else:
            pages[page] = {"from": f"{owner}() 内联", "options": records(args)}
            pages[page]["inline_in"] = owner

    for name in called:
        body = None
        for mname, start, end in spans:
            if mname == name:
                body = source[start:end]
                break
        if body is None:
            continue
        for page, info in pages.items():
            if info["from"] == name:
                info["options"] = records(body)

    # 3) 选项方法自己的效果与去向
    bodies = {name: source[start:end] for name, start, end in spans}
    helpers = _helper_methods(source)
    option_methods: dict[str, dict] = {}
    for page, info in pages.items():
        for option in info["options"]:
            if option.get("locked"):
                # 锁住的选项：真机就是"显示但点不动"（`EventOption.IsLocked`）。
                option.update({"effects": [], "unsupported": [], "choices": 0,
                               "outcome": {"kind": "none"}, "reasons": []})
                continue
            token = (option.get("on_chosen") or "").strip()
            body = option.get("inline_body")
            lambda_expr = option.get("lambda_expr")
            if body is None and lambda_expr:
                # 表达式 lambda：`() => SomeMethod(...)` → 跟到那个方法
                call = re.fullmatch(r"(\w+)\s*\((.*)\)", lambda_expr, re.S)
                if call and not call.group(2).strip():
                    body = bodies.get(call.group(1))
                else:
                    # 带参数的调用（`RiderChosen(riders[0])`）→ 参数会改变效果，
                    # 不能只取方法体，如实报缺口
                    option.update({"effects": [], "unsupported": [],
                                   "choices": 0, "outcome": {"kind": "none"},
                                   "reasons": [f"选项 lambda 带参数：{lambda_expr[:48]}"]})
                    continue
            if body is None and re.fullmatch(r"\w+", token):
                body = bodies.get(token)
            if body is None:
                option.update({"effects": [], "unsupported": [], "choices": 0,
                               "outcome": {"kind": "none"},
                               "reasons": ["选项体抽不出（写法未识别）"]})
                continue
            # ⭐ **把私有辅助方法摊平进来**：事件选项常常只写一句
            # `await AddGuilty(base.DynamicVars["BatheCurses"].IntValue);`
            # （``Wellspring.cs:48``）或 `await Trade(potion);`
            # （``TheFutureOfPotions.GenerateInitialOptions`` 里的内联 delegate），
            # 真正的命令在那个私有方法体里。不摊平的话这个选项会被抽成
            # **"有效果、没去向"甚至"什么都没有"** —— 而它照样会被计成"可用"，
            # 点下去什么也不发生。实测 Wellspring 的 BATHE 就因此少了
            # "往牌组塞 1 张 Guilty"，是**白拿一次移除**。
            body = inline_helpers(body, helpers)
            effects, unsupported, choices = extract_effects(body, key_map)
            outcome = {"kind": "none"}
            finished = re.search(r'SetEventFinished\s*\(\s*L10NLookup\s*\(\s*"([^"]+)"',
                                 body)
            goto = re.search(r'SetEventState\s*\(\s*L10NLookup\s*\(\s*"([^"]+)"', body)
            if finished:
                outcome = {"kind": "finished",
                           "page": page_of_key(finished.group(1)) or ""}
            elif goto:
                outcome = {"kind": "goto",
                           "page": page_of_key(goto.group(1)) or ""}
            reasons = []
            for marker, reason in SUBSYSTEM_MARKERS:
                if marker in body and reason not in reasons:
                    reasons.append(reason)
            # ⚠️ **选项体改写动态变量**（``AbyssalBaths.OnImmerse``：
            # ``DynamicVars.Damage.BaseValue += 1m``）＝ 跨页累计的量。
            # 引擎的事件图是**静态**的（一页的效果只由选项原文决定），
            # 表达不了"第二次点同一个选项伤害 +1" —— 照原样跑会**少掉伤害**，
            # 而日志看起来完全正常。如实标残缺（宁可不可用）。
            if DYNAMIC_VAR_MUTATION.search(body):
                reason = "选项体改写动态变量（跨页累计），引擎不支持"
                if reason not in reasons:
                    reasons.append(reason)
            # `CardPileCmd.AddCursesToDeck(<诅咒集合>, owner)`：**复数**重载，
            # 参数是一组具体诅咒而不是泛型参数，`extract_cards` 只认
            # `AddCurseToDeck<T>()` 的写法，于是整条被报成缺口
            # （并且被错标成"需要随机诅咒"）。真机的调用**全都是具体的诅咒**
            # （``UnrestSite.cs:48`` PoorSleep、``Wellspring.cs:54`` Guilty、
            # ``FieldOfManSizedHoles.cs:48`` Normality …）。
            curses_call = re.search(r"CardPileCmd\.AddCursesToDeck\s*\(", body)
            if curses_call:
                call_args, _ce = balanced(body, curses_call.end() - 1)
                curse = re.search(r"ModelDb\.Card\s*<\s*(\w+)\s*>", call_args)
                if curse is not None:
                    # 份数：`Enumerable.Repeat(card, N)` 的第 2 个参数
                    amount = 1
                    repeat = re.search(r"Enumerable\.Repeat\s*\(", call_args)
                    if repeat:
                        repeat_args = split_top(balanced(call_args, repeat.end() - 1)[0])
                        if len(repeat_args) >= 2:
                            amount = _var_value(repeat_args[1], vars_) or 1
                    effects.append({"op": "add_card", "amount": amount,
                                    "amount_var": None, "power": None,
                                    "target": "self", "times": 1, "amount_raw": "",
                                    "card": snake(curse.group(1)), "pile": "deck"})
                    unsupported = [item for item in unsupported
                                   if "AddCursesToDeck" not in item]
                    reasons = [reason for reason in reasons
                               if reason != "需要随机诅咒"]
            # 摊平之后会出现"选牌 + 它的执行命令"这种**重复**：
            # ``ZenWeaver.RemoveCardsAndProceed`` 里写的是
            # ``CardPileCmd.RemoveFromDeck((await CardSelectCmd.FromDeckForRemoval(…)).ToList())``
            # —— 参数是"表达式"而不是一个局部变量，``extract_cards`` 的
            # ``SELECTION_ACTIONS`` 去重判据（参数必须是选牌结果那个局部变量）
            # 认不出，于是除了那条 ``select_card(removal)`` 之外**又多了一条**
            # ``remove_card_from_deck``。多出来的那条会让引擎**再移除一张**。
            # 判据：选项里既有 ``select_card(removal)``，又有一条"没有指定卡"的
            # ``remove_card_from_deck``（指定了具体卡的不能丢，那是另一回事）。
            if any(e.get("op") == "select_card" and e.get("purpose") == "removal"
                   for e in effects):
                effects = [e for e in effects
                           if not (e.get("op") == "remove_card_from_deck"
                                   and not e.get("card"))]
            # ⭐ 选牌的**执行命令**在"循环 / 分句"写法下，参数是循环变量而不是
            # 选牌结果本身（``foreach (CardModel item in list) { await
            # CardCmd.TransformToRandom(item, base.Rng, …); }``，
            # ``MorphicGrove.cs:48-51``）。``extract_cards`` 的 ``SELECTION_ACTIONS``
            # 只认"参数就是那次选牌的局部变量"，于是这里会留下一条**假缺口**
            # ``TransformToRandom(卡 id 解析不出)`` —— 而用途已经由那条
            # ``select_card(purpose="transform")`` 完整表达了。
            #
            # 只在"选项里确实有一次该用途的选牌"时丢弃，且只丢弃**这一条**
            # 精确的假缺口文本（别的 ``_card_unresolved`` 照旧如实上报）。
            purposes = {effect.get("purpose") for effect in effects
                        if effect.get("op") == "select_card"}
            if "transform" in purposes:
                unsupported = [item for item in unsupported
                               if item not in ("TransformToRandom(卡 id 解析不出)",
                                               "Transform(卡 id 解析不出)")]
                # 同理：`CardCmd.TransformToRandom` 正是那次选牌的**执行**，
                # 引擎已经实现了它（`runeffects._transform_deck_card`），
                # 这条子系统 marker 在这里是过期误报。
                reasons = [reason for reason in reasons
                           if reason != "需要随机转化（transformations 流）"]
            # `PlayerCmd.LoseGold(base.Owner.Gold, …)`（``MorphicGrove.cs:46``）：
            # 量是**当前金币全额**，不是抽不出的数字。`LoseGold` 会把结果夹到 0
            # （``PlayerCmd.cs:198``），所以等价于"金币清零"。
            for effect in effects:
                if effect.get("op") == "lose_gold" and \
                        str(effect.get("amount_raw") or "").strip() == "base.Owner.Gold":
                    effect["op"] = "lose_all_gold"
                    effect["amount"] = 0
            # ⭐ `RewardsCmd.OfferCustom`：奖励列表写在别处（私有方法 / 局部变量 /
            # 循环里累加），`extract_effects` 只看到"有个没实现的命令"。
            # 这里专门解析它 —— 解析不出就**照实报缺口**（绝不"少给一条奖励"）。
            if any("OfferCustom" in item for item in unsupported):
                specs, failure = extract_event_rewards(body, source, vars_)
                if specs:
                    unsupported = [item for item in unsupported
                                   if "OfferCustom" not in item]
                    for spec in specs:
                        count = int(spec.pop("count", 1))
                        effects.append({
                            "op": "offer_rewards", "amount": count,
                            "target": "self", "times": 1,
                            "filter": tuple(sorted(spec.items())),
                        })
                else:
                    unsupported = [item for item in unsupported
                                   if "OfferCustom" not in item]
                    unsupported.append(f"RewardsCmd.OfferCustom({failure})")
            option.update({"effects": effects, "unsupported": sorted(set(unsupported)),
                           "choices": choices, "outcome": outcome,
                           "reasons": reasons})
    return pages, option_methods


def rng_rolls(source: str, key_map: dict[str, str]) -> list[dict]:
    """``CalculateVars`` 里的随机掷点（``base.DynamicVars.X.BaseValue = base.Rng…``）。"""
    body = None
    for name, start, end in method_spans(source):
        if name == "CalculateVars":
            body = source[start:end]
            break
    if body is None:
        return []
    rolls = []
    for match in re.finditer(
            r"DynamicVars(?:\[\s*\"(\w+)\"\s*\]|\.(\w+))\.BaseValue\s*=\s*([^;]+);", body):
        name = match.group(1) or match.group(2)
        expr = match.group(3).strip()
        rolls.append({"var": key_map.get(name, name), "raw": expr,
                      "uses_event_rng": "Rng." in expr})
    return rolls


def gate_of(source: str) -> dict | None:
    """``IsAllowed(runState)`` 的原文（能不能静态求值由引擎侧判）。

    把原文留下来是为了**可审计**：不静态求值的条件一律让事件标残缺，
    绝不"看起来像 true"就放行（``docs/09`` §5.3）。

    ⚠️ 方法体要按**花括号**配平（``balanced_braces``），不能用 ``balanced``
    ——后者配平的是圆括号，会在方法体里第一个右括号处截断，
    于是原文里混进**后面整个类**的内容，下游的条件匹配全部落空。
    实测 35 个事件因此被判成"条件写法未支持"。
    """
    match = re.search(r"public override bool IsAllowed\s*\([^)]*\)\s*\{", source)
    if not match:
        return None
    body = balanced_braces(source, match.end() - 1)[0]
    return {"body": body.strip()[:600]}


def parse_event(path: Path, codex: dict) -> dict:
    source = path.read_text(encoding="utf-8", errors="replace")
    class_name = path.stem
    eid = snake(class_name)
    vars_ = parse_vars(source)
    key_map = build_key_map(vars_)
    #: ``CanonicalVars`` 的字面量（``{名: 值}``）：奖励份数（``DynamicVars["X"].IntValue``）
    #: 与 ``IsAllowed`` 的阈值都要用它解析成具体数字。
    literal_vars = {str(v["name"]): int(v["base"])
                    for v in vars_ if v.get("base") is not None}
    spans = method_spans(source)
    pages, _ = collect_pages(source, spans, key_map, class_name, literal_vars)

    reasons: list[str] = []
    if "EnterCombatWithoutExitingEvent" in source:
        reasons.append("需要事件战斗交接")
    if len(pages) > 1:
        pass                                   # 多阶段本身不是缺口（引擎要支持）
    entry = codex.get(eid) or {}
    return {
        "eid": eid,
        "class": class_name,
        "act": entry.get("act"),
        "name": entry.get("name"),
        "pages": pages,
        "vars": vars_,
        "rng_rolls": rng_rolls(source, key_map),
        "gate": gate_of(source),
        "combat_encounter": (re.search(r"EnterCombatWithoutExitingEvent\s*<\s*(\w+)\s*>",
                                       source) or [None, None])[1]
        if "EnterCombatWithoutExitingEvent<" in source else None,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(SRC_DEFAULT))
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    codex_rows = json.loads(CODEX.read_text(encoding="utf-8")) if CODEX.exists() else []
    codex = {codex_key(str(r.get("id", ""))): r for r in codex_rows}

    records: list[dict] = []
    stats: collections.Counter = collections.Counter()
    reason_hist: collections.Counter = collections.Counter()
    for path in sorted(Path(args.src).glob("*.cs")):
        record = parse_event(path, codex)
        stats["events"] += 1
        options = [o for page in record["pages"].values() for o in page["options"]]
        stats["pages"] += len(record["pages"])
        stats["options"] += len(options)
        for option in options:
            if option.get("unsupported"):
                stats["option_unsupported"] += 1
            for reason in option.get("reasons") or ():
                reason_hist[reason] += 1
            if option.get("outcome", {}).get("kind") == "goto":
                stats["option_goto"] += 1
        if record["gate"]:
            stats["gated"] += 1
        if record["rng_rolls"]:
            stats["with_rng_rolls"] += 1
        records.append(record)

    Path(args.out).write_text(json.dumps(records, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    if not args.quiet:
        print(f"写出 {args.out}（{len(records)} 个事件）")
        print(f"  页 {stats['pages']} | 选项 {stats['options']} | "
              f"跳页 {stats['option_goto']} | 带缺口选项 {stats['option_unsupported']}")
        print(f"  有进入条件 {stats['gated']} | 用事件 RNG 掷点 {stats['with_rng_rolls']}")
        print("  子系统原因分布：")
        for reason, count in reason_hist.most_common():
            print(f"    {count:4d}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
