"""从反编译源码抽取**怪物招式的效果**（``docs/09`` §15 的 3.4）。

为什么必须做
------------
招式表（``monsters.json`` 的 ``moves``）来自**社区库**，而 AI 状态机来自**源码**。
社区库对很多招式只记了意图、**没记效果**，于是这些怪在模拟器里
"那一回合什么也不做"，而意图还显示着 buff / debuff ——
玩家按错误的威胁评估决策，模型学到"这只怪那回合很安全"。

实测缺口：**39 条 buff/debuff 招式效果为空**，涉及 35 只怪。

做法
----
怪物招式处理函数体用的是**和卡牌同一套命令**（``PowerCmd.Apply`` /
``CreatureCmd.GainBlock`` / ``CardPileCmd.AddToCombatAndPreview`` …），
所以直接复用 ``extract_cards.extract_effects``。

两处适配：

1. **常量内联**。卡牌写 ``base.DynamicVars.Damage``，怪物写裸标识符
   （``IncantationAmount`` / ``BiteDamage``）。抽取器认前者不认后者，
   所以先把函数体里的常量名替换成字面量再抽。
2. **进阶值单列**。真机写 ``GetValueIfAscension(DeadlyEnemies, 32, 26)`` ——
   A9 起数值不同。内联的是**普通值**，进阶值另存一份 ``ascension_effects``，
   由引擎按当前进阶选择。只存普通值会让 A9+ 的怪静默变弱。

用法
----
    python tools/extract_moves.py            # 写 data/content/repo/monster_moves.json
    python tools/extract_moves.py --report   # 只看质量报告
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools.extract_cards import extract_effects  # noqa: E402
from tools.extract_monsters import (  # noqa: E402
    DECOMPILED, _snake, collect_constants, method_body,
)

OUT_PATH = pathlib.Path("data/content/repo/monster_moves.json")
AI_PATH = pathlib.Path("data/content/repo/monster_ai.json")

#: 招式处理函数名的结尾（``IncantationMove`` / ``BiteMove``）。
HANDLER_SUFFIX = "Move"
#: 字符串字面量：内联常量时要**跳过**它们，否则会把卡面文本里的单词也换掉。
STRING_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"')


def inline_constants(body: str, consts: dict[str, dict]) -> tuple[str, set[str]]:
    """把裸常量名换成字面量，返回 ``(新函数体, 用到的常量名)``。

    ⚠️ 必须跳过字符串字面量：``TalkCmd.Play("DevourStartTrigger", …)``
    这类文本里的词一旦被替换，抽出来的就是垃圾效果 —— 而且不会报错。
    """
    used: set[str] = set()
    if not consts:
        return body, used

    # 先把字符串字面量挖出来占位，替换完再放回去
    holes: list[str] = []

    def stash(match: re.Match) -> str:
        holes.append(match.group(0))
        return f"\x00{len(holes) - 1}\x00"

    masked = STRING_LITERAL.sub(stash, body)

    def swap(match: re.Match) -> str:
        name = match.group(0)
        entry = consts.get(name)
        if entry is None or entry.get("normal") is None:
            return name
        used.add(name)
        return f"{entry['normal']}m"

    masked = re.sub(r"\b[A-Z]\w*\b", swap, masked)
    restored = re.sub(r"\x00(\d+)\x00", lambda m: holes[int(m.group(1))], masked)
    return restored, used


def ascension_variants(consts: dict[str, dict], used: set[str]) -> set[str]:
    """用到的常量里，哪些**进阶值与普通值不同**。"""
    out = set()
    for name in used:
        entry = consts.get(name) or {}
        if entry.get("ascension") is not None and \
                entry.get("ascension") != entry.get("normal"):
            out.add(name)
    return out


# ==========================================================================
# 「玩家从临时卡里二选一」的招式载荷（KnowledgeDemon 的 Curse of Knowledge）
# ==========================================================================
#: 候选卡的源码目录：抽 ``PowerVar<XxxPower>(Nm)`` 要知道那张卡施放什么能力、
#: 几层（``MindRot.CanonicalVars => new PowerVar<MindRotPower>(1m)``）。
#: ⚠️ ``DECOMPILED`` 指向的是**怪物**目录（``…Models.Monsters``），
#: 卡在它的兄弟目录里 —— 直接拼会得到一个不存在的路径，
#: 而症状只是"候选卡读不到 → 整条识别放弃"（不会报错，静默退回残缺）。
CARD_DIR = DECOMPILED.parent / "MegaCrit.Sts2.Core.Models.Cards"

#: 这一段形状的四个标志，缺一不可（保守：对不上就**不抽**，让这招保持残缺）。
CHOOSE_SCREEN = re.compile(r"FromChooseACardScreen")
ON_CHOSEN = re.compile(r"\.OnChosen\(\)")
#: 候选项列表的分组：``new IChoosable[2] { ModelDb.Card<Disintegration>(), … }``。
CHOICE_GROUP = re.compile(r"new\s+IChoosable\[\s*(\d+)\s*\]\s*\{([^}]*)\}")
#: 分组里的卡：``ModelDb.Card<MindRot>()``。
CARD_MODEL = re.compile(r"ModelDb\.Card<(\w+)>\(\)")
#: 静态 int 数组：``new int[3] { 6, 7, 8 }``（Disintegration 的三轮数值）。
INT_ARRAY = re.compile(r"new\s+int\[\s*\d+\s*\]\s*\{\s*([^}]*)\}")
#: 候选列表按哪个计数器取：``_curseOfKnowledgeSets[CurseOfKnowledgeCounter]``。
SETS_COUNTER = re.compile(r"_curseOfKnowledgeSets\s*\[\s*(\w+)\s*\]")
#: 卡类里的能力变量：``new PowerVar<MindRotPower>(1m)``。
POWER_VAR = re.compile(r"new\s+PowerVar<(\w+)>\(\s*([\d.]+)m?\s*\)")
#: 方法签名（用来找"哪个方法里有选择命令"）。反编译的签名独占一行、``{`` 在下一行。
METHOD_SIG = re.compile(
    r"(?:private|public|protected|internal)[^\n=;{}]*?\b(\w+)\s*\([^)]*\)", re.M)


def _choice_helper_names(source: str) -> set[str]:
    """文件里"造卡并让玩家二选一"的**辅助方法**名。

    真机把这段逻辑单独抽成了一个方法（``KnowledgeDemon.ChooseCurse``），
    招式体里只有一句 ``list.Add(ChooseCurse(target))``。所以判断"这一招是不是
    二选一招式"必须问两件事：**选择命令在不在招式体里**，或者
    **招式体有没有调用某个含选择命令的辅助方法**。

    ⚠️ 少了后半条，整只怪的**每一个**招式都会被贴上同一份候选（实测：
    KnowledgeDemon 的 Slap / KnowledgeOverwhelming / Ponder 三条攻击招式
    在 ``counter < 3`` 时全部变成"不打伤害、改去让玩家选卡"），
    而且日志完全正常 —— 这是本轮抓到的最危险的一处。
    """
    names: set[str] = set()
    for match in METHOD_SIG.finditer(source):
        name = match.group(1)
        if name in names:
            continue
        body = method_body(source, name)
        if body and CHOOSE_SCREEN.search(body) and ON_CHOSEN.search(body):
            names.add(name)
    return names


def _card_power_var(class_name: str) -> tuple[str, int] | None:
    """卡的 ``CanonicalVars`` 里那个 ``PowerVar``：``(能力 id, 层数)``。

    ``IChoosable.OnChosen`` 施加的就是它::

        await PowerCmd.Apply<MindRotPower>(ctx, Owner.Creature,
                                           DynamicVars["MindRotPower"].IntValue,
                                           Owner.Creature, this);

    读源码而不是读 ``cards_source.json``：抽取器不该依赖另一份抽取产物
    （那会让"卡数据错"与"招式数据错"互相掩盖）。
    """
    path = CARD_DIR / f"{class_name}.cs"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = POWER_VAR.search(text)
    if match is None:
        return None
    power = _snake(match.group(1).removesuffix("Power"))
    return power, int(float(match.group(2)))


def extract_choose_sets(source: str, body: str,
                        choice_helpers: set[str] | None = None) -> dict | None:
    """抽出"玩家从临时卡里二选一"的载荷（引擎的 ``effects_by_counter``）。

    形状（``KnowledgeDemon.cs:36-52`` 与 ``:157-176``）::

        private static readonly int[] _disintegrationDamageValues = new int[3] { 6, 7, 8 };
        private static readonly IReadOnlyList<IReadOnlyList<IChoosable>> _curseOfKnowledgeSets =
            new …(new IReadOnlyList<IChoosable>[3] {
                new …(new IChoosable[2] { ModelDb.Card<Disintegration>(), ModelDb.Card<MindRot>() }),
                new …(new IChoosable[2] { ModelDb.Card<Disintegration>(), ModelDb.Card<Sloth>() }),
                new …(new IChoosable[2] { ModelDb.Card<Disintegration>(), ModelDb.Card<WasteAway>() }) });

        // 招式体
        int disintegrationDamage = _disintegrationDamageValues[CurseOfKnowledgeCounter];
        … await CardSelectCmd.FromChooseACardScreen(…) …
        if (cardModel != null) await ((IChoosable)cardModel).OnChosen();
        …
        if (base.CombatState.IsLiveCombat()) CurseOfKnowledgeCounter++;

    ⚠️ 保守到底：任何一步对不上（组数 ≠ 数值个数、候选卡读不到 ``PowerVar``、
    找不到计数器名）就返回 ``None``，让这一招保持"未识别"被报出来。
    猜一个候选集合的后果是**偷偷改掉玩家的选项**，比残缺危险得多。

    ⚠️ **两个方法体都要看**：真机把选择写在**辅助方法** ``ChooseCurse`` 里
    （``FromChooseACardScreen`` / ``OnChosen`` 在那里），而招式体
    ``CurseOfKnowledgeMove`` 只负责 `foreach … ChooseCurse(target)` 与递增计数器。
    只看招式体会找不到选择命令；只看文件又会把别的怪误判进来 ——
    所以"选择命令"扫**整个文件**，"计数器递增"扫**招式体**。

    ⚠️ 范围：真机对**每个 target** 各跑一次 ``ChooseCurse``。单人只有一个玩家，
    所以这里是 1 次；多人（S08）需要按玩家逐个挂起，届时这条要重做。
    """
    helpers = choice_helpers if choice_helpers is not None else _choice_helper_names(source)
    # 这一招必须**真的**会问到玩家：要么选择命令写在招式体里，要么招式体调用了
    # 那个含选择命令的辅助方法。只扫整个文件是不够的（见 `_choice_helper_names`）。
    asks_player = bool(CHOOSE_SCREEN.search(body)) or any(
        re.search(rf"\b{re.escape(name)}\s*\(", body) for name in helpers)
    if not asks_player:
        return None
    groups_raw = CHOICE_GROUP.findall(source)
    counter_match = SETS_COUNTER.search(source)
    values_match = INT_ARRAY.search(source)
    if not groups_raw or counter_match is None or values_match is None:
        return None
    # ⚠️ 必须规范成**登记名**（`_curseOfKnowledgeCounter`）：源码里写的是属性名
    # `CurseOfKnowledgeCounter`，而引擎查的是 `enemy.ai_counters` 的键。
    counter = _counter_key(counter_match.group(1))
    damage_values = [int(v) for v in re.findall(r"-?\d+", values_match.group(1))]
    if len(groups_raw) != len(damage_values) or not damage_values:
        return None
    groups: list[dict] = []
    for index, (size, cards_source) in enumerate(groups_raw):
        names = CARD_MODEL.findall(cards_source)
        if len(names) != int(size) or not names:
            return None
        choices: list[dict] = []
        for class_name in names:
            var = _card_power_var(class_name)
            if var is None:
                return None
            power, amount = var
            if class_name == "Disintegration":
                # 这一档的伤害由 `_disintegrationDamageValues[counter]` **运行时**改写
                # （`cardModel2.DynamicVars["DisintegrationPower"].BaseValue = …`）。
                amount = damage_values[index]
            choices.append({
                "card": _snake(class_name),
                "effects": [{"op": "apply_power", "power": power,
                             "amount": amount, "target": "enemy"}],
            })
        groups.append({"value": index,
                       "effects": [{"op": "choose_one", "choices": choices}]})
    return {
        "counter": counter,
        "groups": groups,
    }


#: 招式体里的实例计数器自增：``CurseOfKnowledgeCounter++``。
INCREMENT = re.compile(r"\b([A-Za-z_]\w*)\s*\+\+")


def _counter_key(name: str) -> str:
    """源码里的属性名 → ``monster_ai.COUNTER_SOURCES`` 的**登记名**。

    ``CurseOfKnowledgeCounter``（属性）与 ``_curseOfKnowledgeCounter``（字段）
    只差一个下划线与首字母大小写，而引擎查 ``enemy.ai_counters`` 用的是**登记名**。
    不规范化的话，``effects_by_counter`` 查表会永远落到默认值 0 ——
    也就是**每一轮都给第一组载荷**，日志看起来完全正常（实测踩过）。
    """
    from sts2_sim import monster_ai
    for key in monster_ai.COUNTER_SOURCES:
        if key.lstrip("_").lower() == name.lstrip("_").lower():
            return key
    return name


def counter_increment(body: str) -> tuple[str, bool]:
    """招式体里自增的实例计数器：``(登记名, 是否只在战斗仍进行时递增)``。

    只认 ``monster_ai.COUNTER_SOURCES`` 里登记过的名字 —— 否则循环变量
    ``i++`` / ``list`` 之类也会被当成计数器，而"招式跑完把 i 加一"
    没有任何可见后果，只是让计数器表里多一个垃圾键。

    ⚠️ 源码里的自增写的是**属性名**（``CurseOfKnowledgeCounter++``），
    而登记表用的是**字段名**（``_curseOfKnowledgeCounter``）—— 两者只差一个
    下划线与首字母大小写。所以这里按"去掉前导下划线后**大小写无关**"匹配，
    匹配上就返回**登记表里的键**（引擎查 `ai_counters` 用的就是它）。
    直接拼字符串会静默返回空串（实测踩过）。

    ⚠️ 守卫也要照抄：``KnowledgeDemon`` 写的是
    ``if (base.CombatState.IsLiveCombat()) CurseOfKnowledgeCounter++;``，
    而 ``TestSubject`` 是裸的 ``Respawns++;``。战斗已结束时还记账，
    会让"第 3 次复活"这类分支在战斗结束后被多推一格。
    """
    from sts2_sim import monster_ai
    for name in INCREMENT.findall(body or ""):
        for key in monster_ai.COUNTER_SOURCES:
            if key.lstrip("_").lower() != name.lower():
                continue
            guarded = re.search(
                rf"IsLiveCombat[^{{}}]*\{{[^}}]*\b{re.escape(name)}\s*\+\+", body or "")
            return key, bool(guarded)
    return "", False

def extract_monster(path: pathlib.Path) -> dict | None:
    source = path.read_text(encoding="utf-8", errors="replace")
    machine = method_body(source, "GenerateMoveStateMachine")
    if machine is None:
        return None
    consts = collect_constants(source)

    # 状态 → 处理函数名（`new MoveState("ID", Handler, intent…)`）
    handlers: dict[str, str] = {}
    for match in re.finditer(
            r'new\s+MoveState\s*\(\s*"([^"]+)"\s*,\s*(\w+)', machine):
        handlers[match.group(1)] = match.group(2)

    # ⭐ "玩家二选一"的辅助方法名，每只怪算一次（见 `_choice_helper_names`）。
    choice_helpers = _choice_helper_names(source)
    moves: dict[str, dict] = {}
    for handler in sorted(set(handlers.values())):
        body = method_body(source, handler)
        if body is None:
            continue
        plain, used = inline_constants(body, consts)
        # `damage_from_intent=True`：招式的攻击力由**意图**给出（`GetValueIfAscension`），
        # 招式函数体里的 `DamageCmd.Attack(...)` 只是把那个数转成一次攻击。
        # 这里若再抽一份，就会多出一个可能和进阶档位对不上的数（见
        # `sts2_sim.content._with_intent_damage`）；卡牌路径则相反，必须抽出来。
        effects, unsupported, choices = extract_effects(
            plain, {}, damage_from_intent=True)
        # 进阶值不同的常量 → 再抽一份"进阶版"
        asc_effects: list[dict] = []
        varying = ascension_variants(consts, used)
        if varying:
            bumped = dict(consts)
            for name in varying:
                bumped[name] = {**consts[name], "normal": consts[name]["ascension"]}
            asc_plain, _ = inline_constants(body, bumped)
            asc_effects, _, _ = extract_effects(
                asc_plain, {}, damage_from_intent=True)
        choose_sets = extract_choose_sets(source, body, choice_helpers)
        unsupported_out = sorted(set(unsupported))
        if choose_sets:
            # `choose_one` 已经把"造卡 → 选一张 → OnChosen"整段表达出来了，
            # 所以这段代码里的 `list.Add` 不再是"未识别的写法"
            # （那是 `List<Task> list` 收集**多目标**的并行选择，单人只有一个目标）。
            unsupported_out = [item for item in unsupported_out
                               if "list.Add" not in item]
        moves[handler] = {
            "effects": effects,
            "unsupported": unsupported_out,
            "choices": choices,
            "constants_used": sorted(used),
            "ascension_varying": sorted(varying),
            "ascension_effects": asc_effects,
            # ⭐ 自爆 / 逃跑：真机在招式结尾写 `CreatureCmd.Kill(base.Creature)` /
            # `CreatureCmd.Escape(base.Creature)`。这两个**不是普通效果**，
            # 而是"这招打完这只怪就没了"，必须单独成标志 ——
            # 少了它，下一回合会去 roll 后继并抛"没有后继状态"
            # （实测 `gas_bomb` 直接把 Run 层炸掉）。
            "kills_self": bool(re.search(r"CreatureCmd\.Kill\(\s*base\.Creature", body)),
            "escapes": bool(re.search(r"CreatureCmd\.Escape\(\s*base\.Creature", body)),
        }
        if choose_sets:
            moves[handler]["effects_by_counter"] = choose_sets
        increment, live_only = counter_increment(body)
        if increment:
            # ⭐ 招式**走完**要自增的实例计数器（真机写在招式函数末尾：
            # `CurseOfKnowledgeCounter++` / `Respawns++`）。引擎在
            # `core._finish_enemy_move` 里递增 —— 正常跑完与挂起续跑两条路径都走它。
            moves[handler]["counter_increment"] = increment
            if live_only:
                moves[handler]["counter_increment_live_only"] = True
    if not moves:
        return None
    return {"eid": _snake(path.stem), "class": path.stem, "moves": moves}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从反编译源码抽取怪物招式效果")
    parser.add_argument("--src", default=str(DECOMPILED))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)

    root = pathlib.Path(args.src)
    records = []
    report: collections.Counter = collections.Counter()
    for path in sorted(root.glob("*.cs")):
        record = extract_monster(path)
        if record is None:
            report["没有状态机（跳过）"] += 1
            continue
        with_effects = sum(1 for m in record["moves"].values() if m["effects"])
        report["有状态机的怪"] += 1
        report["招式总数"] += len(record["moves"])
        report["有效果的招式"] += with_effects
        report["空效果招式"] += len(record["moves"]) - with_effects
        for move in record["moves"].values():
            for command in move["unsupported"]:
                report[f"未支持::{command}"] += 1
            if move["ascension_varying"]:
                report["含进阶差异的招式"] += 1
        records.append(record)

    if not args.report:
        pathlib.Path(args.out).write_text(
            json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"写出 {args.out}（{len(records)} 只怪）")
    print()
    for key, value in report.most_common(24):
        print(f"  {value:5d}  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
