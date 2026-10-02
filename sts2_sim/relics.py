"""遗物求值（``docs/09`` L5）。

数据来自 ``tools/extract_relics.py``，内容层加载进 :data:`sts2_sim.content.RELICS`；
这里只负责**把「遗物声明的增量 + 它的触发条件」算成此刻的具体数值**。

三条设计约束（都是踩出来的）
--------------------------

1. **守卫是"满足则*不*生效"**。源码一律是 early-return 形态：

   .. code-block:: csharp

      if (playerCombatState.TurnNumber > 1) return cardsToDraw;   // 准备背包
      return cardsToDraw + base.DynamicVars.Cards.IntValue;

   即 ``turn_gt 1`` 表示**只在第 1 回合**多抽 2 张。把它读成"第 2 回合起生效"
   会让准备背包完全反过来，而且数值看着还挺合理。

2. **算不出的条件一律不采纳**，不做"近似就上"。``TurnsSeen`` / ``KindleCount``
   这类跨回合计数器活在 Run 层，战斗内复刻不出来 —— 宁可少一个遗物，
   也不能让"每 3 回合多抽 1 张"变成"每回合多抽 1 张"。

3. **房间范围决定一切**。``AfterRoomEntered`` 在**所有**房间类型都会触发
   （源码 ``AbstractModel.cs:64``），所以"什么时候生效"完全取决于函数体里的
   ``room is CombatRoom`` / ``RoomType.Boss`` / ``RoomType.Elite`` 判断。
   抽取器把这个判断抽成 ``scope.room``，这里只认战斗类的取值。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .content import RELICS, Effect

#: 战斗内**算得出**的条件种类。
#:
#: * ``turn_*`` —— 回合数（``PlayerCombatState.TurnNumber``）
#: * ``card_type`` —— 打出的牌是不是某一类（``cardPlay.Card.Type``）。
#:   `GamePiece`（仅能力牌 → 抽牌）/ `LostWisp` / `Permafrost` / `RainbowRing`
#:   整个行为就是"只在某类牌上触发"；丢掉它等于**每张牌都触发**（静默变强）。
TURN_GUARDS = frozenset({"turn_gt", "turn_gte", "turn_lt", "turn_lte",
                         "turn_eq", "turn_ne"})
CARD_TYPE_GUARD = "card_type"
#: ⭐ 「每打出 N 张某类牌触发一次」（``Kunai`` 每 3 张攻击 +1 敏捷 /
#: ``Shuriken`` 每 3 张攻击 +1 力量 / ``LetterOpener`` 每 3 张技能造成 5 点伤害 /
#: ``OrnamentalFan`` 每 3 张攻击 +4 格挡）。
#:
#: 守卫的 ``value`` 是 ``{"counter": "attack"|"skill"|"card", "n": N, "var": "Cards"}``
#: —— ``n`` 与变量名都由**抽取器**从 ``DynamicVars`` 解析好（见
#: ``tools/extract_relics.modulo_guard``），运行期只需要计数。
#: 计数由调用方经 ``played`` 传进来（core 的 ``_fire_after_card_played`` 把
#: 本回合已打出的张数传下来），**包含当前这张牌** ——
#: 与源码"先 ``Counter++`` 再判取模"完全一致。
EVERY_N_GUARD = "every_n_turn"

#: 引擎**算得出**的守卫种类。``content._relic_hook`` 用它把"引擎求值不了的守卫"
#: 挡在门外 —— 求值不了的守卫会被 :func:`hooks_at` **静默跳过**
#: （``_applies(...) is not True``），而报告会说这个遗物"完全复刻"。
EVALUABLE_GUARDS = TURN_GUARDS | {CARD_TYPE_GUARD, EVERY_N_GUARD}

#: 抽取器标成战斗范围的取值。``BeforeCombatStart`` 本身只在战斗里跑，
#: 所以它的 ``any`` 也当成"任意战斗"。
COMBAT_ROOMS = frozenset({"combat", "elite", "boss"})

#: 房间种类 → 它算不算"精英 / Boss 战"。遗物要按这个筛。
ROOM_ALIASES = {
    "monster": "combat", "elite": "elite", "boss": "boss",
    "normal": "combat", "combat": "combat",
}


def _applies(guards: Sequence[tuple[str, object]], turn: int,
             card_type: str = "",
             played: dict[str, int] | None = None) -> bool | None:
    """这些条件是否全部满足（效果应当生效）？算不出返回 ``None``。

    ⚠️ **两种极性必须分开处理**，这是整批遗物最容易整体反号的地方：

    * ``skip_when``（排除式守卫）：源码写 ``if (TurnNumber > 1) return amount;``，
      即"条件成立则**不**生效"。数值钩子（``ModifyMaxEnergy``/``ModifyHandDraw``）
      全是这个形态。
    * ``require``（包含式条件）：源码写 ``if (… && TurnNumber <= 1) { 给能量 }``，
      即"条件成立**才**生效"。效果钩子（``BeforeSideTurnStart`` /
      ``AfterSideTurnStart`` …）几乎都是这个形态。

    把包含式条件当成排除式，会让提灯在**第 2 回合起**每回合给能量、
    弹珠袋**永远不上易伤** —— 数值全错，日志却完全正常。

    ``card_type`` 是**打出的那张牌**的类型（``attack`` / ``skill`` / ``power``）。
    只有带 ``card_type`` 守卫的钩子才需要它；没有该守卫时传空串即可。

    ``played`` 是**本回合已打出的张数**（``{"attack": N, "skill": N, "card": N}``），
    只有 :data:`EVERY_N_GUARD` 需要它。**必须包含当前这张牌** ——
    真机是"先 ``Counter++`` 再判 ``% N == 0``"。没传（``None``）时算不出 → 返回
    ``None``，由调用方跳过（安全方向：绝不默认命中）。
    """
    satisfied = True
    for guard in guards:
        kind, value, polarity = _guard_parts(guard)
        if kind == CARD_TYPE_GUARD:
            if not card_type:
                # 钩子要求"某类牌"，但调用方没给出牌型（例如不是打牌触发的时机）
                # → 算不出，交给调用方跳过。**不能**默认成"命中"。
                return None
            hit = str(value).lower() == card_type.lower()
        elif kind == EVERY_N_GUARD:
            if not isinstance(value, dict) or played is None:
                return None
            counter = str(value.get("counter") or "")
            every = value.get("n")
            count = played.get(counter)
            if count is None or not isinstance(every, int) or every <= 0:
                return None
            # 一张都没打时取模必然为 0，会**白送一次**触发 —— 必须显式排掉。
            hit = count > 0 and count % every == 0
        elif kind in TURN_GUARDS and isinstance(value, int):
            hit = {
                "turn_gt": turn > value,
                "turn_gte": turn >= value,
                "turn_lt": turn < value,
                "turn_lte": turn <= value,
                "turn_eq": turn == value,
                "turn_ne": turn != value,
            }[kind]
        else:
            return None                       # 跨回合计数器之类 → 算不出
        if polarity == "require" and not hit:
            return False                      # 包含式条件不满足 → 不生效
        if polarity != "require" and hit:
            return False                      # 排除式守卫命中 → 不生效
    return satisfied


def _guard_parts(guard) -> tuple[str, object, str]:
    """兼容两种守卫写法：``(kind, value)`` 与 ``(kind, value, polarity)``。"""
    if len(guard) >= 3:
        return str(guard[0]), guard[1], str(guard[2])
    return str(guard[0]), guard[1], "skip_when"


def _numeric(relics: Iterable[str], field: str, turn: int) -> int:
    """累加所有遗物在 ``field``（``max_energy`` / ``hand_draw``）上的增量。

    ⚠️ 算不出的**整条跳过**并计入 :func:`unevaluable`，不猜、不当 0。
    """
    total = 0
    for rid in relics:
        definition = RELICS.get(rid)
        if definition is None:
            continue
        modifier = getattr(definition, field)
        if modifier is None:
            continue
        _var, sign, guards = modifier
        allowed = _applies(guards, turn)
        if allowed:
            total += sign * int(definition.values.get(_var, 0))
    return total


def energy_bonus(relics: Iterable[str], turn: int) -> int:
    """每回合额外的能量（``ModifyMaxEnergy``）。"""
    return _numeric(relics, "max_energy", turn)


def hand_draw_bonus(relics: Iterable[str], turn: int) -> int:
    """每回合额外的抽牌（``ModifyHandDraw``）。可以是负数（``big_mushroom``）。"""
    return _numeric(relics, "hand_draw", turn)


def room_kind(room: str) -> str:
    return ROOM_ALIASES.get(str(room).lower(), "combat")


#: 引擎认识的时机名，按真机 ``CombatManager.StartTurn`` + ``SetupPlayerTurn`` 的顺序。
#:
#: ⚠️ ``AfterCardPlayed`` / ``AfterSideTurnEnd`` **暂时不在表里**：这两个时机上的
#: 遗物钩子几乎都带条件（牌型 / 取模计数 / 随机目标），而抽取器建不出这些条件 ——
#: 采纳会变成"每张牌都触发"，静默变强。理由与启用前置条件见
#: ``content.HOOK_TIMING`` 的注释与 docs/12 §2.6「遗物口径」。
TIMING_ORDER: tuple[str, ...] = (
    "combat_start",
    "before_side_turn_start",
    "after_energy_reset",
    "after_player_turn_start",
    "after_side_turn_start",
    # 回合内（`GamePiece` / `LostWisp` / `Permafrost` / `RainbowRing` 在这里）
    "after_card_played",
)


def hooks_at(timing: str, relics: Iterable[str], turn: int,
             room: str = "monster", card_type: str = "",
             played: dict[str, int] | None = None
             ) -> tuple[list[Effect], list[str]]:
    """取某个时机下**实际生效**的效果，按房间与回合守卫过滤。

    返回 ``(效果序列, 说明)``。``pantograph``（只在本场是 Boss 战时回血）、
    ``sling_of_courage``（只在精英战给力量）、``lantern``（只在第 1 回合给能量）
    全靠这里区分 —— 守卫一丢，提灯就变成"每回合 +1 能量"。

    ``card_type`` 是**打出的那张牌**的类型，只有 ``after_card_played`` 这类
    时机需要它；带 ``card_type`` 守卫而这里没给牌型时，该钩子**跳过**
    （算不出就不生效，绝不默认命中）。

    ``played`` 见 :func:`_applies`：``after_card_played`` 时机上"每 N 张某类牌"
    的遗物（`Kunai` / `Shuriken` / `LetterOpener` / `OrnamentalFan`）需要它。
    """
    kind = room_kind(room)
    effects: list[Effect] = []
    notes: list[str] = []
    for rid in relics:
        definition = RELICS.get(rid)
        if definition is None:
            continue
        for hook in definition.hooks:
            if hook.timing != timing:
                continue
            if hook.room and hook.room != kind:
                continue
            if _applies(hook.guards, turn, card_type, played) is not True:
                continue
            effects.extend(hook.effects)
            notes.append(f"遗物 {definition.name}：{hook.note}")
    return effects, notes


def apply_hook(state, timing: str, relics: Iterable[str],
               events: list[str], card_type: str = "",
               played: dict[str, int] | None = None) -> list[str]:
    """在正确的时机施加遗物效果，返回说明（调用方已并入日志）。

    只处理**目标为自己或全体敌人**的效果 —— 遗物的开局 / 回合开始效果都是这两类；
    需要选牌的那种（``stone_cracker`` 升级随机牌）在内容层就被挡在外面了。
    """
    from . import core

    effects, notes = hooks_at(timing, relics, getattr(state, "turn", 1),
                              getattr(state, "room", "monster"), card_type,
                              played)
    if not effects:
        return []
    core._apply_effects(state, effects, state.player, -1, events)
    events.extend(notes)
    return notes


def run_effects(timing: str, relic_ids: Iterable[str]) -> tuple:
    """取某个 **Run 层时机**（``obtained`` / ``combat_end`` / ``combat_victory``）的效果。

    与 :func:`hooks_at` 的区别：这些效果作用于**牌组 / 金币 / 最大生命**，
    没有"回合"与"房间"的概念，所以不带战斗内的两个守卫。
    """
    effects: list[Effect] = []
    notes: list[str] = []
    for rid in relic_ids:
        definition = RELICS.get(rid)
        if definition is None:
            continue
        for hook in definition.hooks:
            if hook.timing != timing:
                continue
            if hook.guards and _applies(hook.guards, 1) is not True:
                continue
            effects.extend(hook.effects)
            notes.append(f"遗物 {definition.name}：{hook.note}")
    return tuple(effects), notes


def apply_run_hook(state, timing: str, relic_ids: Iterable[str],
                   events: list[str]) -> int:
    """在 Run 层时机施加遗物效果（拾取 / 战斗结束 / 战斗胜利）。返回结算条数。"""
    from . import runeffects

    effects, notes = run_effects(timing, relic_ids)
    if not effects:
        return 0
    done = runeffects.apply_run_effects(state, effects, events)
    events.extend(notes)
    return done


def dispatch(holder, ctx) -> int:
    """**钩子总线的遗物处理器**，与 :func:`sts2_sim.powers._dispatch` 对称。

    ``holder`` 是玩家（遗物挂在玩家身上），``ctx["hook"]`` 是总线上的钩子名
    （``relic_*``）。返回实际调用了几个遗物效果 —— 总线用它决定要不要记轨迹。
    """
    from . import hooks as hook_bus

    hook_name = ctx.get("hook", "")
    timing = next((t for t, h in hook_bus.RELIC_TIMING_TO_HOOK.items()
                   if h == hook_name), None)
    if timing is None:
        return 0
    state = ctx["state"]
    events = ctx["events"]
    relics = tuple(getattr(state, "relics", ()) or ())
    # ``card_type`` 由调用方（`core._fire_after_card_played`）传进来：
    # "只在打出某类牌时触发"的遗物全靠它区分。
    # ``played_counts`` 同理 —— "每 N 张攻击/技能"的遗物靠它判取模。
    notes = apply_hook(state, timing, relics, events,
                       str(ctx.get("card_type") or ""),
                       ctx.get("played_counts"))
    return len(notes)


def unevaluable(relics: Iterable[str]) -> list[str]:
    """哪些遗物的**改数值**能力在战斗内算不出来（会被静默跳过）。

    ⚠️ 这个函数存在的意义就是"不许静默"：跳过是对的，但必须**可见**。
    """
    blocked: list[str] = []
    for rid in relics:
        definition = RELICS.get(rid)
        if definition is None:
            continue
        for field in ("max_energy", "hand_draw"):
            modifier = getattr(definition, field)
            if modifier is None:
                continue
            _var, _sign, guards = modifier
            if _applies(guards, 1) is None:
                blocked.append(f"{rid}.{field}")
    return blocked


def unimplemented(relics: Iterable[str]) -> list[str]:
    """哪些遗物**还有没实现的钩子**（照实报告，供覆盖率统计）。"""
    out: list[str] = []
    for rid in relics:
        definition = RELICS.get(rid)
        if definition is not None and definition.unmodeled_hooks:
            out.append(rid)
    return out
