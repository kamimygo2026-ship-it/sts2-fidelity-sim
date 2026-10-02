"""模拟器体检：一条命令跑完全链路，回答"现在这套内容下模拟器工作是否正常"。

导入了真实内容之后**第一件事**就是跑它——内容换了，很多东西会以你没预料到的方式坏掉
（起始牌组缺牌、卡牌解析出空效果、词表错位、地图不连通……）。

    python tools/healthcheck.py                    # 用当前生效的内容
    python tools/healthcheck.py --content data/content/0.105.0
    python tools/healthcheck.py --combats 200 --runs 50

退出码 0 = 全部通过；非 0 = 有失败项（会逐条列出）。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sts2_sim import (  # noqa: E402
    Action, RuleBot, RunBot, RunEnv, SpireEnv, content as content_module,
    generate_map, legal_actions, lint, observe, reroll_hidden, start_combat,
    unverified_report,
)
from sts2_sim.content import (  # noqa: E402
    CARD_DB, DEFAULT_CHARACTER, ENEMY_DB, ENCOUNTERS, STARTING_DECK,
)
from sts2_sim.core import Hidden  # noqa: E402
from sts2_sim.env import STAGE_COMBAT, STAGE_META  # noqa: E402
from sts2_sim.featurize import (  # noqa: E402
    MAX_ACTIONS, batch_encode, encode_observation, shapes, vocab_fingerprint,
)
from sts2_sim.rng import RngSet  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []

#: 动态挑选的遭遇（不写死怪物 id）
_ENCOUNTER: tuple[str, ...] = ()
_SOLO: tuple[str, ...] = ()


def check(name: str, condition: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(condition), detail))
    mark = "✅" if condition else "❌"
    print(f"  {mark} {name}" + (f"  —— {detail}" if detail else ""))
    return bool(condition)


def warn(name: str, detail: str = "") -> None:
    """保真度缺口：**已知且已上报**，不计入失败，但绝不能忽略。"""
    print(f"  ⚠️ {name}" + (f"  —— {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def pick_encounter(kind: str = "monster", max_members: int = 2,
                   max_total_hp: int = 250, min_total_hp: int = 0) -> tuple[str, ...]:
    """从当前遭遇表里挑一场**规模与血量都可控**的战斗。

    **不要写死怪物 id**——换了内容表之后那些 id 就不存在了，整段体检会直接崩掉。
    必须按血量过滤：数据里有 9999 血的占位实体，挑到它们战斗永远打不完。

    ``min_total_hp`` 用于 SL 检查：太弱的怪 2 回合就死，轨迹太短，
    "重开后轨迹一致"与"重掷会改变轨迹"都会失去鉴别力（真实踩过的坑）。
    """
    table = ENCOUNTERS.get(kind) or ()

    def total_hp(encounter: tuple[str, ...]) -> int:
        return sum(ENEMY_DB[e].hp[1] for e in encounter if e in ENEMY_DB)

    candidates = [e for e in table
                  if 0 < len(e) <= max_members
                  and min_total_hp <= total_hp(e) <= max_total_hp]
    if not candidates:
        candidates = [e for e in table if total_hp(e) <= max_total_hp] or list(table)
    if not candidates:
        first = next(iter(ENEMY_DB), None)
        return (first,) if first else ()
    return min(candidates, key=lambda e: (len(e), total_hp(e)))


# ==========================================================================
def check_content() -> None:
    global _ENCOUNTER, _SOLO
    _ENCOUNTER = pick_encounter("monster", 2)
    _SOLO = pick_encounter("monster", 1) or _ENCOUNTER
    section("1 · 内容表")
    print(f"  来源：{content_module.CONTENT_SOURCE}")
    print(f"  卡牌 {len(CARD_DB)} 张 | 怪物 {len(ENEMY_DB)} 个")
    fingerprint = vocab_fingerprint()
    print(f"  词表指纹：{fingerprint}")

    check("卡牌数 > 0", len(CARD_DB) > 0)
    check("怪物数 > 0", len(ENEMY_DB) > 0)

    problems = lint()
    check("静态检查通过", not problems, f"{len(problems)} 个问题" if problems else "")
    for problem in problems[:10]:
        print(f"      · {problem}")

    missing = [cid for cid in STARTING_DECK if cid not in CARD_DB]
    check("起始牌组完整", not missing, f"缺 {missing}" if missing else "")

    # ⚠️ 口径与 `sts2_sim/eligibility.py` **同源**（审计 F11：工具之间各报一套
    # 数字，最后没人知道哪个是真的）。这里只报"可打出的牌却没有任何效果"，
    # 诅咒/状态这类"本来就该没效果"的牌不算缺口。
    from sts2_sim import eligibility
    empty = [cid for cid, card in CARD_DB.items()
             if eligibility.R_NO_EFFECT in eligibility.card_reasons(cid)]
    if empty:
        warn(f"{len(empty)}/{len(CARD_DB)} 张**可打出**的牌没有任何效果",
             "属已知缺口：要么抽取器解析不了，要么效果超出 DSL 的表达力")
    else:
        check("所有可打出的牌都有可执行效果", True)

    report = unverified_report()
    print(f"  ⚠️ 未对拍：卡牌 {len(report['cards'])} / 怪物 {len(report['enemies'])}"
          f"（训练前应对拍，见 docs/06 §6.5）")
    admission = eligibility.report(DEFAULT_CHARACTER)
    print(f"  内容准入（eligibility）：引擎可执行 {admission['cards_admitted']}"
          f"/{admission['cards_total']} 张 | "
          f"{DEFAULT_CHARACTER} 可抽池 {admission['draft_pool']} 张")
    print(f"    拒绝原因：{admission['reject_reasons']}")

    # 怪物 AI 的保真度。⚠️ 以前这里用 `e.cycle` 判断"忠实 vs 近似"，
    # 而 115 只怪现在**全部**挂上了源码状态机 —— 那个口径早已失效（审计 F11）。
    # 正确的分法：有真机状态机 → 忠于转移结构；没有才退回 cycle / 权重近似。
    with_machine = [e.eid for e in ENEMY_DB.values()
                    if getattr(e, "ai_record", None)]
    fallback_cycle = [e.eid for e in ENEMY_DB.values()
                      if not getattr(e, "ai_record", None) and e.cycle]
    fallback_approx = [e.eid for e in ENEMY_DB.values()
                       if not getattr(e, "ai_record", None) and not e.cycle]
    print(f"  出招状态机：真机状态机 {len(with_machine)} / {len(ENEMY_DB)} 只 | "
          f"退回 cycle {len(fallback_cycle)} | 退回权重近似 {len(fallback_approx)}")
    if fallback_cycle or fallback_approx:
        warn("有怪物没有源码状态机",
             f"cycle {fallback_cycle[:4]} 近似 {fallback_approx[:4]}")
    print("    ⚠️ 挂上状态机 ≠ 运行时条件正确：专属 flags/counters 与招式副作用"
          "需要逐条用例（审计 F05，现已有 Fabricator 回归）")

    from sts2_sim.powers import IMPLEMENTED
    unmodeled = sorted({pid for e in ENEMY_DB.values()
                        for pid, _amount in e.innate_powers
                        if pid not in IMPLEMENTED})
    if unmodeled:
        warn(f"{len(unmodeled)} 种怪物固有属性未实现", f"{unmodeled[:8]}")
    else:
        check("怪物固有属性全部已实现", True)

    # ⭐ **出招表闭包**（docs/13 §7）：状态机里指向的每个 move 状态都必须能落到
    # 一条招式。缺口的后果是"这只怪出场时抛错打断整局"，而且可能几十万步后才发生。
    from sts2_sim import content as sim_content
    gaps = sim_content._enemy_move_gaps()
    if gaps:
        warn(f"{len(gaps)} 只怪的招式表有闭包缺口", f"{sorted(gaps)[:5]}")
    else:
        check("出招表闭包完整（状态机指向的招式都存在）", True)

    from sts2_sim.content import _LAST_ENCOUNTER_LOAD
    print(f"  遭遇表：monster {len(ENCOUNTERS.get('monster', ()))} | "
          f"elite {len(ENCOUNTERS.get('elite', ()))} | "
          f"boss {len(ENCOUNTERS.get('boss', ()))}"
          f"  （排除：事件遭遇 {_LAST_ENCOUNTER_LOAD.get('skipped_event', 0)}、"
          f"成员缺失 {_LAST_ENCOUNTER_LOAD.get('skipped_missing_monster', 0)}、"
          f"超出敌人槽位 {_LAST_ENCOUNTER_LOAD.get('skipped_exceeds_slots', 0)}）")


def check_map() -> None:
    section("2 · 地图（三幕 / 四张图）")
    ok = True
    # 四张幕地图都要连通、都要能走到 Boss —— 也是"四幕地图存在"的最小证据。
    acts = ("overgrowth", "underdocks", "hive", "glory")
    for act in acts:
        for seed in range(10):
            graph = generate_map(seed, act)
            boss = graph.by_id(graph.boss)
            seen = {graph.ancient}
            frontier = [graph.ancient]
            while frontier:
                node = graph.by_id(frontier.pop())
                for child in node.edges:
                    if child not in seen:
                        seen.add(child)
                        frontier.append(child)
            if boss.node_id not in seen or len(seen) != len(graph):
                ok = False
                break
    check("4 张幕 × 10 seed 全部从远古点连通到 Boss（无孤立点）", ok)
    for act in acts:
        graph = generate_map(0, act)
        print(f"  {act:11s} 层数 {graph.rows:2d} | 节点 {len(graph):3d} 个 | "
              f"首行 {len(graph.start)} 个")


def check_combat(combats: int) -> None:
    section(f"3 · 战斗层（{combats} 场）")
    bot = RuleBot()
    wins = 0
    turns = 0
    problems: list[str] = []
    for seed in range(combats):
        env = SpireEnv(seed=seed, encounter=_ENCOUNTER, attempt_budget=0)
        obs, info = env.reset()
        steps = 0
        while info["stage"] != "done" and steps < 400:
            state = env.raw_state
            if not (0 <= state.player.hp <= state.player.max_hp):
                problems.append(f"seed={seed} HP 越界")
            if state.energy < 0:
                problems.append(f"seed={seed} 能量为负")
            if len(state.all_piles()) != len(STARTING_DECK) + state.added_cards:
                problems.append(f"seed={seed} 卡牌不守恒")
            action = bot.act(obs, info["legal_actions"])
            obs, _r, done, _t, info = env.step(action)
            steps += 1
            if done:
                break
        wins += info["phase"] == "won"
        turns += env.raw_state.turn
        if steps >= 400:
            problems.append(f"seed={seed} 疑似死循环")
    check("不变量无违规", not problems, f"{problems[:3]}" if problems else "")
    check("全部正常结束", len(problems) == 0)
    print(f"  规则 bot 胜率 {wins / combats:.0%} | 平均回合 {turns / combats:.1f}")


def check_run_layer(runs: int) -> None:
    section(f"4 · Run 层（{runs} 局）")
    bot = RunBot()
    wins = 0
    floors = []
    problems: list[str] = []
    for seed in range(runs):
        env = RunEnv(seed=seed, attempt_budget=1)
        step = env.reset()
        steps = 0
        while step.phase not in ("won", "lost") and steps < 2000:
            state = env.raw_state
            if not (0 <= state.player.hp <= state.player.max_hp):
                problems.append(f"seed={seed} HP 越界")
            action = bot.act(step)
            if action is None:
                problems.append(f"seed={seed} 无可用动作")
                break
            step, _r, done, _t, _info = env.step(action)
            steps += 1
            if done:
                break
        wins += step.phase == "won"
        floors.append(env.raw_state.floor)
    check("完整 run 均正常结束", not problems, f"{problems[:3]}" if problems else "")
    check("有可用的 run 循环", len(floors) == runs)
    print(f"  通关率 {wins / runs:.0%} | 平均到达第 "
          f"{sum(floors) / max(1, len(floors)):.1f} 层")


def check_observation() -> None:
    section("5 · 观测与特征化")
    state = start_combat(STARTING_DECK, _SOLO, seed=7)
    before = observe(state).to_json()
    reroll_hidden(state, 999_999)
    state.hidden.move_queue = ["chomp"]
    check("观测对隐藏信息不变", before == observe(state).to_json())

    order_before = encode_observation(observe(state), legal_actions(state))["pile_counts"]
    state.draw_pile.reverse()
    order_after = encode_observation(observe(state), legal_actions(state))["pile_counts"]
    check("牌堆顺序不进入编码", bool((order_before == order_after).all()))

    encoded = encode_observation(observe(state), legal_actions(state))
    for key, shape in shapes().items():
        if encoded[key].shape != shape:
            check(f"形状 {key}", False, f"{encoded[key].shape} != {shape}")
            return
    check("全部张量形状符合声明", True)
    check("动作数未超容量", int(encoded["act_mask"].sum()) <= MAX_ACTIONS)
    batch = batch_encode([(observe(state), legal_actions(state))])
    check("批量编码可用", batch["token_num"].shape[0] == 1)


def _scripted_trace(env: SpireEnv, steps: int = 14) -> list[tuple]:
    """跑一段与观测无关的固定脚本，记录可见轨迹。

    注意：**开局手牌在 ``start_combat`` 时就抽好了**，它不依赖后来的隐藏状态，
    所以"重掷后手牌不同"是个错误断言。真正依赖隐藏 RNG 的是后续的洗牌与敌人选招，
    因此必须比较**一段轨迹**而不是第一帧。
    """
    trace = []
    obs, info = env._observe(), env._info()
    for _ in range(steps):
        if info["stage"] != STAGE_COMBAT:
            break
        trace.append((
            obs.turn,
            tuple(c.cid for c in obs.hand),
            tuple((e.hp, e.intent_mid, e.intent_value) for e in obs.enemies),
        ))
        action = next((a for a in info["legal_actions"]
                       if a.kind == "play_card" and a.hand_index == 0),
                      Action("end_turn"))
        obs, _r, done, _t, info = env.step(action)
        if done:
            break
    return trace


def check_sl() -> None:
    section("6 · SL（重开语义 + 最优解）")
    # 挑一场**打得起劲**的战斗：太弱的怪 2 回合就死，对拍会失去鉴别力
    encounter = pick_encounter("monster", 2, 250, 80) or _ENCOUNTER
    env = SpireEnv(seed=1234, encounter=encounter, attempt_budget=3)
    obs, info = env.reset()
    first_hand = tuple(c.cid for c in obs.hand)

    check("combat 阶段有 restart 动作",
          "restart" in {a.kind for a in info["legal_actions"]})
    check("观测携带预算", obs.attempts_left == 3)

    baseline = _scripted_trace(env)
    check("轨迹长度足够（有鉴别力）", len(baseline) > 8, f"{len(baseline)} 步")

    _obs, _r, _d, _t, info = env.step(Action("restart"))
    check("重开后仍面对同一副牌序",
          tuple(c.cid for c in env._observe().hand) == first_hand)
    # ⭐ 知识检查必须紧跟在重开之后：再跑一遍会让第二遍也结束、知识变成 2 条
    check("重开后知识被记录", len(env._observe().prior_attempts) == 1)
    check("预算正确扣减", env.attempts_left == 2)
    again = _scripted_trace(env)
    check("重开后整条轨迹一致（SL 语义成立）", baseline == again)

    # 反向验证：故意让 restore 重掷隐藏 RNG，轨迹必须改变。
    # 没有这一条，"重开轨迹一致"可能只是恒真命题。
    env2 = SpireEnv(seed=1234, encounter=encounter, attempt_budget=3)
    env2.reset()
    env2._node_snapshot.state.hidden = Hidden(RngSet(4242))
    env2.restart_attempt()
    check("故意重掷会改变轨迹（对拍有鉴别力）",
          _scripted_trace(env2) != baseline)

    # 预算耗尽 → 保留最优解
    env3 = SpireEnv(seed=5, encounter=_SOLO, attempt_budget=1)
    obs3, info3 = env3.reset()
    steps = 0
    while info3["stage"] == STAGE_COMBAT and steps < 200:
        action = next((a for a in info3["legal_actions"]
                       if a.kind == "play_card" and a.hand_index == 0),
                      Action("end_turn"))
        obs3, _r, done, _t, info3 = env3.step(action)
        steps += 1
        if done:
            break
    check("战斗结束后进入 meta 等待决策", info3["stage"] == STAGE_META)
    _obs3, reward, done3, _t3, info4 = env3.step(Action("accept"))
    check("接受后进入 done 并返回最优解回报", done3 and info4["stage"] == "done",
          f"reward={reward:.3f} kept={info4['kept_attempt']}")


def check_training_stack() -> None:
    section("7 · 训练栈（需要 torch）")
    try:
        import torch
        from sts2_rl.nets import PolicyValueNet
        from sts2_rl.ppo import CombatPool, to_torch
    except Exception as exc:  # pragma: no cover
        check("torch 可用", False, str(exc))
        return
    check("torch 可用", True, torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(64, 2, 2).to(device)
    pool = CombatPool(8, base_seed=1, attempt_budget=0)
    batch = to_torch(pool.current_batch(), device)
    logits, value = net(batch)
    check("前向输出有限", bool(torch.isfinite(logits).all()))
    check("价值头输出有限", bool(torch.isfinite(value).all()))

    action, logp, val = net.act(batch)          # net.act 是 no_grad 的，仅用于采样
    legal = batch["act_mask"].gather(-1, action.unsqueeze(-1)).squeeze(-1)
    check("采样动作全部合法", bool((legal > 0.5).all()))

    # 反向传播必须**重新前向**（no_grad 出来的 logp 没有 grad_fn）
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    logits2, value2 = net(batch)
    log_probs = torch.log_softmax(logits2, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
    chosen = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
    loss = (-chosen.mean() - 0.01 * entropy
            + 0.5 * torch.nn.functional.mse_loss(value2[:, 0],
                                                 torch.zeros_like(value2[:, 0])))
    check("损失有限", bool(torch.isfinite(loss)))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    bad = [n for n, p in net.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    check("反传梯度有限（-inf 掩码坑）", not bad, f"{bad[:3]}" if bad else "")
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    bad = [n for n, p in net.named_parameters() if not torch.isfinite(p).all()]
    check("更新一步后参数有限", not bad, f"{bad[:3]}" if bad else "")
    print(f"  设备 {device} | 参数 {net.n_parameters():,}")


# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="模拟器体检")
    parser.add_argument("--content", default="", help="内容表目录（JSON）")
    parser.add_argument("--combats", type=int, default=120)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args(argv)

    if args.content:
        from sts2_sim.featurize import configure_content
        info = configure_content(args.content)
        print(f"[内容] 已加载 {args.content}：{info}")

    sections = [
        ("内容表", lambda: check_content()),
        ("地图", lambda: check_map()),
        ("战斗层", lambda: check_combat(args.combats)),
        ("Run 层", lambda: check_run_layer(args.runs)),
        ("观测", lambda: check_observation()),
        ("SL", lambda: check_sl()),
        ("训练栈", lambda: check_training_stack()),
    ]
    for title, fn in sections:
        try:
            fn()
        except Exception:
            check(f"{title} 段抛出异常", False)
            traceback.print_exc()

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{'=' * 70}")
    print(f"体检结果：{len(RESULTS) - len(failed)}/{len(RESULTS)} 项通过")
    if failed:
        print("失败项：")
        for name in failed:
            print(f"  ❌ {name}")
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
