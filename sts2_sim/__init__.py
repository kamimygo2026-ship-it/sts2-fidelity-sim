"""尖塔模型 · 抽象模拟器骨架（Python 原型）。

对应 ``docs/02``（模拟器设计）与 ``docs/01``（信息边界 / SL 规则）。

    from sts2_sim import SpireEnv, RuleBot

    env = SpireEnv(seed=7, encounter=("jaw_worm",), attempt_budget=4)
    obs, info = env.reset()
    bot = RuleBot()
    while True:
        action = bot.act(obs, info["legal_actions"])
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break
        if 打得不好 and env.attempts_left:
            env.restart_attempt()      # 同一副牌序，重开再来

⚠️ 骨架的数值与 AI 权重**均未对拍**（``content.unverified_report()``），
不可用于训练，只用于验证架构机制。
"""

from .content import (
    ASCENSION_TABLE, CARD_DB, ENCOUNTERS, ENEMY_DB, EVENT_DB, RELIC_POOL,
    lint, unverified_report,
)
from .core import (
    Action, CombatState, Hidden, Intent, Snapshot, action_mask, legal_actions,
    render_text, reroll_hidden, restore, snapshot, start_combat, step,
)
from .env import (
    MAX_ATTEMPT_BUDGET_REAL, MAX_ATTEMPT_BUDGET_SIM, STAGE_COMBAT, STAGE_DONE,
    STAGE_META, SpireEnv,
)
from .mapgen import (
    ANCIENT, BOSS, ELITE, MONSTER, REST_SITE, SHOP, TREASURE, UNKNOWN,
    MapGraph, MapNode, generate_map,
)
from .acts import ActDef, default_act_list, random_act_list
from .pointodds import UnknownMapPointOdds
from .observe import (
    FORBIDDEN_SUBSTRINGS, MapNodeView, Observation, RunObservation, observe,
    observe_run,
)
from .rng import STREAMS, RngSet
from .run import MetaAction, RunEnv, RunState, legal_meta_actions
from .bot import RuleBot
from .runbot import RunBot

__all__ = [
    "Action", "CombatState", "Hidden", "Intent", "Snapshot", "action_mask",
    "legal_actions", "render_text", "reroll_hidden", "restore", "snapshot",
    "start_combat", "step", "SpireEnv", "Observation", "observe",
    "STAGE_COMBAT", "STAGE_META", "STAGE_DONE",
    "MAX_ATTEMPT_BUDGET_SIM", "MAX_ATTEMPT_BUDGET_REAL",
    "FORBIDDEN_SUBSTRINGS", "RngSet", "STREAMS", "RuleBot",
    "CARD_DB", "ENEMY_DB", "ENCOUNTERS", "EVENT_DB", "RELIC_POOL",
    "ASCENSION_TABLE", "lint", "unverified_report",
    "RunEnv", "RunState", "MetaAction", "legal_meta_actions",
    "RunObservation", "observe_run", "MapNodeView", "generate_map",
    "MapGraph", "MapNode", "ActDef", "default_act_list", "random_act_list",
    "UnknownMapPointOdds",
    "MONSTER", "ELITE", "UNKNOWN", "REST_SITE", "SHOP", "TREASURE", "BOSS",
    "ANCIENT",
]
