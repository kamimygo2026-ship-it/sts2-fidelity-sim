"""策略 / 价值网络（``docs/03``）。

结构：**实体 Transformer 编码器 + 指针式动作头 + 多任务价值头**。

```
token 序列 ──► EntityEncoder ──► z ──┬──► 动作打分（对每个合法动作打一个分）
（全局/手牌/敌人/遗物/药水）          └──► 价值头（胜率 / 剩余 HP 比例）
     ▲
     └── 牌堆多重集计数 ──► bag 池化（与手牌共用同一张卡牌 embedding 表）
```

与 ``docs/03`` §3.5 的一处**有意的偏离**：文档写的是自回归因式分解
（kind → card → target）。这里实现的是**对合法动作集合打分**（指针网络）。
两者都能处理变长动作集，但指针式实现更短、mask 更直接、调试更容易。
对于本项目的动作语法（≤64 个合法动作），表达力足够。若将来需要，可以再加一级
因式分解而不改动编码器。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sts2_sim.content import CARD_DB, ENEMY_DB, POTIONS, RELICS
from sts2_sim.featurize import (
    ACTION_KINDS, MAX_ACTION_SLOTS, MAX_ACTIONS, MAX_ENEMIES, MAX_HAND,
    MAX_POTIONS, MAX_TOKENS, NUM_FEATURES, NUM_PILES, NUM_TOKEN_TYPES,
    POWER_VOCAB, TOKEN_ENEMY, TOKEN_GLOBAL, TOKEN_HAND, TOKEN_ORB, TOKEN_POTION,
    TOKEN_POWER, TOKEN_RELIC,
)
from sts2_sim.orbs import ORB_DEFS

N_ACTION_KINDS = len(ACTION_KINDS)
#: 空槽位都用下标 0（词表整体 +1 偏移）
NULL = 0


def vocab_sizes() -> dict[str, int]:
    """**当前**内容表下每类实体词表的大小。

    ⚠️ 必须用函数而不是模块级常量（审计 F12 的同类错误）：``N_CARDS = len(CARD_DB)``
    在 import 时求值，那时内容表还是内置占位内容（12 张卡）。等真正加载了 597 张卡
    再建网络，embedding 表只有 13 行 —— 一查表就 ``IndexError``，而且错误信息
    完全不提"内容没加载"。现在改成**建网络时现算**，并在前向里显式校验。
    """
    return {
        "cards": len(CARD_DB),
        "enemies": len(ENEMY_DB),
        "relics": len(RELICS),
        "potions": len(POTIONS),
        "orbs": len(ORB_DEFS),
        # ⭐ 能力实体（外部复核 R4）：词表由 `featurize.rebuild_vocab` 现算，
        # 这里只报大小 —— 网络按它建 embedding 表。
        "powers": len(POWER_VOCAB),
    }

#: 非法动作的 logit 掩码值。
#:
#: ⚠️ **必须用有限的大负数，不能用 ``-inf``。**
#: 用 ``-inf`` 时熵项是 ``exp(logp) * logp``，在屏蔽位上就是 ``0 * (-inf) = NaN``；
#: 前向会被 ``nan_to_num`` 掩盖，但**反向会得到 NaN 梯度**，污染整个网络——
#: 表现为"loss 有限、grad_norm 是 NaN、参数在一步之后全变 NaN"。
#: 这个坑是 tests/test_training.py 的"一次更新后参数必须仍然有限"抓出来的。
MASK_LOGIT = -1e9


def _shift(index: torch.Tensor) -> torch.Tensor:
    """-1 → 0（空），其余 +1。用于 embedding 查表。"""
    return torch.where(index < 0, torch.zeros_like(index), index + 1)


class EntityEncoder(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 3, n_heads: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        # ⚠️ 词表大小**在这里现算**（不是模块级常量）：内容表可以在 import 之后
        # 才加载，常量会把网络钉死在内置占位内容上。
        sizes = vocab_sizes()
        self.vocab_sizes = sizes
        self.d_model = d_model
        self.type_emb = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        # ⭐ 全项目**只有这一张**卡牌 embedding 表：手牌 token 与牌堆池化共用
        self.card_emb = nn.Embedding(sizes["cards"] + 1, d_model, padding_idx=NULL)
        self.enemy_emb = nn.Embedding(sizes["enemies"] + 1, d_model, padding_idx=NULL)
        # ⭐ 审计 F04：遗物 / 药水 / 充能球以前是空占位，模型看不到它们。
        # 现在各自一张嵌入表（**每类一张**：卡、敌、遗物、药水、球的下标空间不同，
        # 混用会让"第 3 号遗物"和"第 3 号药水"在表示上撞车）。
        self.relic_emb = nn.Embedding(sizes["relics"] + 1, d_model, padding_idx=NULL)
        self.potion_emb = nn.Embedding(sizes["potions"] + 1, d_model, padding_idx=NULL)
        self.orb_emb = nn.Embedding(sizes["orbs"] + 1, d_model, padding_idx=NULL)
        # ⭐ 能力实体（外部复核 R4）：每类实体一张表（下标空间不同，混用会撞车）
        self.power_emb = nn.Embedding(sizes["powers"] + 1, d_model, padding_idx=NULL)
        self.feat_proj = nn.Sequential(
            nn.Linear(NUM_FEATURES, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        # ⭐ 牌堆池化收 **2×NUM_PILES + 1** 行：三种牌堆各一份"全部"与一份
        # "其中升级过的"（审计 F04：牌堆编码忽略升级标记）+ 已揭示知识。
        self.pile_proj = nn.Sequential(
            nn.Linear((NUM_PILES * 2 + 1) * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        token_type = batch["token_type"]                       # (B, T)
        entity = batch["token_entity"]                         # (B, T)
        numbers = batch["token_num"]                           # (B, T, F)
        mask = batch["token_mask"]                             # (B, T)
        piles = batch["pile_counts"]                           # (B, P, C)
        piles_up = batch["pile_upgraded"]                       # (B, P, C)

        is_hand = (token_type == TOKEN_HAND)
        is_enemy = (token_type == TOKEN_ENEMY)
        is_relic = (token_type == TOKEN_RELIC)
        is_potion = (token_type == TOKEN_POTION)
        is_orb = (token_type == TOKEN_ORB)
        is_power = (token_type == TOKEN_POWER)
        self._check_vocab_matches(token_type, entity)
        # ⚠️ 各张词表大小不同，**必须先按 token 类型把下标清零再查表**。
        # 之前对全部 token 同时查两张表，手牌的卡牌下标会越界到敌人词表 → IndexError。
        def _lookup(table: nn.Embedding, select: torch.Tensor) -> torch.Tensor:
            index = torch.where(select, _shift(entity), torch.zeros_like(entity))
            return table(index)

        entity_vec = (
            _lookup(self.card_emb, is_hand)
            + _lookup(self.enemy_emb, is_enemy)
            + _lookup(self.relic_emb, is_relic)
            + _lookup(self.potion_emb, is_potion)
            + _lookup(self.orb_emb, is_orb)
            + _lookup(self.power_emb, is_power)
        )

        hidden = entity_vec + self.feat_proj(numbers) + self.type_emb(token_type)

        # bag 池化：counts @ card_emb（置换不变 → 结构上不含顺序信息）
        weighted = piles @ self.card_emb.weight[1:]            # (B, P, d)
        # ⭐ 升级过的牌单独一行：牌堆里"升级打击"和"打击"不再等同（审计 F04）
        weighted_up = piles_up @ self.card_emb.weight[1:]
        # ⭐ SL 知识也走同一张卡牌 embedding：已揭示的牌不该另起一套表示
        revealed = (batch["revealed_counts"] @ self.card_emb.weight[1:]).unsqueeze(1)
        pooled = torch.cat([weighted, weighted_up, revealed], dim=1).flatten(1)
        pile_vec = self.pile_proj(pooled)                       # (B, d)
        hidden = hidden + pile_vec.unsqueeze(1) * (token_type == TOKEN_GLOBAL).unsqueeze(-1)

        padding = mask < 0.5
        encoded = self.encoder(hidden, src_key_padding_mask=padding)
        return self.norm(encoded[:, 0])                         # 全局 token

    def _check_vocab_matches(self, token_type: torch.Tensor,
                             entity: torch.Tensor) -> None:
        """内容表换了而网络没重建时**立刻报错**。

        审计 F12 的同类症状：内容变化后同名 embedding 下标会指向另一个实体，
        而且**不会报错**（只是数值全错）。这里把"内容与网络不一致"变成显式失败。
        """
        current = vocab_sizes()
        if current != self.vocab_sizes:
            raise ValueError(
                f"网络是在词表 {self.vocab_sizes} 下构建的，现在是 {current}。"
                f"内容表变化后必须重建网络（docs/06 §6.4）。")
        limits = {
            TOKEN_HAND: (self.card_emb.num_embeddings, "卡牌"),
            TOKEN_ENEMY: (self.enemy_emb.num_embeddings, "敌人"),
            TOKEN_RELIC: (self.relic_emb.num_embeddings, "遗物"),
            TOKEN_POTION: (self.potion_emb.num_embeddings, "药水"),
            TOKEN_ORB: (self.orb_emb.num_embeddings, "充能球"),
            TOKEN_POWER: (self.power_emb.num_embeddings, "能力"),
        }
        for type_id, (size, name) in limits.items():
            selected = entity[token_type == type_id]
            if selected.numel() and int(selected.max()) >= size - 1:
                raise ValueError(
                    f"{name} 词表下标越界（max={int(selected.max())} ≥ {size - 1}）："
                    f"编码与网络的词表版本不一致")


class PolicyValueNet(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 3, n_heads: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = EntityEncoder(d_model, n_layers, n_heads, dropout)
        d = d_model
        # 动作侧的词表尺寸同样**现算**（内容表可以在 import 之后才加载）。
        sizes = vocab_sizes()

        self.kind_emb = nn.Embedding(N_ACTION_KINDS + 1, d, padding_idx=NULL)
        # ⚠️ 槽位表按 ``MAX_ACTION_SLOTS``（= max(手牌上限, 候选上限)）：
        # `select_card` 的下标是**候选**下标，可以远大于手牌上限（审计 F09）。
        self.slot_emb = nn.Embedding(MAX_ACTION_SLOTS + 2, d, padding_idx=NULL)
        self.target_emb = nn.Embedding(MAX_ENEMIES + 2, d, padding_idx=NULL)
        #: 道具身份（药水）。审计 F09：药水动作要能表达"用哪一种药水"。
        self.item_emb = nn.Embedding(sizes["potions"] + 1, d, padding_idx=NULL)
        self.act_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.value_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2),
        )
        self.logit_scale = 1.0 / math.sqrt(d)

    def forward(self, batch: dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # ⚠️ 守卫：若某一行没有任何合法动作，masked_fill(-inf) 之后 softmax 会算出
        # 0/0 = NaN，而且 NaN 会静默污染整个 batch（表现为"所有行 logits 都是 nan"）。
        # 这类错误极难从结果反推，所以在这里直接报错。
        if bool((batch["act_mask"].sum(-1) < 0.5).any()):
            raise ValueError("存在没有任何合法动作的样本——mask 或环境有 bug")
        if bool((batch["token_mask"].sum(-1) < 0.5).any()):
            raise ValueError("存在没有任何 token 的样本——特征化有 bug")

        z = self.encoder(batch)                                 # (B, d)
        act_vec = (self.kind_emb(_shift(batch["act_kind"]))
                   + self.encoder.card_emb(_shift(batch["act_card"]))
                   + self.slot_emb(_shift(batch["act_slot"]))
                   + self.target_emb(_shift(batch["act_target"]))
                   + self.item_emb(_shift(batch["act_item"])))
        act_vec = self.act_mlp(act_vec)                          # (B, A, d)

        logits = (act_vec * z.unsqueeze(1)).sum(-1) * self.logit_scale   # (B, A)
        logits = logits.masked_fill(batch["act_mask"] < 0.5, MASK_LOGIT)
        value = self.value_head(z)                               # (B, 2): [胜率, 剩余HP比例]
        return logits, value

    @torch.no_grad()
    def act(self, batch: dict[str, torch.Tensor], deterministic: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(batch)
        log_probs = torch.log_softmax(logits, dim=-1)
        if deterministic:
            action = log_probs.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(logits=logits).sample()
        return action, log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1), value

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
