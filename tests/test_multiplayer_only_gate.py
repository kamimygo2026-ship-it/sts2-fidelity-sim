"""**多人专用卡门禁**：单人 profile 里遇不到的牌不许进训练池（`docs/12` §2.32）。

来源：`CardMultiplayerConstraint`（``CardMultiplayerConstraint.cs``）只有三个值 ——
``None`` / ``MultiplayerOnly`` / ``SingleplayerOnly``；本地构建里 ``MultiplayerOnly``
有 37 张、``SingleplayerOnly`` 为 0。

这条门禁是三处**必须一致**的：

1. 反编译源码里 ``MultiplayerConstraint => CardMultiplayerConstraint.MultiplayerOnly``；
2. ``data/content/repo/cards_source.json`` 里的 ``multiplayer_only`` 字段；
3. 装载后的 ``CardDef.multiplayer_only``（``content._load_cards_source`` 在所有分支之后统一落）。

少了任何一处，"多人专用卡进了单人池"这件事就会以不同形态复发（`docs/12` §2.31.3
测出当时有 16 张混进了池子，其中两张是 §2.28 那批刚放行的）。

⚠️ 这条判据**不是**"引擎做不到"：多人专用卡的效果在源码里可能是完整的。
它回答的是"这个 profile 里有没有这张牌"。
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"
DECOMPILED = ROOT / "data" / "decompiled" / "sts2"

#: 源码里声明的多人专用卡（本批实测 37 张）。用作**下限**断言：
#: 真机更新后这个数字可以变大，但不能变小（变小说明抽取器漏了）。
MIN_MULTIPLAYER_ONLY = 37


def setup_content():
    if not CONTENT.exists():
        raise unittest.SkipTest(f"缺内容目录 {CONTENT}")
    from sts2_sim.featurize import configure_content
    configure_content(str(CONTENT))


def restore_builtin() -> None:
    from sts2_sim.content import load_builtin
    from sts2_sim.featurize import rebuild_vocab
    load_builtin()
    rebuild_vocab()


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class MultiplayerOnlyGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_content()
        cls.addClassCleanup(restore_builtin)

    def source_ids(self) -> set[str]:
        if not DECOMPILED.exists():
            self.skipTest(f"缺反编译源码 {DECOMPILED}")
        found = set()
        for path in DECOMPILED.rglob("*.cs"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(
                    r"MultiplayerConstraint\s*=>\s*"
                    r"CardMultiplayerConstraint\.MultiplayerOnly", text):
                found.add(_snake(path.stem))
        return found

    def test_three_layers_agree(self):
        """源码 / 抽取产物 / 装载后的表三处必须是**同一个集合**。"""
        from sts2_sim import content
        source = self.source_ids()
        self.assertGreaterEqual(len(source), MIN_MULTIPLAYER_ONLY)
        recorded = {r["cid"] for r in json.loads(
            (CONTENT / "cards_source.json").read_text(encoding="utf-8"))
            if r.get("multiplayer_only")}
        loaded = {cid for cid, card in content.CARD_DB.items()
                  if card.multiplayer_only}
        self.assertEqual(recorded, source, "抽取产物与源码不一致")
        self.assertEqual(loaded, recorded, "装载后的表与抽取产物不一致")

    def test_no_multiplayer_only_card_is_admitted(self):
        from sts2_sim import eligibility
        bad = [cid for cid in eligibility.admitted_cards()
               if eligibility.CARD_DB[cid].multiplayer_only]
        self.assertEqual(bad, [], "多人专用卡不该出现在单人可执行集合里")

    def test_reason_is_reported(self):
        from sts2_sim import eligibility
        for cid in ("tank", "coordinate", "fade", "tag_team"):
            with self.subTest(cid=cid):
                self.assertIn("multiplayer_only", eligibility.card_reasons(cid))

    def test_normal_cards_are_not_affected(self):
        """对照组：普通卡不会因为这条门禁被误伤。"""
        from sts2_sim import eligibility
        for cid in ("strike_ironclad", "defend_ironclad", "bash", "strangle"):
            with self.subTest(cid=cid):
                self.assertNotIn("multiplayer_only", eligibility.card_reasons(cid))


if __name__ == "__main__":
    unittest.main()
