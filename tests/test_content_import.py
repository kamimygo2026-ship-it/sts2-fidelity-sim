"""内容导入链路测试：原始记录 → 内容表 JSON → 模拟器加载 → 词表重建。

这条链路**必须在没有网络的情况下可测**——真正的抓取要用户在自己终端跑，
但"抓到之后能不能用"必须由测试保证。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from sts2_sim import content as content_module
from sts2_sim.content import CARD_DB, ENEMY_DB, lint, load_builtin
from sts2_sim.featurize import (
    CARD_VOCAB, configure_content, rebuild_vocab, vocab_fingerprint,
)
from tools.import_content import convert_card, parse_effects, _uncovered

#: 一条模拟的社区原始记录（字段名按 docs/06 §6.9 的侦察结论构造）
RAW_CARD = {
    "id": "CARD.SEVER_SOUL",
    "name": "裂魂",
    "cost": 2,
    "type": "Attack",
    "rarity": "Uncommon",
    "description": "造成 16 点伤害。如果这张牌杀死了敌人，获得 2 点力量。",
    "upgraded_description": "造成 20 点伤害。",
}

RAW_CARDS_FOR_LOAD = [
    {"id": "CARD.STRIKE", "name": "打击", "cost": 1, "type": "Attack",
     "rarity": "Basic", "description": "造成 6 点伤害。"},
    {"id": "CARD.DEFEND", "name": "防御", "cost": 1, "type": "Skill",
     "rarity": "Basic", "description": "获得 5 点格挡。"},
    {"id": "CARD.BASH", "name": "重击", "cost": 2, "type": "Attack",
     "rarity": "Basic", "description": "造成 8 点伤害。施加 2 层易伤。"},
    {"id": "CARD.RAGE", "name": "暴怒", "cost": 0, "type": "Skill",
     "rarity": "Uncommon", "description": "获得 3 点格挡。抽 1 张牌。"},
]


class TestEffectParsing(unittest.TestCase):
    def test_common_patterns(self):
        effects, leftovers = parse_effects("造成 8 点伤害。获得 5 点格挡。")
        ops = sorted(e["op"] for e in effects)
        self.assertEqual(ops, ["block", "damage"])
        self.assertEqual(leftovers, [])

    def test_power_patterns(self):
        effects, _ = parse_effects("造成 8 点伤害。施加 2 层易伤。")
        powers = [e for e in effects if e["op"] == "apply_power"]
        self.assertEqual(len(powers), 1)
        self.assertEqual(powers[0]["power"], "vulnerable")
        self.assertEqual(powers[0]["amount"], 2)
        self.assertEqual(powers[0]["target"], "enemy")

    def test_self_buff_power_targets_self(self):
        """力量是自增益。被误判成"对敌施加"会让卡牌定义语义完全错误。"""
        effects, _ = parse_effects("获得 2 点力量。")
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0]["power"], "strength")
        self.assertEqual(effects[0]["target"], "self")

    def test_skill_target_is_not_enemy(self):
        """纯格挡/抽牌的技能牌不该被标成需要选目标。"""
        from tools.import_content import infer_target
        effects, _ = parse_effects("获得 5 点格挡。")
        self.assertEqual(infer_target(effects), "none")

    def test_aoe_pattern(self):
        effects, _ = parse_effects("对所有敌人造成 8 点伤害。")
        self.assertEqual(effects[0]["op"], "damage_all")

    def test_english_patterns(self):
        effects, _ = parse_effects("Deal 11 damage. Gain 5 Block.")
        self.assertEqual(sorted(e["op"] for e in effects), ["block", "damage"])

    def test_unparsed_text_is_reported_not_silently_dropped(self):
        """⭐ 解析不了的部分必须**被报出来**，绝不能静默丢弃。

        静默丢弃 = 模型学到一个不完整的卡，而且没有任何信号提示。
        """
        effects, leftovers = parse_effects("造成 6 点伤害。如果这张牌杀死了敌人，获得 2 点力量。")
        self.assertTrue(any(e["op"] == "damage" for e in effects))
        self.assertTrue(leftovers, "未识别的句子应当进入 leftover 报告")
        self.assertTrue(any("杀死" in piece for piece in leftovers))

    def test_uncovered_spans(self):
        pieces = _uncovered("ABCDEF", [(1, 3)])
        self.assertEqual(pieces, ["A", "DEF"])


class TestConvertCard(unittest.TestCase):
    def test_maps_core_fields(self):
        report: Counter = Counter()
        card = convert_card(RAW_CARD, report)
        self.assertIsNotNone(card)
        assert card is not None
        self.assertEqual(card["cid"], "sever_soul")
        self.assertEqual(card["cost"], 2)
        self.assertEqual(card["card_type"], "attack")
        self.assertEqual(card["rarity"], "uncommon")
        self.assertFalse(card["verified"], "导入的内容一律未对拍")
        self.assertTrue(any(e["op"] == "damage" and e["amount"] == 16
                            for e in card["effects"]))
        self.assertTrue(card["upgrade"], "升级后文本应当解析出升级效果")

    def test_missing_id_is_reported(self):
        report: Counter = Counter()
        self.assertIsNone(convert_card({"name": "无 id"}, report))
        self.assertEqual(report["card_without_id"], 1)

    def test_unparsed_fragments_are_counted(self):
        report: Counter = Counter()
        convert_card(RAW_CARD, report)
        self.assertTrue(any(key.startswith("card_effect_unparsed::") for key in report))


class TestContentLoading(unittest.TestCase):
    """导入产物 → 模拟器可用。**调用顺序错了会静默出错**，所以这里也测顺序。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        self.addCleanup(self._restore)

    def _restore(self):
        load_builtin()
        rebuild_vocab()
        self._tmp.cleanup()

    def _write_cards(self, records) -> None:
        payload = [convert_card(r, Counter()) for r in records]
        (self.directory / "cards.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_configure_content_replaces_cards_and_rebuilds_vocab(self):
        self._write_cards(RAW_CARDS_FOR_LOAD)
        before_hash = vocab_fingerprint()["vocab_hash"]
        info = configure_content(self.directory)

        self.assertEqual(info["cards"], 4)
        self.assertEqual(info["missing_starting_cards"], [],
                         "起始牌组的三张牌都在，不该报缺失")
        self.assertEqual(set(CARD_DB), {"strike", "defend", "bash", "rage"})
        self.assertIn("rage", CARD_VOCAB, "词表必须随内容重建")
        self.assertNotEqual(vocab_fingerprint()["vocab_hash"], before_hash)

    def test_loaded_content_passes_lint(self):
        self._write_cards(RAW_CARDS_FOR_LOAD)
        configure_content(self.directory)
        self.assertEqual(lint(), [], "导入的内容应当通过静态检查")

    def test_missing_starting_cards_are_reported(self):
        """起始牌组缺牌会导致一开局就崩，必须显式报出来而不是等到运行时报错。"""
        self._write_cards([RAW_CARD])           # 只有一张，没有 strike/defend/bash
        info = configure_content(self.directory)
        self.assertIn("strike", info["missing_starting_cards"])

    def test_loading_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            configure_content(self.directory / "不存在")

    def test_builtin_is_restored_after_reload(self):
        self._write_cards(RAW_CARDS_FOR_LOAD)
        configure_content(self.directory)
        load_builtin()
        rebuild_vocab()
        self.assertEqual(content_module.CONTENT_SOURCE, "builtin")
        self.assertIn("cleave", CARD_DB, "内置占位内容应当完整还原")
        self.assertIn("cleave", CARD_VOCAB)

    def test_env_var_autoload_is_opt_in(self):
        """默认**不**自动加载 JSON，否则测试与训练会悄悄用上不同内容。"""
        import os
        self.assertNotIn(content_module.CONTENT_DIR_ENV, os.environ)


class TestVocabFingerprint(unittest.TestCase):
    def test_fingerprint_changes_with_content(self):
        self.addCleanup(lambda: (load_builtin(), rebuild_vocab()))
        baseline = vocab_fingerprint()
        self.assertGreater(baseline["n_cards"], 0)
        CARD_DB["__tmp_card__"] = CARD_DB[next(iter(CARD_DB))]
        rebuild_vocab()
        self.assertNotEqual(vocab_fingerprint()["vocab_hash"],
                            baseline["vocab_hash"],
                            "内容变了指纹就必须变——否则 checkpoint 无法自检")


class TestLanguageSelection(unittest.TestCase):
    """导出包里每个语言各有一份 cards.json —— 全读进来会让记录数翻倍且互相覆盖。

    这是拿到真实导出包之后**最容易踩的坑**，所以单独测。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, lang: str, name: str) -> None:
        path = self.root / lang
        path.mkdir(parents=True, exist_ok=True)
        (path / "cards.json").write_text(json.dumps([
            {"id": f"CARD.{lang.upper()}_STRIKE", "name": f"{lang}打击", "cost": 1,
             "type": "Attack", "rarity": "Basic", "description": "造成 6 点伤害。"},
        ], ensure_ascii=False), encoding="utf-8")

    def test_language_of_detects_directory(self):
        from tools.import_content import language_of
        self.assertEqual(language_of("zhs/cards.json"), "zhs")
        self.assertEqual(language_of("data/eng/monsters.json"), "eng")
        self.assertEqual(language_of("cards.json"), "")

    def test_only_preferred_language_is_used(self):
        from tools.import_content import collect_buckets
        from tools.import_content import dataset_name  # noqa: F401
        for lang in ("zhs", "eng"):
            self._write(lang, "cards.json")
        files = sorted(self.root.rglob("*.json"))
        buckets, chosen, counts, _prov = collect_buckets(files, preferred_lang="zhs")

        self.assertEqual(chosen, "zhs")
        self.assertEqual(counts, {"zhs": 1, "eng": 1})
        ids = [c["id"] for c in buckets["cards"]]
        self.assertEqual(ids, ["CARD.ZHS_STRIKE"],
                         "只能读入选中的那个语言，不能把两个语言混在一起")

    def test_falls_back_to_the_largest_language(self):
        from tools.import_content import collect_buckets
        self._write("eng", "cards.json")
        self._write("eng", "monsters.json")     # eng 记录更多
        self._write("kor", "cards.json")
        files = sorted(self.root.rglob("*.json"))
        _, chosen, _, _prov = collect_buckets(files, preferred_lang="zhs")
        self.assertEqual(chosen, "eng")

    def test_zip_only_dataset_is_filled_from_archive(self):
        """散装端点没有的数据集必须从导出包补上。

        旧实现是 `sources = plain if plain else archives` 的**全有或全无**：
        只要散装文件存在，ZIP 就整个不读，ZIP 独有的 7 个数据集
        （enchantments / orbs / afflictions…）被静默丢掉。这是"修重复导入"
        时引入的相反方向的 bug，必须两边都钉住。
        """
        import zipfile
        from tools.import_content import collect_buckets
        (self.root / "cards.json").write_text(json.dumps(
            [{"id": "CARD.A", "name": "A", "description": "Deal 6 damage."}]),
            encoding="utf-8")
        archive = self.root / "export_zhs.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("cards.json", json.dumps(
                [{"id": "CARD.ZHS", "name": "打击", "description": "造成 6 点伤害。"}]))
            zf.writestr("enchantments.json", json.dumps(
                [{"id": "ENCH.SHARP", "name": "Sharp"}]))
        files = sorted(p for p in self.root.rglob("*") if p.suffix in (".json", ".zip"))
        buckets, _chosen, _counts, prov = collect_buckets(files, preferred_lang="zhs")

        self.assertEqual([c["id"] for c in buckets["cards"]], ["CARD.A"],
                         "散装端点是英文，必须压过导出包里的中文")
        self.assertEqual([e["id"] for e in buckets["enchantments"]], ["ENCH.SHARP"],
                         "ZIP 独有的数据集不能丢")
        self.assertEqual(prov["cards"]["lang"], "eng")
        self.assertEqual(prov["enchantments"]["lang"], "zhs")

    def test_langless_plain_file_is_treated_as_default_english(self):
        """没有语言目录的散装文件 = 默认英文（裸端点返回英文）。

        这条曾经叫 `..._treated_as_unknown`。改成 `eng` 是有依据的：卡牌效果
        解析器吃英文描述，而 spire-codex 的裸端点就是英文；继续叫 unknown
        会让"散装 vs 导出包"的优先级判断失去依据。
        """
        from tools.import_content import collect_buckets
        (self.root / "cards.json").write_text(json.dumps(
            [{"id": "CARD.X", "name": "X", "description": "Gain 5 Block."}]),
            encoding="utf-8")
        files = sorted(self.root.rglob("*.json"))
        buckets, chosen, _counts, prov = collect_buckets(files, preferred_lang="zhs")
        self.assertEqual(chosen, "eng")
        self.assertEqual(prov["cards"]["lang"], "eng")
        self.assertEqual(len(buckets["cards"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
