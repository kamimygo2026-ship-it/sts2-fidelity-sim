"""训练配置清单的回归测试（``docs/13`` §7 / 工单 T00）。

清单必须能回答"这个 checkpoint 是在哪一版**源码 / 内容 / 引擎 / 协议**上训出来的"。
审计 F12 的教训：内容换了而词表下标没换，同名 embedding 会指向另一张牌，
**而且不会报错**。所以清单的每一层哈希都要能被独立验证。

| 症状 | 根因 | 对应测试 |
|---|---|---|
| 换了内容，清单看起来没变 | 只哈希了文件名/数量 | `test_content_hash_tracks_file_contents` |
| 改了引擎代码，清单不变 | 没把引擎模块纳入指纹 | `test_engine_hash_tracks_engine_sources` |
| 协议改了仍能复用旧清单 | 清单不记录观察/SL 协议 | `test_protocol_is_recorded` |
| 受限课程被当成完整范围 | 少算覆盖率 | `test_curriculum_is_labelled_not_assumed` |
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "data" / "content" / "repo"


def _load_manifest_tool():
    path = ROOT / "tools" / "training_manifest.py"
    spec = importlib.util.spec_from_file_location("training_manifest", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["training_manifest"] = module
    spec.loader.exec_module(module)
    return module


class TestTrainingManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CONTENT.exists():
            raise unittest.SkipTest(f"缺内容目录 {CONTENT}")
        from sts2_sim.featurize import configure_content
        configure_content(str(CONTENT))
        cls.tool = _load_manifest_tool()
        cls.manifest = cls.tool.build_manifest("ironclad")
        cls.addClassCleanup(cls._restore)

    @staticmethod
    def _restore():
        from sts2_sim.content import load_builtin
        from sts2_sim.featurize import rebuild_vocab
        load_builtin()
        rebuild_vocab()

    def test_required_fields_are_present(self):
        """``docs/13`` §7 列出的每一项都必须能在清单里找到。"""
        manifest = self.manifest
        for key in ("game_build", "source", "content", "engine", "protocol",
                    "scope", "allowed_content", "reject_reasons",
                    "unknown_mechanic_policy"):
            self.assertIn(key, manifest)
        self.assertTrue(manifest["source"]["hash"])
        self.assertTrue(manifest["content"]["files_hash"])
        self.assertTrue(manifest["engine"]["hash"])

    def test_protocol_is_recorded(self):
        from sts2_sim.observe import OBSERVATION_PROTOCOL
        self.assertEqual(self.manifest["protocol"]["observation"],
                         OBSERVATION_PROTOCOL)
        self.assertTrue(self.manifest["protocol"]["sl"])

    def test_scope_records_character_and_ascension(self):
        scope = self.manifest["scope"]
        self.assertEqual(scope["character"], "ironclad")
        self.assertEqual(scope["ascension"], 0)
        # 解锁状态必须显式标出来源不明，不能假装知道（docs/13 §15）
        self.assertEqual(scope["unlocks"], "unknown")

    def test_sampler_and_manifest_agree_on_the_pool(self):
        """清单里的可抽池必须**就是**采样器用的那个池（同源）。"""
        from sts2_rl.ppo import draft_pool
        from sts2_rl.ppo import encounter_pool
        self.assertEqual(self.manifest["allowed_content"]["draft_pool"],
                         len(draft_pool("ironclad")))
        self.assertEqual(self.manifest["allowed_content"]["encounters"]["monster"],
                         len(encounter_pool("ironclad")))

    def test_curriculum_is_labelled_not_assumed(self):
        """受限课程必须被标注出来，并且带一句"不能声称完成真实分布"。"""
        scope = self.manifest["scope"]
        self.assertIn(scope["curriculum"],
                      (self.tool.CURRICULUM_FULL,
                       self.tool.CURRICULUM_RESTRICTED))
        coverage = self.manifest["allowed_content"]["draft_pool_coverage"]
        if coverage < 0.95:
            self.assertEqual(scope["curriculum"], self.tool.CURRICULUM_RESTRICTED)
            self.assertIn("不能", scope["curriculum_note"])

    def test_content_hash_tracks_file_contents(self):
        """反向验证：改一个内容文件的内容，哈希必须变。

        只哈希文件名或数量的话，"补了 10 张卡的效果"这种改动**看不出来** ——
        而那正是本项目每天都在做的事。
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text('{"x": 1}', encoding="utf-8")
            before, _ = self.tool._sha256_files(root, ("*.json",))
            (root / "a.json").write_text('{"x": 2}', encoding="utf-8")
            after, _ = self.tool._sha256_files(root, ("*.json",))
            self.assertNotEqual(before, after, "内容哈希没有跟着文件内容变")

    def test_content_hash_tracks_relative_paths(self):
        """改名也算内容变化（清单换了名字就是换了清单）。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text("{}", encoding="utf-8")
            before, _ = self.tool._sha256_files(root, ("*.json",))
            (root / "a.json").rename(root / "b.json")
            after, _ = self.tool._sha256_files(root, ("*.json",))
            self.assertNotEqual(before, after, "相对路径没进哈希")

    def test_engine_hash_tracks_engine_sources(self):
        """引擎代码也是"机制的唯一真相"的一部分，必须进指纹。"""
        self.assertIn("core.py", self.manifest["engine"]["files"])
        self.assertIn("powers.py", self.manifest["engine"]["files"])
        self.assertIn("eligibility.py", self.manifest["engine"]["files"])

    def test_manifest_id_covers_content_outside_cards_and_enemies(self):
        """⭐ 文件名必须覆盖**全部**内容，不只是卡牌/怪物。

        `content_fingerprint` 只哈希卡牌与怪物，于是"只改了药水/事件/遗物"
        的两份内容会算出同一个文件名 —— 后一份把前一份**覆盖**掉，
        而两份清单说的根本不是同一套内容。
        """
        first = self.tool.manifest_id(self.manifest)
        changed = json.loads(json.dumps(self.manifest))
        changed["content"]["files_hash"] = "f" * 64      # 只改内容文件哈希
        self.assertNotEqual(first, self.tool.manifest_id(changed))

    def test_manifest_id_is_stable_for_the_same_content(self):
        again = self.tool.build_manifest("ironclad")
        self.assertEqual(self.tool.manifest_id(self.manifest),
                         self.tool.manifest_id(again))

    def test_manifest_id_changes_with_engine_code(self):
        changed = json.loads(json.dumps(self.manifest))
        changed["engine"]["hash"] = "e" * 64
        self.assertNotEqual(self.tool.manifest_id(self.manifest),
                            self.tool.manifest_id(changed))

    def test_check_mode_detects_a_stale_manifest(self):
        saved = json.loads(json.dumps(self.manifest))
        saved["content"]["files_hash"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stale.json"
            path.write_text(json.dumps(saved, ensure_ascii=False),
                            encoding="utf-8")
            code = self.tool.main(["--check", str(path)])
        self.assertEqual(code, 1, "陈旧清单必须被 --check 判为不一致")


if __name__ == "__main__":
    unittest.main(verbosity=2)
