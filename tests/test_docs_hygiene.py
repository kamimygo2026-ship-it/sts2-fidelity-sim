"""文档卫生的护栏测试（``tools/check_docs.py``）。

为什么值得一条测试：文档腐烂是**静默**的 —— 链接指向已改名的文件、
正文引用已归档的编号、绝对路径带着别人机器上的行号。
这些都不会让任何代码报错，但会让"照文档做"的人扑空。

所以这里把 `tools/check_docs.py` 的检查接进回归套件：文档一旦腐烂就红。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


class TestDocsHygiene(unittest.TestCase):
    def test_no_broken_links_or_stale_doc_refs(self):
        import check_docs

        problems = check_docs.check(verbose=False)
        self.assertEqual(problems, [], "文档卫生问题：\n" + "\n".join(problems))

    def test_archived_docs_exist_and_are_pointed_at(self):
        """归档目录必须存在，且每份归档都带"已归档"横幅（否则读者会当活文档看）。"""
        archive = ROOT / "docs" / "archive"
        self.assertTrue(archive.is_dir(), "缺少 docs/archive/")
        self.assertTrue((archive / "README.md").exists(), "缺少归档索引 README")
        files = sorted(p for p in archive.glob("*.md") if p.name != "README.md")
        self.assertGreaterEqual(len(files), 5, "归档文件太少，整理时可能漏了")
        for path in files:
            head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:6])
            self.assertIn("已归档", head, f"{path.name} 缺少归档横幅")

    def test_current_doc_set_is_complete(self):
        """当前文档集编号**不许有洞**（有洞说明整理只做了一半）。

        ⚠️ 判据是"**连续**"，不是"上界等于 14"：写成 `list(range(1, 15))` 之后，
        每加一份新文档（例如 `docs/15-架构复核-2026-09-22.md`）都会把这条测试
        变成假红 —— 而它想拦的是"编号跳到 03、05 就没了"这种**整理做一半**，
        与文档总数无关。所以这里只钉连续性 + 一个下界。
        """
        docs = ROOT / "docs"
        numbers = sorted(int(p.name[:2]) for p in docs.glob("[0-9][0-9]-*.md"))
        self.assertGreaterEqual(len(numbers), 14,
                                f"当前文档集太少，整理时可能漏了：{numbers}")
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)),
                         f"文档编号有洞：{numbers}")

    def test_readme_nav_points_at_existing_docs(self):
        """README 的文档导航表里每一行链接都必须能点开。"""
        import re

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for target in re.findall(r"\]\((docs/[^)]+)\)", readme):
            path = target.split("#")[0]
            self.assertTrue((ROOT / path).exists(), f"README 链接失效：{target}")


if __name__ == "__main__":
    unittest.main()
