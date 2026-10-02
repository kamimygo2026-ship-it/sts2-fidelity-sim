"""文档卫生检查：链接能不能点、引用有没有过期、绝对路径有没有漏进来。

为什么需要它：文档里最容易腐烂的东西不是结论，而是**指路**。

  * `D:/桌面/尖塔模型/sts2_sim/core.py:1232` 这类**绝对路径 + 行号**：
    路径只在本机成立，行号在下次重构后就错了 —— 而且点开还能跳到"某个地方"，
    于是没人发现它指的是别的函数。
  * 指向**已归档**文档的编号（本文档集把 `docs/14`/`15`/`16` 合并进
    `docs/12`/`13` 后归档，见 `docs/archive/README.md`）。

规则（`--check` 会逐条报）：

1. **相对链接必须能解析**（`](...)` 的目标文件真实存在；带 `:行号` 的按去行号后的路径判）。
2. **不允许绝对路径**（`D:/…` / `/home/…` / `file://`）—— 文档要跟着仓库走。
3. **引擎/工具/测试文件的引用不带行号**（它们会变）；`data/decompiled/` 这种
   **冻结**的源码树允许带行号（对拍时要能直接跳到那一行）。
4. **不再引用已归档的编号**：`docs/14` / `docs/15` / `docs/16`。

用法：

    python -X utf8 tools/check_docs.py            # 检查（有问题返回 1）
    python -X utf8 tools/check_docs.py --fix      # 把绝对路径改成相对路径 + 去掉易腐行号
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: 会被当成"文档"来检查的文件。
DOC_GLOBS = ("*.md", "docs/*.md")

#: 行号**可以**保留的路径前缀：反编译源码树是冻结的，行号不会漂。
FROZEN_PREFIXES = ("data/decompiled/",)

#: 允许引用已归档编号的**自我声明**。合并后的文档里必然有一张"旧编号对照表"，
#: 它的用途就是写清 `docs/14` 的哪一节去了哪里 —— 那不是"还在引用活文档"。
#: 在文件里任意位置写这一行即可放开该文件的第 4 条规则：
#:     <!-- check_docs: allow-archived-refs -->
ALLOW_ARCHIVED_MARKER = "check_docs: allow-archived-refs"


def archived_targets() -> tuple[set[str], set[str]]:
    """``(归档文件名, 只存在于归档的编号)``。

    ⚠️ **不能只按编号判断**：编号会被复用 —— 例如 `docs/14` 现在是
    "未纳入内容与共性分析"，而归档里的 `14-审计整改记录.md` 是另一份东西。
    所以这里同时认两件事：
      * **归档文件名**（`14-审计整改记录.md`）：正文提到它就该指向 `docs/archive/`；
      * **只存在于归档的编号**（如 `15` / `16`）：`docs/15` 这种引用一律报错。
    """
    archive = ROOT / "docs" / "archive"
    names = {p.name for p in archive.glob("[0-9][0-9]-*.md")}
    live = {p.name for p in (ROOT / "docs").glob("[0-9][0-9]-*.md")}
    only_archived = {name[:2] for name in names} - {name[:2] for name in live}
    return names, only_archived

ABSOLUTE_PREFIXES = ("D:/", "D:\\", "/home/", "/Users/", "file://")

LINK = re.compile(r"\]\(([^)]+)\)")

#: 正文里提到的源码文件（不一定是链接）——它们也必须真实存在，
#: 否则读者按名字去找会扑空。改名/删文件时最容易漏掉这种引用。
SOURCE_REF = re.compile(
    r"\b((?:sts2_sim|sts2_rl|tools|tests)/[A-Za-z_0-9]+\.py|train_combat\.py)")

LINE_SUFFIX = re.compile(r":(\d+)$")


def doc_files() -> list[Path]:
    files: list[Path] = []
    for pattern in DOC_GLOBS:
        files.extend(ROOT.glob(pattern))
    return sorted(set(files))


def _strip_line(target: str) -> str:
    return LINE_SUFFIX.sub("", target)


def _is_frozen(path: str) -> bool:
    return any(prefix in path for prefix in FROZEN_PREFIXES)


def check(verbose: bool = True) -> list[str]:
    """返回问题清单（空 = 干净）。"""
    problems: list[str] = []

    for doc in doc_files():
        text = doc.read_text(encoding="utf-8")
        rel_doc = doc.relative_to(ROOT).as_posix()

        # ---- 1/2. 链接与绝对路径 -------------------------------------
        for target in LINK.findall(text):
            if target.startswith(("#", "mailto:")):
                continue
            if target.startswith(ABSOLUTE_PREFIXES):
                problems.append(f"{rel_doc}: 绝对路径链接 —— {target}")
                continue
            if target.startswith(("http://", "https://")):
                continue
            path = _strip_line(target.split("#")[0].replace("\\", "/"))
            if not path:
                continue
            if not (doc.parent / path).exists():
                problems.append(f"{rel_doc}: 链接指向不存在的文件 —— {target}")

        # ---- 3. 引擎文件不该带行号 -----------------------------------
        for target in LINK.findall(text):
            path = target.split("#")[0].replace("\\", "/")
            if _is_frozen(path):
                continue
            if LINE_SUFFIX.search(path) and re.search(r"\.(py|md)(:\d+)$", path):
                problems.append(f"{rel_doc}: 引擎/文档引用带了易腐行号 —— {target}")

        # ---- 4. 不再引用已归档的东西 ---------------------------------
        # 对照表那一节是例外：它的用途就是写清旧编号去了哪（见 ALLOW_ARCHIVED_MARKER）。
        if ALLOW_ARCHIVED_MARKER not in text:
            archived_names, only_archived_numbers = archived_targets()
            for line in text.splitlines():
                if "归档" in line or "archive" in line:
                    continue
                hit = None
                for number in only_archived_numbers:
                    if f"docs/{number}" in line or f"docs\\{number}" in line:
                        hit = f"docs/{number}（编号只存在于归档）"
                        break
                if hit is None:
                    for name in archived_names:
                        if name in line:
                            hit = f"{name}（已归档，应指向 docs/archive/）"
                            break
                if hit is not None:
                    problems.append(
                        f"{rel_doc}: 引用了已归档文档 —— {hit} ← {line.strip()[:60]}")

        # ---- 5. 正文提到的源码文件必须存在 ---------------------------
        for ref in sorted(set(SOURCE_REF.findall(text))):
            if not (ROOT / ref).exists():
                problems.append(f"{rel_doc}: 提到的源码文件不存在 —— {ref}")

    if verbose:
        if problems:
            print(f"文档卫生：发现 {len(problems)} 处问题")
            for item in problems:
                print(f"  ✗ {item}")
        else:
            print(f"文档卫生：{len(doc_files())} 份文档全部通过 ✅")
    return problems


def _to_relative(doc: Path, target: str) -> str:
    """把绝对路径改成**相对本文件**的路径；引擎文件顺手去掉行号。

    ⚠️ 相对路径要按"从本文件所在目录出发"算：`docs/12.md` 引用
    `docs/14.md` 的正确写法是 `14.md`，不是 `../docs/14.md`。
    （早先版本用 `doc.parent` 的深度当 `..` 个数，于是 `docs/` 下的链接
    全多爬了一层 —— 这类错误肉眼很难发现，所以本工具自己也得被检查。）
    """
    import os

    for prefix in ABSOLUTE_PREFIXES:
        if not target.startswith(prefix):
            continue
        # file:// 之类的先不处理（文档里不该出现）
        if target.startswith("file://"):
            return target
        raw = Path(target)
        try:
            relative = raw.relative_to(ROOT)
        except ValueError:
            return target                        # 仓库外的路径：留着让人自己看
        result = Path(os.path.relpath(ROOT / relative, start=doc.parent)).as_posix()
        if not _is_frozen(result):
            result = _strip_line(result)          # 引擎文件的行号会腐烂
        return result
    return target


def fix() -> int:
    changed = 0
    for doc in doc_files():
        text = doc.read_text(encoding="utf-8")
        new_text = LINK.sub(
            lambda m: "](" + _to_relative(doc, m.group(1)) + ")",
            text)
        if new_text != text:
            doc.write_text(new_text, encoding="utf-8")
            changed += 1
            print(f"  改写 {doc.relative_to(ROOT).as_posix()}")
    print(f"文档链接：改写 {changed} 份")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="文档卫生检查")
    parser.add_argument("--fix", action="store_true",
                        help="把绝对路径改成相对路径并去掉易腐行号")
    args = parser.parse_args(argv)
    if args.fix:
        fix()
    return 1 if check() else 0


if __name__ == "__main__":
    sys.exit(main())
