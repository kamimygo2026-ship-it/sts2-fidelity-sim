"""把"源码里抽不出效果、又不是诅咒/状态"的卡连同中英文文本一起打出来。

用途：**核对它们到底是"真有效果但抽不出"还是"本来就什么都不做"**。
只靠英文描述容易看走眼，中文文本更直观。
"""

from __future__ import annotations

import json
import pathlib
import re
import zipfile

CONTENT = pathlib.Path("data/content/repo")
ZHS = pathlib.Path("data/raw/spire-codex/repo/export_zhs.zip")


def clean(text: str | None) -> str:
    return re.sub(r"\[[^\]]*\]", "", text or "").replace("\n", " / ")


def main() -> None:
    source = {r["cid"]: r for r in
              json.loads((CONTENT / "cards_source.json").read_text(encoding="utf-8"))}
    # 归一化表用 cid；导出的双语表用大写 id
    english = {c["cid"]: c for c in
               json.loads((CONTENT / "cards.json").read_text(encoding="utf-8"))}
    with zipfile.ZipFile(ZHS) as archive:
        chinese = {c["id"].lower(): c for c in
                   json.loads(archive.read("cards.json").decode("utf-8"))}

    rows = []
    for cid, record in source.items():
        if record.get("effects") or record.get("triggers"):
            continue
        entry = english.get(cid)
        if entry is None:
            continue
        if (entry.get("card_type") or "").lower() in ("curse", "status"):
            continue
        zh = chinese.get(cid, {})
        rows.append((cid, entry.get("card_type"), entry.get("cost"),
                     zh.get("name", "?"), clean(zh.get("description")),
                     clean(entry.get("text")),
                     record.get("unsupported") or [],
                     record.get("choice_commands") or 0))
    rows.sort()
    print(f"共 {len(rows)} 张（源码无可转换效果，且不是诅咒/状态）\n")
    for cid, ctype, cost, zname, ztext, etext, unsupported, choices in rows:
        print("-" * 76)
        print(f"{cid:24s} {ctype or '?':8s} cost={cost}  中文名: {zname}")
        print(f"  中: {ztext[:160]}")
        print(f"  英: {etext[:160]}")
        tail = f"  选牌×{choices}" if choices else ""
        print(f"  抽不出的命令: {unsupported[:4]}{tail}")


if __name__ == "__main__":
    main()
