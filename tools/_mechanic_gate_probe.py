"""诊断（用完即删）：用候选的"缺失机制变量"集合，看会拦下哪些卡、它们的引擎效果是什么。"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2_sim.featurize import configure_content

configure_content("data/content/repo")
from sts2_sim import eligibility
from sts2_sim.content import CARD_DB

CANDIDATES = ("Repeat", "Increase", "Shivs", "PutBack", "PlayMax", "Cards",
              "CalculationBase", "CalculationExtra", "CalculatedCards",
              "CalculatedChannels", "Energy")
records = {r["cid"]: r for r in json.loads(
    Path("data/content/repo/cards_source.json").read_text(encoding="utf-8"))}

hits = []
for cid in sorted(eligibility.admitted_cards()):
    record = records.get(cid)
    if record is None:
        continue
    blob = json.dumps(record.get("effects") or [], ensure_ascii=False)
    blob += json.dumps(record.get("triggers") or [], ensure_ascii=False)
    for var in record.get("vars") or []:
        name = str(var.get("name") or "")
        if name not in CANDIDATES:
            continue
        if re.search(rf'"{name}"', blob) or re.search(rf"\.{name}\b", blob):
            continue
        card = CARD_DB[cid]
        hits.append((cid, name, [(e.op, e.amount, e.power) for e in card.effects]))

print(f"会被拦下 {len(hits)} 处（{len({h[0] for h in hits})} 张卡）")
for cid, name, effects in hits:
    print(f"  {cid:22s} {name:18s} 引擎效果={effects}")
