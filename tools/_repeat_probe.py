"""诊断（用完即删）：看 Repeat 形状的卡目前抽成什么。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.extract_cards import build_key_map, extract_effects, method_body, parse_vars

for name in ("Capacitor", "IceLance", "CloakAndDagger", "Quadcast",
             "BouncingFlask", "DeathsDoor"):
    path = list(Path("data/decompiled/sts2").rglob(f"{name}.cs"))[0]
    source = path.read_text(encoding="utf-8", errors="replace")
    body = method_body(source, "OnPlay")
    key_map = build_key_map(parse_vars(source))
    effects, unsupported, choices = extract_effects(body, key_map)
    print(f"== {name}")
    print(f"   effects     {effects}")
    print(f"   unsupported {unsupported}")
