"""审计回归：引擎**读到**的能力 id 必须与 ``powers.RULES`` 对得上（``docs/12`` 静默错一族）。

两类错都不会报错：

* 读到**不存在**的 id（拼写错）→ 那一条逻辑永远取到 0，静默失效；
* 读到了内容里存在、但 ``RULES`` 里**没登记**的能力 → 它的行为其实已经在消费者
  那一侧生效了（``orbs._focused`` 读 ``focus``），可门禁按 ``RULES`` 判断
  "能力未实现"，于是**所有施加它的卡都被拒**。这是反向的错：能力能用，卡进不来。

实现放在 ``tools/audit_power_refs.py``，测试只负责把它钉住。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestPowerRefs(unittest.TestCase):
    def test_every_read_power_id_exists_and_is_registered(self):
        from tools.audit_power_refs import audit

        typos, unregistered = audit()
        self.assertEqual(
            typos, [],
            "引擎读到了内容里不存在的能力 id（该分支会静默失效）："
            + ", ".join(f"{pid} ← {mods}" for pid, mods in typos))
        self.assertEqual(
            unregistered, [],
            "这些能力的行为已经在引擎里（有消费者读它），但没有登记进 powers.RULES，"
            "门禁会把所有施加它的卡拒掉："
            + ", ".join(f"{pid} ← {mods}" for pid, mods in unregistered))

    def test_focus_is_registered_and_consumer_matches_source(self):
        """``focus`` 是"消费者已在引擎里"的样例：登记它才能放行 ``biased_cognition``。"""
        from sts2_sim import powers
        self.assertIn("focus", powers.RULES)
        self.assertIn("ModifyOrbValue", powers.RULES["focus"].source)


if __name__ == "__main__":
    unittest.main()
