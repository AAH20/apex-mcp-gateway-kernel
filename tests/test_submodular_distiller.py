"""
Unit tests for SubmodularObservationDistiller.
"""

import unittest
from apex_mcp_gateway_kernel.core.submodular_distiller import SubmodularObservationDistiller


class TestSubmodularObservationDistiller(unittest.TestCase):
    def setUp(self):
        self.distiller = SubmodularObservationDistiller(target_token_limit=150)

    def test_preserves_error_anchors(self):
        raw_log = """[2026-09-28 00:00:01] INFO: Service starting
[2026-09-28 00:00:02] DEBUG: Loading config from disk
[2026-09-28 00:00:03] DEBUG: Loading cache index
Traceback (most recent call last):
  File "/workspace/app.py", line 42, in run
    raise ValueError("Invalid schema version")
ValueError: Invalid schema version
exit code: 2
[2026-09-28 00:00:04] DEBUG: Thread pool cleanup complete
"""
        report = self.distiller.distill(raw_log)
        self.assertGreater(report.error_anchors_preserved, 0)
        self.assertIn("Traceback", report.distilled_text)
        self.assertIn("ValueError: Invalid schema version", report.distilled_text)
        self.assertIn("exit code: 2", report.distilled_text)
        self.assertGreaterEqual(report.compression_ratio_pct, 0.0)

    def test_empty_string(self):
        report = self.distiller.distill("")
        self.assertEqual(report.raw_tokens, 0)
        self.assertEqual(report.distilled_tokens, 0)
        self.assertEqual(report.compression_ratio_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
