"""
Unit tests for ConvexWaterFillingLeveler.
"""

import unittest
from apex_mcp_gateway_kernel.core.models import RateLimitBucket
from apex_mcp_gateway_kernel.core.water_filling_leveler import ConvexWaterFillingLeveler


class TestWaterFillingLeveler(unittest.TestCase):
    def setUp(self):
        self.leveler = ConvexWaterFillingLeveler(safety_margin_pct=0.05)
        self.buckets = [
            RateLimitBucket(
                provider_id="provider_a",
                capacity_tpm=100_000,
                available_tokens=80_000,
            ),
            RateLimitBucket(
                provider_id="provider_b",
                capacity_tpm=200_000,
                available_tokens=150_000,
            ),
        ]

    def test_burst_within_capacity(self):
        demanded = 50_000
        report = self.leveler.allocate_burst(demanded, self.buckets)
        self.assertEqual(report.total_burst_tokens, demanded)
        total_allocated = sum(report.allocations.values())
        self.assertEqual(total_allocated, demanded)
        self.assertEqual(report.http_429_drop_probability, 0.0)

    def test_burst_exceeding_capacity(self):
        demanded = 500_000  # Exceeds total available ~230k
        report = self.leveler.allocate_burst(demanded, self.buckets)
        self.assertGreater(report.http_429_drop_probability, 0.0)

    def test_zero_demanded(self):
        report = self.leveler.allocate_burst(0, self.buckets)
        self.assertEqual(sum(report.allocations.values()), 0)
        self.assertEqual(report.http_429_drop_probability, 0.0)


if __name__ == "__main__":
    unittest.main()
