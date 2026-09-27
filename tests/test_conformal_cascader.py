"""
Unit tests for ConformalParetoCascader.
"""

import unittest
from apex_mcp_gateway_kernel.core.models import ModelTierSpec
from apex_mcp_gateway_kernel.core.conformal_cascader import ConformalParetoCascader


class TestConformalParetoCascader(unittest.TestCase):
    def setUp(self):
        self.cascader = ConformalParetoCascader(conformal_risk_alpha=0.05, cost_sensitivity=20.0)
        self.tiers = [
            ModelTierSpec(
                tier_id="tier_draft",
                model_name="qwen-2.5-coder-7b",
                input_cost_per_m=0.20,
                output_cost_per_m=0.40,
                p90_latency_ms=100.0,
                base_reasoning_capability=0.60,
            ),
            ModelTierSpec(
                tier_id="tier_standard",
                model_name="claude-3-5-haiku",
                input_cost_per_m=0.80,
                output_cost_per_m=4.00,
                p90_latency_ms=250.0,
                base_reasoning_capability=0.85,
            ),
            ModelTierSpec(
                tier_id="tier_frontier",
                model_name="claude-3-7-sonnet",
                input_cost_per_m=3.00,
                output_cost_per_m=15.00,
                p90_latency_ms=600.0,
                base_reasoning_capability=0.98,
            ),
        ]
        self.cascader.register_tiers(self.tiers)

    def test_simple_read_query_routes_to_draft_or_standard(self):
        dec = self.cascader.evaluate_cascade(
            query_text="What is the git status?",
            tool_count=1,
            requires_mutation=False,
        )
        self.assertFalse(dec.escalated_to_frontier)
        self.assertLess(dec.estimated_dollar_cost, 0.001)
        self.assertLess(dec.entropy_score, 0.40)

    def test_complex_mutation_query_escalates_to_frontier(self):
        dec = self.cascader.evaluate_cascade(
            query_text="Execute atomic database partition re-indexing and rollback on lock timeout failure",
            tool_count=8,
            requires_mutation=True,
        )
        self.assertTrue(dec.escalated_to_frontier)
        self.assertEqual(dec.selected_tier, "tier_frontier")
        self.assertEqual(dec.model_name, "claude-3-7-sonnet")

    def test_unregistered_tiers_raises_error(self):
        fresh_cascader = ConformalParetoCascader()
        with self.assertRaises(RuntimeError):
            fresh_cascader.evaluate_cascade("test query", 1)


if __name__ == "__main__":
    unittest.main()
