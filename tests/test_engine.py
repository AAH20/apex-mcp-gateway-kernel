"""
Unit tests for ApexMCPGatewayEngine and full pipeline benchmark.
"""

import unittest
from apex_mcp_gateway_kernel.engine import (
    ApexMCPGatewayEngine,
    generate_synthetic_workload,
)


class TestApexMCPGatewayEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ApexMCPGatewayEngine()
        (
            self.tools,
            self.workers,
            self.tiers,
            self.locks,
            self.raw_obs,
            self.buckets,
        ) = generate_synthetic_workload()

    def test_synthetic_workload_generation(self):
        self.assertGreater(len(self.tools), 15)
        self.assertGreater(len(self.workers), 2)
        self.assertGreater(len(self.tiers), 2)
        self.assertGreater(len(self.locks), 3)
        self.assertIn("Traceback", self.raw_obs)
        self.assertGreater(len(self.buckets), 2)

    def test_end_to_end_methods(self):
        # 1. Prune
        k_rep = self.engine.prune_tools_for_agent(self.tools, {"git", "database"}, token_budget=3000)
        self.assertLessEqual(k_rep.pruned_tokens, 3000)

        # 2. Route
        r_dec = self.engine.route_to_gpu_cluster(["system_prompt_coding_agent", "def", "main"], self.workers)
        self.assertIn(r_dec.selected_node_id, [w.node_id for w in self.workers])

        # 3. Cascade
        c_dec = self.engine.cascade_inference("Check db connection", 3, self.tiers, requires_mutation=False)
        self.assertIsNotNone(c_dec.selected_tier)

        # 4. Deadlock
        d_rep = self.engine.arbitrate_agent_locks(self.locks)
        self.assertGreaterEqual(d_rep.cycles_detected, 1)

        # 5. Distill
        dist_rep = self.engine.distill_tool_observation(self.raw_obs)
        self.assertGreater(dist_rep.error_anchors_preserved, 0)

        # 6. Rate Level
        wf_rep = self.engine.level_burst_traffic(300_000, self.buckets)
        self.assertEqual(wf_rep.total_burst_tokens, 300_000)

    def test_run_full_pipeline_benchmark(self):
        report = self.engine.run_full_pipeline_benchmark(iterations=10)
        self.assertGreater(report.total_runtime_ms, 0.0)
        self.assertIn("avg_solve_time_ms", report.schema_knapsack_summary)
        self.assertIn("avg_route_latency_us", report.radix_cache_summary)
        self.assertIn("avg_solve_latency_us", report.conformal_cascade_summary)
        self.assertIn("avg_arbitration_latency_us", report.deadlock_lct_summary)
        self.assertIn("avg_distill_latency_us", report.submodular_distillation_summary)
        self.assertIn("avg_solve_latency_us", report.water_filling_summary)


if __name__ == "__main__":
    unittest.main()
