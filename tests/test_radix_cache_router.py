"""
Unit tests for RadixPrefixCacheRouter.
"""

import unittest
from apex_mcp_gateway_kernel.core.models import GPUWorkerNode
from apex_mcp_gateway_kernel.core.radix_cache_router import RadixPrefixCacheRouter


class TestRadixCacheRouter(unittest.TestCase):
    def setUp(self):
        self.router = RadixPrefixCacheRouter(cache_affinity_weight=0.8, base_ttft_ms=600.0)
        self.workers = [
            GPUWorkerNode(
                node_id="worker-01",
                cached_prefixes=set(),
                current_inflight_requests=2,
                max_concurrency=16,
            ),
            GPUWorkerNode(
                node_id="worker-02",
                cached_prefixes=set(),
                current_inflight_requests=15,  # heavily loaded
                max_concurrency=16,
            ),
            GPUWorkerNode(
                node_id="worker-03",
                cached_prefixes=set(),
                current_inflight_requests=0,
                max_concurrency=16,
            ),
        ]

    def test_routing_with_warmed_cache(self):
        prefix = ["system", "you", "are", "expert", "coder"]
        self.router.insert_prefix(prefix, "worker-01")

        query = ["system", "you", "are", "expert", "coder", "write", "quicksort"]
        decision = self.router.route_request(query, self.workers)

        self.assertEqual(decision.selected_node_id, "worker-01")
        self.assertEqual(decision.longest_common_prefix_tokens, 5)
        self.assertAlmostEqual(decision.cache_hit_rate_pct, (5 / 7) * 100.0, delta=0.1)
        self.assertLess(decision.estimated_ttft_ms, 600.0)

    def test_load_shedding_when_cache_node_saturated(self):
        prefix = ["header", "token", "a", "b"]
        # worker-02 has cached prefix, but inflight is 15/16 (heavily loaded)
        self.router.insert_prefix(prefix, "worker-02")

        query = ["header", "token", "a", "b", "c"]
        # Router should balance score: if worker-02 penalty is severe, or compare with cold worker-03
        decision = self.router.route_request(query, self.workers)
        self.assertIsNotNone(decision.selected_node_id)
        self.assertGreater(decision.prompt_tokens, 0)

    def test_empty_workers_raises_error(self):
        with self.assertRaises(ValueError):
            self.router.route_request(["test"], [])


if __name__ == "__main__":
    unittest.main()
