"""
Radix-Tree Token Prefix-Cache Affinity Router.
Routes incoming agent prompts to distributed GPU inference workers (vLLM / SGLang)
to maximize Key-Value (KV) cache reuse and minimize Time-to-First-Token (TTFT).
Zero external pip dependencies. Pure Python 3.10+.
"""

from __future__ import annotations
import math
import time
from typing import List, Dict, Set, Tuple, Optional
from apex_mcp_gateway_kernel.core.models import (
    GPUWorkerNode,
    RadixCacheRouteDecision,
)


class RadixTrieNode:
    """Node in token-level Radix prefix trie tracking worker KV-cache residency."""
    __slots__ = ("token_chunk", "children", "cached_by_workers")

    def __init__(self, token_chunk: str = ""):
        self.token_chunk = token_chunk
        self.children: Dict[str, RadixTrieNode] = {}
        self.cached_by_workers: Set[str] = set()


class RadixPrefixCacheRouter:
    """
    Sub-millisecond Radix-Tree Prefix Cache Affinity Router.
    Evaluates token-level Longest Common Prefix (LCP) against multi-GPU clusters.
    """

    def __init__(self, cache_affinity_weight: float = 0.75, base_ttft_ms: float = 850.0):
        self.alpha = cache_affinity_weight
        self.base_ttft = base_ttft_ms
        self.root = RadixTrieNode()

    def insert_prefix(self, prefix_tokens: List[str], worker_id: str) -> None:
        """Inserts token sequence into global Radix Trie and marks worker ownership."""
        curr = self.root
        curr.cached_by_workers.add(worker_id)

        for tok in prefix_tokens:
            if tok not in curr.children:
                curr.children[tok] = RadixTrieNode(token_chunk=tok)
            curr = curr.children[tok]
            curr.cached_by_workers.add(worker_id)

    def _compute_worker_lcp(self, prompt_tokens: List[str]) -> Dict[str, int]:
        """Traverses the Radix Trie to compute exact LCP token length for all workers."""
        worker_lcp: Dict[str, int] = {}
        curr = self.root

        for idx, tok in enumerate(prompt_tokens):
            if tok in curr.children:
                curr = curr.children[tok]
                for w in curr.cached_by_workers:
                    worker_lcp[w] = idx + 1
            else:
                break

        return worker_lcp

    def route_request(
        self,
        prompt_tokens: List[str],
        workers: List[GPUWorkerNode],
    ) -> RadixCacheRouteDecision:
        """
        Selects optimal GPU worker balancing KV-cache prefix hits against queue load.
        """
        start_t = time.perf_counter()
        total_tokens = max(1, len(prompt_tokens))

        if not workers:
            raise ValueError("No GPU workers available in cluster.")

        # Compute Longest Common Prefix (LCP) across Radix Trie
        worker_lcp = self._compute_worker_lcp(prompt_tokens)

        best_worker = workers[0]
        max_score = -float("inf")
        best_lcp = 0

        for w in workers:
            lcp = worker_lcp.get(w.node_id, 0)
            cache_hit_ratio = lcp / total_tokens

            # Capacity penalty: non-linear quadratic load term
            load_factor = w.current_inflight_requests / max(1, w.max_concurrency)
            load_penalty = (load_factor ** 2) if load_factor < 1.0 else 10.0

            # Convex score: alpha * cache_hit - (1 - alpha) * load_penalty
            score = (self.alpha * cache_hit_ratio) - ((1.0 - self.alpha) * load_penalty)

            if score > max_score:
                max_score = score
                best_worker = w
                best_lcp = lcp

        # Update cache state on chosen worker
        self.insert_prefix(prompt_tokens, best_worker.node_id)
        best_worker.cached_prefixes.add("".join(prompt_tokens[:min(10, total_tokens)]))

        hit_rate_pct = (best_lcp / total_tokens) * 100.0
        # TTFT = base_ttft * (1 - cache_hit_ratio) + queue_latency
        queue_latency = (best_worker.current_inflight_requests / best_worker.max_concurrency) * 150.0
        est_ttft = max(20.0, self.base_ttft * (1.0 - (hit_rate_pct / 100.0)) + queue_latency)

        elapsed_us = (time.perf_counter() - start_t) * 1_000_000.0

        return RadixCacheRouteDecision(
            selected_node_id=best_worker.node_id,
            longest_common_prefix_tokens=best_lcp,
            prompt_tokens=total_tokens,
            cache_hit_rate_pct=round(hit_rate_pct, 2),
            estimated_ttft_ms=round(est_ttft, 2),
            route_latency_us=round(elapsed_us, 2),
        )
