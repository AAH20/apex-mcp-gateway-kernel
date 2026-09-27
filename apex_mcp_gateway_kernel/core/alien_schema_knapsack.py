"""
Alien's Trick (WQS Binary Search) Submodular MCP Schema Knapsack Pruner.
Solves the Budgeted Maximum Coverage problem to select optimal MCP tools
under token context limits without quadratic dynamic programming overhead.
Zero external pip dependencies. Pure Python 3.10+.
"""

from __future__ import annotations
import math
import time
from typing import List, Dict, Set, Tuple, Optional
from apex_mcp_gateway_kernel.core.models import (
    MCPToolDefinition,
    SchemaKnapsackReport,
)


class AlienSchemaKnapsackPruner:
    """
    Sub-millisecond Alien's Trick Submodular Knapsack Pruner.
    Employs Lagrangian relaxation (WQS binary search on continuous penalty lambda)
    to enforce strict token budgets while maximizing query capability coverage.
    """

    def __init__(self, gamma_coverage: float = 1.8, max_bsearch_iters: int = 40):
        self.gamma = gamma_coverage
        self.max_iters = max_bsearch_iters

    def _resolve_dependencies(
        self,
        selected_tools: List[MCPToolDefinition],
        all_tools_map: Dict[str, MCPToolDefinition],
    ) -> List[MCPToolDefinition]:
        """Ensures all transitive tool prerequisites are included."""
        resolved: Set[str] = {t.tool_name for t in selected_tools}
        added = True
        while added:
            added = False
            for t_name in list(resolved):
                tool = all_tools_map.get(t_name)
                if tool:
                    for dep in tool.dependencies:
                        if dep in all_tools_map and dep not in resolved:
                            resolved.add(dep)
                            added = True
        return [all_tools_map[name] for name in resolved if name in all_tools_map]

    def _eval_unconstrained_lambda(
        self,
        tools: List[MCPToolDefinition],
        query_capabilities: Set[str],
        penalty_lambda: float,
    ) -> Tuple[List[MCPToolDefinition], float, int]:
        """
        Solves unconstrained penalized submodular objective for a fixed lambda:
        max_S [ SubmodularGain(S) - lambda * Tokens(S) ]
        """
        selected: List[MCPToolDefinition] = []
        covered_caps: Dict[str, int] = {c: 0 for c in query_capabilities}
        current_gain = 0.0
        current_tokens = 0

        # Sort candidate tools by marginal density: |caps in query| / (schema_tokens + 1e-4)
        candidates = sorted(
            tools,
            key=lambda t: len(t.capabilities.intersection(query_capabilities)) / (t.schema_tokens + 1e-4),
            reverse=True,
        )

        for tool in candidates:
            matched_caps = tool.capabilities.intersection(query_capabilities)
            if not matched_caps and not tool.is_mutation:
                continue

            # Marginal submodular gain: delta = sum_{c} [ log(1 + gamma*(cov+1)) - log(1 + gamma*cov) ]
            marginal_gain = 0.0
            for c in matched_caps:
                cov = covered_caps[c]
                delta_c = math.log(1.0 + self.gamma * (cov + 1.0)) - math.log(1.0 + self.gamma * cov)
                marginal_gain += delta_c

            if tool.is_mutation:
                marginal_gain += 0.5  # Mutation bonus for stateful tools

            # Net value after Lagrangian penalty: Net = MarginalGain - lambda * schema_tokens
            net_value = marginal_gain - (penalty_lambda * tool.schema_tokens)

            if net_value > 0.0:
                selected.append(tool)
                current_gain += marginal_gain
                current_tokens += tool.schema_tokens
                for c in matched_caps:
                    covered_caps[c] += 1

        return selected, current_gain, current_tokens

    def prune_schemas(
        self,
        tools: List[MCPToolDefinition],
        query_capabilities: Set[str],
        token_budget: int = 4000,
    ) -> SchemaKnapsackReport:
        """
        Executes WQS Binary Search over lambda to hit the target token budget.
        """
        start_t = time.perf_counter()
        total_tools_available = len(tools)
        unpruned_tokens = sum(t.schema_tokens for t in tools)
        tools_map = {t.tool_name: t for t in tools}

        if unpruned_tokens <= token_budget:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return SchemaKnapsackReport(
                total_tools_available=total_tools_available,
                selected_tools=tools,
                unpruned_tokens=unpruned_tokens,
                pruned_tokens=unpruned_tokens,
                token_reduction_pct=0.0,
                coverage_score=1.0,
                lambda_penalty_converged=0.0,
                solve_time_ms=round(elapsed_ms, 3),
            )

        # Binary search range for lambda
        # When lambda = 0: picks maximum tools (overflows budget)
        # When lambda = high: picks 0 tools
        low_lambda = 0.0
        high_lambda = 10.0
        best_selection: List[MCPToolDefinition] = []
        best_gain = 0.0
        best_tokens = 0

        for _ in range(self.max_iters):
            mid_lambda = (low_lambda + high_lambda) * 0.5
            sel, gain, tokens = self._eval_unconstrained_lambda(tools, query_capabilities, mid_lambda)

            if tokens <= token_budget:
                best_selection = sel
                best_gain = gain
                best_tokens = tokens
                # Try a smaller penalty to see if we can pack more tools
                high_lambda = mid_lambda
            else:
                # Exceeded budget, increase penalty lambda
                low_lambda = mid_lambda

            if abs(high_lambda - low_lambda) < 1e-6:
                break

        # Resolve transitive dependencies
        final_selected = self._resolve_dependencies(best_selection, tools_map)
        final_tokens = sum(t.schema_tokens for t in final_selected)
        reduction_pct = ((unpruned_tokens - final_tokens) / unpruned_tokens * 100.0) if unpruned_tokens > 0 else 0.0

        # Compute coverage score
        all_caps_needed = len(query_capabilities)
        covered_caps_count = sum(
            1 for c in query_capabilities
            if any(c in t.capabilities for t in final_selected)
        )
        cov_score = (covered_caps_count / max(1, all_caps_needed))

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return SchemaKnapsackReport(
            total_tools_available=total_tools_available,
            selected_tools=final_selected,
            unpruned_tokens=unpruned_tokens,
            pruned_tokens=final_tokens,
            token_reduction_pct=round(reduction_pct, 2),
            coverage_score=round(cov_score, 4),
            lambda_penalty_converged=round(low_lambda, 6),
            solve_time_ms=round(elapsed_ms, 3),
        )
