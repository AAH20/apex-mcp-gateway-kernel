"""
Data models and typed structures for Apex MCP Gateway & Inference Router Kernel.
Zero external pip dependencies. Strict Python 3.10+ typing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple


@dataclass(slots=True)
class MCPToolDefinition:
    tool_name: str
    server_id: str
    description: str
    schema_tokens: int
    capabilities: Set[str]
    dependencies: List[str] = field(default_factory=list)
    is_mutation: bool = False
    latency_p50_ms: float = 50.0
    cost_per_call: float = 0.0


@dataclass(slots=True)
class SchemaKnapsackReport:
    total_tools_available: int
    selected_tools: List[MCPToolDefinition]
    unpruned_tokens: int
    pruned_tokens: int
    token_reduction_pct: float
    coverage_score: float
    lambda_penalty_converged: float
    solve_time_ms: float


@dataclass(slots=True)
class GPUWorkerNode:
    node_id: str
    cached_prefixes: Set[str]
    current_inflight_requests: int
    max_concurrency: int = 16
    gpu_device: str = "H100-SXM5-80GB"


@dataclass(slots=True)
class RadixCacheRouteDecision:
    selected_node_id: str
    longest_common_prefix_tokens: int
    prompt_tokens: int
    cache_hit_rate_pct: float
    estimated_ttft_ms: float
    route_latency_us: float


@dataclass(slots=True)
class ModelTierSpec:
    tier_id: str
    model_name: str
    input_cost_per_m: float
    output_cost_per_m: float
    p90_latency_ms: float
    base_reasoning_capability: float  # [0, 1]


@dataclass(slots=True)
class ConformalCascadeDecision:
    selected_tier: str
    model_name: str
    entropy_score: float
    conformal_risk_bound: float
    escalated_to_frontier: bool
    estimated_dollar_cost: float
    solve_latency_us: float


@dataclass(slots=True)
class AgentTaskLock:
    task_id: str
    resource_id: str
    priority: int
    is_exclusive: bool = True


@dataclass(slots=True)
class DeadlockDefusalReport:
    active_tasks: int
    cycles_detected: int
    preempted_task_id: Optional[str]
    safe_execution_order: List[str]
    arbitration_latency_us: float


@dataclass(slots=True)
class DistilledObservationReport:
    raw_tokens: int
    distilled_tokens: int
    compression_ratio_pct: float
    error_anchors_preserved: int
    distilled_text: str
    processing_time_us: float


@dataclass(slots=True)
class RateLimitBucket:
    provider_id: str
    capacity_tpm: int
    available_tokens: int
    cost_multiplier: float = 1.0


@dataclass(slots=True)
class WaterFillingReport:
    total_burst_tokens: int
    allocations: Dict[str, int]
    http_429_drop_probability: float
    solve_latency_us: float


@dataclass(slots=True)
class ApexMCPGatewayBenchmarkReport:
    total_runtime_ms: float
    timestamp: str
    schema_knapsack_summary: Dict[str, float]
    radix_cache_summary: Dict[str, float]
    conformal_cascade_summary: Dict[str, float]
    deadlock_lct_summary: Dict[str, float]
    submodular_distillation_summary: Dict[str, float]
    water_filling_summary: Dict[str, float]
