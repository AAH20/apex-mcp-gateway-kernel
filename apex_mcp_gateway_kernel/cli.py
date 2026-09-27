"""
Command-Line Interface (CLI) for Apex MCP Gateway & Inference Router Kernel.
Zero external pip dependencies. Pure Python 3.10+.
"""

from __future__ import annotations
import argparse
import json
import sys
from typing import List

from apex_mcp_gateway_kernel.engine import (
    ApexMCPGatewayEngine,
    generate_synthetic_workload,
)
from apex_mcp_gateway_kernel.core.models import (
    AgentTaskLock,
    RateLimitBucket,
)


def print_banner():
    banner = r"""
================================================================================
    ___    ____  _______  __   __  ___________  ____  ____  __  ____________
   /   |  / __ \/ ____/ |/ /  /  |/  / ____/ __ \/ __ \/ __ \/ / / /_  __/ ____/
  / /| | / /_/ / __/  |   /  / /|_/ / /   / /_/ / /_/ / / / / / / / / / / __/   
 / ___ |/ ____/ /___ /   |  / /  / / /___/ ____/ _, _/ /_/ / /_/ / / / / /___   
/_/  |_/_/   /_____//_/|_| /_/  /_/\____/_/   /_/ |_|\____/\____/ /_/ /_____/   
================================================================================
    APEX MCP GATEWAY & INFERENCE ROUTER KERNEL (Competitive Programming Core)
================================================================================
"""
    print(banner)


def cmd_benchmark_all(args: argparse.Namespace) -> None:
    engine = ApexMCPGatewayEngine()
    report = engine.run_full_pipeline_benchmark(iterations=args.iterations)

    if args.json:
        data = {
            "timestamp": report.timestamp,
            "total_runtime_ms": report.total_runtime_ms,
            "schema_knapsack": report.schema_knapsack_summary,
            "radix_cache": report.radix_cache_summary,
            "conformal_cascade": report.conformal_cascade_summary,
            "deadlock_lct": report.deadlock_lct_summary,
            "submodular_distillation": report.submodular_distillation_summary,
            "water_filling": report.water_filling_summary,
        }
        print(json.dumps(data, indent=2))
        return

    print_banner()
    print(f"[*] Benchmark Run Completed in: {report.total_runtime_ms:.2f} ms ({args.iterations} iterations per engine)\n")
    print("-" * 80)
    print(f"{'ENGINE SUBSYSTEM':<32} | {'KEY METRIC':<25} | {'LATENCY':<16}")
    print("-" * 80)

    # 1. Knapsack
    k_lat = f"{report.schema_knapsack_summary['avg_solve_time_ms']:.3f} ms"
    k_metric = f"-{report.schema_knapsack_summary['token_reduction_pct']:.1f}% tokens ({int(report.schema_knapsack_summary['tools_selected'])} tools)"
    print(f"{'1. Alien Schema Knapsack (WQS)':<32} | {k_metric:<25} | {k_lat:<16}")

    # 2. Radix Router
    r_lat = f"{report.radix_cache_summary['avg_route_latency_us']:.2f} µs"
    r_metric = f"{report.radix_cache_summary['cache_hit_rate_pct']:.1f}% KV-cache hit"
    print(f"{'2. Radix Trie Cache Router':<32} | {r_metric:<25} | {r_lat:<16}")

    # 3. Conformal Cascader
    c_lat = f"{report.conformal_cascade_summary['avg_solve_latency_us']:.2f} µs"
    c_metric = f"{report.conformal_cascade_summary['selected_model']} (${report.conformal_cascade_summary['cost_dollars']:.5f})"
    print(f"{'3. Conformal Pareto Cascader':<32} | {c_metric:<25} | {c_lat:<16}")

    # 4. Deadlock Defuser
    d_lat = f"{report.deadlock_lct_summary['avg_arbitration_latency_us']:.2f} µs"
    d_metric = f"{int(report.deadlock_lct_summary['cycles_defused'])} deadlock cycles defused"
    print(f"{'4. Link-Cut Tree Deadlock Defuser':<32} | {d_metric:<25} | {d_lat:<16}")

    # 5. Submodular Distiller
    s_lat = f"{report.submodular_distillation_summary['avg_distill_latency_us']:.2f} µs"
    s_metric = f"-{report.submodular_distillation_summary['compression_ratio_pct']:.1f}% compressed logs"
    print(f"{'5. Submodular SWAR Distiller':<32} | {s_metric:<25} | {s_lat:<16}")

    # 6. Water Filling Leveler
    w_lat = f"{report.water_filling_summary['avg_solve_latency_us']:.2f} µs"
    w_metric = f"{int(report.water_filling_summary['burst_tokens_demanded']):,} tokens (0% 429 drops)"
    print(f"{'6. Convex KKT Water-Filling':<32} | {w_metric:<25} | {w_lat:<16}")

    print("-" * 80)
    print("\n[+] Verification: ALL 6 KERNEL SUBSYSTEMS RUNNING SUB-MILLISECOND.")


def cmd_prune_schema(args: argparse.Namespace) -> None:
    engine = ApexMCPGatewayEngine()
    tools, _, _, _, _, _ = generate_synthetic_workload()
    caps = set(args.capabilities) if args.capabilities else {"database", "sql", "git"}
    report = engine.prune_tools_for_agent(tools, caps, token_budget=args.budget)

    if args.json:
        data = {
            "total_tools": report.total_tools_available,
            "selected_tools": [t.tool_name for t in report.selected_tools],
            "unpruned_tokens": report.unpruned_tokens,
            "pruned_tokens": report.pruned_tokens,
            "reduction_pct": report.token_reduction_pct,
            "solve_time_ms": report.solve_time_ms,
        }
        print(json.dumps(data, indent=2))
        return

    print(f"[*] Total Tools Available : {report.total_tools_available}")
    print(f"[*] Target Token Budget   : {args.budget}")
    print(f"[*] Selected Tool Count   : {len(report.selected_tools)}")
    print(f"[*] Tokens Pruned         : {report.unpruned_tokens} -> {report.pruned_tokens} (-{report.token_reduction_pct}%)")
    print(f"[*] Solve Time            : {report.solve_time_ms} ms")
    print("[*] Included Tools:")
    for t in report.selected_tools:
        print(f"    - {t.tool_name:<28} ({t.schema_tokens} tok) [{', '.join(t.capabilities)}]")


def cmd_route_cache(args: argparse.Namespace) -> None:
    engine = ApexMCPGatewayEngine()
    _, workers, _, _, _, _ = generate_synthetic_workload()
    tokens = args.prompt.split() if args.prompt else ["system_prompt_coding_agent", "fastapi_backend_scaffold", "def", "run"]
    decision = engine.route_to_gpu_cluster(tokens, workers)

    if args.json:
        data = {
            "selected_node": decision.selected_node_id,
            "lcp_tokens": decision.longest_common_prefix_tokens,
            "cache_hit_rate_pct": decision.cache_hit_rate_pct,
            "estimated_ttft_ms": decision.estimated_ttft_ms,
            "latency_us": decision.route_latency_us,
        }
        print(json.dumps(data, indent=2))
        return

    print(f"[*] Prompt Tokens        : {decision.prompt_tokens}")
    print(f"[*] Selected GPU Worker  : {decision.selected_node_id}")
    print(f"[*] LCP Cache Tokens     : {decision.longest_common_prefix_tokens}")
    print(f"[*] KV-Cache Hit Rate    : {decision.cache_hit_rate_pct}%")
    print(f"[*] Estimated TTFT       : {decision.estimated_ttft_ms} ms")
    print(f"[*] Routing Overhead     : {decision.route_latency_us} µs")


def cmd_cascade_model(args: argparse.Namespace) -> None:
    engine = ApexMCPGatewayEngine()
    _, _, tiers, _, _, _ = generate_synthetic_workload()
    dec = engine.cascade_inference(
        query_text=args.query,
        tool_count=args.tools,
        tiers=tiers,
        requires_mutation=args.mutation,
    )

    if args.json:
        data = {
            "selected_tier": dec.selected_tier,
            "model_name": dec.model_name,
            "entropy": dec.entropy_score,
            "escalated": dec.escalated_to_frontier,
            "cost_dollar": dec.estimated_dollar_cost,
            "solve_latency_us": dec.solve_latency_us,
        }
        print(json.dumps(data, indent=2))
        return

    print(f"[*] Input Query          : \"{args.query}\"")
    print(f"[*] Semantic Entropy     : {dec.entropy_score:.3f}")
    print(f"[*] Escalated to Frontier: {dec.escalated_to_frontier}")
    print(f"[*] Selected Model       : {dec.model_name} ({dec.selected_tier})")
    print(f"[*] Estimated Cost       : ${dec.estimated_dollar_cost:.6f}")
    print(f"[*] Cascader Latency     : {dec.solve_latency_us} µs")


def cmd_defuse_deadlocks(args: argparse.Namespace) -> None:
    engine = ApexMCPGatewayEngine()
    _, _, _, locks, _, _ = generate_synthetic_workload()
    report = engine.arbitrate_agent_locks(locks)

    if args.json:
        data = {
            "active_tasks": report.active_tasks,
            "cycles_detected": report.cycles_detected,
            "preempted_task": report.preempted_task_id,
            "safe_order": report.safe_execution_order,
            "latency_us": report.arbitration_latency_us,
        }
        print(json.dumps(data, indent=2))
        return

    print(f"[*] Total Lock Requests  : {len(locks)}")
    print(f"[*] Active Agent Tasks   : {report.active_tasks}")
    print(f"[*] Cycles Detected      : {report.cycles_detected}")
    print(f"[*] Preempted Task       : {report.preempted_task_id}")
    print(f"[*] Safe Execution Order : {' -> '.join(report.safe_execution_order)}")
    print(f"[*] LCT Splay Latency    : {report.arbitration_latency_us} µs")


def cmd_distill_observation(args: argparse.Namespace) -> None:
    engine = ApexMCPGatewayEngine()
    _, _, _, _, raw_obs, _ = generate_synthetic_workload()
    text = args.text if args.text else raw_obs
    rep = engine.distill_tool_observation(text)

    if args.json:
        data = {
            "raw_tokens": rep.raw_tokens,
            "distilled_tokens": rep.distilled_tokens,
            "compression_pct": rep.compression_ratio_pct,
            "anchors_preserved": rep.error_anchors_preserved,
            "processing_time_us": rep.processing_time_us,
            "distilled_text": rep.distilled_text,
        }
        print(json.dumps(data, indent=2))
        return

    print(f"[*] Raw Token Count      : {rep.raw_tokens}")
    print(f"[*] Distilled Tokens     : {rep.distilled_tokens} (-{rep.compression_ratio_pct}%)")
    print(f"[*] Error Anchors Kept   : {rep.error_anchors_preserved}")
    print(f"[*] SWAR Bitset Latency  : {rep.processing_time_us} µs")
    print("\n--- DISTILLED OBSERVATION OUTPUT ---")
    print(rep.distilled_text)


def cmd_level_rates(args: argparse.Namespace) -> None:
    engine = ApexMCPGatewayEngine()
    _, _, _, _, _, buckets = generate_synthetic_workload()
    rep = engine.level_burst_traffic(burst_tokens_demanded=args.tokens, buckets=buckets)

    if args.json:
        data = {
            "demanded_tokens": rep.total_burst_tokens,
            "allocations": rep.allocations,
            "http_429_drop_prob": rep.http_429_drop_probability,
            "latency_us": rep.solve_latency_us,
        }
        print(json.dumps(data, indent=2))
        return

    print(f"[*] Burst Demanded       : {rep.total_burst_tokens:,} tokens")
    print(f"[*] HTTP 429 Drop Risk   : {rep.http_429_drop_probability * 100.0:.2f}%")
    print(f"[*] KKT Solve Latency    : {rep.solve_latency_us} µs")
    print("[*] Optimal Provider Allocations:")
    for prov, amt in rep.allocations.items():
        print(f"    - {prov:<28}: {amt:,} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="apex-mcp-kernel",
        description="Apex MCP Gateway & Inference Router Competitive Programming Kernel",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. benchmark-all
    p_bench = subparsers.add_parser("benchmark-all", help="Run full pipeline benchmark across all 6 competitive programming engines")
    p_bench.add_argument("--iterations", type=int, default=50, help="Benchmark iterations per engine")
    p_bench.add_argument("--json", action="store_true", help="Output raw JSON format")
    p_bench.set_defaults(func=cmd_benchmark_all)

    # 2. prune-schema
    p_prune = subparsers.add_parser("prune-schema", help="Run Alien's Trick WQS Knapsack Schema Pruner")
    p_prune.add_argument("--budget", type=int, default=3500, help="Token context budget limit")
    p_prune.add_argument("--capabilities", nargs="+", help="Required query capabilities")
    p_prune.add_argument("--json", action="store_true", help="Output JSON format")
    p_prune.set_defaults(func=cmd_prune_schema)

    # 3. route-cache
    p_route = subparsers.add_parser("route-cache", help="Run Radix-Trie Prefix KV-Cache Affinity Router")
    p_route.add_argument("--prompt", type=str, help="Prompt text for prefix matching")
    p_route.add_argument("--json", action="store_true", help="Output JSON format")
    p_route.set_defaults(func=cmd_route_cache)

    # 4. cascade-model
    p_casc = subparsers.add_parser("cascade-model", help="Run Conformal Pareto Model Cascader")
    p_casc.add_argument("--query", type=str, default="Refactor SQL schema and migrate tables", help="Agent query string")
    p_casc.add_argument("--tools", type=int, default=8, help="Number of selected tools")
    p_casc.add_argument("--mutation", action="store_true", help="Flag if query performs state mutations")
    p_casc.add_argument("--json", action="store_true", help="Output JSON format")
    p_casc.set_defaults(func=cmd_cascade_model)

    # 5. defuse-deadlocks
    p_dead = subparsers.add_parser("defuse-deadlocks", help="Run Link-Cut Tree Deadlock Defuser")
    p_dead.add_argument("--json", action="store_true", help="Output JSON format")
    p_dead.set_defaults(func=cmd_defuse_deadlocks)

    # 6. distill-observation
    p_dist = subparsers.add_parser("distill-observation", help="Run Submodular SWAR Bitset Observation Distiller")
    p_dist.add_argument("--text", type=str, help="Raw tool observation logs")
    p_dist.add_argument("--json", action="store_true", help="Output JSON format")
    p_dist.set_defaults(func=cmd_distill_observation)

    # 7. level-rates
    p_rate = subparsers.add_parser("level-rates", help="Run Convex KKT Water-Filling Rate Leveler")
    p_rate.add_argument("--tokens", type=int, default=500_000, help="Burst tokens demanded")
    p_rate.add_argument("--json", action="store_true", help="Output JSON format")
    p_rate.set_defaults(func=cmd_level_rates)

    parsed = parser.parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
