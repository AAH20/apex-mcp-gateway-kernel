"""
Unit tests for AlienSchemaKnapsackPruner.
"""

import unittest
from apex_mcp_gateway_kernel.core.models import MCPToolDefinition
from apex_mcp_gateway_kernel.core.alien_schema_knapsack import AlienSchemaKnapsackPruner


class TestAlienSchemaKnapsack(unittest.TestCase):
    def setUp(self):
        self.pruner = AlienSchemaKnapsackPruner(gamma_coverage=1.8, max_bsearch_iters=35)
        self.tools = [
            MCPToolDefinition(
                tool_name="git_clone",
                server_id="mcp-git",
                description="Clone git repo",
                schema_tokens=400,
                capabilities={"git", "read"},
                dependencies=[],
            ),
            MCPToolDefinition(
                tool_name="git_push",
                server_id="mcp-git",
                description="Push git repo",
                schema_tokens=500,
                capabilities={"git", "mutation"},
                dependencies=["git_clone"],
                is_mutation=True,
            ),
            MCPToolDefinition(
                tool_name="sql_exec",
                server_id="mcp-sql",
                description="Execute SQL",
                schema_tokens=800,
                capabilities={"sql", "database"},
                dependencies=[],
                is_mutation=True,
            ),
            MCPToolDefinition(
                tool_name="k8s_deploy",
                server_id="mcp-k8s",
                description="Deploy pods",
                schema_tokens=1200,
                capabilities={"infra", "k8s"},
                dependencies=[],
                is_mutation=True,
            ),
            MCPToolDefinition(
                tool_name="file_read",
                server_id="mcp-fs",
                description="Read file",
                schema_tokens=250,
                capabilities={"fs", "read"},
                dependencies=[],
            ),
        ]

    def test_pruning_within_budget(self):
        query_caps = {"git", "mutation", "sql"}
        report = self.pruner.prune_schemas(self.tools, query_caps, token_budget=1000)

        self.assertLessEqual(report.pruned_tokens, 1000)
        self.assertGreater(len(report.selected_tools), 0)
        self.assertGreaterEqual(report.token_reduction_pct, 0.0)
        self.assertLess(report.solve_time_ms, 50.0)

    def test_transitive_dependency_resolution(self):
        query_caps = {"mutation"}  # git_push has mutation, depends on git_clone
        report = self.pruner.prune_schemas(self.tools, query_caps, token_budget=1200)
        names = {t.tool_name for t in report.selected_tools}
        if "git_push" in names:
            self.assertIn("git_clone", names)

    def test_unconstrained_budget_returns_all(self):
        report = self.pruner.prune_schemas(self.tools, {"git", "sql", "infra"}, token_budget=10_000)
        self.assertEqual(len(report.selected_tools), len(self.tools))
        self.assertEqual(report.token_reduction_pct, 0.0)
        self.assertEqual(report.pruned_tokens, report.unpruned_tokens)

    def test_empty_tools(self):
        report = self.pruner.prune_schemas([], {"git"}, token_budget=500)
        self.assertEqual(len(report.selected_tools), 0)
        self.assertEqual(report.pruned_tokens, 0)


if __name__ == "__main__":
    unittest.main()
