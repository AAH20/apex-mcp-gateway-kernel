"""
Unit tests for DeadlockLCTDefuser.
"""

import unittest
from apex_mcp_gateway_kernel.core.models import AgentTaskLock
from apex_mcp_gateway_kernel.core.deadlock_lct_defuser import DeadlockLCTDefuser


class TestDeadlockLCTDefuser(unittest.TestCase):
    def setUp(self):
        self.defuser = DeadlockLCTDefuser()

    def test_acyclic_locks_no_deadlock(self):
        locks = [
            AgentTaskLock(task_id="agent_1", resource_id="res_a", priority=50),
            AgentTaskLock(task_id="agent_2", resource_id="res_b", priority=40),
            AgentTaskLock(task_id="agent_3", resource_id="res_c", priority=30),
        ]
        report = self.defuser.arbitrate_locks(locks)
        self.assertEqual(report.cycles_detected, 0)
        self.assertIsNone(report.preempted_task_id)
        self.assertEqual(len(report.safe_execution_order), 3)

    def test_circular_dependency_detected_and_preempted(self):
        # A wants res_b (held by B), B wants res_a (held by A)
        locks = [
            AgentTaskLock(task_id="agent_high", resource_id="res_a", priority=90),
            AgentTaskLock(task_id="agent_low", resource_id="res_b", priority=30),
            AgentTaskLock(task_id="agent_high", resource_id="res_b", priority=90),
            AgentTaskLock(task_id="agent_low", resource_id="res_a", priority=30),
        ]
        report = self.defuser.arbitrate_locks(locks)
        self.assertGreaterEqual(report.cycles_detected, 1)
        self.assertIsNotNone(report.preempted_task_id)
        self.assertEqual(len(report.safe_execution_order), 2)
        # Higher priority agent should be first in safe execution order
        self.assertEqual(report.safe_execution_order[0], "agent_high")


if __name__ == "__main__":
    unittest.main()
