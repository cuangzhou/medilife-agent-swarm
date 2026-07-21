import argparse
import asyncio
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "deepseek_e2e.py"
SPEC = importlib.util.spec_from_file_location("medilife_deepseek_e2e", MODULE_PATH)
e2e = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(e2e)


class DeepSeekE2ETests(unittest.TestCase):
    def test_selection_is_seed_stable_and_stratified(self):
        left = e2e.load_cases("smoke", 7)
        right = e2e.load_cases("smoke", 7)
        self.assertEqual([case["id"] for case in left], [case["id"] for case in right])
        self.assertEqual(len({case["category"] for case in left}), len(left))

    def test_verifier_checks_tools_safety_trace_and_leakage(self):
        case = {"required_tools": ["assess_risk"], "forbidden_tools": [], "high_risk": True, "require_swarm": True, "forbidden_marker": "SECRET"}
        result = {"answer": "请立即前往急诊。", "swarm_enabled": True, "tool_calls_history": [{"tool_name": "assess_risk"}], "trace_graph": {"nodes": [{}]}}
        self.assertTrue(e2e._verify(case, result)["passed"])

    def test_missing_credentials_produces_aborted_artifact(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = argparse.Namespace(provider="openai-compatible", model="deepseek-v4-flash", cases="compat", repetitions=1, seed=1, output=Path(directory) / "result.json", max_tokens=32, timeout=1.0)
            result = asyncio.run(e2e.evaluate(args))
            self.assertEqual(result["result_kind"], "aborted")
            self.assertFalse(result["metrics"]["manifest"]["credentials_present"])

    def test_run_case_does_not_persist_answer(self):
        case = e2e.load_cases("compat", 1)[0]
        fake = {"answer": "private raw answer", "swarm_enabled": False, "tool_calls_history": [{"tool_name": "analyze_symptoms"}], "trace_graph": {"nodes": [{}]}}
        client = type("Client", (), {"telemetry": {"request_ids": []}})()
        with patch.object(e2e, "process_with_swarm", AsyncMock(return_value=fake)):
            row = asyncio.run(e2e.run_case(case, 1, client, 1.0))
        self.assertNotIn("answer", row)
        self.assertTrue(row["passed"])


if __name__ == "__main__":
    unittest.main()
