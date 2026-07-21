import json
from pathlib import Path

from evaluation.capability_parity import MANIFEST, run


def test_parity_manifest_is_versioned_and_has_required_capabilities():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    required = {"medical_skills", "worker_agents", "function_calling", "agent_loop", "swarm", "milvus_lite_adapter", "long_term_memory", "deep_research", "trace_graph", "api", "governed_evidence_memory"}
    assert payload["schema_version"] == "legacy-capability-parity.v1" and required == payload["capabilities"].keys()


def test_offline_parity_runner_passes_hard_contracts(tmp_path: Path):
    report = run(tmp_path / "parity.json")
    assert report["status"] == "passed" and report["network_used"] is False and report["real_llm_used"] is False
    hard = [probe for probe in report["probes"] if probe["name"] != "optional_adapters"]
    assert all(probe["status"] == "passed" for probe in hard)
    assert len(report["manifest_digest"]) == 64
