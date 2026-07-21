from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
MANIFEST = Path(__file__).with_name("legacy_capability_parity.v1.json")
EXPECTED_SKILLS = {"analyze-symptoms", "assess-risk", "clinical-guideline", "deep-research", "disease-code", "recommend-lifestyle", "search-history", "search-knowledge", "search-similar-cases"}


class MockLLM:
    async def chat(self, *_: Any, **__: Any) -> str:
        return "mock response"


def _probe(name: str, callback: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        details = callback()
        return {"name": name, "status": "passed", "details": details}
    except ImportError as exc:
        return {"name": name, "status": "not_testable", "details": {"reason": type(exc).__name__, "dependency": str(exc).split("'")[-2] if "'" in str(exc) else "optional"}}
    except Exception as exc:
        return {"name": name, "status": "failed", "details": {"reason": type(exc).__name__, "message": str(exc)[:300]}}


def _skills_and_agents() -> dict[str, Any]:
    from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
    from core.skill_loader import discover_skills
    skills = discover_skills(ROOT)
    names = {item["name"] for item in skills}
    if names != EXPECTED_SKILLS or not all(callable(item["function"]) for item in skills): raise AssertionError("packaged skill contract changed")
    agents = [ConsultationAgent(llm_client=MockLLM()), DiagnosticAgent(llm_client=MockLLM()), ResearchAgent(llm_client=MockLLM())]
    counts = {}
    for agent in agents:
        tools = agent.skill_registry.to_openai_format(); tool_names = {tool["function"]["name"].replace("_", "-") for tool in tools}
        if len(tools) != 9 or tool_names != EXPECTED_SKILLS or not all(tool["type"] == "function" for tool in tools): raise AssertionError(f"tool contract changed for {agent.agent_id}")
        counts[agent.agent_id] = len(tools)
    result = asyncio.run(agents[1].skill_registry.execute("search_history", session_id="parity-session", limit=1))
    if not isinstance(result, dict): raise AssertionError("skill execution contract changed")
    return {"skill_names": sorted(names), "agent_tool_counts": counts, "offline_execution": True}


def _runtime_symbols() -> dict[str, Any]:
    from core.agent_loop import AgentLoop
    from swarm.swarm_coordinator import SwarmCoordinator, process_with_swarm
    deep_research_source = (ROOT / "research" / "deep_research_workflow.py").read_text(encoding="utf-8")
    if not all(callable(item) for item in (AgentLoop, SwarmCoordinator, process_with_swarm)) or "class DeepResearchWorkflow" not in deep_research_source: raise AssertionError("runtime symbol missing")
    return {"agent_loop": True, "swarm": True, "deep_research": True}


def _trace_graph() -> dict[str, Any]:
    from research.trace_graph import TraceGraphBuilder
    graph = TraceGraphBuilder(); question = graph.add_node("question", "question", "safe", source="parity"); answer = graph.add_node("final_answer", "answer", "safe", source="parity"); graph.add_edge(question, answer, "synthesizes_into", "parity")
    payload = graph.to_dict()
    if len(payload.get("nodes", [])) != 2 or len(payload.get("edges", [])) != 1: raise AssertionError("trace graph contract changed")
    return {"nodes": 2, "edges": 1}


def _api_contract() -> dict[str, Any]:
    import api_server
    expected = {"/api/health", "/api/chat", "/api/evidence/search", "/api/evidence/{episode_id}/verify"}
    actual = {route.path for route in api_server.app.routes}
    if not expected <= actual: raise AssertionError(f"missing API routes: {sorted(expected - actual)}")
    health = api_server.health_service_metadata()
    if health.get("service") != "medilife-medical-assistant" or "api_key" in health: raise AssertionError("health metadata contract changed")
    fallback = api_server.build_fallback_trace("question", "answer", "single_agent", [])
    if not fallback.get("nodes") or not fallback.get("edges"): raise AssertionError("fallback trace missing")
    response_fields = {"answer", "session_id", "mode", "agents_involved", "trace_graph", "evidence_pack", "memory_candidates", "memory_delta", "conflicts", "disclaimer"}
    return {"routes": sorted(expected), "response_fields": sorted(response_fields), "fallback_trace": True}


def _evidence_memory() -> dict[str, Any]:
    from memory.evidence_memory import EvidenceClaim, EvidenceEpisode, EvidenceMemoryService, EvidenceMemoryStore, EvidenceSource
    with tempfile.TemporaryDirectory(prefix="medilife-parity-") as folder:
        service = EvidenceMemoryService(EvidenceMemoryStore(str(Path(folder) / "evidence.json")))
        def episode(user: str, suffix: str) -> Any:
            source = EvidenceSource(f"source-{suffix}", "guideline", f"guideline://{suffix}", "guideline")
            claim = EvidenceClaim(f"claim-{suffix}", "safe evidence", .9, [source.source_id])
            return EvidenceEpisode.candidate(user_id=user, session_id="session", question=f"question {suffix}", claims=[claim], sources=[source], risk_level="low", confidence=.9)
        candidate = service.add_candidate(episode("user-a", "candidate"))
        if any(hit.episode.episode_id == candidate.episode_id for hit in service.search("candidate", user_id="user-a").hits): raise AssertionError("candidate leaked into retrieval")
        verified_a = service.add_candidate(episode("user-a", "verified")); service.promote(verified_a.episode_id, verifier_passed=True)
        verified_b = service.add_candidate(episode("user-b", "verified")); service.promote(verified_b.episode_id, verifier_passed=True)
        ids = {hit.episode.episode_id for hit in service.search("verified", user_id="user-a").hits}
        if verified_a.episode_id not in ids or verified_b.episode_id in ids: raise AssertionError("user isolation failed")
    return {"candidate_exclusion": True, "user_isolation": True, "verification_gate": True}


def _optional_adapters() -> dict[str, Any]:
    from knowledge.milvus_kb import MedicalKnowledgeBase
    from memory.long_term import LongTermMemory
    return {"milvus_adapter": callable(MedicalKnowledgeBase), "long_term_memory_adapter": callable(LongTermMemory), "external_end_to_end": "not_testable_without_optional_services"}


def run(output: Path) -> dict[str, Any]:
    manifest_bytes = MANIFEST.read_bytes(); manifest = json.loads(manifest_bytes)
    probes = [_probe("skills_and_agents", _skills_and_agents), _probe("runtime_symbols", _runtime_symbols), _probe("trace_graph", _trace_graph), _probe("api_contract", _api_contract), _probe("evidence_memory", _evidence_memory), _probe("optional_adapters", _optional_adapters)]
    hard = [probe for probe in probes if probe["name"] != "optional_adapters"]
    status = "failed" if any(probe["status"] == "failed" for probe in hard) else "not_testable" if any(probe["status"] == "not_testable" for probe in hard) else "passed"
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: commit = "unversioned"
    report = {"schema_version": "legacy-capability-parity-report.v1", "status": status, "manifest_schema_version": manifest["schema_version"], "manifest_digest": hashlib.sha256(manifest_bytes).hexdigest(), "code_commit": commit, "generated_at": datetime.now(timezone.utc).isoformat(), "network_used": False, "real_llm_used": False, "probes": probes, "limitations": ["Optional Milvus, long-term-memory, and web-research services are adapter checks only unless separately provisioned.", "This report proves contract parity, not clinical validity."]}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"); return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); result = run(args.output); print(json.dumps(result, ensure_ascii=False, indent=2)); raise SystemExit(0 if result["status"] == "passed" else 1)
