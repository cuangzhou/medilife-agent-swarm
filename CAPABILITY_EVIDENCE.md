# Capability evidence

This page maps MediLife's public project claims to implementation and tests. MediLife is a medical-information and risk-guidance prototype, not a diagnostic device or a substitute for professional care.

| Capability | Implementation | Verification |
|---|---|---|
| Coordinator–Lead–Worker Swarm | routing in `swarm/swarm_coordinator.py`, decomposition and synthesis in `swarm/lead_agent.py`, parallel worker tasks through `asyncio.gather` | `tests/test_capability_parity.py`, capability-parity artifact |
| Three Worker Agents | consultation, diagnostic-risk, and research agents under `agents/` | `tests/test_capability_parity.py`, capability-parity artifact |
| Nine medical Skills | packaged `medilife_skills/*/SKILL.md` resources discovered through `core/skill_loader.py` | `tests/test_skills_and_resources.py` |
| Function calling and validation | OpenAI tool schemas plus strict Pydantic input/output contracts in `core/skill_registry.py` | type, extra-field, output, and nine-tool tests in `tests/test_skills_and_resources.py` |
| Milvus Lite RAG | optional vector adapter in `knowledge/milvus_kb.py` | adapter and resource tests; optional service checks are reported as `not_testable` when unavailable |
| Mem0 long-term memory | optional adapter in `memory/long_term.py`; disabled when credentials are absent | memory tests and capability-parity contract |
| DeepResearch and Trace Graph | workflow and evidence synthesis under `research/` | trace and capability-parity tests |
| Governed Evidence Memory | candidate/verified gate, relations, conflicts, provenance, high-risk manual gate, user isolation | `tests/test_evidence_memory.py`, `evaluation/innovation_benchmark.py` |
| FastAPI contract | chat, health, evidence search, and verification routes in `api_server.py` | `tests/test_capability_parity.py`, `tests/test_observability_stream.py` |

## Reproduce locally

```powershell
python -m pip install -e ".[test]"
python -m pytest
python evaluation/capability_parity.py --output evaluation/artifacts/capability-parity.json
python evaluation/evidence_memory_benchmark.py --mode full
python evaluation/innovation_benchmark.py
```

## Evidence boundaries

- The 50-case measured artifacts use anonymous synthetic longitudinal cases and deterministic assertions. They validate governance logic, not clinical effectiveness.
- The measured governance checks report 50/50 for multi-hop evidence retrieval, conflict recognition, temporal supersession, provenance tracking, and high-risk blocking, with 0/50 cross-user leakage for that dataset and version.
- Milvus Lite, Mem0, and web research are optional dependencies. Adapter availability is not reported as an end-to-end external-service result.
- The current retrieval implementation is vector-first. BM25, hybrid retrieval, and an independent reranker are not claimed as implemented.
- Production authentication, authorization, compliance controls, and clinical validation remain outside the current prototype boundary.
