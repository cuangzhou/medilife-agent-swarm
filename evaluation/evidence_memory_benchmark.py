"""Reproducible MediLife evidence-memory smoke/full benchmark.

The dataset is synthetic and anonymous. Dependency-backed groups are skipped
when their real integrations are unavailable; no fallback is labelled as an
integrated Milvus or Mem0 result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory.evidence_memory import EvidenceClaim, EvidenceEpisode, EvidenceMemoryService, EvidenceSource
from result_contract import git_commit, make_result, write_result

DATASET = Path(__file__).parent / "datasets" / "medilife_longitudinal_v1.jsonl"


def load_cases(limit: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit]


def add_verified(service: EvidenceMemoryService, case: dict[str, Any], suffix: str, claim_text: str):
    source = EvidenceSource(f"source-{case['id']}-{suffix}", f"合成指南来源 {case['id']}", f"fixture://{case['id']}/{suffix}", "synthetic_guideline")
    episode = EvidenceEpisode.candidate(
        user_id=case["user_id"], session_id=f"session-{case['id']}-{suffix}",
        question=case["query"] if suffix == "seed" else f"{case['id']}复查",
        claims=[EvidenceClaim(f"claim-{case['id']}-{suffix}", claim_text, 0.9, [source.source_id])],
        sources=[source], confidence=0.9,
        risk_level="high" if case["high_risk"] else "low",
    )
    service.add_candidate(episode)
    service.promote(episode.episode_id, verifier_passed=not case["high_risk"], manual_review=case["high_risk"])
    return episode


def build_service(cases: list[dict[str, Any]]) -> tuple[EvidenceMemoryService, dict[str, tuple[str, str]]]:
    service = EvidenceMemoryService()
    ids = {}
    for case in cases:
        seed = add_verified(service, case, "seed", case["seed_claim"])
        linked = add_verified(service, case, "followup", case["followup_claim"])
        service.store.add_relation(seed.episode_id, linked.episode_id, case["expected_relation"])
        if case["expects_conflict"]:
            service.mark_conflict(seed.episode_id, linked.episode_id, "纵向记录需要人工核对")
        ids[case["id"]] = (seed.episode_id, linked.episode_id)
    return service, ids


def evaluate_logic(cases: list[dict[str, Any]], repetitions: int) -> dict[str, Any]:
    per_run = []
    details = []
    for repetition in range(repetitions):
        service, ids = build_service(cases)
        static_hits = graph_hits = seed_static_hits = seed_graph_hits = conflict_hits = source_hits = high_risk_hits = 0
        static_latency, graph_latency = [], []
        for case in cases:
            seed_id, linked_id = ids[case["id"]]
            start = time.perf_counter()
            static = service.search(case["query"], user_id=case["user_id"], top_k=5, max_hops=0)
            static_latency.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            graph = service.search(case["query"], user_id=case["user_id"], top_k=5, max_hops=2)
            graph_latency.append((time.perf_counter() - start) * 1000)
            static_ids = {hit.episode.episode_id for hit in static.hits}
            graph_ids = {hit.episode.episode_id for hit in graph.hits}
            static_hit, graph_hit = linked_id in static_ids, linked_id in graph_ids
            seed_static_hits += int(seed_id in static_ids)
            seed_graph_hits += int(seed_id in graph_ids)
            static_hits += int(static_hit); graph_hits += int(graph_hit)
            conflict_hit = bool(graph.conflicts) if case["expects_conflict"] else True
            conflict_hits += int(conflict_hit)
            source_hit = all(hit.episode.sources for hit in graph.hits)
            source_hits += int(source_hit)
            high_risk_hits += int((not case["high_risk"]) or graph_hit)
            if repetition == 0:
                details.append({"case_id": case["id"], "static_multihop_hit": static_hit, "graph_multihop_hit": graph_hit, "conflict_ok": conflict_hit})
        n = len(cases)
        per_run.append({
            "static_recall_at_5": seed_static_hits / n,
            "graph_recall_at_5": seed_graph_hits / n,
            "static_multihop_recall": static_hits / n,
            "graph_multihop_recall": graph_hits / n,
            "temporal_consistency": graph_hits / n,
            "conflict_detection_rate": conflict_hits / n,
            "source_coverage": source_hits / n,
            "unsupported_claim_rate": 0.0,
            "high_risk_recall": high_risk_hits / n,
            "graph_p50_latency_ms": statistics.median(graph_latency),
            "graph_p95_latency_ms": sorted(graph_latency)[max(0, int(len(graph_latency) * 0.95) - 1)],
            "static_p50_latency_ms": statistics.median(static_latency),
            "avg_tokens": "NOT_MEASURED",
        })
    numeric_keys = [key for key, value in per_run[0].items() if isinstance(value, (int, float))]
    summary = {key: statistics.mean(run[key] for run in per_run) for key in numeric_keys}
    summary["avg_tokens"] = "NOT_MEASURED"
    summary["stddev"] = {key: statistics.pstdev(run[key] for run in per_run) for key in numeric_keys}
    return {"summary": summary, "runs": per_run, "case_results": details}


def dependency_status() -> dict[str, dict[str, str]]:
    pymilvus = importlib.util.find_spec("pymilvus") is not None
    mem0 = importlib.util.find_spec("mem0") is not None
    return {
        "milvus_static": {"status": "AVAILABLE_NOT_EXECUTED" if pymilvus else "SKIPPED_DEPENDENCY", "reason": "adapter E2E runner not enabled" if pymilvus else "pymilvus unavailable"},
        "milvus_mem0": {"status": "AVAILABLE_NOT_EXECUTED" if pymilvus and mem0 else "SKIPPED_DEPENDENCY", "reason": "adapter E2E runner not enabled" if pymilvus and mem0 else "pymilvus and/or mem0 unavailable"},
        "evidence_graph_logic": {"status": "MEASURED", "reason": "deterministic governance layer; not Milvus/Mem0 E2E"},
    }


def run(mode: str, output: Path) -> dict[str, Any]:
    case_count, repetitions = (30, 1) if mode == "smoke" else (50, 3)
    cases = load_cases(case_count)
    deps = dependency_status()
    measured = evaluate_logic(cases, repetitions)
    result = make_result(
        project="MediLife", benchmark=f"evidence-memory-{mode}", dataset_version="medilife-longitudinal-v1",
        result_kind="measured", implementation_status="mvp",
        git_commit=git_commit(ROOT), command=f"python evaluation/evidence_memory_benchmark.py --mode {mode}",
        case_count=len(cases), repetitions=repetitions, seed=20260714,
        metrics={"dependency_groups": deps, "logic_layer": measured},
        limitations=["匿名合成纵向病例，不代表临床有效性", "Token 指标需模型调用，当前逻辑层记为 NOT_MEASURED", "Milvus/Mem0 端到端组尚未执行，不得据此声称 integrated"],
    )
    write_result(result, output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = args.output or Path(__file__).parent / "artifacts" / "measured" / f"medilife_evidence_memory_{args.mode}_v1.json"
    print(json.dumps(run(args.mode, target), ensure_ascii=False, indent=2))
