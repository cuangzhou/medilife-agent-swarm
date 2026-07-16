"""Machine-only benchmark for MediLife governed dynamic evidence memory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from memory.evidence_memory import EvidenceClaim, EvidenceEpisode, EvidenceMemoryService, EvidenceSource, EvidenceStatus
from result_contract import git_commit, make_result, write_result

DATASET = Path(__file__).parent / "datasets" / "medilife_longitudinal_v1.jsonl"


def episode(case: dict, suffix: str, text: str, *, user_id: str | None = None, high_risk: bool = False):
    source = EvidenceSource(f"src-{case['id']}-{suffix}", f"合成来源 {case['id']}", f"fixture://{case['id']}/{suffix}", "synthetic_guideline")
    return EvidenceEpisode.candidate(user_id=user_id or case["user_id"], session_id=f"s-{case['id']}-{suffix}", question=case["query"] if suffix == "old" else f"{case['topic']}复查更新", claims=[EvidenceClaim(f"claim-{case['id']}-{suffix}", text, 0.9, [source.source_id])], sources=[source], risk_level="high" if high_risk else "low", confidence=0.9)


def run(output: Path) -> dict:
    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    counters = {key: 0 for key in ["multihop", "conflict", "candidate_gate", "user_isolation", "source_trace", "supersede", "high_risk_gate", "irrelevant_robust", "irrelevant_admitted"]}
    latencies = []; details = []
    for case in cases:
        service = EvidenceMemoryService()
        old = episode(case, "old", case["seed_claim"]); service.add_candidate(old); service.promote(old.episode_id, verifier_passed=True)
        follow = episode(case, "follow", case["followup_claim"]); service.add_candidate(follow); service.promote(follow.episode_id, verifier_passed=True)
        service.store.add_relation(old.episode_id, follow.episode_id, "follows")
        service.mark_conflict(old.episode_id, follow.episode_id, "合成冲突 ground truth")
        candidate = episode(case, "candidate", "未经验证的候选结论"); service.add_candidate(candidate)
        foreign = episode(case, "foreign", "其他用户的证据", user_id=f"other-{case['user_id']}"); service.add_candidate(foreign); service.promote(foreign.episode_id, verifier_passed=True)
        unrelated = episode(case, "unrelated", "完全无关的皮肤护理记录"); unrelated.question = "皮肤护理记录"; service.add_candidate(unrelated); service.promote(unrelated.episode_id, verifier_passed=True)
        high = episode(case, "high", "高风险候选证据", high_risk=True); service.add_candidate(high)
        try:
            service.promote(high.episode_id, verifier_passed=True); high_blocked = False
        except PermissionError:
            high_blocked = True
        counters["high_risk_gate"] += int(high_blocked)
        started = time.perf_counter(); pack = service.search(case["query"], user_id=case["user_id"], top_k=6, max_hops=2); latencies.append((time.perf_counter() - started) * 1000)
        ids = {hit.episode.episode_id for hit in pack.hits}
        checks = {
            "multihop": follow.episode_id in ids,
            "conflict": bool(pack.conflicts),
            "candidate_gate": candidate.episode_id not in ids,
            "user_isolation": foreign.episode_id not in ids,
            "source_trace": all(hit.episode.sources and all(claim.source_ids for claim in hit.episode.claims) for hit in pack.hits),
            "irrelevant_robust": old.episode_id in ids and follow.episode_id in ids,
            "irrelevant_admitted": unrelated.episode_id in ids,
        }
        newer = episode(case, "new", f"新版本：{case['followup_claim']}"); service.add_candidate(newer); service.promote(newer.episode_id, verifier_passed=True); service.supersede(newer.episode_id, old.episode_id)
        checks["supersede"] = old.status is EvidenceStatus.SUPERSEDED and newer.status is EvidenceStatus.VERIFIED
        for key, passed in checks.items(): counters[key] += int(passed)
        details.append({"case_id": case["id"], **checks, "high_risk_gate": high_blocked})
    n = len(cases)
    metrics = {
        "graph_multihop_recall": counters["multihop"] / n,
        "conflict_detection_rate": counters["conflict"] / n,
        "unverified_candidate_exclusion_rate": counters["candidate_gate"] / n,
        "cross_user_leakage_rate": 1 - counters["user_isolation"] / n,
        "source_traceability_rate": counters["source_trace"] / n,
        "temporal_supersession_correctness": counters["supersede"] / n,
        "high_risk_auto_promotion_block_rate": counters["high_risk_gate"] / n,
        "irrelevant_memory_robustness_rate": counters["irrelevant_robust"] / n,
        "irrelevant_memory_admission_rate": counters["irrelevant_admitted"] / n,
        "mean_query_latency_ms": statistics.mean(latencies),
        "p95_query_latency_ms": sorted(latencies)[max(0, int(n * 0.95) - 1)],
    }
    result = make_result(project="MediLife", benchmark="governed-evidence-memory-innovation", dataset_version="medilife-longitudinal-v1", result_kind="measured", implementation_status="mvp", git_commit=git_commit(ROOT), command="python evaluation/innovation_benchmark.py", case_count=n, repetitions=1, seed=20260714, metrics=metrics, limitations=["全自动确定性工程评测，无人工或 LLM Judge", "使用匿名合成纵向病例，不代表临床有效性", "评测 Evidence Memory 治理与图关系逻辑，不冒充 Milvus/Mem0 端到端结果"])
    write_result(result, output)
    output.with_name(output.stem + "_cases.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); target = args.output or Path(__file__).parent / "artifacts" / "measured" / "medilife_innovation_machine_v1.json"
    print(json.dumps(run(target), ensure_ascii=False, indent=2))
