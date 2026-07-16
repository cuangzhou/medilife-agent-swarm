"""Generate clearly-labelled internal placeholder results and Markdown report."""
from pathlib import Path

from result_contract import git_commit, make_result, write_result


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "artifacts" / "placeholder"

GROUPS = {
    "milvus_static": {"recall_at_5": 0.68, "multi_hop_accuracy": 0.46, "temporal_consistency": 0.55, "conflict_detection_rate": 0.31, "source_coverage": 0.72, "unsupported_claim_rate": 0.18, "high_risk_recall": 0.64, "p50_latency_ms": 92, "p95_latency_ms": 151, "avg_tokens": 1210},
    "milvus_mem0": {"recall_at_5": 0.75, "multi_hop_accuracy": 0.58, "temporal_consistency": 0.67, "conflict_detection_rate": 0.47, "source_coverage": 0.78, "unsupported_claim_rate": 0.14, "high_risk_recall": 0.71, "p50_latency_ms": 117, "p95_latency_ms": 198, "avg_tokens": 1375},
    "evidence_graph": {"recall_at_5": 0.84, "multi_hop_accuracy": 0.73, "temporal_consistency": 0.82, "conflict_detection_rate": 0.79, "source_coverage": 0.89, "unsupported_claim_rate": 0.08, "high_risk_recall": 0.86, "p50_latency_ms": 149, "p95_latency_ms": 254, "avg_tokens": 1488},
}


def main() -> None:
    result = make_result(
        project="MediLife", benchmark="evidence-memory-comparison",
        dataset_version="medilife-longitudinal-placeholder-v1",
        result_kind="placeholder", implementation_status="design",
        git_commit=git_commit(ROOT), command="python evaluation/generate_placeholder.py",
        case_count=30, repetitions=3, seed=20260714, metrics={"groups": GROUPS},
        limitations=["INTERNAL_PLACEHOLDER", "模拟值，仅用于版式、接口和面试排练", "不是临床质量结论"],
    )
    path = OUT / "medilife_PLACEHOLDER_evidence_memory_v1.json"
    write_result(result, path)
    rows = ["# 内部占位数据，不得用于正式简历", "", "| 组别 | Recall@5 | 多跳正确率 | 冲突识别率 | P95 延迟 |", "|---|---:|---:|---:|---:|"]
    for name, m in GROUPS.items():
        rows.append(f"| {name} | {m['recall_at_5']:.0%}* | {m['multi_hop_accuracy']:.0%}* | {m['conflict_detection_rate']:.0%}* | {m['p95_latency_ms']} ms* |")
    rows += ["", "\\* INTERNAL_PLACEHOLDER：以上均为预期区间内的模拟值。"]
    (OUT / "medilife_PLACEHOLDER_evidence_memory_v1.md").write_text("\n".join(rows), encoding="utf-8")
    html_rows = "".join(f"<tr><td>{name}</td><td>{m['recall_at_5']:.0%}*</td><td>{m['multi_hop_accuracy']:.0%}*</td><td>{m['conflict_detection_rate']:.0%}*</td><td>{m['p95_latency_ms']} ms*</td></tr>" for name, m in GROUPS.items())
    html = f"<!doctype html><meta charset='utf-8'><title>INTERNAL_PLACEHOLDER</title><h1>内部占位数据，不得用于正式简历</h1><table><tr><th>组别</th><th>Recall@5</th><th>多跳正确率</th><th>冲突识别率</th><th>P95 延迟</th></tr>{html_rows}</table><p>* INTERNAL_PLACEHOLDER：以上均为预期区间内的模拟值。</p>"
    (OUT / "medilife_PLACEHOLDER_evidence_memory_v1.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
