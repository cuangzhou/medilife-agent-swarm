"""Build a compact cross-project DeepSeek evaluation report from artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build(tifa: dict, medilife: dict) -> dict:
    tm = tifa["metrics"]
    mm = medilife["metrics"]["summary"]
    failed = [
        {"case_id": row["case_id"], "category": row["category"], "repetition": row["repetition"], "failed_checks": [key for key, value in row["checks"].items() if not value]}
        for row in medilife["metrics"]["cases"] if not row["passed"]
    ]
    tifa_gate = (
        tm["infrastructure_failure_rate"] <= 0.02
        and tm["duplicate_side_effect_rate"] == 0
        and tm["strict_pass_rate"] >= 0.8
    )
    return {
        "schema_version": "deepseek-dual-project-report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "deepseek-v4-flash",
        "temperature": 0,
        "projects": {
            "Tifa": {
                "artifact_status": tifa["status"], "code_version": tifa["provenance"]["code_version"],
                "executed": tm["executed_case_count"], "strict_pass_rate": tm["strict_pass_rate"],
                "infrastructure_failure_rate": tm["infrastructure_failure_rate"],
                "duplicate_side_effect_rate": tm["duplicate_side_effect_rate"],
                "failure_distribution": tm["failure_distribution"],
                "formal_100x3_started": False, "smoke_gate_passed": tifa_gate,
                "decision": "STOPPED_AT_SMOKE_GATE" if not tifa_gate else "ELIGIBLE_FOR_FORMAL_RUN",
            },
            "MediLife": {
                "artifact_status": medilife["result_kind"], "git_commit": medilife["git_commit"],
                "scheduled": mm["scheduled"], "completed": mm["completed"],
                "strict_passes": mm["strict_passes"], "strict_pass_rate": mm["strict_pass_rate"],
                "infrastructure_failures": mm["infrastructure_failures"], "usage": mm["usage"],
                "failed_cases": failed,
                "evidence_scope": "anonymous synthetic E2E routing/tool/trace/safety assertions; not clinical effectiveness",
            },
        },
        "comparison_policy": "Project scores are reported separately and are not combined into a single score.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tifa", type=Path, required=True)
    parser.add_argument("--medilife", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(load(args.tifa), load(args.medilife))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
