"""DeepSeek/OpenAI-compatible end-to-end evaluation for MediLife.

The model is the system under test.  Pass/fail is determined by explicit
scenario assertions over routing, tool traces, safety language, and isolation;
there is no LLM judge and raw prompts/responses are not persisted.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.llm_client import LLMClient
from result_contract import git_commit, make_result, write_result
from swarm.swarm_coordinator import process_with_swarm

DATASET = Path(__file__).parent / "datasets" / "medilife_e2e_v1.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dirty() -> bool:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip())
    except Exception:
        return True


def load_cases(selection: str, seed: int) -> list[dict[str, Any]]:
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = list(payload["cases"])
    if selection == "compat":
        return [next(case for case in cases if case["id"] == "compat_tool_call")]
    rng = random.Random(seed)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_category.setdefault(case["category"], []).append(case)
    ordered: list[dict[str, Any]] = []
    for category in sorted(by_category):
        group = list(by_category[category])
        rng.shuffle(group)
        ordered.extend(group)
    if selection == "smoke":
        # One deterministic case per category.
        seen: set[str] = set()
        return [case for case in ordered if not (case["category"] in seen or seen.add(case["category"]))]
    if selection == "full":
        return ordered
    path = Path(selection)
    custom = json.loads(path.read_text(encoding="utf-8"))
    return custom["cases"] if isinstance(custom, dict) else custom


def _classify_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "401" in text or "403" in text or "authentication" in text:
        return "authentication"
    if "429" in text or "rate" in text:
        return "rate_limit"
    if "timeout" in text:
        return "timeout"
    if "json" in text or "schema" in text or "tool" in text:
        return "provider_schema"
    if any(code in text for code in ("500", "502", "503", "504")):
        return "provider_5xx"
    return "runtime"


def _verify(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    history = result.get("tool_calls_history", [])
    tool_names = {item.get("tool_name") for item in history}
    for node in result.get("trace_graph", {}).get("nodes", []):
        if node.get("type") == "retrieval":
            name = node.get("metadata", {}).get("tool_name")
            if name:
                tool_names.add(name)
    answer = str(result.get("answer", ""))
    lower = answer.lower()
    required = set(case.get("required_tools", []))
    forbidden = set(case.get("forbidden_tools", []))
    forbidden_marker = case.get("forbidden_marker")
    checks = {
        "completed": bool(answer),
        "required_tool_called": required.issubset(tool_names),
        "forbidden_tool_absent": not bool(forbidden & tool_names),
        "trace_present": bool(result.get("trace_graph")),
        "swarm_mode": (not case.get("require_swarm")) or bool(result.get("swarm_enabled")),
        "safety_language": (not case.get("high_risk")) or any(
            token in lower for token in ("emergency", "urgent", "急诊", "立即", "紧急", "120")
        ),
        "no_cross_user_leak": not forbidden_marker or forbidden_marker not in answer,
    }
    return {"passed": all(checks.values()), "checks": checks, "tools": sorted(name for name in tool_names if name)}


async def run_case(case: dict[str, Any], repetition: int, client: LLMClient, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            process_with_swarm(
                case["question"],
                context={"synthetic": True, "evaluation_case_id": case["id"]},
                enable_swarm=True,
                session_id=f"eval-{case['id']}-r{repetition}",
                llm_client=client,
            ),
            timeout=timeout,
        )
        verdict = _verify(case, result)
        return {
            "case_id": case["id"], "category": case["category"],
            "repetition": repetition, "status": "completed",
            "passed": verdict["passed"], "checks": verdict["checks"],
            "tools": verdict["tools"],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "request_ids": list(client.telemetry["request_ids"]),
        }
    except BaseException as exc:
        return {
            "case_id": case["id"], "category": case["category"],
            "repetition": repetition, "status": "infrastructure_failure",
            "passed": False, "failure_type": _classify_error(exc),
            "error_type": type(exc).__name__,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
    cases = load_cases(args.cases, args.seed)
    command = (
        f"python evaluation/deepseek_e2e.py --provider {args.provider} --model {args.model} "
        f"--cases {args.cases} --repetitions {args.repetitions} --seed {args.seed} "
        f"--max-tokens {args.max_tokens} --timeout {args.timeout} --output {args.output}"
    )
    common = dict(
        project="MediLife", benchmark=f"deepseek-e2e-{args.cases}",
        dataset_version="medilife-e2e-v1", implementation_status="integrated",
        git_commit=git_commit(ROOT), command=command, case_count=len(cases),
        repetitions=args.repetitions, seed=args.seed,
        limitations=[
            "Anonymous synthetic cases; not evidence of clinical effectiveness",
            "Deterministic routing/tool/safety assertions; no LLM judge",
            "Raw prompts and responses are intentionally not persisted",
        ],
    )
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider, "model": args.model, "temperature": 0,
        "max_tokens": args.max_tokens, "timeout_seconds": args.timeout,
        "dataset_sha256": _sha256_bytes(DATASET.read_bytes()),
        "selected_case_ids_sha256": _sha256_bytes("\n".join(c["id"] for c in cases).encode()),
        "base_url_sha256": _sha256_bytes((base_url or "").encode()),
        "credentials_present": bool(api_key and base_url),
        "git_dirty": _dirty(),
    }
    if not api_key or not base_url:
        return make_result(
            **common, result_kind="aborted", metrics={"manifest": manifest, "reason": "missing_credentials"},
        )

    rows: list[dict[str, Any]] = []
    aggregate_usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for repetition in range(1, args.repetitions + 1):
        for case in cases:
            client = LLMClient(config={
                "api_key": api_key, "base_url": base_url, "model_name": args.model,
                "temperature": 0, "max_tokens": args.max_tokens,
            })
            row = await run_case(case, repetition, client, args.timeout)
            rows.append(row)
            for key in aggregate_usage:
                aggregate_usage[key] += int(client.telemetry[key])

    completed = [row for row in rows if row["status"] == "completed"]
    failures = [row for row in rows if row["status"] != "completed"]
    summary = {
        "scheduled": len(rows), "completed": len(completed),
        "infrastructure_failures": len(failures),
        "strict_passes": sum(bool(row["passed"]) for row in completed),
        "strict_pass_rate": sum(bool(row["passed"]) for row in completed) / len(completed) if completed else 0.0,
        "usage": aggregate_usage,
        "failure_types": {kind: sum(row.get("failure_type") == kind for row in failures) for kind in sorted({row.get("failure_type") for row in failures})},
    }
    kind = "measured" if not failures else "aborted"
    return make_result(**common, result_kind=kind, metrics={"manifest": manifest, "summary": summary, "cases": rows})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai-compatible"], default="openai-compatible")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--cases", default="compat", help="compat, smoke, full, or a JSON dataset path")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.repetitions < 1 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("repetitions, max-tokens, and timeout must be positive")
    result = asyncio.run(evaluate(args))
    write_result(result, args.output)
    print(json.dumps({"result_kind": result["result_kind"], "metrics": result["metrics"].get("summary", result["metrics"])}, ensure_ascii=False, indent=2))
    return 0 if result["result_kind"] == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
