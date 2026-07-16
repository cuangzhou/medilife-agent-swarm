"""Shared benchmark result contract and safe resume-metric export helpers."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
from typing import Any


REQUIRED_FIELDS = {
    "project", "benchmark", "dataset_version", "result_kind",
    "implementation_status", "environment", "git_commit", "command",
    "case_count", "repetitions", "seed", "metrics", "limitations",
    "generated_at",
}


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL, timeout=3,
        ).strip()
    except Exception:
        return "unversioned"


def environment() -> dict[str, str]:
    return {"python": platform.python_version(), "platform": platform.platform()}


def make_result(**values: Any) -> dict[str, Any]:
    result = dict(values)
    result.setdefault("environment", environment())
    result.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    validate_result(result)
    return result


def validate_result(result: dict[str, Any]) -> None:
    missing = REQUIRED_FIELDS - result.keys()
    if missing:
        raise ValueError(f"benchmark result missing fields: {sorted(missing)}")
    if result["result_kind"] not in {"placeholder", "measured"}:
        raise ValueError("result_kind must be placeholder or measured")
    if result["implementation_status"] not in {"design", "mvp", "integrated"}:
        raise ValueError("invalid implementation_status")


def write_result(result: dict[str, Any], path: Path) -> None:
    validate_result(result)
    if result["result_kind"] == "placeholder" and "_PLACEHOLDER_" not in path.name:
        raise ValueError("placeholder filename must contain _PLACEHOLDER_")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def export_measured_metrics(source: Path, output: Path) -> None:
    if "upstream_artifacts" in source.parts:
        raise ValueError("REFUSED: upstream example artifacts are not MediLife measured evidence")
    result = json.loads(source.read_text(encoding="utf-8"))
    validate_result(result)
    if result["result_kind"] != "measured":
        raise ValueError("REFUSED: only measured benchmark results may be exported")
    payload = {
        "project": result["project"],
        "benchmark": result["benchmark"],
        "dataset_version": result["dataset_version"],
        "metrics": result["metrics"],
        "limitations": result["limitations"],
        "source": str(source.resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
