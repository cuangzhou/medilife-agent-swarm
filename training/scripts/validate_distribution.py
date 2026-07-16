"""Dependency-free static validation for the isolated training distribution."""
from __future__ import annotations

import ast
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = [ROOT / "verl", ROOT / "examples", ROOT / "evaluation", ROOT / "LICENSE", ROOT / "MODIFICATIONS.md"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing training resources: {missing}")

    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in ROOT.rglob("*.yaml"):
        yaml.safe_load(path.read_text(encoding="utf-8"))
    for path in ROOT.rglob("*.yml"):
        yaml.safe_load(path.read_text(encoding="utf-8"))

    reward_dir = ROOT / "examples" / "reward_function"
    expected_rewards = {"dapo.py", "math.py", "medical.py", "r1v.py"}
    found = {path.name for path in reward_dir.glob("*.py")}
    if not expected_rewards <= found:
        raise SystemExit(f"missing reward functions: {sorted(expected_rewards - found)}")
    print("training distribution static validation passed")


if __name__ == "__main__":
    main()

