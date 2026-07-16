from pathlib import Path
import argparse

from result_contract import export_measured_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export measured metrics only")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_measured_metrics(args.source, args.output)
