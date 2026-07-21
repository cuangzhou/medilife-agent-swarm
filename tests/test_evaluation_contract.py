from pathlib import Path
import sys
import tempfile
import unittest

EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))
from result_contract import export_measured_metrics, make_result, validate_result, write_result


class EvaluationContractTests(unittest.TestCase):
    def test_upstream_artifacts_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "upstream_artifacts"
            root.mkdir()
            source = root / "example.json"
            source.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "upstream example"):
                export_measured_metrics(source, root / "resume.json")

    def test_placeholder_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = make_result(project="MediLife", benchmark="test", dataset_version="v1", result_kind="placeholder", implementation_status="design", git_commit="test", command="test", case_count=1, repetitions=1, seed=1, metrics={}, limitations=["test"])
            source = root / "medilife_PLACEHOLDER_test.json"
            write_result(result, source)
            with self.assertRaisesRegex(ValueError, "REFUSED"):
                export_measured_metrics(source, root / "resume.json")

    def test_placeholder_requires_marked_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            result = make_result(project="MediLife", benchmark="test", dataset_version="v1", result_kind="placeholder", implementation_status="design", git_commit="test", command="test", case_count=1, repetitions=1, seed=1, metrics={}, limitations=["test"])
            with self.assertRaisesRegex(ValueError, "_PLACEHOLDER_"):
                write_result(result, Path(directory) / "unsafe.json")

    def test_aborted_result_is_valid_but_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            result = make_result(project="MediLife", benchmark="test", dataset_version="v1", result_kind="aborted", implementation_status="integrated", git_commit="test", command="test", case_count=1, repetitions=1, seed=1, metrics={"reason": "timeout"}, limitations=["test"])
            validate_result(result)
            source = Path(directory) / "aborted.json"
            write_result(result, source)
            with self.assertRaisesRegex(ValueError, "only measured"):
                export_measured_metrics(source, Path(directory) / "resume.json")


if __name__ == "__main__": unittest.main()
