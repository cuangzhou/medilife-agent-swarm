import importlib
import os
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class BrandAndConfigTests(unittest.TestCase):
    def test_repository_has_no_retired_identifier(self):
        retired = "medi" + "x"
        excluded = {".git", "__pycache__", ".pytest_cache", "dist", "build"}
        allowed_files = {
            ROOT / "THIRD_PARTY_NOTICES.md",
            ROOT / "training" / "LICENSE",
            ROOT / "training" / "README.md",
        }
        for path in ROOT.rglob("*"):
            if not path.is_file() or any(part in excluded for part in path.parts):
                continue
            if path in allowed_files or "upstream_artifacts" in path.parts:
                continue
            if path.suffix.lower() in {".db", ".pyc"}:
                data = path.read_bytes().lower()
                self.assertNotIn(retired.encode(), data, str(path))
            else:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
                self.assertNotIn(retired, text, str(path))

    def test_environment_only_llm_config(self):
        with patch.dict(os.environ, {
            "LLM_API_KEY": "test-key",
            "LLM_MODEL_NAME": "test-model",
            "LLM_BASE_URL": "https://example.invalid/v1",
        }, clear=False):
            module = importlib.import_module("core.llm_client")
            config = module._load_llm_config()
        self.assertEqual(config["api_key"], "test-key")
        self.assertEqual(config["model_name"], "test-model")

    def test_api_brand_metadata(self):
        api = importlib.import_module("api_server")
        self.assertEqual(api.app.title, "MediLife Medical Assistant API")
        self.assertEqual(api.health_service_metadata()["service"], "medilife-medical-assistant")


if __name__ == "__main__":
    unittest.main()
