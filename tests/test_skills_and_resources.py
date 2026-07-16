import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
from core.skill_loader import discover_skills
from knowledge.resources import bundled_knowledge_db, prepare_knowledge_db


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SKILLS = {
    "analyze-symptoms", "assess-risk", "clinical-guideline", "deep-research",
    "disease-code", "recommend-lifestyle", "search-history", "search-knowledge",
    "search-similar-cases",
}


class MockLLM:
    async def chat(self, *args, **kwargs):
        return "mock response"


class SkillsAndResourcesTests(unittest.TestCase):
    def test_discovers_all_packaged_skills(self):
        skills = discover_skills(ROOT)
        self.assertEqual({item["name"] for item in skills}, EXPECTED_SKILLS)
        for item in skills:
            self.assertTrue(callable(item["function"]))
            self.assertTrue(item["metadata"].get("description"))

    def test_every_worker_registers_nine_openai_tools(self):
        agents = [
            ConsultationAgent(llm_client=MockLLM()),
            DiagnosticAgent(llm_client=MockLLM()),
            ResearchAgent(llm_client=MockLLM()),
        ]
        for agent in agents:
            tools = agent.skill_registry.to_openai_format()
            self.assertEqual(len(tools), 9, agent.agent_id)
            self.assertEqual({tool["function"]["name"].replace("_", "-") for tool in tools}, EXPECTED_SKILLS)
            self.assertTrue(all(tool["type"] == "function" for tool in tools))

    def test_tool_execution_without_external_service(self):
        agent = DiagnosticAgent(llm_client=MockLLM())
        result = asyncio.run(agent.skill_registry.execute("search_history", session_id="test-session", limit=1))
        self.assertIsInstance(result, dict)
        self.assertIn("total_messages", result)

    def test_bundled_database_copies_to_writable_runtime(self):
        self.assertTrue(bundled_knowledge_db().is_file())
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MEDILIFE_DATA_DIR": directory}):
            target = prepare_knowledge_db()
            self.assertEqual(target.parent, Path(directory))
            self.assertEqual(target.read_bytes(), bundled_knowledge_db().read_bytes())


if __name__ == "__main__":
    unittest.main()

