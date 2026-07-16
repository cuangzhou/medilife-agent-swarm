import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory.entropy_manager import MemoryEntropyManager
from memory.evidence_memory import (
    EvidenceClaim,
    EvidenceEpisode,
    EvidenceMemoryService,
    EvidenceMemoryStore,
    EvidenceSource,
    EvidenceStatus,
)


def episode(user_id="user-a", question="高血压如何管理", risk="low", source=True):
    sources = [EvidenceSource("guideline-1", "高血压临床指南", "guideline://1", "guideline")] if source else []
    claims = [EvidenceClaim("claim-1", "应控制盐摄入", 0.9, ["guideline-1"])] if source else []
    return EvidenceEpisode.candidate(
        user_id=user_id,
        session_id="session-1",
        question=question,
        observations=["血压升高"],
        claims=claims,
        sources=sources,
        risk_level=risk,
        confidence=0.9,
    )


class EvidenceMemoryTests(unittest.TestCase):
    def setUp(self):
        self.service = EvidenceMemoryService()

    def test_content_contract_deduplicates_real_content(self):
        manager = MemoryEntropyManager()
        records = [
            {"memory_id": "1", "content": "问题：高血压怎么办"},
            {"memory_id": "2", "content": "问题：高血压怎么办"},
            {"memory_id": "3", "content": "问题：感冒怎么办"},
        ]
        self.assertEqual(2, len(manager.deduplicate_sessions(records)))

    def test_user_isolation_and_candidate_exclusion(self):
        a = self.service.add_candidate(episode("user-a"))
        self.service.promote(a.episode_id, verifier_passed=True)
        b = self.service.add_candidate(episode("user-b"))
        self.service.promote(b.episode_id, verifier_passed=True)
        self.assertEqual([a.episode_id], [hit.episode.episode_id for hit in self.service.search("高血压", user_id="user-a").hits])

        candidate = self.service.add_candidate(episode("user-a", "高血压候选记录"))
        hidden_ids = {hit.episode.episode_id for hit in self.service.search("候选记录", user_id="user-a").hits}
        self.assertNotIn(candidate.episode_id, hidden_ids)

    def test_verification_gate(self):
        no_source = self.service.add_candidate(episode(source=False))
        with self.assertRaises(ValueError):
            self.service.promote(no_source.episode_id, verifier_passed=True)

        high_risk = self.service.add_candidate(episode(risk="high"))
        with self.assertRaises(PermissionError):
            self.service.promote(high_risk.episode_id, verifier_passed=True)
        promoted = self.service.promote(high_risk.episode_id, manual_review=True)
        self.assertEqual(EvidenceStatus.VERIFIED, promoted.status)

    def test_graph_expansion_conflict_and_supersede(self):
        first = self.service.add_candidate(episode(question="胸痛检查"))
        second = self.service.add_candidate(episode(question="心电图结果"))
        third = self.service.add_candidate(episode(question="旧治疗方案"))
        for item in (first, second, third):
            self.service.promote(item.episode_id, verifier_passed=True)
        self.service.store.add_relation(first.episode_id, second.episode_id, "supports")
        self.service.mark_conflict(second.episode_id, third.episode_id, "指南结论不同")

        pack = self.service.search("胸痛", user_id="user-a", top_k=3, max_hops=2)
        ids = {hit.episode.episode_id for hit in pack.hits}
        self.assertIn(second.episode_id, ids)
        self.assertTrue(pack.conflicts)

        replacement = self.service.add_candidate(episode(question="新治疗方案"))
        self.service.promote(replacement.episode_id, verifier_passed=True)
        self.service.supersede(replacement.episode_id, third.episode_id)
        self.assertEqual(EvidenceStatus.SUPERSEDED, self.service.store.episodes[third.episode_id].status)

    def test_freshness_and_persistence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evidence.json"
            service = EvidenceMemoryService(EvidenceMemoryStore(str(path)))
            old = episode(question="过期高血压证据")
            old.observed_at = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
            old.freshness_days = 30
            service.add_candidate(old)
            service.promote(old.episode_id, verifier_passed=True)
            self.assertFalse(service.search("高血压", user_id="user-a").hits)

            reloaded = EvidenceMemoryStore(str(path))
            self.assertIn(old.episode_id, reloaded.episodes)


if __name__ == "__main__":
    unittest.main()
