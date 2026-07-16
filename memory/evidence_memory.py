"""Governed, traceable evidence memory for longitudinal medical interactions.

Milvus (or another adapter) supplies semantic candidates.  This module owns the
relationship graph, temporal/conflict filters, and promotion safety gate.  It
has a deterministic lexical fallback so the governance layer remains testable
without model downloads or external services.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set
import uuid


SCHEMA_VERSION = "evidence-memory.v1"
HIGH_RISK_LEVELS = {"high", "critical", "emergency"}
RELATION_TYPES = {
    "supports", "contradicts", "derived_from", "follows", "supersedes", "similar_to"
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceStatus(str, Enum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


@dataclass
class EvidenceSource:
    source_id: str
    title: str
    uri: str = ""
    source_type: str = "unknown"
    published_at: Optional[str] = None


@dataclass
class EvidenceClaim:
    claim_id: str
    text: str
    confidence: float = 0.0
    source_ids: List[str] = field(default_factory=list)


@dataclass
class EvidenceEpisode:
    episode_id: str
    user_id: str
    session_id: str
    question: str
    observations: List[str] = field(default_factory=list)
    claims: List[EvidenceClaim] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    sources: List[EvidenceSource] = field(default_factory=list)
    observed_at: str = field(default_factory=utc_now)
    valid_at: Optional[str] = None
    risk_level: str = "low"
    confidence: float = 0.0
    status: EvidenceStatus = EvidenceStatus.CANDIDATE
    freshness_days: int = 90
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def candidate(
        cls,
        *,
        user_id: str,
        session_id: str,
        question: str,
        observations: Optional[Sequence[str]] = None,
        claims: Optional[Sequence[EvidenceClaim]] = None,
        recommendations: Optional[Sequence[str]] = None,
        sources: Optional[Sequence[EvidenceSource]] = None,
        risk_level: str = "low",
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvidenceEpisode":
        return cls(
            episode_id=f"episode_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            session_id=session_id,
            question=question,
            observations=list(observations or []),
            claims=list(claims or []),
            recommendations=list(recommendations or []),
            sources=list(sources or []),
            risk_level=risk_level.lower(),
            confidence=max(0.0, min(1.0, confidence)),
            metadata=dict(metadata or {}),
        )

    def searchable_text(self) -> str:
        fields = [self.question, *self.observations]
        fields.extend(claim.text for claim in self.claims)
        fields.extend(self.recommendations)
        fields.extend(source.title for source in self.sources)
        return " ".join(part for part in fields if part)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvidenceEpisode":
        data = dict(payload)
        data["claims"] = [EvidenceClaim(**item) for item in data.get("claims", [])]
        data["sources"] = [EvidenceSource(**item) for item in data.get("sources", [])]
        data["status"] = EvidenceStatus(data.get("status", EvidenceStatus.CANDIDATE.value))
        return cls(**data)


@dataclass
class EvidenceRelation:
    relation_id: str
    source_episode_id: str
    target_episode_id: str
    relation_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.relation_type not in RELATION_TYPES:
            raise ValueError(f"unsupported evidence relation: {self.relation_type}")


@dataclass
class EvidenceHit:
    episode: EvidenceEpisode
    score: float
    matched_by: str
    relation_path: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode": self.episode.to_dict(),
            "score": self.score,
            "matched_by": self.matched_by,
            "relation_path": self.relation_path,
        }


@dataclass
class EvidencePack:
    query: str
    user_id: str
    hits: List[EvidenceHit]
    conflicts: List[Dict[str, Any]]
    generated_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "user_id": self.user_id,
            "hits": [hit.to_dict() for hit in self.hits],
            "conflicts": self.conflicts,
            "generated_at": self.generated_at,
            "schema_version": self.schema_version,
        }


class EvidenceMemoryStore:
    """Small event-style store with optional JSON persistence."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self.episodes: Dict[str, EvidenceEpisode] = {}
        self.relations: Dict[str, EvidenceRelation] = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported evidence memory schema")
        self.episodes = {
            item["episode_id"]: EvidenceEpisode.from_dict(item)
            for item in payload.get("episodes", [])
        }
        self.relations = {
            item["relation_id"]: EvidenceRelation(**item)
            for item in payload.get("relations", [])
        }

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "episodes": [episode.to_dict() for episode in self.episodes.values()],
            "relations": [asdict(relation) for relation in self.relations.values()],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def put_episode(self, episode: EvidenceEpisode) -> EvidenceEpisode:
        self.episodes[episode.episode_id] = episode
        self._save()
        return episode

    def add_relation(
        self,
        source_episode_id: str,
        target_episode_id: str,
        relation_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRelation:
        if source_episode_id not in self.episodes or target_episode_id not in self.episodes:
            raise KeyError("both evidence episodes must exist before relating them")
        relation = EvidenceRelation(
            relation_id=f"relation_{uuid.uuid4().hex[:12]}",
            source_episode_id=source_episode_id,
            target_episode_id=target_episode_id,
            relation_type=relation_type,
            metadata=dict(metadata or {}),
        )
        self.relations[relation.relation_id] = relation
        self._save()
        return relation

    def neighbors(self, episode_id: str, relation_types: Optional[Set[str]] = None) -> List[EvidenceRelation]:
        return [
            relation for relation in self.relations.values()
            if (relation.source_episode_id == episode_id or relation.target_episode_id == episode_id)
            and (not relation_types or relation.relation_type in relation_types)
        ]


class EvidenceMemoryService:
    """Semantic seed retrieval plus governed graph expansion and write-back."""

    def __init__(
        self,
        store: Optional[EvidenceMemoryStore] = None,
        vector_search: Optional[Callable[[str, str, int], Iterable[Dict[str, Any]]]] = None,
        vector_upsert: Optional[Callable[[EvidenceEpisode], None]] = None,
    ):
        self.store = store or EvidenceMemoryStore()
        self.vector_search = vector_search
        self.vector_upsert = vector_upsert

    def add_candidate(self, episode: EvidenceEpisode) -> EvidenceEpisode:
        episode.status = EvidenceStatus.CANDIDATE
        return self.store.put_episode(episode)

    def promote(
        self,
        episode_id: str,
        *,
        manual_review: bool = False,
        verifier_passed: bool = False,
    ) -> EvidenceEpisode:
        episode = self.store.episodes[episode_id]
        if episode.status is EvidenceStatus.VERIFIED:
            return episode
        if not episode.sources:
            raise ValueError("evidence without a source cannot be verified")
        source_ids = {source.source_id for source in episode.sources}
        if any(not claim.source_ids or not set(claim.source_ids).issubset(source_ids) for claim in episode.claims):
            raise ValueError("every claim must reference a known evidence source")
        if episode.risk_level in HIGH_RISK_LEVELS and not manual_review:
            raise PermissionError("high-risk evidence requires manual review")
        if not manual_review and not verifier_passed:
            raise PermissionError("verification gate did not pass")
        episode.status = EvidenceStatus.VERIFIED
        episode.metadata["verified_at"] = utc_now()
        episode.metadata["verification"] = "manual" if manual_review else "deterministic"
        episode = self.store.put_episode(episode)
        if self.vector_upsert:
            self.vector_upsert(episode)
        return episode

    def reject(self, episode_id: str, reason: str) -> EvidenceEpisode:
        episode = self.store.episodes[episode_id]
        episode.status = EvidenceStatus.REJECTED
        episode.metadata["rejection_reason"] = reason
        return self.store.put_episode(episode)

    def supersede(self, new_episode_id: str, old_episode_id: str) -> EvidenceRelation:
        new_episode = self.store.episodes[new_episode_id]
        old_episode = self.store.episodes[old_episode_id]
        if new_episode.status is not EvidenceStatus.VERIFIED:
            raise ValueError("only verified evidence may supersede prior evidence")
        old_episode.status = EvidenceStatus.SUPERSEDED
        self.store.put_episode(old_episode)
        return self.store.add_relation(new_episode_id, old_episode_id, "supersedes")

    def mark_conflict(self, left_episode_id: str, right_episode_id: str, reason: str) -> EvidenceRelation:
        return self.store.add_relation(
            left_episode_id,
            right_episode_id,
            "contradicts",
            {"reason": reason},
        )

    def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        max_hops: int = 2,
        include_candidates: bool = False,
        now: Optional[datetime] = None,
    ) -> EvidencePack:
        allowed = {EvidenceStatus.VERIFIED}
        if include_candidates:
            allowed.add(EvidenceStatus.CANDIDATE)
        candidates = [
            episode for episode in self.store.episodes.values()
            if episode.user_id == user_id and episode.status in allowed and self._is_fresh(episode, now)
        ]

        if not candidates:
            return EvidencePack(query=query, user_id=user_id, hits=[], conflicts=[])

        scored: Dict[str, EvidenceHit] = {}
        if self.vector_search:
            for item in self.vector_search(query, user_id, max(top_k * 2, 10)):
                episode_id = item.get("episode_id")
                episode = self.store.episodes.get(episode_id)
                if episode in candidates:
                    scored[episode_id] = EvidenceHit(
                        episode=episode,
                        score=float(item.get("score", 0.0)),
                        matched_by="vector",
                    )
        if not scored:
            for episode in candidates:
                score = self._lexical_score(query, episode.searchable_text())
                if score > 0:
                    scored[episode.episode_id] = EvidenceHit(episode, score, "lexical")

        seed_ids = [item.episode.episode_id for item in sorted(scored.values(), key=lambda hit: hit.score, reverse=True)[:top_k]]
        frontier = [(episode_id, 0, []) for episode_id in seed_ids]
        visited = set(seed_ids)
        while frontier:
            episode_id, depth, path = frontier.pop(0)
            if depth >= max_hops:
                continue
            for relation in self.store.neighbors(episode_id):
                neighbor_id = (
                    relation.target_episode_id
                    if relation.source_episode_id == episode_id
                    else relation.source_episode_id
                )
                if neighbor_id in visited:
                    continue
                neighbor = self.store.episodes.get(neighbor_id)
                if not neighbor or neighbor.user_id != user_id or neighbor.status not in allowed:
                    continue
                visited.add(neighbor_id)
                relation_path = [*path, relation.relation_type]
                scored[neighbor_id] = EvidenceHit(
                    episode=neighbor,
                    score=max(0.01, scored[episode_id].score * 0.75),
                    matched_by="graph",
                    relation_path=relation_path,
                )
                frontier.append((neighbor_id, depth + 1, relation_path))

        hits = sorted(scored.values(), key=lambda hit: hit.score, reverse=True)[:top_k]
        selected_ids = {hit.episode.episode_id for hit in hits}
        conflicts = [
            {
                "left": relation.source_episode_id,
                "right": relation.target_episode_id,
                "reason": relation.metadata.get("reason", ""),
            }
            for relation in self.store.relations.values()
            if relation.relation_type == "contradicts"
            and (relation.source_episode_id in selected_ids or relation.target_episode_id in selected_ids)
        ]
        return EvidencePack(query=query, user_id=user_id, hits=hits, conflicts=conflicts)

    @staticmethod
    def _is_fresh(episode: EvidenceEpisode, now: Optional[datetime]) -> bool:
        if episode.freshness_days <= 0:
            return True
        current = now or datetime.now(timezone.utc)
        observed = datetime.fromisoformat(episode.observed_at)
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        return (current - observed).days <= episode.freshness_days

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        def tokens(value: str) -> Set[str]:
            latin = re.findall(r"[a-z0-9_]+", value.lower())
            chinese = [char for char in value if "\u4e00" <= char <= "\u9fff"]
            return set(latin + chinese)

        query_tokens = tokens(query)
        text_tokens = tokens(text)
        if not query_tokens or not text_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens | text_tokens)
