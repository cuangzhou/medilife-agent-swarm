"""Lazy Milvus Lite index for verified EvidenceEpisode semantic retrieval."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger


class MilvusEvidenceVectorIndex:
    """Separate collection so episodic evidence never pollutes the static KB."""

    def __init__(
        self,
        db_path: str = "./knowledge/data/evidence_memory.db",
        collection_name: str = "evidence_episodes",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self._client = None
        self._model = None

    def _ensure(self) -> bool:
        if self._client is not None and self._model is not None:
            return True
        try:
            from pymilvus import MilvusClient
            from sentence_transformers import SentenceTransformer

            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(self.embedding_model_name, device="cpu")
            dimension = self._model.get_sentence_embedding_dimension()
            self._client = MilvusClient(self.db_path)
            if not self._client.has_collection(self.collection_name):
                self._client.create_collection(
                    collection_name=self.collection_name,
                    dimension=dimension,
                    metric_type="COSINE",
                    auto_id=True,
                    enable_dynamic_field=True,
                )
            return True
        except Exception as exc:
            logger.warning(f"Evidence vector index unavailable; lexical fallback active: {exc}")
            self._client = None
            self._model = None
            return False

    def upsert(self, episode) -> None:
        if not self._ensure():
            return
        vector = self._model.encode([episode.searchable_text()])[0].tolist()
        self._client.insert(
            self.collection_name,
            [{
                "vector": vector,
                "episode_id": episode.episode_id,
                "user_id": episode.user_id,
                "status": episode.status.value,
                "observed_at": episode.observed_at,
            }],
        )

    def search(self, query: str, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self._ensure():
            return []
        vector = self._model.encode([query])[0].tolist()
        safe_user_id = user_id.replace('"', '\\"')
        try:
            results = self._client.search(
                collection_name=self.collection_name,
                data=[vector],
                limit=limit,
                filter=f'user_id == "{safe_user_id}" and status == "verified"',
                output_fields=["episode_id", "user_id", "status"],
            )
        except Exception as exc:
            logger.warning(f"Evidence vector search failed; lexical fallback active: {exc}")
            return []
        hits: List[Dict[str, Any]] = []
        for group in results:
            for hit in group:
                entity = hit.get("entity", {})
                distance = float(hit.get("distance", 1.0))
                hits.append({
                    "episode_id": entity.get("episode_id"),
                    "score": 1.0 - distance,
                })
        return hits
