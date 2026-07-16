"""
Trace graph helpers for explainable medical retrieval and agent reasoning.

The graph is intentionally plain JSON-compatible data so API clients can render
it as a relationship chain without depending on Python-specific classes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid


TraceGraph = Dict[str, List[Dict[str, Any]]]


class TraceGraphBuilder:
    """Small helper for building stable trace graph payloads."""

    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self._node_ids = set()

    def add_node(
        self,
        node_type: str,
        label: str,
        summary: str = "",
        source: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        node_id = node_id or f"{node_type}_{uuid.uuid4().hex[:8]}"
        if node_id in self._node_ids:
            return node_id

        self.nodes.append({
            "id": node_id,
            "type": node_type,
            "label": label,
            "summary": summary or "",
            "source": source or "",
            "confidence": confidence,
            "metadata": metadata or {},
        })
        self._node_ids.add(node_id)
        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        edge_id = f"edge_{uuid.uuid4().hex[:8]}"
        self.edges.append({
            "id": edge_id,
            "source": source,
            "target": target,
            "type": edge_type,
            "label": label or "",
            "metadata": metadata or {},
        })
        return edge_id

    def merge(self, graph: Optional[TraceGraph], prefix: Optional[str] = None) -> Dict[str, str]:
        """Merge another trace graph and return old-id to new-id mapping."""
        mapping: Dict[str, str] = {}
        if not graph:
            return mapping

        for node in graph.get("nodes", []):
            old_id = node.get("id") or f"external_{uuid.uuid4().hex[:8]}"
            new_id = f"{prefix}_{old_id}" if prefix else old_id
            mapping[old_id] = new_id
            self.add_node(
                node_type=node.get("type", "evidence"),
                label=node.get("label", "外部证据"),
                summary=node.get("summary", ""),
                source=node.get("source", ""),
                confidence=node.get("confidence"),
                metadata=node.get("metadata", {}),
                node_id=new_id,
            )

        for edge in graph.get("edges", []):
            source = mapping.get(edge.get("source"))
            target = mapping.get(edge.get("target"))
            if source and target:
                self.add_edge(
                    source=source,
                    target=target,
                    edge_type=edge.get("type", "supports"),
                    label=edge.get("label", ""),
                    metadata=edge.get("metadata", {}),
                )

        return mapping

    def to_dict(self) -> TraceGraph:
        return {"nodes": self.nodes, "edges": self.edges}


def empty_trace_graph() -> TraceGraph:
    return {"nodes": [], "edges": []}
