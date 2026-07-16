#!/usr/bin/env python3
"""
MediLife medical assistant API.

This API exposes the core MediLife Agent Swarm assistant and returns an
explainable trace graph that clients can render as an evidence chain.
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

SWARM_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SWARM_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SWARM_ROOT) not in sys.path:
    sys.path.insert(0, str(SWARM_ROOT))

from research.trace_graph import TraceGraphBuilder, empty_trace_graph
from memory.evidence_memory import (
    EvidenceClaim,
    EvidenceEpisode,
    EvidenceMemoryService,
    EvidenceMemoryStore,
    EvidenceSource,
)
from knowledge.evidence_vector_index import MilvusEvidenceVectorIndex
from knowledge.resources import bundled_knowledge_db, evidence_index_db, evidence_memory_store

try:
    from core.llm_client import LLM_CONFIG
except Exception:
    LLM_CONFIG = {
        "api_key": os.getenv("LLM_API_KEY", ""),
        "model_name": os.getenv("LLM_MODEL_NAME", "gpt-4.1-mini"),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    }
MEM0_CONFIG = {"api_key": os.getenv("MEM0_API_KEY", "")}

try:
    from swarm import process_with_swarm
except Exception:  # pragma: no cover - API should still start for health checks
    process_with_swarm = None


app = FastAPI(title="MediLife Medical Assistant API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    enable_swarm: bool = True
    explain: bool = True
    user_id: str = Field(default="anonymous", min_length=1)
    evidence_memory: bool = True


class EvidenceVerifyRequest(BaseModel):
    manual_review: bool = False
    verifier_passed: bool = False


EVIDENCE_MEMORY_PATH = evidence_memory_store()
evidence_vector_index = MilvusEvidenceVectorIndex(
    db_path=str(evidence_index_db())
)
evidence_memory_service = EvidenceMemoryService(
    store=EvidenceMemoryStore(str(EVIDENCE_MEMORY_PATH)),
    vector_search=evidence_vector_index.search,
    vector_upsert=evidence_vector_index.upsert,
)


def llm_available() -> bool:
    return bool((LLM_CONFIG or {}).get("api_key"))


def mem0_available() -> bool:
    return bool((MEM0_CONFIG or {}).get("api_key"))


def knowledge_base_available() -> bool:
    return bundled_knowledge_db().exists()


def health_service_metadata() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "medilife-medical-assistant",
        "llmConfigured": llm_available(),
        "swarmAvailable": process_with_swarm is not None,
        "knowledgeBase": knowledge_base_available(),
        "mem0Configured": mem0_available(),
        "evidenceMemory": True,
    }


def build_fallback_trace(question: str, answer: str, mode: str, agents: List[str]) -> Dict[str, Any]:
    graph = TraceGraphBuilder()
    question_node = graph.add_node(
        "question",
        "用户医学问题",
        question,
        source="user",
        confidence=1.0
    )
    plan_node = graph.add_node(
        "plan",
        "MediLife 医疗助手路由",
        f"选择 {mode} 模式处理问题。",
        source="api",
        confidence=0.5,
        metadata={"agents_involved": agents}
    )
    graph.add_edge(question_node, plan_node, "decomposes_to", "路由决策")
    answer_node = graph.add_node(
        "final_answer",
        "最终医学回答",
        answer,
        source="medilife",
        confidence=0.5
    )
    graph.add_edge(plan_node, answer_node, "synthesizes_into", "生成回答")
    return graph.to_dict()


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return health_service_metadata()


@app.post("/api/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    evidence_pack = (
        evidence_memory_service.search(request.question, user_id=request.user_id).to_dict()
        if request.evidence_memory
        else {"query": request.question, "user_id": request.user_id, "hits": [], "conflicts": []}
    )
    if not process_with_swarm:
        answer = "MediLife Swarm 未能加载，请检查项目依赖和启动路径。"
        return {
            "answer": answer,
            "session_id": request.session_id or "",
            "mode": "single_agent",
            "agents_involved": [],
            "trace_graph": build_fallback_trace(request.question, answer, "single_agent", []),
            "evidence_pack": evidence_pack,
            "memory_candidates": [],
            "memory_delta": {"created": [], "verified": [], "superseded": []},
            "conflicts": evidence_pack.get("conflicts", []),
            "disclaimer": "以上信息仅供学习和研究参考，不能替代专业医生诊断或治疗。",
            "error": "swarm_unavailable",
        }

    if not llm_available():
        answer = "LLM API key 未配置，MediLife 医疗助手暂时无法生成 Agent 回答。请在上级目录 config.py 中配置 LLM_CONFIG['api_key'] 后重试。"
        return {
            "answer": answer,
            "session_id": request.session_id or "",
            "mode": "single_agent",
            "agents_involved": [],
            "trace_graph": build_fallback_trace(request.question, answer, "single_agent", []),
            "evidence_pack": evidence_pack,
            "memory_candidates": [],
            "memory_delta": {"created": [], "verified": [], "superseded": []},
            "conflicts": evidence_pack.get("conflicts", []),
            "disclaimer": "以上信息仅供学习和研究参考，不能替代专业医生诊断或治疗。",
            "suggestions": [],
            "metadata": {
                "swarm_enabled": False,
                "subtasks_completed": None,
                "timeout_occurred": False,
                "iterations": 0,
            },
            "error": "llm_not_configured",
        }

    try:
        result = await process_with_swarm(
            question=request.question,
            context={
                "api": "chat",
                "explain": request.explain,
                "user_id": request.user_id,
                "evidence_pack": evidence_pack,
            },
            enable_swarm=request.enable_swarm,
            session_id=request.session_id,
        )
    except Exception as e:
        answer = f"处理问题时发生错误：{e}"
        return {
            "answer": answer,
            "session_id": request.session_id or "",
            "mode": "single_agent",
            "agents_involved": [],
            "trace_graph": build_fallback_trace(request.question, answer, "single_agent", []),
            "evidence_pack": evidence_pack,
            "memory_candidates": [],
            "memory_delta": {"created": [], "verified": [], "superseded": []},
            "conflicts": evidence_pack.get("conflicts", []),
            "disclaimer": "以上信息仅供学习和研究参考，不能替代专业医生诊断或治疗。",
            "error": str(e),
        }

    answer = result.get("answer", "")
    mode = "swarm" if result.get("swarm_enabled") else "single_agent"
    agents = result.get("agents_involved") or [result.get("agent_id", "consultation_agent")]
    agents = [agent for agent in agents if agent]

    if request.explain:
        trace_graph = result.get("trace_graph") or build_fallback_trace(request.question, answer, mode, agents)
    else:
        trace_graph = empty_trace_graph()

    memory_candidates: List[str] = []
    memory_delta: Dict[str, Any] = {"created": [], "verified": [], "superseded": []}
    if request.evidence_memory and answer:
        sources: List[EvidenceSource] = []
        for node in trace_graph.get("nodes", []):
            if node.get("type") != "evidence" or node.get("metadata", {}).get("error"):
                continue
            source_id = node.get("id") or f"source_{len(sources) + 1}"
            sources.append(EvidenceSource(
                source_id=source_id,
                title=node.get("label") or "医学证据",
                uri=node.get("source") or "",
                source_type=node.get("metadata", {}).get("source_type", "trace_graph"),
            ))
        source_ids = [source.source_id for source in sources]
        claims = [EvidenceClaim(
            claim_id=f"answer_{(result.get('session_id') or request.session_id or 'session')}",
            text=answer[:1000],
            confidence=float(result.get("confidence", 0.0) or 0.0),
            source_ids=source_ids,
        )] if sources else []
        episode = EvidenceEpisode.candidate(
            user_id=request.user_id,
            session_id=result.get("session_id") or request.session_id or "",
            question=request.question,
            claims=claims,
            recommendations=result.get("suggestions", []),
            sources=sources,
            risk_level=str(result.get("risk_level", "low")),
            confidence=float(result.get("confidence", 0.0) or 0.0),
            metadata={"mode": mode, "agents_involved": agents},
        )
        evidence_memory_service.add_candidate(episode)
        memory_candidates.append(episode.episode_id)
        memory_delta["created"].append(episode.episode_id)

    return {
        "answer": answer,
        "session_id": result.get("session_id") or request.session_id or "",
        "mode": mode,
        "agents_involved": agents,
        "trace_graph": trace_graph,
        "evidence_pack": evidence_pack,
        "memory_candidates": memory_candidates,
        "memory_delta": memory_delta,
        "conflicts": evidence_pack.get("conflicts", []),
        "disclaimer": result.get(
            "disclaimer",
            "以上信息仅供学习和研究参考，不能替代专业医生诊断或治疗。"
        ),
        "suggestions": result.get("suggestions", []),
        "metadata": {
            "swarm_enabled": result.get("swarm_enabled", False),
            "subtasks_completed": result.get("subtasks_completed"),
            "timeout_occurred": result.get("timeout_occurred", False),
            "iterations": result.get("iterations"),
        },
    }


@app.get("/api/evidence/search")
async def search_evidence(query: str, user_id: str, top_k: int = 5) -> Dict[str, Any]:
    return evidence_memory_service.search(
        query,
        user_id=user_id,
        top_k=max(1, min(top_k, 20)),
    ).to_dict()


@app.post("/api/evidence/{episode_id}/verify")
async def verify_evidence(episode_id: str, request: EvidenceVerifyRequest) -> Dict[str, Any]:
    episode = evidence_memory_service.promote(
        episode_id,
        manual_review=request.manual_review,
        verifier_passed=request.verifier_passed,
    )
    return {"episode": episode.to_dict(), "memory_delta": {"verified": [episode_id]}}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8787)
