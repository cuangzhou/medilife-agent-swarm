"""
医学知识库模块

提供基于 Milvus 的向量数据库封装
"""
try:
    from .milvus_kb import MedicalKnowledgeBase
except ImportError:  # API health/fallback mode must work without optional Milvus deps.
    MedicalKnowledgeBase = None

from .evidence_vector_index import MilvusEvidenceVectorIndex

__all__ = ['MedicalKnowledgeBase', 'MilvusEvidenceVectorIndex']
