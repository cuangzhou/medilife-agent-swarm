"""
深度研究工作流

编排多步骤研究流程：查询规划 → 搜索 → 检索 → 综合 → 验证
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import asyncio
import json
import re

from core import LLMClient
from research.web_search import WebSearchTool, SearchResult
from knowledge.milvus_kb import MedicalKnowledgeBase
from research.evidence_synthesizer import EvidenceSynthesizer, ResearchReport
from research.trace_graph import TraceGraphBuilder

# 全局知识库实例（单例）
_kb_instance = None

def get_knowledge_base():
    """获取知识库单例"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = MedicalKnowledgeBase()
    return _kb_instance


class DeepResearchWorkflow:
    """
    深度研究工作流

    功能：
    - 多步骤研究流程编排
    - 查询规划和优化
    - 并行搜索和检索
    - 证据综合和质量控制
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        use_web_search: bool = True,
        use_knowledge_base: bool = True
    ):
        """
        初始化工作流

        Args:
            llm_client: LLM 客户端
            use_web_search: 是否使用网络搜索
            use_knowledge_base: 是否使用 Milvus 知识库
        """
        self.llm_client = llm_client or LLMClient()
        self.use_web_search = use_web_search
        self.use_knowledge_base = use_knowledge_base

        # 初始化组件
        self.web_search = WebSearchTool() if use_web_search else None
        # 使用 Milvus 知识库单例（和其他 Skills 共享，避免重复加载模型）
        self.knowledge_base = get_knowledge_base() if use_knowledge_base else None
        self.synthesizer = EvidenceSynthesizer(llm_client=self.llm_client)

    async def run(
        self,
        question: str,
        max_web_results: int = 10,
        max_kb_results: int = 5,
        explainable: bool = True,
        max_research_rounds: int = 3
    ) -> ResearchReport:
        """
        执行深度研究

        Args:
            question: 研究问题
            max_web_results: 最大网络搜索结果数
            max_kb_results: 最大知识库检索结果数

        Returns:
            研究报告
        """
        logger.info(f"Starting DeepResearch for: {question}")

        if not explainable:
            return await self._run_parallel(question, max_web_results, max_kb_results)

        return await self._run_explainable(
            question=question,
            max_web_results=max_web_results,
            max_kb_results=max_kb_results,
            max_research_rounds=max_research_rounds
        )

    async def _run_parallel(
        self,
        question: str,
        max_web_results: int = 10,
        max_kb_results: int = 5
    ) -> ResearchReport:
        """Legacy one-shot planning + parallel retrieval workflow."""

        # Step 1: 查询规划
        sub_queries = await self._plan_queries(question)
        logger.info(f"Planned {len(sub_queries)} sub-queries")

        # Step 2: 并行搜索
        web_results: List[SearchResult] = []
        kb_results: List[Document] = []

        search_tasks = []

        if self.use_web_search and self.web_search:
            # 网络搜索
            for query in sub_queries[:3]:  # 限制子查询数量
                search_tasks.append(
                    self.web_search.search(query, max_results=max_web_results // len(sub_queries))
                )

        if self.use_knowledge_base and self.knowledge_base:
            # 从 Milvus 知识库检索
            for query in sub_queries[:3]:
                search_tasks.append(
                    self._search_milvus(query, top_k=max_kb_results // len(sub_queries))
                )

        # 并行执行
        if search_tasks:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # 分离结果
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Search task failed: {result}")
                    continue

                if isinstance(result, list):
                    if len(result) > 0:
                        if isinstance(result[0], SearchResult):
                            web_results.extend(result)
                        elif isinstance(result[0], dict):
                            # Milvus 返回的是字典列表
                            kb_results.extend(result)

        logger.info(f"Collected {len(web_results)} web results, {len(kb_results)} KB results")

        # Step 3: 证据综合
        report = await self.synthesizer.synthesize(
            query=question,
            web_results=web_results,
            kb_results=kb_results
        )
        if not report.key_findings:
            logger.warning("Report has no key findings")

        if not report.summary:
            logger.warning("Report has no summary")

        logger.info("DeepResearch completed")
        return report

    async def _run_explainable(
        self,
        question: str,
        max_web_results: int = 10,
        max_kb_results: int = 5,
        max_research_rounds: int = 3
    ) -> ResearchReport:
        """
        Explainable step-by-step workflow.

        Each round executes at most one web search and one Milvus search, records
        the observations, then asks a planner whether to continue, refine, or
        synthesize. A deterministic fallback is used when the planner LLM is not
        available.
        """
        graph = TraceGraphBuilder()
        research_steps: List[Dict[str, Any]] = []
        web_results: List[SearchResult] = []
        kb_results: List[Dict[str, Any]] = []

        question_node = graph.add_node(
            "question",
            "用户医学问题",
            question,
            source="user",
            confidence=1.0
        )

        sub_queries = await self._plan_queries(question)
        sub_queries = (sub_queries or [question])[:max_research_rounds]
        plan_node = graph.add_node(
            "plan",
            "分步检索规划",
            "；".join(sub_queries),
            source="planner",
            confidence=0.8,
            metadata={"max_research_rounds": max_research_rounds}
        )
        graph.add_edge(question_node, plan_node, "decomposes_to", "拆解为检索计划")

        pending_queries = list(sub_queries)
        completed_queries: List[str] = []
        round_index = 0

        while pending_queries and round_index < max_research_rounds:
            round_index += 1
            query = pending_queries.pop(0)
            sub_query_node = graph.add_node(
                "sub_query",
                f"医学子问题 {round_index}",
                query,
                source="planner",
                confidence=0.8,
                metadata={"round": round_index}
            )
            graph.add_edge(plan_node, sub_query_node, "decomposes_to", "规划子问题")

            step: Dict[str, Any] = {
                "round": round_index,
                "query": query,
                "web_results": 0,
                "kb_results": 0,
                "decision": "continue",
                "errors": []
            }

            round_web_results: List[SearchResult] = []
            round_kb_results: List[Dict[str, Any]] = []

            if self.use_knowledge_base and self.knowledge_base:
                retrieval_node = graph.add_node(
                    "retrieval",
                    "Milvus 医学知识库检索",
                    query,
                    source="milvus",
                    metadata={"round": round_index, "top_k": max(1, max_kb_results)}
                )
                graph.add_edge(sub_query_node, retrieval_node, "searches", "语义检索")
                try:
                    round_kb_results = await self._search_milvus(query, top_k=max(1, max_kb_results))
                    kb_results.extend(round_kb_results)
                    step["kb_results"] = len(round_kb_results)
                    for index, doc in enumerate(round_kb_results[:3], 1):
                        metadata = doc.get("metadata", {})
                        evidence_node = graph.add_node(
                            "evidence",
                            metadata.get("title") or f"知识库证据 {index}",
                            doc.get("content", "")[:500],
                            source=metadata.get("source", "Milvus 医学知识库"),
                            confidence=doc.get("score"),
                            metadata={
                                "round": round_index,
                                "result_index": index,
                                "type": metadata.get("type"),
                                "doc_id": doc.get("id")
                            }
                        )
                        graph.add_edge(retrieval_node, evidence_node, "returns", "返回知识库证据")
                except Exception as e:
                    logger.warning(f"Milvus retrieval failed in round {round_index}: {e}")
                    step["errors"].append(f"milvus: {e}")
                    error_node = graph.add_node(
                        "evidence",
                        "Milvus 检索失败",
                        str(e),
                        source="milvus",
                        confidence=0.0,
                        metadata={"round": round_index, "error": True}
                    )
                    graph.add_edge(retrieval_node, error_node, "returns", "检索失败")

            if self.use_web_search and self.web_search:
                retrieval_node = graph.add_node(
                    "retrieval",
                    "医学网络检索",
                    query,
                    source="web_search",
                    metadata={"round": round_index, "max_results": max(1, max_web_results)}
                )
                graph.add_edge(sub_query_node, retrieval_node, "searches", "网络检索")
                try:
                    round_web_results = await self.web_search.search(query, max_results=max(1, max_web_results))
                    web_results.extend(round_web_results)
                    step["web_results"] = len(round_web_results)
                    for index, result in enumerate(round_web_results[:3], 1):
                        evidence_node = graph.add_node(
                            "evidence",
                            result.title or f"网络证据 {index}",
                            result.snippet[:500],
                            source=result.url or "web",
                            confidence=0.6,
                            metadata={"round": round_index, "result_index": index}
                        )
                        graph.add_edge(retrieval_node, evidence_node, "returns", "返回网络证据")
                except Exception as e:
                    logger.warning(f"Web retrieval failed in round {round_index}: {e}")
                    step["errors"].append(f"web: {e}")
                    error_node = graph.add_node(
                        "evidence",
                        "网络检索失败",
                        str(e),
                        source="web_search",
                        confidence=0.0,
                        metadata={"round": round_index, "error": True}
                    )
                    graph.add_edge(retrieval_node, error_node, "returns", "检索失败")

            completed_queries.append(query)
            decision = await self._decide_next_step(
                question=question,
                completed_queries=completed_queries,
                pending_queries=pending_queries,
                latest_web_results=round_web_results,
                latest_kb_results=round_kb_results,
                round_index=round_index,
                max_research_rounds=max_research_rounds
            )
            step["decision"] = decision.get("action", "continue")
            step["decision_reason"] = decision.get("reason", "")
            research_steps.append(step)

            if decision.get("action") == "refine" and decision.get("query"):
                pending_queries.insert(0, str(decision["query"]))
            elif decision.get("action") == "synthesize":
                break

        report = await self.synthesizer.synthesize(
            query=question,
            web_results=web_results,
            kb_results=kb_results
        )

        synthesis_node = graph.add_node(
            "synthesis",
            "证据综合",
            report.summary or "综合多来源医学证据",
            source="evidence_synthesizer",
            confidence=report.confidence,
            metadata={
                "evidence_level": report.evidence_level,
                "conflicts": report.conflicts,
                "sources": len(report.sources)
            }
        )
        graph.add_edge(plan_node, synthesis_node, "synthesizes_into", "汇总检索结果")

        final_node = graph.add_node(
            "final_answer",
            "最终医学回答",
            report.summary or "生成最终回答",
            source="deep_research",
            confidence=report.confidence,
            metadata={"recommendations": report.recommendations}
        )
        graph.add_edge(synthesis_node, final_node, "synthesizes_into", "形成最终答案")

        for node in list(graph.nodes):
            if node.get("type") == "evidence" and not node.get("metadata", {}).get("error"):
                graph.add_edge(node["id"], synthesis_node, "supports", "支持综合结论")

        report.trace_graph = graph.to_dict()
        report.research_steps = research_steps

        if not report.key_findings:
            logger.warning("Report has no key findings")

        if not report.summary:
            logger.warning("Report has no summary")

        logger.info("Explainable DeepResearch completed")
        return report

    async def _search_milvus(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        从 Milvus 知识库搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            文档列表（字典格式）
        """
        try:
            results = self.knowledge_base.search(query=query, top_k=top_k, filter_type=None)
            logger.debug(f"Milvus search returned {len(results)} results for: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return []

    async def _plan_queries(self, question: str) -> List[str]:
        """
        查询规划：将复杂问题拆解为多个子查询

        Args:
            question: 原始问题

        Returns:
            子查询列表
        """
        prompt = f"""你是医学研究助手。请将以下问题拆解为 2-3 个更具体的子查询，以便进行深度研究。

原始问题：{question}

要求：
1. 每个子查询应该聚焦一个特定方面
2. 子查询应该互补，覆盖问题的不同角度
3. 子查询应该简洁明确

输出格式：
每行一个子查询，不需要编号。

示例：
原始问题：2型糖尿病如何治疗？
子查询1：2型糖尿病的药物治疗方案
子查询2：2型糖尿病的生活方式管理
子查询3：2型糖尿病的并发症预防
"""

        try:
            response = await self.llm_client.chat([
                {"role": "user", "content": prompt}
            ])

            # 解析子查询
            lines = response.strip().split('\n')
            sub_queries = []

            for line in lines:
                line = line.strip()
                # 移除可能的编号
                line = line.lstrip('0123456789.-:：）) ')
                if line and len(line) > 5:  # 过滤太短的行
                    sub_queries.append(line)

            # 至少包含原始问题
            if not sub_queries:
                sub_queries = [question]

            # 限制数量
            sub_queries = sub_queries[:3]

            return sub_queries

        except Exception as e:
            logger.error(f"Query planning error: {e}")
            # 降级：返回原始问题
            return [question]

    async def _decide_next_step(
        self,
        question: str,
        completed_queries: List[str],
        pending_queries: List[str],
        latest_web_results: List[SearchResult],
        latest_kb_results: List[Dict[str, Any]],
        round_index: int,
        max_research_rounds: int
    ) -> Dict[str, Any]:
        """Decide whether to continue retrieval, refine the next query, or synthesize."""
        if round_index >= max_research_rounds:
            return {"action": "synthesize", "reason": "已达到最大检索轮数"}

        if latest_web_results or latest_kb_results:
            if not pending_queries:
                return {"action": "synthesize", "reason": "已获得可综合的医学证据"}
        elif not pending_queries:
            return {"action": "synthesize", "reason": "没有更多待检索子问题"}

        evidence_summary = {
            "latest_web_results": [
                {"title": item.title, "snippet": item.snippet[:160]}
                for item in latest_web_results[:3]
            ],
            "latest_kb_results": [
                {
                    "title": item.get("metadata", {}).get("title", "医学知识"),
                    "content": item.get("content", "")[:160],
                    "score": item.get("score")
                }
                for item in latest_kb_results[:3]
            ],
            "pending_queries": pending_queries,
        }
        prompt = f"""你是医学检索 planner。请根据已完成检索结果决定下一步。

原始问题：{question}
已完成子问题：{completed_queries}
最新观察：{json.dumps(evidence_summary, ensure_ascii=False)}

只输出 JSON：
{{"action":"continue|refine|synthesize","query":"可选的新子问题","reason":"一句话原因"}}
"""
        try:
            response = await self.llm_client.chat([{"role": "user", "content": prompt}], temperature=0.2)
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                decision = json.loads(match.group())
                action = decision.get("action")
                if action in {"continue", "refine", "synthesize"}:
                    return decision
        except Exception as e:
            logger.debug(f"Planner decision fallback: {e}")

        return {"action": "continue", "reason": "继续执行预设医学子问题检索"}

    async def research_with_refinement(
        self,
        question: str,
        max_iterations: int = 2
    ) -> ResearchReport:
        """
        带细化的研究（多轮迭代）

        Args:
            question: 研究问题
            max_iterations: 最大迭代次数

        Returns:
            最终研究报告
        """
        logger.info(f"Starting iterative research (max_iterations={max_iterations})")

        report = None

        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")

            # 执行研究
            report = await self.run(question)

            # 检查质量
            if report.confidence >= 0.7 and len(report.key_findings) >= 3:
                logger.info(f"High-quality report achieved in iteration {iteration + 1}")
                break

            # 如果是最后一轮，直接返回
            if iteration == max_iterations - 1:
                break

            # 细化查询（基于当前结果）
            if report.key_findings:
                question = f"{question}（关注：{report.key_findings[0]}）"

        return report


# 便捷函数
async def deep_research(
    question: str,
    use_web: bool = True,
    use_kb: bool = True
) -> ResearchReport:
    """
    快速执行深度研究

    Args:
        question: 研究问题
        use_web: 是否使用网络搜索
        use_kb: 是否使用知识库

    Returns:
        研究报告
    """
    workflow = DeepResearchWorkflow(
        use_web_search=use_web,
        use_knowledge_base=use_kb
    )
    return await workflow.run(question)
