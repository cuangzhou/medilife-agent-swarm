# MediLife

MediLife 是一个面向医疗信息咨询、症状风险提示、指南检索与循证研究的多 Agent 原型系统。它采用 Skills–Agent 两层架构，将 Agent Loop、Swarm 协作、Milvus Lite 向量检索、DeepResearch、记忆与可解释 Trace Graph 组合到统一 API 中。

> 本项目仅用于学习、研究和产品原型验证，不提供诊断或治疗，不可替代专业医务人员。评测使用匿名合成病例，不代表临床有效性。

## 核心能力

- `ConsultationAgent`、`DiagnosticAgent`、`ResearchAgent` 分别处理健康咨询、风险分析和循证检索。
- `SwarmCoordinator` 在单 Agent 与多 Agent 协作之间路由。
- `SkillRegistry` 将本地 Skills 转换为 OpenAI function calling 工具定义；当前不是 MCP Server。
- 静态知识库使用 Milvus Lite 向量检索；当前未实现 BM25、混合检索或独立 Rerank。
- DeepResearch 与 Trace Graph 展示问题拆解、检索、证据、综合和最终回答之间的关系。
- 可选 Mem0 长期记忆；无凭据时自动禁用。

## Governed Evidence Memory

2026-07-14 的升级增加了受治理 Evidence Memory：

- 普通回答只产生 `candidate`，通过 verifier 或人工审核后才能晋升为 `verified`。
- 高风险证据不得仅凭自动 verifier 晋升。
- 支持 `supports`、`contradicts`、`derived_from`、`follows`、`supersedes`、`similar_to` 关系。
- 检索严格按 `user_id` 隔离，并排除未验证候选证据。
- 独立 Milvus collection 仅作为已验证 EvidenceEpisode 的语义候选索引，关系图负责扩展和冲突呈现。

Swarm、Agent Loop、DeepResearch、Trace Graph、Milvus、Mem0 和 Skills 是此前已有能力，不属于这次升级新增。

## 安装

需要 Python 3.11 或 3.12。

```bash
python -m pip install -e ".[server]"
```

按需安装完整的 RAG、研究与记忆依赖：

```bash
python -m pip install -e ".[all]"
```

通过环境变量配置 OpenAI-compatible 模型：

```bash
export LLM_API_KEY="your-key"
export LLM_MODEL_NAME="your-model"
export LLM_BASE_URL="https://api.openai.com/v1"
```

Windows PowerShell 使用 `$env:LLM_API_KEY="your-key"`。也可复制 `config.example.py` 为未跟踪的 `config.py` 做本地开发配置。

## 运行

交互式 CLI：

```bash
medilife
```

FastAPI：

```bash
python -m uvicorn api_server:app --host 127.0.0.1 --port 8787
```

健康检查：`GET /api/health`。

问答接口：

```http
POST /api/chat
Content-Type: application/json

{
  "question": "高血压患者日常应该注意什么？",
  "session_id": "demo-session",
  "user_id": "demo-user",
  "enable_swarm": true,
  "explain": true,
  "evidence_memory": true
}
```

响应保留 `trace_graph`，并包含 `evidence_pack`、`memory_candidates`、`memory_delta` 和 `conflicts`。

证据检索与审核：

```http
GET /api/evidence/search?query=高血压&user_id=demo-user&top_k=5
POST /api/evidence/{episode_id}/verify
Content-Type: application/json

{"manual_review": true, "verifier_passed": false}
```

## 评测与验证

```bash
python -m pytest
python evaluation/evidence_memory_benchmark.py --mode smoke
python evaluation/evidence_memory_benchmark.py --mode full
python evaluation/innovation_benchmark.py
```

`evaluation/artifacts/measured/` 保存机器实测结果。Evidence Memory full 与 innovation 评测使用 50 组匿名合成纵向病例，验证治理、用户隔离、图关系和冲突逻辑；它们不是人工或 LLM Judge，也不冒充 Milvus/Mem0 端到端或临床质量评测。

`evaluation/artifacts/placeholder/` 仅保存明确标记的设计占位结果，不得作为已实现能力或简历指标证据。

## 安全边界

- 默认 API 没有生产级身份认证、授权和合规控制。
- 不应输入可识别患者信息或将其发送到未经批准的模型、搜索或记忆服务。
- 面向生产环境还需补充租户隔离、审计、加密、数据保留策略与人工复核流程。

详细说明见 [SECURITY.md](SECURITY.md)。
