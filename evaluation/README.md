# MediLife 双轨评测

```powershell
python evaluation/generate_placeholder.py
python evaluation/evidence_memory_benchmark.py --mode smoke
python evaluation/evidence_memory_benchmark.py --mode full
python evaluation/innovation_benchmark.py
python evaluation/export_resume_metrics.py evaluation/artifacts/measured/medilife_evidence_memory_smoke_v1.json resume_metrics.json
```

- `artifacts/placeholder` 是内部模拟数据，正式指标导出器会拒绝读取。
- `artifacts/measured` 来自固定 JSONL fixture 和实际执行结果。
- 缺少 Pymilvus/Mem0 时相应对照组输出 `SKIPPED_DEPENDENCY`，确定性 Evidence Memory 逻辑层仍可评测，但不得表述为完整端到端集成结果。
- 数据集为匿名合成纵向病例，结果不代表临床有效性。
- `innovation_benchmark.py` 使用确定性 ground truth，自动测量多跳召回、冲突发现、候选隔离、跨用户泄漏、来源追溯、时序覆盖、高风险晋升门禁和无关记忆干扰，不依赖人工或 LLM Judge。
