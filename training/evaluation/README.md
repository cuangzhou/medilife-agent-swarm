# Training evaluation

- `primary/` contains the main generation, reference-based judge and scoring pipeline, including MMMU-Medical support.
- `secondary/` contains the second retained evaluation implementation for compatibility and comparison.
- Generated logs and results belong outside these source directories. Historical upstream examples are isolated under `../upstream_artifacts/`.

Both pipelines require separately installed model-serving and evaluation dependencies. Their outputs must not be treated as MediLife measured evidence until executed in a recorded environment with an immutable command, model identifier, dataset version and result contract.

