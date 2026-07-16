# MediLife-R1 research training subsystem

This isolated subsystem provides medical vision-language reinforcement-learning and evaluation code. It supports GRPO, DAPO and GSPO configurations, composite medical rewards, a vLLM reward server workflow, FSDP checkpoint merging, and text/VLM evaluation.

It is adapted from the upstream **MediX-R1** research project and retains the upstream CC BY-NC-SA 4.0 non-commercial restriction. See `LICENSE`, `MODIFICATIONS.md`, and the root `THIRD_PARTY_NOTICES.md` before use.

## Environment boundary

Training is intentionally not installed by the main MediLife wheel. Use a separate Linux GPU environment with CUDA, FlashAttention, Ray and vLLM. The main FastAPI service never imports this package at startup.

```bash
cd training
python -m pip install -e .
```

The training package keeps the upstream `verl` module name for framework compatibility. Product-facing experiment names and output paths use MediLife-R1.

## Workflows

1. Configure the reward model and GPU allocation in the training scripts.
2. Start the vLLM reward server required by the composite accuracy reward.
3. Run a configuration from `examples/`, such as the 2B DAPO or 8B GSPO profile.
4. Merge an FSDP checkpoint with `merge_model.sh`.
5. Run generation, judge and score stages under `evaluation/`.

Training data is obtained separately from the upstream MBZUAI medical RL dataset. Do not commit datasets, model weights, checkpoints, secrets, or identifiable patient information.

`upstream_artifacts/` contains clearly labelled upstream example curves, logs and outputs. They are not MediLife measured evidence and cannot be exported as resume metrics.

