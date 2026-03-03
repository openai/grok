# OpenAI Grok Curve Experiments

> **Note:** This is `openai/grok` — the grokking research codebase. This fork also includes [`xai-org/grok-1`](../grok-1-main/) with a bridge between the two. See the [root README](../README.md) for context.

## Paper

This is the code for the paper [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) by Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra.

## Installation and Training

```bash
pip install -e .
./scripts/train.py
```

## Grok-1 Architecture Mode

This fork adds Grok-1's architectural innovations (MoE, RoPE, RMSNorm, gated GELU) as an alternative architecture for grokking experiments:

```bash
# Standard grokking experiment (original)
./scripts/train.py --math_operator + --train_data_pct 5

# Grok-1 architecture — same task, different optimizer geometry
./scripts/train.py --architecture grok1 --math_operator + --num_experts 8

# Auto-scaled miniature Grok-1
./scripts/train.py --architecture grok1_mini
```

New classes in `grok/transformer.py`: `GrokOneTransformer`, `GrokOneMoELayer`, `RotaryPositionalEmbedding`, `RMSNorm`.
New metrics in `grok/metrics.py`: `expert_utilization_entropy`, `expert_specialization_score`, `routing_collapse_index`.
