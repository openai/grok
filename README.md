Readme · MD
<div align="center">

# Does Grok grok grokking?

<p>
<em>A functional bridge between</em>
<a href="https://github.com/openai/grok"><code>openai/grok</code></a>
<em>and</em>
<a href="https://github.com/xai-org/grok-1"><code>xai-org/grok-1</code></a>
</p>

<p>
<em>Two projects that share a name, a history, and nothing else.</em>
</p>

</div>

---

> **grok** /ɡrɒk/ *verb.* To understand something so thoroughly that it becomes part of you.
>
> — Robert A. Heinlein, *Stranger in a Strange Land* (1961)

---

Three words. Three meanings. One question.

**Grok** (noun) — xAI's 314-billion-parameter Mixture-of-Experts language model, [open-sourced](https://github.com/xai-org/grok-1) on March 17, 2024, under Apache 2.0. Named after Heinlein's verb.

**grok** (verb) — to understand something with such intimacy that observer and observed merge. The word entered hacker culture in the 1960s and never left.

**grokking** (noun, ML) — a phenomenon discovered by [Power et al. (2022)](https://arxiv.org/abs/2201.02177) where neural networks, long after memorizing their training set, suddenly and sharply generalize to unseen data. The [research code](https://github.com/openai/grok) was published by OpenAI.

This repository forces the two `grok` repositories into a single project — not because they belong together, but because naming things is hard, irony compounds, and sometimes the most interesting things happen when you make incompatible things work together through sheer will.

---

## The Two Groks

| | `openai/grok` | `xai-org/grok-1` |
|---|---|---|
| **What** | Research code for the grokking phenomenon | Inference code for the Grok-1 LLM |
| **Lines** | ~500, PyTorch | ~1400, JAX/Haiku |
| **Parameters** | Trains models with ~100K params | Loads a model with 314B params |
| **GitHub Stars** | ~4.2K | ~49K |
| **Framework** | PyTorch + Lightning | JAX + Haiku |
| **Task** | Modular arithmetic (42 + 55 mod 97) | General language modeling |
| **Named after** | An ML phenomenon | A sci-fi verb |

> Two `grok`s. One studying how small models suddenly *understand*.
> The other claiming to *be* understanding.
>
> This fork asks: what if we made them talk to each other?

---

## Background

In February 2023, Elon Musk publicly accused OpenAI of betraying its founding mission. On [February 19](https://x.com/elonmusk/status/1626516035863212034), he posted:

> *"OpenAI was created as an open source (which is why I named it 'Open' AI), non-profit company to serve as a counterweight to Google, but now it has become a closed source, maximum-profit company effectively controlled by Microsoft."*

By November 2023, xAI launched **Grok** as a closed product. On February 29, 2024, Musk [filed a lawsuit](https://www.courtlistener.com/docket/68235965/musk-v-altman/) against OpenAI. Eleven days later, on March 11, he [announced](https://x.com/elonmusk/status/1767108624038449405) that xAI would open-source Grok — and did so on March 17.

Meanwhile, on GitHub:

```
$ git merge openai/grok xai-org/grok-1

CONFLICT (rename/rename):  both repos are named "grok"
CONFLICT (content):        PyTorch ≠ JAX
CONFLICT (scale):          100,000 params ≠ 314,000,000,000 params
CONFLICT (purpose):        studying understanding ≠ claiming to understand

Automatic merge failed; fix conflicts and then commit the result.
```

No one had thought to connect them. This fork does.

---

## Architecture

The bridge replicates Grok-1's architectural components in PyTorch and plugs them into OpenAI's training framework:

```
         openai/grok                                        xai-org/grok-1
    ┌───────────────────┐                              ┌──────────────────────┐
    │                   │                              │                      │
    │  Sinusoidal PE    │ ─── replicated as RoPE ────► │  Rotary PE (RoPE)    │
    │                   │                              │                      │
    │  Multi-Head Attn  │ ─── replicated as MHA ─────► │  GQA (48q / 8kv)     │
    │                   │                              │                      │
    │  FFN              │                              │  MoE FFN             │
    │  ReLU(xW₁)W₂     │ ─── replicated as MoE ─────► │  8 experts, top-2    │
    │                   │                              │  GELU(xW₁)⊙(xWᵥ)W₂  │
    │                   │                              │                      │
    │  LayerNorm        │ ─── replicated as RMSNorm ─► │  RMSNorm             │
    │                   │                              │                      │
    └───────────────────┘                              └──────────────────────┘
      ~100K params                                       314,000,000K params
      PyTorch + Lightning                                JAX + Haiku
      task: 42 + 55 mod 97                               task: everything
```

The resulting class — `GrokOneTransformer` — is a miniature Grok-1 that can be trained on the same arithmetic datasets, with the same training loop, on a fundamentally different optimizer landscape.

---

## What This Does

This is not a toy. The bridge is functional.

### Bridge A: Grok-1 Architecture → OpenAI Framework

Grok-1's architectural innovations transplanted into PyTorch for grokking experiments:

- **Mixture of Experts** — 8 expert FFNs with top-2 routing, exactly as in Grok-1
- **Rotary Positional Embeddings (RoPE)** — replacing sinusoidal encoding
- **RMSNorm** — replacing standard LayerNorm
- **Gated GELU (SwiGLU-style)** — replacing ReLU FFN

```bash
# Standard grokking experiment (original, unchanged)
./grok-main/scripts/train.py --math_operator + --train_data_pct 5

# Grok-1 architecture — same task, different geometry
./grok-main/scripts/train.py --architecture grok1 --math_operator + --num_experts 8

# Auto-scaled miniature Grok-1
./grok-main/scripts/train.py --architecture grok1_mini
```

New metrics track MoE-specific grokking signals per epoch:

- **Routing entropy** — does expert selection become more uniform during generalization?
- **Expert specialization** — do memorizing models rely on "shortcut" experts?
- **Collapse index** — does routing collapse correlate with the training plateau?

### Bridge B: OpenAI Tasks → Grok-1 Inference

OpenAI's arithmetic evaluation brought to Grok-1's inference pipeline. Does a 314B language model know that 42 + 55 ≡ 0 (mod 97)?

```bash
# Dry run — inspect problem generation, no checkpoint needed
python grok-1-main/run.py --eval-grokking --operator + --n-samples 50 --dry-run

# Full evaluation (requires the 314B checkpoint, ~300GB)
python grok-1-main/run.py --eval-grokking --operator + --n-samples 100
```

### Bridge C: Config Export

Grok-1's `TransformerConfig` can now export itself as a scaled-down PyTorch-compatible config:

```python
from model import TransformerConfig

config = TransformerConfig(
    emb_size=6144, num_layers=64, num_q_heads=48,
    num_kv_heads=8, num_experts=8, num_selected_experts=2
)

# Scale down by 1/24 for a trainable experiment
mini = config.to_grokking_config(scale_factor=1/24)
# → {'d_model': 256, 'n_layers': 2, 'n_heads': 2, 'num_experts': 8, ...}
```

---

## The Research Question

Beyond the provocation, there is a real empirical question.

The 2022 grokking paper showed that dense transformers exhibit a sharp phase transition from memorization to generalization on algorithmic tasks. The transition is abrupt, nearly discontinuous, and poorly understood. But MoE models have a fundamentally different optimization geometry — the router introduces a discrete dispatch decision that creates a non-smooth loss landscape, load-balancing pressures, and the possibility of *routing collapse*.

**Does the Mixture-of-Experts architecture change the grokking phenomenon?**

Three testable hypotheses:

1. **Routing entropy as a leading indicator** — Does the expert distribution become more uniform *before* the grokking transition shows up in the loss curve? If so, routing entropy might predict generalization before it happens.

2. **Expert collapse → memorization trap** — If all tokens route to one expert, that expert memorizes everything. Grokking might require "breaking out" of this collapse first.

3. **MoE capacity delays onset** — More experts means more room to memorize without pressure to generalize. Does a larger expert count push the phase transition further out in training time, or does the routing bottleneck actually accelerate it?

None of these have been tested. This fork provides the infrastructure to test them.

---

## Quick Start

```bash
# Verify both architectures instantiate and run forward passes (no GPU needed)
python does_grok_grok.py --demo

# Run a comparative grokking experiment (standard vs Grok-1, logs to ./logs/)
python does_grok_grok.py --experiment --operator + --max-steps 50000

# Test Grok-1's arithmetic ability (dry run, no checkpoint needed)
python does_grok_grok.py --eval-grok1 --dry-run --n-samples 20
```

Expected output from `--demo`:

```
[1] Standard transformer (dense, sinusoidal PE)
    Params: 329,860  ·  2L / 4H / 128D

[2] Grok-1 architecture (MoE + RoPE + RMSNorm + gated GELU)
    Params: 4,610,148  ·  2L / 2H / 256D / 8 experts (top-2)

[3] MoE routing (layer 0):
    Expert 0  ████████░░░░░░░░  0.125
    Expert 1  ████████░░░░░░░░  0.125
    Expert 2  ████████░░░░░░░░  0.125
    ...

    Routing entropy: 2.079 / 2.079 (perfectly uniform)

[4] Config export (Grok-1 → PyTorch):
    {'d_model': 256, 'n_layers': 2, 'n_heads': 2, 'num_experts': 8, ...}

Both architectures operational. Bridge verified.
```

---

## Structure

```
.
├── does_grok_grok.py              ← unified entry point
├── README.md                      ← you are here
│
├── grok-main/                     ← openai/grok (grokking research)
│   ├── grok/
│   │   ├── transformer.py         ← MODIFIED: +GrokOneTransformer, +MoE, +RoPE, +RMSNorm
│   │   ├── training.py            ← MODIFIED: +architecture selection, +MoE metrics logging
│   │   ├── metrics.py             ← MODIFIED: +expert_utilization_entropy, +specialization
│   │   ├── data.py                ← MODIFIED: +format_for_grok1(), +eval suite generator
│   │   ├── __init__.py            ← MODIFIED: +bridge exports
│   │   ├── measure.py             ← unchanged
│   │   └── visualization.py       ← unchanged
│   ├── setup.py                   ← MODIFIED: version bump
│   ├── scripts/                   ← unchanged
│   └── nbs/                       ← unchanged
│
└── grok-1-main/                   ← xai-org/grok-1 (Grok-1 314B)
    ├── model.py                   ← MODIFIED: +to_grokking_config(), +architecture_summary()
    ├── run.py                     ← MODIFIED: +--eval-grokking mode, +arithmetic eval
    ├── runners.py                 ← unchanged
    ├── checkpoint.py              ← unchanged
    ├── tokenizer.model            ← unchanged
    └── checkpoints/               ← unchanged (download separately)
```

Design principle: **every original file still works exactly as before.** All additions are appended, never replacing. Every bridge addition is marked with a `# Bridge:` comment. `git diff` against the upstream repos shows only additions.

---

## Why

Because both projects are named `grok`, and someone had to do it.

Because Musk accused OpenAI of abandoning open source, then named his AI after a word that means to understand deeply, then open-sourced it eleven days after filing a lawsuit — while OpenAI had a research project with the same name, sitting quietly on GitHub, studying how models *learn to understand*.

Because the word "grok" deserves better than to be caught in the crossfire.

And because the answer to "Does Grok grok grokking?" is genuinely worth knowing.

---

## Sources

- [openai/grok](https://github.com/openai/grok) — Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
- [xai-org/grok-1](https://github.com/xai-org/grok-1) — Grok-1 open weights (314B MoE)
- [Power et al. (2022)](https://arxiv.org/abs/2201.02177) — "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
- [Su et al. (2021)](https://arxiv.org/abs/2104.09864) — "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- [Heinlein (1961)](https://en.wikipedia.org/wiki/Stranger_in_a_Strange_Land) — *Stranger in a Strange Land*
- [Musk on OpenAI](https://x.com/elonmusk/status/1626516035863212034) — February 19, 2023
- [xAI open-sources Grok](https://x.ai/blog/grok-os) — March 17, 2024
- [Musk v. Altman](https://www.courtlistener.com/docket/68235965/musk-v-altman/) — Filed February 29, 2024

---

## License

- `grok-main/` — [MIT License](grok-main/LICENSE) (original)
- `grok-1-main/` — [Apache 2.0](grok-1-main/LICENSE.txt) (original)
- Bridge code — public domain

---

<div align="center">
<sub>
<em>"The word is much wider in meaning than any English word conceived to date —<br>
it means to understand so thoroughly that the observer becomes a part of the observed."</em>
<br><br>
— Heinlein, via Jubal Harshaw
</sub>
</div>
