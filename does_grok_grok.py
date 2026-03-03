#!/usr/bin/env python
"""
does_grok_grok.py — The unified entry point of 强兼 (forceful compatibility)

                    ╔═══════════════════════════════╗
                    ║    Does Grok grok grokking?    ║
                    ╚═══════════════════════════════╝

This script bridges two open-source projects that share a name but nothing else:

  • openai/grok  — Studies "grokking": the phenomenon where neural networks
                   suddenly generalize after prolonged memorization. Trains
                   small transformers on modular arithmetic. (PyTorch)

  • xai-org/grok-1 — The 314B-parameter Mixture-of-Experts language model
                      from xAI, open-sourced under Apache 2.0. (JAX/Haiku)

This script asks a simple question with a beautiful double meaning:
Does Grok (the model) grok (deeply understand) grokking (the phenomenon)?

We answer it in two ways:

  MODE 1: "Grok-1 Architecture Grokking" (--experiment)
    Train a miniature version of Grok-1's architecture (MoE + RoPE + RMSNorm
    + gated GELU) on OpenAI's grokking benchmark tasks, and compare the
    grokking dynamics against the original dense transformer. Do MoE models
    grok differently? Does the expert routing change during the phase
    transition from memorization to generalization?

  MODE 2: "Does Grok-1 Know Arithmetic?" (--eval-grok1)
    Generate modular arithmetic problems from OpenAI's grokking dataset
    and evaluate Grok-1 on them. (Requires Grok-1 checkpoint.)

Usage:
    # Compare grokking: standard transformer vs. Grok-1 architecture
    python does_grok_grok.py --experiment --operator + --max-steps 50000

    # Quick demo (no training, just shows the architecture bridge)
    python does_grok_grok.py --demo

    # Evaluate Grok-1 on arithmetic (requires checkpoint)
    python does_grok_grok.py --eval-grok1 --checkpoint ./grok-1-main/checkpoints/

Copyright: This bridge is a work of 行为艺术 (behavioral art).
License: Both source projects' licenses apply to their respective code.
         The bridge code itself is unlicensed — do whatever you want with it.
"""

import argparse
import sys
import os
import json
import time

# Add both project directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "grok-main"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "grok-1-main"))


BANNER = r"""
   ╭──────────────────────────────────────────────────────────────╮
   │                                                              │
   │           ██████╗ ██████╗  ██████╗ ██╗  ██╗                  │
   │          ██╔════╝ ██╔══██╗██╔═══██╗██║ ██╔╝                  │
   │          ██║  ███╗██████╔╝██║   ██║█████╔╝                   │
   │          ██║   ██║██╔══██╗██║   ██║██╔═██╗                   │
   │          ╚██████╔╝██║  ██║╚██████╔╝██║  ██╗                  │
   │           ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝                │
   │                                                              │
   │                         强  兼                               │
   │              — forceful compatibility —                       │
   │                                                              │
   │           openai/grok  ←──bridge──→  xai-org/grok-1          │
   │           (grokking)                   (Grok-1 314B)         │
   │           PyTorch                      JAX/Haiku             │
   │                                                              │
   │           "Does Grok grok grokking?"                         │
   │                                                              │
   ╰──────────────────────────────────────────────────────────────╯
"""


def demo():
    """
    Demonstrate the 强兼 bridge without training or inference.
    Shows that Grok-1's architecture can be instantiated in PyTorch
    at grokking-experiment scale.
    """
    print(BANNER)
    print("=" * 66)
    print("  MODE: Demo — Architecture Bridge Verification")
    print("=" * 66)

    # Import from OpenAI's grokking framework
    from grok.transformer import Transformer, GrokOneTransformer
    from grok.data import ArithmeticDataset, MODULUS

    print("\n[1] Original OpenAI transformer (dense, sinusoidal PE):")
    standard = Transformer(
        n_layers=2, n_heads=4, d_model=128,
        dropout=0.0, max_context_len=50, vocab_len=100,
    )
    n_params_standard = sum(p.numel() for p in standard.parameters())
    print(f"    Params: {n_params_standard:,}")
    print(f"    Architecture: {standard.n_layers}L / {standard.n_heads}H / {standard.d_model}D")

    print("\n[2] Grok-1 architecture (MoE + RoPE + RMSNorm + gated GELU):")
    grok1_mini = GrokOneTransformer.from_grok1_config(
        scale_factor=1/24, vocab_len=100, max_context_len=50,
    )
    n_params_grok1 = sum(p.numel() for p in grok1_mini.parameters())
    print(f"    Params: {n_params_grok1:,}")
    print(f"    Architecture: {grok1_mini.n_layers}L / {grok1_mini.n_heads}H / "
          f"{grok1_mini.d_model}D / {grok1_mini.num_experts}E (top-{grok1_mini.num_selected_experts})")

    print(f"\n[3] Parameter ratio: Grok-1-mini is {n_params_grok1/n_params_standard:.1f}x "
          f"the standard transformer")
    print(f"    (Real Grok-1 is ~314B params — "
          f"{314_000_000_000/n_params_grok1:.0f}x this miniature)")

    # Quick forward pass test
    import torch
    print("\n[4] Forward pass test:")
    dummy_input = torch.randint(0, 100, (2, 10))  # batch=2, seq=10

    with torch.no_grad():
        out_std, _, _ = standard(dummy_input)
        out_grok, _, _ = grok1_mini(dummy_input)

    print(f"    Standard:  input {tuple(dummy_input.shape)} → output {tuple(out_std.shape)}")
    print(f"    Grok-1:    input {tuple(dummy_input.shape)} → output {tuple(out_grok.shape)}")

    # Show routing info
    if grok1_mini.last_router_probs:
        rp = grok1_mini.last_router_probs[0]
        print(f"\n[5] MoE Routing (layer 0):")
        print(f"    Router probs shape: {tuple(rp.shape)}")
        mean_probs = rp.mean(dim=(0, 1))
        for i, p in enumerate(mean_probs):
            bar = "█" * int(p.item() * 40)
            print(f"    Expert {i}: {p.item():.3f} {bar}")

    # Cross-framework config export
    print("\n[6] Cross-framework config export (JAX → PyTorch):")
    try:
        from model import TransformerConfig as Grok1TransformerConfig
        grok1_config = Grok1TransformerConfig(
            emb_size=48 * 128, widening_factor=8, key_size=128,
            num_q_heads=48, num_kv_heads=8, num_layers=64,
            num_experts=8, num_selected_experts=2,
        )
        exported = grok1_config.to_grokking_config()
        print(f"    Grok-1 ({grok1_config.emb_size}D, {grok1_config.num_layers}L) →")
        print(f"    Grokking ({exported['d_model']}D, {exported['n_layers']}L, "
              f"{exported['num_experts']}E)")
    except ImportError:
        print("    (Grok-1 model.py not in path — run from project root)")

    print("\n" + "=" * 66)
    print("  Bridge verified. Both architectures are alive and compatible.")
    print("  Run with --experiment to compare grokking dynamics.")
    print("=" * 66 + "\n")


def run_experiment(args):
    """
    Train both architectures on the same grokking task and compare.
    """
    print(BANNER)
    print("=" * 66)
    print("  MODE: Comparative Grokking Experiment")
    print(f"  Task: {args.operator} mod 97  |  Steps: {args.max_steps}")
    print("=" * 66)

    from grok.training import train, add_args

    # Prepare base arguments
    parser = add_args()
    base_args = [
        "--math_operator", args.operator,
        "--max_steps", str(args.max_steps),
        "--train_data_pct", str(args.train_pct),
        "--d_model", str(args.d_model),
        "--n_layers", str(args.n_layers),
        "--n_heads", str(args.n_heads),
    ]

    # Run 1: Standard transformer
    print("\n" + "─" * 66)
    print("  [1/2] Training STANDARD transformer (OpenAI's original)")
    print("─" * 66)
    std_args = parser.parse_args(base_args + [
        "--architecture", "standard",
        "--logdir", os.path.join(args.logdir, "standard"),
    ])
    t0 = time.time()
    train(std_args)
    t_std = time.time() - t0

    # Run 2: Grok-1 architecture
    print("\n" + "─" * 66)
    print("  [2/2] Training GROK-1 architecture (MoE + RoPE + RMSNorm)")
    print("─" * 66)
    grok1_args = parser.parse_args(base_args + [
        "--architecture", "grok1",
        "--num_experts", str(args.num_experts),
        "--num_selected_experts", str(args.num_selected_experts),
        "--logdir", os.path.join(args.logdir, "grok1"),
    ])
    t0 = time.time()
    train(grok1_args)
    t_grok1 = time.time() - t0

    print("\n" + "=" * 66)
    print("  EXPERIMENT COMPLETE")
    print(f"  Standard: {t_std:.1f}s  |  Grok-1: {t_grok1:.1f}s")
    print(f"  Logs: {args.logdir}/")
    print("=" * 66)
    print("\n  Compare training curves to see if MoE affects grokking dynamics.")
    print("  Key questions:")
    print("    • Does the Grok-1 architecture grok faster or slower?")
    print("    • Does expert routing entropy change at the grokking transition?")
    print("    • Do MoE models find different generalization shortcuts?\n")


def eval_grok1(args):
    """Evaluate Grok-1 on grokking arithmetic (delegates to grok-1-main/run.py)."""
    print(BANNER)
    os.chdir(os.path.join(os.path.dirname(__file__), "grok-1-main"))
    from run import eval_grokking
    eval_args = argparse.Namespace(
        operator=args.operator,
        n_samples=args.n_samples,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        output=args.output,
        dry_run=args.dry_run,
    )
    eval_grokking(eval_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="强兼 — Does Grok grok grokking?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python does_grok_grok.py --demo
  python does_grok_grok.py --experiment --operator + --max-steps 50000
  python does_grok_grok.py --eval-grok1 --dry-run
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true",
                      help="Demo: verify the architecture bridge works")
    mode.add_argument("--experiment", action="store_true",
                      help="Run comparative grokking experiment")
    mode.add_argument("--eval-grok1", action="store_true",
                      help="Evaluate Grok-1 on arithmetic tasks")

    # Experiment args
    parser.add_argument("--operator", type=str, default="+")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--train-pct", type=float, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--num-selected-experts", type=int, default=2)
    parser.add_argument("--logdir", type=str, default="logs/qiangjian")

    # Eval args
    parser.add_argument("--checkpoint", type=str, default="./grok-1-main/checkpoints/")
    parser.add_argument("--tokenizer", type=str, default="./grok-1-main/tokenizer.model")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.experiment:
        run_experiment(args)
    elif args.eval_grok1:
        eval_grok1(args)
