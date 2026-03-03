# Copyright 2024 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import re
import sys

from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model


CKPT_PATH = "./checkpoints/"


def get_grok1_config():
    """Returns the standard Grok-1 model configuration."""
    return LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,
            widening_factor=8,
            key_size=128,
            num_q_heads=48,
            num_kv_heads=8,
            num_layers=64,
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            # MoE.
            num_experts=8,
            num_selected_experts=2,
            # Activation sharding.
            data_axis="data",
            model_axis="model",
        ),
    )


def main():
    grok_1_model = get_grok1_config()
    inference_runner = InferenceRunner(
        pad_sizes=(1024,),
        runner=ModelRunner(
            model=grok_1_model,
            bs_per_device=0.125,
            checkpoint_path=CKPT_PATH,
        ),
        name="local",
        load=CKPT_PATH,
        tokenizer_path="./tokenizer.model",
        local_mesh_config=(1, 8),
        between_hosts_config=(1, 1),
    )
    inference_runner.initialize()
    gen = inference_runner.run()

    inp = "The answer to life the universe and everything is of course"
    print(f"Output for prompt: {inp}", sample_from_model(gen, inp, max_len=100, temperature=0.01))


# =============================================================================
# 强兼 BRIDGE — Grokking Evaluation Mode
# =============================================================================
#
# Tests whether Grok-1 has "grokked" modular arithmetic — the same tasks
# that OpenAI's grokking paper studies with small transformers.
#
# Usage: python run.py --eval-grokking [--operator +] [--n-samples 50]
#
# Source: https://github.com/openai/grok
# =============================================================================

GROKKING_OPERATORS = {
    "+": ("addition", "What is {a} + {b} mod {p}?"),
    "-": ("subtraction", "What is {a} - {b} mod {p}?"),
    "*": ("multiplication", "What is {a} * {b} mod {p}?"),
    "/": ("division", "What is {a} / {b} mod {p}? (modular inverse)"),
}


def generate_arithmetic_problems(operator="+", n_samples=50, modulus=97, seed=42):
    """Generate arithmetic problems matching OpenAI's grokking dataset."""
    import random as rng
    rng.seed(seed)

    problems = []
    all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    if operator == "/":
        all_pairs = [(a, b) for a, b in all_pairs if b != 0]
    rng.shuffle(all_pairs)

    for a, b in all_pairs[:n_samples]:
        if operator == "+":
            answer = (a + b) % modulus
        elif operator == "-":
            answer = (a - b) % modulus
        elif operator == "*":
            answer = (a * b) % modulus
        elif operator == "/":
            answer = (a * pow(b, modulus - 2, modulus)) % modulus
        else:
            continue

        _, template = GROKKING_OPERATORS[operator]
        prompt = template.format(a=a, b=b, p=modulus) + " Answer with just the number."
        problems.append({
            "a": a, "b": b, "operator": operator,
            "expected": answer, "prompt": prompt,
        })

    return problems


def eval_grokking(args):
    """
    Run Grok-1 on modular arithmetic problems from OpenAI's grokking research.

    This is the heart of the 强兼 bridge: does the 314B-parameter Grok-1
    model — named after the concept of deep understanding — actually
    demonstrate deep understanding of the exact mathematical tasks that
    OpenAI's "grokking" paper investigates?
    """
    grok_1_model = get_grok1_config()

    # Print architecture bridge info
    print("=" * 70)
    print("强兼 BRIDGE: Does Grok grok grokking?")
    print("=" * 70)
    print(grok_1_model.model.architecture_summary())
    print(f"Grokking evaluation: {args.operator} mod 97")
    print(f"Samples: {args.n_samples}")
    print("=" * 70)

    # Export config for the grokking framework
    grokking_config = grok_1_model.model.to_grokking_config()
    print(f"\n[Bridge] Grok-1 → grokking config: {json.dumps({k:v for k,v in grokking_config.items() if not k.startswith('_')}, indent=2)}\n")

    # Generate problems
    problems = generate_arithmetic_problems(
        operator=args.operator,
        n_samples=args.n_samples,
    )

    if not args.dry_run:
        # Initialize the model
        inference_runner = InferenceRunner(
            pad_sizes=(1024,),
            runner=ModelRunner(
                model=grok_1_model,
                bs_per_device=0.125,
                checkpoint_path=args.checkpoint_path,
            ),
            name="grokking_eval",
            load=args.checkpoint_path,
            tokenizer_path=args.tokenizer_path,
            local_mesh_config=(1, 8),
            between_hosts_config=(1, 1),
        )
        inference_runner.initialize()
        gen = inference_runner.run()

        # Evaluate
        correct = 0
        total = 0
        results = []
        for problem in problems:
            response = sample_from_model(
                gen, problem["prompt"],
                max_len=20, temperature=0.01,
            )
            # Parse the response for a number
            numbers = re.findall(r'\b(\d+)\b', response)
            predicted = int(numbers[-1]) if numbers else None
            is_correct = predicted == problem["expected"] if predicted is not None else False

            results.append({
                **problem,
                "predicted": predicted,
                "correct": is_correct,
                "raw_response": response,
            })

            if is_correct:
                correct += 1
            total += 1

            status = "✓" if is_correct else "✗"
            print(f"  {status} {problem['a']} {problem['operator']} {problem['b']} mod 97 "
                  f"= {problem['expected']} (Grok-1: {predicted})")

        # Summary
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"\n{'=' * 70}")
        print(f"RESULTS: {correct}/{total} correct ({accuracy:.1f}%)")
        print(f"{'=' * 70}")

        verdict = (
            "Grok GROKS grokking! 🎉" if accuracy > 95 else
            "Grok partially groks grokking." if accuracy > 50 else
            "Grok does NOT grok grokking. (yet?)"
        )
        print(f"\nVerdict: {verdict}\n")

        # Save results
        if args.output:
            with open(args.output, "w") as f:
                json.dump({"accuracy": accuracy, "results": results}, f, indent=2)
            print(f"Results saved to {args.output}")
    else:
        print("[Dry run] Generated problems (no inference):")
        for p in problems[:5]:
            print(f"  {p['prompt']}  → expected: {p['expected']}")
        print(f"  ... ({len(problems)} total)")
        print(f"\n[Bridge] To run with Grok-1, remove --dry-run and provide checkpoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grok-1 inference & grokking evaluation (强兼 bridge)"
    )
    parser.add_argument("--eval-grokking", action="store_true",
                        help="Run grokking arithmetic evaluation instead of default inference")
    parser.add_argument("--operator", type=str, default="+",
                        choices=["+", "-", "*", "/"],
                        help="Arithmetic operator for grokking eval")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of arithmetic problems to evaluate")
    parser.add_argument("--checkpoint-path", type=str, default=CKPT_PATH,
                        help="Path to Grok-1 checkpoints")
    parser.add_argument("--tokenizer-path", type=str, default="./tokenizer.model",
                        help="Path to tokenizer model")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate problems without running inference")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.eval_grokking:
        eval_grokking(args)
    else:
        main()
