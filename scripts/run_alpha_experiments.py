#!/usr/bin/env python
"""
Run grokking experiments with different alpha (training fraction) values.

This script automates running multiple experiments to study how the
training data fraction affects the grokking phenomenon in modular addition.

Usage:
    python scripts/run_alpha_experiments.py --modulus 97 --alpha_values 0.1 0.3 0.5 0.7

Example configurations:
    1. Quick test (small p, few alphas):
       python scripts/run_alpha_experiments.py --modulus 59 --alpha_values 0.2 0.4 0.6

    2. Full experiment (multiple alphas):
       python scripts/run_alpha_experiments.py --modulus 97 --alpha_values 0.05 0.1 0.2 0.3 0.4 0.5

    3. Multiple primes and alphas:
       python scripts/run_alpha_experiments.py --modulus 59 97 113 --alpha_values 0.1 0.3 0.5
"""

import argparse
import os
import subprocess
import sys
import json
import time
from typing import List
from pathlib import Path
import itertools


def run_single_experiment(
    modulus: int,
    alpha: float,
    base_args: argparse.Namespace,
) -> dict:
    """
    Run a single experiment with given modulus and alpha.

    :param modulus: Prime modulus
    :param alpha: Training fraction
    :param base_args: Base arguments for the experiment
    :returns: Dictionary with experiment results
    """
    print("\n" + "=" * 80)
    print(f"Running experiment: p={modulus}, α={alpha}")
    print("=" * 80)

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "grok.train_modular_addition",
        "--modulus",
        str(modulus),
        "--train_fraction",
        str(alpha),
        "--num_layers",
        str(base_args.num_layers),
        "--num_heads",
        str(base_args.num_heads),
        "--d_model",
        str(base_args.d_model),
        "--learning_rate",
        str(base_args.learning_rate),
        "--weight_decay",
        str(base_args.weight_decay),
        "--batch_size",
        str(base_args.batch_size),
        "--max_steps",
        str(base_args.max_steps),
        "--warmup_steps",
        str(base_args.warmup_steps),
        "--seed",
        str(base_args.seed),
        "--log_dir",
        base_args.log_dir,
        "--gpu",
        str(base_args.gpu),
    ]

    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
        success = True
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.CalledProcessError as e:
        success = False
        stdout = e.stdout
        stderr = e.stderr
        print(f"Error running experiment: {e}")

    elapsed_time = time.time() - start_time

    # Extract log directory from stdout (last line should contain path)
    log_dir = None
    if success:
        for line in stdout.split("\n"):
            if "Logs saved to:" in line or "Results saved to:" in line:
                log_dir = line.split(":")[-1].strip()

    result_dict = {
        "modulus": modulus,
        "alpha": alpha,
        "success": success,
        "elapsed_time": elapsed_time,
        "log_dir": log_dir,
    }

    if success:
        print(f"✓ Experiment completed successfully in {elapsed_time:.1f}s")
        print(f"  Log directory: {log_dir}")
    else:
        print(f"✗ Experiment failed after {elapsed_time:.1f}s")
        if stderr:
            print(f"  Error: {stderr[:200]}")

    return result_dict


def save_experiment_summary(results: List[dict], output_dir: str):
    """
    Save summary of all experiments.

    :param results: List of result dictionaries
    :param output_dir: Directory to save summary
    """
    summary_file = os.path.join(output_dir, "experiment_summary.json")

    summary = {
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "total_time": sum(r["elapsed_time"] for r in results),
        "results": results,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExperiment summary saved to: {summary_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(
        f"{'Modulus':<10} {'Alpha':<10} {'Status':<10} {'Time (s)':<12} {'Log Dir'}"
    )
    print("-" * 80)

    for r in results:
        status = "✓ Success" if r["success"] else "✗ Failed"
        log_dir = r["log_dir"] if r["log_dir"] else "N/A"
        # Shorten log dir for display
        if len(log_dir) > 40:
            log_dir = "..." + log_dir[-37:]

        print(
            f"{r['modulus']:<10} {r['alpha']:<10.3f} {status:<10} "
            f"{r['elapsed_time']:<12.1f} {log_dir}"
        )

    print("-" * 80)
    print(
        f"Total: {summary['successful']}/{summary['total_experiments']} successful "
        f"(Total time: {summary['total_time']:.1f}s)"
    )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run grokking experiments with different alpha values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Experiment parameters
    parser.add_argument(
        "--modulus",
        type=int,
        nargs="+",
        default=[97],
        help="Prime modulus values to test (can specify multiple)",
    )
    parser.add_argument(
        "--alpha_values",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Training fraction values to test (can specify multiple)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Random seeds for multiple runs (default: [0])",
    )

    # Model parameters
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--warmup_steps", type=int, default=50)

    # Logging
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/alpha_experiments",
        help="Base directory for all experiment logs",
    )

    # Hardware
    parser.add_argument("--gpu", type=int, default=0, help="GPU device (-1 for CPU)")

    # Execution control
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print experiments to run without executing",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.log_dir, exist_ok=True)

    # Generate all experiment combinations
    experiments = list(
        itertools.product(args.modulus, args.alpha_values, args.seeds)
    )

    print("\n" + "=" * 80)
    print("ALPHA GROKKING EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Total experiments: {len(experiments)}")
    print(f"Modulus values: {args.modulus}")
    print(f"Alpha values: {args.alpha_values}")
    print(f"Seeds: {args.seeds}")
    print(f"Model: {args.num_layers} layers, {args.num_heads} heads, d={args.d_model}")
    print(f"Training: {args.max_steps} steps, lr={args.learning_rate}, wd={args.weight_decay}")
    print(f"Output directory: {args.log_dir}")
    print("=" * 80)

    if args.dry_run:
        print("\nDRY RUN - Experiments to be executed:")
        for i, (modulus, alpha, seed) in enumerate(experiments, 1):
            print(f"  {i}. p={modulus}, α={alpha:.3f}, seed={seed}")
        print(f"\nTotal: {len(experiments)} experiments")
        return

    # Confirm with user
    response = input(
        f"\nProceed with {len(experiments)} experiments? [y/N]: "
    )
    if response.lower() not in ["y", "yes"]:
        print("Aborted.")
        return

    # Run all experiments
    results = []
    start_time = time.time()

    for i, (modulus, alpha, seed) in enumerate(experiments, 1):
        print(f"\n\n{'=' * 80}")
        print(f"Experiment {i}/{len(experiments)}")
        print(f"{'=' * 80}")

        # Update seed in args
        args.seed = seed

        result = run_single_experiment(
            modulus=modulus, alpha=alpha, base_args=args
        )
        results.append(result)

    total_time = time.time() - start_time

    # Save summary
    save_experiment_summary(results, args.log_dir)

    print(f"\n\nAll experiments completed in {total_time:.1f}s")
    print(f"Results saved to: {args.log_dir}")


if __name__ == "__main__":
    main()
