#!/usr/bin/env python
"""
Simple example of running a grokking experiment programmatically.

This script demonstrates how to use the modular addition grokking
modules directly in Python code without command-line arguments.
"""

import torch
from argparse import Namespace
from grok.train_modular_addition import train


def run_simple_experiment():
    """
    Run a simple grokking experiment with default parameters.

    This demonstrates:
    - Setting up hyperparameters
    - Running training
    - Monitoring results
    """
    print("\n" + "=" * 80)
    print("Simple Grokking Experiment")
    print("=" * 80 + "\n")

    # Define hyperparameters
    hparams = Namespace(
        # Data parameters
        modulus=97,                    # Prime modulus
        train_fraction=0.3,            # 30% of data for training
        seed=42,                       # Random seed

        # Model parameters
        num_layers=2,                  # 2 transformer layers
        num_heads=4,                   # 4 attention heads
        d_model=128,                   # 128-dimensional embeddings
        dropout=0.0,                   # No dropout
        max_len=50,                    # Maximum sequence length
        non_linearity="relu",          # ReLU activation
        weight_noise=0.0,              # No weight noise

        # Training parameters
        batch_size=512,                # Batch size for training
        val_batch_size=512,            # Batch size for validation
        learning_rate=1e-3,            # Learning rate
        weight_decay=1.0,              # Weight decay (important for grokking)
        warmup_steps=50,               # Number of warmup steps
        max_steps=10000,               # Maximum training steps
        max_epochs=None,               # No epoch limit (use max_steps)

        # Logging parameters
        log_dir="logs/simple_example",
        experiment_name="simple_grokking_p97_alpha0.3",

        # Hardware
        gpu=0 if torch.cuda.is_available() else -1,
    )

    # Print experiment setup
    print("Experiment Configuration:")
    print(f"  Problem: (x + y) mod {hparams.modulus}")
    print(f"  Training fraction (α): {hparams.train_fraction}")
    print(f"  Dataset size: {hparams.modulus}² = {hparams.modulus**2:,}")
    print(f"  Training samples: {int(hparams.modulus**2 * hparams.train_fraction):,}")
    print(f"  Validation samples: {int(hparams.modulus**2 * (1-hparams.train_fraction)):,}")
    print(f"\nModel Configuration:")
    print(f"  Layers: {hparams.num_layers}")
    print(f"  Heads: {hparams.num_heads}")
    print(f"  Model dimension: {hparams.d_model}")
    print(f"\nTraining Configuration:")
    print(f"  Max steps: {hparams.max_steps:,}")
    print(f"  Learning rate: {hparams.learning_rate}")
    print(f"  Weight decay: {hparams.weight_decay}")
    print(f"  Batch size: {hparams.batch_size}")
    print(f"\n" + "=" * 80 + "\n")

    # Run training
    try:
        log_dir = train(hparams)

        print("\n" + "=" * 80)
        print("Experiment Complete!")
        print("=" * 80)
        print(f"\nResults saved to: {log_dir}")
        print("\nTo visualize results, run:")
        print(f"  python scripts/visualize_grokking.py --log_dir {log_dir}")
        print("\n" + "=" * 80 + "\n")

        return log_dir

    except Exception as e:
        print(f"\nError during training: {e}")
        raise


def run_alpha_comparison():
    """
    Run multiple experiments with different alpha values for comparison.

    This demonstrates how to run systematic experiments programmatically.
    """
    print("\n" + "=" * 80)
    print("Alpha Comparison Experiment")
    print("=" * 80 + "\n")

    # Alpha values to test
    alpha_values = [0.1, 0.3, 0.5, 0.7]

    print(f"Running {len(alpha_values)} experiments with α = {alpha_values}\n")

    results = []

    for i, alpha in enumerate(alpha_values, 1):
        print(f"\n{'='*80}")
        print(f"Experiment {i}/{len(alpha_values)}: α = {alpha}")
        print(f"{'='*80}\n")

        # Create hyperparameters for this experiment
        hparams = Namespace(
            modulus=97,
            train_fraction=alpha,
            seed=0,

            num_layers=2,
            num_heads=4,
            d_model=128,
            dropout=0.0,
            max_len=50,
            non_linearity="relu",
            weight_noise=0.0,

            batch_size=512,
            val_batch_size=512,
            learning_rate=1e-3,
            weight_decay=1.0,
            warmup_steps=50,
            max_steps=20000,  # Shorter for comparison
            max_epochs=None,

            log_dir="logs/alpha_comparison",
            experiment_name=f"alpha_{alpha:.2f}",

            gpu=0 if torch.cuda.is_available() else -1,
        )

        try:
            log_dir = train(hparams)
            results.append({
                "alpha": alpha,
                "log_dir": log_dir,
                "success": True
            })
        except Exception as e:
            print(f"Error in experiment with α={alpha}: {e}")
            results.append({
                "alpha": alpha,
                "log_dir": None,
                "success": False
            })

    # Print summary
    print("\n" + "=" * 80)
    print("All Experiments Complete!")
    print("=" * 80)
    print(f"\n{'Alpha':<10} {'Status':<15} {'Log Directory'}")
    print("-" * 80)

    for result in results:
        status = "✓ Success" if result["success"] else "✗ Failed"
        log_dir = result["log_dir"] if result["log_dir"] else "N/A"
        print(f"{result['alpha']:<10.2f} {status:<15} {log_dir}")

    print("-" * 80)
    print(f"\nTo visualize all results, run:")
    print(f"  python scripts/visualize_grokking.py --log_dir logs/alpha_comparison")
    print("\n" + "=" * 80 + "\n")

    return results


if __name__ == "__main__":
    import sys

    # Parse command line argument
    if len(sys.argv) > 1 and sys.argv[1] == "comparison":
        # Run alpha comparison
        run_alpha_comparison()
    else:
        # Run simple experiment
        run_simple_experiment()

        print("\nTip: To run alpha comparison, use:")
        print("  python examples/simple_grokking_experiment.py comparison\n")
