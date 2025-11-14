#!/usr/bin/env python
"""
Visualize grokking phenomenon from experiment results.

This script creates plots to analyze how training data fraction (alpha)
affects the grokking phenomenon in modular addition tasks.

Usage:
    python scripts/visualize_grokking.py --log_dir logs/alpha_experiments
    python scripts/visualize_grokking.py --log_dir logs/alpha_experiments --modulus 97
"""

import argparse
import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use non-interactive backend if no display available
mpl.use('Agg')


def load_metrics_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load metrics from PyTorch Lightning CSV logger.

    :param csv_path: Path to metrics.csv file
    :returns: DataFrame with metrics
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def find_experiment_logs(base_dir: str, modulus: int = None) -> List[Dict]:
    """
    Find all experiment log directories.

    :param base_dir: Base log directory
    :param modulus: Optional filter by modulus
    :returns: List of experiment info dictionaries
    """
    experiments = []

    # Find all subdirectories
    for subdir in Path(base_dir).rglob("*"):
        if not subdir.is_dir():
            continue

        # Look for hparams.json
        hparams_file = subdir / "hparams.json"
        if not hparams_file.exists():
            continue

        # Load hparams
        with open(hparams_file, "r") as f:
            hparams = json.load(f)

        # Filter by modulus if specified
        if modulus is not None and hparams.get("modulus") != modulus:
            continue

        # Look for metrics CSV
        csv_files = list(subdir.rglob("metrics.csv"))
        if not csv_files:
            csv_files = list(subdir.rglob("*/metrics.csv"))

        if csv_files:
            experiments.append(
                {
                    "path": str(subdir),
                    "csv_path": str(csv_files[0]),
                    "hparams": hparams,
                }
            )

    return experiments


def detect_grokking(df: pd.DataFrame, threshold: float = 95.0) -> Dict:
    """
    Detect grokking phenomenon in training curve.

    Grokking is characterized by:
    1. Initial learning: train accuracy increases
    2. Delayed generalization: val accuracy stays low while train accuracy is high
    3. Sudden generalization: val accuracy rapidly increases to match train accuracy

    :param df: DataFrame with train_acc and val_acc columns
    :param threshold: Accuracy threshold to consider "solved"
    :returns: Dictionary with grokking statistics
    """
    # Remove NaN values
    df = df.dropna(subset=["train_acc", "val_acc"])

    if len(df) == 0:
        return {
            "grokking_detected": False,
            "train_solve_step": None,
            "val_solve_step": None,
            "grokking_delay": None,
        }

    # Find when train accuracy crosses threshold
    train_solved = df[df["train_acc"] >= threshold]
    train_solve_step = train_solved["step"].min() if len(train_solved) > 0 else None

    # Find when val accuracy crosses threshold
    val_solved = df[df["val_acc"] >= threshold]
    val_solve_step = val_solved["step"].min() if len(val_solved) > 0 else None

    # Calculate grokking delay
    if train_solve_step is not None and val_solve_step is not None:
        grokking_delay = val_solve_step - train_solve_step
        grokking_detected = grokking_delay > 100  # Significant delay
    else:
        grokking_delay = None
        grokking_detected = False

    return {
        "grokking_detected": grokking_detected,
        "train_solve_step": float(train_solve_step) if train_solve_step is not None else None,
        "val_solve_step": float(val_solve_step) if val_solve_step is not None else None,
        "grokking_delay": float(grokking_delay) if grokking_delay is not None else None,
        "final_train_acc": float(df["train_acc"].iloc[-1]),
        "final_val_acc": float(df["val_acc"].iloc[-1]),
    }


def plot_single_experiment(
    df: pd.DataFrame, hparams: Dict, output_path: str
) -> None:
    """
    Plot training curves for a single experiment.

    :param df: DataFrame with metrics
    :param hparams: Hyperparameters dictionary
    :param output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Accuracy curves
    ax = axes[0]
    if "train_acc" in df.columns:
        ax.plot(
            df["step"],
            df["train_acc"],
            label="Train Accuracy",
            linewidth=2,
            alpha=0.8,
        )
    if "val_acc" in df.columns:
        ax.plot(
            df["step"],
            df["val_acc"],
            label="Validation Accuracy",
            linewidth=2,
            alpha=0.8,
        )
    if "full_train_acc" in df.columns:
        ax.plot(
            df["step"],
            df["full_train_acc"],
            label="Full Train Accuracy",
            linewidth=1,
            alpha=0.6,
            linestyle="--",
        )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Grokking: p={hparams['modulus']}, α={hparams['train_fraction']:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Plot 2: Loss curves
    ax = axes[1]
    if "train_loss" in df.columns:
        ax.plot(
            df["step"],
            df["train_loss"],
            label="Train Loss",
            linewidth=2,
            alpha=0.8,
        )
    if "val_loss" in df.columns:
        ax.plot(
            df["step"],
            df["val_loss"],
            label="Validation Loss",
            linewidth=2,
            alpha=0.8,
        )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Loss Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved plot: {output_path}")


def plot_alpha_comparison(
    experiments: List[Dict], output_path: str, modulus: int = None
) -> None:
    """
    Plot comparison of different alpha values.

    :param experiments: List of experiment dictionaries
    :param output_path: Path to save plot
    :param modulus: Modulus to filter by
    """
    if modulus is not None:
        experiments = [
            e for e in experiments if e["hparams"]["modulus"] == modulus
        ]

    if len(experiments) == 0:
        print("No experiments found for comparison")
        return

    # Sort by alpha
    experiments = sorted(
        experiments, key=lambda e: e["hparams"]["train_fraction"]
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))

    # Plot 1: Training accuracy
    ax = axes[0]
    for i, exp in enumerate(experiments):
        df = load_metrics_from_csv(exp["csv_path"])
        if df is None or "train_acc" not in df.columns:
            continue

        alpha = exp["hparams"]["train_fraction"]
        label = f"α={alpha:.3f}"

        df_clean = df.dropna(subset=["train_acc"])
        ax.plot(
            df_clean["step"],
            df_clean["train_acc"],
            label=label,
            linewidth=2,
            alpha=0.7,
            color=colors[i],
        )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Train Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Training Accuracy vs Alpha (p={modulus if modulus else 'various'})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Plot 2: Validation accuracy
    ax = axes[1]
    for i, exp in enumerate(experiments):
        df = load_metrics_from_csv(exp["csv_path"])
        if df is None or "val_acc" not in df.columns:
            continue

        alpha = exp["hparams"]["train_fraction"]
        label = f"α={alpha:.3f}"

        df_clean = df.dropna(subset=["val_acc"])
        ax.plot(
            df_clean["step"],
            df_clean["val_acc"],
            label=label,
            linewidth=2,
            alpha=0.7,
            color=colors[i],
        )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title(
        "Validation Accuracy vs Alpha (Grokking Effect)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved comparison plot: {output_path}")


def plot_grokking_analysis(
    experiments: List[Dict], output_path: str, modulus: int = None
) -> None:
    """
    Create grokking analysis plot showing delay vs alpha.

    :param experiments: List of experiment dictionaries
    :param output_path: Path to save plot
    :param modulus: Modulus to filter by
    """
    if modulus is not None:
        experiments = [
            e for e in experiments if e["hparams"]["modulus"] == modulus
        ]

    # Analyze each experiment
    analysis_data = []
    for exp in experiments:
        df = load_metrics_from_csv(exp["csv_path"])
        if df is None:
            continue

        grokking_stats = detect_grokking(df)
        analysis_data.append(
            {
                "alpha": exp["hparams"]["train_fraction"],
                "modulus": exp["hparams"]["modulus"],
                **grokking_stats,
            }
        )

    if len(analysis_data) == 0:
        print("No data available for grokking analysis")
        return

    df_analysis = pd.DataFrame(analysis_data)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Grokking delay vs alpha
    ax = axes[0, 0]
    mask = df_analysis["grokking_delay"].notna()
    if mask.sum() > 0:
        ax.scatter(
            df_analysis[mask]["alpha"],
            df_analysis[mask]["grokking_delay"],
            s=100,
            alpha=0.6,
        )
        ax.set_xlabel("Training Fraction (α)", fontsize=12)
        ax.set_ylabel("Grokking Delay (steps)", fontsize=12)
        ax.set_title("Grokking Delay vs Alpha", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Plot 2: Steps to solve vs alpha
    ax = axes[0, 1]
    mask_train = df_analysis["train_solve_step"].notna()
    mask_val = df_analysis["val_solve_step"].notna()

    if mask_train.sum() > 0:
        ax.scatter(
            df_analysis[mask_train]["alpha"],
            df_analysis[mask_train]["train_solve_step"],
            s=100,
            alpha=0.6,
            label="Train",
        )
    if mask_val.sum() > 0:
        ax.scatter(
            df_analysis[mask_val]["alpha"],
            df_analysis[mask_val]["val_solve_step"],
            s=100,
            alpha=0.6,
            label="Validation",
        )

    ax.set_xlabel("Training Fraction (α)", fontsize=12)
    ax.set_ylabel("Steps to Solve (>95% acc)", fontsize=12)
    ax.set_title("Steps to Solve vs Alpha", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Final accuracies
    ax = axes[1, 0]
    ax.scatter(
        df_analysis["alpha"],
        df_analysis["final_train_acc"],
        s=100,
        alpha=0.6,
        label="Train",
    )
    ax.scatter(
        df_analysis["alpha"],
        df_analysis["final_val_acc"],
        s=100,
        alpha=0.6,
        label="Validation",
    )
    ax.set_xlabel("Training Fraction (α)", fontsize=12)
    ax.set_ylabel("Final Accuracy (%)", fontsize=12)
    ax.set_title("Final Accuracy vs Alpha", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Plot 4: Grokking detection summary
    ax = axes[1, 1]
    grokking_counts = df_analysis["grokking_detected"].value_counts()
    colors_pie = ["#ff7f0e", "#2ca02c"]
    ax.pie(
        grokking_counts.values,
        labels=["No Grokking", "Grokking Detected"],
        autopct="%1.1f%%",
        colors=colors_pie,
        startangle=90,
    )
    ax.set_title(
        f"Grokking Detection (p={modulus if modulus else 'various'})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved grokking analysis: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("GROKKING ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"{'Alpha':<10} {'Train Step':<15} {'Val Step':<15} {'Delay':<15} {'Grokking'}")
    print("-" * 80)

    for _, row in df_analysis.iterrows():
        train_step = (
            f"{row['train_solve_step']:.0f}"
            if pd.notna(row["train_solve_step"])
            else "N/A"
        )
        val_step = (
            f"{row['val_solve_step']:.0f}"
            if pd.notna(row["val_solve_step"])
            else "N/A"
        )
        delay = (
            f"{row['grokking_delay']:.0f}"
            if pd.notna(row["grokking_delay"])
            else "N/A"
        )
        grokking = "✓" if row["grokking_detected"] else "✗"

        print(
            f"{row['alpha']:<10.3f} {train_step:<15} {val_step:<15} {delay:<15} {grokking}"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize grokking phenomenon from experiment results"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Base directory containing experiment logs",
    )
    parser.add_argument(
        "--modulus",
        type=int,
        default=None,
        help="Filter experiments by modulus (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as log_dir)",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.log_dir, "plots")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("GROKKING VISUALIZATION")
    print("=" * 80)
    print(f"Log directory: {args.log_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.modulus:
        print(f"Filtering by modulus: {args.modulus}")
    print("=" * 80)

    # Find experiments
    print("\nSearching for experiments...")
    experiments = find_experiment_logs(args.log_dir, args.modulus)
    print(f"Found {len(experiments)} experiments")

    if len(experiments) == 0:
        print("No experiments found. Exiting.")
        return

    # Plot individual experiments
    print("\nGenerating individual plots...")
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Processing experiment:")
        print(f"  Alpha: {exp['hparams']['train_fraction']:.3f}")
        print(f"  Modulus: {exp['hparams']['modulus']}")

        df = load_metrics_from_csv(exp["csv_path"])
        if df is None:
            print("  Skipping (no metrics found)")
            continue

        # Create plot filename
        plot_name = (
            f"p{exp['hparams']['modulus']}_"
            f"alpha{exp['hparams']['train_fraction']:.3f}.png"
        )
        plot_path = os.path.join(args.output_dir, plot_name)

        plot_single_experiment(df, exp["hparams"], plot_path)

    # Create comparison plots
    print("\n\nGenerating comparison plots...")

    # Get unique modulus values
    moduli = set(exp["hparams"]["modulus"] for exp in experiments)

    for mod in sorted(moduli):
        print(f"\nCreating comparison for p={mod}...")

        # Alpha comparison
        comparison_path = os.path.join(
            args.output_dir, f"alpha_comparison_p{mod}.png"
        )
        plot_alpha_comparison(experiments, comparison_path, modulus=mod)

        # Grokking analysis
        analysis_path = os.path.join(
            args.output_dir, f"grokking_analysis_p{mod}.png"
        )
        plot_grokking_analysis(experiments, analysis_path, modulus=mod)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
