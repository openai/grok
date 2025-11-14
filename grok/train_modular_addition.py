#!/usr/bin/env python
"""
Training script for modular addition grokking experiments.

This script trains a transformer model on the modular addition task:
(x, y) -> (x + y) mod p

Key parameters:
- p: Prime modulus (larger p = harder problem)
- α (alpha): Training data fraction = train_size / p²
"""

import argparse
import json
import math
import os
import time
from typing import Dict, Any, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger

from grok.modular_arithmetic import (
    ModularAdditionDataset,
    ModularAdditionIterator,
)
from grok.transformer import Transformer


class ModularAdditionTransformer(LightningModule):
    """Transformer model for modular addition with training logic."""

    def __init__(self, hparams: argparse.Namespace) -> None:
        """
        Initialize the model.

        :param hparams: Hyperparameters from command line
        """
        super().__init__()
        self.save_hyperparameters(hparams)

        # Create dataset
        self.dataset = ModularAdditionDataset(
            modulus=hparams.modulus,
            train_fraction=hparams.train_fraction,
            seed=hparams.seed,
        )

        self.dataset_info = self.dataset.get_dataset_info()
        vocab_size = self.dataset_info["vocab_size"]

        # Create transformer model
        self.transformer = Transformer(
            num_layers=hparams.num_layers,
            num_heads=hparams.num_heads,
            d_model=hparams.d_model,
            dropout=hparams.dropout,
            max_len=hparams.max_len,
            vocab_size=vocab_size,
            non_linearity=hparams.non_linearity,
            weight_noise=hparams.weight_noise,
        )

        # Training state
        self.train_batch_size = hparams.batch_size
        self.val_batch_size = hparams.val_batch_size

        # Logging control
        self.next_train_log_step = 0
        self.next_val_log_step = 0

        print("\n" + "=" * 80)
        print("Model Initialized")
        print("=" * 80)
        print(f"Dataset Info:")
        for key, value in self.dataset_info.items():
            print(f"  {key}: {value}")
        print(f"Model Parameters:")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters: {total_params:,}")
        print("=" * 80 + "\n")

    def forward(self, x: Tensor, **kwargs) -> Tuple:
        """Forward pass through transformer."""
        return self.transformer(x, **kwargs)

    def configure_optimizers(self):
        """Configure AdamW optimizer with learning rate schedule."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=self.hparams.weight_decay,
        )

        # Learning rate schedule with warmup
        def lr_schedule(step):
            warmup_steps = self.hparams.warmup_steps

            if step < warmup_steps:
                # Linear warmup
                return float(step) / max(float(warmup_steps), 1.0)
            else:
                # Constant after warmup (or could add decay here)
                return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        """Create training data loader."""
        train_data = self.dataset.get_train_dataset()
        device = next(self.parameters()).device

        iterator = ModularAdditionIterator(
            data=train_data,
            batch_size=self.train_batch_size,
            device=device,
            shuffle=True,
        )

        return iterator

    def val_dataloader(self):
        """Create validation data loader."""
        val_data = self.dataset.get_val_dataset()
        device = next(self.parameters()).device

        iterator = ModularAdditionIterator(
            data=val_data,
            batch_size=self.val_batch_size,
            device=device,
            shuffle=False,
        )

        return iterator

    def _compute_loss_and_accuracy(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute loss and accuracy for a batch.

        Loss is computed only on the right-hand side of the equation (after =).

        :param batch: Batch with 'text' and 'target' keys
        :returns: Tuple of (loss, accuracy)
        """
        x = batch["text"]  # (batch_size, seq_len)
        y = batch["target"]  # (batch_size, seq_len)

        # Forward pass
        y_hat, _, _ = self(x)  # (batch_size, seq_len, vocab_size)
        y_hat = y_hat.transpose(-2, -1)  # (batch_size, vocab_size, seq_len)

        # Find position of '=' token
        eq_token_index = self.dataset.tokenizer.stoi["="]
        eq_position = (y[0, :] == eq_token_index).nonzero(as_tuple=True)[0].item()

        # Only compute loss on right-hand side (after '=')
        y_rhs = y[:, eq_position + 1 :]  # (batch_size, rhs_len)
        y_hat_rhs = y_hat[:, :, eq_position + 1 :]  # (batch_size, vocab_size, rhs_len)

        # Compute cross-entropy loss
        loss = F.cross_entropy(y_hat_rhs, y_rhs, reduction="mean")

        # Compute accuracy: all tokens must match
        with torch.no_grad():
            # Get predictions
            predictions = torch.argmax(y_hat_rhs, dim=1)  # (batch_size, rhs_len)
            # Check if all tokens match
            correct = (predictions == y_rhs).all(dim=1)  # (batch_size,)
            accuracy = correct.float().mean() * 100.0  # Convert to percentage

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, accuracy = self._compute_loss_and_accuracy(batch)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, accuracy = self._compute_loss_and_accuracy(batch)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)

        return {"val_loss": loss, "val_acc": accuracy}

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Compute accuracy on full training set
        train_data = self.dataset.get_train_dataset()
        device = next(self.parameters()).device

        with torch.no_grad():
            batch_size = min(512, len(train_data))
            total_correct = 0
            total_samples = 0

            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i : i + batch_size]
                text = batch_data[:, :-1].to(device)
                target = batch_data[:, 1:].to(device)

                batch = {"text": text, "target": target}
                _, accuracy = self._compute_loss_and_accuracy(batch)

                total_correct += accuracy.item() * len(batch_data) / 100.0
                total_samples += len(batch_data)

            full_train_acc = (total_correct / total_samples) * 100.0
            self.log("full_train_acc", full_train_acc)


class MetricsCallback(Callback):
    """Callback to log additional metrics during training."""

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log additional metrics at end of validation."""
        # Get current learning rate
        current_lr = trainer.optimizers[0].param_groups[0]["lr"]
        pl_module.log("learning_rate", current_lr)

        # Log epoch and step
        pl_module.log("epoch", float(trainer.current_epoch))
        pl_module.log("global_step", float(trainer.global_step))


def add_args() -> argparse.ArgumentParser:
    """Add command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train transformer on modular addition"
    )

    # Data parameters
    parser.add_argument(
        "--modulus",
        type=int,
        default=97,
        help="Prime modulus p for modular addition",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.3,
        help="Fraction of data for training (alpha), between 0 and 1",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )

    # Model parameters
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--d_model", type=int, default=128, help="Model dimension"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--max_len", type=int, default=50, help="Maximum sequence length"
    )
    parser.add_argument(
        "--non_linearity",
        type=str,
        default="relu",
        choices=["relu", "gelu"],
        help="Non-linearity function",
    )
    parser.add_argument(
        "--weight_noise", type=float, default=0.0, help="Weight noise factor"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Training batch size"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=512, help="Validation batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1.0, help="Weight decay (L2 regularization)"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=50, help="Number of warmup steps"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--max_steps", type=int, default=50000, help="Maximum number of training steps"
    )

    # Logging parameters
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for logs",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name for logging",
    )

    # Hardware parameters
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU device ID (-1 for CPU)"
    )

    return parser


def train(hparams: argparse.Namespace) -> str:
    """
    Main training function.

    :param hparams: Hyperparameters
    :returns: Path to log directory
    """
    # Set random seeds for reproducibility
    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create experiment name if not provided
    if hparams.experiment_name is None:
        hparams.experiment_name = (
            f"p{hparams.modulus}_alpha{hparams.train_fraction:.3f}_"
            f"layers{hparams.num_layers}_heads{hparams.num_heads}_"
            f"d{hparams.d_model}_seed{hparams.seed}"
        )

    # Setup logging
    log_dir = os.path.join(hparams.log_dir, hparams.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    # Save hyperparameters
    with open(os.path.join(log_dir, "hparams.json"), "w") as f:
        json.dump(vars(hparams), f, indent=2)

    print(f"\nExperiment: {hparams.experiment_name}")
    print(f"Log directory: {log_dir}\n")

    # Create model
    model = ModularAdditionTransformer(hparams)

    # Create logger
    logger = CSVLogger(save_dir=hparams.log_dir, name=hparams.experiment_name)

    # Create callbacks
    callbacks = [MetricsCallback()]

    # Create trainer
    trainer_args = {
        "max_epochs": hparams.max_epochs if hparams.max_epochs else 10000000,
        "max_steps": hparams.max_steps,
        "val_check_interval": 0.25,  # Validate 4 times per epoch
        "logger": logger,
        "callbacks": callbacks,
        "enable_progress_bar": True,
        "log_every_n_steps": 10,
    }

    # Add GPU support
    if torch.cuda.is_available() and hparams.gpu >= 0:
        trainer_args["accelerator"] = "gpu"
        trainer_args["devices"] = [hparams.gpu]
    else:
        trainer_args["accelerator"] = "cpu"

    trainer = Trainer(**trainer_args)

    # Train
    print("\nStarting training...\n")
    start_time = time.time()
    trainer.fit(model)
    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Logs saved to: {log_dir}\n")

    return log_dir


if __name__ == "__main__":
    parser = add_args()
    args = parser.parse_args()

    # Validate arguments
    assert 0 < args.train_fraction < 1, "train_fraction must be between 0 and 1"
    assert args.modulus > 1, "modulus must be greater than 1"

    log_dir = train(args)
    print(f"\nExperiment complete! Results saved to: {log_dir}")
