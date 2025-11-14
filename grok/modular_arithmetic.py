#!/usr/bin/env python
"""
Modular Arithmetic Data Generation for Grokking Experiments

This module provides specialized data generation for modular addition:
(x, y) -> (x + y) mod p, where x, y ∈ Z_p and p is a prime number.
"""

import itertools
import numpy as np
import torch
from torch import Tensor, LongTensor
from typing import List, Dict, Tuple, Optional
import sympy


EOS_TOKEN = "<|eos|>"
EQ_TOKEN = "="


def is_prime(n: int) -> bool:
    """Check if a number is prime using sympy."""
    return sympy.isprime(n)


def get_next_prime(n: int) -> int:
    """Get the next prime number after n."""
    return int(sympy.nextprime(n))


class ModularAdditionTokenizer:
    """Tokenizer specifically for modular addition tasks."""

    def __init__(self, modulus: int) -> None:
        """
        Initialize tokenizer for modular addition.

        :param modulus: The modulus p for the operation (x + y) mod p
        """
        self.modulus = modulus
        self.itos = self._build_vocabulary()
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}

    def _build_vocabulary(self) -> List[str]:
        """
        Build vocabulary for modular addition.

        Vocabulary consists of:
        - <|eos|>: End of sequence token
        - =: Equality token
        - +: Addition operator
        - 0 to modulus-1: All possible numbers in Z_p
        """
        tokens = [EOS_TOKEN, EQ_TOKEN, "+"]
        tokens += [str(i) for i in range(self.modulus)]
        return tokens

    def encode(self, text: str) -> Tensor:
        """
        Encode a text string to tensor of token IDs.

        :param text: String to encode (e.g., "<|eos|> 5 + 3 = 8 <|eos|>")
        :returns: Tensor of token IDs
        """
        tokens = text.split(" ")
        return LongTensor([self.stoi[t] for t in tokens])

    def encode_batch(self, texts: List[str]) -> Tensor:
        """
        Encode a batch of text strings.

        :param texts: List of strings to encode
        :returns: 2D tensor of token IDs (batch_size x seq_len)
        """
        return torch.stack([self.encode(t) for t in texts], dim=0)

    def decode(self, tensor: Tensor) -> str:
        """
        Decode a tensor of token IDs back to text.

        :param tensor: Tensor of token IDs
        :returns: Decoded string
        """
        tokens = [self.itos[i.item()] for i in tensor]
        return " ".join(tokens)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.itos)


class ModularAdditionDataset:
    """
    Dataset for modular addition: (x + y) mod p

    This generates all possible equations for x, y in Z_p.
    Total dataset size is p^2.
    """

    def __init__(
        self,
        modulus: int,
        train_fraction: float = 0.5,
        seed: int = 0
    ) -> None:
        """
        Initialize modular addition dataset.

        :param modulus: Prime number p for modular arithmetic
        :param train_fraction: Fraction of data to use for training (alpha)
        :param seed: Random seed for reproducibility
        """
        if not is_prime(modulus):
            raise ValueError(f"Modulus {modulus} must be a prime number")

        self.modulus = modulus
        self.train_fraction = train_fraction
        self.seed = seed
        self.tokenizer = ModularAdditionTokenizer(modulus)

        # Generate all equations
        self.all_equations = self._generate_all_equations()

        # Split into train and validation
        self.train_data, self.val_data = self._split_data()

    def _generate_all_equations(self) -> List[str]:
        """
        Generate all possible modular addition equations.

        For modulus p, generates p^2 equations:
        x + y = (x + y) mod p for all x, y in [0, p-1]

        :returns: List of equation strings
        """
        equations = []

        # Generate all pairs (x, y) where x, y ∈ Z_p
        for x, y in itertools.product(range(self.modulus), repeat=2):
            result = (x + y) % self.modulus
            eq = f"{x} + {y} = {result}"
            # Wrap with EOS tokens
            eq_with_eos = f"{EOS_TOKEN} {eq} {EOS_TOKEN}"
            equations.append(eq_with_eos)

        return equations

    def _split_data(self) -> Tuple[Tensor, Tensor]:
        """
        Split data into training and validation sets.

        :returns: Tuple of (train_tensor, val_tensor)
        """
        # Shuffle equations with fixed seed for reproducibility
        rng = np.random.RandomState(seed=self.seed)
        indices = np.arange(len(self.all_equations))
        rng.shuffle(indices)

        # Split based on train_fraction
        train_size = int(len(self.all_equations) * self.train_fraction)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Get equations for each split
        train_equations = [self.all_equations[i] for i in train_indices]
        val_equations = [self.all_equations[i] for i in val_indices]

        # Encode to tensors
        train_tensor = self.tokenizer.encode_batch(train_equations)
        val_tensor = self.tokenizer.encode_batch(val_equations)

        return train_tensor, val_tensor

    def get_train_dataset(self) -> Tensor:
        """Return training data tensor."""
        return self.train_data

    def get_val_dataset(self) -> Tensor:
        """Return validation data tensor."""
        return self.val_data

    def get_dataset_info(self) -> Dict:
        """
        Return information about the dataset.

        :returns: Dictionary with dataset statistics
        """
        return {
            "modulus": self.modulus,
            "train_fraction": self.train_fraction,
            "total_size": len(self.all_equations),
            "train_size": len(self.train_data),
            "val_size": len(self.val_data),
            "vocab_size": len(self.tokenizer),
            "sequence_length": self.train_data.shape[1] if len(self.train_data) > 0 else 0,
        }


class ModularAdditionIterator:
    """Iterator for batching modular addition data."""

    def __init__(
        self,
        data: Tensor,
        batch_size: int,
        device: torch.device,
        shuffle: bool = True
    ) -> None:
        """
        Initialize iterator.

        :param data: Data tensor (num_examples x seq_len)
        :param batch_size: Batch size
        :param device: Device to place batches on
        :param shuffle: Whether to shuffle data each epoch
        """
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.num_batches = (len(data) + batch_size - 1) // batch_size
        self.reset()

    def reset(self):
        """Reset iterator for new epoch."""
        self.current_idx = 0
        if self.shuffle:
            self.permutation = torch.randperm(len(self.data))
        else:
            self.permutation = torch.arange(len(self.data))

    def __iter__(self):
        """Return iterator."""
        return self

    def __next__(self) -> Dict[str, Tensor]:
        """
        Get next batch.

        :returns: Dictionary with 'text' (input) and 'target' (shifted output)
        """
        if self.current_idx >= len(self.data):
            self.reset()
            raise StopIteration

        # Get batch indices
        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, len(self.data))
        batch_indices = self.permutation[start_idx:end_idx]

        # Get batch data
        batch = self.data[batch_indices]

        # Create input (all but last token) and target (all but first token)
        text = batch[:, :-1].to(self.device)
        target = batch[:, 1:].to(self.device)

        self.current_idx = end_idx

        return {"text": text, "target": target}

    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches


def create_modular_addition_datasets(
    modulus: int,
    train_fraction: float,
    seed: int = 0
) -> Tuple[ModularAdditionDataset, Dict]:
    """
    Convenience function to create modular addition datasets.

    :param modulus: Prime modulus p
    :param train_fraction: Fraction of data for training (alpha)
    :param seed: Random seed
    :returns: Tuple of (dataset, info_dict)
    """
    dataset = ModularAdditionDataset(
        modulus=modulus,
        train_fraction=train_fraction,
        seed=seed
    )

    info = dataset.get_dataset_info()

    return dataset, info


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Modular Addition Dataset Example")
    print("=" * 80)

    # Create dataset with p=97, alpha=0.3 (30% training data)
    modulus = 97
    train_fraction = 0.3

    print(f"\nCreating dataset with:")
    print(f"  - Modulus (p): {modulus}")
    print(f"  - Train fraction (α): {train_fraction}")

    dataset, info = create_modular_addition_datasets(
        modulus=modulus,
        train_fraction=train_fraction,
        seed=42
    )

    print(f"\nDataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print(f"\nExample equations:")
    train_data = dataset.get_train_dataset()
    for i in range(min(5, len(train_data))):
        equation = dataset.tokenizer.decode(train_data[i])
        print(f"  {equation}")

    print(f"\n" + "=" * 80)
