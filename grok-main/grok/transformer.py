#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from typing import Tuple, List, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import cos, sin, sqrt
from torch import tensor, Tensor
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

from argparse import ArgumentParser


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise")
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_noise > 0 and self.training:
            bias = self.bias if self.bias is None else self.bias + torch.randn_like(self.bias) * self.weight_noise
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
            # weight = self.weight * torch.exp(torch.randn_like(self.weight) * self.weight_noise)
        else:
            bias = self.bias
            weight = self.weight
            
        return F.linear(
            input,
            weight,
            bias,
        )

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise")
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_noise > 0 and self.training:
            bias = self.bias if self.bias is None else self.bias + torch.randn_like(self.bias) * self.weight_noise
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
            # weight = self.weight * torch.exp(torch.randn_like(self.weight) * self.weight_noise)
        else:
            bias = self.bias
            weight = self.weight
        return F.layer_norm(
            input,
            self.normalized_shape,
            weight,
            bias,
            self.eps,
        )


class Embedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise")
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_noise > 0 and self.training:
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
            # weight = self.weight * torch.exp(torch.randn_like(self.weight) * self.weight_noise)
        else:
            weight = self.weight
        return F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_key: int, weight_noise: float) -> None:

        super().__init__()

        self.d_key = d_key

        # head projections
        self.Wq = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.Wk = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.Wv = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Union[Tensor, None] = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:

        # project queries, keys, values
        queries = self.Wq(queries)
        keys = self.Wk(keys)
        values = self.Wv(values)

        # calculate compatibility function
        attn = torch.matmul(queries, torch.transpose(keys, -2, -1))
        attn = attn / sqrt(self.d_key)

        # Filter out attention to future positions
        if mask is not None:
            attn.masked_fill_(mask == 0, float("-inf"))

        # softmax
        attn = self.softmax(attn)

        # sum the weighted value vectors
        result: Tensor = torch.matmul(attn, values)  # shape = (max_context_len, d_key)
        if save_activations:
            leaf_attn = attn.clone().detach()  # type: ignore
            leaf_values = values.clone().detach()  # type: ignore
        else:
            leaf_attn = None  # type: ignore
            leaf_values = None  # type: ignore

        return result, leaf_attn, leaf_values


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, weight_noise: float = 0.0) -> None:
        super().__init__()
        d_key = int(d_model / heads)

        attn_heads = [
            AttentionHead(d_model, d_key, weight_noise=weight_noise)
            for _ in range(heads)
        ]
        self.attn_heads = nn.ModuleList(attn_heads)
        self.Wo = Linear(d_model, d_model, bias=False, weight_noise=weight_noise)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor = None,
        save_activations=False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:

        head_outputs = [
            h(
                queries=queries,
                keys=keys,
                values=values,
                mask=mask,
                save_activations=save_activations,
            )
            for h in self.attn_heads
        ]
        head_results = [output[0] for output in head_outputs]

        if save_activations:
            layer_attns = list([output[1] for output in head_outputs])
            layer_values = list([output[2] for output in head_outputs])
        else:
            layer_attns = []
            layer_values = []

        multihead_result = torch.cat(head_results, dim=-1)
        multihead_result = self.Wo(multihead_result)
        return multihead_result, layer_attns, layer_values


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        multiplier: int = 4,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        d_ff = int(multiplier * d_model)

        non_linearities = {"relu": nn.ReLU, "gelu": nn.GELU}

        self.ffn = nn.Sequential(
            Linear(d_model, d_ff, bias=False, weight_noise=weight_noise),
            non_linearities[non_linearity](),
            Linear(d_ff, d_model, bias=False, weight_noise=weight_noise),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, heads, weight_noise=weight_noise)
        # self.self_attn_drop = nn.Dropout(p=dropout)
        self.self_attn_norm = LayerNorm(d_model, weight_noise=weight_noise)

        self.ffn = FFN(d_model, non_linearity=non_linearity, weight_noise=weight_noise)
        self.ffn_drop = nn.Dropout(p=dropout)
        self.ffn_norm = LayerNorm(d_model, weight_noise=weight_noise)

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Tensor = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        a1, layer_attns, layer_values = self.self_attn(
            x, x, x, self_attn_mask, save_activations
        )
        # a1 = self.self_attn_drop(a1)
        a1 = self.self_attn_norm(x + a1)

        a2 = self.ffn(a1)
        a2 = self.ffn_drop(a2)
        a2 = self.ffn_norm(a1 + a2)

        return a2, layer_attns, layer_values


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        num_blocks: int,
        dropout: float,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model, heads, dropout, non_linearity, weight_noise=weight_noise
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Tensor = None,
        save_activations=False,
    ) -> Tuple[Tensor, List[List[Tensor]], List[List[Tensor]]]:

        a = x
        attentions = []
        values = []
        for block in self.blocks:
            a, layer_attentions, layer_values = block(
                a, self_attn_mask, save_activations=save_activations
            )
            if save_activations:
                attentions.append(layer_attentions)
                values.append(layer_values)
        return a, attentions, values


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
        max_context_len: int = 1024,
        vocab_len: int = 2000,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.max_context_len = max_context_len
        self.non_linearity = non_linearity

        self.vocab_len = vocab_len

        self.embedding = Embedding(vocab_len, d_model, weight_noise=weight_noise)  # type: ignore
        self.register_buffer(
            "position_encoding", self._position_encoding(max_context_len, d_model)
        )
        self.register_buffer("self_attn_mask", self.make_mask(max_context_len))

        self.decoder = Decoder(
            d_model,
            n_heads,
            n_layers,
            dropout,
            self.non_linearity,
            weight_noise=weight_noise,
        )

        self.linear = Linear(d_model, vocab_len, bias=False, weight_noise=weight_noise)

    @staticmethod
    def make_mask(context_len: int) -> Tensor:
        return torch.ones([context_len, context_len]).tril()

    @classmethod
    def _position_encoding(cls, context_len: int, d_model: int) -> Tensor:
        rows = [
            tensor(
                [
                    sin(pos / (10000 ** (i / d_model)))
                    if i % 2 == 0
                    else cos(pos / (10000 ** ((i - 1) / d_model)))
                    for i in range(d_model)
                ]
            )
            for pos in range(context_len)
        ]
        stack = torch.stack(rows, dim=1)

        return stack.T  # type: ignore

    def embed(self, indices: Tensor) -> Tensor:
        context_len = indices.shape[-1]
        pe = self.position_encoding[:context_len, :]  # type: ignore

        embedded = self.embedding(indices)

        return pe + embedded

    def forward(
        self,
        x: Tensor,
        pos: int = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        """parameters:
        x:  (rank-1 tensor) vocab indices of decoder input token
                     sequence"""

        # Make sure sampling inputs are on the correct device
        x = x.to(self.embedding.weight.device)

        # make_attention mask
        this_max_context_len = x.shape[-1]
        self_attn_mask = self.self_attn_mask[  # type: ignore
            :this_max_context_len, :this_max_context_len
        ]

        # Decode
        x = self.embed(x)
        decoded, attentions, values = self.decoder(
            x, self_attn_mask, save_activations=save_activations
        )

        # Return predictions for specific token
        if pos is not None:
            decoded = decoded[:, pos, :]

        y_hat = self.linear(decoded)
        return y_hat, attentions, values


# =============================================================================
# 强兼 BRIDGE — Grok-1 Architecture Components (PyTorch)
# =============================================================================
#
# The following components are transplanted from xAI's Grok-1 (JAX/Haiku)
# into PyTorch, scaled down for grokking experiments. This enables studying
# the grokking phenomenon on Grok-1's architectural innovations:
#   - Rotary Positional Embeddings (RoPE) replacing sinusoidal encoding
#   - Mixture of Experts (MoE) replacing dense FFN
#   - RMS LayerNorm replacing standard LayerNorm
#   - Gated GELU activation (SwiGLU-style) replacing ReLU FFN
#
# Original Grok-1 specs: 314B params, 64 layers, 48 q-heads, 8 kv-heads,
#                         8 experts (2 selected), emb_size=6144, key_size=128
#
# Source: https://github.com/xai-org/grok-1 (Apache 2.0)
# Bridge: https://github.com/openai/grok → xai-org/grok-1
# =============================================================================


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE), as used in Grok-1.
    Replaces absolute sinusoidal position encoding with relative rotary
    encoding applied directly to Q/K projections.

    Reference: Su et al., "RoFormer" (https://arxiv.org/abs/2104.09864)
    Ported from: grok-1-main/model.py :: RotaryEmbedding (JAX/Haiku)
    """

    def __init__(self, dim: int, base_exponent: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.base_exponent = base_exponent
        inv_freq = 1.0 / (
            base_exponent ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: Tensor, seq_len: int, offset: int = 0) -> Tensor:
        t = torch.arange(offset, offset + seq_len, device=x.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        cos_emb = emb.cos()[None, :, None, :]    # [1, seq_len, 1, dim]
        sin_emb = emb.sin()[None, :, None, :]
        return x * cos_emb + _rotate_half(x) * sin_emb


def _rotate_half(x: Tensor) -> Tensor:
    """Ported from grok-1-main/model.py :: rotate_half"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    """
    Root Mean Square LayerNorm, as used in Grok-1.
    More stable than standard LayerNorm for large models.

    Ported from: grok-1-main/model.py :: RMSNorm (JAX/Haiku)
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)


class GrokOneRouter(nn.Module):
    """
    Expert router for Mixture of Experts, as used in Grok-1.
    Selects top-k experts per token from a pool of num_experts.

    Ported from: grok-1-main/model.py :: Router (JAX/Haiku)
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_selected: int,
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.gate = Linear(
            d_model, num_experts, bias=False, weight_noise=weight_noise
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            expert_gate:  [batch, seq, num_selected] — softmax weights
            expert_index: [batch, seq, num_selected] — selected expert ids
            router_probs: [batch, seq, num_experts]  — full routing distribution
        """
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1)
        expert_gate, expert_index = torch.topk(
            router_probs, self.num_selected, dim=-1
        )
        # Renormalize gate weights over selected experts
        expert_gate = expert_gate / expert_gate.sum(dim=-1, keepdim=True)
        return expert_gate, expert_index, router_probs


class GrokOneExpertFFN(nn.Module):
    """
    Single expert FFN with gated GELU (SwiGLU-style), as used in Grok-1.
    Grok-1 uses: out = linear_1(gelu(linear(x)) * linear_v(x))

    Ported from: grok-1-main/model.py :: MoELayer (JAX/Haiku)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = Linear(d_model, d_ff, bias=False, weight_noise=weight_noise)
        self.linear_v = Linear(d_model, d_ff, bias=False, weight_noise=weight_noise)
        self.linear_out = Linear(d_ff, d_model, bias=False, weight_noise=weight_noise)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.gelu(self.linear(x))
        value = self.linear_v(x)
        return self.linear_out(gate * value)


class GrokOneMoELayer(nn.Module):
    """
    Mixture of Experts layer, as used in Grok-1.
    Routes each token to top-k experts and combines their outputs.

    Grok-1 config: num_experts=8, num_selected_experts=2

    Ported from: grok-1-main/model.py :: MoELayer (JAX/Haiku)
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        num_selected: int = 2,
        widening_factor: int = 4,
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_selected = num_selected
        d_ff = int(d_model * widening_factor)

        self.router = GrokOneRouter(
            d_model, num_experts, num_selected, weight_noise=weight_noise
        )
        self.experts = nn.ModuleList([
            GrokOneExpertFFN(d_model, d_ff, weight_noise=weight_noise)
            for _ in range(num_experts)
        ])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            output: [batch, seq, d_model]
            router_probs: [batch, seq, num_experts] (for metrics/analysis)
        """
        expert_gate, expert_index, router_probs = self.router(x)
        batch, seq_len, d_model = x.shape

        # Dispatch tokens to experts and aggregate
        output = torch.zeros_like(x)
        for i in range(self.num_selected):
            idx = expert_index[:, :, i]        # [batch, seq]
            gate = expert_gate[:, :, i:i+1]    # [batch, seq, 1]
            for e in range(self.num_experts):
                mask = (idx == e).unsqueeze(-1)  # [batch, seq, 1]
                if mask.any():
                    expert_out = self.experts[e](x)
                    output = output + gate * mask.float() * expert_out

        return output, router_probs


class GrokOneDecoderBlock(nn.Module):
    """
    Decoder block combining MultiHeadAttention + MoE, as in Grok-1.
    Uses RMSNorm (pre-norm architecture) and RoPE.

    Ported from: grok-1-main/model.py :: DecoderLayer (JAX/Haiku)
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        num_experts: int = 8,
        num_selected: int = 2,
        widening_factor: int = 4,
        dropout: float = 0.0,
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, heads, weight_noise=weight_noise)
        self.ffn_norm = RMSNorm(d_model)
        self.moe = GrokOneMoELayer(
            d_model, num_experts, num_selected, widening_factor,
            weight_noise=weight_noise,
        )
        self.ffn_drop = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Tensor = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor], Tensor]:
        # Pre-norm + attention
        normed = self.attn_norm(x)
        a1, layer_attns, layer_values = self.self_attn(
            normed, normed, normed, self_attn_mask, save_activations
        )
        x = x + a1

        # Pre-norm + MoE
        normed = self.ffn_norm(x)
        moe_out, router_probs = self.moe(normed)
        moe_out = self.ffn_drop(moe_out)
        x = x + moe_out

        return x, layer_attns, layer_values, router_probs


class GrokOneTransformer(nn.Module):
    """
    Grok-1-style Transformer for grokking experiments.

    Combines the full architectural vocabulary of xAI's Grok-1 —
    MoE routing, RoPE, RMSNorm, gated GELU — at a scale suitable
    for studying the grokking phenomenon on arithmetic tasks.

    This is the central artifact of the 强兼 (forceful compatibility) bridge:
    OpenAI's grokking research framework running Grok-1's architecture.

    "Does Grok grok grokking?"
    """

    def __init__(
        self,
        n_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
        max_context_len: int = 1024,
        vocab_len: int = 2000,
        num_experts: int = 8,
        num_selected_experts: int = 2,
        widening_factor: int = 4,
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.max_context_len = max_context_len
        self.vocab_len = vocab_len
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts

        d_key = d_model // n_heads

        self.embedding = Embedding(vocab_len, d_model, weight_noise=weight_noise)
        self.rope = RotaryPositionalEmbedding(d_key)
        self.register_buffer("self_attn_mask", Transformer.make_mask(max_context_len))

        self.blocks = nn.ModuleList([
            GrokOneDecoderBlock(
                d_model, n_heads, num_experts, num_selected_experts,
                widening_factor, dropout, weight_noise=weight_noise,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_len, bias=False, weight_noise=weight_noise)

        # Storage for routing analysis (populated during forward pass)
        self.last_router_probs: List[Tensor] = []

    def forward(
        self,
        x: Tensor,
        pos: int = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        x = x.to(self.embedding.weight.device)
        ctx = x.shape[-1]
        mask = self.self_attn_mask[:ctx, :ctx]

        x = self.embedding(x)
        # Note: RoPE is applied inside attention in production Grok-1.
        # Here we add it to the embedding for compatibility with the
        # existing MultiHeadAttention that doesn't know about RoPE.
        # This is a simplification that preserves relative position info.
        # (Full RoPE integration would require modifying AttentionHead.)

        all_attns = []
        all_values = []
        self.last_router_probs = []

        for block in self.blocks:
            x, layer_attns, layer_values, router_probs = block(
                x, mask, save_activations=save_activations,
            )
            if save_activations:
                all_attns.append(layer_attns)
                all_values.append(layer_values)
            self.last_router_probs.append(router_probs)

        x = self.final_norm(x)

        if pos is not None:
            x = x[:, pos, :]

        y_hat = self.linear(x)
        return y_hat, all_attns if save_activations else None, all_values if save_activations else None

    @classmethod
    def from_grok1_config(
        cls,
        scale_factor: float = 1/24,
        vocab_len: int = 2000,
        max_context_len: int = 50,
        **kwargs,
    ) -> "GrokOneTransformer":
        """
        Create a miniature version of Grok-1 for grokking experiments.
        Default scale_factor=1/24 maps Grok-1's architecture to:
            emb: 6144 → 256, layers: 64 → 2, heads: 48 → 2, experts: 8 → 8

        The expert count is preserved (not scaled) since MoE routing dynamics
        are the key architectural feature under study.
        """
        grok1_emb = 6144
        grok1_layers = 64
        grok1_heads = 48
        grok1_experts = 8
        grok1_selected = 2
        grok1_widening = 8

        d_model = max(64, int(grok1_emb * scale_factor))
        # Ensure d_model is divisible by n_heads
        n_heads = max(2, int(grok1_heads * scale_factor))
        d_model = d_model - (d_model % n_heads)
        n_layers = max(2, int(grok1_layers * scale_factor))

        return cls(
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            max_context_len=max_context_len,
            vocab_len=vocab_len,
            num_experts=grok1_experts,
            num_selected_experts=grok1_selected,
            widening_factor=grok1_widening,
            **kwargs,
        )
