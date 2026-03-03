import torch
import math
import copy
import torch.nn as nn
from typing import Callable

# References:
# https://github.com/nitarshan/robust-generalization-measures
# https://github.com/bneyshabur/generalization-bounds
# https://github.com/bneyshabur/over-parametrization


def compute_measure(
    model: nn.Module,
    init_model: nn.Module,
    measure_func: Callable,
    operator: str,
    kwargs: dict = {},
    p: int = 1,
) -> float:
    """
    Computes measure value for each layer given trained network and network at
    initialization.  Then aggregates values per layer using specified operator.

    :param model: trained network
    :param init_model: network at initialization
    :param measure_func: callable for the measure to compute
    :param operator: 'log_product', 'sum', 'max', 'product', or 'norm'
    :param p: p in L^p
    :return: value of the desired measure
    """

    measure_value = 0
    # weight_modules = ["Linear", "Embedding"]
    weight_modules = ["Linear"]

    if operator == "product":
        measure_value = math.exp(
            compute_measure(model, init_model, measure_func, "log_product", kwargs, p)
        )
    elif operator == "norm":
        measure_value = (
            compute_measure(model, init_model, measure_func, "sum", kwargs, p=p)
        ) ** (1 / p)
    else:
        measure_value = 0
        for child, init_child in zip(model.children(), init_model.children()):
            module_name = child._get_name()
            if module_name in weight_modules:
                if operator == "log_product":
                    measure_value += math.log(measure_func(child, init_child, **kwargs))
                elif operator == "sum":
                    measure_value += (measure_func(child, init_child, **kwargs)) ** p
                elif operator == "max":
                    measure_value = max(
                        measure_value, measure_func(child, init_child, **kwargs)
                    )
            else:
                measure_value += compute_measure(
                    child, init_child, measure_func, operator, kwargs, p=p
                )
    return measure_value


def norm(module, init_module, p=2, q=2):
    """
    Calculates l_pq norm of a parameter matrix
      l_p norm of incoming weights to each hidden unit
      l_q norm on the hidden units
    """
    return module.weight.view(module.weight.size(0), -1).norm(p=p, dim=1).norm(q).item()


def op_norm(module, init_module, p=float("Inf")):
    """
    Calculates l_p norm of eigenvalues of parameter matrix
    """
    _, S, _ = module.weight.view(module.weight.size(0), -1).svd()
    return S.norm(p).item()


def dist(module, init_module, p=2, q=2):
    """
    Calculates l_pq distance of the parameter matrix of a layer from the random
    initialization:
        l_p norm of incoming weights to each hidden unit
        l_q norm on the hidden units
    """
    return (
        (module.weight - init_module.weight)
        .view(module.weight.size(0), -1)
        .norm(p=p, dim=1)
        .norm(q)
        .item()
    )


def h_dist(module, init_module, p=2, q=2):
    """
    Calculate l_pq distance of parameters of trained network from random init
    Includes extra factor depending on number of hidden units
    """
    return (n_hidden(module, init_module) ** (1 - 1 / q)) * dist(
        module, init_module, p=p, q=q
    )


def h_dist_op_norm(module, init_module, p=2, q=2, p_op=float("Inf")):
    """
    Calculate ratio of h_dist to operator norm
    """
    return h_dist(module, init_module, p=p, q=q) / op_norm(module, init_module, p=p_op)


def n_hidden(module, init_module):
    """
    Number of hidden units
    """
    return module.weight.size(0)


def depth(module, init_module):
    """
    Depth (always == 1 for any linear layer)
    """
    return 1


def n_param(module, init_module):
    """
    Num parameters
    """
    bparam = 0 if module.bias is None else module.bias.size(0)
    return bparam + module.weight.size(0) * module.weight.view(
        module.weight.size(0), -1
    ).size(1)


def lp_path_norm(model, device, p=2, input_size=[3, 32, 32]):
    """
    Path norm (Neyshabur 2015)
    """

    tmp_model = copy.deepcopy(model)
    tmp_model.eval()
    for param in tmp_model.parameters():
        if param.requires_grad:
            param.abs_().pow_(p)
    data_ones = torch.ones(input_size).to(device)
    return (tmp_model(data_ones).sum() ** (1 / p)).item()


def calculate(trained_model, init_model, device, dataset_size, margin, input_dim):
    """
    Calculates various measures given trained model and model at init
    Computes:
        measures: norm based measures on the model
        bounds: generalization bounds on the model
    """

    model = copy.deepcopy(trained_model)

    # depth
    d = compute_measure(model, init_model, depth, "sum", {})

    # number of parameters (not including batch norm)
    nparam = compute_measure(model, init_model, n_param, "sum", {})

    measure, bound = {}, {}
    with torch.no_grad():

        # Compute measures
        measure["L_{1,inf} norm"] = (
            compute_measure(
                model, init_model, norm, "product", {"p": 1, "q": float("Inf")}
            )
            / margin
        )
        measure["Frobenius norm"] = (
            compute_measure(model, init_model, norm, "product", {"p": 2, "q": 2})
            / margin
        )
        measure["L_{3,1.5} norm"] = (
            compute_measure(model, init_model, norm, "product", {"p": 3, "q": 1.5})
            / margin
        )
        measure["Spectral norm"] = (
            compute_measure(model, init_model, op_norm, "product", {"p": float("Inf")})
            / margin
        )
        measure["L_1.5 operator norm"] = (
            compute_measure(model, init_model, op_norm, "product", {"p": 1.5}) / margin
        )
        measure["Trace norm"] = (
            compute_measure(model, init_model, op_norm, "product", {"p": 1}) / margin
        )

        # input_size = [context_len, emb_dim]
        # measure["L1_path norm"] = (
        #     lp_path_norm(
        #         model, device, p=1, input_size=input_size
        #     )
        #     / margin
        # )
        # measure["L1.5_path norm"] = (
        #     lp_path_norm(
        #         model, device, p=1.5, input_size=input_size
        #     )
        #     / margin
        # )
        # measure["L2_path norm"] = (
        #     lp_path_norm(
        #         model, device, p=2, input_size=input_size
        #     )
        #     / margin
        # )

        # Compute generalization bounds without constant or additive logarithmic factors

        # Golowich 2018
        # https://arxiv.org/pdf/1712.06541.pdf
        alpha = math.sqrt(d + math.log(1 * input_dim * input_dim))

        # Bartlett Mendelson 2002
        bound["L1_max Bound"] = (
            alpha * measure["L_{1,inf} norm"] / math.sqrt(dataset_size)
        )

        # Neyshabur 2015
        bound["Frobenius Bound"] = (
            alpha * measure["Frobenius norm"] / math.sqrt(dataset_size)
        )

        # Neyshabur 2015
        bound["L_{3,1.5} Bound"] = (
            alpha * measure["L_{3,1.5} norm"] / (dataset_size ** (1 / 3))
        )

        beta = math.log(dataset_size) * math.log(nparam)
        ratio = compute_measure(
            model,
            init_model,
            h_dist_op_norm,
            "norm",
            {"p": 2, "q": 1, "p_op": float("Inf")},
            p=2 / 3,
        )

        # Spectral L_{2, 1} Bound
        # Bartlett 2017
        bound["Spec_L_{2,1} Bound"] = (
            beta * measure["Spectral norm"] * ratio / math.sqrt(dataset_size)
        )

        ratio = compute_measure(
            model,
            init_model,
            h_dist_op_norm,
            "norm",
            {"p": 2, "q": 2, "p_op": float("Inf")},
            p=2,
        )

        # Spectral Frobenius
        # Neyshabur 2018
        # https://arxiv.org/pdf/1706.08947.pdf
        bound["Spec_Fro Bound"] = (
            d * measure["Spectral norm"] * ratio / math.sqrt(dataset_size)
        )

    return measure, bound


# =============================================================================
# 强兼 BRIDGE — Mixture of Experts Grokking Metrics
# =============================================================================
#
# Additional metrics for studying the grokking phenomenon in MoE architectures.
# These capture dynamics specific to Grok-1's expert routing that standard
# dense transformer metrics miss.
#
# Key insight: MoE models may "grok" differently — expert specialization
# could correlate with the phase transition from memorization to generalization.
# =============================================================================


def expert_utilization_entropy(router_probs: torch.Tensor) -> float:
    """
    Shannon entropy of the expert selection distribution.
    Higher entropy = more uniform expert utilization = better load balance.

    In the context of grokking: if entropy increases during the phase
    transition, it suggests generalization requires distributing computation
    more evenly across experts (moving away from memorization shortcuts).

    :param router_probs: [batch, seq_len, num_experts] probability tensor
    :returns: scalar entropy value
    """
    mean_probs = router_probs.mean(dim=(0, 1))  # [num_experts]
    entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum().item()
    return entropy


def expert_specialization_score(router_probs: torch.Tensor) -> float:
    """
    Measures how specialized individual experts are (vs. uniform).
    Score of 0 = all experts identical, 1 = maximum specialization.

    In grokking: high specialization during training plateau (memorization)
    that drops during the grokking phase transition would indicate that
    memorization relies on expert shortcuts while generalization doesn't.

    :param router_probs: [batch, seq_len, num_experts] probability tensor
    :returns: specialization score in [0, 1]
    """
    mean_probs = router_probs.mean(dim=(0, 1))  # [num_experts]
    num_experts = mean_probs.shape[0]
    uniform = 1.0 / num_experts
    max_deviation = 1.0 - uniform
    actual_deviation = (mean_probs - uniform).abs().max().item()
    return actual_deviation / max_deviation if max_deviation > 0 else 0.0


def routing_collapse_index(router_probs: torch.Tensor) -> float:
    """
    Detects expert routing collapse — when the model routes most tokens
    to a single expert, wasting MoE capacity.

    Defined as: max(expert_load) / mean(expert_load)
    Collapse threshold: > 2.0 indicates significant imbalance
    Perfect balance: 1.0

    :param router_probs: [batch, seq_len, num_experts] probability tensor
    :returns: collapse index (1.0 = balanced, higher = more collapsed)
    """
    mean_probs = router_probs.mean(dim=(0, 1))
    return (mean_probs.max() / mean_probs.mean()).item()


def compute_moe_grokking_metrics(model: nn.Module, init_model: nn.Module) -> dict:
    """
    Computes all MoE-specific grokking metrics for a GrokOneTransformer.
    Call after a forward pass to analyze the cached router probabilities.

    :param model: trained GrokOneTransformer
    :param init_model: GrokOneTransformer at initialization
    :returns: dict of metric name → value
    """
    from grok.transformer import GrokOneTransformer

    moe_metrics = {}
    if isinstance(model, GrokOneTransformer) and model.last_router_probs:
        for layer_idx, rp in enumerate(model.last_router_probs):
            if rp is not None:
                prefix = f"layer_{layer_idx}"
                moe_metrics[f"{prefix}/routing_entropy"] = expert_utilization_entropy(rp)
                moe_metrics[f"{prefix}/specialization"] = expert_specialization_score(rp)
                moe_metrics[f"{prefix}/collapse_index"] = routing_collapse_index(rp)

        # Aggregate across layers
        all_rp = torch.stack(model.last_router_probs)
        moe_metrics["avg_routing_entropy"] = expert_utilization_entropy(
            all_rp.mean(dim=0)
        )
        moe_metrics["avg_specialization"] = expert_specialization_score(
            all_rp.mean(dim=0)
        )
        moe_metrics["avg_collapse_index"] = routing_collapse_index(
            all_rp.mean(dim=0)
        )

    return moe_metrics
