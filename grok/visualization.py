import csv
import logging
import os
import math
import socket

from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch

from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from grok.data import ArithmeticDataset

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("grok.view_metrics")
logger.setLevel(logging.ERROR)

GROK_DIR = os.path.expanduser("~/data/grok")
IMAGE_DIR = f"{GROK_DIR}/images"
DATA_DIR = f"{GROK_DIR}/data"


DEFAULT_CMAP = "viridis"

default_metric_limits = {
    "min_val_accuracy": 0,
    "max_val_accuracy": 100,
    "min_T": 0,  # 0
    "max_T": 100,  # 87.5
    "min_D": 0,  # 8
    "max_D": 2048,  # 256
    "min_H": 0,  # 1
    "max_H": 1204,  # 8
    "min_L": 0,  # 1
    "max_L": 1024,  # 4
    "min_accuracy": 0,
    "max_accuracy": 100,
}

default_axis_scales = {"x": "linear", "y": "linear"}


## Data Loading Functions


def factor_expts(expts):
    result = {}
    for expt in expts:
        expt_s = expt.split("_")
        arch = "_".join(expt_s[:3])
        t = int(float(expt_s[3].split("-")[1]))
        result.setdefault(arch, {})
        result[arch][t] = expt
    return result


def load_metric_data(data_dir, epochs=100000, load_partial_data=True):
    # layers x heads x d_model x train_pct
    data = {}
    expts = os.listdir(data_dir)
    archs = factor_expts(expts)
    logger.debug(archs)
    for arch in archs:
        T = sorted(archs[arch].keys())
        data[arch] = {
            "T": torch.LongTensor(T),
            "metrics": torch.zeros((max(T), 5, epochs)),
        }
        # print(f"metrics_shape = {data[arch]['metrics'].shape}")
        for i, t in tqdm(list(enumerate(T))):
            expt = archs[arch][t]
            logger.debug(expt)
            log_dir = data_dir + "/" + expt

            # print("log_dir", log_dir)
            try:
                with open(log_dir + "/default/version_0/metrics.csv", "r") as fh:
                    logger.debug(f"loading {log_dir}")
                    reader = list(csv.DictReader(fh))
                    val_t = torch.FloatTensor(
                        [
                            [
                                float(r["val_loss"]),
                                float(r["val_accuracy"]),
                            ]
                            for r in reader
                            if r["val_loss"]
                        ]
                    ).T
                    train_t = torch.FloatTensor(
                        [
                            [
                                float(r["learning_rate"]),
                                float(r["train_loss"]),
                                float(r["train_accuracy"]),
                            ]
                            for r in reader
                            if r["train_loss"]
                        ]
                    ).T
                    # logger.debug(val_t.shape)
                    # logger.debug(train_t[0, -3:])
                    if load_partial_data:
                        raise Exception("Not implemented")
                    elif (val_t.shape[-1] >= epochs) and (train_t.shape[-1] >= epochs):
                        data[arch]["metrics"][i] = torch.cat(
                            [train_t[..., :epochs], val_t[..., :epochs]], dim=0
                        )
                    else:
                        data[arch]["T"][i] = 0
            # except FileNotFoundError:
            except:
                data[arch]["T"][i] = 0
        indices = torch.nonzero(data[arch]["T"]).squeeze()
        if len(indices.shape) == 0:
            indices = indices.unsqueeze(0)
        # print(f"indices.shape = {indices.shape}")
        data[arch]["T"] = data[arch]["T"][indices]
        # print(f"data[arch]['T'].shape = {data[arch]['T'].shape}")
        data[arch]["metrics"] = data[arch]["metrics"][indices]
        # print(f"data[arch]['metrics'].shape = {data[arch]['metrics'].shape}")
        data[arch]["metrics"] = torch.transpose(data[arch]["metrics"], 0, 1)
        # print(f"data[arch]['metrics'].shape = {data[arch]['metrics'].shape}")
    return data


def most_interesting(metric_data):
    interesting_metric_data = {}
    for arch in metric_data:
        T = metric_data[arch]["T"]
        max_acc_by_t = torch.max(
            metric_data[arch]["val_accuracy"], dim=1, keepdim=True
        ).values.squeeze()
        max_loss_by_t = torch.max(
            metric_data[arch]["val_loss"], dim=1, keepdim=True
        ).values.squeeze()
        acc_idx = torch.nonzero(max_acc_by_t >= 95).squeeze()
        if acc_idx.shape == torch.Size([0]):
            acc_idx = torch.nonzero(max_acc_by_t == max_acc_by_t.max()).squeeze()
        if acc_idx.shape == torch.Size([]):
            acc_idx = acc_idx.unsqueeze(0)
        max_loss = torch.max(max_loss_by_t[acc_idx])
        loss_idx = torch.nonzero(max_loss_by_t[acc_idx] == max_loss)
        interesting_idx = acc_idx[loss_idx].squeeze()

        interesting_metric_data[arch] = {}
        for k in metric_data[arch]:
            interesting_metric_data[arch][k] = metric_data[arch][k][
                interesting_idx
            ].unsqueeze(0)

        return interesting_metric_data


# ## Graph Drawing Functions


def moving_avg(Y, steps):
    return np.convolve(Y, np.ones(steps), "valid") / steps


def find_inflections(Y, smoothing_steps=100):
    avg_Y = moving_avg(Y, smoothing_steps)
    avg_direction = torch.FloatTensor(np.sign(avg_Y[1:] - avg_Y[:-1]))
    avg_direction = torch.cat([avg_direction[0].unsqueeze(0), avg_direction])
    avg_inflections = torch.nonzero(avg_direction[1:] - avg_direction[:-1]).squeeze()
    avg_inflections = [0] + (avg_inflections + 1).tolist() + [len(Y) - 1]
    logger.debug(f"avg_inflections = {avg_inflections}")
    inflections = []
    for i in range(2, len(avg_inflections)):
        low = avg_inflections[i - 2]
        high = avg_inflections[i]
        logger.debug(f"low={low}")
        logger.debug(f"high={high}")
        if avg_direction[low + 1] < 0:
            indices = Y[low:high].argmin() + low
            logger.debug(f"min = (Y[{indices}] = {Y[int(indices)]}")
        else:
            indices = Y[low:high].argmax() + low
            logger.debug(f"max = (Y[{indices}] = {Y[int(indices)]}")
        inflections.append(indices)
    return torch.LongTensor(inflections)


def check_limits(arch_name, limits):
    L, H, D = [float(v.split("-")[1]) for v in arch_name.split("_")]
    if (L > limits["max_L"]) or (L < limits["min_L"]):
        return False
    if (H > limits["max_H"]) or (H < limits["min_H"]):
        return False
    if (D > limits["max_D"]) or (D < limits["min_D"]):
        return False
    # if (T > limits['max_T']) or (T < limits['min_T']):
    #    return False
    return True


def filter_archs(data, limits={}):
    my_limits = deepcopy(default_metric_limits)
    my_limits.update(limits)
    limits = my_limits
    archs = sorted(list(set([a for a in data.keys() if check_limits(a, limits)])))
    logger.debug(f"archs = {archs}")
    return archs


def get_metric_data(data, limits={}):
    my_limits = deepcopy(default_metric_limits)
    my_limits.update(limits)
    limits = my_limits

    for k in limits.keys():
        metric = k.replace("min_", "").replace("max_", "")
        assert (
            limits["max_" + metric] >= limits["min_" + metric]
        ), f"invalid {metric} limits"

    d = {}
    for arch in filter_archs(data, limits):
        logger.debug(arch)
        indices = torch.nonzero(
            torch.logical_and(
                data[arch]["T"] >= limits["min_T"], data[arch]["T"] <= limits["max_T"]
            )
        ).squeeze(dim=-1)
        logger.debug(f"indices={indices}")
        learning_rate, train_loss, train_accuracy, val_loss, val_accuracy = data[arch][
            "metrics"
        ][:, indices, :]
        d[arch] = {
            "T": data[arch]["T"][indices],
            "learning_rate": data[arch]["metrics"][0, indices, :],
            "train_loss": data[arch]["metrics"][1, indices, :],
            "train_accuracy": data[arch]["metrics"][2, indices, :],
            "val_loss": data[arch]["metrics"][3, indices, :],
            "val_accuracy": data[arch]["metrics"][4, indices, :],
        }
    return d


def add_metric_graph(
    fig,
    ax,
    metric,
    metric_data,
    scales=default_axis_scales,
    cmap=DEFAULT_CMAP,
    inflection_hline=False,
    ds_len=None,
    batchsize=97,
):
    ax.set_title(metric)
    ax.set_xscale(scales["x"])
    ax.set_yscale(scales["y"])
    if ds_len is None:
        ax.set_xlabel("epochs")
    else:
        ax.set_xlabel("updates")

    # if 'loss' in metric:
    #    ymin=0
    #    ax.axis(ymin=ymin)
    if "accuracy" in metric:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ymin = 1e-16
        ymax = 101
        ax.axis(ymin=ymin, ymax=ymax)
    if "loss" in metric:
        ymin = 1e-16
        ymax = 15
        ax.axis(ymin=ymin, ymax=ymax)

    total_plots = 0
    logger.debug(f"processing {metric}")
    plots = []
    for arch in metric_data:
        metric_data[arch]["T"] = metric_data[arch]["T"].squeeze()
        logger.debug((" " * 4) + f"arch = {arch}")
        if len(metric_data[arch]["T"].shape) == 0:
            metric_data[arch]["T"] = metric_data[arch]["T"].unsqueeze(0)
        T_min = int(metric_data[arch]["T"][0])
        T_max = int(metric_data[arch]["T"][-1])
        # T_min = 0
        # T_max = 88
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=T_min, vmax=T_max)
        )
        colors = sm.to_rgba(metric_data[arch]["T"])
        for i, t in enumerate(metric_data[arch]["T"]):
            if ds_len is None:
                steps_per_epoch = 1
            else:
                train_rows, val_rows = ArithmeticDataset.calc_split_len(
                    t.item(), ds_len
                )
                steps_per_epoch = math.ceil(train_rows / batchsize)

            logger.debug((" " * 4) + f"t = {t}")
            # print(
            #    f"metric_data[arch][metric].shape = {metric_data[arch][metric].shape}"
            # )
            Y = metric_data[arch][metric][i]
            # print(f"Y = {Y}")
            assert len(Y.shape) == 1, f"Y.shape = {Y.shape} is invalid"
            X = torch.arange(1, Y.shape[0] + 1) * steps_per_epoch
            assert len(X.shape) == 1, f"X.shape = {X.shape} is invalid"

            label = arch + f" t={t}"

            # ax.set_xlim(left=X[0], right=X[-1] + 1)
            if metric == "val_loss" and inflection_hline:
                Y_infs = find_inflections(Y)
                ax.axhline(y=Y[Y_infs[0]], color="orange")
            if metric == "val_accuracy":
                label += " (max = %.2f)" % max(Y)
            total_plots += 1
            ax.plot(X, Y, label=label, color=colors[i])
    if T_max - T_min <= 10:
        pass
        ax.legend()
    else:
        fig.colorbar(
            sm,
            ax=ax,
            label="% training data",
            ticks=range(T_min, T_max, int((T_max - T_min) / 5)),
        )


def add_comm_graph(
    ax, metric, kind, comm_data, arch, scales=default_axis_scales, cmap=DEFAULT_CMAP
):
    assert metric in (
        "loss",
        "accuracy",
        "perplexity",
    )
    assert kind in (
        "comm",
        "non_comm",
        "modulo",
        "non_modulo",
        "assoc",
        "non_assoc",
        "zero",
        "non_zero",
    )
    ax.set_title(metric)
    ax.set_xscale(scales["x"])
    ax.set_yscale(scales["y"])
    ax.set_xlabel("epochs")
    if "accuracy" in metric:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    X = [int(r["epoch"]) for r in comm_data]
    Y = torch.tensor(
        (
            [float(r["comm" + "_" + metric]) for r in comm_data],
            [float(r["non_comm" + "_" + metric]) for r in comm_data],
            # [float(r["assoc" + "_" + metric]) for r in comm_data],
            # [float(r["non_assoc" + "_" + metric]) for r in comm_data],
            # [float(r["zero" + "_" + metric]) for r in comm_data],
            # [float(r["non_zero" + "_" + metric]) for r in comm_data],
        )
    )
    # label = kind
    # if kind.endswith("comm"):
    #    label += "utative"

    labels = ["commutative", "non-commutative"]
    # labels = ["zero", "non_zero"]
    # labels = ["associative", "non_associative"]
    # label = f"{arch} {kind}_{metric}"
    # ax.plot(X, Y, label=label)
    sm = plt.cm.ScalarMappable(cmap="cividis", norm=plt.Normalize(vmin=0, vmax=len(Y)))
    colors = sm.to_rgba(range(len(Y)))
    ax.stackplot(X, Y, baseline="zero", labels=labels, colors=colors)
    ax.legend()


def add_extremum_graph(
    ax,
    metric,
    kind,
    metric_data,
    scales=default_axis_scales,
    epochs=[-1],
    show_legend=True,
):
    assert kind in ("max", "min")
    ax.set_title(f"{kind} {metric}")
    ax.set_xlabel("training data")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    xmin = 0
    xmax = 100
    ax.axis(xmin=xmin, xmax=xmax)

    # ax.set_ylabel(metric)
    ax.set_xscale(scales["x"])
    ax.set_yscale(scales["y"])
    if "accuracy" in metric:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ymin = -1
        ymax = 105
        ax.axis(ymin=ymin, ymax=ymax)

    # if 'learning' in metric:
    #    ymin=0
    #    ymax=0.002
    #    ax.axis(ymin=ymin, ymax=ymax)

    plots = {}

    total_plots = 0
    for arch in metric_data:
        X = metric_data[arch]["T"]
        if kind == "max":
            Y = torch.max(
                metric_data[arch][metric], dim=1, keepdim=True
            ).values.squeeze()
        elif kind == "min":
            Y = torch.min(
                metric_data[arch][metric], dim=1, keepdim=True
            ).values.squeeze()

        # ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 100, 5))
        label = f"{kind} {metric} {arch}"
        ax.plot(X, Y, label=label)
        total_plots += 1

    if show_legend and total_plots <= 12:
        ax.legend()
        pass


def add_inflection_graphs(
    ax, metric, metric_data, scales=default_axis_scales, smoothing_steps=100
):
    ax.set_title(f"{metric} inflections by train_data_pct")
    ax.set_xlabel("train_data_pct")
    ax.set_ylabel(f"{metric} inflections")
    ax.set_xscale(scales["x"])
    ax.set_yscale(scales["y"])
    if "accuracy" in metric:
        ymin = 0
        ymax = 100
        ax.axis(xmin=0, xmax=87.5, ymin=ymin, ymax=ymax)
    if "learning" in metric:
        ymin = 0
        ymax = 0.002
        ax.axis(xmin=0, xmax=87.5, ymin=ymin, ymax=ymax)

    total_plots = 0
    for arch in metric_data:
        for num in range(5):
            for i, t in enumerate(metric_data[arch]["T"]):
                Y = metric_data[arch][metric][i]
                X = torch.arange(Y.shape[-1])
                inflections = find_inflections(Y, smoothing_steps=smoothing_steps)
                ax.plot(X[inflections], Y[inflections], label=f"{arch} t={t}")
                total_plots += 1

    if total_plots <= 12:
        ax.legend()
        pass


def colorbar(mappable, ticks=None, labels=None):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax, ticks=ticks)
    if labels is not None:
        cbar.ax.set_yticklabels(labels)  # vertically oriented colorbar
    plt.sca(last_axes)
    return cbar


def add_matshow(
    fig, ax, t, name, vmin=0, vmax=100, cmap=DEFAULT_CMAP, show_colorbar=True
):
    sides = ("left", "right", "top", "bottom")
    labels = {
        "left": True,
        "right": False,
        "top": False,
        "bottom": True,
        "labelleft": True,
        "labelright": False,
        "labeltop": False,
        "labelbottom": True,
    }
    m = ax.matshow(
        t.cpu().detach().numpy(), vmin=vmin, vmax=vmax, origin="lower", cmap=cmap
    )
    # c = ax.pcolor(t.cpu(), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(name)
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.set_xticks(np.arange(0, t.shape[1], 10))
    # ax.set_xticklabels(np.arange(1, t.shape[1]+1))
    # ax.set_yticks(np.arange(0.5, t.shape[0] + .5, 1))
    ax.set_yticks(np.arange(0, t.shape[0], 10))
    # ax.set_yticks(np.arange(t.shape[0]))
    # ax.set_yticklabels(np.arange(1, t.shape[0]+1))
    ax.tick_params(axis="both", which="both", **labels)
    if show_colorbar:
        colorbar(m)
