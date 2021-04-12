#!/usr/bin/env python
# coding: utf-8

# Render metrics graphs

import csv
import logging
import os
import glob
import socket
from argparse import ArgumentParser

from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch

from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from sklearn.manifold import TSNE

import grok
from grok.visualization import *

# from grok_runs import RUNS

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("grok.view_metrics")
logger.setLevel(logging.ERROR)

RUNS = {
    "subtraction": (
        9409,
        "subtraction/2021-02-05-03-33-56-alethea-sjjf",
    ),
}


limits = {
    "min_val_accuracy": 0,
    "max_val_accuracy": 100,
    "min_T": 0,  # 0
    "max_T": 100,  # 87.5
    "min_D": 0,  # 8
    "max_D": 256,  # 256
    "min_H": 0,  # 1
    "max_H": 4,  # 8
    "min_L": 0,  # 1
    "max_L": 4,  # 4
    "min_accuracy": 0,
    "max_accuracy": 100,
}

for k in limits.keys():
    metric = k.replace("min_", "").replace("max_", "")
    assert (
        limits["max_" + metric] >= limits["min_" + metric]
    ), f"invalid {metric} limits"


parser = ArgumentParser()
parser.add_argument("-i", "--image_dir", type=str, default=IMAGE_DIR)
args = parser.parse_args()


def create_loss_curves(
    metric_data,
    epochs,
    run,
    most_interesting_only=False,
    image_dir=args.image_dir,
    ds_len=None,
    cmap=DEFAULT_CMAP,
):
    scales = {
        "x": "log",
        "y": "linear",
    }


    arch = list(metric_data.keys())[0]

    ncols = 2
    nrows = 3
    fig_width = ncols * 8
    fig_height = nrows * 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    add_metric_graph(
        fig, axs[0, 0], "val_loss", metric_data, scales, cmap=cmap, ds_len=ds_len
    )
    add_metric_graph(
        fig, axs[0, 1], "val_accuracy", metric_data, scales, cmap, ds_len=ds_len
    )
    add_metric_graph(
        fig, axs[1, 0], "train_loss", metric_data, scales, cmap, ds_len=ds_len
    )
    add_metric_graph(
        fig, axs[1, 1], "train_accuracy", metric_data, scales, cmap, ds_len=ds_len
    )
    add_metric_graph(
        fig,
        axs[2, 0],
        "learning_rate",
        metric_data,
        scales,
        cmap,  # ds_len=ds_len
    )
    fig.suptitle(f"{operation} {list(data.keys())[0]}")
    fig.tight_layout()

    img_file = f"{image_dir}/loss_curves/{operation}_loss_curves_{arch}"
    if ds_len is not None:
        img_file += "_by_update"
    if most_interesting_only:
        img_file += "_most_interesting"
    img_file += ".png"
    d = os.path.split(img_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {img_file}")
    fig.savefig(img_file)
    plt.close(fig)


def create_max_accuracy_curves(
    metric_data, epochs, run, image_dir=args.image_dir, ds_len=None
):
    scales = {
        "x": "linear",
        "y": "linear",
    }

    ncols = 1
    nrows = 2
    fig_width = ncols * 8
    fig_height = nrows * 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    def get_ax(row=0, col=0, nrows=nrows, ncols=ncols, axs=axs):
        if nrows == 0:
            if ncols == 1:
                return axs
            else:
                return axs[col]
        else:
            if ncols == 1:
                return axs[row]
            else:
                return axs[row, col]

    add_extremum_graph(
        get_ax(0, 0), "val_accuracy", "max", metric_data, show_legend=False
    )
    add_extremum_graph(
        get_ax(1, 0), "train_accuracy", "max", metric_data, show_legend=False
    )
    fig.suptitle(f"{operation} {list(data.keys())[0]}")
    fig.tight_layout()

    expt = list(metric_data.keys())[0]
    img_file = f"{image_dir}/max_accuracy/{operation}_max_accuracy_{arch}.png"
    d = os.path.split(img_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {img_file}")
    fig.savefig(img_file)
    plt.close(fig)


def create_tsne_graphs(operation, expt, run_dir, image_dir=args.image_dir):

    saved_pt_dir = f"{run_dir}/activations"
    saved_pts = []

    loss_ts = []
    accuracy_ts = []
    epochs_ts = []
    print(f'glob = {saved_pt_dir + "/activations_*.pt"}')
    files = sorted(glob.glob(saved_pt_dir + "/activations_*.pt"))
    print(f"files = {files}")

    for file in files:
        print(f"Loading {file}")
        saved_pt = torch.load(file)
        saved_pts.append(saved_pt)
        loss_ts.append(saved_pt["val_loss"].mean(dim=-1))
        accuracy_ts.append(saved_pt["val_accuracy"])
        epochs_ts.append(saved_pt["epochs"].squeeze())

    loss_t = torch.cat(loss_ts, dim=0).T.detach()
    accuracy_t = torch.cat(accuracy_ts, dim=0).T.detach()
    epochs_t = torch.cat(epochs_ts, dim=0).detach()
    print(loss_t.shape)
    print(accuracy_t.shape)
    print(epochs_t.shape)
    ######
    a = 0
    num_eqs = len(loss_t)
    b = a + num_eqs

    print("Doing T-SNE..")
    loss_tsne = TSNE(n_components=2, init="pca").fit_transform(loss_t)
    print("...done T-SNE.")

    ncols = 1
    nrows = 1
    fig_width = ncols * 8
    fig_height = nrows * 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    axs.scatter(loss_tsne[:, 0], loss_tsne[:, 1])

    img_file = f"{image_dir}/tsne/{operation}_{expt}.png"
    d = os.path.split(img_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {img_file}")
    fig.savefig(img_file)
    plt.close(fig)


for operation in RUNS:
    print("")
    print("")
    print(f"Processing {operation}", flush=True)

    if operation.endswith("-epochs"):
        epochs = int(operation.split("/")[-1].split("-")[0])
    else:
        epochs = 5000

    ####

    ds_len, run = RUNS[operation]


    data = load_metric_data(f"{DATA_DIR}/{run}", epochs=epochs, load_partial_data=False)

    # check it
    for arch in data:
        # print(data[arch]["metrics"].shape)
        metrics, expts, epochs = data[arch]["metrics"].shape
        message = (
            f"{arch} : loaded {metrics} metrics, {expts} experiments, {epochs} epochs"
        )
        assert metrics == 5, "INVALID metrics count: " + message
        assert expts < 88, "INVALID experiments count: " + message
        assert epochs == epochs, f"INVALID epochs count: " + message
        print(message)

    # ## Set filters on the data to view

    metric_data = get_metric_data(data, limits)

    # Draw loss and accuracy curves

    create_max_accuracy_curves(metric_data, epochs, run)

    create_loss_curves(metric_data, epochs, run)
    create_loss_curves(metric_data, epochs, run, ds_len=ds_len)

    most_interesting_metric_data = most_interesting(metric_data)

    create_loss_curves(
        most_interesting_metric_data, epochs, run, most_interesting_only=True
    )
    create_loss_curves(
        most_interesting_metric_data,
        epochs,
        run,
        most_interesting_only=True,
        ds_len=ds_len,
    )

    # Draw max accuracy curves

    # T-SNE of loss curves:
    try:
        for arch in most_interesting_metric_data:
            t = int(most_interesting_metric_data[arch]["T"][0].item())
            expt = f"{arch}_T-{t}_DROP-0.0"
            create_tsne_graphs(operation, expt, run_dir=f"{DATA_DIR}/{run}/{expt}")
    except:
        print("TSNE failed")
