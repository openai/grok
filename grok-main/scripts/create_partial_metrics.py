#!/usr/bin/env python

import logging

logging.basicConfig(level=logging.ERROR)
import csv
import copy
import glob
import os
import grok
import numpy as np
import subprocess
import torch
import sys
from torch.multiprocessing import Process
from grok import trainer
from tqdm import tqdm
from argparse import ArgumentParser
from collections import Counter
from grok_runs import RUNS
from grok_metrics_lib import (
    DATA_DIR,
    load_metric_data,
    get_metric_data,
    most_interesting,
)


# Make N_EPOCHS exponentially spaced sets of epochs from 1 to 10,000
N_EPOCHS = 32
BASE = 9999 ** (1.0 / (N_EPOCHS - 1))
epochs = (BASE ** torch.arange(1, N_EPOCHS).float()).long().tolist()
DEFAULT_EPOCHS = ",".join([str(i) for i in epochs])

parser = ArgumentParser()
parser.add_argument("--expt_dir", type=str, help="where to find the runs")
parser.add_argument("--epochs", type=str, default=DEFAULT_EPOCHS)


def child(hparams):
    expt_dir = hparams.expt_dir
    epochs = [int(e) for e in hparams.epochs.split(",")]
    # print("epochs = ", epochs)
    device = torch.device(f"cuda:{hparams.gpu}")
    ckpt_dir = expt_dir + "/" + "checkpoints"
    # ckpt_files = [ckpt_dir + f"/epoch={epoch}.ckpt" for epoch in epochs]
    hparams.logdir = expt_dir

    results = {
        "val_loss": None,
        "val_accuracy": None,
    }

    processed_epochs = []
    # with tqdm(epochs, unit="epochs", initial=epochs[0], total=epochs[-1]) as pbar:
    #    last_epoch = epochs[0]
    for idx, epoch in tqdm(list(enumerate(epochs))):
        # pbar.update(epoch - last_epoch)
        # last_epoch = epoch
        ckpt_files = glob.glob(ckpt_dir + f"/epoch={epoch}-step=*.ckpt")
        ckpt_files += glob.glob(ckpt_dir + f"/epoch={epoch}.ckpt")
        try:
            ckpt_file = ckpt_files[-1]
            ckpt = torch.load(
                ckpt_file,
                map_location=f"cuda:{0}",  # FIXME
            )
            processed_epochs.append(epoch)
        except FileNotFoundError:
            continue

        for k, v in ckpt["hyper_parameters"].items():
            setattr(hparams, k, v)

        new_state_dict = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("transformer."):
                new_state_dict[k] = v
            else:
                new_state_dict["transformer." + k] = v
        ckpt["state_dict"] = new_state_dict

        model = trainer.TrainableTransformer(hparams).float()
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device).eval()
        dl = model.test_dataloader()
        dl.reset_iteration(shuffle=False)

        outputs = [model.test_step(batch, idx) for (idx, batch) in enumerate(dl)]
        r = model.test_epoch_end(outputs)["log"]
        if results["val_loss"] is None:
            results["val_loss"] = r["test_loss"].squeeze().unsqueeze(0)
            results["val_accuracy"] = r["test_accuracy"].squeeze().unsqueeze(0)
        else:
            results["val_loss"] = torch.cat(
                [results["val_loss"], r["test_loss"].squeeze().unsqueeze(0)], dim=0
            )
            results["val_accuracy"] = torch.cat(
                [
                    results["val_accuracy"],
                    r["test_accuracy"].squeeze().unsqueeze(0),
                ],
                dim=0,
            )

    for k, v in results.items():
        results[k] = v.to("cpu")
    results["epochs"] = torch.LongTensor(processed_epochs, device="cpu")
    results["dl"] = dl

    os.makedirs(expt_dir + "/activations", exist_ok=True)
    ptfile = (
        expt_dir + f"/activations/activations_{epochs[0]:010d}_{epochs[-1]:010d}.pt"
    )
    torch.save(results, ptfile)


if __name__ == "__main__":
    hparams = trainer.get_args(parser)
    if hparams.expt_dir is not None:
        child(hparams)
    else:
        for operation in RUNS:
            print(f"running {operation}")
            ds_len, run = RUNS[operation]
            data = load_metric_data(
                f"{DATA_DIR}/{run}", epochs=10000, load_partial_data=False
            )
            metric_data = get_metric_data(data)
            metric_data = most_interesting(metric_data)
            for arch in metric_data:
                interesting_t = int(metric_data[arch]["T"][0].item())
                expt = f"{arch}_T-{interesting_t}"
                print(f"--> expt {expt}")
                glb = f"{DATA_DIR}/{run}/{expt}_*"
                # print(f"glb {glb}")
                expt_dir = glob.glob(glb)[0]
                cmd = [sys.argv[0], "--expt_dir", expt_dir]
                subprocess.run(cmd, check=False, shell=False)
                # child(hparams)
