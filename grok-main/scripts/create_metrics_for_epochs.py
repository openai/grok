#!/usr/bin/env python

import logging

logging.basicConfig(level=logging.ERROR)
import csv
import copy
import os
import grok
import numpy as np
import sys
import subprocess
import torch
from torch.multiprocessing import Process
from grok import trainer
from tqdm import tqdm
from argparse import ArgumentParser
from collections import Counter


torch.multiprocessing.freeze_support()
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

# Get args
EPOCHS = (
    list(range(10))
    + list(range(10, 200, 2))
    + list(range(200, 5000, 10))
    + list(range(5000, 10000, 50))
    + [10000]
)

parser = ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, help="where to find the runs", required=True
)
parser.add_argument("--expt", type=str, default=None)
parser.add_argument("--epochs_per_run", type=int, default=40)


def parent(expts):
    for expt in expts:
        print(f"Processing {expt}")
        all_results = {}
        for first_epoch in range(0, len(EPOCHS), hparams.epochs_per_run):
            these_epochs = [
                str(e)
                for e in EPOCHS[first_epoch : first_epoch + hparams.epochs_per_run]
            ]
            expt_dir = data_dir + "/" + expt
            cmd = [
                "./create_partial_metrics.py",
                f"--gpu={hparams.gpu}",
                f"--expt_dir={expt_dir}",
                f'--epochs={",".join(these_epochs)}',
            ]
            result = subprocess.run(cmd, capture_output=False, shell=False)
            if result.returncode != 0:
                sys.exit(result.returncode)


hparams = trainer.get_args(parser)

data_dir = hparams.data_dir

if hparams.expt is not None:
    expts = [hparams.expt]
else:
    expts = os.listdir(data_dir)

parent(expts)
