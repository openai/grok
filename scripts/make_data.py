#!/usr/bin/env python

from argparse import ArgumentParser
from grok.data import create_data_files, DEFAULT_DATA_DIR


parser = ArgumentParser()
parser.add_argument("-d", "--data_directory", type=str, default=DEFAULT_DATA_DIR)
args = parser.parse_args()
create_data_files(args.data_directory)