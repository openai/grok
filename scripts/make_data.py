#!/usr/bin/env python

from argparse import ArgumentParser

from grok.data import DEFAULT_DATA_DIR, create_data_files

parser = ArgumentParser()
parser.add_argument("-d", "--data_directory", type=str, default=DEFAULT_DATA_DIR)
args = parser.parse_args()
create_data_files(args.data_directory)
