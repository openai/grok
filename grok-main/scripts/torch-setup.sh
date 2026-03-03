#!/bin/bash

# Set up torch with magma support

DIR="`mktemp -d`"
cd $DIR

# Install deps
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

# Install torch, magma
conda install -y -c pytorch magma-cuda110

# Build torch from scratch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# If updating an existing checkout
# git submodule sync
# git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install