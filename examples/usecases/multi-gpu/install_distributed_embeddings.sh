#!/bin/bash

set -e

WORK_DIR=$(pwd)
ROOT_DIR="/tmp"

cd $ROOT_DIR

git clone https://github.com/NVIDIA-Merlin/distributed-embeddings.git

git config --global --add safe.directory $ROOT_DIR/distributed-embeddings
git config --global --add safe.directory $ROOT_DIR/distributed-embeddings/third_party/thrust

cd $ROOT_DIR/distributed-embeddings

git submodule update --init --recursive
make pip_pkg
python -m pip install --force-reinstall artifacts/*.whl
python setup.py install

cd $WORK_DIR

python -c "import distributed_embeddings"
