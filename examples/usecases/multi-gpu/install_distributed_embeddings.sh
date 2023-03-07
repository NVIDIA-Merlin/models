#!/bin/bash

set -e

WORK_DIR=$(pwd)
ROOT_DIR="/tmp"

cd $ROOT_DIR

if [ ! -d "distributed-embeddings" ]; then
  git clone https://github.com/NVIDIA-Merlin/distributed-embeddings.git
fi

cd distributed-embeddings

git submodule update --init --recursive
make pip_pkg
python -m pip install --force-reinstall artifacts/*.whl
python setup.py install

cd $WORK_DIR

python -c "import distributed_embeddings"