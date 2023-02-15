#!/bin/bash

set -e

ROOT_DIR="/tmp"

pushd $ROOT_DIR

git clone https://github.com/edknv/distributed-embeddings.git

git config --global --add safe.directory $ROOT_DIR/distributed-embeddings
git config --global --add safe.directory $ROOT_DIR/distributed-embeddings/third_party/thrust

pushd $ROOT_DIR/distributed-embeddings

git checkout fix_shape_graph_mode

git submodule update --init --recursive
make pip_pkg
python -m pip install --force-reinstall artifacts/*.whl
python setup.py install

popd +2

python -c "import distributed_embeddings"
