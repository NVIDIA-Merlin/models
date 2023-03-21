#!/bin/bash

set -e

ROOT_DIR=$1

cd $ROOT_DIR

rm -rf hugectr/

git clone -b release-23.02 https://github.com/NVIDIA-Merlin/HugeCTR.git hugectr

cd hugectr/sparse_operation_kit/
python setup.py install

rm -rf ${HUGECTR_HOME}
