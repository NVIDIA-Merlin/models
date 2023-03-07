#!/bin/bash

cd /tmp
git clone --depth 1 https://github.com/NVIDIA-Merlin/HugeCTR hugectr
cd hugectr
git submodule update --init --recursive
cd sparse_operation_kit
python -m pip install scikit-build
python setup.py install
