#!/bin/bash

# Get latest models version
cd /models/
git pull origin main

container=$1

## Tensorflow container
if [ "$container" == "merlin-tensorflow-training" ]; then
    make tests-tf
# Pytorch container
elif [ "$container" == "merlin-pytorch-training" ]; then
    make tests-torch
# Inference container
elif [ "$container" == "merlin-inference" ]; then
    make tests-tf make tests-torch
fi
