# [Merlin Models](https://github.com/NVIDIA-Merlin/models/) | [Documentation](https://nvidia-merlin.github.io/models/main/)

<p align="left">
    <a href="https://github.com/NVIDIA-Merlin/models/actions/workflows/pytorch.yml?branch=main">
        <img alt="Build" src="https://github.com/NVIDIA-Merlin/models/actions/workflows/pytorch.yml/badge.svg?branch=main">
    </a>
    <a href="https://github.com/NVIDIA-Merlin/models/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/NVIDIA-Merlin/models.svg?color=blue">
    </a>
    <a href="https://nvidia-merlin.github.io/models/main/">
        <img alt="Documentation" src="https://img.shields.io/website.svg?down_color=red&down_message=offline&up_message=online&url=https%3A%2F%2Fnvidia-merlin.github.io%2Fmodels%2Fmain%2F">
    </a>
    <a href="https://github.com/NVIDIA-Merlin/models/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/NVIDIA-Merlin/models.svg">
    </a>
</p>

NVIDIA Merlin Models provides standard models for recommender systems. The
models are high quality implementations that range from classic machine learning
models, to more advanced deep learning models.

## Highlights

- TODO

- TODO

- TODO

## Quick tour

TODO

## Use cases

TODO&mdash;Does this apply?

## Installation

### Installing with pip

```shell
pip install merlin-models
```

### Installing with conda

```shell
conda install -c nvidia merlin-models
```

### Installing with Docker

Merlin Models is installed in the following NVIDIA Merlin Docker containers that
are available in the NVIDIA container repository:

<!-- prettier-ignore-start -->

| Container Name             | Container Location | Functionality |
| -------------------------- | ------------------ | ------------- |
| merlin-tensorflow-training | [https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow-training](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow-training) | Transformers4Rec, NVTabular, TensorFlow, and HugeCTR Tensorflow Embedding plugin |
| merlin-pytorch-training    | [https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-pytorch-training](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-pytorch-training)    | Transformers4Rec, NVTabular and PyTorch |
| merlin-inference           | [https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-inference](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-inference)           | Transformers4Rec, NVTabular, PyTorch, and Triton Inference |

<!-- prettier-ignore-end -->

> Before you can use these Docker containers, you need to install the
> [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide
> GPU support for Docker. You can use the NGC links referenced in the preceding
> table to get more information about how to launch and run these containers.

## Feedback and Support

If you'd like to contribute to the project, see the
[CONTRIBUTING.md](https://github.com/NVIDIA-Merlin/models/blob/main/CONTRIBUTING.md)
file. We're interested in contributions or feature requests for our feature
engineering and preprocessing operations. To further advance our Merlin Roadmap,
we encourage you to share all the details about your Recommender System pipeline
in this [survey](https://developer.nvidia.com/merlin-devzone-survey).
