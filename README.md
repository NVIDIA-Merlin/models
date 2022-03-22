# Merlin Models

[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)

The Merlin Models library provides standard models for recommender systems with an aim for high quality implementations
that range from classic machine learning models to highly-advanced deep learning models.

The goal of this library is to make it easy for users in industry to train and deploy recommender models with the best
practices that are already baked into the library. The library simplifies how users in industry can train standard models against their own
dataset and put high-performance, GPU-accelerated models into production. The library also enables researchers to build custom
models by incorporating standard components of deep learning recommender models and then researchers can benchmark the new models on
example offline
datasets.

## Installation

### Installing Merlin Models with pip


```shell
pip install merlin-models
```

### Installing Merlin Models with conda

```shell
conda install -c nvidia merlin-models
```

### Docker Containers that include Merlin Models

Merlin Models is pre-installed in the NVIDIA Merlin Docker containers that are available in the [NVIDIA container repository](https://ngc.nvidia.com/catalog/containers/nvidia:merlin) in three different containers:

<!-- prettier-ignore-start -->

| Container Name             | Container Location | Functionality |
| -------------------------- | ------------------ | ------------- |
| merlin-tensorflow-training | [https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow-training](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow-training) | Transformers4Rec, NVTabular, TensorFlow, and HugeCTR Tensorflow Embedding plugin |
| merlin-pytorch-training    | [https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-pytorch-training](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-pytorch-training)    | Transformers4Rec, NVTabular and PyTorch
| merlin-inference           | [https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-inference](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-inference)           | Transformers4Rec, NVTabular, PyTorch, and Triton Inference |  |


<!-- prettier-ignore-end -->

To use these Docker containers, you'll first need to install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide GPU support for Docker. You can use the NGC links referenced in the table above to obtain more information about how to launch and run these containers.

<!-- Need core benefits, Common use cases, or Highlights -->

## Core Features

To learn about the core features of Merlin Models, see the [Models Overview](docs/source/models_overview.md) page.
The key retrieval models are as follows:

* Matrix factorization
* YouTube DNN retrieval
* Two tower
* Ranking models such as DLRM and DCN-V2
* Multi-task learning with Mixture-of-experts or Progressive layered extraction

<!--
## Sample Notebooks

* Link to each notebook directory when #190 is merged.
-->

## Feedback and Support

If you'd like to contribute to the library directly, see the [CONTRIBUTING.md](CONTRIBUTING.md) file.
We're particularly interested in contributions or feature requests for our feature engineering and preprocessing operations.
To further advance our Merlin Roadmap, we encourage you to share all the details regarding your recommender system pipeline in this [survey](https://developer.nvidia.com/merlin-devzone-survey).

<!-- TODO
If you're interested in learning more about how Merlin Models works, see our documentation.
We also have API documentation that outlines the specifics of the available calls within the library.
-->
