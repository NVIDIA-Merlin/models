# Merlin Models 
[![PyPI version shields.io](https://img.shields.io/pypi/v/merlin-models.svg)](https://pypi.python.org/pypi/merlin-models/)
![GitHub License](https://img.shields.io/github/license/NVIDIA-Merlin/models)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/models/main/)

The Merlin Models library provides standard models for recommender systems with an aim for high-quality implementations
that range from classic machine learning models to highly-advanced deep learning models.

The goal of this library is to make it easy for users in the industry to train and deploy recommender models with the best
practices that are already baked into the library. The library simplifies how users in the industry can train standard models against their dataset and put high-performance, GPU-accelerated models into production. The library also enables researchers to build custom
models by incorporating standard components of deep learning recommender models and then researchers can benchmark the new models on
example offline
datasets.

In our initial releases, Merlin Models features a TensorFlow API. The PyTorch API will come later.

## Contents

To learn about the core features of Merlin Models, see the [Models Overview](docs/source/models_overview.md) page.

**[RecSys models provided](https://nvidia-merlin.github.io/models/main/models_overview.html)** - We provide a high-level API for classic and state-of-the-art deep learning architectures for recommender models including both Retrieval (e.g. Matrix Factorization, Two tower, YouTube DNN, ..) and Ranking (e.g. DLRM, DCN-v2, DeepFM, ...) models.

**Building blocks** - Within Merlin Models, recommender models are built based on reusable building blocks, making it easy to combine those blocks to define new architectures. It provides model definition blocks (MLP layers, Factorization Layers, input blocks, negative samplers, loss functions), training models (data loaders from Parquet files) and evaluation (e.g. ranking metrics).

**Integration with Merlin platform** - Merlin Models is deeply integrated with the other Merlin components, for example with NVTabular for pre-processing and Merlin Systems for inference, making it straightforward to build performant end-to-end recsys pipelines.

**[Merlin Models DataLoaders](https://nvidia-merlin.github.io/models/main/api.html#loader-utility-functions)** - Merlin provides seamless integration with common deep learning frameworks, such as TensorFlow, PyTorch, and HugeCTR. When training deep learning recommender system models, dataloading can be a bottleneck. Therefore, we've developed custom, highly-optimized dataloaders to accelerate existing TensorFlow and PyTorch training pipelines. The Merlin dataloaders can lead to a speedup that is nine times faster than the same training pipeline used with the GPU. With the Merlin dataloaders, you can:
- remove bottlenecks from dataloading by processing large chunks of data at a time instead of item by item.
- process datasets that don't fit within the GPU or CPU memory by streaming from the disk.
- prepare batches asynchronously into the GPU to avoid CPU-GPU communication.
- integrate easily into existing TensorFlow or PyTorch training pipelines by using a similar API.
## Installation

### Installing Merlin Models with pip

Merlin Models can be installed with `pip` by running the following command:
```shell
pip install merlin-models
```
Note: Installing Merlin Models with `pip` will not install some additional GPU dependencies (like CUDA Toolkit). Prefer Docker where possible.

### Docker Containers that include Merlin Models

Merlin Models is pre-installed in the NVIDIA Merlin Docker containers that are available in the [NVIDIA container repository](https://ngc.nvidia.com/catalog/containers/nvidia:merlin). There are six different containers:


%TODO: Do not include the list of components within each Container here (to avoid updating multiple places when needed), but rather keep them in a Merlin top-level readme.
<!-- prettier-ignore-start -->

| Container Name             | Container Location | Functionality |
| -------------------------- | ------------------ | ------------- |
| merlin-tensorflow-training | [https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow-training](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow-training) | Transformers4Rec, NVTabular, TensorFlow, and HugeCTR Tensorflow Embedding plugin |
| merlin-pytorch-training    | [https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-pytorch-training](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-pytorch-training)    | Transformers4Rec, NVTabular and PyTorch
| merlin-inference           | [https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-inference](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-inference)           | Transformers4Rec, NVTabular, PyTorch, and Triton Inference |  |


<!-- prettier-ignore-end -->

To use these Docker containers, you'll first need to install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide GPU support for Docker. You can use the NGC links referenced in the table above to obtain more information about how to launch and run these containers.

### Installing project from Source

Merlin Models can be installed from source by running the following commands: 
```
git clone https://github.com/NVIDIA-Merlin/models
cd models && pip install -e .
```
<!-- Need core benefits, Common use cases, or Highlights -->

## Getting Started
Merlin Models makes it straighforward to define architectures that adapt to different input features, based on the schema output by NVTabular . You can read more about how the **schema** object in this [example](https://github.com/NVIDIA-Merlin/models/blob/main/examples/02-Merlin-Models-and-NVTabular-applying-to-your-own-dataset.ipynb).

You can easily build popular recsys architectures like [DLRM](http://arxiv.org/abs/1906.00091), as exemplified below. To build the internal input layer, it identifies from the schema object what are continuous features or categorical features, for which embedding tables are created. To define the body of the architecture, MLP layers are used with configurable dimensions. Then the head of the architecture is created from the chosen task (`BinaryClassificationTask`), whose target binary feature is also infered from the schema (i.e., as tagged as 'TARGET'). You can find an example on how to build DLRM using our low-level API [here](https://nvidia-merlin.github.io/models/main/models_overview.html#deep-learning-recommender-model).

Then you can train and evaluate as with a typical Keras model.

```python
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io.dataset import Dataset

train_ds = Dataset(os.path.join(data_path, 'train/*.parquet'))
valid_ds = Dataset(os.path.join(output_path 'valid/*.parquet'))

model = mm.DLRMModel(
    train_ds.schema,
    embedding_dim=64,
    bottom_block=mm.MLPBlock([128, 64]),
    top_block=mm.MLPBlock([128, 64, 32]),
    prediction_tasks=mm.BinaryClassificationTask(schema)
    ),
)

opt = tf.keras.optimizers.Adagrad(learning_rate=1e-4)
model.compile(optimizer=opt, run_eagerly=False)
model.fit(train, validation_data=valid_ds, batch_size=batch_size)
eval_metrics = model.evaluate(valid_ds, return_dict=True)
```



<!--
## Sample Notebooks

* Link to each notebook directory when #190 is merged.
-->

## Notebook Examples and Tutorials
The [examples](https://github.com/NVIDIA-Merlin/models/tree/main/examples) folder includes a series of notebooks to help you getting familiar with Merlin Models. You can learn more about these notebooks in this [examples' readme](https://github.com/NVIDIA-Merlin/models/tree/main/examples#merlin-models-example-notebooks) doc.


## Feedback and Support

If you'd like to contribute to the library directly, see the [CONTRIBUTING.md](CONTRIBUTING.md) file.
We're particularly interested in contributions or feature requests for our feature engineering and preprocessing operations.
To further advance our Merlin Roadmap, we encourage you to share all the details regarding your recommender system pipeline in this [survey](https://developer.nvidia.com/merlin-devzone-survey).

<!-- TODO
If you're interested in learning more about how Merlin Models works, see our documentation.
We also have API documentation that outlines the specifics of the available calls within the library.
-->