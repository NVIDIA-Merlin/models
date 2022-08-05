## Merlin Models

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

In our initial releases, Merlin Models features a TensorFlow API. The PyTorch API is initiated, but incomplete.

### Benefits of Merlin Models

**[RecSys model implementations](https://nvidia-merlin.github.io/models/main/models_overview.html)** - The library provides a high-level API for classic and state-of-the-art deep learning architectures for recommender models.
These models include both retrieval (e.g. Matrix Factorization, Two tower, YouTube DNN, ..) and ranking (e.g. DLRM, DCN-v2, DeepFM, ...) models.

**Building blocks** - Within Merlin Models, recommender models are built on reusable building blocks.
The design makes it easy to combine the blocks to define new architectures.
The library provides model definition blocks (MLP layers, factorization layers, input blocks, negative samplers, loss functions), training models (data loaders from Parquet files), and evaluation (e.g. ranking metrics).

**Integration with Merlin platform** - Merlin Models is deeply integrated with the other Merlin components.
For example, models depend on NVTabular for pre-processing and integrate easily with Merlin Systems for inference.
The thoughtfully-designed integration makes it straightforward to build performant end-to-end RecSys pipelines.

**[Merlin Models DataLoaders](https://nvidia-merlin.github.io/models/main/api.html#loader-utility-functions)** - Merlin provides seamless integration with common deep learning frameworks, such as TensorFlow, PyTorch, and HugeCTR.
When training deep learning recommender system models, data loading can be a bottleneck.
To address the challenge, Merlin has custom, highly-optimized dataloaders to accelerate existing TensorFlow and PyTorch training pipelines.
The Merlin dataloaders can lead to a speedup that is nine times faster than the same training pipeline used with the GPU.

With the Merlin dataloaders, you can:

- Remove bottlenecks from data loading by processing large chunks of data at a time instead of item by item.
- Process datasets that don't fit within the GPU or CPU memory by streaming from the disk.
- Prepare batches asynchronously into the GPU to avoid CPU-to-GPU communication.
- Integrate easily into existing TensorFlow or PyTorch training pipelines by using a similar API.

To learn about the core features of Merlin Models, see the [Models Overview](https://nvidia-merlin.github.io/models/main/models_overview.html) page.

### Installation

#### Installing Merlin Models Using Pip

Merlin Models can be installed with `pip` by running the following command:

```shell
pip install merlin-models
```

> Installing Merlin Models with `pip` does not install some additional GPU dependencies, such as the CUDA Toolkit.
> When you run Merlin Models in one of our Docker containers, the dependencies are already installed.

#### Docker Containers that include Merlin Models

Merlin Models is included in the Merlin Containers.

Refer to the [Merlin Containers](https://nvidia-merlin.github.io/Merlin/main/containers.html) documentation page for information about the Merlin container names, URLs to the container images on the NVIDIA GPU Cloud catalog, and key Merlin components.

#### Installing Merlin Models from Source

Merlin Models can be installed from source by running the following commands:

```shell
git clone https://github.com/NVIDIA-Merlin/models
cd models && pip install -e .
```

### Getting Started

Merlin Models makes it straightforward to define architectures that adapt to different input features.
This adaptability is provided by building on a core feature of the NVTabular library.
When you use NVTabular for feature engineering, NVTabular creates a schema that identifies the input features.
You can see the `Schema` object in action by looking at the [From ETL to Training RecSys models - NVTabular and Merlin Models integrated example](https://nvidia-merlin.github.io/models/main/examples/02-Merlin-Models-and-NVTabular-integration.html) example notebook.

You can easily build popular RecSys architectures like [DLRM](http://arxiv.org/abs/1906.00091), as shown in the following code sample.
After you define the model, you can train and evaluate it with a typical Keras model.

```python
import merlin.models.tf as mm
from merlin.io.dataset import Dataset

train = Dataset(PATH_TO_TRAIN_DATA)
valid = Dataset(PATH_TO_VALID_DATA)

model = mm.DLRMModel(
    train.schema,                                                   # 1
    embedding_dim=64,
    bottom_block=mm.MLPBlock([128, 64]),                            # 2
    top_block=mm.MLPBlock([128, 64, 32]),
    prediction_tasks=mm.BinaryClassificationTask(train.schema)      # 3
)

model.compile(optimizer="adagrad", run_eagerly=False)
model.fit(train, validation_data=valid, batch_size=1024)
eval_metrics = model.evaluate(valid, batch_size=1024, return_dict=True)
```

1.  To build the internal input layer, the model identifies them from the schema object.
    The schema identifies the continuous features and categorical features, for which embedding tables are created.
2.  To define the body of the architecture, MLP layers are used with configurable dimensions.
3.  The head of the architecture is created from the chosen task, `BinaryClassificationTask` in this example.
    The target binary feature is also inferred from the schema (i.e., tagged as 'TARGET').

You can find more details and information about a low-level API in our overview of the
[Deep Learning Recommender Model](https://nvidia-merlin.github.io/models/main/models_overview.html#deep-learning-recommender-model).

### Notebook Examples and Tutorials

View the example notebooks in the [documentation](https://nvidia-merlin.github.io/models/main/examples/README.html) to help you become familiar with Merlin Models.

The same notebooks are available in the `examples` directory from the [Merlin Models](https://github.com/NVIDIA-Merlin/models) GitHub repostory.

### Feedback and Support

If you'd like to contribute to the library directly, see the [CONTRIBUTING.md](CONTRIBUTING.md) file.
We're particularly interested in contributions or feature requests for our feature engineering and preprocessing operations.
To further advance our Merlin Roadmap, we encourage you to share all the details regarding your recommender system pipeline in this [survey](https://developer.nvidia.com/merlin-devzone-survey).
