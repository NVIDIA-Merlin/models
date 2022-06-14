# Merlin Models Example Notebooks

The example notebooks demonstrate how to use Merlin Models with TensorFlow on a variety of datasets.

## Inventory

- [Getting Started: Develop a Model for MovieLens](01-Getting-started.ipynb): Get started with Merlin Models by training [Facebook's DLRM](https://arxiv.org/pdf/1906.00091.pdf) architecture with only 3 commands.

- [From ETL to Training RecSys models - NVTabular and Merlin Models integrated example](02-Merlin-Models-and-NVTabular-integration.ipynb): Learn how the `Schema` object connects the feature engineering and training steps. You'll learn how to apply Merlin Models to your own dataset structures.

- [Iterating over Deep Learning Models](03-Exploring-different-models.ipynb): Explore different ranking model architectures, such as [Neural Collaborative Filtering (NCF)](https://arxiv.org/pdf/1708.05031.pdf), MLP, [DRLM](https://arxiv.org/abs/1906.00091.pdf), and [Deep & Cross Network (DCN)](https://arxiv.org/pdf/1708.05123.pdf).

- [Exporting Ranking Models](04-Exporting-ranking-models.ipynb): Save a `Workflow` object and a `Model` object in preparation of deploying the model to production.

- [Two-Stage Recommender Systems](05-Retrieval-Model.ipynb): Build a two-tower model using synthetic data that mimics the [Ali-CCP: Alibaba Click and Conversion Prediction](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408#1) dataset.

- [The Next Step: Define Your Own Architecture](06-Define-your-own-architecture-with-Merlin-Models.ipynb): See how to combine pre-existing blocks to define a custom model.

## Running the Example Notebooks

You can run the examples with Docker containers.
Docker containers are available from the NVIDIA GPU Cloud.
Access the catalog of containers at <http://ngc.nvidia.com/catalog/containers>.

Most example notebooks demonstrate how to use Merlin Models with TensorFlow.
The following container can train a model and perform inference and is capable for all the notebooks:

- [merlin-tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow) (contains Merlin Core, Merlin Models, Merlin Systems, NVTabular, TensorFlow, and Triton Inference Server)

Alternatively, you can [install Merlin Models from source](https://github.com/NVIDIA-Merlin/models#installing-merlin-models-from-source) and other required libraries to run the notebooks on your host by following the instructions in the README from the GitHub repository.

To run the example notebooks using Docker containers, perform the following steps:

1. Pull and start the container by running the following command:

   ```shell
   docker run --gpus all --rm -it \
     -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host \
     <docker container> /bin/bash
   ```

   The container opens a shell when the run command execution is completed.
   Your shell prompt should look similar to the following example:

   ```shell
   root@2efa5b50b909:
   ```

1. Start the JupyterLab server by running the following command:

   ```shell
   jupyter-lab --allow-root --ip='0.0.0.0'
   ```

   View the messages in your terminal to identify the URL for JupyterLab.
   The messages in your terminal show similar lines to the following example:

   ```shell
   Or copy and paste one of these URLs:
   http://2efa5b50b909:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
   or http://127.0.0.1:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
   ```

1. Open a browser and use the `127.0.0.1` URL provided in the messages by JupyterLab.

1. After you log in to JupyterLab, navigate to the `/models/examples` directory to try out the example notebooks.
