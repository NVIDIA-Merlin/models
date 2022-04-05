Merlin Models Example Notebooks
===============================

The example notebooks demonstrate how to use Merlin Models with TensorFlow on a variety of datasets.

Inventory
---------

* `Getting Started: Develop a Model for MovieLens <01-Getting-started.ipynb>`_: Get started with Merlin Models by training `Facebook's DLRM <https://arxiv.org/pdf/1906.00091.pdf>`_ architecture with only 3 commands.

* `From ETL to Training RecSys models - NVTabular and Merlin Models integrated example <02-Merlin-Models-and-NVTabular-integration.ipynb>`_: Learn how the `Schema` object connects the feature engineering and training steps. You'll learn how to apply Merlin Models to your own dataset structures.

* `Iterating over Deep Learning Models <03-Exploring-different-models.ipynb>`_: Explore different ranking model architectures, such as `Neural Collaborative Filtering (NCF) <https://arxiv.org/pdf/1708.05031.pdf>`_, MLP, `DRLM <https://arxiv.org/abs/1906.00091>`_, and `Deep & Cross Network (DCN) <https://arxiv.org/pdf/1708.05123.pdf>`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   01-Getting-started.ipynb
   02-Merlin-Models-and-NVTabular-integration.ipynb
   03-Exploring-different-models.ipynb

Running the Example Notebooks
-----------------------------

You can run the examples with Docker containers.
Docker containers are available from the NVIDIA GPU Cloud.
Access the catalog of containers at http://ngc.nvidia.com/catalog/containers.

Depending on which example you want to run, you should use any one of these Docker containers:

- `Merlin-Tensorflow-Training <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow-training>`_ (contains Merlin Core, Merlin Models, NVTabular and TensorFlow)
- `Merlin-Inference <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow-inference>`_ (contains Merlin Core, Merlin Models, Merlin Systems, NVTabular, TensorFlow and Triton Inference Server)

Alternatively, you can `install Merlin Models from source <../README.md#installing-merlin-models-from-source>`_ and other required libraries to run the notebooks on your host.

To run the example notebooks using Docker containers, perform the following steps:

1. Pull and start the container by running the following command:

   .. code-block:: shell

      docker run --gpus all --rm -it \
        -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host \
        <docker container> /bin/bash

   The container opens a shell when the run command execution is completed.
   Your shell prompt should look similar to the following example:

   .. code-block:: shell

      root@2efa5b50b909:

2. Install JupyterLab with ``pip`` by running the following command:

   .. code-block:: shell

      pip install jupyterlab

   For more information, see the JupyterLab `Installation Guide <https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html>`_.

3. Start the JupyterLab server by running the following command:

   .. code-block:: shell

      jupyter-lab --allow-root --ip='0.0.0.0'

   View the messages in your terminal to identify the URL for JupyterLab.
   The messages in your terminal show similar lines to the following example:

   .. code-block:: shell

      Or copy and paste one of these URLs:
         http://2efa5b50b909:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
      or http://127.0.0.1:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d

4. Open a browser and use the ``127.0.0.1`` URL provided in the messages by JupyterLab.

5. After you log in to JupyterLab, navigate to the ``/merlin/examples`` directory to try out the example notebooks.

