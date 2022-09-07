#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2022 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_models_01-getting-started/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Getting Started with Merlin Models: Develop a Model for MovieLens
# 
# This notebook is created using the latest stable [merlin-tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow/tags) container. 
# 
# ## Overview
# 
# [Merlin Models](https://github.com/NVIDIA-Merlin/models/) is a library for training recommender models. Merlin Models let Data Scientists and ML Engineers easily train standard RecSys models on their own dataset, getting GPU-accelerated models with best practices baked into the library. This will also let researchers to build custom models by incorporating standard components of deep learning recommender models, and then benchmark their new models on example offline datasets. Merlin Models is part of the [Merlin open source framework](https://developer.nvidia.com/nvidia-merlin).
# 
# Core features are:
# - Many different recommender system architectures (tabular, two-tower, sequential) or tasks (binary, multi-class classification, multi-task)
# - Flexible APIs targeted to both production and research
# - Deep integration with NVIDIA Merlin platform, including NVTabular for ETL and Merlin Systems model serving
# 
# 
# ### Learning objectives
# 
# - Training [Facebook's DLRM model](https://arxiv.org/pdf/1906.00091.pdf) very easily with our high-level API.
# - Understanding Merlin Models high-level API

# ## Downloading and preparing the dataset

# In[2]:


import os
import merlin.models.tf as mm

from merlin.datasets.entertainment import get_movielens


# We provide the `get_movielens()` function as a convenience to download the dataset, perform simple preprocessing, and split the data into training and validation datasets.

# In[3]:


input_path = os.environ.get("INPUT_DATA_DIR", os.path.expanduser("~/merlin-models-data/movielens/"))
train, valid = get_movielens(variant="ml-1m", path=input_path)


# ## Training the DLRM Model with Merlin Models

# We define the DLRM model, whose prediction task is a binary classification. From the `schema`, the categorical features are identified (and embedded) and the target column is also automatically inferred, because of the schema tags. We talk more about the schema in the next [example notebook (02)](02-Merlin-Models-and-NVTabular-integration.ipynb),

# In[4]:


model = mm.DLRMModel(
    train.schema,
    embedding_dim=64,
    bottom_block=mm.MLPBlock([128, 64]),
    top_block=mm.MLPBlock([128, 64, 32]),
    prediction_tasks=mm.BinaryClassificationTask(train.schema),
)

model.compile(optimizer="adam")


# Next, we train the model.

# In[5]:


model.fit(train, batch_size=1024)


# We evaluate the model...

# In[6]:


metrics = model.evaluate(valid, batch_size=1024, return_dict=True)


# ... and check the evaluation metrics. We use by default typical binary classification metrics -- Precision, Recall, Accuracy and AUC. But you can also provide your own metrics list by setting `BinaryClassificationTask(..., metrics=[])`.

# In[7]:


metrics


# ## Conclusion

# Merlin Models enables users to define and train a deep learning recommeder model with only 3 commands.
# 
# ```python
# model = mm.DLRMModel(
#     train.schema,
#     embedding_dim=64,
#     bottom_block=mm.MLPBlock([128, 64]),
#     top_block=mm.MLPBlock([128, 64, 32]),
#     prediction_tasks=mm.BinaryClassificationTask(
#         train.schema.select_by_tag(Tags.TARGET).column_names[0]
#     ),
# )
# model.compile(optimizer="adam")
# model.fit(train, batch_size=1024)
# ```

# ## Next steps

# In the next example notebooks, we will show how the integration with NVTabular and how to explore different recommender models.
