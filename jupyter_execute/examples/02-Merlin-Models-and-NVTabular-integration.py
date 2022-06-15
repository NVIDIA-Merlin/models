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


# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # From ETL to Training RecSys models - NVTabular and Merlin Models integrated example
# 
# This notebook is created using the latest stable [merlin-tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow/tags) container. 
# 
# ## Overview
# 
# In [01-Getting-started.ipynb](01-Getting-started.ipynb), we provide a getting started example to train a DLRM model on the MovieLens 1M dataset. In this notebook, we will explore how Merlin Models uses the ETL output from [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular/).<br><br>
# 
# ### Learning objectives
# 
# This notebook provides details on how NVTabular and Merlin Models are linked together. We will discuss the concept of the `schema` file.
# 
# ## Merlin
# 
# [Merlin](https://developer.nvidia.com/nvidia-merlin) is an open-source framework for building large-scale (deep learning) recommender systems. It is designed to support recommender systems end-to-end from ETL to training to deployment on CPU or GPU. Common deep learning frameworks are integrated such as TensorFlow (and PyTorch in the future). Among its key benefits are the easy-to-use and flexible APIs, availability of popular recsys architectures, accelerated training and evaluation with GPU and scaling to multi-GPU or multi-node systems.
# 
# Merlin Models and NVTabular are components of Merlin. They are designed to work closely together. 
# 
# [Merlin Models](https://github.com/NVIDIA-Merlin/models/) is a library to make it easy for users in industry or academia to train and deploy recommender models with best practices baked into the library. Data Scientists and ML Engineers can easily train standard and state-of-the art models on their own dataset, getting high performance GPU accelerated models into production. Researchers can build custom models by incorporating standard components of deep learning recommender models and benchmark their new models on example offline datasets.
# 
# [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular/) is a feature engineering and preprocessing library for tabular data that is designed to easily manipulate terabyte scale datasets and train deep learning (DL) based recommender systems. It provides high-level abstraction to simplify code and accelerates computation on the GPU using the RAPIDS Dask-cuDF library under the hood.
# 
# ## Integration of NVTabular and Merlin Models
# 
# <img src="images/schema.png">
# 
# In this notebook, we focus on an important piece of an ML pipeline: feature engineering and model training.
# 
# If you use NVTabular for feature engineering, NVTabular will output (in addition to the preprocessed parquet files), a **schema** file describing the dataset structures. The schema contains columns statistics, tags and metadata collected by NVTabular. Here are some examples of such metadata computed by some NVTabular preprocessing ops:
# 
# - **Categorify:** This op transforms categorical columns into contiguous integers (`0, ..., |C|`) for embedding layers. The columns that are processed by this op have save in the schema its cardinality `|C|` and are also tagged as **CATEGORICAL**.
# - **Normalize**: This op applies standardization to normalize continuous features. The mean and stddev of the columns are saved to the schema, also being tagged as **CONTINUOUS**.
# 
# The users can also define their own tags in the preprocessing pipeline to group together related features, for further modeling purposes.
# 
# **Let's take a look on the MovieLens 1M example.**

# In[2]:


import os
import pandas as pd
import nvtabular as nvt
from merlin.models.utils.example_utils import workflow_fit_transform
import merlin.io

import merlin.models.tf as mm

from nvtabular import ops
from merlin.core.utils import download_file
from merlin.datasets.entertainment import get_movielens
from merlin.schema.tags import Tags


# We will use the utils function to download, extract and preprocess the dataset.

# In[3]:


input_path = os.environ.get("INPUT_DATA_DIR", os.path.expanduser("~/merlin-models-data/movielens/"))
train, valid = get_movielens(variant="ml-1m", path=input_path)


# P.s. You can also choose to generate synthetic data to test your models using `generate_data()`. The `input` argument can be either the name of one of the supported public datasets (e.g. "movielens-1m", "criteo") or the schema of a dataset (which is explained next). For example:
# 
# ```python
# from merlin.datasets.synthetic import generate_data
# train, valid = generate_data(input="movielens-1m", num_rows=1000000, set_sizes=(0.8, 0.2))
# ```

# ## Understanding the Schema File and Structure

# When NVTabular process the data, it will persist the schema as a file to disk. You can access the `schema` from the Merlin `Dataset` class (like below).

# The `schema` can be interpreted as a list of features in the dataset, where each element describes metadata of the feature. It contains the name, some properties (e.g. statistics) depending on the feature type and multiple tags. 

# In[4]:


train.schema


# We can select the features by **name**.

# In[5]:


train.schema.select_by_name("userId")


# We can also select features by **tags**. As we described earlier in the notebook, categorical and continuous features are automatically tagged when using ops like `Categorify()` and `Normalize()`.
# In our example preprocessing workflow for this dataset, we also set the `Tags` for the the `user` and `item` features, and also for the `user_id` and `item_id`, which are important for collaborative filtering architectures. 

# Alternatively, we can select them by `Tag`. We add `column_names` to the object to receive only names without all the additional metadata.

# In[6]:


# All categorical features
train.schema.select_by_tag(Tags.CATEGORICAL).column_names


# In[7]:


# All continuous features
train.schema.select_by_tag(Tags.CONTINUOUS).column_names


# In[8]:


# All targets
train.schema.select_by_tag(Tags.TARGET).column_names


# In[9]:


# All features related to the item
train.schema.select_by_tag(Tags.ITEM).column_names


# In[10]:


# The item id feature name
train.schema.select_by_tag(Tags.ITEM_ID).column_names


# In[11]:


# All features related to the user
train.schema.select_by_tag(Tags.USER).column_names


# In[12]:


# The user id feature name
train.schema.select_by_tag(Tags.USER_ID).column_names


# We can also query all properties of a feature. Here we see that the cardinality (number of unique values) of the `movieId` feature is `3682`, which is an important information to build the corresponding embedding table.

# In[13]:


train.schema.select_by_tag(Tags.ITEM_ID)


# The `schema` is a great interface between feature engineering and modeling libraries, describing the available features and their metadata/statistics. It makes it easy to build generic models definition, as the features names and types are automatically inferred from schema and represented properly in the neural networks architectures. That means that when the dataset changes (e.g. features are added or removed), you don't have to change the modeling code to leverage the new dataset!
# 
# For example, the `DLRMModel` embeds categorical features and applies an MLP (called bottom MLP) to combine the continuous features. As another example, The `TwoTowerModel` (for retrieval) builds one MLP tower to combine user features and another MLP tower for the item features, factorizing both towers in the output.

# ## Integrated pipeline with NVTabular and Merlin Models
# 
# Now you have a solid understanding of the importance of the schema and how the schema works. 
# 
# The best way is to use [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular/) for the feature engineering step, so that the schema file is automatically created for you. We will look on a minimal example for the MovieLens dataset.

# ### Download and prepare the data

# We will download the dataset, if it is not already downloaded and cached locally.

# In[14]:


input_path = os.environ.get("INPUT_DATA_DIR", os.path.expanduser("~/merlin-models-data/movielens/"))
name = "ml-1m"
download_file(
    "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
    os.path.join(input_path, "ml-1m.zip"),
    redownload=False,
)


# We preprocess the dataset and split it into training and validation.

# In[15]:


ratings = pd.read_csv(
    os.path.join(input_path, "ml-1m/ratings.dat"),
    sep="::",
    names=["userId", "movieId", "rating", "timestamp"],
)
# Shuffling rows
ratings = ratings.sample(len(ratings), replace=False)

num_valid = int(len(ratings) * 0.2)
train = ratings[:-num_valid]
valid = ratings[-num_valid:]
train.to_parquet(os.path.join(input_path, name, "train.parquet"))
valid.to_parquet(os.path.join(input_path, name, "valid.parquet"))


# ### Feature Engineering and Generating Schema File with NVTabular

# We use NVTabular to define a preprocessing and feature engineering pipeline. 
# 
# NVTabular has already implemented multiple transformations, called `ops` that can be applied to a `ColumnGroup` from an overloaded `>>` operator.<br><br>
# **Example:**<br>
# ```python
# features = [ column_name, ...] >> op1 >> op2 >> ...
# ```
# 
# We need to perform following steps:
# - Categorify userId and movieId, that the values are contiguous integers from 0 ... |C|
# - Transform the rating column ([1,5] interval) to a binary target by using as threshold the value `3`
# - Add Tags with `ops.AddMetadata` for `item_id`, `user_id`, `item`, `user` and `target`.

# Categorify will transform categorical columns into contiguous integers (`0, ..., |C|`) for embedding layers. It collects the cardinality of the embedding table and tags it as categorical.

# In[16]:


cat_features = ["userId", "movieId"] >> ops.Categorify(dtype="int32")


# The tags for `user`, `userId`, `item` and `itemId` cannot be inferred from the dataset. Therefore, we need to provide them manually during the NVTabular workflow. Actually, the `DLRMModel` does not differentiate between `user` and `item` features. But other architectures, such as the `TwoTowerModel` depends on the `user` and `item` features distinction. We will show how to tag features manually in a NVTabular workflow below. 

# In[17]:


feats_itemId = cat_features["movieId"] >> ops.TagAsItemID()
feats_userId = cat_features["userId"] >> ops.TagAsUserID()
feats_target = (
    nvt.ColumnSelector(["rating"])
    >> ops.LambdaOp(lambda col: (col > 3).astype("int32"))
    >> ops.AddTags(["binary_classification", "target"])
    >> nvt.ops.Rename(name="rating_binary")
)
output = feats_itemId + feats_userId + feats_target


# We fit the workflow to our train set and apply to the valid and test sets.

# In[18]:


get_ipython().run_cell_magic('time', '', 'train_path = os.path.join(input_path, name, "train.parquet")\nvalid_path = os.path.join(input_path, name, "valid.parquet")\noutput_path = os.path.join(input_path, name + "_integration")\n\nworkflow_fit_transform(output, train_path, valid_path, output_path)\n')


# ### Training a Recommender Model with Merlin Models

# We can load the data as a Merlin Dataset object. The Dataset expect the schema as Protobuf text format (`.pbtxt`) file in the train/valid folder, which NVTabular automatically generates.

# In[19]:


train = merlin.io.Dataset(
    os.path.join(input_path, name + "_integration", "train"), engine="parquet"
)
valid = merlin.io.Dataset(
    os.path.join(input_path, name + "_integration", "valid"), engine="parquet"
)


# We can see that the `schema` object contains the features tags and the cardinalities of the categorical features.
# As we prepared only a minimal example, our schema has only tree features `movieId`, `userId` and `rating_binary`.|

# In[20]:


train.schema.column_names


# In[21]:


train.schema


# Here we train our model.

# In[22]:


model = mm.DLRMModel(
    train.schema,
    embedding_dim=64,
    bottom_block=mm.MLPBlock([128, 64]),
    top_block=mm.MLPBlock([128, 64, 32]),
    prediction_tasks=mm.BinaryClassificationTask(
        train.schema.select_by_tag(Tags.TARGET).column_names[0]
    ),
)

model.compile(optimizer="adam")
model.fit(train, batch_size=1024)


# Let's run the evaluation on validations set. We use by default typical binary classification metrics -- Precision, Recall, Accuracy and AUC. But you also can provide your own metrics list by setting `BinaryClassificationTask(..., metrics=[])`.

# In[23]:


metrics = model.evaluate(valid, batch_size=1024, return_dict=True)


# In[24]:


metrics


# ## Conclusion
# 
# This example shows the easiness and flexilibity provided by the integration between NVTabular and Merlin Models.
# Feature engineering and model training are depending on each other. The `schema` object is a convient way to provide information from the available features for dynamically setting the model definition. It allows for the modeling code to capture changes in the available features and avoids hardcoding feature names.
# 
# The dataset features are `tagged` automatically (and manually if needed) to group together features, for further modeling usage. 
# 
# The recommended practice is to use `NVTabular` for feature engineering, which generates a `schema` file. NVTabular can automatically add `Tags` for certrain operations. For example, the output of `Categorify` is always a categorical feature and will be tagged. Similar, the output of `Normalize` is always continuous. If you choose to use another preprocessing library, you can create the `schema` file manually, using either the Protobuf text format (`.pbtxt`) or `json` format.
# 
# 
# ## Next Steps
# 
# In the next notebooks, we will explore multiple ranking models with Merlin Models.
# 
# You can learn more about NVTabular, its functionality and supported ops by visiting our [github repository](https://github.com/NVIDIA-Merlin/NVTabular/) or exploring the [examples](https://github.com/NVIDIA-Merlin/NVTabular/tree/main/examples), such as [`Getting Started MovieLens`](https://github.com/NVIDIA-Merlin/NVTabular/blob/main/examples/getting-started-movielens/02-ETL-with-NVTabular.ipynb) or [`Scaling Criteo`](https://github.com/NVIDIA-Merlin/NVTabular/tree/main/examples/scaling-criteo).
