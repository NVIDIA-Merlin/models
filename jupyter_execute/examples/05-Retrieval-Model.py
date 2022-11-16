#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2021 NVIDIA Corporation. All Rights Reserved.
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
# ================================


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_models_05-retrieval-model/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Two-Stage Recommender Systems
# 
# This notebook is created using the latest stable [merlin-tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow/tags) container. 
# 
# In large scale recommender systems pipelines, the size of the item catalog (number of unique items) might be in the order of millions. At such scale, a typical setup is having two-stage pipeline, where a faster candidate retrieval model quickly extracts thousands of relevant items and a then a more powerful ranking model (i.e. with more features and more powerful architecture) ranks the top-k items that are going to be displayed to the user. For ML-based candidate retrieval model, as it needs to quickly score millions of items for a given user, a popular choices are models that can produce recommendation scores by just computing the dot product the user embeddings and item embeddings. Popular choices of such models are **Matrix Factorization**, which learns low-rank user and item embeddings, and the **Two-Tower architecture**, which is a neural network with two MLP towers where both user and item features are fed to generate user and item embeddings in the output.

# 
# ### Dataset
# 
# In this notebook, we are building a Two-Tower model for Item Retrieval task using synthetic datasets that are mimicking the real [Ali-CCP: Alibaba Click and Conversion Prediction](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408#1) dataset.
# ### Learning objectives
# - Preparing the data with NVTabular
# - Training and evaluating Two-Tower model with Merlin Models
# - Exporting the model for deployment

# ### Importing Libraries

# In[2]:


import os

import nvtabular as nvt
from nvtabular.ops import *
from merlin.models.utils.example_utils import workflow_fit_transform

from merlin.schema.tags import Tags

import merlin.models.tf as mm
from merlin.io.dataset import Dataset

import tensorflow as tf


# In[3]:


# disable INFO and DEBUG logging everywhere
import logging

logging.disable(logging.WARNING)


# ### Feature Engineering with NVTabular

# Let's generate synthetic train and validation dataset objects.

# In[4]:


from merlin.datasets.synthetic import generate_data

DATA_FOLDER = os.environ.get("DATA_FOLDER", "/workspace/data/")
NUM_ROWS = os.environ.get("NUM_ROWS", 1000000)
SYNTHETIC_DATA = eval(os.environ.get("SYNTHETIC_DATA", "True"))

if SYNTHETIC_DATA:
    train, valid = generate_data("aliccp-raw", int(NUM_ROWS), set_sizes=(0.7, 0.3))
else:
    train = nvt.Dataset(DATA_FOLDER + "/train/*.parquet")
    valid = nvt.Dataset(DATA_FOLDER + "/valid/*.parquet")


# In[5]:


# define output path for the processed parquet files
output_path = os.path.join(DATA_FOLDER, "processed")


# We keep only positive interactions where clicks==1 in the dataset with `Filter()` op.

# In[6]:


user_id = ["user_id"] >> Categorify() >> TagAsUserID()
item_id = ["item_id"] >> Categorify() >> TagAsItemID()

item_features = ["item_category", "item_shop", "item_brand"] >> Categorify() >> TagAsItemFeatures()

user_features = (
    [
        "user_shops",
        "user_profile",
        "user_group",
        "user_gender",
        "user_age",
        "user_consumption_2",
        "user_is_occupied",
        "user_geography",
        "user_intentions",
        "user_brands",
        "user_categories",
    ]
    >> Categorify()
    >> TagAsUserFeatures()
)

inputs = user_id + item_id + item_features + user_features + ["click"]

outputs = inputs >> Filter(f=lambda df: df["click"] == 1)


# With `transform_aliccp` function, we can execute fit() and transform() on the raw dataset applying the operators defined in the NVTabular workflow pipeline above. The processed parquet files are saved to output_path.

# In[7]:


from merlin.datasets.ecommerce import transform_aliccp

transform_aliccp((train, valid), output_path, nvt_workflow=outputs)


# ## Building a Two-Tower Model with Merlin Models

# We will use Two-Tower Model for item retrieval task. Real-world large scale recommender systems have hundreds of millions of items (products) and users. Thus, these systems often composed of two stages: candidate generation (retrieval) and ranking (scoring the retrieved items). At candidate generation step, a subset of relevant items from large item corpus is retrieved. You can read more about two stage Recommender Systems here. In this example, we're going to focus on the retrieval stage.
# 
# A Two-Tower Model consists of item (candidate) and user (query) encoder towers. With two towers, the model can learn representations (embeddings) for queries and candidates separately. 
# 
# <img src="./images/TwoTower.png"  width="30%">
# 
# Image Adapted from: [Off-policy Learning in Two-stage Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3366423.3380130)

# We use the `schema` object to define our model.

# In[8]:


output_path


# In[9]:


train = Dataset(os.path.join(output_path, "train", "*.parquet"))
valid = Dataset(os.path.join(output_path, "valid", "*.parquet"))


# Select features with user and item tags, and be sure to exclude target column.

# In[10]:


schema = train.schema.select_by_tag([Tags.ITEM_ID, Tags.USER_ID, Tags.ITEM, Tags.USER])
train.schema = schema
valid.schema = schema


# We can print out the feature column names.

# In[11]:


schema.column_names


# We expect the label names to be empty.

# In[12]:


label_names = schema.select_by_tag(Tags.TARGET).column_names
label_names


# ### Negative sampling
# Many datasets for recommender systems contain implicit feedback with logs of user interactions like clicks, add-to-cart, purchases, music listening events, rather than explicit ratings that reflects user preferences over items. To be able to learn from implicit feedback, we use the general (and naive) assumption that the interacted items are more relevant for the user than the non-interacted ones.
# In Merlin Models we provide some scalable negative sampling algorithms for the Item Retrieval Task. In particular, we use in this example the in-batch sampling algorithm which uses the items interacted by other users as negatives within the same mini-batch.

# ### Building the Model

# Now, let's build our Two-Tower model. In a nutshell, we aggregate all user features to feed in user tower and feed the item features to the item tower. Then we compute the positive score by multiplying the user embedding with the item embedding and sample negative items (read more about negative sampling [here](https://openreview.net/pdf?id=824xC-SgWgU) and [here](https://medium.com/mlearning-ai/overview-negative-sampling-on-recommendation-systems-230a051c6cd7)), whose item embeddings are also multiplied by the user embedding. Then we apply the loss function on top of the positive and negative scores.

# In[13]:


model = mm.TwoTowerModel(
    schema,
    query_tower=mm.MLPBlock([128, 64], no_activation_last_layer=True),
    samplers=[mm.InBatchSampler()],
    embedding_options=mm.EmbeddingOptions(infer_embedding_sizes=True),
)


# Let's explain the parameters in the TwoTowerModel():
# - no_activation_last_layer: when set True, no activation is used for top hidden layer. Learn more [here](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/b9f4e78a8830fe5afcf2f0452862fb3c0d6584ea.pdf).
# - infer_embedding_sizes: when set True, automatically defines the embedding dimension from the feature cardinality in the schema
# 
# **Metrics:**
# 
# The following information retrieval metrics are used to compute the Top-10 accuracy of recommendation lists containing all items:
# 
# - **Normalized Discounted Cumulative Gain (NDCG@10)**: NDCG accounts for rank of the relevant item in the recommendation list and is a more fine-grained metric than HR, which only verifies whether the relevant item is among the top-k items.
# 
# - **Recall@10**: Also known as HitRate@n when there is only one relevant item in the recommendation list. Recall just verifies whether the relevant item is among the top-n items.

# We need to initialize the dataloaders.

# In[14]:


model.compile(optimizer="adam", run_eagerly=False, metrics=[mm.RecallAt(10), mm.NDCGAt(10)])
model.fit(train, validation_data=valid, batch_size=4096, epochs=3)


# ## Exporting Retrieval Models

# So far we have trained and evaluated our Retrieval model. Now, the next step is to deploy our model and generate top-K recommendations given a user (query). We can efficiently serve our model by indexing the trained item embeddings into an **Approximate Nearest Neighbors (ANN)** engine. Basically, for a given user query vector, that is generated passing the user features into user tower of retrieval model, we do an ANN search query to find the ids of nearby item vectors, and at serve time, we score user embeddings over all indexed top-K item embeddings within the ANN engine.
# 
# In doing so, we need to export
#  
# - user (query) tower
# - item and user features
# - item embeddings

# #### Save User (query) tower

# We are able to save the user tower model as a TF model to disk. The user tower model is needed to generate a user embedding vector when a user feature vector <i>x</i> is fed into that model.

# In[15]:


query_tower = model.retrieval_block.query_block()
query_tower.save("query_tower")


# #### Extract and save User features

# With `unique_rows_by_features` utility function we can easily extract both unique user and item features tables as cuDF dataframes. Note that for user features table, we use `USER` and `USER_ID` tags.

# In[16]:


from merlin.models.utils.dataset import unique_rows_by_features

user_features = (
    unique_rows_by_features(train, Tags.USER, Tags.USER_ID).compute().reset_index(drop=True)
)


# In[17]:


user_features.head()


# In[18]:


user_features.shape


# In[19]:


# save to disk
user_features.to_parquet("user_features.parquet")


# #### Extract and save Item features

# In[20]:


item_features = (
    unique_rows_by_features(train, Tags.ITEM, Tags.ITEM_ID).compute().reset_index(drop=True)
)


# In[21]:


item_features.head()


# In[22]:


# save to disk
item_features.to_parquet("item_features.parquet")


# #### Extract and save Item embeddings

# In[23]:


item_embs = model.item_embeddings(Dataset(item_features, schema=schema), batch_size=1024)
item_embs_df = item_embs.compute(scheduler="synchronous")


# In[24]:


item_embs_df


# In[25]:


# select only embedding columns
item_embeddings = item_embs_df.iloc[:, 4:]


# In[26]:


item_embeddings.head()


# In[27]:


# save to disk
item_embeddings.to_parquet("item_embeddings.parquet")


# That's it. You have learned how to train and evaluate your Two-Tower retrieval model, and then how to export the required components to be able to deploy this model to generate recommendations. In order to learn more on serving a model to [Triton Inference Server](https://github.com/triton-inference-server/server), please explore the examples in the [Merlin](https://github.com/NVIDIA-Merlin/Merlin) and [Merlin Systems](https://github.com/NVIDIA-Merlin/systems) repos.
