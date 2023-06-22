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

# Each user is responsible for checking the content of datasets and the
# applicable licenses and determining if suitable for the intended use.


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_models_05-retrieval-model/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Building a Retrieval Model with Merlin Models
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
# - Generating the top K recommendations from the trained model

# ### Importing Libraries

# In[2]:


import os

import nvtabular as nvt
from nvtabular.ops import *
from merlin.models.utils.example_utils import workflow_fit_transform

from merlin.schema.tags import Tags

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.models.utils.dataset import unique_rows_by_features

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


train = train.to_ddf().compute()
valid = valid.to_ddf().compute()


# We keep only positive interactions where clicks==1 in the dataset.

# In[6]:


train = train.loc[train['click']==1].reset_index(drop=True)
valid = valid.loc[valid['click']==1].reset_index(drop=True)


# We can drop the target column since in this example we will only use positive interactions and then generate negative samples via negative sampling technique.

# In[7]:


train = train.drop(['click', 'conversion'], axis=1)
valid = valid.drop(['click', 'conversion'], axis=1)


# Create Dataset objects

# In[8]:


train = Dataset(train)
valid = Dataset(valid)


# Define output path for the processed parquet files

# In[9]:


output_path = os.path.join(DATA_FOLDER, "processed")


# In[10]:


category_temp_directory = os.path.join(DATA_FOLDER, "categories")
user_id = ["user_id"] >> Categorify(out_path=category_temp_directory) >> TagAsUserID()
item_id = ["item_id"] >> Categorify(out_path=category_temp_directory) >> TagAsItemID()

item_features = ["item_category", "item_shop", "item_brand"] >> Categorify(out_path=category_temp_directory) >> TagAsItemFeatures()

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
    >> Categorify(out_path=category_temp_directory)
    >> TagAsUserFeatures()
)

outputs = user_id + item_id + item_features + user_features


# With `transform_aliccp` function, we can execute fit() and transform() on the raw dataset applying the operators defined in the NVTabular workflow pipeline above. The processed parquet files are saved to output_path.

# In[11]:


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

# In[12]:


train = Dataset(os.path.join(output_path, "train", "*.parquet"))
valid = Dataset(os.path.join(output_path, "valid", "*.parquet"))


# Select features with user and item tags, and be sure to exclude target column.

# In[13]:


schema = train.schema.select_by_tag([Tags.ITEM_ID, Tags.USER_ID, Tags.ITEM, Tags.USER])
train.schema = schema
valid.schema = schema


# We can print out the feature column names.

# In[14]:


schema


# We expect the label names to be empty.

# In[15]:


label_names = schema.select_by_tag(Tags.TARGET).column_names
label_names


# ### Negative sampling
# 
# Many datasets for recommender systems contain implicit feedback with logs of user interactions like clicks, add-to-cart, purchases, music listening events, rather than explicit ratings that reflects user preferences over items. To be able to learn from implicit feedback, we use the general (and naive) assumption that the interacted items are more relevant for the user than the non-interacted ones.
# In Merlin Models we provide some scalable negative sampling algorithms for the Item Retrieval Task. In particular, in this example, we use the `in-batch` sampling algorithm which uses the items interacted by other users as negatives within the same mini-batch.

# ### Building the Model
# 
# Now, let's build our Two-Tower model. In a nutshell, we aggregate all user features to feed in user tower and feed the item features to the item tower. Then we compute the positive score by multiplying the user embedding with the item embedding and sample negative items (read more about negative sampling [here](https://openreview.net/pdf?id=824xC-SgWgU) and [here](https://medium.com/mlearning-ai/overview-negative-sampling-on-recommendation-systems-230a051c6cd7)), whose item embeddings are also multiplied by the user embedding. Then we apply the loss function on top of the positive and negative scores.

# We make sure that the mlp blocks used for the user and query towers have the same last dimension. This is needed because we will compute the dot product between the two towers' outputs to get the similarity scores.

# In[16]:


tower_dim = 64 

# create user schema using USER tag
user_schema = schema.select_by_tag(Tags.USER)
# create user (query) tower input block
user_inputs = mm.InputBlockV2(user_schema)
# create user (query) encoder block
query = mm.Encoder(user_inputs, mm.MLPBlock([128, tower_dim], no_activation_last_layer=True))

# create item schema using ITEM tag
item_schema = schema.select_by_tag(Tags.ITEM)
# create item (candidate) tower input block
item_inputs = mm.InputBlockV2(item_schema)
# create item (candidate) encoder block
candidate = mm.Encoder(item_inputs, mm.MLPBlock([128, tower_dim], no_activation_last_layer=True))


# `no_activation_last_layer:` when set True, no activation is used for top hidden layer. Learn more [here](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/b9f4e78a8830fe5afcf2f0452862fb3c0d6584ea.pdf).

# Build the model class.

# In[17]:


model = mm.TwoTowerModelV2(query, candidate)


# Note that in the `TwoTowerModelV2` function we did not set `negative_samplers` arg, that means it is set to None. In that case, Two-tower model is trained with contrastive learning and `in-batch` negative sampling strategy.
# 
# **Metrics:**
# 
# The following information retrieval metrics are used to compute the Top-10 accuracy of recommendation lists containing all items:
# 
# - **Normalized Discounted Cumulative Gain (NDCG@10)**: NDCG accounts for rank of the relevant item in the recommendation list and is a more fine-grained metric than HR, which only verifies whether the relevant item is among the top-k items.
# 
# - **Recall@10**: Also known as HitRate@n when there is only one relevant item in the recommendation list. Recall just verifies whether the relevant item is among the top-n items.

# We need to initialize the dataloaders.

# In[18]:


model.compile(optimizer="adam", run_eagerly=False, metrics=[mm.RecallAt(10), mm.NDCGAt(10)])
model.fit(train, validation_data=valid, batch_size=4096, epochs=2)


# The validation metric values are calculated given the positive and negative scores in each batch, and then averaged over batches per epoch. That means validation metrics are not computed using the entire item catalog.

# ### Evaluate the model accuracy

# Note that above when we  set `validation_data=valid` in the `model.fit()`, we compute evaluation metrics on validation set using the negative sampling strategy used for training. To determine the exact accuracy of our trained retrieval model, we need to compute the similarity score between a given query and all possible candidates. The higher the score of the positive candidate (the one that is already interacted with, i.e. target item_id returned by dataloader), the more accurate the model is. We can do this using the `topk_model` model that we create below via `to_top_k_encoder` method, and the following section shows how to instantiate it. The `to_top_k_encoder()` is a method of the [RetrievalModelV2](https://github.com/NVIDIA-Merlin/models/blob/stable/merlin/models/tf/models/base.py) class. 
# 
# `unique_rows_by_features` : A utility function allows extracting both unique user and item features tables as Merlin Dataset object that can easily be converted to a cuDF data frame. The function extracts unique rows from a specified dataset (transformed train set) based on a specified id-column tags (`ITEM` and `ITEM_ID`).

# In[19]:


# Top-K evaluation
candidate_features = unique_rows_by_features(train, Tags.ITEM, Tags.ITEM_ID)
candidate_features.head()


# Below, by using the `topk_model` we can evaluate the trained retrieval model using the entire item catalog. This is applying dot product for entire catalog, and by default it is brute force.

# In[20]:


topk = 20
topk_model = model.to_top_k_encoder(candidate_features, k=topk, batch_size=128)

# we can set `metrics` param in the `compile(), if we want
topk_model.compile(run_eagerly=False)


# In[21]:


eval_loader = mm.Loader(valid, batch_size=1024).map(mm.ToTarget(schema, "item_id"))

metrics = topk_model.evaluate(eval_loader, return_dict=True)
metrics


# ### Generate top-K recommendations

# We trained a model, now we can generate recommendations offline using `to_top_k_encoder` method. The `to_top_k_encoder()` uses the pre-trained candidate and query encoders to initialize a top-k encoder model, called as `topk_model` in this example. Practically, this method applies the candidate_encoder on the provided candidate_features dataset to set the top-k index of the `topk_model`. Therefore, topk_model object is the one responsible of generating the top-k predictions.
# 
# Let's generate top-K (k=20 in our example) recommendations for a given batch of 8 samples. The `to_top_k_encoder()` method uses the candidate (item) features dataset as the identifiers, i.e., we extract the`  candidate_id` arg using `Tags.ITEM_ID` tag by default and set it as `index` when calculating the candidate embeddings. The forward method of `topk_model` takes as the query features as input, and computes the dot product scores between the given query embeddings and all the candidates of the top-k index. Then, it returns the top-k (k=20) item ids with the highest scores. Note that instead of calculating the candidate (item) tower embeddings for each user query, we compute the output of the item tower once and store it in the `TopKEncoder` class  to use for the Top-k index. This is computationally more efficient.

# In[22]:


eval_loader = mm.Loader(valid, batch_size=8, shuffle=False)
batch =next(iter(eval_loader))


# Let's check the `user_id` column in a given batch.

# In[23]:


batch[0]['user_id']


# The recommended top 20 item ids are returned below. The output of the method is a named tuple `TopKPrediction`, where the first element is the dot product scores and the second element is the encoded item ids (not the original ids).

# In[24]:


topk_model(batch[0])


# ## Exporting Retrieval Models

# So far we have trained and evaluated our Retrieval model. Now, the next step is to deploy our model and generate top-K recommendations given a user (query). We can efficiently serve our model by indexing the trained item embeddings into an **Approximate Nearest Neighbors (ANN)** engine. Basically, for a given user query vector, that is generated passing the user features into user tower of retrieval model, we do an ANN search query to find the ids of nearby item vectors, and at serve time, we score user embeddings over all indexed top-K item embeddings within the ANN engine.
# 
# In doing so, we need to export
#  
# - user (query) tower
# - item and user features
# - item embeddings

# #### Save and Load User (query) tower

# We are able to save the user tower model as a TF model to disk. The user tower model is needed to generate a user embedding vector when a user feature vector <i>x</i> is fed into that model.

# In[25]:


query_tower = model.query_encoder
query_tower.save(os.path.join(DATA_FOLDER, "query_tower"))

## we can load back the saved model via the following script.
#query_tower_loaded = tf.keras.models.load_model(os.path.join(DATA_FOLDER, 'query_tower'))


# #### Extract and save User features

# With `unique_rows_by_features` utility function we can easily extract both unique user and item features tables as cuDF dataframes. Note that for user features table, we use `USER` and `USER_ID` tags.

# In[26]:


user_features = (
    unique_rows_by_features(train, Tags.USER, Tags.USER_ID).compute().reset_index(drop=True)
)


# In[27]:


user_features.head()


# In[28]:


# save to disk
user_features.to_parquet(os.path.join(DATA_FOLDER, "user_features.parquet"))


# #### Generate Query embeddings for entire user catalog

# In[29]:


queries = model.query_embeddings(Dataset(user_features, schema=schema), batch_size=1024, index=Tags.USER_ID)
query_embs_df = queries.compute(scheduler="synchronous").reset_index()


# In[30]:


query_embs_df.head()


# #### Extract and save Item features

# In[31]:


item_features = (
    unique_rows_by_features(train, Tags.ITEM, Tags.ITEM_ID).compute().reset_index(drop=True)
)


# In[32]:


item_features.head()


# In[33]:


# save to disk
item_features.to_parquet(os.path.join(DATA_FOLDER, "item_features.parquet"))


# #### Extract and save Item embeddings

# In[34]:


item_embs = model.candidate_embeddings(Dataset(item_features, schema=schema), batch_size=1024, index=Tags.ITEM_ID)


# In[35]:


item_embs_df = item_embs.compute(scheduler="synchronous")


# In[36]:


item_embs_df


# In[37]:


# save to disk
item_embs_df.to_parquet(os.path.join(DATA_FOLDER, "item_embeddings.parquet"))


# That's it. You have learned how to train and evaluate your Two-Tower retrieval model, and then how to export the required components to be able to deploy this model to generate recommendations. In order to learn more on serving a model to [Triton Inference Server](https://github.com/triton-inference-server/server), please explore the examples in the [Merlin](https://github.com/NVIDIA-Merlin/Merlin) and [Merlin Systems](https://github.com/NVIDIA-Merlin/systems) repos.
