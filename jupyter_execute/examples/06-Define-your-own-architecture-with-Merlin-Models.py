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


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_models_06-define-your-own-architecture-with-merlin-models/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Taking the Next Step with Merlin Models: Define Your Own Architecture
# 
# This notebook is created using the latest stable [merlin-tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow/tags) container. 
# 
# In [Iterating over Deep Learning Models using Merlin Models](https://nvidia-merlin.github.io/models/main/examples/03-Exploring-different-models.html), we conducted a benchmark of standard and deep learning-based ranking models provided by the high-level Merlin Models API. The library also includes the standard components of deep learning that let recsys practitioners and researchers to define custom models, train and export them for inference.
# 
# 
# In this example, we combine pre-existing blocks and demonstrate how to create the [DLRM](https://arxiv.org/abs/1906.00091) architecture.
# 
# 
# ### Learning objectives
# - Understand the building blocks of Merlin Models
# - Define a model architecture from scratch

# ### Introduction to Merlin-models core building blocks

# The [Block](https://nvidia-merlin.github.io/models/review/pr-294/generated/merlin.models.tf.Block.html#merlin.models.tf.Block) is the core abstraction in Merlin Models and is the class from which all blocks inherit.
# The class extends the [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) base class and implements a number of properties that simplify the creation of custom blocks and models. These properties include the `Schema` object for determining the embedding dimensions, input shapes, and output shapes. Additionally, the `Block` has a `ModelContext` instance to store and retrieve public variables and share them with other blocks in the same model as additional meta-data. 
# 
# Before deep-diving into the definition of the DLRM architecture, let's start by listing the core components you need to know to define a model from scratch:

# #### Features Blocks

# They include input blocks to process various inputs based on their types and shapes. Merlin Models supports three main blocks: 
# - `EmbeddingFeatures`: Input block for embedding-lookups for categorical features.
# - `SequenceEmbeddingFeatures`: Input block for embedding-lookups for sequential categorical features (3D tensors).
# - `ContinuousFeatures`: Input block for continuous features.

# #### Transformations Blocks

# They include various operators commonly used to transform tensors in various parts of the model, such as: 
# 
# - `ToDense`: It takes a dictionary of raw input tensors and transforms the sparse tensors into dense tensors.
# - `L2Norm`: It takes a single or a dictionary of hidden tensors and applies an L2-normalization along a given axis. 
# - `LogitsTemperatureScaler`: It scales the output tensor of predicted logits to lower the model's confidence. 

# #### Aggregations Blocks

# They include common aggregation operations to combine multiple tensors, such as:
# - `ConcatFeatures`: Concatenate dictionary of tensors along a given dimension.
# - `StackFeatures`: Stack dictionary of tensors along a given dimension.
# - `CosineSimilarity`: Calculate the cosine similarity between two tensors. 
# 

# #### Connects Methods

# The base class `Block` implements different connects methods that control how to link a given block to other blocks: 
# 
# - `connect`: Connect the block to other blocks sequentially. The output is a tensor returned by the last block. 
# - `connect_branch`: Link the block to other blocks in parallel. The output is a dictionary containing the output tensor of each block.
# - `connect_with_shortcut`: Connect the block to other blocks sequentially and apply a skip connection with the block's output. 
# - `connect_with_residual`: Connect the block to other blocks sequentially and apply a residual sum with the block's output.

# #### Prediction Tasks

# Merlin Models introduces the `PredictionTask` layer that defines the necessary blocks and transformation operations to compute the final prediction scores. It also provides the default loss and metrics related to the given prediction task.\
# Merlin Models supports the core tasks:  `BinaryClassificationTask`, `MultiClassClassificationTask`, and`RegressionTask`. In addition to the preceding tasks, Merlin Models provides tasks that are specific to recommender systems: `NextItemPredictionTask`, and `ItemRetrievalTask`.
# 
# 
# 

# ### Implement the DLRM model with MovieLens-1M data

# Now that we have introduced the core blocks of Merlin Models, let's take a look at how we can combine them to define the DLRM architecture:

# In[2]:


import tensorflow as tf
import merlin.models.tf as mm

from merlin.datasets.entertainment import get_movielens
from merlin.schema.tags import Tags


# We use the `get_movielens` function to download, extract, and preprocess the MovieLens 1M  dataset:

# In[3]:


train, valid = get_movielens(variant="ml-1m")


# We display the first five rows of the validation data and use them to check the outputs of each building block: 

# In[4]:


valid.head()


# We convert the first five rows of the `valid` dataset to a batch of input tensors:  

# In[5]:


batch = mm.sample_batch(valid, batch_size=5, shuffle=False, include_targets=False)
batch["userId"]


# #### Define the inputs block

# For the sake of simplicity, let's create a schema with a subset of the following continuous and categorical features: 

# In[6]:


sub_schema = train.schema.select_by_name(
    [
        "userId",
        "movieId",
        "title",
        "gender",
        "TE_zipcode_rating",
        "TE_movieId_rating",
        "rating_binary",
    ]
)


# We define the continuous layer based on the schema:

# In[7]:


continuous_block = mm.ContinuousFeatures.from_schema(sub_schema, tags=Tags.CONTINUOUS)


# We display the output tensor of the continuous block by using the data from the first batch. We can see the raw tensors of the continuous features:

# In[8]:


continuous_block(batch)


# We connect the continuous block to a `MLPBlock` instance to project them into the same dimensionality as the embedding width of categorical features:

# In[9]:


deep_continuous_block = continuous_block.connect(mm.MLPBlock([64]))
deep_continuous_block(batch).shape


# We define the categorical embedding block based on the schema:

# In[10]:


embedding_block = mm.EmbeddingFeatures.from_schema(sub_schema)


# We display the output tensor of the categorical embedding block using the data from the first batch. We can see the embeddings tensors of categorical features with a default dimension of 64:

# In[11]:


embeddings = embedding_block(batch)
embeddings.keys(), embeddings["userId"].shape


# Let's store the continuous and categorical representations in a single dictionary using a `ParallelBlock` instance:

# In[12]:


dlrm_input_block = mm.ParallelBlock(
    {"embeddings": embedding_block, "deep_continuous": deep_continuous_block}
)
print("Output shapes of DLRM input block:")
for key, val in dlrm_input_block(batch).items():
    print("\t%s : %s" % (key, val.shape))


# By looking at the output, we can see that the `ParallelBlock` class applies embedding and continuous blocks, in parallel, to the same input batch.  Additionally, it merges the resulting tensors into one dictionary.

# #### Define the interaction block

# Now that we have a vector representation of each input feature, we will create the DLRM interaction block. It consists of three operations: 
# - Apply a dot product between all continuous and categorical features to learn pairwise interactions. 
# - Concat the resulting pairwise interaction with the deep representation of conitnuous features (skip-connection). 
# - Apply an `MLPBlock` with a series of dense layers to the concatenated tensor. 

# First, we use the `connect_with_shortcut` method to create first two operations of the DLRM interaction block:

# In[13]:


from merlin.models.tf.blocks.dlrm import DotProductInteractionBlock

dlrm_interaction = dlrm_input_block.connect_with_shortcut(
    DotProductInteractionBlock(), shortcut_filter=mm.Filter("deep_continuous"), aggregation="concat"
)


# The `Filter` operation allows us to select the `deep_continuous` tensor from the `dlrm_input_block` outputs. 

# The following diagram provides a visualization of the operations that we constructed in the `dlrm_interaction` object.
# 
# <img src="./images/residual_interaction.png"  width="30%">
# 

# In[14]:


dlrm_interaction(batch)


# Then, we project the learned interaction using a series of dense layers:

# In[15]:


deep_dlrm_interaction = dlrm_interaction.connect(mm.MLPBlock([64, 128, 512]))
deep_dlrm_interaction(batch)


# #### Define the Prediction block

# At this stage, we have created the DLRM block that accepts a dictionary of categorical and continuous tensors as input. The output of this block is the interaction representation vector of shape `512`. The next step is to use this hidden representation to conduct a given prediction task. In our case, we use the label `rating_binary` and the objective is: to predict if a user `A` will give a high rating to a movie `B` or not. 

# We use the `BinaryClassificationTask` class and evaluate the performances using the `AUC` metric. We also use the `LogitsTemperatureScaler` block as a pre-transformation operation that scales the logits returned by the task before computing the loss and metrics:

# In[16]:


from merlin.models.tf.transforms.bias import LogitsTemperatureScaler

binary_task = mm.BinaryClassificationTask(
    sub_schema,
    pre=LogitsTemperatureScaler(temperature=2),
)


# #### Define, train, and evaluate the final DLRM Model

# We connect the deep DLRM interaction to the binary task and the method automatically generates the `Model` class for us.
# We note that the `Model` class inherits from [tf.keras.Model](https://keras.io/api/models/model/) class:

# In[17]:


model = mm.Model(deep_dlrm_interaction, binary_task)
type(model)


# We train the model using the built-in tf.keras `fit` method: 

# In[18]:


model.compile(optimizer="adam", metrics=[tf.keras.metrics.AUC()])
model.fit(train, batch_size=1024, epochs=1)


# Let's check out the model evaluation scores:

# In[19]:


metrics = model.evaluate(valid, batch_size=1024, return_dict=True)
metrics


# Note that the `evaluate()` progress bar shows the loss score for every batch, whereas the final loss stored in the dictionary represents the total loss across all batches. 

# Save the model so we can use it for serving predictions in production or for resuming training with new observations:

# In[20]:


model.save("custom_dlrm")


# ## Conclusion 
# 
# Merlin Models provides common and state-of-the-art RecSys architectures in a high-level API as well as all the required low-level building blocks for you to create your own architecture (input blocks, MLP layers, prediction tasks, loss functions, etc.). In this example, we explored a subset of these pre-existing blocks to create the DLRM model, but you can view our [documentation](https://nvidia-merlin.github.io/models/main/) to discover more. You can also [contribute](https://github.com/NVIDIA-Merlin/models/blob/main/CONTRIBUTING.md) to the library by submitting new RecSys architectures and custom building Blocks.  
# 
# 
# 
# ## Next steps
# To learn more about how to deploy the trained DLRM model, please visit [Merlin Systems](https://github.com/NVIDIA-Merlin/systems) library and execute the `Serving-Ranking-Models-With-Merlin-Systems.ipynb` notebook that deploys an ensemble of a [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) Workflow and a trained model from Merlin Models to [Triton Inference Server](https://github.com/triton-inference-server/server). 
# 
# 
# 
