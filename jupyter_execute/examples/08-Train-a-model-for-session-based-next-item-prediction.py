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
# ==============================================================================

# Each user is responsible for checking the content of datasets and the
# applicable licenses and determining if suitable for the intended use.


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_models_ecommerce-session-based-next-item-prediction-for-fashion/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Session-Based Next Item Prediction for Fashion E-Commerce
# 
# This notebook is created using the latest stable [merlin-tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow/tags) container. 
# 
# ## Overview
# 
# NVIDIA-Merlin team participated in [Recsys2022 challenge](http://www.recsyschallenge.com/2022/index.html) and secured 3rd position. This notebook contains the various techniques used in the solution.
# 
# In this notebook we train several different architectures with the last one being a transformer model. We only cover training. If you would be interested also in putting your model in production and serving predictions using the industry standard Triton Inference Server, please consult [this notebook](https://github.com/NVIDIA-Merlin/Merlin/blob/main/examples/Next-Item-Prediction-with-Transformers/tf/transformers-next-item-prediction.ipynb).
# 
# ### Learning Objective
# 
# In this notebook, we will apply important concepts that improve recommender systems. We leveraged them for our RecSys solution:
# - MultiClass next item prediction head with Merlin Models
# - Sequential input features representing user sessions
# - Label Smoothing 
# - Temperature Scaling
# - Weight Tying
# - Learning Rate Scheduler
# 
# ### Brief Description of the Concepts
# 
# **Label smoothing**
# 
# In recommender systems, we often have noisy datasets. A user cannot view all items to make the best decision. Noisy examples can result in high gradients and confuse the model. Label smoothing addresses the problem of noisy examples by smoothing the porbabilities to avoid high confident predictions.
# 
# $$  \begin{array}{l}
# y_{l} \ =\ ( 1\ -\ \alpha \ ) \ *\ y_{o} \ +\ ( \alpha \ /\ L)\\
# \alpha :\ Label\ smoothing\ hyper-parameter\ ( 0 \leq \alpha \leq 1 ) \\
# L:\ Total\ number\ of\ label\ classes\\
# y_{o} :\ One-hot\ encoded\ label\ vector
# \end{array}
# $$
# 
# When α is 0, we have the original one-hot encoded labels, and as α increases, we move towards smoothed labels. Read [this](https://arxiv.org/abs/1906.02629) paper to learn more about it.
# 
# 
# **Temperature Scaling**
# 
# Similar to Label Smoothing, Temperature Scaling is done to reduce the overconfidence of a model. In this, we divide the logits (inputs to the softmax function) by a scalar parameter (T) . For more information on Temperature Scaling read [this](https://arxiv.org/pdf/1706.04599.pdf) paper.
# $$ softmax\ =\ \frac{e\ ^{( z_{i} \ /\ \ T)}}{\sum _{j} \ e^{( z_{j} \ /\ T)} \ } $$
# 
# 
# **Weight-tying**
# 
# Weight-tying can be applied for Multi-Class Classification problems, when we try to predict items and have previous viewed items as an input. The final output layer (without activation function) is multiplied with the same item embeddings table to represent the input items, resulting in a vector with a logit for each item id. The advantage is that the gradients flow to the item embeddings are short. For more information read [this](https://arxiv.org/pdf/1608.05859v3.pdf) paper.

# ## Downloading and preparing the dataset

# We will import the required libraries.

# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
import glob
import gc

import nvtabular as nvt
from merlin.io import Dataset
from merlin.schema import Schema, Tags
from nvtabular.ops import (
    AddMetadata,
)

from tensorflow.keras import regularizers

import merlin.models.tf as mm
from merlin.models.tf import InputBlock
from merlin.models.tf.models.base import Model
from merlin.models.tf.transforms.bias import LogitsTemperatureScaler
from merlin.models.tf.prediction_tasks.next_item import ItemsPredictionWeightTying

from merlin.core.dispatch import get_lib


# ###  Dressipi
# [Dressipi](http://www.recsyschallenge.com/2022/dataset.html) hosted the [Recsys2022 challenge](http://www.recsyschallenge.com/2022/index.html) and provided an anonymized dataset. It contains 1.1 M online retail sessions that resulted in a purchase. It provides details about items that were viewed in a session, the item purchased at the end of the session and numerous features of those items. The item features are categorical IDs and are not interpretable.
# 
# The task of this competition was, given a sequence of items predict which item will be purchased at the end of a session.
# 
# <img src="http://www.recsyschallenge.com/2022/images/session_purchase_data.jpeg" alt="dressipi_dataset" style="width: 400px; float: center;">  
# 

# ### Dataset
# 
# We provide a function `get_dressipi2022` which preprocess the dataset. Currently, we can't download this dataset automatically so this needs to be downloaded manually. To use this function, prepare the data by following these 3 steps:
# 1. Sign up and download the data from [dressipi-recsys2022.com](https://www.dressipi-recsys2022.com/).
# 2. Unzip the raw data to a directory.
# 3. Define `DATA_FOLDER` to the directory
# 
# In case you do not want to use this dataset to run our examples, you can also opt for synthetic data. Synthetic data can be generated by running::
# 
# ```python
#     from merlin.datasets.synthetic import generate_data
#     train, valid = generate_data("dressipi2022-preprocessed", num_rows=10000, set_sizes=(0.8, 0.2))
# ```

# In[3]:


from merlin.datasets.ecommerce import get_dressipi2022

DATA_FOLDER = os.environ.get(
    "DATA_FOLDER", 
    '/workspace/data/dressipi_recsys2022'
)

train, valid = get_dressipi2022(DATA_FOLDER)


# The dataset contains:
# - `session_id`, id of a session, in which a user viewed and purchased an item. 
# - `item_id` which was viewed at a given `timestamp` in a session
# - `purchase_id` which is the id of item bought at the end of the session 
# 
# In addition to `timestamp`, we have `day` and `date` features for representing the chronological order in which items were viewed.
# 
# The items in the Dresspi dataset had a many features out of which we took 22 most important features, namely 
# `f_3 ,f_4 ,f_5 ,f_7 ,f_17 ,f_24 ,f_30 ,f_45 ,f_46 ,f_47 ,f_50 ,f_53 ,f_55 ,f_56 ,f_58 ,f_61 ,f_63 ,f_65 ,f_68 ,f_69 ,f_72 ,f_73`.

# In[4]:


train.to_ddf().head()


# ## Feature Engineering with NVTabular
# 
# We use NVTabular for Feature Engineering. If you want to learn more about NVTabular, we recommend the [examples in the NVTabular GitHub Repository](https://github.com/NVIDIA-Merlin/NVTabular/tree/stable/examples).

# ### Categorify
# 
# We want to use embedding layers for our categorical features. First, we need to Categorify them, that they are contiguous integers. 
# 
# The features `item_id` and `purchase_id` belongs to the same category. If `item_id` is 8432 and `purchase_id` is 8432, they are the same item. When we want to apply Categorify, we want to keep the connection. We can achieve this by encoding them jointly by providing them as a list in the list `[['item_id', 'purchase_id']]`.

# We will use only 2 of the categorical item features in this example.

# In[5]:


get_ipython().run_cell_magic('time', '', "item_features_names = ['f_' + str(col) for col in [47, 68]]\ncat_features = [['item_id', 'purchase_id']] + item_features_names >> nvt.ops.Categorify()\n\nfeatures = ['session_id', 'timestamp'] + cat_features\n")


# ### GroupBy the data by sessions.
# 
# Currently, every row is a viewed item in the dataset. Our goal is to predict the item purchased after the last view in a session. Therefore, we groupby the dataset by `session_id` to have one row for each prediction.
# 
# Each row will have a sequence of encoded items ids with which a user interacted. The last item of a session has special importance as it is closer to the user's intention. We will keep the viewed item as a separate feature.
# 
# The NVTabular `GroupBy` op enables the transformation. 
# 
# First, we define how the different columns should be aggregates:
# - Keep the first occurrence of `timestamp`
# - Keep the last item and concatenate all items to a list (results are 2 features)
# - Keep the first occurrence of `purchase_id` (purchase_id should be the same for all rows of one session)

# In[6]:


to_aggregate = {}
to_aggregate['timestamp'] = ["first"]
to_aggregate['item_id'] = ["last", "list"]
to_aggregate['purchase_id'] = ["first"]   


# In addition, we concatenate each item features to a list.

# In[7]:


for name in item_features_names: 
    to_aggregate[name] = ['list']


# In[8]:


to_aggregate


# We want to sort the dataframe by `timestamp` and groupby the columns by `session_id`.

# In[9]:


groupby_features = features >> nvt.ops.Groupby(
    groupby_cols=["session_id"], 
    sort_cols=["timestamp"],
    aggs= to_aggregate,
    name_sep="_")


# Merlin Models can infer the neural network architecture from the dataset schema. We will Tag the columns accordingly based on the type of each column. If you want to learn more, we recommend our [Dataset Schema Example](https://github.com/NVIDIA-Merlin/models/blob/stable/examples/02-Merlin-Models-and-NVTabular-integration.ipynb).

# In[10]:


item_last = (
    groupby_features['item_id_last'] >> 
    AddMetadata(tags=[Tags.ITEM, Tags.ITEM_ID])
)
item_list = (
    groupby_features['item_id_list'] >> 
    AddMetadata(
        tags=[Tags.ITEM, Tags.ITEM_ID, Tags.LIST, Tags.SEQUENCE]
    )
)
feature_list = (
    groupby_features[[name+'_list' for name in item_features_names]] >> 
    AddMetadata(
        tags=[Tags.SEQUENCE, Tags.ITEM, Tags.LIST]
    )
)

other_features = groupby_features['session_id', 'timestamp_first']

groupby_features = item_last + item_list + feature_list + other_features +  groupby_features['purchase_id_first']


# ### Truncate for a Maximum Sequence Length
# 
# We want to truncate and pad the sequential features. We define the columns, which are sequential features and the non-sequential ones. We truncate the sequence by keeping the last 3 elements.

# In[11]:


list_features = [name+'_list' for name in item_features_names] + ['item_id_list']
nonlist_features = ['session_id', 'timestamp_first', 'item_id_last', 'purchase_id_first']


# In[12]:


SESSIONS_MAX_LENGTH = 3
truncated_features = (
    groupby_features[list_features] 
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH) 
    >> nvt.ops.Rename(postfix = '_seq')
    >> nvt.ops.ValueCount()
)

final_features = groupby_features[nonlist_features] + truncated_features


# We initialize our NVTabular workflow.

# In[13]:


workflow = nvt.Workflow(final_features)


# We call fit and transform similar to the scikit learn API.
# 
# Categorify will map item_ids (and purchase_ids), which does not occur in the train dataset, to a special category `0` in the validation dataset. This can bias the validation metrics. In our example, almost all item_ids in validation are available in train and we neglect it.

# In[14]:


# fit data
workflow.fit(train)

# transform and save data
workflow.transform(train).to_parquet(os.path.join(DATA_FOLDER, "train/"), output_files=2)
workflow.transform(valid).to_parquet(os.path.join(DATA_FOLDER, "valid/"))


# ### Sort the Training Dataset by Time
# 
# The train dataset contains the data from Jan 2020 to April 2021 and the validation dataset is May 2021. As the data is split by time, we noticed that we achieve higher validation scores, when we sort the training data by time and do not apply shuffling.

# In[15]:


df = get_lib().read_parquet(
    glob.glob(
        os.path.join(DATA_FOLDER, "train/*.parquet")
    )
)
df = df.sort_values('timestamp_first').reset_index(drop=True)
df.to_parquet(os.path.join(DATA_FOLDER, "train_sorted.parquet"))


# Let's review the transformed dataset.

# In[16]:


df.head()


# ## Training an MLP with sequential input with Merlin Models
# 
# We train a Sequential-Multi-Layer Perceptron model, which averages the sequential input features (e.g. `item_id_list_seq`) and concatenate the resulting embeddings with the categorical embeddings (e.g. `item_id_last`). We visualize the architecture in the figure below.
# 
# <img src="images/mlp_ecommerce.png"  width="30%">

# ### Dataloader
# 
# We initialize the dataloaders to train the neural network models. First, we define NVTabular dataset.

# In[17]:


train = Dataset(os.path.join(DATA_FOLDER, 'train_sorted.parquet'))
valid = Dataset(os.path.join(DATA_FOLDER, 'valid/*.parquet'))


# As we loaded, sorted and saved the train dataset without using NVTabular, the parquet file doesn't contain a schema, anymore. We can copy the schema from valid to train.

# In[18]:


valid.schema = valid.schema.select_by_name(
    ['item_id_list_seq', 'item_id_last','f_47_list_seq', 'f_68_list_seq', 'purchase_id_first']
)
train.schema = valid.schema
schema_model = train.schema.select_by_name(
        ['item_id_list_seq', 'item_id_last','f_47_list_seq', 'f_68_list_seq']
)


# #### Hyperparameters
# 
# We use the following hyperparameters that we found during experimentations.

# In[19]:


EPOCHS = int(os.environ.get(
    "EPOCHS", 
    '3'
))

dmodel = int(os.environ.get(
    "dmodel", 
    '256'
))

BATCH_SIZE = 1024
LEARNING_RATE = 0.005
DROPOUT = 0.2 
LABEL_SMOOTHING = 0.2
TEMPERATURE_SCALING = 2


# #### Data loader

# The default dataloader does shuffle by default. We will initialize the Loader for the training dataset, and set the shuffle to `False`.

# In[20]:


loader = mm.Loader(train, batch_size=BATCH_SIZE, shuffle=False).map(mm.ToTarget(train.schema, "purchase_id_first", one_hot=True))
val_loader = mm.Loader(valid, batch_size=BATCH_SIZE, shuffle=False).map(mm.ToTarget(train.schema, "purchase_id_first", one_hot=True))


# ### Build the Sequential MLP with Merlin Models
# 
# Now we will create an InputBlock which takes sequential features, concatenate them and return the sequence of interaction embeddings. Note that we define the embedding dimensions, manually.

# In[21]:


manual_dims = {
    'item_id_list_seq': dmodel, 
    'item_id_last': dmodel,
    'f_47_list_seq': 16,
    'f_68_list_seq': 16
}


# In[22]:


input_block = mm.InputBlockV2(
        schema_model,
        categorical=mm.Embeddings(schema_model,
                                 dim=manual_dims
                                )
)


# Before the loss is calculated, we want to transform the model output:
# 
# 1. We apply `weight-tying` and multiply the model output with the embedding weights from ITEM_ID. The embedding dimensions and the model output dimensions have to be the same (256 in our example).
# 2. We transform the ground truth into OneHot representation
# 3. We apply Temperature Scaling

# Now, we will build a model with a 2-layer MLPBlock, `input_block` as the input and `prediction_task` as the task. The output dimension of MLPBlock should match with the embedding dimension of the `item_id_list_seq` since we are using weight tying technique.

# In[23]:


mlp_block = mm.MLPBlock(
        [128, dmodel], 
        no_activation_last_layer=True, 
        dropout=0.2
    )


# In[24]:


item_id_name = train.schema.select_by_tag(Tags.ITEM_ID).first.properties['domain']['name']
item_id_name


# Next, we define the prediction task. Our objective is multi-class classification - which is the item purchased at the end of the session. Therefore, this is a multi-class classification task, and the default_loss in the `CategoricalOutput` class is  "categorical_crossentropy". [CategoricalOutput](https://github.com/NVIDIA-Merlin/models/blob/stable/merlin/models/tf/outputs/classification.py#L112) class has the functionality to do `weight-tying`, when we provide the `EmbeddingTable` related to the target feature in the `to_call` method. Note that in our example we feed the embedding table for the `item_id_purchase_id` domain name, since it reflects the fact that the `item_id_list_seq` and `item_id_last` input columns were jointly encoded and they share the same embedding table.

# In[25]:


prediction_task= mm.CategoricalOutput(
        to_call=input_block["categorical"][item_id_name],
        logits_temperature=TEMPERATURE_SCALING,
        target='purchase_id_first',
    )


# In[26]:


model_mlp = mm.Model(input_block, mlp_block, prediction_task)


# ### Fit the Model

# 
# We initialize the optimizer with `ExponentialDecay` learning rate scheduler and compile the model - similar to other TensorFlow Keras API. The competition was evaluated based on MRR@100.

# In[27]:


initial_learning_rate = LEARNING_RATE

exp_decay_lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)


# In[28]:


optimizer = tf.keras.optimizers.Adam(
    learning_rate=exp_decay_lr_scheduler,
)

model_mlp.compile(
    optimizer=optimizer,
    run_eagerly=False,
    loss=tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, 
        label_smoothing=LABEL_SMOOTHING
    ),
    metrics=mm.TopKMetricsAggregator.default_metrics(top_ks=[100])
)


# We call `.fit` to train the model.

# In[29]:


get_ipython().run_cell_magic('time', '', 'history = model_mlp.fit(\n    loader,\n    validation_data=val_loader,\n    epochs=EPOCHS,\n)\n')


# We can evaluate the model.

# In[30]:


metrics_mlp = model_mlp.evaluate(val_loader, batch_size=BATCH_SIZE, return_dict=True)
metrics_mlp['mrr_at_100']


# Note that final score `mrr_at_100` we printed out above is the average over all steps (batches), whereas the value we get from progress bar shows the score of the last batch.

# In[31]:


del model_mlp
gc.collect()


# ## Training a Bi-LSTM with Merlin Models

# In this section, we train a Bi-LSTM model, an extension of traditional LSTMs, which enables straight (past) and reverse traversal of input (future) sequence to be used. The input block concatenates the embedding vectors for all sequential features (`item_id_list_seq`, `f_47_list_seq`, `f_68_list_seq`) per step (e.g. here 3). The concatenated vectors are processed by a BiLSTM architecture. The hidden state of the BiLSTM is concatenated with the embedding vectors of the categorical features (`item_id_last`). Then we connect it with a Multi-Layer Perceptron Block. We visualize the architecture in the figure below.
# 
# <img src="images/bi-lstm_ecommerce.png"  width="30%">

# ### Build Bi-LSTM model
# 
# Now we will create a Bi-LSTM model by using [tf.keras.layers.Bidirectional](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional) api. We connect the sequence input block for sequential features with `BiLSTM` block via `connect` method. First, we  create two input blocks which takes sequential and categorical features, respectively, concatenate them and return the interaction embeddings.

# We define the embedding dimensions, manually.

# In[32]:


seq_inputs = mm.InputBlockV2(
        schema_model.select_by_tag(Tags.SEQUENCE),
        categorical=mm.Embeddings(
            schema_model.select_by_tag(Tags.SEQUENCE),
            sequence_combiner=None,
            dim=manual_dims                 
        )
)

cat_inputs = mm.InputBlockV2(
        schema_model.select_by_name(['item_id_last']),
        categorical=mm.Embeddings(
            schema_model.select_by_name(['item_id_last']),
            dim=manual_dims                        
        )
)


# The argument `sequence_combiner` is used  for combining embeddings of each positions of the same input feature. In other words, it is a string specifying how to combine embedding results for each entry. Here we set it to None, since, we do not want to combine the embeddings for each position in a given sequence.

# Connect the sequential input block to the BiLSTM model. We leverage `tf.keras.layers.Bidirectional` api.

# In[33]:


dense_block =seq_inputs.connect(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(64,
        return_sequences=False, 
        dropout=0.05,
        kernel_regularizer=regularizers.l2(1e-4),
    )
))


# Now, we combine dense block with input block of categorical features by concatenating them.

# In[34]:


concats = mm.ParallelBlock(
    {'dense_block': dense_block, 
     'cat_inputs': cat_inputs},
    aggregation='concat'
)


# Next, we build a 2-layer MLPBlock which is used as a projection layer.

# In[35]:


mlp_block = mm.MLPBlock(
                [128,dmodel],
                activation='relu',
                no_activation_last_layer=True,
                dropout=DROPOUT,
            )


# Now, we define the prediction task. As we saw above in our MLP implementation, our objective is multi-class classification.
# 
# Before the loss is calculated, we want to transform the model output:
# 
# - We apply  `weight-tying` (see in the beginning) and multiply the model output with the embedding weights from ITEM_ID. The embedding dimensions and the model output dimensions have to be the same (256 in our example).
# - We transform the ground truth into OneHot representation
# - We apply Temperature Scaling
# 
# The pipeline is executed before the loss is calculated.

# In[36]:


item_id_name = train.schema.select_by_tag(Tags.ITEM_ID).first.properties['domain']['name']


# In[37]:


prediction_task= mm.CategoricalOutput(
    to_call=seq_inputs["categorical"][item_id_name],
    logits_temperature=TEMPERATURE_SCALING,
    target='purchase_id_first',
)


# Now, we will build a model by chaining the `concats`, the `mlp_block` and the `prediction_task` layers.

# In[38]:


model_bi_lstm = Model(concats, mlp_block, prediction_task)


# ### Fit the Model
# We initialize the optimizer and compile the model - similar to other TensorFlow Keras API. The competition was evaluated based on MRR@100.

# In[39]:


optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model_bi_lstm.compile(
    optimizer=optimizer,
    run_eagerly=False,
    loss=tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, 
        label_smoothing=LABEL_SMOOTHING
    ),
    metrics=mm.TopKMetricsAggregator.default_metrics(top_ks=[100])
)


# Now, we train the model

# In[40]:


history = model_bi_lstm.fit(
    loader,
    validation_data=val_loader,
    epochs=EPOCHS,
)


# We can evaluate the model

# In[41]:


metrics_bi_lstm = model_bi_lstm.evaluate(val_loader, batch_size=BATCH_SIZE, return_dict=True)
metrics_bi_lstm['mrr_at_100']


# In[42]:


del model_bi_lstm
gc.collect()


# ## Training a Transformer-based Model
# 
# In recent years, several deep learning-based algorithms have been proposed for recommendation systems while its adoption in industry deployments have been steeply growing. In particular, NLP inspired approaches have been successfully adapted for sequential and session-based recommendation problems, which are important for many domains like e-commerce, news and streaming media. Session-Based Recommender Systems (SBRS) have been proposed to model the sequence of interactions within the current user session, where a session is a short sequence of user interactions typically bounded by user inactivity. They have recently gained popularity due to their ability to capture short-term or contextual user preferences towards items.
# 
# The field of NLP has evolved significantly within the last decade, particularly due to the increased usage of deep learning. As a result, state of the art NLP approaches have inspired RecSys practitioners and researchers to adapt those architectures, especially for sequential and session-based recommendation problems. Here, we use one of the state-of-the-art Transformer-based architecture, [XLNet](https://arxiv.org/abs/1906.08237) with Causal Language Modeling (CLM) training technique for multi-class classification task. For this, we leverage the popular HuggingFace’s Transformers NLP library and make it possible to experiment with cutting-edge implementation of such architectures for sequential and session-based recommendation problems.

# Now, we replace the BiLSTM model with a transformer-based architecture. We train an `XLNet` model which concatenates the embedding vectors for all sequential features (`item_id_list_seq`, `f_47_list_seq`, `f_68_list_seq`) per step in the sequential input block, and uses self-attention mechanism. To learn more about the self-attention mechanism you can take a look at this [paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) and this [post](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html).
# 
# <img src="../images/XLNet_ecommerce.png"  width="30%">

# In[43]:


item_id_name = train.schema.select_by_tag(Tags.ITEM_ID).first.properties['domain']['name']


# In[44]:


seq_inputs = mm.InputBlockV2(
        schema_model.select_by_tag(Tags.SEQUENCE),
        categorical=mm.Embeddings(
            schema_model.select_by_tag(Tags.SEQUENCE),
            sequence_combiner=None,
            dim=manual_dims                 
        )
)

cat_inputs = mm.InputBlockV2(
        schema_model.select_by_name(
            ['item_id_last']
        ),
        categorical=mm.Embeddings(
            schema_model.select_by_name(
            ['item_id_last']
        ),
            dim=manual_dims
                                
        )
)


# We can check the output from the sequential input block and its dimension. We obtain a 3-D sequence representation (batch_size, sequence_length, sum_of_emb_dim_of_features).

# In[45]:


batch = mm.sample_batch(train, batch_size=128, include_targets=False, prepare_features=True)


# In[46]:


seq_inputs(batch).shape


# In[47]:


cat_inputs(batch).shape


# The sequence_length dimension is printed out as None, because it is a variable length given a batch. That's why we get the sequence_length dim printed as `None`.

# Let's create a sequential block where we connect sequential inputs block (i.e., a SequentialLayer represents a sequence of Keras layers) with MLPBlock and then XLNetBlock. MLPBlock is used as a projection block to match the output dimensions of the seq_inputs block with the transformer block. In otherwords, due to residual connection in the Transformer model, we add an MLPBlock in the model pipeline. The output dim of the input block should match with the hidden dimension (d_model) of the XLNetBlock.

# Let's set the hidden dimension of the XLNet. This is a hyper parameter that user can decide the value of it. Please note that we did not do hyper-parameter optimization (HPO) for the models used here. We recommend users to perform HPO when they adopt these architectures with their custom datasets at their end.

# In[48]:


mlp_block1 = mm.MLPBlock(
                [128,dmodel],
                activation='relu',
                no_activation_last_layer=True,
                dropout=DROPOUT,
            )


# In[49]:


dense_block =mm.SequentialBlock(
    seq_inputs,
    mlp_block1,
    mm.XLNetBlock(
        d_model=dmodel,
        n_head=4,
        n_layer=2,
        post='sequence_mean',
    )
)


# The output of XLNetBlock is a 2D tensor `(batch_size, d_model)`, and it is then fed to final output layer.

# In[50]:


dense_block(batch).shape


# Let's concatenate the output of transformer block with the output of the categorical block.

# In[51]:


concats = mm.ParallelBlock(
    {'dense_block': dense_block, 
     'cat_inputs': cat_inputs},
    aggregation='concat'
)


# The concat layer shape would be the total output dimension (256 + 256) of two layers that are concatenated.

# In[52]:


concats(batch).shape


# In[53]:


mlp_block2 = mm.MLPBlock(
                [128,dmodel],
                activation='relu',
                no_activation_last_layer=True,
                dropout=DROPOUT,
            )


# In[54]:


prediction_task= mm.CategoricalOutput(
    to_call=seq_inputs["categorical"][item_id_name],
    logits_temperature=TEMPERATURE_SCALING,
    target='purchase_id_first',
)


# In[55]:


model_transformer = mm.Model(concats, mlp_block2, prediction_task)


# We train the model

# In[56]:


optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model_transformer.compile(
    optimizer=optimizer,
    run_eagerly=False,
    loss=tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, 
        label_smoothing=LABEL_SMOOTHING
    ),
    metrics=mm.TopKMetricsAggregator.default_metrics(top_ks=[100])
)


# In[57]:


model_transformer.fit(loader, 
                      validation_data=val_loader,
                      epochs=EPOCHS
                     )


# We evaluate the model

# In[58]:


metrics_transformer = model_transformer.evaluate(val_loader, return_dict=True)
metrics_transformer['mrr_at_100']


# In[59]:


del model_transformer
gc.collect()


# ## Summary

# In this example, we focused on concepts which are relevant for a broad range of recommender system use cases- session-based recommendation task. If you compare the MRR to the ACM RecSys'22 competition, you will notice, that the MRR can be much higher. Following are additional techniques that can be applied to improve the MRR:
# - Data Augmentations - in the RecSys'22 challenge, we used a lot of different techniques to increase the training dataset. The techniques are specific to the dataset and we did not include it in the example:
# - Additional item features - we focused on only a few item features
# - Stacking - we stacked 17 models with a two-step approach
# - Ensemble - we ensembled 3 different stacked models
# - Hyperparameter Search - we ran multiple HPO jobs to find the best hyperparameters
# 
# In addition, the MRR on the June month (test data) was in general higher than in May (validation)
