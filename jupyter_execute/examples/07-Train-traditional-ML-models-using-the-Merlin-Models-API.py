#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
# See the License for the specific language governing permissions anda
# limitations under the License.
# ==============================================================================


# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Train traditional ML models using the Merlin Models API
# 
# ## Overview
# 
# Merlin Models exposes a high-level API that can be used with models from other libraries. For the Merlin Models v0.6.0 release, some `XGBoost`, `implicit` and `lightFM` models are supported.
# 
# Relying on this high level API enables you to iterate more effectively. You do not have to switch between various APIs as you evaluate additional models on your data.
# 
# Furthermore, you can use your data represented as a `Dataset` across all your models.
# 
# We begin by training and `XGBoost` model. In this section we go into more details on some of the best practices around training `XGBoost` models and the technical aspects of training (using `DaskDeviceQuantileDMatrix` and the  `Distributed` context manager for efficient resource usage).
# 
# Subsequently, we provide brief examples of using the Merlin Models high level API to train `lightFM` and `implicit` models on Merlin Datasets.
# 
# ### Learning objectives
# 
# - Training an `XGBoost` model with `DaskDeviceQuantileDMatrix` and early stopping evaluated on the validation set
# - Starting a local dask cluster with the `Distributed` context manager
# - Training `implicit` and `lightFM` models
# - Understanding the interplay between column tagging and setting the objective for a model for target selection
# - Using the Merlin Models high level API

# ## Preparing the dataset

# In[3]:


from merlin.core.utils import Distributed
from merlin.models.xgb import XGBoost

from merlin.datasets.entertainment import get_movielens
from merlin.schema.tags import Tags


# We will use the `movielens-100k` dataset. The dataset consists of `userId` and `movieId` pairings. For each record, a user rates a movie and the record includes additional information such as genre of the movie, age of the user, and so on.

# In[4]:


train, valid = get_movielens(variant='ml-100k')


# The `get_movielens` function downloads the `movielens-100k` data for us and returns it materialized as a Merlin `Dataset`.

# In[5]:


train, valid


# One of the features that the Merlin Models API supports is tagging. You can tag your data once, during preprocessing, and this information is picked up during later steps such as additional preprocessing steps, training your model, serving the model, and so on.
# 
# Here, we will make use of the `Tags.TARGET` to identify the objective for our `XGBoost` model.
# 
# During preprocessing that is performed by the `get_movielens` function, two columns in the dataset are assigned the `Tags.TARGET` tag:

# In[6]:


train.schema.select_by_tag(Tags.TARGET)


# You can specify the target to train by passing `target_columns` when you construct the model. We would like to use `rating_binary` as our target, so we could do the following:
# 
# `model = XGBoost(target_columns='rating_binary', ...`
# 
# However, we can also do something better. Instead of providing this argument to the constructor of our model, we can instead specify the `objective` for our `XGBoost` model and have the Merlin Models API do the rest of the work for us.
# 
# Later in this example, we will set our booster's objective to `'binary:logistic'`. Given this piece of information, the Merlin Model code can infer that we want to train with a target that has the `Tags.BINARY_CLASSIFICATION` tag assigned to it and there will be nothing else we will need to do.
# 
# Before we begin to train, let us remove the `title` column from our schema. In the dataset, the title is a string, and unless we preprocess it further, it is not useful in training.

# In[7]:


schema_without_title = train.schema.remove_col('title')


# To summarize, we will train an `XGBoost` model that predicts the rating of a movie.
# 
# For the `rating_binary` column, a value of `1` indicates that the user has given the movie a high rating, and a target of `0` indicates that the user has given the movie a low rating.

# ## Training an XGBoost model

# Before we begin training, let's define a couple of custom parameters.
# 
# Specifying `gpu_hist` as our `tree_method` will run the training on the GPU. Also, it will trigger representing our datasets as `DaskDeviceQuantileDMatrix` instead of the standard `DaskDMatrix`. This class is introduced in the XGBoost 1.1 release and this data format provides more efficient training with lower memory footprint. You can read more about it in this [article](https://medium.com/rapids-ai/new-features-and-optimizations-for-gpus-in-xgboost-1-1-fc153dc029ce) from the RAPIDS AI channel.
# 
# Additionally, we will train with early stopping and evaluate the stopping criteria on a validation set. If we were to train without early stopping, `XGboost` would continue to improve results on the train set until it would reach a perfect score. That would result in a low training loss but we would lose any ability to generalize to unseen data. Instead, by training with early stopping, the training ceases as soon as the model starts overfitting to the train set and the results on the validation set will start to deteriorate.
# 
# The `verbose_eval` parameter specifies how often metrics are reported during training.

# In[8]:


xgb_booster_params = {
    'objective':'binary:logistic',
    'tree_method':'gpu_hist',
}

xgb_train_params = {
    'num_boost_round': 100,
    'verbose_eval': 20,
    'early_stopping_rounds': 10,
}


# We are now ready to train.
# 
# In order to facilitate training on data larger than the available GPU memory, the training will leverage Dask. All the complexity of starting a local dask cluster is hidden in the `Distributed` context manager.
# 
# Without further ado, let's train.

# In[9]:


with Distributed():
    model = XGBoost(schema=schema_without_title, **xgb_booster_params)
    model.fit(
        train,
        evals=[(valid, 'validation_set'),],
        **xgb_train_params
    )
    metrics = model.evaluate(valid)


# ## Training an implicit model

# There are two `implicit` models you can train. The `AlternatingLeastSquares` and `BayesianPersonalizedRanking` models. We will train a `BayesianPersonalizedRanking` model

# In[10]:


from merlin.models.implicit import BayesianPersonalizedRanking


# `merlin.models.implicit` doesn't have the same facility as `merlin.models.xgb.XGBoost` for identifying which target column it should use.
# 
# Let's remove the `rating` column from the schema so that only `rating_binary` is left.

# In[11]:


train.schema = schema_without_title.remove_col('rating')
valid.schema = schema_without_title.remove_col('rating')


# In[12]:


implicit = BayesianPersonalizedRanking()
implicit.fit(train)


# Having trained the model, we can now evaluate it.

# In[13]:


implicit_metrics = implicit.evaluate(valid)


# And last but not least, lets use our trained implicit model to output predictions.

# In[14]:


implicit_preds = implicit.predict(valid)


# ## Training a LightFM model

# [LightFM](https://github.com/lyst/lightfm) implements of a number of popular recommendation algorithms for both implicit and explicit feedback, including efficient implementation of BPR and WARP ranking losses.
# 
# You can specify what type of model to train on through the use of the `loss` argument. Here we will train with a `warp` loss (Weighted Approximate-Rank Pairwise loss). You can read more about available losses as well as the parameters that can be used for training [here](https://making.lyst.com/lightfm/docs/lightfm.html).

# In[15]:


from merlin.models.lightfm import LightFM

lightfm = LightFM(loss='warp')


# We can now train our model.

# In[16]:


lightfm.fit(train)


# Now that the model is trained let's validate its performance.

# In[17]:


lightfm_metrics = lightfm.evaluate(valid)


# We can now use the model to predict on our data.

# In[18]:


lightfm_preds = lightfm.predict(valid)

