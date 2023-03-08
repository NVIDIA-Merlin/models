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
# See the License for the specific language governing permissions anda
# limitations under the License.
# ==============================================================================

# Each user is responsible for checking the content of datasets and the
# applicable licenses and determining if suitable for the intended use.


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_models_07-train-traditional-ml-models-using-the-merlin-models-api/nvidia_logo.png" style="width: 90px; float: right;">
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

# In[2]:


from merlin.core.utils import Distributed
from merlin.models.xgb import XGBoost

from merlin.datasets.entertainment import get_movielens
from merlin.schema.tags import Tags
from merlin.io import Dataset


# We will use the `movielens-100k` dataset. The dataset consists of `userId` and `movieId` pairings. For each record, a user rates a movie and the record includes additional information such as genre of the movie, age of the user, and so on.

# In[3]:


train, valid = get_movielens(variant='ml-100k')


# The `get_movielens` function downloads the `movielens-100k` data for us and returns it materialized as a Merlin `Dataset`.

# In[4]:


train, valid


# One of the features that the Merlin Models API supports is tagging. You can tag your data once, during preprocessing, and this information is picked up during later steps such as additional preprocessing steps, training your model, serving the model, and so on.
# 
# Here, we will make use of the `Tags.TARGET` to identify the objective for our `XGBoost` model.
# 
# During preprocessing that is performed by the `get_movielens` function, two columns in the dataset are assigned the `Tags.TARGET` tag:

# In[5]:


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

# In[6]:


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

# In[7]:


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

# In[8]:


with Distributed():
    model = XGBoost(schema=schema_without_title, **xgb_booster_params)
    model.fit(
        train,
        evals=[(valid, 'validation_set'),],
        **xgb_train_params
    )
    metrics = model.evaluate(valid)


# ## Training an implicit model

# `Implicit` provides fast Python implementations of several different popular recommendation algorithms for implicit feedback datasets.
# 
# These models are designed to work with implicit datasets, that is datasets that don't have explicit labels! What this translates to is that we will not be able to use these algorithms for training on data with labels such as `ratings` or `number of likes received`, etc. These models are geared toward predicting a binary target of the form of how likely a user is to interact with an item of given id.
# 
# There are two implicit models you can train with Merlin Models. One approach would be to pass only user-item id pairs. In this case the model will treat the pairs we pass as positive examples and will generate negative examples by itself. Alternatively, we can pass in a column of zeros and ones where ones indicate a positive example. We need to tag that column with `Tags.TARGET` (as outlined in the "Preparing the data" section above). From there on, all that remains is to pass the data to the `fit` method of our model to train it.
# 
# There are two `implicit` models you can train with `Merlin Models`:
# 
# * `AlternatingLeastSquares` from [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf))
# * `BayesianPersonalizedRanking` from [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)
# 
# In this example, we will train a `BayesianPersonalizedRanking` model.

# In[9]:


from merlin.models.implicit import BayesianPersonalizedRanking


# `merlin.models.implicit` doesn't have the same facility as `merlin.models.xgb.XGBoost` for identifying which target column it should use.
# 
# Let's remove the `rating` column from the schema so that only `rating_binary` is left.
# 
# The `rating` column contains explicit information, that is a rating on a scale from 1 to 5 that a user assigned to a movie. The `rating_binary` column emulates implicit data. It could indicate that a user interacted with a given item, or that they watched a movie for some number of minutes, etc.
# 
# Essentially, implicit data is one where we don't ask the user to give us *explicit* information, but infer labeling from their actions!

# In[10]:


train.schema = schema_without_title.remove_col('rating')
valid.schema = schema_without_title.remove_col('rating')


# This is a shape of data that will go into our model:

# In[11]:


train.compute().head()


# However, only the columns tagged with `Tags.USER_ID`, `Tags.ITEM_ID` or `Tags.TARGET` will be used.

# In[12]:


train.schema.select_by_tag([Tags.USER_ID, Tags.ITEM_ID, Tags.TARGET])


# Let's train our model.
# 
# There are several options we can specify. Here are the 3 most important ones:
# 
# * factors - the number of latent factors to compute
# * learning_rate – the learning rate to apply for SGD updates during training
# * regularization – the regularization factor to use
# 
# Further information on the arguments that `BayesianPersonalizedRanking` accepts can be found in [implicit's documentation](https://implicit.readthedocs.io/en/latest/bpr.html).
# 
# We can also train without passing in any of the above values in which case `BayesianPersonalizedRanking` will use the defaults.

# In[13]:


implicit = BayesianPersonalizedRanking()
implicit.fit(train)


# Having trained the model, we can now evaluate it.
# 
# Implicit models can be best thought of as retrieval models and so we have the usual set of retrieval metrics at our disposal.
# 
# The metrics that are available to us are:
# * [precision](https://en.wikipedia.org/wiki/Precision_and_recall)
# * [mean average precision](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)
# * [normalized discounted cumulative gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG)
# * [area under the ROC operating curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)

# In[14]:


implicit_metrics = implicit.evaluate(valid)
implicit_metrics


# And last but not least, let's use our trained implicit model to output predictions.
# 
# We can pass the `valid` Dataset with several columns, however, using the schema, the model will select the `userId` column and output predictions which will be an ordered list of movies that the user is most likely to enjoy.

# In[15]:


valid.head()


# Let us now pass this information to our model and predict.

# In[16]:


implicit_preds = implicit.predict(valid)
implicit_preds


# The predictions are item ids for each of the user id that we passed in as our data. They are ordered from the most likely to be interacted with by the user to the least likely as predicted by our `implicit` model. These IDs are contained in the first array.
# 
# The second array consists of scores from the model. The higher the score, the better the chance a user will interact with a movie. They are not on any particular scale nor have a probabilistic interpretation.

# ## Training a LightFM model

# [LightFM](https://github.com/lyst/lightfm) implements of a number of popular recommendation algorithms for both implicit and explicit feedback, including efficient implementation of BPR and WARP ranking losses.
# 
# You can specify what type of model to train on through the use of the `loss` argument. Here we will train with a `warp` loss (Weighted Approximate-Rank Pairwise loss). You can read more about available losses as well as the parameters that can be used for training [here](https://making.lyst.com/lightfm/docs/lightfm.html).
# 
# Let us train a model that will again predict a score of how likely a user is to interact with a given item, following the same approach as we did above with the `implicit` model.

# In[17]:


from merlin.models.lightfm import LightFM

lightfm = LightFM(loss='warp')


# We can now train our model.

# In[18]:


lightfm.fit(train)


# Now that the model is trained let's validate its performance.

# In[19]:


lightfm_metrics = lightfm.evaluate(valid)
lightfm_metrics


# We can now use the model to predict on our data.
# 
# We pass our `valid` Dataset again -- this time however our model will take the `userId` column and the `movieId` column and output a score for each pairing. The higher the score the higher the chance (according to the model) of a user interacting with a given movie.

# In[20]:


lightfm_preds = lightfm.predict(valid)
lightfm_preds

