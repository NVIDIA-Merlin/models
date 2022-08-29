# #
# # Copyright (c) 2021, NVIDIA CORPORATION.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
#
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import regularizers

import merlin.models.tf as ml
from merlin.datasets.synthetic import generate_data
from merlin.io import Dataset
from merlin.models.tf.dataset import BatchedDataset
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_mf_model_single_binary_task(ecommerce_data, run_eagerly):
    model = ml.MatrixFactorizationModel(
        ecommerce_data.schema,
        dim=4,
        aggregation="cosine",
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model(music_streaming_data, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_age", "click"]
    )
    model = ml.DLRMModel(
        music_streaming_data.schema,
        embedding_dim=2,
        bottom_block=ml.MLPBlock([2]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    loaded_model, _ = testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)

    features = testing_utils.get_model_inputs(
        music_streaming_data.schema.remove_by_tag(Tags.TARGET)
    )
    testing_utils.test_model_signature(loaded_model, features, ["click/binary_classification_task"])


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model_with_embeddings(music_streaming_data, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_age", "click"]
    )
    schema = music_streaming_data.schema
    embedding_dim = 4
    model = ml.DLRMModel(
        schema,
        embeddings=ml.Embeddings(schema.select_by_tag(Tags.CATEGORICAL), dim=embedding_dim),
        bottom_block=ml.MLPBlock([embedding_dim]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model_with_sample_weights_and_weighted_metrics(music_streaming_data, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_age", "click"]
    )

    def add_sample_weight(features, labels, sample_weight_col_name="user_age"):
        sample_weight = tf.cast(features.pop(sample_weight_col_name), tf.float32)
        return features, labels, sample_weight

    batched_ds = BatchedDataset(music_streaming_data, batch_size=10)
    batched_ds = batched_ds.map(add_sample_weight)
    batch = next(iter(batched_ds))

    model = ml.DLRMModel(
        music_streaming_data.schema.select_by_name(["item_id", "click"]),
        embedding_dim=2,
        bottom_block=ml.MLPBlock([2]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    weighted_metrics = (
        tf.keras.metrics.Precision(name="weighted_precision"),
        tf.keras.metrics.Recall(name="weighted_recall"),
        tf.keras.metrics.BinaryAccuracy(name="weighted_binary_accuracy"),
        tf.keras.metrics.AUC(name="weighted_auc"),
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly, weighted_metrics=weighted_metrics)

    metrics = model.train_step(batch)

    assert set(metrics.keys()) == set(
        [
            "loss",
            "regularization_loss",
            "binary_accuracy",
            "recall",
            "precision",
            "auc",
            "weighted_binary_accuracy",
            "weighted_recall",
            "weighted_precision",
            "weighted_auc",
        ]
    )


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model_without_continuous_features(ecommerce_data, run_eagerly):
    ecommerce_data.schema = ecommerce_data.schema.select_by_name(
        ["user_categories", "item_category", "click"]
    )
    model = ml.DLRMModel(
        ecommerce_data.schema,
        embedding_dim=2,
        top_block=ml.MLPBlock([2]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("stacked", [True, False])
@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dcn_model(music_streaming_data, stacked, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_age", "click"]
    )
    model = ml.DCNModel(
        music_streaming_data.schema,
        depth=1,
        deep_block=ml.MLPBlock([2]),
        stacked=stacked,
        embedding_options=ml.EmbeddingOptions(
            embedding_dims=None,
            embedding_dim_default=2,
            infer_embedding_sizes=True,
            infer_embedding_sizes_multiplier=0.2,
        ),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model_multi_task(music_streaming_data, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_age", "click", "play_percentage"]
    )
    tasks_blocks = dict(click=ml.MLPBlock([2]), play_percentage=ml.MLPBlock([2]))
    model = ml.Model(
        ml.DLRMBlock(
            music_streaming_data.schema,
            embedding_dim=2,
            bottom_block=ml.MLPBlock([2]),
            top_block=ml.MLPBlock([2]),
        ),
        ml.MLPBlock([2]),
        ml.PredictionTasks(music_streaming_data.schema, task_blocks=tasks_blocks),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("prediction_task", [ml.BinaryClassificationTask, ml.RegressionTask])
def test_serialization_model(ecommerce_data: Dataset, prediction_task):
    model = ml.Model(
        ml.InputBlock(ecommerce_data.schema), ml.MLPBlock([2]), prediction_task("click")
    )

    testing_utils.model_test(model, ecommerce_data, reload_model=True)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_wide_deep_model(music_streaming_data, run_eagerly):

    # prepare wide_schema
    wide_schema = music_streaming_data.schema.select_by_name(["country"])
    deep_schema = music_streaming_data.schema.select_by_name(["country", "user_age"])

    model = ml.WideAndDeepModel(
        music_streaming_data.schema,
        wide_schema=wide_schema,
        deep_schema=deep_schema,
        deep_block=ml.MLPBlock([32, 16]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_wide_deep_model_wide_categorical_one_hot(ecommerce_data, run_eagerly):

    wide_schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    deep_schema = ecommerce_data.schema

    model = ml.WideAndDeepModel(
        ecommerce_data.schema,
        wide_schema=wide_schema,
        deep_schema=deep_schema,
        wide_preprocess=ml.CategoryEncoding(wide_schema, sparse=True),
        deep_block=ml.MLPBlock([32, 16]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_wide_deep_model_hashed_cross(ecommerce_data, run_eagerly):

    wide_schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    deep_schema = ecommerce_data.schema

    model = ml.WideAndDeepModel(
        ecommerce_data.schema,
        wide_schema=wide_schema,
        deep_schema=deep_schema,
        wide_preprocess=ml.HashedCross(wide_schema, 1000, sparse=True),
        deep_block=ml.MLPBlock([32, 16]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_wide_deep_embedding_custom_inputblock(music_streaming_data, run_eagerly):

    schema = music_streaming_data.schema
    # prepare wide_schema
    wide_schema = schema.select_by_name(["country", "user_age"])
    deep_embedding = ml.Embeddings(schema.select_by_tag(Tags.CATEGORICAL), dim=16)

    model = ml.WideAndDeepModel(
        schema,
        deep_input_block=ml.InputBlockV2(schema=schema, embeddings=deep_embedding),
        wide_schema=wide_schema,
        wide_preprocess=ml.HashedCross(wide_schema, 1000, sparse=True),
        deep_block=ml.MLPBlock([32, 16]),
        deep_regularizer=regularizers.l2(1e-5),
        wide_regularizer=regularizers.l2(1e-5),
        deep_dropout=0.1,
        wide_dropout=0.2,
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_wide_deep_model_wide_onehot_multihot_feature_interaction(ecommerce_data, run_eagerly):
    ml_dataset = generate_data("movielens-1m", 100)
    # data_ddf = ml_dataset.to_ddf()
    # data_ddf = data_ddf[[c for c in list(data_ddf.columns) if c != "rating"]]

    # Removing the rating regression target
    schema = ml_dataset.schema.remove_col("rating")
    target_col = schema.select_by_tag(Tags.TARGET).column_names[0]

    cat_schema = schema.select_by_tag(Tags.CATEGORICAL)
    cat_schema_onehot = cat_schema.remove_col("genres")
    cat_schema_multihot = cat_schema.select_by_name("genres")

    ignore_combinations = [["age", "userId"], ["userId", "occupation"]]

    wide_preprocessing_blocks = [
        # One-hot features
        ml.SequentialBlock(
            ml.Filter(cat_schema_onehot),
            ml.CategoryEncoding(cat_schema_onehot, sparse=True, output_mode="one_hot"),
        ),
        # Multi-hot features
        ml.SequentialBlock(
            ml.Filter(cat_schema_multihot),
            ml.AsDenseFeatures(max_seq_length=6),
            ml.CategoryEncoding(cat_schema_multihot, sparse=True, output_mode="multi_hot"),
        ),
        # 2nd level feature interactions of one-hot features
        ml.SequentialBlock(
            ml.Filter(cat_schema),
            ml.AsDenseFeatures(max_seq_length=6),
            ml.HashedCrossAll(
                cat_schema,
                num_bins=100,
                max_level=2,
                output_mode="multi_hot",
                sparse=True,
                ignore_combinations=ignore_combinations,
            ),
        ),
    ]

    batch = ml.sample_batch(ml_dataset, batch_size=100, include_targets=False)

    output_wide_features = ml.ParallelBlock(wide_preprocessing_blocks)(batch)
    assert set(output_wide_features.keys()) == set(
        [
            "userId",
            "movieId",
            "title",
            "gender",
            "age",
            "occupation",
            "zipcode",
            "genres",
            "cross_movieId_userId",
            "cross_title_userId",
            "cross_genres_userId",
            "cross_gender_userId",
            "cross_userId_zipcode",
            "cross_movieId_title",
            "cross_genres_movieId",
            "cross_gender_movieId",
            "cross_age_movieId",
            "cross_movieId_occupation",
            "cross_movieId_zipcode",
            "cross_genres_title",
            "cross_gender_title",
            "cross_age_title",
            "cross_occupation_title",
            "cross_title_zipcode",
            "cross_gender_genres",
            "cross_age_genres",
            "cross_genres_occupation",
            "cross_genres_zipcode",
            "cross_age_gender",
            "cross_gender_occupation",
            "cross_gender_zipcode",
            "cross_age_occupation",
            "cross_age_zipcode",
            "cross_occupation_zipcode",
        ]
    )

    assert all([isinstance(t, tf.SparseTensor) for t in output_wide_features.values()])
    assert all([len(t.shape) == 2 for t in output_wide_features.values()])
    assert all([t.shape[1] > 1 for t in output_wide_features.values()])
    assert all(
        [
            np.all(np.sum(tf.sparse.to_dense(v).numpy(), axis=1) == 1.0)
            for k, v in output_wide_features.items()
            if "genres" not in k
        ]
    ), "All features should be one-hot, except 'genres'"
    assert (
        np.max(np.sum(tf.sparse.to_dense(output_wide_features["genres"]).numpy(), axis=1)) > 1.0
    ), "'genres' should be multi-hot"

    model = ml.WideAndDeepModel(
        cat_schema,
        wide_schema=cat_schema,
        deep_schema=cat_schema,
        wide_preprocess=ml.ParallelBlock(
            wide_preprocessing_blocks,
            aggregation="concat",
        ),
        deep_block=ml.MLPBlock([32, 16]),
        prediction_tasks=ml.BinaryClassificationTask(target_col),
    )

    testing_utils.model_test(model, ml_dataset, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_wide_deep_model_wide_feature_interaction_multi_optimizer(ecommerce_data, run_eagerly):

    deep_schema = ecommerce_data.schema.select_by_name(
        names=["user_categories", "item_category", "user_brands", "item_brand"]
    )
    wide_schema = ecommerce_data.schema.select_by_name(
        names=["user_categories", "item_category", "user_brands", "item_brand"]
    )

    model = ml.WideAndDeepModel(
        ecommerce_data.schema,
        wide_schema=wide_schema,
        deep_schema=deep_schema,
        wide_preprocess=ml.ParallelBlock(
            [
                # One-hot representations of categorical features
                ml.CategoryEncoding(wide_schema, sparse=True),
                # One-hot representations of hashed 2nd-level feature interactions
                ml.HashedCrossAll(wide_schema, num_bins=1000, max_level=2, sparse=True),
            ],
            aggregation="concat",
        ),
        deep_block=ml.MLPBlock([31, 16]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )
    # print(model.summary(expand_nested=True, show_trainable=True, line_length=80))
    """
     Layer (type)                       Output Shape                    Param #     Trainable
    ===========================================================================================
    parallel_block_2 (ParallelBlock)   multiple                        9186        Y
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | sequential_block_6 (SequentialBloc  multiple                     2           Y          |
    | k)                                                                                      |
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || parallel_block_1 (ParallelBlock)  multiple                     0           Y          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| tabular_block_1 (TabularBlock)  multiple                     0           Y          |||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || sequential_block_5 (SequentialBloc  multiple                   2           Y          ||
    || k)                                                                                    ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| private__dense_3 (_Dense)    multiple                        2           Y          |||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | sequential_block_3 (SequentialBloc  multiple                     9184        Y          |
    | k)                                                                                      |
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || sequential_block_1 (SequentialBloc  multiple                   9167        Y          ||
    || k)                                                                                    ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| parallel_block (ParallelBlock)  multiple                     7632        Y          |||
    ||||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||||
    |||| embeddings (ParallelBlock)  multiple                       7632        Y          ||||
    |||||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||||
    ||||| user_categories (EmbeddingTable)  multiple               4816        Y          |||||
    |||||                                                                                 |||||
    ||||| item_category (EmbeddingTable)  multiple                 808         Y          |||||
    |||||                                                                                 |||||
    ||||| item_brand (EmbeddingTable)  multiple                    2008        Y          |||||
    ||||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| sequential_block (SequentialBlock)  multiple                 1535        Y          |||
    ||||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||||
    |||| private__dense (_Dense)    multiple                        1023        Y          ||||
    ||||                                                                                   ||||
    |||| private__dense_1 (_Dense)  multiple                        512         Y          ||||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || sequential_block_2 (SequentialBloc  multiple                   17          Y          ||
    || k)                                                                                    ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| private__dense_2 (_Dense)    multiple                        17          Y          |||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
    click/binary_classification_task (  multiple                       2           Y
    BinaryClassificationTask)

    model_context (ModelContext)       multiple                        0           Y

    ===========================================================================================
    """

    testing_utils.model_test(model, ecommerce_data, run_eagerly=True)

    # Get the names of wide model and deep model from model.summary()
    wide_model = model.get_blocks_by_name("sequential_block_6")
    deep_model = model.get_blocks_by_name("sequential_block_3")

    multi_optimizer = ml.MultiOptimizer(
        default_optimizer="adagrad",
        optimizers_and_blocks=[
            ml.OptimizerBlocks("ftrl", wide_model),
            ml.OptimizerBlocks("adagrad", deep_model),
        ],
    )
    testing_utils.model_test(
        model, ecommerce_data, run_eagerly=run_eagerly, optimizer=multi_optimizer
    )
