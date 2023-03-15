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

import merlin.models.tf as mm
from merlin.datasets.synthetic import generate_data
from merlin.io import Dataset
from merlin.models.tf.transforms.features import expected_input_cols_from_schema
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_mf_model_single_binary_task(ecommerce_data, run_eagerly):
    model = mm.MatrixFactorizationModel(
        ecommerce_data.schema,
        dim=4,
        aggregation="cosine",
        prediction_tasks=mm.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize(
    "prediction_blocks", [None, mm.BinaryOutput("click"), mm.BinaryClassificationTask("click")]
)
def test_dlrm_model(music_streaming_data, run_eagerly, prediction_blocks):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_id", "user_age", "item_genres", "click"]
    )
    model = mm.DLRMModel(
        music_streaming_data.schema,
        embedding_dim=2,
        bottom_block=mm.MLPBlock([5, 2], dropout=0.05),
        prediction_tasks=prediction_blocks,
    )

    loaded_model, _ = testing_utils.model_test(
        model, music_streaming_data, run_eagerly=run_eagerly, reload_model=True
    )

    expected_features = expected_input_cols_from_schema(music_streaming_data.schema)
    expected_output_signature = (
        "click/binary_classification_task"
        if isinstance(prediction_blocks, mm.PredictionTask)
        else "click/binary_output"
    )
    testing_utils.test_model_signature(loaded_model, expected_features, [expected_output_signature])


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model_with_embeddings(music_streaming_data, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_age", "click"]
    )
    schema = music_streaming_data.schema
    embedding_dim = 4
    model = mm.DLRMModel(
        schema,
        embeddings=mm.Embeddings(schema.select_by_tag(Tags.CATEGORICAL), dim=embedding_dim),
        bottom_block=mm.MLPBlock([embedding_dim]),
        prediction_tasks=mm.BinaryOutput("click"),
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

    loader = mm.Loader(music_streaming_data, batch_size=10).map(add_sample_weight)
    batch = next(iter(loader))

    model = mm.DLRMModel(
        music_streaming_data.schema.select_by_name(["item_id", "click"]),
        embedding_dim=2,
        bottom_block=mm.MLPBlock([2]),
        prediction_tasks=mm.BinaryOutput("click"),
    )

    weighted_metrics = (
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly, weighted_metrics=weighted_metrics)

    metrics = model.train_step(batch)

    assert set(metrics.keys()) == set(
        [
            "loss",
            "loss_batch",
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
    model = mm.DLRMModel(
        ecommerce_data.schema,
        embedding_dim=2,
        top_block=mm.MLPBlock([2]),
        prediction_tasks=mm.BinaryOutput("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("stacked", [True, False])
@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dcn_model(music_streaming_data, stacked, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_age", "click"]
    )
    model = mm.DCNModel(
        music_streaming_data.schema,
        depth=1,
        deep_block=mm.MLPBlock([2]),
        stacked=stacked,
        prediction_tasks=mm.BinaryOutput("click"),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_deepfm_model_only_categ_feats(music_streaming_data, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "item_category", "user_id", "click"]
    )
    model = mm.DeepFMModel(
        music_streaming_data.schema,
        embedding_dim=16,
        deep_block=mm.MLPBlock([16]),
        prediction_tasks=mm.BinaryOutput("click"),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_deepfm_model_categ_and_continuous_feats(music_streaming_data, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "item_category", "user_id", "user_age", "click"]
    )
    model = mm.DeepFMModel(
        music_streaming_data.schema,
        embedding_dim=16,
        deep_block=mm.MLPBlock([16]),
        prediction_tasks=mm.BinaryOutput("click"),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model_multi_task(music_streaming_data, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_age", "click", "play_percentage"]
    )
    tasks_blocks = dict(click=mm.MLPBlock([2]), play_percentage=mm.MLPBlock([2]))
    model = mm.Model(
        mm.DLRMBlock(
            music_streaming_data.schema,
            embedding_dim=2,
            bottom_block=mm.MLPBlock([2]),
            top_block=mm.MLPBlock([2]),
        ),
        mm.MLPBlock([2]),
        mm.PredictionTasks(music_streaming_data.schema, task_blocks=tasks_blocks),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model_multi_task_v2(music_streaming_data, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_age", "click", "play_percentage"]
    )
    tasks_blocks = dict(click=mm.MLPBlock([2]), play_percentage=mm.MLPBlock([2]))
    model = mm.Model(
        mm.DLRMBlock(
            music_streaming_data.schema,
            embedding_dim=2,
            bottom_block=mm.MLPBlock([2]),
            top_block=mm.MLPBlock([2]),
        ),
        mm.MLPBlock([2]),
        mm.OutputBlock(music_streaming_data.schema, task_blocks=tasks_blocks),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize(
    "prediction_task",
    [mm.BinaryClassificationTask, mm.RegressionTask, mm.BinaryOutput, mm.RegressionOutput],
)
def test_serialization_model(ecommerce_data: Dataset, prediction_task):
    model = mm.Model(
        mm.InputBlockV2(ecommerce_data.schema), mm.MLPBlock([2]), prediction_task("click")
    )

    testing_utils.model_test(model, ecommerce_data, reload_model=True)


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize(
    "prediction_blocks", [None, mm.BinaryOutput("click"), mm.BinaryClassificationTask("click")]
)
def test_wide_deep_model(music_streaming_data, run_eagerly, prediction_blocks):
    # prepare wide_schema
    wide_schema = music_streaming_data.schema.select_by_name(["country"])
    deep_schema = music_streaming_data.schema.select_by_name(["country", "user_age"])

    model = mm.WideAndDeepModel(
        music_streaming_data.schema,
        wide_schema=wide_schema,
        deep_schema=deep_schema,
        deep_block=mm.MLPBlock([32, 16]),
        prediction_tasks=prediction_blocks,
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_wide_deep_model_wide_categorical_one_hot(ecommerce_data, run_eagerly):
    wide_schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    deep_schema = ecommerce_data.schema

    model = mm.WideAndDeepModel(
        ecommerce_data.schema,
        wide_schema=wide_schema,
        deep_schema=deep_schema,
        wide_preprocess=mm.CategoryEncoding(wide_schema, sparse=True),
        deep_block=mm.MLPBlock([32, 16]),
        prediction_tasks=mm.BinaryOutput("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_wide_deep_model_hashed_cross(ecommerce_data, run_eagerly):
    wide_schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    deep_schema = ecommerce_data.schema

    model = mm.WideAndDeepModel(
        ecommerce_data.schema,
        wide_schema=wide_schema,
        deep_schema=deep_schema,
        wide_preprocess=mm.HashedCross(wide_schema, 1000, sparse=True),
        deep_block=mm.MLPBlock([32, 16]),
        prediction_tasks=mm.BinaryOutput("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_wide_deep_embedding_custom_inputblock(music_streaming_data, run_eagerly):
    schema = music_streaming_data.schema
    # prepare wide_schema
    wide_schema = schema.select_by_name(["country", "user_age"])
    deep_embedding = mm.Embeddings(schema.select_by_tag(Tags.CATEGORICAL), dim=16)

    model = mm.WideAndDeepModel(
        schema,
        deep_input_block=mm.InputBlockV2(schema=schema, categorical=deep_embedding),
        wide_schema=wide_schema,
        wide_preprocess=mm.HashedCross(wide_schema, 1000, sparse=True),
        deep_block=mm.MLPBlock([32, 16]),
        deep_regularizer=regularizers.l2(1e-5),
        wide_regularizer=regularizers.l2(1e-5),
        deep_dropout=0.1,
        wide_dropout=0.2,
        prediction_tasks=mm.BinaryOutput("click"),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_wide_deep_model_wide_onehot_multihot_feature_interaction(run_eagerly):
    ml_dataset = generate_data("movielens-1m", 100, max_session_length=4)

    # Removing the rating regression target
    schema = ml_dataset.schema.remove_col("rating")
    target_col = schema.select_by_tag(Tags.TARGET).column_names[0]

    cat_schema = schema.select_by_tag(Tags.CATEGORICAL)
    cat_schema_onehot = cat_schema.remove_col("genres")
    cat_schema_multihot = cat_schema.select_by_name("genres")

    ignore_combinations = [["age", "userId"], ["userId", "occupation"]]

    wide_preprocessing_blocks = [
        # One-hot features
        mm.SequentialBlock(
            mm.Filter(cat_schema_onehot),
            mm.CategoryEncoding(cat_schema_onehot, sparse=True, output_mode="one_hot"),
        ),
        # Multi-hot features
        mm.SequentialBlock(
            mm.Filter(cat_schema_multihot),
            mm.ToDense(cat_schema_multihot),
            mm.CategoryEncoding(cat_schema_multihot, sparse=True, output_mode="multi_hot"),
        ),
        # 2nd level feature interactions of one-hot features
        mm.SequentialBlock(
            mm.Filter(cat_schema),
            mm.ToDense(cat_schema),
            mm.HashedCrossAll(
                cat_schema,
                num_bins=100,
                max_level=2,
                output_mode="multi_hot",
                sparse=True,
                ignore_combinations=ignore_combinations,
            ),
        ),
    ]

    batch, _ = mm.sample_batch(ml_dataset, batch_size=100, prepare_features=True)
    output_wide_features = mm.ParallelBlock(wide_preprocessing_blocks)(batch)
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

    model = mm.WideAndDeepModel(
        cat_schema,
        wide_schema=cat_schema,
        deep_schema=cat_schema,
        wide_preprocess=mm.ParallelBlock(
            wide_preprocessing_blocks,
            aggregation="concat",
        ),
        deep_block=mm.MLPBlock([32, 16]),
        prediction_tasks=mm.BinaryOutput(target_col),
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

    model = mm.WideAndDeepModel(
        ecommerce_data.schema,
        wide_schema=wide_schema,
        deep_schema=deep_schema,
        wide_preprocess=mm.ParallelBlock(
            [
                # One-hot representations of categorical features
                mm.CategoryEncoding(wide_schema, sparse=True),
                # One-hot representations of hashed 2nd-level feature interactions
                mm.HashedCrossAll(wide_schema, num_bins=1000, max_level=2, sparse=True),
            ],
            aggregation="concat",
        ),
        deep_block=mm.MLPBlock([31, 16]),
        prediction_tasks=mm.BinaryOutput("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=True)

    wide_model = model.blocks[0].parallel_layers["wide"]
    deep_model = model.blocks[0].parallel_layers["deep"]

    multi_optimizer = mm.MultiOptimizer(
        default_optimizer="adagrad",
        optimizers_and_blocks=[
            mm.OptimizerBlocks("ftrl", wide_model),
            mm.OptimizerBlocks("adagrad", deep_model),
        ],
    )
    testing_utils.model_test(
        model, ecommerce_data, run_eagerly=run_eagerly, optimizer=multi_optimizer
    )
