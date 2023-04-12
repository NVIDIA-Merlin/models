from pathlib import Path

import nvtabular as nvt
import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.core.dispatch import make_df
from merlin.io import Dataset
from merlin.models.tf.metrics.topk import (
    AvgPrecisionAt,
    MRRAt,
    NDCGAt,
    PrecisionAt,
    RecallAt,
    TopKMetricsAggregator,
)
from merlin.models.tf.outputs.base import DotProduct
from merlin.models.tf.transforms.features import expected_input_cols_from_schema
from merlin.models.tf.utils import testing_utils
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.schema import Tags
from tests.common.tf.retrieval import retrieval_tests_common


def test_two_tower_shared_embeddings():
    train = make_df(
        {
            "user_id": [1, 3, 3, 4, 3, 1, 2, 4, 6, 7, 8, 9] * 100,
            "item_id": [1, 2, 3, 4, 11, 12, 5, 1, 1, 3, 5, 11] * 100,
            "item_id_hist": [
                [1, 3, 10],
                [1, 5],
                [4, 2, 1],
                [1, 2, 3],
                [1],
                [3, 4],
                [1, 3, 10],
                [11, 3, 10],
                [3, 4],
                [1, 3, 10],
                [11, 3, 10],
                [1, 11],
            ]
            * 100,
        }
    )

    user_id = ["user_id"] >> nvt.ops.Categorify() >> nvt.ops.TagAsUserID()

    joint_feats = [["item_id_hist", "item_id"]] >> nvt.ops.Categorify()

    item_id = joint_feats["item_id"] >> nvt.ops.TagAsItemID()
    user_feat = joint_feats["item_id_hist"] >> nvt.ops.TagAsUserFeatures()
    outputs = user_id + item_id + user_feat

    train_dataset = Dataset(train)

    workflow = nvt.Workflow(outputs)
    workflow.fit(train_dataset)
    train = workflow.transform(train_dataset)
    schema = train.schema

    input_block = mm.InputBlockV2(schema)
    item_tower = input_block.select_by_tag(Tags.ITEM)
    query_tower = input_block.select_by_tag(Tags.USER)

    model = mm.TwoTowerModel(
        schema,
        query_tower=query_tower.connect(mm.MLPBlock([256, 128], no_activation_last_layer=True)),
        item_tower=item_tower.connect(mm.MLPBlock([256, 128], no_activation_last_layer=True)),
        samplers=[mm.InBatchSampler()],
    )

    model.compile(optimizer="adam", run_eagerly=False, metrics=[])

    model.fit(train, batch_size=128, epochs=5)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_matrix_factorization_model(music_streaming_data: Dataset, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(["user_id", "item_id"])

    model = mm.MatrixFactorizationModel(music_streaming_data.schema, dim=4)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(music_streaming_data, batch_size=50, epochs=1, steps_per_epoch=1)
    assert len(losses.epoch) == 1
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


def test_matrix_factorization_model_l2_reg(testing_data: Dataset):
    testing_data.schema = testing_data.schema.select_by_name(["user_id", "item_id"])
    model = mm.MatrixFactorizationModel(testing_data.schema, dim=4, embeddings_l2_reg=0.1)

    _ = model(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    l2_emb_losses = model.losses

    assert (
        len(l2_emb_losses) == 2
    ), "The number of reg losses should be 2 (for user and item embeddings)"

    for reg_loss in l2_emb_losses:
        assert reg_loss > 0.0


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_matrix_factorization_model_v2(music_streaming_data: Dataset, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(["user_id", "item_id"])

    model = mm.MatrixFactorizationModelV2(
        music_streaming_data.schema, negative_samplers="in-batch", dim=4
    )
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    _, losses = testing_utils.model_test(model, music_streaming_data, reload_model=True)

    assert len(losses.epoch) == 1
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_matrix_factorization_topk_evaluation(music_streaming_data: Dataset, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(["user_id", "item_id"])
    model = mm.MatrixFactorizationModelV2(
        music_streaming_data.schema, negative_samplers="in-batch", dim=4
    )
    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    _, losses = testing_utils.model_test(model, music_streaming_data, reload_model=False)

    # Top-K evaluation
    candidate_features = unique_rows_by_features(music_streaming_data, Tags.ITEM, Tags.ITEM_ID)
    topk_model = model.to_top_k_encoder(candidate_features, k=20, batch_size=16)
    topk_model.compile(run_eagerly=run_eagerly)

    loader = mm.Loader(music_streaming_data, batch_size=32).map(
        mm.ToTarget(music_streaming_data.schema, "item_id")
    )

    metrics = topk_model.evaluate(loader, return_dict=True)
    assert all([metric >= 0 for metric in metrics.values()])


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize(
    "task",
    [
        mm.BinaryOutput,
        mm.RegressionOutput,
    ],
)
def test_matrix_factorization_model_with_binary_task(ecommerce_data: Dataset, run_eagerly, task):
    task = task("click", pre=DotProduct())
    model = mm.MatrixFactorizationModelV2(ecommerce_data.schema, dim=4, outputs=task)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    losses = model.fit(ecommerce_data, batch_size=50, epochs=1, steps_per_epoch=1)
    assert len(losses.epoch) == 1
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


def test_matrix_factorization_model_v2_l2_reg(testing_data: Dataset):
    testing_data.schema = testing_data.schema.select_by_name(["user_id", "item_id"])

    model = mm.MatrixFactorizationModelV2(
        testing_data.schema,
        dim=4,
        embeddings_l2_batch_regularization=0.1,
        negative_samplers="in-batch",
    )

    _ = model(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    l2_emb_losses = model.losses

    assert (
        len(l2_emb_losses) == 2
    ), "The number of reg losses should be 2 (for user and item embeddings)"

    for reg_loss in l2_emb_losses:
        assert reg_loss > 0.0


def test_matrix_factorization_model_v2_save(tmpdir, testing_data: Dataset):
    testing_data.schema = testing_data.schema.select_by_name(["user_id", "item_id"])

    model = mm.MatrixFactorizationModelV2(
        testing_data.schema,
        dim=4,
        embeddings_l2_batch_regularization=0.1,
        negative_samplers="in-batch",
    )

    _ = testing_utils.model_test(model, testing_data, reload_model=True)

    query_tower = model.query_encoder
    query_tower_path = Path(tmpdir) / "query_tower"
    query_tower.save(query_tower_path)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_two_tower_model(music_streaming_data: Dataset, run_eagerly, num_epochs=2):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_genres"]
    )

    model = mm.TwoTowerModel(music_streaming_data.schema, query_tower=mm.MLPBlock([2]))
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(music_streaming_data, batch_size=50, epochs=num_epochs, steps_per_epoch=1)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])

    query_features = expected_input_cols_from_schema(
        music_streaming_data.schema.select_by_tag(Tags.USER),
    )
    testing_utils.test_model_signature(model.first.query_block(), query_features, ["output_1"])

    item_features = expected_input_cols_from_schema(
        music_streaming_data.schema.select_by_tag(Tags.ITEM),
    )
    testing_utils.test_model_signature(model.first.item_block(), item_features, ["output_1"])


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_two_tower_retrieval_model_evaluate_after_fit_validation_should_raise(
    ecommerce_data: Dataset, run_eagerly
):
    ecommerce_data.schema = ecommerce_data.schema.remove_by_tag(Tags.TARGET)
    df = ecommerce_data.to_ddf().compute()
    train_ds = Dataset(df[: len(df) // 2], schema=ecommerce_data.schema)
    eval_ds = Dataset(df[len(df) // 2 :], schema=ecommerce_data.schema)

    model = mm.TwoTowerModel(
        schema=ecommerce_data.schema,
        query_tower=mm.MLPBlock([128, 64]),
        samplers=[mm.InBatchSampler()],
    )
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, run_eagerly=run_eagerly)

    num_epochs = 3
    losses = model.fit(
        train_ds,
        batch_size=64,
        epochs=num_epochs,
        train_metrics_steps=3,
        validation_data=eval_ds,
        validation_steps=3,
    )

    expected_metrics = ["recall_at_10", "mrr_at_10", "ndcg_at_10", "map_at_10", "precision_at_10"]
    expected_loss_metrics = ["loss", "loss_batch", "regularization_loss"]
    expected_metrics_all = expected_metrics + expected_loss_metrics
    expected_metrics_valid = [f"val_{k}" for k in expected_metrics_all]
    assert set(losses.history.keys()) == set(expected_metrics_all + expected_metrics_valid)

    with pytest.raises(Exception) as exc_info:
        _ = model.evaluate(eval_ds, item_corpus=train_ds, batch_size=10, return_dict=True)
    assert "The model.evaluate() was called before without `item_corpus`" in str(exc_info.value)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_two_tower_model_v2(music_streaming_data: Dataset, run_eagerly, num_epochs=2):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_genres"]
    )
    query = mm.Encoder(music_streaming_data.schema.select_by_tag(Tags.USER), mm.MLPBlock([2]))
    candidate = mm.Encoder(music_streaming_data.schema.select_by_tag(Tags.ITEM), mm.MLPBlock([2]))

    model = mm.TwoTowerModelV2(query, candidate)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(music_streaming_data, batch_size=50, epochs=num_epochs, steps_per_epoch=1)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])

    query_features = expected_input_cols_from_schema(
        music_streaming_data.schema.select_by_tag(Tags.USER)
    )
    testing_utils.test_model_signature(model.query_encoder, query_features, ["output_1"])

    item_features = expected_input_cols_from_schema(
        music_streaming_data.schema.select_by_tag(Tags.ITEM),
    )
    testing_utils.test_model_signature(model.candidate_encoder, item_features, ["output_1"])


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize(
    "task",
    [
        mm.BinaryOutput,
        mm.RegressionOutput,
    ],
)
def test_two_tower_model_with_different_tasks(ecommerce_data: Dataset, run_eagerly, task):
    query = mm.Encoder(ecommerce_data.schema.select_by_tag(Tags.USER), mm.MLPBlock([2]))
    candidate = mm.Encoder(ecommerce_data.schema.select_by_tag(Tags.ITEM), mm.MLPBlock([2]))
    task = task("click", pre=DotProduct())
    model = mm.TwoTowerModelV2(query, candidate, outputs=task)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    losses = model.fit(ecommerce_data, batch_size=50, epochs=1, steps_per_epoch=1)
    assert len(losses.epoch) == 1
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


def test_two_tower_model_save(tmpdir, ecommerce_data: Dataset):
    dataset = ecommerce_data
    schema = dataset.schema
    model = mm.TwoTowerModel(
        schema,
        query_tower=mm.MLPBlock([4], no_activation_last_layer=True),
        samplers=[mm.InBatchSampler()],
        embedding_options=mm.EmbeddingOptions(infer_embedding_sizes=True),
    )

    testing_utils.model_test(model, dataset, reload_model=True)

    query_tower = model.retrieval_block.query_block()
    query_tower_path = Path(tmpdir) / "query_tower"
    query_tower.save(query_tower_path)


def test_two_tower_model_v2_save(tmpdir, ecommerce_data: Dataset):
    dataset = ecommerce_data
    schema = dataset.schema
    query = mm.Encoder(
        schema.select_by_tag(Tags.USER), mm.MLPBlock([4], no_activation_last_layer=True)
    )
    candidate = mm.Encoder(
        schema.select_by_tag(Tags.ITEM), mm.MLPBlock([4], no_activation_last_layer=True)
    )
    model = mm.TwoTowerModelV2(
        query,
        candidate,
        negative_samplers=["in-batch"],
    )

    _ = testing_utils.model_test(model, dataset, reload_model=True)

    query_tower = model.query_encoder
    query_tower_path = Path(tmpdir) / "query_tower"
    query_tower.save(query_tower_path)


def test_two_tower_model_l2_reg(testing_data: Dataset):
    testing_data.schema = testing_data.schema.excluding_by_name(["event_timestamp"])

    model = mm.TwoTowerModel(
        testing_data.schema,
        query_tower=mm.MLPBlock([2]),
        embedding_options=mm.EmbeddingOptions(
            embedding_dim_default=2,
            embeddings_l2_reg=0.1,
        ),
    )

    _ = model(mm.sample_batch(testing_data, batch_size=10, include_targets=False))

    l2_emb_losses = model.losses

    assert len(l2_emb_losses) == len(
        testing_data.schema.select_by_tag(Tags.CATEGORICAL)
    ), "The number of reg losses should be 4 (for user and item categorical features)"

    for reg_loss in l2_emb_losses:
        assert reg_loss > 0.0


def test_two_tower_model_v2_l2_reg(testing_data: Dataset):
    testing_data.schema = testing_data.schema.excluding_by_name(["event_timestamp"])

    user_schema = testing_data.schema.select_by_tag(Tags.USER)
    user_inputs = mm.InputBlockV2(
        user_schema,
        categorical=mm.Embeddings(
            user_schema.select_by_tag(Tags.CATEGORICAL),
            dim=2,
            l2_batch_regularization_factor=0.1,
        ),
    )
    query = mm.Encoder(user_inputs, mm.MLPBlock([4], no_activation_last_layer=True))

    item_schema = testing_data.schema.select_by_tag(Tags.ITEM)
    item_inputs = mm.InputBlockV2(
        item_schema,
        categorical=mm.Embeddings(
            item_schema.select_by_tag(Tags.CATEGORICAL),
            dim=2,
            l2_batch_regularization_factor=0.1,
        ),
    )
    candidate = mm.Encoder(item_inputs, mm.MLPBlock([4], no_activation_last_layer=True))

    model = mm.TwoTowerModelV2(query, candidate)
    _ = model(mm.sample_batch(testing_data, batch_size=10, include_targets=False))

    l2_emb_losses = model.losses

    assert len(l2_emb_losses) == len(
        testing_data.schema.select_by_tag(Tags.CATEGORICAL)
    ), "The number of reg losses should be 4 (for user and item categorical features)"

    for reg_loss in l2_emb_losses:
        assert reg_loss > 0.0


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_two_tower_model_topk_evaluation(ecommerce_data: Dataset, run_eagerly):
    dataset = ecommerce_data
    schema = dataset.schema
    query = mm.Encoder(
        schema.select_by_tag(Tags.USER), mm.MLPBlock([4], no_activation_last_layer=True)
    )
    candidate = mm.Encoder(
        schema.select_by_tag(Tags.ITEM), mm.MLPBlock([4], no_activation_last_layer=True)
    )
    model = mm.TwoTowerModelV2(
        query,
        candidate,
        negative_samplers=["in-batch"],
    )
    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    _ = testing_utils.model_test(model, dataset)

    # Top-K evaluation
    candidate_features = unique_rows_by_features(ecommerce_data, Tags.ITEM, Tags.ITEM_ID)
    topk_model = model.to_top_k_encoder(candidate_features, k=20, batch_size=16)
    topk_model.compile(run_eagerly=run_eagerly)

    loader = mm.Loader(ecommerce_data, batch_size=32).map(mm.ToTarget(schema, "item_id"))

    metrics = topk_model.evaluate(loader, return_dict=True)
    assert all([metric >= 0 for metric in metrics.values()])


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("logits_pop_logq_correction", [True, False])
@pytest.mark.parametrize("loss", ["categorical_crossentropy", "bpr-max", "binary_crossentropy"])
def test_two_tower_model_with_custom_options(
    ecommerce_data: Dataset,
    run_eagerly,
    logits_pop_logq_correction,
    loss,
):
    from tensorflow.keras import regularizers

    from merlin.models.tf.transforms.bias import PopularityLogitsCorrection
    from merlin.models.utils import schema_utils

    data = ecommerce_data
    data.schema = data.schema.select_by_name(["user_categories", "item_id"])

    metrics = [
        tf.keras.metrics.AUC(from_logits=True, name="auc"),
        mm.RecallAt(5),
        mm.RecallAt(10),
        mm.MRRAt(10),
        mm.NDCGAt(10),
    ]

    post_logits = None
    if logits_pop_logq_correction:
        cardinalities = schema_utils.categorical_cardinalities(data.schema)
        item_id_cardinalities = cardinalities[
            data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        ]
        items_frequencies = tf.sort(
            tf.random.uniform((item_id_cardinalities,), minval=0, maxval=1000, dtype=tf.int32)
        )
        post_logits = PopularityLogitsCorrection(
            items_frequencies,
            schema=data.schema,
        )

    retrieval_task = mm.ItemRetrievalTask(
        samplers=[mm.InBatchSampler()],
        schema=data.schema,
        logits_temperature=0.1,
        post_logits=post_logits,
        store_negative_ids=True,
    )

    model = mm.TwoTowerModel(
        data.schema,
        query_tower=mm.MLPBlock(
            [2],
            activation="relu",
            no_activation_last_layer=True,
            dropout=0.1,
            kernel_regularizer=regularizers.l2(1e-5),
            bias_regularizer=regularizers.l2(1e-6),
        ),
        embedding_options=mm.EmbeddingOptions(
            infer_embedding_sizes=True,
            infer_embedding_sizes_multiplier=3.0,
            infer_embeddings_ensure_dim_multiple_of_8=True,
            embeddings_l2_reg=1e-5,
        ),
        prediction_tasks=retrieval_task,
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly, loss=loss, metrics=metrics)

    losses = model.fit(data, batch_size=50, epochs=1, steps_per_epoch=1)
    assert len(losses.epoch) == 1
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])

    metrics = model.evaluate(data, batch_size=10, item_corpus=data, return_dict=True, steps=1)
    assert set(metrics.keys()) == set(
        [
            "loss",
            "loss_batch",
            "regularization_loss",
            "auc",
            "recall_at_5",
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
        ]
    )


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("logits_pop_logq_correction", [True, False])
@pytest.mark.parametrize("loss", ["categorical_crossentropy", "bpr-max", "binary_crossentropy"])
def test_two_tower_model_v2_with_custom_options(
    ecommerce_data: Dataset,
    run_eagerly,
    logits_pop_logq_correction,
    loss,
):
    from functools import partial

    from tensorflow.keras import regularizers

    from merlin.models.tf.outputs.base import DotProduct
    from merlin.models.tf.transforms.bias import PopularityLogitsCorrection
    from merlin.models.utils import schema_utils

    data = ecommerce_data
    data.schema = data.schema.select_by_name(["user_categories", "item_id"])
    user_schema = data.schema.select_by_tag(Tags.USER)
    item_schema = data.schema.select_by_tag(Tags.ITEM)

    metrics = [
        tf.keras.metrics.AUC(from_logits=True, name="auc"),
        mm.RecallAt(5),
        mm.RecallAt(10),
        mm.MRRAt(10),
        mm.NDCGAt(10),
    ]

    post_logits = None
    if logits_pop_logq_correction:
        cardinalities = schema_utils.categorical_cardinalities(data.schema)
        item_id_cardinalities = cardinalities[
            data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        ]
        items_frequencies = tf.sort(
            tf.random.uniform((item_id_cardinalities,), minval=0, maxval=1000, dtype=tf.int32)
        )
        post_logits = PopularityLogitsCorrection(
            items_frequencies,
            schema=data.schema,
        )

    user_inputs = mm.InputBlockV2(
        user_schema,
        categorical=mm.Embeddings(
            user_schema.select_by_tag(Tags.CATEGORICAL),
            infer_dim_fn=partial(schema_utils.infer_embedding_dim, multiplier=3.0),
            l2_batch_regularization_factor=1.0e-5,
        ),
    )

    tower_block = mm.MLPBlock(
        [2],
        activation="relu",
        no_activation_last_layer=True,
        dropout=0.1,
        kernel_regularizer=regularizers.l2(1e-5),
        bias_regularizer=regularizers.l2(1e-6),
    )
    query = mm.Encoder(user_inputs, tower_block)

    item_inputs = mm.InputBlockV2(
        item_schema,
        categorical=mm.Embeddings(
            item_schema.select_by_tag(Tags.CATEGORICAL),
            infer_dim_fn=partial(schema_utils.infer_embedding_dim, multiplier=3.0),
            l2_batch_regularization_factor=1.0e-5,
        ),
    )
    candidate = mm.Encoder(item_inputs, tower_block.copy())

    output = mm.ContrastiveOutput(
        DotProduct(),
        logits_temperature=0.1,
        post=post_logits,
        negative_samplers="in-batch",
        schema=data.schema.select_by_tag(Tags.ITEM_ID),
        store_negative_ids=True,
    )

    model = mm.TwoTowerModelV2(
        query,
        candidate,
        outputs=output,
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly, loss=loss, metrics=metrics)
    losses = model.fit(data, batch_size=50, epochs=1, steps_per_epoch=1)
    assert len(losses.epoch) == 1
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])

    metrics = model.evaluate(data, batch_size=10, return_dict=True, steps=1)
    assert set(metrics.keys()) == set(
        [
            "loss",
            "loss_batch",
            "regularization_loss",
            "auc",
            "recall_at_5",
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
        ]
    )


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize(
    "loss", ["categorical_crossentropy", "bpr", "bpr-max", "binary_crossentropy"]
)
def test_two_tower_retrieval_model_with_metrics(ecommerce_data: Dataset, run_eagerly, loss):
    ecommerce_data.schema = ecommerce_data.schema.select_by_name(["user_categories", "item_id"])

    metrics = [RecallAt(5), MRRAt(5), NDCGAt(5), AvgPrecisionAt(5), PrecisionAt(5)]
    model = mm.TwoTowerModel(schema=ecommerce_data.schema, query_tower=mm.MLPBlock([4]))
    model.compile(optimizer="adam", run_eagerly=run_eagerly, metrics=metrics, loss=loss)

    # Training
    losses = model.fit(
        ecommerce_data,
        batch_size=10,
        epochs=1,
        steps_per_epoch=1,
        train_metrics_steps=3,
    )
    assert len(losses.epoch) == 1

    # Checking train metrics
    expected_metrics = ["recall_at_5", "mrr_at_5", "ndcg_at_5", "map_at_5", "precision_at_5"]
    expected_loss_metrics = ["loss", "loss_batch", "regularization_loss"]
    expected_metrics_all = expected_metrics + expected_loss_metrics
    assert set(losses.history.keys()) == set(expected_metrics_all)

    # TODO: This fails sometimes now
    # for metric_name in expected_metrics + expected_loss_metrics:
    #     assert len(losses.history[metric_name]) == num_epochs
    #     if metric_name in expected_metrics:
    #         assert losses.history[metric_name][1] >= losses.history[metric_name][0]
    #     elif metric_name in expected_loss_metrics:
    #         assert losses.history[metric_name][1] <= losses.history[metric_name][0]

    metrics = model.evaluate(
        ecommerce_data, batch_size=10, item_corpus=ecommerce_data, return_dict=True, steps=1
    )
    assert set(metrics.keys()) == set(expected_metrics_all)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_two_tower_retrieval_model_with_topk_metrics_aggregator(
    ecommerce_data: Dataset, run_eagerly
):
    ecommerce_data.schema = ecommerce_data.schema.select_by_name(["user_categories", "item_id"])

    metrics_agg = TopKMetricsAggregator(
        RecallAt(5), MRRAt(5), NDCGAt(5), AvgPrecisionAt(5), PrecisionAt(5)
    )
    model = mm.TwoTowerModel(schema=ecommerce_data.schema, query_tower=mm.MLPBlock([4]))
    model.compile(optimizer="adam", run_eagerly=run_eagerly, metrics=[metrics_agg])

    # Training
    losses = model.fit(
        ecommerce_data,
        batch_size=10,
        epochs=1,
        steps_per_epoch=1,
        train_metrics_steps=3,
    )
    assert len(losses.epoch) == 1

    # Checking train metrics
    expected_metrics = ["recall_at_5", "mrr_at_5", "ndcg_at_5", "map_at_5", "precision_at_5"]
    expected_loss_metrics = ["loss", "loss_batch", "regularization_loss"]
    expected_metrics_all = expected_metrics + expected_loss_metrics
    assert set(losses.history.keys()) == set(expected_metrics_all)

    metrics = model.evaluate(
        ecommerce_data, batch_size=10, item_corpus=ecommerce_data, return_dict=True, steps=1
    )
    assert set(metrics.keys()) == set(expected_metrics_all)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_two_tower_retrieval_model_v2_with_topk_metrics_aggregator(
    ecommerce_data: Dataset, run_eagerly
):
    ecommerce_data.schema = ecommerce_data.schema.select_by_name(["user_categories", "item_id"])

    metrics_agg = TopKMetricsAggregator(
        RecallAt(5), MRRAt(5), NDCGAt(5), AvgPrecisionAt(5), PrecisionAt(5)
    )

    query = mm.Encoder(
        ecommerce_data.schema.select_by_tag(Tags.USER),
        mm.MLPBlock([4], no_activation_last_layer=True),
    )
    candidate = mm.Encoder(
        ecommerce_data.schema.select_by_tag(Tags.ITEM),
        mm.MLPBlock([4], no_activation_last_layer=True),
    )

    model = mm.TwoTowerModelV2(query, candidate)
    model.compile(optimizer="adam", run_eagerly=run_eagerly, metrics=[metrics_agg])

    # Training
    losses = model.fit(
        ecommerce_data,
        batch_size=10,
        epochs=1,
        steps_per_epoch=1,
        train_metrics_steps=3,
        validation_data=ecommerce_data,
        validation_steps=3,
    )
    assert len(losses.epoch) == 1

    # Checking train metrics
    expected_metrics = ["recall_at_5", "mrr_at_5", "ndcg_at_5", "map_at_5", "precision_at_5"]
    expected_loss_metrics = ["loss", "loss_batch", "regularization_loss"]
    expected_metrics_all = expected_metrics + expected_loss_metrics
    expected_metrics_valid = [f"val_{k}" for k in expected_metrics_all]
    assert set(losses.history.keys()) == set(expected_metrics_all + expected_metrics_valid)

    metrics = model.evaluate(ecommerce_data, batch_size=10, return_dict=True, steps=1)
    assert set(metrics.keys()) == set(expected_metrics_all)


def test_two_tower_advanced_options(ecommerce_data):
    ecommerce_data.schema = ecommerce_data.schema.select_by_name(["user_id", "item_id"])
    train_ds, eval_ds = ecommerce_data, ecommerce_data
    metrics = retrieval_tests_common.train_eval_two_tower_for_lastfm(
        train_ds,
        eval_ds,
        train_epochs=1,
        train_steps_per_epoch=None,
        eval_steps=None,
        train_batch_size=16,
        eval_batch_size=16,
        topk_metrics_cutoffs="10",
        log_to_wandb=False,
    )
    assert metrics["loss-final"] > 0.0
    assert metrics["runtime_sec-final"] > 0.0
    assert metrics["avg_examples_per_sec-final"] > 0.0
    assert metrics["recall_at_10-final"] > 0.0


def test_mf_advanced_options(ecommerce_data):
    ecommerce_data.schema = ecommerce_data.schema.select_by_name(["user_id", "item_id"])
    train_ds, eval_ds = ecommerce_data, ecommerce_data
    metrics = retrieval_tests_common.train_eval_mf_for_lastfm(
        train_ds,
        eval_ds,
        train_epochs=1,
        train_steps_per_epoch=None,
        eval_steps=None,
        train_batch_size=16,
        eval_batch_size=16,
        topk_metrics_cutoffs="10",
        log_to_wandb=False,
    )
    assert metrics["loss-final"] > 0.0
    assert metrics["runtime_sec-final"] > 0.0
    assert metrics["avg_examples_per_sec-final"] > 0.0
    assert metrics["recall_at_10-final"] > 0.0


# def test_retrieval_evaluation_without_negatives(ecommerce_data: Dataset):
#     model = mm.TwoTowerModel(schema=ecommerce_data.schema, query_tower=mm.MLPBlock([64]))
#     model.compile(optimizer="adam", run_eagerly=True)
#     model.fit(ecommerce_data, batch_size=50)
#     with pytest.raises(ValueError) as exc_info:
#         model.evaluate(ecommerce_data, batch_size=10)
#         assert "You need to specify the set of negatives to use for evaluation" in str(
#             exc_info.value
#         )


@pytest.mark.skip(
    reason="The YoutubeDNNRetrievalModel is outdated, "
    "was never officially released and is going to be deprecated in favor "
    "of YoutubeDNNRetrievalModelV2"
)
def test_youtube_dnn_retrieval(sequence_testing_data: Dataset):
    """This test works both for eager mode and graph mode when
    ran individually. But when both tests are run by pytest
    the last one fails. So somehow pytest is sharing some
    graph state between tests. I keep now only the graph mode test"""

    to_remove = ["event_timestamp"] + (
        sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
        .select_by_tag(Tags.CONTINUOUS)
        .column_names
    )
    sequence_testing_data.schema = sequence_testing_data.schema.excluding_by_name(to_remove)

    model = mm.YoutubeDNNRetrievalModel(
        schema=sequence_testing_data.schema,
        top_block=mm.MLPBlock([2]),
        l2_normalization=True,
        sampled_softmax=True,
        num_sampled=100,
        embedding_options=mm.EmbeddingOptions(
            embedding_dim_default=2,
        ),
    )
    model.compile(optimizer="adam", run_eagerly=False)

    prepare_features = mm.PrepareFeatures(sequence_testing_data.schema)

    class LastInteractionAsTarget(tf.keras.layers.Layer):
        def call(self, inputs, **kwargs):
            inputs = prepare_features(inputs)
            items = inputs["item_id_seq"]
            _items = items[:, :-1]
            targets = tf.reshape(items[:, -1:].to_tensor(), (1, -1))

            inputs["item_id_seq"] = _items

            return inputs, targets

    dataloader = mm.Loader(sequence_testing_data, batch_size=50)

    losses = model.fit(dataloader, epochs=1, pre=LastInteractionAsTarget())

    assert losses is not None


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("target_augmentation", [mm.SequencePredictLast, mm.SequencePredictRandom])
def test_youtube_dnn_retrieval_v2(sequence_testing_data: Dataset, run_eagerly, target_augmentation):
    # remove sequential continuous features because second dimension (=[None]) is raising an error
    # in the `compute_output_shape` of  `ConcatFeatures`)
    to_remove = ["event_timestamp"] + (
        sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
        .select_by_tag(Tags.CONTINUOUS)
        .column_names
    )
    sequence_testing_data.schema = sequence_testing_data.schema.excluding_by_name(to_remove)

    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    target_augmentation = target_augmentation(schema=seq_schema, target=target)

    model = mm.YoutubeDNNRetrievalModelV2(
        schema=sequence_testing_data.schema, top_block=mm.MLPBlock([32]), num_sampled=1000
    )

    dataloader = mm.Loader(sequence_testing_data, batch_size=50)

    _, losses = testing_utils.model_test(
        model,
        dataloader,
        reload_model=True,
        run_eagerly=run_eagerly,
        fit_kwargs=dict(pre=target_augmentation),
    )

    assert losses is not None


def test_two_tower_v2_export_embeddings(
    ecommerce_data: Dataset,
):
    user_schema = ecommerce_data.schema.select_by_tag(Tags.USER_ID)
    candidate_schema = ecommerce_data.schema.select_by_tag(Tags.ITEM_ID)

    query = mm.Encoder(user_schema, mm.MLPBlock([8]))
    candidate = mm.Encoder(candidate_schema, mm.MLPBlock([8]))
    model = mm.TwoTowerModelV2(
        query_tower=query, candidate_tower=candidate, negative_samplers=["in-batch"]
    )

    model, _ = testing_utils.model_test(model, ecommerce_data, reload_model=False)

    queries = model.query_embeddings(ecommerce_data, batch_size=16, index=Tags.USER_ID).compute()
    _check_embeddings(queries, 100, 8, "user_id")

    candidates = model.candidate_embeddings(
        ecommerce_data, batch_size=16, index=Tags.ITEM_ID
    ).compute()
    _check_embeddings(candidates, 100, 8, "item_id")


def test_mf_v2_export_embeddings(
    ecommerce_data: Dataset,
):
    model = mm.MatrixFactorizationModelV2(
        ecommerce_data.schema,
        dim=8,
        negative_samplers="in-batch",
    )

    model, _ = testing_utils.model_test(model, ecommerce_data, reload_model=False)

    queries = model.query_embeddings(ecommerce_data, batch_size=16, index=Tags.USER_ID).compute()
    _check_embeddings(queries, 100, 8, "user_id")

    candidates = model.candidate_embeddings(
        ecommerce_data, batch_size=16, index=Tags.ITEM_ID
    ).compute()
    _check_embeddings(candidates, 100, 8, "item_id")


def _check_embeddings(embeddings, extected_len, num_dim=8, index_name=None):
    import pandas as pd

    if not isinstance(embeddings, pd.DataFrame):
        embeddings = embeddings.to_pandas()

    assert isinstance(embeddings, pd.DataFrame)
    assert list(embeddings.columns) == [str(i) for i in range(num_dim)]
    assert len(embeddings.index) == extected_len
    assert embeddings.index.name == index_name


def test_youtube_dnn_v2_export_embeddings(sequence_testing_data: Dataset):
    to_remove = ["event_timestamp"] + (
        sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
        .select_by_tag(Tags.CONTINUOUS)
        .column_names
    )
    sequence_testing_data.schema = sequence_testing_data.schema.excluding_by_name(to_remove)

    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_next = mm.SequencePredictLast(schema=seq_schema, target=target)

    model = mm.YoutubeDNNRetrievalModelV2(
        schema=sequence_testing_data.schema, top_block=mm.MLPBlock([32]), num_sampled=1000
    )

    dataloader = mm.Loader(sequence_testing_data, batch_size=50)
    model, _ = testing_utils.model_test(
        model, dataloader, reload_model=False, fit_kwargs=dict(pre=predict_next)
    )

    candidates = model.candidate_embeddings().compute()
    assert list(candidates.columns) == [str(i) for i in range(32)]
    assert len(candidates.index) == 51997

    # Export the query embeddings is raising an error from dask, related to
    # the support of multi-hot input features.

    # queries = model.query_embeddings(
    #    sequence_testing_data, batch_size=10, index=Tags.USER_ID
    # ).compute()
    # _check_embeddings(queries, 100, 32, "user_id")


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_youtube_dnn_topk_evaluation(sequence_testing_data: Dataset, run_eagerly):
    to_remove = ["event_timestamp"] + (
        sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
        .select_by_tag(Tags.CONTINUOUS)
        .column_names
    )
    sequence_testing_data.schema = sequence_testing_data.schema.excluding_by_name(to_remove)

    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_next = mm.SequencePredictLast(schema=seq_schema, target=target)

    model = mm.YoutubeDNNRetrievalModelV2(
        schema=sequence_testing_data.schema, top_block=mm.MLPBlock([32]), num_sampled=1000
    )

    dataloader = mm.Loader(sequence_testing_data, batch_size=50)

    model, _ = testing_utils.model_test(
        model, dataloader, reload_model=False, fit_kwargs=dict(pre=predict_next)
    )

    # Top-K evaluation
    topk_model = model.to_top_k_encoder(k=20)
    topk_model.compile(run_eagerly=run_eagerly)

    metrics = topk_model.evaluate(dataloader, return_dict=True, pre=predict_next)
    assert all([metric >= 0 for metric in metrics.values()])
