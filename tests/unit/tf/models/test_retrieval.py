from pathlib import Path

import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.metrics.topk import (
    AvgPrecisionAt,
    MRRAt,
    NDCGAt,
    PrecisionAt,
    RecallAt,
    TopKMetricsAggregator,
)
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags
from tests.common.tf.retrieval import retrieval_tests_common


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_matrix_factorization_model(music_streaming_data: Dataset, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(["user_id", "item_id"])

    model = mm.MatrixFactorizationModel(music_streaming_data.schema, dim=4)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(music_streaming_data, batch_size=50, epochs=1, steps_per_epoch=1)
    assert len(losses.epoch) == 1
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


def test_matrix_factorization_model_l2_reg(testing_data: Dataset):
    model = mm.MatrixFactorizationModel(testing_data.schema, dim=4, embeddings_l2_reg=0.1)

    _ = model(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    l2_emb_losses = model.losses

    assert (
        len(l2_emb_losses) == 2
    ), "The number of reg losses should be 2 (for user and item embeddings)"

    for reg_loss in l2_emb_losses:
        assert reg_loss > 0.0


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

    query_features = testing_utils.get_model_inputs(
        music_streaming_data.schema.select_by_tag(Tags.USER), ["user_genres"]
    )
    testing_utils.test_model_signature(model.first.query_block(), query_features, ["output_1"])

    item_features = testing_utils.get_model_inputs(
        music_streaming_data.schema.select_by_tag(Tags.ITEM),
    )
    testing_utils.test_model_signature(model.first.item_block(), item_features, ["output_1"])


def test_two_tower_model_save(tmpdir, ecommerce_data: Dataset):
    dataset = ecommerce_data
    schema = dataset.schema
    model = mm.TwoTowerModel(
        schema,
        query_tower=mm.MLPBlock([4], no_activation_last_layer=True),
        samplers=[mm.InBatchSampler()],
        embedding_options=mm.EmbeddingOptions(infer_embedding_sizes=True),
    )

    testing_utils.model_test(model, dataset, reload_model=False)

    query_tower = model.retrieval_block.query_block()
    query_tower_path = Path(tmpdir) / "query_tower"
    query_tower.save(query_tower_path)


def test_two_tower_model_l2_reg(testing_data: Dataset):
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
        validation_data=ecommerce_data,
        validation_steps=3,
    )
    assert len(losses.epoch) == 1

    # Checking train metrics
    expected_metrics = ["recall_at_5", "mrr_at_5", "ndcg_at_5", "map_at_5", "precision_at_5"]
    expected_loss_metrics = ["loss", "regularization_loss"]
    expected_metrics_all = expected_metrics + expected_loss_metrics
    expected_metrics_valid = [f"val_{k}" for k in expected_metrics_all]
    assert set(losses.history.keys()) == set(expected_metrics_all + expected_metrics_valid)

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
        validation_data=ecommerce_data,
        validation_steps=3,
    )
    assert len(losses.epoch) == 1

    # Checking train metrics
    expected_metrics = ["recall_at_5", "mrr_at_5", "ndcg_at_5", "map_at_5", "precision_at_5"]
    expected_loss_metrics = ["loss", "regularization_loss"]
    expected_metrics_all = expected_metrics + expected_loss_metrics
    expected_metrics_valid = [f"val_{k}" for k in expected_metrics_all]
    assert set(losses.history.keys()) == set(expected_metrics_all + expected_metrics_valid)

    metrics = model.evaluate(
        ecommerce_data, batch_size=10, item_corpus=ecommerce_data, return_dict=True, steps=1
    )
    assert set(metrics.keys()) == set(expected_metrics_all)


def test_two_tower_advanced_options(ecommerce_data):
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


def test_youtube_dnn_retrieval(sequence_testing_data: Dataset):
    """This test works both for eager mode and graph mode when
    ran individually. But when both tests are run by pytest
    the last one fails. So somehow pytest is sharing some
    graph state between tests. I keep now only the graph mode test"""

    to_remove = (
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

    as_ragged = mm.ListToRagged()

    def last_interaction_as_target(inputs, targets):
        inputs = as_ragged(inputs)
        items = inputs["item_id_seq"]
        _items = items[:, :-1]
        targets = items[:, -1:].flat_values

        inputs["item_id_seq"] = _items

        return inputs, targets

    dataloader = mm.Loader(
        sequence_testing_data, batch_size=50, transform=last_interaction_as_target
    )

    losses = model.fit(dataloader, epochs=1)

    assert losses is not None
