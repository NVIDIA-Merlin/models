import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.metrics.ranking import AvgPrecisionAt, MRRAt, NDCGAt, PrecisionAt, RecallAt
from merlin.schema import Tags


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_matrix_factorization_model(music_streaming_data: Dataset, run_eagerly, num_epochs=2):
    music_streaming_data.schema = music_streaming_data.schema.remove_by_tag(Tags.TARGET)

    model = mm.MatrixFactorizationModel(music_streaming_data.schema, dim=64)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(music_streaming_data, batch_size=50, epochs=num_epochs)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_two_tower_model(music_streaming_data: Dataset, run_eagerly, num_epochs=2):
    music_streaming_data.schema = music_streaming_data.schema.remove_by_tag(Tags.TARGET)

    model = mm.TwoTowerModel(music_streaming_data.schema, query_tower=mm.MLPBlock([512, 256]))
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(music_streaming_data, batch_size=50, epochs=num_epochs)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("logits_pop_logq_correction", [True, False])
def test_two_tower_model_with_custom_options(
    music_streaming_data: Dataset, run_eagerly, logits_pop_logq_correction, num_epochs=2
):

    from tensorflow.keras import regularizers

    from merlin.models.tf.blocks.core.transformations import PopularityLogitsCorrection
    from merlin.models.utils import schema_utils

    music_streaming_data.schema = music_streaming_data.schema.remove_by_tag(Tags.TARGET)
    metrics = [
        tf.keras.metrics.AUC(from_logits=True),
        mm.RecallAt(10),
        mm.RecallAt(50),
        mm.MRRAt(50),
        mm.NDCGAt(50),
    ]

    post_logits = None
    if logits_pop_logq_correction:
        cardinalities = schema_utils.categorical_cardinalities(music_streaming_data.schema)
        item_id_cardinalities = cardinalities[
            music_streaming_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        ]
        items_frequencies = tf.sort(
            tf.random.uniform((item_id_cardinalities,), minval=0, maxval=1000, dtype=tf.int32)
        )
        post_logits = PopularityLogitsCorrection(
            items_frequencies,
            schema=music_streaming_data.schema,
        )

    retrieval_task = mm.ItemRetrievalTask(
        samplers=[mm.InBatchSampler()],
        schema=music_streaming_data.schema,
        loss="bpr-max",
        logits_temperature=0.1,
        metrics=metrics,
        post_logits=post_logits,
        store_negative_ids=True,
    )

    model = mm.TwoTowerModel(
        music_streaming_data.schema,
        query_tower=mm.MLPBlock(
            [512, 256],
            activation="relu",
            no_activation_last_layer=True,
            dropout=0.1,
            kernel_regularizer=regularizers.l2(1e-5),
            bias_regularizer=regularizers.l2(1e-6),
        ),
        embedding_options=mm.EmbeddingOptions(
            infer_embedding_sizes=True,
            infer_embedding_sizes_multiplier=3.0,
        ),
        prediction_tasks=retrieval_task,
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(music_streaming_data, batch_size=50, epochs=num_epochs)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("loss", ["categorical_crossentropy", "bpr", "binary_crossentropy"])
def test_two_tower_retrieval_model_with_metrics(ecommerce_data: Dataset, run_eagerly, loss):
    ecommerce_data.schema = ecommerce_data.schema.remove_by_tag(Tags.TARGET)

    metrics = [RecallAt(5), MRRAt(5), NDCGAt(5), AvgPrecisionAt(5), PrecisionAt(5)]
    model = mm.TwoTowerModel(
        schema=ecommerce_data.schema,
        query_tower=mm.MLPBlock([128, 64]),
        samplers=[mm.InBatchSampler()],
        metrics=metrics,
        loss=loss,
    )
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    # Training
    num_epochs = 2
    losses = model.fit(
        ecommerce_data,
        batch_size=10,
        epochs=num_epochs,
        train_metrics_steps=3,
        validation_data=ecommerce_data,
        validation_steps=3,
    )
    assert len(losses.epoch) == num_epochs

    # Checking train metrics
    expected_metrics = ["recall_at_5", "mrr_at_5", "ndcg_5", "map_at_5", "precision_at_5"]
    expected_loss_metrics = ["loss", "regularization_loss", "total_loss"]
    expected_metrics_all = expected_metrics + expected_loss_metrics
    assert len(expected_metrics_all) == len(
        set(losses.history.keys()).intersection(set(expected_metrics_all))
    )
    for metric_name in expected_metrics + expected_loss_metrics:
        assert len(losses.history[metric_name]) == num_epochs
        if metric_name in expected_metrics:
            assert losses.history[metric_name][1] >= losses.history[metric_name][0]
        elif metric_name in expected_loss_metrics:
            assert losses.history[metric_name][1] <= losses.history[metric_name][0]

    metrics = model.evaluate(ecommerce_data, batch_size=10, item_corpus=ecommerce_data)

    assert len(metrics) == 8


# def test_retrieval_evaluation_without_negatives(ecommerce_data: Dataset):
#     model = mm.TwoTowerModel(schema=ecommerce_data.schema, query_tower=mm.MLPBlock([64]))
#     model.compile(optimizer="adam", run_eagerly=True)
#     model.fit(ecommerce_data, batch_size=50)
#     with pytest.raises(ValueError) as exc_info:
#         model.evaluate(ecommerce_data, batch_size=10)
#         assert "You need to specify the set of negatives to use for evaluation" in str(
#             exc_info.value
#         )


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_youtube_dnn_retrieval(
    sequence_testing_data: Dataset,
    run_eagerly: bool,
):
    model = mm.YoutubeDNNRetrievalModel(schema=sequence_testing_data.schema, max_seq_length=4)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(sequence_testing_data, batch_size=50, epochs=2)

    assert len(losses.epoch) == 2
    for metric in losses.history.keys():
        assert type(losses.history[metric]) is list
    batch = mm.sample_batch(
        sequence_testing_data, batch_size=10, include_targets=False, to_dense=True
    )
    out = model({k: tf.cast(v, tf.int64) for k, v in batch.items()})

    assert out.shape[-1] == 51997
