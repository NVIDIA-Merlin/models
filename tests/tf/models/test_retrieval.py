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
    # Setting up evaluation
    model.set_retrieval_candidates_for_evaluation(ecommerce_data)
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

    _ = model.evaluate(ecommerce_data, batch_size=10)


def test_retrieval_evaluation_without_negatives(ecommerce_data: Dataset):
    model = mm.TwoTowerModel(schema=ecommerce_data.schema, query_tower=mm.MLPBlock([64]))
    model.compile(optimizer="adam", run_eagerly=True)
    model.fit(ecommerce_data, batch_size=50)
    with pytest.raises(ValueError) as exc_info:
        model.evaluate(ecommerce_data, batch_size=10)
        assert "You need to specify the set of negatives to use for evaluation" in str(
            exc_info.value
        )


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
