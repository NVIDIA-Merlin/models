import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.models.data.synthetic import SyntheticData
from merlin.models.tf.metrics.ranking import AvgPrecisionAt, MRRAt, NDCGAt, PrecisionAt, RecallAt
from merlin.schema import Tags


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_matrix_factorization_model(music_streaming_data: SyntheticData, run_eagerly, num_epochs=2):
    music_streaming_data._schema = music_streaming_data.schema.remove_by_tag(Tags.TARGET)

    model = mm.MatrixFactorizationModel(music_streaming_data.schema, dim=64)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(music_streaming_data.dataset, batch_size=50, epochs=num_epochs)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_two_tower_model(music_streaming_data: SyntheticData, run_eagerly, num_epochs=2):
    music_streaming_data._schema = music_streaming_data.schema.remove_by_tag(Tags.TARGET)

    model = mm.TwoTowerModel(music_streaming_data.schema, query_tower=mm.MLPBlock([512, 256]))
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    data = music_streaming_data.dataset
    losses = model.fit(data, batch_size=50, epochs=num_epochs)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_two_tower_retrieval_model_with_metrics(ecommerce_data: SyntheticData, run_eagerly):
    ecommerce_data._schema = ecommerce_data.schema.remove_by_tag(Tags.TARGET)

    metrics = [RecallAt(10), MRRAt(10), NDCGAt(10), AvgPrecisionAt(10), PrecisionAt(10)]
    model = mm.TwoTowerModel(
        schema=ecommerce_data.schema,
        query_tower=mm.MLPBlock([128, 64]),
        samplers=[mm.InBatchSampler()],
        metrics=metrics,
        loss="categorical_crossentropy",
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    # Training
    num_epochs = 2
    losses = model.fit(
        ecommerce_data.tf_dataloader(batch_size=50),
        epochs=num_epochs,
        train_metrics_steps=3,
        # validation_data=ecommerce_data.tf_dataloader(batch_size=50),
        # validation_steps=3,
    )
    assert len(losses.epoch) == num_epochs

    # Checking train metrics
    expected_metrics = ["recall_at_10", "mrr_at_10", "ndcg_10", "map_at_10", "precision_at_10"]
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

    # Setting up evaluation
    item_features = ecommerce_data.schema.select_by_tag(Tags.ITEM).column_names
    item_dataset = ecommerce_data.dataframe[item_features].drop_duplicates()
    item_dataset = Dataset(item_dataset)
    model = model.load_topk_evaluation(item_dataset, k=20)
    _ = model.evaluate(ecommerce_data.tf_dataloader(batch_size=50))


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_youtube_dnn_retrieval(
    sequence_testing_data: SyntheticData,
    run_eagerly: bool,
):
    model = mm.YoutubeDNNRetrievalModel(schema=sequence_testing_data.schema, max_seq_length=4)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(sequence_testing_data.dataset, batch_size=50, epochs=2)

    assert len(losses.epoch) == 2
    for metric in losses.history.keys():
        assert type(losses.history[metric]) is list
    out = model({k: tf.cast(v, tf.int64) for k, v in sequence_testing_data.tf_tensor_dict.items()})

    assert out.shape[-1] == 51997
