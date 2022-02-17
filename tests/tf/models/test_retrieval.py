import pytest

import merlin_models.tf as ml
from merlin_models.data.synthetic import SyntheticData
from merlin_standard_lib import Tag


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_matrix_factorization_model(music_streaming_data: SyntheticData, run_eagerly, num_epochs=2):
    music_streaming_data._schema = music_streaming_data.schema.remove_by_tag(Tag.TARGETS)

    model = ml.MatrixFactorizationModel(music_streaming_data.schema, dim=64)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(music_streaming_data.tf_dataloader(batch_size=50), epochs=num_epochs)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_two_tower_model(music_streaming_data: SyntheticData, run_eagerly, num_epochs=2):
    music_streaming_data._schema = music_streaming_data.schema.remove_by_tag(Tag.TARGETS)

    model = ml.TwoTowerModel(music_streaming_data.schema, query_tower=ml.MLPBlock([512, 256]))
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(music_streaming_data.tf_dataloader(batch_size=50), epochs=num_epochs)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_youtube_dnn_retrieval(
    sequence_testing_data: SyntheticData,
    run_eagerly: bool,
):
    model = ml.YoutubeDNNRetrievalModel(schema=sequence_testing_data.schema)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(sequence_testing_data.tf_dataloader(batch_size=50), epochs=2)

    assert len(losses.epoch) == 2
    for metric in losses.history.keys():
        assert type(losses.history[metric]) is list
    out = model(sequence_testing_data.tf_tensor_dict)

    assert out.shape[-1] == 51997
