import pytest

import merlin_models.tf as ml
from merlin_models.data.synthetic import SyntheticData


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
