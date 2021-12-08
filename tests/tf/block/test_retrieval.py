import pytest

from merlin_models.data.synthetic import SyntheticData

tf = pytest.importorskip("tensorflow")
ml = pytest.importorskip("merlin_models.tf")


def test_matrix_factorization_block(music_streaming_data: SyntheticData):
    mf = ml.MatrixFactorizationBlock(music_streaming_data.schema, dim=128)

    outputs = mf(music_streaming_data.tf_tensor_dict)

    assert "user_id" in outputs
    assert "item_id" in outputs
