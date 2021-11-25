import pytest
from merlin_standard_lib import Tag

from merlin_models.data.dataset import ParquetDataset
from merlin_models.data.testing.dataset import tabular_testing_data


def test_tabular_sequence_testing_data():
    assert isinstance(tabular_testing_data, ParquetDataset)

    assert tabular_testing_data.schema_path.endswith("merlin_models/data/testing/testing.json")
    assert len(tabular_testing_data.schema) == 11


def test_tf_tensors_generation_cpu():
    tf = pytest.importorskip("tensorflow")
    s = tabular_testing_data.schema
    tensors = tabular_testing_data.tf_synthetic_tensors(
        num_rows=100, min_session_length=5, max_session_length=50
    )

    assert tensors["user_id"].shape == (100,)
    assert tensors["user_age"].dtype == tf.float32
    for val in s.select_by_tag(Tag.LIST).filter_columns_from_dict(tensors).values():
        assert val.shape == (100, 50)

    for val in s.select_by_tag(Tag.CATEGORICAL).filter_columns_from_dict(tensors).values():
        assert val.dtype == tf.int32
        assert tf.reduce_max(val) < 1000


def test_torch_tensors_generation_cpu():
    torch = pytest.importorskip("torch")
    s = tabular_testing_data.schema
    tensors = tabular_testing_data.torch_synthetic_tensors(
        num_rows=100, min_session_length=5, max_session_length=50
    )

    assert tensors["user_id"].shape == (100,)
    assert tensors["user_age"].dtype == torch.float32
    for val in s.select_by_tag(Tag.LIST).filter_columns_from_dict(tensors).values():
        assert val.shape == (100, 50)

    for val in s.select_by_tag(Tag.CATEGORICAL).filter_columns_from_dict(tensors).values():
        assert val.dtype == torch.int64
        assert val.max() < 1000
