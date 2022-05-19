import os

import pytest

import merlin.io
from merlin.datasets.advertising import get_criteo
from merlin.datasets.synthetic import generate_data

MAYBE_DATA_DIR = os.environ.get("INPUT_DATA_DIR", None)


def test_synthetic_criteo_data():
    dataset = generate_data("criteo", 100)

    assert isinstance(dataset, merlin.io.Dataset)
    assert dataset.num_rows == 100
    assert len(dataset.schema) == 40


@pytest.mark.skipif(
    MAYBE_DATA_DIR is None,
    reason="No data-dir available, pass it through env variable $INPUT_DATA_DIR",
)
def test_get_criteo(tmp_path):
    data_path = os.path.join(MAYBE_DATA_DIR, "criteo")

    train, valid = get_criteo(data_path, num_days=2)

    assert isinstance(train, merlin.io.Dataset)
    assert isinstance(valid, merlin.io.Dataset)
