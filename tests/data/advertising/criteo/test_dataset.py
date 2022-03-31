import os

import pytest

import merlin.io
from merlin.models.data.advertising.criteo import get_criteo

# This is the path to the raw ali-ccp dataset
MAYBE_DATA_DIR = os.environ.get("INPUT_DATA_DIR", None)


@pytest.mark.skipif(
    MAYBE_DATA_DIR is None,
    reason="No data-dir available, pass it through env variable $INPUT_DATA_DIR",
)
def test_get_criteo(tmp_path):
    data_path = os.path.join(MAYBE_DATA_DIR, "criteo")

    train, valid = get_criteo(data_path, num_days=2)

    assert isinstance(train, merlin.io.Dataset)
    assert isinstance(valid, merlin.io.Dataset)
