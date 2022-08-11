import os

import pytest

import merlin.io
from merlin.datasets.entertainment import get_movielens
from merlin.datasets.synthetic import generate_data

MAYBE_DATA_DIR = os.environ.get("INPUT_DATA_DIR", None)


def test_synthetic_music_data():
    dataset = generate_data("music-streaming", 100)

    assert isinstance(dataset, merlin.io.Dataset)
    assert dataset.num_rows == 100
    assert len(dataset.schema) == 13


def test_movielens_25m_data():
    dataset = generate_data("movielens-25m", 100)

    assert isinstance(dataset, merlin.io.Dataset)
    assert dataset.num_rows == 100
    assert len(dataset.schema) == 8


def test_movielens_1m_data():
    dataset = generate_data("movielens-1m", 100)

    assert isinstance(dataset, merlin.io.Dataset)
    assert dataset.num_rows == 100
    assert len(dataset.schema) == 16


def test_movielens_100k_data():
    dataset = generate_data("movielens-100k", 100)

    assert isinstance(dataset, merlin.io.Dataset)
    assert dataset.num_rows == 100
    assert len(dataset.schema) == 11


@pytest.mark.skipif(
    MAYBE_DATA_DIR is None,
    reason="No data-dir available, pass it through env variable $INPUT_DATA_DIR",
)
@pytest.mark.parametrize("variant", ["ml-25m", "ml-1m", "ml-100k"])
def test_get_movielens(tmp_path, variant):
    data_path = os.path.join(MAYBE_DATA_DIR, "movielens")

    train, valid = get_movielens(data_path, variant=variant, overwrite=True)

    assert isinstance(train, merlin.io.Dataset)
    assert isinstance(valid, merlin.io.Dataset)


def test_unsupported_variant():
    with pytest.raises(ValueError) as excinfo:
        get_movielens("/tmp", variant="unknown")
    assert "MovieLens dataset variant not supported" in str(excinfo.value)
