import pytest

from merlin.models.utils.schema_utils import get_embedding_size_from_cardinality


@pytest.mark.parametrize(
    "cardinality_x_expected_dim",
    [
        (1, 5),
        (5, 8),
        (10, 9),
        (20, 11),
        (50, 14),
        (100, 16),
        (500, 24),
        (1000, 29),
        (5000, 43),
        (10000, 50),
        (50000, 75),
        (100000, 89),
        (500000, 133),
        (1000000, 159),
        (5000000, 237),
        (10000000, 282),
    ],
)
def test_get_embedding_sizes_from_cardinality(cardinality_x_expected_dim):
    multiplier = 5.0

    cardinality, expected_dim = cardinality_x_expected_dim
    dim = get_embedding_size_from_cardinality(cardinality, multiplier)
    assert dim == expected_dim


@pytest.mark.parametrize(
    "cardinality_x_expected_dim",
    [
        (1, 8),
        (5, 8),
        (10, 16),
        (20, 16),
        (50, 16),
        (100, 16),
        (500, 24),
        (1000, 32),
        (5000, 48),
        (10000, 56),
        (50000, 80),
        (100000, 96),
        (500000, 136),
        (1000000, 160),
        (5000000, 240),
        (10000000, 288),
    ],
)
def test_get_embedding_size_from_cardinality_multiple_of_8(cardinality_x_expected_dim):
    multiplier = 5.0

    cardinality, expected_dim = cardinality_x_expected_dim
    dim = get_embedding_size_from_cardinality(cardinality, multiplier, ensure_multiple_of_8=True)
    assert dim == expected_dim
