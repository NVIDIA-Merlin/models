import pytest

from merlin_models.data.synthetic import generate_recsys_data, synthetic_retrieval_schema

pd = pytest.importorskip("pandas")


def test_generate_item_interactions():
    data = generate_recsys_data(500, synthetic_retrieval_schema)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 500
    assert list(data.columns) == [
        "session_id",
        "purchase",
        "price",
        "item_id",
        "category",
        "item_recency",
        "genres",
        "user_id",
        "age",
        "sex",
    ]
    expected_dtypes = {
        "session_id": "int64",
        "item_id": "int64",
        "user_id": "int64",
        "age": "float64",
        "sex": "int64",
        "genres": "int64",
        "category": "int64",
        "item_recency": "float64",
        "purchase": "int64",
        "price": "float64",
    }

    assert all(
        val == expected_dtypes[key] for key, val in dict(data.dtypes).items() if key != "genres"
    )
