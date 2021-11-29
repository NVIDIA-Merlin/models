import pytest

from merlin_models.data.synthetic import generate_user_item_interactions


def test_generate_item_interactions_cpu(tabular_schema):
    pd = pytest.importorskip("pandas")
    data = generate_user_item_interactions(
        tabular_schema.remove_by_name("event_timestamp"), num_interactions=500
    )

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 500
    assert list(data.columns) == [
        "user_id",
        "user_country",
        "user_age",
        "item_id",
        "item_age_days_norm",
        "event_hour_sin",
        "event_hour_cos",
        "event_weekday_sin",
        "event_weekday_cos",
        "categories",
    ]
    expected_dtypes = {
        "user_id": "int64",
        "user_country": "int64",
        "user_age": "float64",
        "item_id": "int64",
        "item_age_days_norm": "float64",
        "event_hour_sin": "float64",
        "event_hour_cos": "float64",
        "event_weekday_sin": "float64",
        "event_weekday_cos": "float64",
        "categories": "int64",
    }

    assert all(
        val == expected_dtypes[key] for key, val in dict(data.dtypes).items() if key != "categories"
    )


def test_generate_item_interactions_gpu(tabular_schema):
    cudf = pytest.importorskip("cudf")
    data = generate_user_item_interactions(tabular_schema, num_interactions=500, device="cuda")

    assert isinstance(data, cudf.DataFrame)
    assert len(data) == 500
    assert list(data.columns) == [
        "user_id",
        "user_country",
        "user_age",
        "item_id",
        "item_age_days_norm",
        "event_hour_sin",
        "event_hour_cos",
        "event_weekday_sin",
        "event_weekday_cos",
        "categories",
        "event_timestamp",
    ]
    expected_dtypes = {
        "user_id": "int64",
        "user_country": "int64",
        "user_age": "float64",
        "item_id": "int64",
        "item_age_days_norm": "float64",
        "event_hour_sin": "float64",
        "event_hour_cos": "float64",
        "event_weekday_sin": "float64",
        "event_weekday_cos": "float64",
        "categories": "int64",
        "event_timestamp": "float64",
    }

    assert all(
        val == expected_dtypes[key] for key, val in dict(data.dtypes).items() if key != "categories"
    )
