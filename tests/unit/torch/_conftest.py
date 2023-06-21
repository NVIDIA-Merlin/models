import numpy as np
import pytest

from merlin.schema import ColumnSchema, Tags


@pytest.fixture
def item_id_col_schema() -> ColumnSchema:
    return ColumnSchema(
        "item_id",
        dtype=np.int32,
        properties={"domain": {"min": 0, "max": 10, "name": "item_id"}},
        tags=[Tags.CATEGORICAL, Tags.ITEM_ID],
    )


@pytest.fixture
def user_id_col_schema() -> ColumnSchema:
    return ColumnSchema(
        "user_id",
        dtype=np.int32,
        properties={"domain": {"min": 0, "max": 20, "name": "user_id"}},
        tags=[Tags.CATEGORICAL, Tags.USER_ID],
    )
