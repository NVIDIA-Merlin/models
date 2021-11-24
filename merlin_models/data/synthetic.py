import logging

import merlin_standard_lib as msl
import numpy as np
import pandas as pd
from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.utils.proto_utils import has_field

LOG = logging.getLogger("transformers4rec")


def generate_recsys_data(num_interactions: int, schema: Schema) -> pd.DataFrame:
    """
    Util function to generate synthetic data for retrieval and ranking models from a schema object.
    It supports the generation of user and item features.

    The schema should include a few tags:
    - `Tag.SESSION_ID` to tag the session-id feature.
    - `Tag.USER` for users features.
    - `Tag.ITEM` for item features.

    Parameters:
    ----------
    num_interactions: int
        number of interaction rows to generate.
    schema: Schema
        schema object describing the columns to generate.

    Returns
    -------
    data: pd.DataFrame
        Pandas dataframe with synthetic generated data.
    """
    # get session cols
    session_id_col = schema.select_by_tag(Tag.SESSION_ID).feature[0]
    data = pd.DataFrame(
        np.clip(
            np.random.lognormal(3.0, 1.0, num_interactions).astype(np.int32),
            1,
            session_id_col.int_domain.max,
        ).astype(np.int64),
        columns=["session_id"],
    ).astype(np.int64)

    features = schema.select_by_tag(Tag.SESSION).feature
    data = generate_conditional_features(data, features, session_id_col)

    # get ITEM cols
    item_id_col = schema.select_by_tag(Tag.ITEM_ID).feature[0]
    data[item_id_col.name] = np.clip(
        np.random.lognormal(3.0, 1.0, num_interactions).astype(np.int32),
        1,
        item_id_col.int_domain.max,
    ).astype(np.int64)
    features = schema.select_by_tag(Tag.ITEM).feature
    data = generate_conditional_features(data, features, item_id_col)

    # get USER cols
    user_id_col = schema.select_by_tag(Tag.USER_ID).feature[0]
    data[user_id_col.name] = np.clip(
        np.random.lognormal(3.0, 1.0, num_interactions).astype(np.int32),
        1,
        user_id_col.int_domain.max,
    ).astype(np.int64)
    features = schema.select_by_tag(Tag.USER).feature
    data = generate_conditional_features(data, features, user_id_col)

    return data


def generate_conditional_features(data, features, parent_feature):
    """
    Generate features conditioned by the value of `parent_feature` feature
    """
    num_interactions = data.shape[0]
    for feature in features:
        is_int_feature = has_field(feature, "int_domain")
        is_list_feature = has_field(feature, "value_count")
        if is_int_feature:
            if is_list_feature:
                list_length = feature.value_count.max
                data[feature.name] = (
                    np.random.randint(1, feature.int_domain.max, (num_interactions, list_length))
                    .astype(np.int64)
                    .tolist()
                )
            else:
                data[feature.name] = pd.cut(
                    data[parent_feature.name],
                    bins=feature.int_domain.max - 1,
                    labels=np.arange(1, feature.int_domain.max),
                ).astype(np.int64)
        else:
            if is_list_feature:
                list_length = feature.value_count.max
                data[feature.name] = (
                    np.random.uniform(
                        feature.float_domain.min,
                        feature.float_domain.max,
                        (num_interactions, list_length),
                    )
                    .astype(np.int64)
                    .tolist()
                )
            else:
                data[feature.name] = np.random.uniform(
                    feature.float_domain.min, feature.float_domain.max, num_interactions
                )
    return data


synthetic_retrieval_schema = Schema(
    [
        msl.ColumnSchema.create_categorical("session_id", num_items=10000, tags=[Tag.SESSION_ID]),
        msl.ColumnSchema.create_categorical("user_id", num_items=1000, tags=[Tag.USER_ID]),
        msl.ColumnSchema.create_continuous(
            "age",
            min_value=0,
            max_value=1,
            tags=[Tag.USER],
        ),
        msl.ColumnSchema.create_categorical(
            "sex",
            num_items=3,
            tags=[Tag.USER],
        ),
        msl.ColumnSchema.create_categorical(
            "item_id",
            num_items=10000,
            tags=[Tag.ITEM_ID],
        ),
        msl.ColumnSchema.create_categorical(
            "category",
            num_items=100,
            tags=[Tag.ITEM],
        ),
        msl.ColumnSchema.create_continuous(
            "item_recency",
            min_value=0,
            max_value=1,
            tags=[Tag.ITEM],
        ),
        msl.ColumnSchema.create_categorical(
            "genres",
            num_items=10,
            tags=[Tag.ITEM, Tag.LIST],
            value_count=msl.schema.ValueCount(1, 10),
        ),
        msl.ColumnSchema.create_categorical(
            "purchase", num_items=3, tags=[Tag.SESSION, Tag.BINARY_CLASSIFICATION]
        ),
        msl.ColumnSchema.create_continuous(
            "price", min_value=0, max_value=1, tags=[Tag.SESSION, Tag.REGRESSION]
        ),
    ]
)
