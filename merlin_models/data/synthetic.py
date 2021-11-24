import logging
from random import randint
from typing import Optional

from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.utils.proto_utils import has_field

LOG = logging.getLogger("transformers4rec")


def generate_user_item_interactions(
    num_interactions: int,
    schema: Schema,
    min_session_length: int = 5,
    max_session_length: Optional[int] = None,
    device: str = "cpu",
):
    """
    Util function to generate synthetic data of user-item interactions from a schema object,
    it supports the generation of conditional user, item and session features.

    The schema should include a few tags:
    - `Tag.SESSION_ID` to tag the session-id feature.
    - `Tag.USER_ID` for user-id feature.
    - `Tag.ITEM_ID` for item-id feature.

    It supports both, GPU-based and CPU-based, generation.

    Parameters:
    ----------
    num_interactions: int
        number of interaction rows to generate.
    schema: Schema
        schema object describing the columns to generate.
    max_session_length: Optional[int]
        The maximum length of the multi-hot/sequence features
    min_session_length: int
        The minimum length of the multi-hot/sequence features
    device: str
        device to use for generating data.

    Returns
    -------
    data: DataFrame
       DataFrame with synthetic generated rows,
       If `cpu`, the function returns a Pandas dataframe,
       otherwise, it retuns a cudf dataframe.
    """
    if device == "cpu":
        import numpy as _array
        import pandas as _frame
    else:
        import cudf as _frame
        import cupy as _array
    data = _frame.DataFrame()

    # get session cols
    session_id_col = schema.select_by_tag(Tag.SESSION_ID)
    if session_id_col:
        session_id_col = session_id_col.feature[0]
        data[session_id_col.name] = _array.clip(
            _array.random.lognormal(3.0, 1.0, num_interactions).astype(_array.int32),
            1,
            session_id_col.int_domain.max,
        ).astype(_array.int64)

        features = schema.select_by_tag(Tag.SESSION).remove_by_tag(Tag.SESSION_ID).feature
        data = generate_conditional_features(
            data,
            features,
            session_id_col,
            min_session_length=min_session_length,
            max_session_length=max_session_length,
            device=device,
        )

    # get USER cols
    user_id_col = schema.select_by_tag(Tag.USER_ID).feature
    if user_id_col:
        user_id_col = user_id_col[0]
        data[user_id_col.name] = _array.clip(
            _array.random.lognormal(3.0, 1.0, num_interactions).astype(_array.int32),
            1,
            user_id_col.int_domain.max,
        ).astype(_array.int64)
        features = schema.select_by_tag(Tag.USER).remove_by_tag(Tag.USER_ID).feature
        data = generate_conditional_features(
            data,
            features,
            user_id_col,
            min_session_length=min_session_length,
            max_session_length=max_session_length,
            device=device,
        )

    # get ITEM cols
    item_id_col = schema.select_by_tag(Tag.ITEM_ID).feature[0]
    data[item_id_col.name] = _array.clip(
        _array.random.lognormal(3.0, 1.0, num_interactions).astype(_array.int32),
        1,
        item_id_col.int_domain.max,
    ).astype(_array.int64)
    features = schema.select_by_tag(Tag.ITEM).remove_by_tag(Tag.ITEM_ID).feature
    data = generate_conditional_features(
        data,
        features,
        item_id_col,
        min_session_length=min_session_length,
        max_session_length=max_session_length,
        device=device,
    )
    return data


def generate_conditional_features(
    data,
    features,
    parent_feature,
    min_session_length: int = 5,
    max_session_length: Optional[int] = None,
    device="cpu",
):
    """
    Generate features conditioned by the value of `parent_feature` feature
    """
    if device == "cpu":
        import numpy as _array
        import pandas as _frame
    else:
        import cudf as _frame
        import cupy as _array

    num_interactions = data.shape[0]
    for feature in features:
        is_int_feature = has_field(feature, "int_domain")
        is_list_feature = has_field(feature, "value_count")

        if is_int_feature:
            if is_list_feature:
                if max_session_length:
                    padded_array = []
                    for _ in range(num_interactions):
                        list_length = randint(min_session_length, max_session_length)
                        actual_values = _array.random.randint(
                            1, feature.int_domain.max, (list_length,)
                        ).astype(_array.int64)

                        padded_array.append(
                            _array.pad(
                                actual_values,
                                [0, max_session_length - list_length],
                                constant_values=0,
                            )
                        )
                    data[feature.name] = _array.stack(padded_array, axis=0).tolist()
                else:
                    list_length = feature.value_count.max
                    data[feature.name] = (
                        _array.random.randint(
                            1, feature.int_domain.max, (num_interactions, list_length)
                        )
                        .astype(_array.int64)
                        .tolist()
                    )

            else:
                data[feature.name] = _frame.cut(
                    data[parent_feature.name],
                    feature.int_domain.max - 1,
                    labels=list(range(1, feature.int_domain.max)),
                ).astype(_array.int64)

        else:
            if is_list_feature:
                list_length = feature.value_count.max
                data[feature.name] = (
                    _array.random.uniform(
                        feature.float_domain.min,
                        feature.float_domain.max,
                        (num_interactions, list_length),
                    )
                    .astype(_array.int64)
                    .tolist()
                )
            else:
                data[feature.name] = _array.random.uniform(
                    feature.float_domain.min, feature.float_domain.max, num_interactions
                )
    return data
