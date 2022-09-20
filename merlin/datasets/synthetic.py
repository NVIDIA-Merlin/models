#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import difflib
import logging
import os
import pathlib
from pathlib import Path
from random import randint
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

import merlin.io
from merlin.models.utils import schema_utils
from merlin.schema import Schema, Tags
from merlin.schema.io.tensorflow_metadata import TensorflowMetadata

LOG = logging.getLogger("merlin-models")
HERE = pathlib.Path(__file__).parent
KNOWN_DATASETS: Dict[str, Path] = {
    "e-commerce": HERE / "ecommerce/small",
    "e-commerce-large": HERE / "ecommerce/large",
    "music-streaming": HERE / "entertainment" / "music_streaming",
    "social": HERE / "social",
    "testing": HERE / "testing",
    "sequence-testing": HERE / "testing" / "sequence_testing",
    "movielens-25m": HERE / "entertainment/movielens/25m",
    "movielens-1m": HERE / "entertainment/movielens/1m",
    "movielens-1m-raw-ratings": HERE / "entertainment/movielens/1m-raw/ratings/",
    "movielens-100k": HERE / "entertainment/movielens/100k",
    "criteo": HERE / "advertising/criteo/transformed",
    "aliccp": HERE / "ecommerce/aliccp/transformed",
    "aliccp-raw": HERE / "ecommerce/aliccp/raw",
}


def generate_data(
    input: Union[Schema, Path, str],
    num_rows: int,
    set_sizes: Sequence[float] = (1.0,),
    min_session_length=5,
    max_session_length=None,
    device="cpu",
) -> Union[merlin.io.Dataset, Tuple[merlin.io.Dataset, ...]]:
    """
    Generate synthetic data from a schema or one of the known datasets.

    Known fully synthetic datasets:
    - e-commerce
    - e-commerce-large
    - music-streaming
    - social
    - testing
    - sequence-testing

    Based on real datasets:
    - criteo
    - aliccp
    - aliccp-raw


    Parameters
    ----------
    input: Union[Schema, Path, str]
        The schema, path to a dataset or name of a known dataset.
    num_rows: int
        The number of rows to generate.
    set_sizes: Sequence[float], default=(1.0,)
        This parameter allows outputting multiple datasets, where each
        dataset is a subset of the original dataset.

        Example::
            train, valid = generate_data(input, 10000, (0.8, 0.2))
    min_session_length: int
        The minimum number of events in a session.
    max_session_length: int
        The maximum number of events in a session.
    device: str
        The device to use for the data generation.
        Supported values: {'cpu', 'gpu'}

    Returns
    -------
    merlin.io.Dataset
    """

    schema: Schema
    if isinstance(input, str):
        if input in KNOWN_DATASETS:
            input = KNOWN_DATASETS[input]
        elif not os.path.exists(input):
            closest_match = difflib.get_close_matches(input, KNOWN_DATASETS.keys(), n=1)
            raise ValueError(f"Unknown dataset {input}, did you mean: {closest_match[0]}?")

        schema = _get_schema(input)
    elif isinstance(input, Schema):
        schema = input
    else:
        raise ValueError(f"Unknown input type: {type(input)}")

    df = generate_user_item_interactions(
        schema, num_rows, min_session_length, max_session_length, device=device
    )

    if list(set_sizes) != [1.0]:
        num_rows = df.shape[0]
        output_datasets = []
        start_i = 0
        for set_size in set_sizes:
            num_rows_set = int(num_rows * set_size)
            end_i = start_i + num_rows_set
            set_df = df.iloc[start_i:end_i]
            start_i = end_i
            output_datasets.append(set_df)

        return tuple([merlin.io.Dataset(d, schema=schema) for d in output_datasets])

    return merlin.io.Dataset(df, schema=schema)


def generate_user_item_interactions(
    schema: Schema,
    num_interactions: int,
    min_session_length: int = 5,
    max_session_length: Optional[int] = None,
    device: str = "cpu",
):
    """
    Util function to generate synthetic data of user-item interactions from a schema object,
    it supports the generation of conditional user, item and session features.

    The schema should include a few tags:
    - `Tags.SESSION_ID` to tag the session-id feature.
    - `Tags.USER_ID` for user-id feature.
    - `Tags.ITEM_ID` for item-id feature.

    It supports both, GPU-based and CPU-based, generation.

    Parameters:
    ----------
    schema: Schema
        schema object describing the columns to generate.
    num_interactions: int
        number of interaction rows to generate.
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
       otherwise, it returns a cudf dataframe.
    """
    if device == "cpu":
        import numpy as _array
        import pandas as _frame
    else:
        import cudf as _frame
        import cupy as _array
    data = _frame.DataFrame()
    processed_cols = []
    # get session cols
    session_id_cols = list(schema.select_by_tag(Tags.SESSION_ID))
    if session_id_cols:
        session_id_col = session_id_cols[0]
        data[session_id_col.name] = _array.clip(
            _array.random.lognormal(3.0, 1.0, num_interactions).astype(_array.int32),
            1,
            session_id_col.int_domain.max,
        ).astype(str(session_id_col.dtype))

        features = list(schema.select_by_tag(Tags.SESSION).remove_by_tag(Tags.SESSION_ID))
        data = generate_conditional_features(
            data,
            features,
            session_id_col,
            min_session_length=min_session_length,
            max_session_length=max_session_length,
            device=device,
        )
        processed_cols += [f.name for f in features] + [session_id_col.name]

    # get USER cols
    user_id_cols = list(schema.select_by_tag(Tags.USER_ID))
    if user_id_cols:
        user_id_col = user_id_cols[0]
        data[user_id_col.name] = _array.clip(
            _array.random.lognormal(3.0, 1.0, num_interactions).astype(_array.int32),
            1,
            user_id_col.int_domain.max,
        ).astype(str(user_id_col.dtype))
        features = list(schema.select_by_tag(Tags.USER).remove_by_tag(Tags.USER_ID))
        data = generate_conditional_features(
            data,
            features,
            user_id_col,
            min_session_length=min_session_length,
            max_session_length=max_session_length,
            device=device,
        )
        processed_cols += [f.name for f in features] + [user_id_col.name]

    # get ITEM cols
    item_schema = schema.select_by_tag(Tags.ITEM_ID)
    if not len(item_schema) > 0:
        raise ValueError("Item ID column is required")
    item_id_col = item_schema.first

    is_list_feature = item_id_col.is_list
    if not is_list_feature:
        shape = num_interactions
    else:
        shape = (num_interactions, item_id_col.value_count.max)  # type: ignore
    tmp = _array.clip(
        _array.random.lognormal(3.0, 1.0, shape).astype(_array.int32),
        1,
        item_id_col.int_domain.max,
    ).astype(str(item_id_col.dtype))
    if isinstance(shape, int):
        data[item_id_col.name] = tmp
    else:
        data[item_id_col.name] = list(tmp)
    features = list(schema.select_by_tag(Tags.ITEM).remove_by_tag(Tags.ITEM_ID))
    data = generate_conditional_features(
        data,
        features,
        item_id_col,
        min_session_length=min_session_length,
        max_session_length=max_session_length,
        device=device,
    )
    processed_cols += [f.name for f in features] + [item_id_col.name]

    # Get remaining features
    remaining = schema.without(processed_cols)

    for feature in remaining.select_by_tag(Tags.BINARY_CLASSIFICATION):
        data[feature.name] = _array.random.randint(0, 2, num_interactions).astype(
            str(feature.dtype)
        )

    for feature in remaining.remove_by_tag(Tags.BINARY_CLASSIFICATION):
        is_int_feature = np.issubdtype(feature.dtype, np.integer)
        is_list_feature = feature.is_list
        if is_list_feature:
            data[feature.name] = generate_random_list_feature(
                feature, num_interactions, min_session_length, max_session_length, device
            )

        elif is_int_feature:
            domain = feature.int_domain
            min_value, max_value = (domain.min, domain.max) if domain else (0, 1)

            data[feature.name] = _array.random.randint(
                min_value, max_value, num_interactions, dtype=str(feature.dtype)
            )

        else:
            domain = feature.float_domain
            min_value, max_value = (domain.min, domain.max) if domain else (0.0, 1.0)

            data[feature.name] = _array.random.uniform(min_value, max_value, num_interactions)

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
        is_int_feature = np.issubdtype(feature.dtype, np.integer)
        is_list_feature = feature.is_list

        if is_list_feature:
            data[feature.name] = generate_random_list_feature(
                feature, num_interactions, min_session_length, max_session_length, device
            )

        elif is_int_feature:
            if not feature.int_domain:
                raise ValueError(
                    "Int domain is required for conditional features, got {}".format(feature)
                )

            data[feature.name] = _frame.cut(
                data[parent_feature.name],
                feature.int_domain.max - 1,
                labels=list(range(1, feature.int_domain.max)),
            ).astype(str(feature.dtype))

        else:
            if feature.float_domain:
                _min, _max = feature.float_domain.min, feature.float_domain.max
            else:
                logging.warning(
                    "Couldn't find the float-domain for feature {}, assuming [0, 1]".format(feature)
                )
                _min, _max = 0.0, 1.0

            data[feature.name] = _array.random.uniform(_min, _max, num_interactions)

    return data


def generate_random_list_feature(
    feature,
    num_interactions,
    min_session_length: int = 5,
    max_session_length: Optional[int] = None,
    device="cpu",
):
    if device == "cpu":
        import numpy as _array
    else:
        import cupy as _array

    is_int_feature = np.issubdtype(feature.dtype, np.integer)
    if is_int_feature:
        if max_session_length:
            padded_array = []
            for _ in range(num_interactions):
                list_length = randint(min_session_length, max_session_length)
                actual_values = _array.random.randint(
                    1, feature.int_domain.max, (list_length,)
                ).astype(str(feature.dtype))

                padded_array.append(
                    _array.pad(
                        actual_values,
                        [0, max_session_length - list_length],
                        constant_values=0,
                    )
                )
            return list(_array.stack(padded_array, axis=0))
        else:
            list_length = feature.value_count.max
            return list(
                _array.random.randint(
                    1, feature.int_domain.max, (num_interactions, list_length)
                ).astype(str(feature.dtype))
            )
    else:
        if max_session_length:
            padded_array = []
            for _ in range(num_interactions):
                list_length = randint(min_session_length, max_session_length)
                actual_values = _array.random.uniform(
                    feature.float_domain.min, feature.float_domain.max, (list_length,)
                )

                padded_array.append(
                    _array.pad(
                        actual_values,
                        [0, max_session_length - list_length],
                        constant_values=0,
                    )
                )
            return list(_array.stack(padded_array, axis=0))
        else:
            list_length = feature.value_count.max
            return list(
                _array.random.uniform(
                    feature.float_domain.min,
                    feature.float_domain.max,
                    (num_interactions, list_length),
                )
            )


def _get_schema(path: Union[str, Path]) -> Schema:
    path = str(path)
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "schema.json")):
            path = os.path.join(path, "schema.json")
        else:
            path = os.path.join(path, "schema.pbtxt")

    if path.endswith(".pb") or path.endswith(".pbtxt"):
        return TensorflowMetadata.from_proto_text_file(
            os.path.dirname(path), os.path.basename(path)
        ).to_merlin_schema()

    return schema_utils.tensorflow_metadata_json_to_schema(path)
