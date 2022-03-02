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
import logging
import os
import pathlib
import tempfile
from pathlib import Path
from random import randint
from typing import Optional, Union

import numpy as np
import pandas as pd

from merlin.models.utils.schema import (
    schema_to_tensorflow_metadata_json,
    tensorflow_metadata_json_to_schema,
)
from merlin.schema import Schema, Tags
from merlin.schema.io.tensorflow_metadata import TensorflowMetadata

LOG = logging.getLogger("merlin-models")
HERE = pathlib.Path(__file__).parent


def _read_data(path: str, num_rows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if num_rows:
        df = df.iloc[:num_rows]

    return df


class SyntheticData:
    DATASETS = {
        "ecommerce": HERE / "ecommerce/small",
        "e-commerce": HERE / "ecommerce/small",
        "ecommerce-large": HERE / "ecommerce/large",
        "e-commerce-large": HERE / "ecommerce/large",
        "music_streaming": HERE / "music_streaming",
        "music-streaming": HERE / "music_streaming",
        "social": HERE / "social",
        "testing": HERE / "testing",
        "sequence_testing": HERE / "sequence_testing",
    }
    FILE_NAME = "data.parquet"

    def __init__(
        self,
        data: Union[str, Path],
        device: str = "cpu",
        schema_file_name="schema.json",
        num_rows=None,
        read_data_fn=_read_data,
    ):
        if not os.path.isdir(data):
            data = self.DATASETS[data]

        self._dir = str(data)
        self.data_path = os.path.join(self._dir, self.FILE_NAME)
        self.schema_path = os.path.join(self._dir, schema_file_name)
        self._schema = self.read_schema(self.schema_path)
        self.device = device
        self._num_rows = num_rows
        self._read_data_fn = read_data_fn

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        output_dir: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        num_rows=100,
        min_session_length=5,
        max_session_length=None,
    ) -> "SyntheticData":
        if not output_dir:
            output_dir = tempfile.mkdtemp()

        if not os.path.exists(os.path.join(output_dir, "schema.json")):
            schema_to_tensorflow_metadata_json(schema, os.path.join(output_dir, "schema.json"))

        output = cls(output_dir, device=device)
        output.generate_interactions(num_rows, min_session_length, max_session_length)

        return output

    @classmethod
    def read_schema(cls, path: Union[str, Path]) -> Schema:
        path = str(path)
        _schema_path = os.path.join(path, "schema.json") if os.path.isdir(path) else path

        if _schema_path.endswith(".pb") or _schema_path.endswith(".pbtxt"):
            return TensorflowMetadata.from_proto_text_file(
                os.path.dirname(_schema_path), os.path.basename(_schema_path)
            ).to_merlin_schema()

        return tensorflow_metadata_json_to_schema(_schema_path)

    @property
    def schema(self) -> Schema:
        return self._schema

    def generate_interactions(
        self, num_rows=100, min_session_length=5, max_session_length=None, save=True
    ):
        data = generate_user_item_interactions(
            self.schema, num_rows, min_session_length, max_session_length, self.device
        )
        if save:
            data.to_parquet(os.path.join(self._dir, self.FILE_NAME))

        return data

    @property
    def tf_tensor_dict(self):
        import tensorflow as tf

        data = self.dataframe.to_dict("list")

        return {key: tf.convert_to_tensor(value) for key, value in data.items()}

    @property
    def tf_features_and_targets(self):
        return self._pull_out_targets(self.tf_tensor_dict)

    @property
    def torch_tensor_dict(self):
        import torch

        data = self.dataframe.to_dict("list")

        return {key: torch.tensor(value).to(self.device) for key, value in data.items()}

    @property
    def torch_features_and_targets(self):
        return self._pull_out_targets(self.torch_tensor_dict)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._read_data_fn(self.data_path, num_rows=self._num_rows)

    def tf_dataloader(self, batch_size=50):
        # TODO: return tf NVTabular loader

        import tensorflow as tf

        data = self.dataframe.to_dict("list")
        tensors = {key: tf.convert_to_tensor(value) for key, value in data.items()}
        dataset = tf.data.Dataset.from_tensor_slices(self._pull_out_targets(tensors)).batch(
            batch_size=batch_size
        )

        return dataset

    def torch_dataloader(self, batch_size=50):
        """return torch NVTabular loader"""
        raise NotImplementedError()

    def _pull_out_targets(self, inputs):
        target_names = self.schema.select_by_tag(Tags.TARGET).column_names
        targets = {}

        for target_name in target_names:
            targets[target_name] = inputs.pop(target_name, None)

        return inputs, targets


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
    session_id_col = list(schema.select_by_tag(Tags.SESSION_ID))
    if session_id_col:
        session_id_col = session_id_col[0]
        data[session_id_col.name] = _array.clip(
            _array.random.lognormal(3.0, 1.0, num_interactions).astype(_array.int32),
            1,
            session_id_col.int_domain.max,
        ).astype(_array.int64)

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
        ).astype(_array.int64)
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
    item_id_col = list(schema.select_by_tag(Tags.ITEM_ID))[0]

    is_list_feature = item_id_col.is_list
    if not is_list_feature:
        shape = num_interactions
    else:
        shape = (num_interactions, item_id_col.value_count.max)
    data[item_id_col.name] = (
        _array.clip(
            _array.random.lognormal(3.0, 1.0, shape).astype(_array.int32),
            1,
            item_id_col.int_domain.max,
        )
        .astype(_array.int64)
        .tolist()
    )
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
        data[feature.name] = _array.random.randint(0, 2, num_interactions).astype(_array.int64)

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
                min_value, max_value, num_interactions
            ).astype(_array.int64)

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
            data[feature.name] = _frame.cut(
                data[parent_feature.name],
                feature.int_domain.max - 1,
                labels=list(range(1, feature.int_domain.max)),
            ).astype(_array.int64)

        else:
            data[feature.name] = _array.random.uniform(
                feature.float_domain.min, feature.float_domain.max, num_interactions
            )

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
                ).astype(_array.int64)

                padded_array.append(
                    _array.pad(
                        actual_values,
                        [0, max_session_length - list_length],
                        constant_values=0,
                    )
                )
            return _array.stack(padded_array, axis=0).tolist()
        else:
            list_length = feature.value_count.max
            return (
                _array.random.randint(1, feature.int_domain.max, (num_interactions, list_length))
                .astype(_array.int64)
                .tolist()
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
            return _array.stack(padded_array, axis=0).tolist()
        else:
            list_length = feature.value_count.max
            return _array.random.uniform(
                feature.float_domain.min,
                feature.float_domain.max,
                (num_interactions, list_length),
            ).tolist()
