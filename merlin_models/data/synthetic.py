import logging
import os
import pathlib
from pathlib import Path
from random import randint
from typing import Optional, Union

import pandas as pd
from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.utils.proto_utils import has_field

LOG = logging.getLogger("merlin-models")
HERE = pathlib.Path(__file__).parent


class SyntheticData:
    def __init__(self, schema: Union[str, Path, Schema], device: str = "cpu"):
        if isinstance(schema, Schema):
            self._schema = schema
        else:
            self.schema_path = str(schema)
            if self.schema_path.endswith(".pb") or self.schema_path.endswith(".pbtxt"):
                self._schema = Schema().from_proto_text(self.schema_path)
            else:
                self._schema = Schema().from_json(self.schema_path)
        self.device = device

    @property
    def schema(self) -> Schema:
        return self._schema

    def generate_interactions(
        self, num_rows=100, min_session_length=5, max_session_length=None, save_path=None
    ):
        data = generate_user_item_interactions(
            self.schema, num_rows, min_session_length, max_session_length, self.device
        )
        if save_path:
            data.to_parquet(save_path)

        return data

    def tf_tensors(
        self,
        num_rows=100,
        min_session_length=5,
        max_session_length=None,
    ):
        import tensorflow as tf

        data = self.generate_interactions(num_rows, min_session_length, max_session_length)
        if self.device != "cpu":
            data = data.to_pandas()
        data = data.to_dict("list")

        return {key: tf.convert_to_tensor(value) for key, value in data.items()}

    def torch_tensors(
        self,
        num_rows=100,
        min_session_length=5,
        max_session_length=None,
    ):
        import torch

        data = self.generate_interactions(num_rows, min_session_length, max_session_length)
        if self.device != "cpu":
            data = data.to_pandas()
        data = data.to_dict("list")

        return {key: torch.tensor(value).to(self.device) for key, value in data.items()}


class SyntheticDataset(SyntheticData):
    def __init__(
        self,
        dir: Union[str, Path],
        parquet_file_name="data.parquet",
        schema_file_name="schema.json",
        schema_path: Optional[str] = None,
        device="cpu",
    ):
        super(SyntheticDataset, self).__init__(
            schema_path or os.path.join(str(dir), schema_file_name), device=device
        )
        self.path = os.path.join(str(dir), parquet_file_name)

    @classmethod
    def create_ecommerce_data(cls) -> "SyntheticDataset":
        """
        Create a synthetic ecommerce dataset.
        """
        return cls(dir=HERE / "ecommerce")

    @classmethod
    def create_testing_data(cls) -> "SyntheticDataset":
        """
        Create a synthetic ecommerce dataset.
        """
        return cls(dir=HERE / "testing")

    @classmethod
    def create_social_data(cls) -> "SyntheticDataset":
        """
        Create a synthetic ecommerce dataset.
        """
        return cls(dir=HERE / "social")

    @classmethod
    def create_music_streaming_data(cls) -> "SyntheticDataset":
        """
        Create a synthetic ecommerce dataset.
        """
        return cls(dir=HERE / "music_streaming")

    def dataframe(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)

    def get_tf_dataloader(self, batch_size=50):
        # TODO: return tf NVTabular loader

        import tensorflow as tf

        data = pd.read_parquet(self.path).to_dict("list")
        tensors = {key: tf.convert_to_tensor(value) for key, value in data.items()}
        dataset = tf.data.Dataset.from_tensor_slices(self._pull_out_targets(tensors)).batch(
            batch_size=batch_size
        )

        return dataset

    def get_torch_dataloader(self):
        """return torch NVTabular loader"""
        raise NotImplementedError()

    def _pull_out_targets(self, inputs):
        target_names = self.schema.select_by_tag("target").column_names
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
    - `Tag.SESSION_ID` to tag the session-id feature.
    - `Tag.USER_ID` for user-id feature.
    - `Tag.ITEM_ID` for item-id feature.

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
        processed_cols.append([f.name for f in features] + [session_id_col.name])

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
        processed_cols.append([f.name for f in features] + [user_id_col.name])

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
    processed_cols.append([f.name for f in features] + [item_id_col.name])

    # Get remaining features
    remaining = schema.remove_by_name(processed_cols)
    for feature in remaining:
        is_int_feature = has_field(feature, "int_domain")
        is_list_feature = has_field(feature, "value_count")
        if is_list_feature:
            data[feature.name] = generate_random_list_feature(
                feature, num_interactions, min_session_length, max_session_length, device
            )

        elif is_int_feature:
            data[feature.name] = _array.random.randint(
                1, feature.int_domain.max, num_interactions
            ).astype(_array.int64)

        else:
            data[feature.name] = _array.random.uniform(
                feature.float_domain.min, feature.float_domain.max, num_interactions
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

    is_int_feature = has_field(feature, "int_domain")
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
