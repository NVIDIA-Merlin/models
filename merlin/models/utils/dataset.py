#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from merlin.io import Dataset
from merlin.schema import Schema, Tags


def dataset_to_coo(
    dataset: Dataset,
    target_column: Optional[str] = None,
):
    """Converts a merlin.io.Dataset object to a scipy coo matrix"""
    user_id_column = get_user_id_column_name(dataset)
    item_id_column = get_item_id_column_name(dataset)

    columns = [user_id_column, item_id_column]

    if not target_column:
        target_column = get_target_column_name(dataset)

    if target_column:
        columns.append(target_column)

    df = dataset.to_ddf()[columns].compute(scheduler="synchronous")

    userids = _to_numpy(df[user_id_column])
    itemids = _to_numpy(df[item_id_column])
    targets = _to_numpy(df[target_column]) if target_column else np.ones(len(userids))
    return coo_matrix((targets.astype("float32"), (userids, itemids)))


def get_schema(dataset_or_schema: Union[Dataset, Schema]) -> Schema:
    if isinstance(dataset_or_schema, Dataset):
        schema = dataset_or_schema.schema
    elif isinstance(dataset_or_schema, Schema):
        schema = dataset_or_schema
    else:
        raise ValueError(f"Cannot get schema from {type(dataset_or_schema)}.")
    return schema


def get_user_id_column_name(dataset_or_schema: Union[Dataset, Schema]) -> str:
    schema = get_schema(dataset_or_schema)
    return schema.select_by_tag(Tags.USER_ID).first.name


def get_item_id_column_name(dataset_or_schema: Union[Dataset, Schema]) -> str:
    schema = get_schema(dataset_or_schema)
    return schema.select_by_tag(Tags.ITEM_ID).first.name


def get_target_column_name(dataset_or_schema: Union[Dataset, Schema]) -> Optional[str]:
    target_column = None
    schema = get_schema(dataset_or_schema)
    target = schema.select_by_tag(Tags.TARGET)
    if len(target) > 1:
        raise ValueError(
            "Found more than one column tagged Tags.TARGET in the dataset schema."
            f" Expected a single target column but found  {target.column_names}"
        )
    elif len(target) == 1:
        target_column = target.first.name
    return target_column


def unique_rows_by_features(
    dataset: Dataset, features_tag: Union[str, Tags], grouping_tag: Union[str, Tags] = Tags.ID
) -> Dataset:
    """
    Select unique rows from a Dataset. Returns columns specified by `features_tag`
     that are unique based on the columns specified by the `grouping_tag`.

    Parameters
    ----------
    dataset : ~merlin.io.Dataset
        Dataset to transform
    features_tag : ~merlin.schema.Tags
        Tag representing the columns to return in the new Dataset
    grouping_tag : ~merlin.schema.Tags
        Tag representing the columns to check for uniqueness.
        Default: Tags.ID

    Returns
    -------
        Dataset
    """
    warnings.warn(
        "`unique_rows_by_features` is deprecated and will be removed in a future version. "
        "Please use `unique_by_tag` instead.",
        DeprecationWarning,
    )
    return unique_by_tag(dataset, features_tag, grouping_tag)


def unique_by_tag(
    dataset: Dataset, features_tag: Union[str, Tags], grouping_tag: Union[str, Tags] = Tags.ID
) -> Dataset:
    """
    Select unique rows from a Dataset. Returns columns specified by `features_tag`
     that are unique based on the columns specified by the `grouping_tag`.

    Parameters
    ----------
    dataset : ~merlin.io.Dataset
        Dataset to transform
    features_tag : ~merlin.schema.Tags
        Tag representing the columns to return in the new Dataset
    grouping_tag : ~merlin.schema.Tags
        Tag representing the columns to check for uniqueness.
        Default: Tags.ID

    Returns
    -------
        Dataset
    """
    ddf = dataset.to_ddf()

    features_schema = dataset.schema.select_by_tag(features_tag)
    columns = features_schema.column_names

    if columns:
        id_col = features_schema.select_by_tag(grouping_tag).first.name
        ddf = ddf[columns].drop_duplicates(id_col, keep="first")

    return Dataset(ddf, schema=features_schema)


def _to_numpy(series):
    """converts a pandas or cudf series to a numpy array"""
    if isinstance(series, pd.Series):
        return series.values
    else:
        return series.values_host
