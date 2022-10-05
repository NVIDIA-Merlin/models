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
from typing import Union

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from merlin.io import Dataset
from merlin.schema import Tags


def dataset_to_coo(dataset: Dataset):
    """Converts a merlin.io.Dataset object to a scipy coo matrix"""
    user_id_column = dataset.schema.select_by_tag(Tags.USER_ID).first.name
    item_id_column = dataset.schema.select_by_tag(Tags.ITEM_ID).first.name

    columns = [user_id_column, item_id_column]
    target_column = None
    target = dataset.schema.select_by_tag(Tags.TARGET)

    if len(target) > 1:
        raise ValueError(
            "Found more than one column tagged Tags.TARGET in the dataset schema."
            f" Expected a single target column but found  {target.column_names}"
        )

    elif len(target) == 1:
        target_column = target.first.name
        columns.append(target_column)

    df = dataset.to_ddf()[columns].compute(scheduler="synchronous")

    userids = _to_numpy(df[user_id_column])
    itemids = _to_numpy(df[item_id_column])
    targets = _to_numpy(df[target_column]) if target_column else np.ones(len(userids))
    return coo_matrix((targets.astype("float32"), (userids, itemids)))


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
