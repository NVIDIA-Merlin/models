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
import math
import os
from typing import Dict, Optional

import numpy as np

from merlin.schema import ColumnSchema, Schema, Tags, TagsType
from merlin.schema.io.tensorflow_metadata import TensorflowMetadata


def select_targets(schema: Schema, extra_tags: Optional[TagsType] = None) -> Schema:
    out = schema.select_by_tag(Tags.BINARY_CLASSIFICATION)
    out += schema.select_by_tag(Tags.TARGET)
    out += schema.select_by_tag(Tags.REGRESSION)

    if extra_tags:
        out += schema.select_by_tag(extra_tags)

    return out


def schema_to_tensorflow_metadata_json(schema, path=None):
    json = TensorflowMetadata.from_merlin_schema(schema).to_json()
    if path:
        with open(path, "w") as o:
            o.write(json)
    return json


def tensorflow_metadata_json_to_schema(value):
    if os.path.isfile(value):
        value = open(value).read()
    return TensorflowMetadata.from_json(value).to_merlin_schema()


def create_categorical_column(
    name,
    num_items,
    dtype=np.int32,
    domain_name=None,
    tags=None,
    properties=None,
    min_value_count=None,
    max_value_count=None,
):
    properties = properties or {}
    if not domain_name:
        domain_name = name
    if num_items:
        properties["domain"] = {"name": domain_name, "min": 0, "max": num_items}

    is_list, is_ragged = False, False
    value_count = {}
    if min_value_count is not None:
        value_count["min"] = min_value_count
    if max_value_count is not None:
        value_count["max"] = max_value_count
    if value_count:
        properties["value_count"] = value_count
        is_list = True
        is_ragged = min_value_count != max_value_count

    tags = tags or []
    if Tags.CATEGORICAL not in tags:
        tags.append(Tags.CATEGORICAL)

    return ColumnSchema(
        name=name,
        tags=tags,
        dtype=dtype,
        properties=properties,
        is_list=is_list,
        is_ragged=is_ragged,
    )


def create_continuous_column(
    name,
    dtype=np.float32,
    tags=None,
    properties=None,
    min_value=None,
    max_value=None,
):
    properties = properties or {}
    domain = {}
    if min_value is not None:
        domain["min"] = min_value
    if max_value is not None:
        domain["max"] = max_value
    if domain:
        properties["domain"] = domain

    tags = tags or []
    if Tags.CONTINUOUS not in tags:
        tags.append(Tags.CONTINUOUS)

    return ColumnSchema(name=name, tags=tags, properties=properties, dtype=dtype)


def filter_dict_by_schema(input_dict, schema):
    """Filters out entries from input_dict, returns a dictionary
    where every entry corresponds to a column in the schema"""
    column_names = set(schema.column_names)
    return {k: v for k, v in input_dict.items() if k in column_names}


def categorical_cardinalities(schema) -> Dict[str, int]:
    outputs = {}
    for col in schema:
        if Tags.CATEGORICAL in col.tags:
            domain = col.int_domain
            if domain:
                outputs[col.name] = domain.max + 1

    return outputs


def categorical_domains(schema) -> Dict[str, str]:
    outputs = {}
    for col in schema:
        if Tags.CATEGORICAL in col.tags:
            domain = col.int_domain
            name = col.name
            if domain and domain.name:
                name = domain.name
            outputs[col.name] = name

    return outputs


def get_embedding_sizes_from_schema(
    schema: Schema, multiplier: float = 2.0, ensure_multiple_of_8: bool = False
) -> Dict[str, int]:
    """Provides a heristic (from Google) that suggests the embedding sizes
    as a function (forth root) of categorical features cardinalities, obtained
    from the schema.

    Parameters
    ----------
    schema : Schema
        Featires schema
    multiplier : float, optional
        Multiplier to be applied on the forth root of the cardinality.
        Google recommends multiplier in the [2.0,10.0] range, by default 2.0
    ensure_multiple_of_8 : bool, optional
        If enabled, adjusts the embedding dim to the smallest greater
        number multiple of 8, to ensure best performance with GPU ops, by default False

    Returns
    -------
    Dict[str, int]
        A dict with the feature names and the suggested embedding sizes based
        on the features cardinalities obtained from the schema
    """
    cardinalities = categorical_cardinalities(schema)

    return {
        key: get_embedding_size_from_cardinality(val, multiplier, ensure_multiple_of_8)
        for key, val in cardinalities.items()
    }


def get_embedding_size_from_cardinality(
    cardinality: int, multiplier: float = 2.0, ensure_multiple_of_8: bool = False
) -> int:
    """Provides a heristic (from Google) that suggests the embedding
    dimension as a function (forth root) of the feature cardinality.

    Parameters
    ----------
    cardinality : int
        The number of unique values of a categorical feature
    multiplier : float, optional
        Multiplier to be applied on the forth root of the cardinality.
        Google recommends multiplier in the [2.0,10.0] range, by default 2.0
    ensure_multiple_of_8 : bool, optional
        If enabled, adjusts the embedding dim to the smallest greater
        number multiple of 8, to ensure best performance with GPU ops, by default False

    Returns
    -------
    int
        The suggested embedding dimension based on the feature cardinality
    """
    # A rule-of-thumb from Google.
    embedding_size = int(math.ceil(math.pow(cardinality, 0.25) * multiplier))

    if ensure_multiple_of_8:
        embedding_size = int(math.ceil((embedding_size / 8)) * 8)

    return embedding_size
