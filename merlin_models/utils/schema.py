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
from typing import Dict, List, Optional, Tuple, Union

from merlin.schema import ColumnSchema, Schema, Tags
from merlin.schema.io.tensorflow_metadata import TensorflowMetadata


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


def create_categorical_column(name, num_items, tags=None, properties=None):
    properties = properties or {}
    if num_items:
        properties["domain"] = {"min": 0, "max": num_items}
    return ColumnSchema(name=name, tags=tags, properties=properties)


def create_continuous_column():
    # TODO
    return None


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


def get_embedding_sizes_from_schema(schema: Schema, multiplier: float = 2.0):
    cardinalities = categorical_cardinalities(schema)

    return {
        key: get_embedding_size_from_cardinality(val, multiplier)
        for key, val in cardinalities.items()
    }


def get_embedding_size_from_cardinality(cardinality: int, multiplier: float = 2.0):
    # A rule-of-thumb from Google.
    embedding_size = int(math.ceil(math.pow(cardinality, 0.25) * multiplier))

    return embedding_size
