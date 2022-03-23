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

from copy import copy, deepcopy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union, Type, List

import tensorflow as tf
from tensorflow.python.ops import init_ops_v2

from merlin.models.tf import BlockContext
from tensorflow.python import to_dlpack
from tensorflow.python.keras import backend
from tensorflow.python.ops.init_ops_v2 import Initializer
from tensorflow.python.tpu.tpu_embedding_v2_utils import FeatureConfig, TableConfig

import merlin.io
from merlin.models.tf.blocks.core.base import Block, BlockType
from merlin.models.tf.blocks.core.combinators import ParallelBlock, SequentialBlock
from merlin.models.tf.blocks.core.tabular import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    Filter,
    TabularAggregationType,
    TabularBlock,
)
from merlin.models.tf.blocks.core.transformations import AsDenseFeatures, AsSparseFeatures

# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter, unexpected-keyword-arg
from merlin.models.tf.typing import TabularData
from merlin.models.utils import schema_utils
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.schema import ColumnSchema, Schema, Tags, TagsType

EMBEDDING_FEATURES_PARAMS_DOCSTRING = """
    feature_config: Dict[str, FeatureConfig]
        This specifies what TableConfig to use for each feature. For shared embeddings, the same
        TableConfig can be used for multiple features.
    item_id: str, optional
        The name of the feature that's used for the item_id.
"""


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class EmbeddingTable(Block):
    def __init__(
            self,
            name: str,
            vocabulary_size: int,
            options: "TableOptions",
            context: Optional[BlockContext] = None,
            **kwargs
    ):
        super().__init__(context=context, name=name, **kwargs)
        self._name = name
        self.options = options
        self.vocabulary_size = vocabulary_size

    @classmethod
    def load(cls, path: str) -> "EmbeddingTable":
        raise NotImplementedError("TODO!")

    def build(self, input_shapes):
        if not getattr(self, "built", False):
            self._create_embedding_table(input_shapes)

        return super().build(input_shapes)

    def _create_embedding_table(self, input_shape):
        add_fn = (
            self.context.add_embedding_weight if hasattr(self, "_context") else self.add_weight
        )
        self.embedding_table = add_fn(
            name=self._name,
            trainable=True,
            initializer=self.options.initialize(),
            shape=(self.vocabulary_size, self.dim),
        )

    def call(self, inputs, output_sequence=False):
        dtype = backend.dtype(inputs)
        if dtype != "int32" and dtype != "int64":
            inputs = tf.cast(inputs, "int32")

        if isinstance(inputs, tf.SparseTensor):
            out = tf.nn.safe_embedding_lookup_sparse(self.embedding_table, inputs, None, combiner=self.combiner)
        else:
            if output_sequence:
                out = tf.gather(self.embedding_table, tf.cast(inputs, tf.int32))
            else:
                if len(self.shape) > 1:
                    # TODO: Check if it is correct to retrieve only the 1st element
                    # of second dim for non-sequential multi-hot categ features
                    out = tf.gather(self.embedding_table, tf.cast(inputs, tf.int32)[:, 0])
                else:
                    out = tf.gather(self.embedding_table, tf.cast(inputs, tf.int32))
        if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
            # Instead of casting the variable as in most layers, cast the output, as
            # this is mathematically equivalent but is faster.
            out = tf.cast(out, self._dtype_policy.compute_dtype)

        return out

    def compute_output_shape(self, input_shapes):
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)

        return tf.TensorShape([batch_size, self.dim])


def truncated_normal_initializer(options: "TableOptions"):
    return init_ops_v2.TruncatedNormal(mean=0.0, stddev=1 / math.sqrt(options.dim))


InitializerFn = Callable[[int], Union[Initializer, tf.Tensor]]


@dataclass
class TableOptions:
    dim: int
    initializer: InitializerFn = truncated_normal_initializer
    block_cls: Type[Block] = EmbeddingTable
    combiner: str = "mean"
    extra_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def infer_dim(cls, col: Union[ColumnSchema, int], multiplier: float = 2.0, **kwargs) -> "TableOptions":
        if isinstance(col, ColumnSchema):
            dim = schema_utils.get_embedding_size_from_col(col, multiplier=multiplier)
        elif isinstance(col, int):
            dim = schema_utils.get_embedding_size_from_cardinality(col, multiplier=multiplier)
        else:
            raise TypeError(f"col must be a ColumnSchema or int, got {type(col)}")

        return cls(dim=dim, **kwargs)

    def to_block(self, column_schema: ColumnSchema, **kwargs) -> "EmbeddingTable":
        cardinality = schema_utils.categorical_cardinality(column_schema)
        if not cardinality:
            raise ValueError("Cannot infer cardinality for column {}".format(column_schema.name))
        return self.block_cls(column_schema.name, cardinality, options=self, **kwargs)

    def initialize(self) -> Union[Initializer, tf.Tensor]:
        return self.initializer(self.dim)


class EmbeddingOptions:
    def __init__(
            self,
            schema: Schema,
            custom_tables: Dict[str, Union[TableOptions, EmbeddingTable]] = None,
            default_embedding_dim: int = 64,
            infer_embedding_sizes: bool = False,
            infer_embedding_sizes_multiplier: float = 2.0,
            default_block_cls: Type[Block] = EmbeddingTable,
            default_combiner: str = "mean",
            default_initializer: InitializerFn = truncated_normal_initializer,
    ):
        self.schema = schema
        self._tables: Dict[str, TableOptions] = {}
        self._features: Dict[str, TableOptions] = {}
        self._feature_mapping: Dict[str, str] = {}

        domains = schema_utils.categorical_domains(schema)
        for name, col in schema.column_schemas:
            table_name = domains[name]
            if table_name not in self._tables:
                table_kwargs = dict(
                    block_cls=default_block_cls,
                    combiner=default_combiner,
                    initializer=default_initializer
                )
                if table_name in custom_tables:
                    table = custom_tables[table_name]
                elif infer_embedding_sizes:
                    table = TableOptions.infer_dim(col, multiplier=infer_embedding_sizes_multiplier, **table_kwargs)
                else:
                    table = TableOptions(dim=default_embedding_dim, **table_kwargs)

                self._tables[table_name] = table

            self._features[name] = self._tables[table_name]
            self._feature_mapping[name] = table_name

    def set_table(self, name: str, table: TableOptions):
        self._tables[name] = table

    def set_feature(self, name: str, table: TableOptions):
        self._features[name] = table

    def to_blocks(self) -> Dict[str, Block]:
        blocks: Dict[str, Block] = {}
        feature_to_blocks: Dict[str, Block] = {}
        for name, col in self.schema.column_schemas.items():
            table_name = self._feature_mapping[name]
            if table_name not in blocks:
                blocks[table_name] = self._tables[table_name].to_block(col)
            feature_to_blocks[name] = blocks[table_name]

        return feature_to_blocks

    def select_by_tag(self, tags: List[Union[str, Tags]]) -> "EmbeddingOptions":
        return EmbeddingOptions(self.schema.select_by_tag(tags), custom_tables=self._tables)

    def select_by_schema(self, schema: Schema) -> "EmbeddingOptions":
        return EmbeddingOptions(schema, custom_tables=self._tables)

    @property
    def features(self) -> Dict[str, TableOptions]:
        return self._features

    @property
    def tables(self) -> Dict[str, TableOptions]:
        return self._tables

    @property
    def feature_mapping(self) -> Dict[str, str]:
        return self._feature_mapping


@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    embedding_features_parameters=EMBEDDING_FEATURES_PARAMS_DOCSTRING,
)
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class EmbeddingFeatures(TabularBlock):
    """Input block for embedding-lookups for categorical features.

    For multi-hot features, the embeddings will be aggregated into a single tensor using the mean.

    Parameters
    ----------
    {embedding_features_parameters}
    {tabular_module_parameters}
    """

    def __init__(
            self,
            embeddings: Union[EmbeddingOptions, Dict[str, EmbeddingTable]],
            pre: Optional[BlockType] = None,
            post: Optional[BlockType] = None,
            aggregation: Optional[TabularAggregationType] = None,
            schema: Optional[Schema] = None,
            name=None,
            add_default_pre=True,
            **kwargs,
    ):
        if isinstance(embeddings, EmbeddingOptions):
            self.embeddings = embeddings.to_blocks()
        else:
            self.embeddings = embeddings

        if add_default_pre:
            embedding_pre = [Filter(list(self.embeddings.keys())), AsSparseFeatures()]
            pre = [embedding_pre, pre] if pre else embedding_pre  # type: ignore

        super().__init__(
            pre=pre,
            post=post,
            aggregation=aggregation,
            name=name,
            schema=schema,
            is_input=True,
            **kwargs,
        )

    def build(self, input_shapes):
        for name, table in self.embeddings.items():
            table.build(input_shapes)
        if isinstance(input_shapes, dict):
            super().build(input_shapes)
        else:
            tf.keras.layers.Layer.build(self, input_shapes)

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        embedded_outputs = {}
        for name, val in inputs.items():
            embedded_outputs[name] = self.embeddings[name](val)

        return embedded_outputs

    def compute_call_output_shape(self, input_shapes):
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)

        output_shapes = {}
        for name, val in input_shapes.items():
            output_shapes[name] = tf.TensorShape([batch_size, self.feature_config[name].table.dim])

        return output_shapes

    def embedding_table_df(self, table_name: Union[str, Tags], gpu=True):
        if isinstance(table_name, Tags):
            table_name = table_name.value
        embeddings = self.embedding_tables[table_name]

        if gpu:
            import cudf

            df = cudf.from_dlpack(to_dlpack(tf.convert_to_tensor(embeddings)))
            df.columns = [str(col) for col in list(df.columns)]
            df.set_index(cudf.RangeIndex(0, embeddings.shape[0]))
        else:
            import pandas as pd

            df = pd.DataFrame(embeddings.numpy())
            df.columns = [str(col) for col in list(df.columns)]
            df.set_index(pd.RangeIndex(0, embeddings.shape[0]))

        return df

    def embedding_table_dataset(self, table_name: Union[str, Tags], gpu=True) -> merlin.io.Dataset:
        return merlin.io.Dataset(self.embedding_table_df(table_name, gpu))

    def export_embedding_table(self, table_name: Union[str, Tags], export_path: str, gpu=True):
        df = self.embedding_table_df(table_name, gpu=gpu)
        df.to_parquet(export_path)

    def get_config(self):
        config = super().get_config()

        feature_configs = {}

        for key, val in self.feature_config.items():
            feature_config_dict = dict(name=val.name, max_sequence_length=val.max_sequence_length)

            feature_config_dict["table"] = serialize_table_config(val.table)
            feature_configs[key] = feature_config_dict

        config["feature_config"] = feature_configs

        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize feature_config
        feature_configs, table_configs = {}, {}
        for key, val in config["feature_config"].items():
            feature_params = deepcopy(val)
            table_params = feature_params["table"]
            if "name" in table_configs:
                feature_params["table"] = table_configs["name"]
            else:
                table = deserialize_table_config(table_params)
                if table.name:
                    table_configs[table.name] = table
                feature_params["table"] = table
            feature_configs[key] = FeatureConfig(**feature_params)
        config["feature_config"] = feature_configs

        # Set `add_default_pre to False` since pre will be provided from the config
        config["add_default_pre"] = False

        return super().from_config(config)


@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    embedding_features_parameters=EMBEDDING_FEATURES_PARAMS_DOCSTRING,
)
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequenceEmbeddingFeatures(EmbeddingFeatures):
    """Input block for embedding-lookups for categorical features. This module produces 3-D tensors,
    this is useful for sequential models like transformers.
    Parameters
    ----------
    {embedding_features_parameters}
    padding_idx: int
        The symbol to use for padding.
    {tabular_module_parameters}
    """

    def __init__(
            self,
            feature_config: Dict[str, FeatureConfig],
            max_seq_length: Optional[int] = None,
            mask_zero: bool = True,
            padding_idx: int = 0,
            pre: Optional[BlockType] = None,
            post: Optional[BlockType] = None,
            aggregation: Optional[TabularAggregationType] = None,
            schema: Optional[Schema] = None,
            name: Optional[str] = None,
            add_default_pre=True,
            **kwargs,
    ):
        if add_default_pre:
            embedding_pre = [Filter(list(feature_config.keys())), AsDenseFeatures(max_seq_length)]
            pre = [embedding_pre, pre] if pre else embedding_pre  # type: ignore

        super().__init__(
            feature_config=feature_config,
            pre=pre,
            post=post,
            aggregation=aggregation,
            name=name,
            schema=schema,
            add_default_pre=False,
            **kwargs,
        )
        self.padding_idx = padding_idx
        self.mask_zero = mask_zero

    def lookup_feature(self, name, val, **kwargs):
        return super(SequenceEmbeddingFeatures, self).lookup_feature(
            name, val, output_sequence=True
        )

    def compute_call_output_shape(self, input_shapes):
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)
        sequence_length = input_shapes[list(self.feature_config.keys())[0]][1]

        output_shapes = {}
        for name, val in input_shapes.items():
            output_shapes[name] = tf.TensorShape(
                [batch_size, sequence_length, self.feature_config[name].table.dim]
            )

        return output_shapes

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        outputs = {}
        for key, val in inputs.items():
            outputs[key] = tf.not_equal(val, self.padding_idx)

        return outputs

    def get_config(self):
        config = super().get_config()
        config["mask_zero"] = self.mask_zero
        config["padding_idx"] = self.padding_idx

        return config


def ContinuousEmbedding(
        inputs: Union[Block, Dict[str, Block]],
        embedding_block: Block,
        aggregation: Optional["TabularAggregationType"] = None,
        continuous_aggregation="concat",
        name: str = "continuous_projection",
        **kwargs,
) -> SequentialBlock:
    """Routes continuous features to an embedding block.

    Parameters
    ----------
    inputs: Union[Block, Dict[str, Block]]
        The input block or dictionary of input blocks.
    embedding_block: Block
        The embedding block to use.
    aggregation: Optional[TabularAggregationType]
        The aggregation to use.
    continuous_aggregation: str
        The aggregation to use for continuous features.
    name: str
        The name of the block.

    Returns
    -------
    SequentialBlock
    """
    _inputs: Block = ParallelBlock(inputs) if isinstance(inputs, dict) else inputs
    continuous_embedding = Filter(Tags.CONTINUOUS, aggregation=continuous_aggregation).connect(
        embedding_block
    )

    outputs = _inputs.connect_branch(
        continuous_embedding.as_tabular(name), add_rest=True, aggregation=aggregation, **kwargs
    )

    return outputs


def serialize_table_config(table_config: TableConfig) -> Dict[str, Any]:
    """Serialize a table config to a dictionary.

    Parameters
    ----------
    table_config: TableConfig
        The table config to serialize.

    Returns
    -------
    dict
        The serialized table config.
    """

    table = deepcopy(table_config.__dict__)
    if "initializer" in table:
        table["initializer"] = tf.keras.initializers.serialize(table["initializer"])
    if "optimizer" in table:
        table["optimizer"] = tf.keras.optimizers.serialize(table["optimizer"])

    return table


def deserialize_table_config(table_params: Dict[str, Any]) -> TableConfig:
    """Deserialize a table config from a dictionary.

    Parameters
    ----------
    table_params: dict
        The serialized table config

    Returns
    -------
    TableConfig

    """

    if "initializer" in table_params and table_params["initializer"]:
        table_params["initializer"] = tf.keras.initializers.deserialize(table_params["initializer"])
    if "optimizer" in table_params and table_params["optimizer"]:
        table_params["optimizer"] = tf.keras.optimizers.deserialize(table_params["optimizer"])
    table = TableConfig(**table_params)

    return table

def serialize_feature_config(feature_config: FeatureConfig) -> Dict[str, Any]:
    """Serialize a feature config to a dictionary.

    Parameters
    ----------
    feature_config: FeatureConfig
        The feature config to serialize.

    Returns
    -------
    dict

    """
    outputs = {}

    for key, val in feature_config.items():
        feature_config_dict = dict(name=val.name, max_sequence_length=val.max_sequence_length)
        feature_config_dict["table"] = serialize_table_config(feature_config_dict["table"])
        outputs[key] = feature_config_dict

    return outputs
