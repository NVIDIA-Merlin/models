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

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

import tensorflow as tf
from tensorflow.python import to_dlpack
from tensorflow.python.keras import backend
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops.init_ops_v2 import Initializer

import merlin.io
from merlin.models.tf import BlockContext
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
from merlin.models.tf.utils import tf_utils
from merlin.models.utils import schema_utils
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.schema import ColumnSchema, Schema, Tags

EMBEDDING_FEATURES_PARAMS_DOCSTRING = """
    feature_config: Dict[str, FeatureConfig]
        This specifies what TableConfig to use for each feature. For shared embeddings, the same
        TableConfig can be used for multiple features.
    item_id: str, optional
        The name of the feature that's used for the item_id.
"""


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class EmbeddingTable(Block):
    """
    This block is used to create a single embedding table for a single feature.

    Parameters
    ----------
    name: str
        The name of embedding table.
    vocabulary_size: int
        The size of the vocabulary.
    options: EmbeddingTableOptions
        The options for the embedding table.
    context: BlockContext
        The context for the embedding table.

    """

    def __init__(
        self,
        name: str,
        vocabulary_size: int,
        options: "EmbeddingTableOptions",
        context: Optional[BlockContext] = None,
        **kwargs,
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
        add_fn = self.context.add_embedding_weight if hasattr(self, "_context") else self.add_weight
        self.table = add_fn(
            name=self._name,
            trainable=True,
            initializer=self.options.initialize(),
            shape=(self.vocabulary_size, self.options.dim),
        )

    def call(self, inputs):
        dtype = backend.dtype(inputs)
        if dtype != "int32" and dtype != "int64":
            inputs = tf.cast(inputs, "int32")

        if isinstance(inputs, tf.SparseTensor):
            out = tf.nn.safe_embedding_lookup_sparse(
                self.table, inputs, None, combiner=self.combiner
            )
        else:
            if self.options.max_seq_length:
                out = tf.gather(self.table, tf.cast(inputs, tf.int32))
            else:
                if len(inputs.shape) > 1:
                    # TODO: Check if it is correct to retrieve only the 1st element
                    # of second dim for non-sequential multi-hot categ features
                    out = tf.gather(self.table, tf.cast(inputs, tf.int32)[:, 0])
                else:
                    out = tf.gather(self.table, tf.cast(inputs, tf.int32))
        if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
            # Instead of casting the variable as in most layers, cast the output, as
            # this is mathematically equivalent but is faster.
            out = tf.cast(out, self._dtype_policy.compute_dtype)

        return out

    def compute_output_shape(self, input_shape):
        if self.options.max_seq_length:
            return tf.TensorShape(input_shape[0], self.options.max_seq_length, self.options.dim)

        return tf.TensorShape([input_shape[0], self.options.dim])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "name": self._name,
                "vocabulary_size": self.vocabulary_size,
                "options": self.options.get_config(),
            }
        )

        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["options"] = EmbeddingTableOptions.from_config(
            config["options"], custom_objects=custom_objects
        )

        return cls(**config)


def truncated_normal_initializer(dim: int):
    return init_ops_v2.TruncatedNormal(mean=0.0, stddev=1 / math.sqrt(dim))


InitializerFn = Callable[[int], Union[Initializer, tf.Tensor]]


@dataclass
class EmbeddingTableOptions:
    """
    Options for the EmbeddingTable.

    Parameters
    ----------
    dim: int
        The dimension of the embedding table.
    initializer: InitializerFn, optional
        The initializer to use for the embedding table.
    block_cls: Block, optional
        The block class to use for the embedding table.
    combiner: str, optional
        The combiner to use for the embedding table.
    max_seq_length: int, optional
        The maximum length of the sequence.
    extra_options: Dict[str, Any], optional
        Extra options for the embedding table.
    """

    dim: int
    initializer: InitializerFn = truncated_normal_initializer
    block_cls: Type[Block] = EmbeddingTable
    combiner: str = "mean"
    max_seq_length: Optional[int] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def infer_dim(
        cls, col: Union[ColumnSchema, int], multiplier: float = 2.0, **kwargs
    ) -> "EmbeddingTableOptions":
        if isinstance(col, ColumnSchema):
            dim = schema_utils.get_embedding_size_from_col(col, multiplier=multiplier)
        elif isinstance(col, int):
            dim = schema_utils.get_embedding_size_from_cardinality(col, multiplier=multiplier)
        else:
            raise TypeError(f"col must be a ColumnSchema or int, got {type(col)}")

        return cls(dim=dim, **kwargs)

    def to_block(self, column_schema: ColumnSchema, **kwargs) -> Block:
        cardinality = schema_utils.categorical_cardinality(column_schema)
        if not cardinality:
            raise ValueError("Cannot infer cardinality for column {}".format(column_schema.name))
        return self.block_cls(column_schema.name, cardinality, options=self, **kwargs)

    def initialize(self) -> Union[Initializer, tf.Tensor]:
        if isinstance(self.initializer, Initializer):
            return self.initializer

        return self.initializer(self.dim)

    def get_config(self):
        out = self.__dict__
        out["initializer"] = tf.keras.initializers.serialize(self.initialize())

        return out

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["initializer"] = tf.keras.initializers.deserialize(config["initializer"])

        return cls(**config)


# TODO: What do to with these parameters for sequential?
# mask_zero: bool = True,
# padding_idx: int = 0,


class EmbeddingOptions:
    """
    Options for the Embedding block.

    Parameters
    ----------
    schema: Schema
        The schema that contains the features of the embedding table.
    custom_tables: Dict[str, Union[EmbeddingTableOptions, EmbeddingTable]], optional
        Custom embedding tables.
    default_embedding_dim: int
        The default embedding dimension.
    infer_embedding_sizes: bool
        Whether to infer the embedding dimension from the schema.
    infer_embedding_sizes_multiplier: float
        The multiplier to use when inferring the embedding dimension.
        Defaults to 2.0.
    default_block_cls: Type[Block], optional
        The default block class to use for the embedding table.
    default_combiner: str, optional
        The default combiner to use for the embedding table.
    default_initializer: InitializerFn
        The default initializer to use for the embedding table.
        Defaults to truncated_normal_initializer.
    max_seq_length: int, optional
        The maximum length of the sequence.
    """

    def __init__(
        self,
        schema: Schema,
        custom_tables: Dict[str, Union[EmbeddingTableOptions, EmbeddingTable]] = None,
        default_embedding_dim: int = 64,
        infer_embedding_sizes: bool = False,
        infer_embedding_sizes_multiplier: float = 2.0,
        default_block_cls: Type[Block] = EmbeddingTable,
        default_combiner: str = "mean",
        default_initializer: InitializerFn = truncated_normal_initializer,
        max_seq_length: Optional[int] = None,
    ):
        self.schema = schema
        self._tables: Dict[str, EmbeddingTableOptions] = {}
        self._features: Dict[str, EmbeddingTableOptions] = {}
        self._feature_mapping: Dict[str, str] = {}
        self._max_seq_length = max_seq_length
        _custom_tables = custom_tables or {}

        domains = schema_utils.categorical_domains(schema)
        for name, col in schema.column_schemas.items():
            if name not in domains:
                continue
            table_name = domains[name]
            if table_name not in self._tables:
                table_kwargs = dict(
                    block_cls=default_block_cls,
                    combiner=default_combiner,
                    initializer=default_initializer,
                )
                if table_name in _custom_tables:
                    table = _custom_tables[table_name]
                elif infer_embedding_sizes:
                    table = EmbeddingTableOptions.infer_dim(
                        col, multiplier=infer_embedding_sizes_multiplier, **table_kwargs
                    )
                else:
                    table = EmbeddingTableOptions(dim=default_embedding_dim, **table_kwargs)

                self._tables[table_name] = table

            self._features[name] = self._tables[table_name]
            self._feature_mapping[name] = table_name

    def set_table(self, name: str, table: EmbeddingTableOptions):
        self._tables[name] = table

    def set_feature(self, name: str, table: EmbeddingTableOptions):
        self._features[name] = table

    def to_blocks(self) -> Dict[str, Block]:
        blocks: Dict[str, Block] = {}
        feature_to_blocks: Dict[str, Block] = {}
        for name, col in self.schema.column_schemas.items():
            if name in self._feature_mapping:
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
    def features(self) -> Dict[str, EmbeddingTableOptions]:
        return self._features

    @property
    def tables(self) -> Dict[str, EmbeddingTableOptions]:
        return self._tables

    @property
    def feature_mapping(self) -> Dict[str, str]:
        return self._feature_mapping

    @property
    def max_seq_length(self) -> Optional[int]:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: Optional[int]):
        self._max_seq_length = value
        for table in self._tables.values():
            table.max_seq_length = value


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
            if not schema:
                schema = embeddings.schema
        elif isinstance(embeddings, dict) and all(
            isinstance(v, EmbeddingTable) for v in embeddings.values()
        ):
            self.embeddings = embeddings
        else:
            raise ValueError(
                "`embeddings` must be an EmbeddingOptions or a dictionary ",
                "from feature_name to EmbeddingTable ",
                f"but got {embeddings} of type {type(embeddings)}",
            )

        if add_default_pre:
            convert_lists = (
                AsDenseFeatures(self.max_seq_length) if self.max_seq_length else AsSparseFeatures()
            )
            embedding_pre = [Filter(list(self.embeddings.keys())), convert_lists]
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

    @classmethod
    def from_schema(cls, schema: Schema, tags=None, allow_none=True, **kwargs):
        _schema = schema.select_by_tag(tags) if tags else schema
        if not _schema.column_schemas:
            if allow_none:
                return None
            else:
                raise ValueError(f"No columns found in schema {schema}")

        options = EmbeddingOptions(schema, **kwargs)

        return cls(options, schema=_schema)

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
        output_shapes = {}
        for name, val in input_shapes.items():
            output_shapes[name] = self.embeddings[name].compute_output_shape(val)

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
        config = tf_utils.maybe_serialize_keras_objects(self, super().get_config(), ["embeddings"])

        return config

    @classmethod
    def from_config(cls, config):
        config = tf_utils.maybe_deserialize_keras_objects(config, ["embeddings"])

        # Set `add_default_pre to False` since pre will be provided from the config
        config["add_default_pre"] = False

        return super().from_config(config)

    @property
    def max_seq_length(self) -> Optional[int]:
        return list(self.embeddings.values())[0].options.max_seq_length

    @property
    def is_sequential(self) -> bool:
        return self.max_seq_length is not None

    def __getitem__(self, item):
        return self.embeddings[item]


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
