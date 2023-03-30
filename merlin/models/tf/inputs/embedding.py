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
import collections
import inspect
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Type, Union

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.python import to_dlpack
from tensorflow.python.tpu.tpu_embedding_v2_utils import FeatureConfig, TableConfig

import merlin.io
from merlin.core.dispatch import DataFrameType
from merlin.io import Dataset
from merlin.models.tf.blocks.mlp import InitializerType, RegularizerType
from merlin.models.tf.core.base import Block, BlockType
from merlin.models.tf.core.combinators import ParallelBlock, SequentialBlock
from merlin.models.tf.core.tabular import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    Filter,
    TabularAggregationType,
    TabularBlock,
)

# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter, unexpected-keyword-arg
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.tf_utils import call_layer, df_to_tensor, tensor_to_df
from merlin.models.utils import schema_utils
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.models.utils.schema_utils import (
    create_categorical_column,
    infer_embedding_dim,
    schema_to_tensorflow_metadata_json,
    tensorflow_metadata_json_to_schema,
)
from merlin.schema import ColumnSchema, Schema, Tags, TagsType

EMBEDDING_FEATURES_PARAMS_DOCSTRING = """
    feature_config: Dict[str, FeatureConfig]
        This specifies what TableConfig to use for each feature. For shared embeddings, the same
        TableConfig can be used for multiple features.
    item_id: str, optional
        The name of the feature that's used for the item_id.
"""


class EmbeddingTableBase(Block):
    def __init__(self, dim: int, *col_schemas: ColumnSchema, trainable=True, **kwargs):
        super(EmbeddingTableBase, self).__init__(trainable=trainable, **kwargs)
        self.dim = dim
        self.features = {}
        if len(col_schemas) == 0:
            raise ValueError("At least one col_schema must be provided to the embedding table.")

        self.col_schema = col_schemas[0]
        for col_schema in col_schemas:
            self.add_feature(col_schema)

    @property
    def _schema(self):
        return Schema([col_schema for col_schema in self.features.values()])

    @classmethod
    def from_pretrained(
        cls,
        data: Union[Dataset, DataFrameType],
        col_schema: Optional[ColumnSchema] = None,
        trainable=True,
        **kwargs,
    ):
        raise NotImplementedError()

    @property
    def input_dim(self):
        return self.col_schema.int_domain.max + 1

    @property
    def table_name(self):
        return self.col_schema.int_domain.name or self.col_schema.name

    def add_feature(self, col_schema: ColumnSchema) -> None:
        """Add a feature to the table.

        Adding more than one feature enables the table to lookup and return embeddings
        for more than one feature when called with tabular data (Dict[str, TensorLike]).

        Additional column schemas must have an int domain that matches the existing ones.

        Parameters
        ----------
        col_schema : ColumnSchema
        """
        if not col_schema.int_domain:
            raise ValueError("`col_schema` needs to have an int-domain")

        if (
            col_schema.int_domain.name
            and self.col_schema.int_domain.name
            and col_schema.int_domain.name != self.col_schema.int_domain.name
        ):
            raise ValueError(
                "`col_schema` int-domain name does not match table domain name. "
                f"{col_schema.int_domain.name} != {self.col_schema.int_domain.name} "
            )

        if col_schema.int_domain.max != self.col_schema.int_domain.max:
            raise ValueError(
                "`col_schema.int_domain.max` does not match existing input dim."
                f"{col_schema.int_domain.max} != {self.col_schema.int_domain.max} "
            )

        self.features[col_schema.name] = col_schema

    def get_config(self):
        config = super().get_config()
        config["dim"] = self.dim

        schema = schema_to_tensorflow_metadata_json(self.schema)
        config["schema"] = schema

        return config

    @classmethod
    def from_config(cls, config):
        dim = config.pop("dim")
        schema = tensorflow_metadata_json_to_schema(config.pop("schema"))

        return cls(dim, *schema, **config)


CombinerType = Union[str, tf.keras.layers.Layer]


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class EmbeddingTable(EmbeddingTableBase):
    """Embedding table that is backed by a standard Keras Embedding Layer.
    It accepts as input features for lookup tf.Tensor, tf.RaggedTensor,
    and tf.SparseTensor which might be 2D (batch_size, 1) for scalars
    or 3d (batch_size, seq_length, 1) for sequential features

     Parameters
     ----------
     dim: Dimension of the dense embedding.
     col_schema: ColumnSchema
         Schema of the column. This is used to infer the cardinality.
     embeddings_initializer: Initializer for the `embeddings`
       matrix (see `keras.initializers`).
     embeddings_regularizer: Regularizer function applied to
       the `embeddings` matrix (see `keras.regularizers`).
     embeddings_constraint: Constraint function applied to
       the `embeddings` matrix (see `keras.constraints`).
     mask_zero: Boolean, whether or not the input value 0 is a special "padding"
       value that should be masked out.
       This is useful when using recurrent layers
       which may take variable length input.
       If this is `True`, then all subsequent layers
       in the model need to support masking or an exception will be raised.
       If mask_zero is set to True, as a consequence, index 0 cannot be
       used in the vocabulary (input_dim should equal size of
       vocabulary + 1).
     input_length: Length of input sequences, when it is constant.
       This argument is required if you are going to connect
       `Flatten` then `Dense` layers upstream
       (without it, the shape of the dense outputs cannot be computed).
    combiner: A string specifying how to combine embedding results for each
       entry ("mean", "sqrtn" and "sum" are supported) or a layer.
       Default is None (no combiner used)
    trainable: Boolean, whether the layer's variables should be trainable.
    name: String name of the layer.
    dtype: The dtype of the layer's computations and weights. Can also be a
       `tf.keras.mixed_precision.Policy`, which allows the computation and weight
       dtype to differ. Default of `None` means to use
       `tf.keras.mixed_precision.global_policy()`, which is a float32 policy
       unless set to different value.
    dynamic: Set this to `True` if your layer should only be run eagerly, and
       should not be used to generate a static computation graph.
       This would be the case for a Tree-RNN or a recursive network,
       for example, or generally for any layer that manipulates tensors
       using Python control flow. If `False`, we assume that the layer can
       safely be used to generate a static computation graph.
    l2_batch_regularization_factor: float, optional
        Factor for L2 regularization of the embeddings vectors (from the current batch only)
        by default 0.0
    **kwargs: Forwarded Keras Layer parameters
    """

    def __init__(
        self,
        dim: int,
        *col_schemas: ColumnSchema,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        activity_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        input_length=None,
        sequence_combiner: Optional[CombinerType] = None,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        table=None,
        l2_batch_regularization_factor=0.0,
        weights=None,
        **kwargs,
    ):
        """Create an EmbeddingTable."""
        super(EmbeddingTable, self).__init__(
            dim,
            *col_schemas,
            trainable=trainable,
            name=name,
            dtype=dtype,
            dynamic=dynamic,
            **kwargs,
        )
        if table is not None:
            self.table = table
        else:
            table_kwargs = dict(
                embeddings_initializer=embeddings_initializer,
                embeddings_regularizer=embeddings_regularizer,
                activity_regularizer=activity_regularizer,
                embeddings_constraint=embeddings_constraint,
                mask_zero=mask_zero,
                input_length=input_length,
                trainable=trainable,
                weights=weights,
            )
            self.table = tf.keras.layers.Embedding(
                input_dim=self.input_dim,
                output_dim=self.dim,
                name=self.table_name,
                **table_kwargs,
            )
        self.sequence_combiner = sequence_combiner
        self.supports_masking = True
        self.l2_batch_regularization_factor = l2_batch_regularization_factor

    def select_by_tag(self, tags: Union[Tags, Sequence[Tags]]) -> Optional["EmbeddingTable"]:
        """Select features in EmbeddingTable by tags.

        Since an EmbeddingTable can be a shared-embedding table, this method filters
        the schema for features that match the tags.

        If none of the features match the tags, it will return None.

        Parameters
        ----------
        tags: Union[Tags, Sequence[Tags]]
            A list of tags.

        Returns
        -------
        An EmbeddingTable if the tags match. If no features match, it returns None.
        """
        if not isinstance(tags, collections.Sequence):
            tags = [tags]

        selected_schema = self.schema.select_by_tag(tags)
        if not selected_schema:
            return
        config = self.get_config()
        config["schema"] = schema_utils.schema_to_tensorflow_metadata_json(selected_schema)
        embedding_table = EmbeddingTable.from_config(config, table=self.table)
        return embedding_table

    @classmethod
    def from_pretrained(
        cls,
        data: Union[Dataset, DataFrameType],
        trainable=True,
        name=None,
        col_schema=None,
        **kwargs,
    ) -> "EmbeddingTable":
        """Create From pre-trained embeddings from a Dataset or DataFrame.
        Parameters
        ----------
        data : Union[Dataset, DataFrameType]
            A dataset containing the pre-trained embedding weights
        trainable : bool
            Whether the layer should be trained or not.
        name : str
            The name of the layer.
        """
        if hasattr(data, "to_ddf"):
            data = data.to_ddf().compute()
        embeddings = df_to_tensor(data, tf.float32)

        num_items, dim = tuple(embeddings.shape)

        if not col_schema:
            if not name:
                raise ValueError("`name` is required when not using a ColumnSchema")
            col_schema = create_categorical_column(name, num_items - 1)

        embedding_table = cls(
            dim,
            col_schema,
            name=name,
            weights=[embeddings],
            embeddings_initializer=None,
            trainable=trainable,
            **kwargs,
        )
        # trigger build of table to make sure that weights are configured
        # without this line the weights are not initialized correctly with
        # the provided values
        embedding_table.table(0)

        return embedding_table

    @classmethod
    def from_dataset(
        cls,
        data: Union[Dataset, DataFrameType],
        trainable=True,
        name=None,
        col_schema=None,
        **kwargs,
    ) -> "EmbeddingTable":
        """Create From pre-trained embeddings from a Dataset or DataFrame.
        Parameters
        ----------
        data : Union[Dataset, DataFrameType]
            A dataset containing the pre-trained embedding weights
        trainable : bool
            Whether the layer should be trained or not.
        name : str
            The name of the layer.
        """
        return cls.from_pretrained(
            data, trainable=trainable, name=name, col_schema=col_schema, **kwargs
        )

    def to_dataset(self, gpu=None) -> merlin.io.Dataset:
        return merlin.io.Dataset(self.to_df(gpu=gpu))

    def to_df(self, gpu=None):
        return tensor_to_df(self.table.embeddings, gpu=gpu)

    def _maybe_build(self, inputs):
        """Creates state between layer instantiation and layer call.
        Invoked automatically before the first execution of `call()`.
        """
        self.table._maybe_build(inputs)

        return super(EmbeddingTable, self)._maybe_build(inputs)

    def build(self, input_shapes):
        if not self.table.built:
            self.table.build(input_shapes)
        return super(EmbeddingTable, self).build(input_shapes)

    def call(
        self, inputs: Union[tf.Tensor, TabularData], **kwargs
    ) -> Union[tf.Tensor, TabularData]:
        """
        Parameters
        ----------
        inputs : Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor]
            Tensors or dictionary of tensors representing the input batch.

        Returns
        -------
        A tensor or dict of tensors corresponding to the embeddings for inputs
        """
        if isinstance(inputs, dict):
            out = {}
            for feature_name in self.schema.column_names:
                if feature_name in inputs:
                    out[feature_name] = self._call_table(inputs[feature_name], **kwargs)
        else:
            out = self._call_table(inputs, **kwargs)

        return out

    def _call_table(self, inputs, **kwargs):
        if isinstance(inputs, (tf.RaggedTensor, tf.SparseTensor)):
            if self.sequence_combiner and isinstance(self.sequence_combiner, str):
                if isinstance(inputs, tf.RaggedTensor):
                    inputs = inputs.to_sparse()

                inputs = tf.sparse.reshape(inputs, tf.shape(inputs)[:-1])

                out = tf.nn.safe_embedding_lookup_sparse(
                    self.table.embeddings, inputs, None, combiner=self.sequence_combiner
                )

            else:
                if isinstance(inputs, tf.SparseTensor):
                    raise ValueError(
                        "Sparse tensors are not supported without sequence_combiner ",
                        "please convert the tensor to a ragged or dense.",
                    )

                inputs = tf.squeeze(inputs, axis=-1)

                out = call_layer(self.table, inputs, **kwargs)
                if len(out.get_shape()) > 2 and isinstance(
                    self.sequence_combiner, tf.keras.layers.Layer
                ):
                    out = call_layer(self.sequence_combiner, out, **kwargs)
        else:
            if inputs.shape.as_list()[-1] == 1:
                inputs = tf.squeeze(inputs, axis=-1)
            out = call_layer(self.table, inputs, **kwargs)
            if len(out.get_shape()) > 2 and self.sequence_combiner is not None:
                if isinstance(self.sequence_combiner, tf.keras.layers.Layer):
                    out = call_layer(self.sequence_combiner, out, **kwargs)
                elif isinstance(self.sequence_combiner, str):
                    out = process_str_sequence_combiner(out, self.sequence_combiner, **kwargs)

        if self.l2_batch_regularization_factor > 0:
            self.add_loss(self.l2_batch_regularization_factor * tf.reduce_sum(tf.square(out)))

        if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
            # Instead of casting the variable as in most layers, cast the output, as
            # this is mathematically equivalent but is faster.
            out = tf.cast(out, self._dtype_policy.compute_dtype)

        return out

    def compute_output_shape(
        self, input_shape: Union[tf.TensorShape, Dict[str, tf.TensorShape]]
    ) -> Union[tf.TensorShape, Dict[str, tf.TensorShape]]:
        if isinstance(input_shape, dict):
            output_shapes = {}
            for feature_name in self.schema.column_names:
                if feature_name in input_shape:
                    output_shapes[feature_name] = self._compute_output_shape_table(
                        input_shape[feature_name]
                    )
        else:
            output_shapes = self._compute_output_shape_table(input_shape)

        return output_shapes

    def _compute_output_shape_table(
        self, input_shape: Union[tf.TensorShape, tuple]
    ) -> tf.TensorShape:
        first_dims = input_shape

        if input_shape.rank > 1:
            if self.sequence_combiner is not None:
                first_dims = [input_shape[0]]

            elif input_shape[-1] == 1:
                first_dims = input_shape[:-1]

        output_shapes = tf.TensorShape(list(first_dims) + [self.dim])

        return output_shapes

    def compute_call_output_shape(self, input_shapes):
        return self.compute_output_shape(input_shapes)

    @classmethod
    def from_config(cls, config, table=None):
        if table:
            config["table"] = table
        else:
            config["table"] = tf.keras.layers.deserialize(config["table"])
        if "combiner-layer" in config:
            config["sequence_combiner"] = tf.keras.layers.deserialize(config.pop("combiner-layer"))

        return super().from_config(config)

    def get_config(self):
        config = super().get_config()
        config["table"] = tf.keras.layers.serialize(self.table)
        if isinstance(self.sequence_combiner, tf.keras.layers.Layer):
            config["combiner-layer"] = tf.keras.layers.serialize(self.sequence_combiner)
        else:
            config["sequence_combiner"] = self.sequence_combiner
        return config


def Embeddings(
    schema: Schema,
    dim: Optional[Union[Dict[str, int], int]] = None,
    infer_dim_fn: Callable[[ColumnSchema], int] = infer_embedding_dim,
    sequence_combiner: Optional[Union[CombinerType, Dict[str, CombinerType]]] = "mean",
    embeddings_initializer: Optional[Union[InitializerType, Dict[str, InitializerType]]] = None,
    embeddings_regularizer: Optional[Union[RegularizerType, Dict[str, RegularizerType]]] = None,
    activity_regularizer: Optional[Union[RegularizerType, Dict[str, RegularizerType]]] = None,
    trainable: Optional[Union[bool, Dict[str, bool]]] = None,
    table_cls: Type[tf.keras.layers.Layer] = EmbeddingTable,
    pre: Optional[BlockType] = None,
    post: Optional[BlockType] = None,
    aggregation: Optional[TabularAggregationType] = None,
    block_name: str = "embeddings",
    l2_batch_regularization_factor: Optional[Union[float, Dict[str, float]]] = 0.0,
    **kwargs,
) -> ParallelBlock:
    """Creates a ParallelBlock with an EmbeddingTable for each categorical feature
    in the schema.

    Parameters
    ----------
    schema: Schema
        Schema of the input data. This Schema object will be automatically generated using
        [NVTabular](https://nvidia-merlin.github.io/NVTabular/main/Introduction.html).
        Next to this, it's also possible to construct it manually.
    dim: Optional[Union[Dict[str, int], int]], optional
        A dim to use for all features, or a
        Dict like {"feature_name": embedding size, ...}, by default None
    infer_dim_fn: Callable[[ColumnSchema], int], defaults to infer_embedding_dim
        The function to use to infer the embedding dimension, by default infer_embedding_dim
    sequence_combiner: Optional[Union[str, tf.keras.layers.Layer]], optional
       A string specifying how to combine embedding results for each
       entry ("mean", "sqrtn" and "sum" are supported) or a layer.
       Default is None (no combiner used)
    embeddings_initializer: Union[InitializerType, Dict[str, InitializerType]], optional
        An initializer function or a dict where keys are feature names and values are
        callable to initialize embedding tables. Pre-trained embeddings can be fed via
        embeddings_initializer arg.
    embeddings_regularizer: Union[RegularizerType, Dict[str, RegularizerType]], optional
        A regularizer function or a dict where keys are feature names and values are
        callable to apply regularization to embedding tables.
    activity_regularizer: Union[RegularizerType, Dict[str, RegularizerType]], optional
        A regularizer function or a dict where keys are feature names and values are
        callable to apply regularization to the activations of the embedding tables.
    trainable: Optional[Dict[str, bool]] = None
        Name of the column(s) whose embeddings should be frozen (or trainable) during training
        trainable will be set to False/True for these column(s), accordingly
    table_cls: Type[tf.keras.layers.Layer], by default EmbeddingTable
        The class to use for each embedding table.
    pre: Optional[BlockType], optional
        Transformation block to apply before the embeddings lookup, by default None
    post: Optional[BlockType], optional
        Transformation block to apply after the embeddings lookup, by default None
    aggregation: Optional[TabularAggregationType], optional
        Transformation block to apply for aggregating the inputs, by default None
    block_name: str, optional
        Name of the block, by default "embeddings"
    l2_batch_regularization_factor: Optional[float, Dict[str, float]] = 0.0
        Factor for L2 regularization of the embeddings vectors (from the current batch only)
        If a dictionary is provided, the keys are feature names and the values are
        regularization factors
    Returns
    -------
    ParallelBlock
        Returns a parallel block with an embedding table for each categorical features
    """
    if trainable:
        kwargs["trainable"] = trainable
    if embeddings_initializer:
        kwargs["embeddings_initializer"] = embeddings_initializer
    if embeddings_regularizer:
        kwargs["embeddings_regularizer"] = embeddings_regularizer
    if activity_regularizer:
        kwargs["activity_regularizer"] = activity_regularizer
    if sequence_combiner:
        kwargs["sequence_combiner"] = sequence_combiner
    if l2_batch_regularization_factor:
        kwargs["l2_batch_regularization_factor"] = l2_batch_regularization_factor

    tables = {}

    for col in schema:
        table_kwargs = _forward_kwargs_to_table(col, table_cls, kwargs)
        table_name = col.int_domain.name or col.name
        if table_name in tables:
            tables[table_name].add_feature(col)
        else:
            tables[table_name] = table_cls(
                _get_dim(col, dim, infer_dim_fn),
                col,
                name=table_name,
                **table_kwargs,
            )

    return ParallelBlock(
        tables, pre=pre, post=post, aggregation=aggregation, name=block_name, schema=schema
    )


def _forward_kwargs_to_table(col, table_cls, kwargs):
    arg_spec = inspect.getfullargspec(table_cls.__init__)
    supported_kwargs = arg_spec.kwonlyargs
    if arg_spec.defaults:
        supported_kwargs += arg_spec.args[-len(arg_spec.defaults) :]

    table_kwargs = {}
    for key, val in kwargs.items():
        if key in supported_kwargs:
            if isinstance(val, dict):
                if col.name in val:
                    table_kwargs[key] = val[col.name]
            else:
                table_kwargs[key] = val

    return table_kwargs


def _get_dim(col, embedding_dims, infer_dim_fn):
    dim = None
    if isinstance(embedding_dims, dict):
        dim = embedding_dims.get(col.name)
    elif isinstance(embedding_dims, int):
        dim = embedding_dims

    if not dim:
        dim = infer_dim_fn(col)

    return dim


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class AverageEmbeddingsByWeightFeature(tf.keras.layers.Layer):
    def __init__(self, weight_feature_name: str, axis=1, **kwargs):
        """Computes the weighted average of a Tensor based
        on one of the input features.
        Typically used as a combiner for EmbeddingTable
        for aggregating sequential embedding features

        Parameters
        ----------
        weight_feature_name : str
            Name of the feature to be used as weight for average
        axis : int, optional
            Axis for reduction, by default 1 (assuming the 2nd dim is
            the sequence length)
        """
        super(AverageEmbeddingsByWeightFeature, self).__init__(**kwargs)
        self.axis = axis
        self.weight_feature_name = weight_feature_name

    def call(self, inputs, features):
        weight_feature = features[self.weight_feature_name]
        if isinstance(inputs, tf.RaggedTensor) and not isinstance(weight_feature, tf.RaggedTensor):
            raise ValueError(
                f"If inputs is a tf.RaggedTensor, the weight feature ({self.weight_feature_name}) "
                f"should also be a tf.RaggedTensor (and not a {type(weight_feature)}), "
                "so that the list length can vary per example for both input embedding "
                "and weight features."
            )

        weights = tf.cast(weight_feature, tf.float32)
        if len(weight_feature.shape) == 2:
            weights = tf.expand_dims(weights, -1)
        output = tf.divide(
            tf.reduce_sum(tf.multiply(inputs, weights), axis=self.axis),
            tf.reduce_sum(weights, axis=self.axis),
        )
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    def from_schema_convention(schema: Schema, weight_features_name_suffix: str = "_weight"):
        """Infers the weight features corresponding to sequential embedding
        features based on the feature name suffix. For example, if a
        sequential categorical feature is called `item_id_seq`, if there is another
        feature in the schema called `item_id_seq_weight`, then it will be used
        for weighted average. If a weight feature cannot be found for a given
        seq cat. feature then standard mean is used as combiner


        Parameters
        ----------
        schema : Schema
            The feature schema
        weight_features_name_suffix : str
            Suffix to look for a corresponding weight feature

        Returns
        -------
        Dict[str, WeightedAverageByFeature]
            A dict where the key is the sequential categorical feature name and the value
            is an instance of WeightedAverageByFeature with the corresponding weight feature name
        """
        cat_cols = schema.select_by_tag(Tags.CATEGORICAL)
        seq_combiners = {}
        for cat_col in cat_cols:
            combiner = None
            if Tags.SEQUENCE in cat_col.tags:
                weight_col_name = f"{cat_col.name}{weight_features_name_suffix}"
                if weight_col_name in schema.column_names:
                    combiner = AverageEmbeddingsByWeightFeature(weight_col_name)
                else:
                    combiner = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))
                seq_combiners[cat_col.name] = combiner

        return seq_combiners

    def get_config(self):
        config = super().get_config()
        config["axis"] = self.axis
        config["weight_feature_name"] = self.weight_feature_name

        return config


@dataclass
class EmbeddingOptions:
    embedding_dims: Optional[Dict[str, int]] = None
    embedding_dim_default: Optional[int] = 64
    infer_embedding_sizes: bool = False
    infer_embedding_sizes_multiplier: float = 2.0
    infer_embeddings_ensure_dim_multiple_of_8: bool = False
    embeddings_initializers: Optional[
        Union[Dict[str, Callable[[Any], None]], Callable[[Any], None], str]
    ] = None
    embeddings_l2_reg: float = 0.0
    combiner: Optional[str] = "mean"


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
        feature_config: Dict[str, "FeatureConfig"],
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name=None,
        add_default_pre=True,
        l2_reg: Optional[float] = 0.0,
        **kwargs,
    ):
        if add_default_pre:
            embedding_pre = [Filter(list(feature_config.keys()))]
            pre = [embedding_pre, pre] if pre else embedding_pre  # type: ignore
        self.feature_config = feature_config
        self.l2_reg = l2_reg

        self.embedding_tables = {}
        tables: Dict[str, TableConfig] = {}
        for _, feature in self.feature_config.items():
            table: TableConfig = feature.table
            if table.name not in tables:
                tables[table.name] = table

        for table_name, table in tables.items():
            self.embedding_tables[table_name] = tf.keras.layers.Embedding(
                table.vocabulary_size,
                table.dim,
                name=table_name,
                embeddings_initializer=table.initializer,
            )

        kwargs["is_input"] = kwargs.get("is_input", True)
        super().__init__(
            pre=pre,
            post=post,
            aggregation=aggregation,
            name=name,
            schema=schema,
            **kwargs,
        )

    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        embedding_options: EmbeddingOptions = EmbeddingOptions(),
        tags: Optional[TagsType] = None,
        max_sequence_length: Optional[int] = None,
        **kwargs,
    ) -> Optional["EmbeddingFeatures"]:
        """Instantiates embedding features from the schema

        Parameters
        ----------
        schema : Schema
            The features chema
        embedding_options : EmbeddingOptions, optional
            An EmbeddingOptions instance, which allows for a number of
            options for the embedding table, by default EmbeddingOptions()
        tags : Optional[TagsType], optional
            If provided, keeps only features from those tags, by default None
        max_sequence_length : Optional[int], optional
            Maximum sequence length of sparse features (if any), by default None

        Returns
        -------
        EmbeddingFeatures
            An instance of EmbeddingFeatures block, with the embedding
            layers created under-the-hood
        """

        if tags:
            schema = schema.select_by_tag(tags)

        embedding_dims = embedding_options.embedding_dims or {}
        if embedding_options.infer_embedding_sizes:
            inferred_embedding_dims = schema_utils.get_embedding_sizes_from_schema(
                schema,
                embedding_options.infer_embedding_sizes_multiplier,
                embedding_options.infer_embeddings_ensure_dim_multiple_of_8,
            )
            # Adding inferred embedding dims only for features where the embedding sizes
            # were not pre-defined
            inferred_embedding_dims = {
                k: v for k, v in inferred_embedding_dims.items() if k not in embedding_dims
            }
            embedding_dims = {**embedding_dims, **inferred_embedding_dims}

        initializer_default = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05)
        embeddings_initializer = embedding_options.embeddings_initializers or initializer_default

        emb_config = {}
        cardinalities = schema_utils.categorical_cardinalities(schema)
        for key, cardinality in cardinalities.items():
            embedding_size = embedding_dims.get(key, embedding_options.embedding_dim_default)
            if isinstance(embeddings_initializer, dict):
                emb_initializer = embeddings_initializer.get(key, initializer_default)
            else:
                emb_initializer = embeddings_initializer
            emb_config[key] = (cardinality, embedding_size, emb_initializer)

        feature_config: Dict[str, FeatureConfig] = {}
        tables: Dict[str, TableConfig] = {}

        domains = schema_utils.categorical_domains(schema)
        for name, (vocab_size, dim, emb_initilizer) in emb_config.items():
            table_name = domains[name]
            table = tables.get(table_name, None)
            if not table:
                table = TableConfig(
                    vocabulary_size=vocab_size,
                    dim=dim,
                    name=table_name,
                    combiner=embedding_options.combiner,
                    initializer=emb_initilizer,
                )
                tables[table_name] = table
            feature_config[name] = FeatureConfig(table)

        if not feature_config:
            return None

        schema = schema.select_by_name(list(feature_config.keys()))

        output = cls(
            feature_config,
            schema=schema,
            l2_reg=embedding_options.embeddings_l2_reg,
            **kwargs,
        )

        return output

    def build(self, input_shapes):
        for name, embedding_table in self.embedding_tables.items():
            embedding_table.build(())

            if hasattr(self, "_context"):
                self._context.add_embedding_table(name, self.embedding_tables[name])

        if isinstance(input_shapes, dict):
            super().build(input_shapes)
        else:
            tf.keras.layers.Layer.build(self, input_shapes)

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        embedded_outputs = {}
        for name, val in inputs.items():
            embedded_outputs[name] = self.lookup_feature(name, val)
            if self.l2_reg > 0:
                self.add_loss(self.l2_reg * tf.reduce_sum(tf.square(embedded_outputs[name])))

        return embedded_outputs

    def compute_call_output_shape(self, input_shapes):
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)

        output_shapes = {}
        for name, val in input_shapes.items():
            output_shapes[name] = tf.TensorShape([batch_size, self.feature_config[name].table.dim])

        return output_shapes

    def lookup_feature(self, name, val, output_sequence=False):
        dtype = backend.dtype(val)
        if dtype != "int32" and dtype != "int64":
            val = tf.cast(val, "int32")

        table: TableConfig = self.feature_config[name].table
        table_var = self.embedding_tables[table.name].embeddings
        if isinstance(val, (tf.RaggedTensor, tf.SparseTensor)):
            if isinstance(val, tf.RaggedTensor):
                val = val.to_sparse()

            val = tf.sparse.reshape(val, tf.shape(val)[:-1])

            out = tf.nn.safe_embedding_lookup_sparse(table_var, val, None, combiner=table.combiner)
        else:
            if output_sequence:
                out = tf.gather(table_var, tf.cast(val, tf.int32))
            else:
                if len(val.shape) > 1 and val.shape.as_list()[-1] == 1:
                    val = tf.squeeze(val, axis=-1)
                out = tf.gather(table_var, tf.cast(val, tf.int32))

            if len(out.get_shape()) > 2 and table.combiner is not None:
                out = process_str_sequence_combiner(out, table.combiner)

        if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
            # Instead of casting the variable as in most layers, cast the output, as
            # this is mathematically equivalent but is faster.
            out = tf.cast(out, self._dtype_policy.compute_dtype)

        return out

    def table_config(self, feature_name: str):
        return self.feature_config[feature_name].table

    def get_embedding_table(self, table_name: Union[str, Tags], l2_normalization: bool = False):
        if isinstance(table_name, Tags):
            feature_names = self.schema.select_by_tag(table_name).column_names
            if len(feature_names) == 1:
                table_name = feature_names[0]
            elif len(feature_names) > 1:
                raise ValueError(
                    f"There is more than one feature associated to the tag {table_name}"
                )
            else:
                raise ValueError(f"Could not find a feature associated to the tag {table_name}")

        embeddings = self.embedding_tables[table_name].embeddings
        if l2_normalization:
            embeddings = tf.linalg.l2_normalize(embeddings, axis=-1)

        return embeddings

    def embedding_table_df(
        self, table_name: Union[str, Tags], l2_normalization: bool = False, gpu: bool = True
    ):
        """Retrieves a dataframe with the embedding table

        Parameters
        ----------
        table_name : Union[str, Tags]
            Tag or name of the embedding table
        l2_normalization : bool, optional
            Whether the L2-normalization should be applied to
            embeddings (common approach for Matrix Factorization
            and Retrieval models in general), by default False
        gpu : bool, optional
            Whether or not should use GPU, by default True

        Returns
        -------
        Union[pd.DataFrame, cudf.DataFrame]
            Returns a dataframe (cudf or pandas), depending on the gpu
        """
        embeddings = self.get_embedding_table(table_name, l2_normalization)
        if gpu:
            import cudf
            import cupy

            # Note: It is not possible to convert Tensorflow tensors to the cudf dataframe
            # directly using dlPack (as the example commented below) because cudf.from_dlpack()
            # expects the 2D tensor to be in Fortran order (column-major), which is not
            # supported by TF (https://github.com/rapidsai/cudf/issues/10754).
            # df = cudf.from_dlpack(to_dlpack(tf.convert_to_tensor(embeddings)))
            embeddings_cupy = cupy.fromDlpack(to_dlpack(tf.convert_to_tensor(embeddings)))
            df = cudf.DataFrame(embeddings_cupy)
            df.columns = [str(col) for col in list(df.columns)]
            df.set_index(cudf.RangeIndex(0, embeddings.shape[0]))
        else:
            import pandas as pd

            df = pd.DataFrame(embeddings.numpy())
            df.columns = [str(col) for col in list(df.columns)]
            df.set_index(pd.RangeIndex(0, embeddings.shape[0]))

        return df

    def embedding_table_dataset(
        self, table_name: Union[str, Tags], l2_normalization: bool = False, gpu=True
    ) -> merlin.io.Dataset:
        """Creates a Dataset for the embedding table

        Parameters
        ----------
        table_name : Union[str, Tags]
            Tag or name of the embedding table
        l2_normalization : bool, optional
            Whether the L2-normalization should be applied to
            embeddings (common approach for Matrix Factorization
            and Retrieval models in general), by default False
        gpu : bool, optional
            Whether or not should use GPU, by default True

        Returns
        -------
        merlin.io.Dataset
            Returns a Dataset with the embeddings
        """
        return merlin.io.Dataset(self.embedding_table_df(table_name, l2_normalization, gpu))

    def export_embedding_table(
        self,
        table_name: Union[str, Tags],
        export_path: str,
        l2_normalization: bool = False,
        gpu=True,
    ):
        """Exports the embedding table to parquet file

        Parameters
        ----------
        table_name : Union[str, Tags]
            Tag or name of the embedding table
        export_path : str
            Path for the generated parquet file
        l2_normalization : bool, optional
            Whether the L2-normalization should be applied to
            embeddings (common approach for Matrix Factorization
            and Retrieval models in general), by default False
        gpu : bool, optional
            Whether or not should use GPU, by default True
        """
        df = self.embedding_table_df(table_name, l2_normalization, gpu=gpu)
        df.to_parquet(export_path)

    def get_config(self):
        config = super().get_config()

        feature_configs = {}

        for key, val in self.feature_config.items():
            feature_config_dict = dict(name=val.name, max_sequence_length=val.max_sequence_length)

            feature_config_dict["table"] = serialize_table_config(val.table)
            feature_configs[key] = feature_config_dict

        config["feature_config"] = feature_configs
        config["l2_reg"] = self.l2_reg

        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize feature_config
        feature_configs = {}
        for key, val in config["feature_config"].items():
            table = deserialize_table_config(val["table"])
            feature_config_params = {**val, "table": table}
            feature_configs[key] = FeatureConfig(**feature_config_params)

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
            embedding_pre = [Filter(list(feature_config.keys()))]
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
    inputs: Block,
    embedding_block: Block,
    aggregation=None,
    continuous_aggregation="concat",
    name: str = "continuous",
    **kwargs,
) -> SequentialBlock:
    continuous_embedding = Filter(Tags.CONTINUOUS, aggregation=continuous_aggregation).connect(
        embedding_block
    )

    outputs = inputs.connect_branch(
        continuous_embedding.as_tabular(name), add_rest=True, aggregation=aggregation, **kwargs
    )

    return outputs


def serialize_table_config(table_config: TableConfig) -> Dict[str, Any]:
    table = deepcopy(table_config.__dict__)
    if "initializer" in table:
        table["initializer"] = tf.keras.initializers.serialize(table["initializer"])
    if "optimizer" in table:
        table["optimizer"] = tf.keras.optimizers.serialize(table["optimizer"])

    return table


def deserialize_table_config(table_params: Dict[str, Any]) -> TableConfig:
    if "initializer" in table_params and table_params["initializer"]:
        table_params["initializer"] = tf.keras.initializers.deserialize(table_params["initializer"])
    if "optimizer" in table_params and table_params["optimizer"]:
        table_params["optimizer"] = tf.keras.optimizers.deserialize(table_params["optimizer"])
    table = TableConfig(**table_params)

    return table


def serialize_feature_config(feature_config: FeatureConfig) -> Dict[str, Any]:
    outputs = {}

    for key, val in feature_config.items():
        feature_config_dict = dict(name=val.name, max_sequence_length=val.max_sequence_length)
        feature_config_dict["table"] = serialize_table_config(feature_config_dict["table"])
        outputs[key] = feature_config_dict

    return outputs


def process_str_sequence_combiner(
    inputs: Union[tf.Tensor, tf.RaggedTensor], combiner: str, **kwargs
) -> tf.Tensor:
    """Process inputs with str sequence combiners ("mean" or "sum")

    Parameters
    ----------
    inputs : Union[tf.Tensor, tf.RaggedTensor]
        Input 3D tensor (batch size, seq length, embedding dim)
    combiner : str
        The combiner: "mean" or "sum"

    Returns
    -------
    tf.Tensor
        A 2D tensor with values combined on axis=1
    """
    if not combiner or len(inputs.get_shape()) <= 2:
        return inputs
    if combiner == "mean":
        combiner = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))
    elif combiner == "sum":
        combiner = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))
    else:
        raise ValueError(
            "Only 'mean' and 'sum' str combiners is implemented for dense"
            " list/multi-hot embedded features. You can also"
            " provide a tf.keras.layers.Layer instance as a sequence combiner."
        )
    return call_layer(combiner, inputs, **kwargs)
