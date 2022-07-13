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
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.python import to_dlpack
from tensorflow.python.tpu.tpu_embedding_v2_utils import FeatureConfig, TableConfig

import merlin.io
from merlin.core.dispatch import DataFrameType
from merlin.io import Dataset
from merlin.models.tf.core.base import Block, BlockType
from merlin.models.tf.core.combinators import SequentialBlock
from merlin.models.tf.core.tabular import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    Filter,
    TabularAggregationType,
    TabularBlock,
)
from merlin.models.tf.core.transformations import AsDenseFeatures, AsSparseFeatures

# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter, unexpected-keyword-arg
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.tf_utils import df_to_tensor
from merlin.models.utils import schema_utils
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.models.utils.schema_utils import (
    create_categorical_column,
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
    def __init__(self, dim: int, col_schema: ColumnSchema, trainable=True, **kwargs):
        super(EmbeddingTableBase, self).__init__(trainable=trainable, **kwargs)
        self.dim = dim

        if not col_schema.int_domain:
            raise ValueError("`col_schema` needs to have a int-domain")

        self.col_schema = col_schema

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

    def get_config(self):
        config = super().get_config()
        config["dim"] = self.dim

        schema = schema_to_tensorflow_metadata_json(Schema([self.col_schema]))
        config["schema"] = schema

        return config

    @classmethod
    def from_config(cls, config):
        schema = tensorflow_metadata_json_to_schema(config.pop("schema"))
        col_schema = schema.first

        return cls(col_schema=col_schema, **config)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class EmbeddingTable(EmbeddingTableBase):
    """Embedding table that is backed by a standard Keras Embedding Layer.

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
       entry. Currently "mean", "sqrtn" and "sum" are supported.
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
    **kwargs: Forwarded Keras Layer parameters
    """

    def __init__(
        self,
        dim: int,
        col_schema: ColumnSchema,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        activity_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        input_length=None,
        combiner: Optional[str] = None,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        table=None,
        weights=None,
        **kwargs,
    ):
        """Create an EmbeddingTable."""
        super(EmbeddingTable, self).__init__(
            dim, col_schema, trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs
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
                weights=weights,
                input_length=input_length,
            )
            self.table = tf.keras.layers.Embedding(
                input_dim=self.input_dim,
                output_dim=self.dim,
                name=self.col_schema.name,
                **table_kwargs,
            )
        self.combiner = combiner

    @classmethod
    def from_pretrained(
        cls,
        data: Union[Dataset, DataFrameType],
        trainable=True,
        name=None,
        col_schema=None,
        **kwargs,
    ):
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

        return cls(
            dim,
            col_schema,
            name=name,
            weights=[tf.Variable(embeddings, trainable=trainable)],
            trainable=trainable,
            **kwargs,
        )

    def _maybe_build(self, inputs):
        """Creates state between layer instantiation and layer call.
        Invoked automatically before the first execution of `call()`.
        """
        self.table._maybe_build(inputs)
        return super(EmbeddingTable, self)._maybe_build(inputs)

    def call(self, inputs, **kwargs):
        """
        Parameters
        ----------
        inputs : Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor]
            Tensors representing the input batch

        Returns
        -------
        A tensor corresponding to the embeddings for inputs
        """
        dtype = backend.dtype(inputs)
        if dtype != "int32" and dtype != "int64":
            inputs = tf.cast(inputs, "int32")

        if self.combiner:
            if not isinstance(inputs, (tf.RaggedTensor, tf.SparseTensor)):
                raise ValueError(
                    "Combiner only supported for RaggedTensor and SparseTensor. "
                    f"Received: {type(inputs)}"
                )
            if isinstance(inputs, tf.RaggedTensor):
                inputs = inputs.to_sparse()
            out = tf.nn.safe_embedding_lookup_sparse(
                self.table.embeddings, inputs, None, combiner=self.combiner
            )
            if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
                # Instead of casting the variable as in most layers, cast the output, as
                # this is mathematically equivalent but is faster.
                out = tf.cast(out, self._dtype_policy.compute_dtype)
        else:
            if not isinstance(inputs, (tf.RaggedTensor, tf.Tensor)):
                raise ValueError(
                    "EmbeddingTable supports only RaggedTensor and Tensor input types. "
                    f"Received: {type(inputs)}"
                )
            out = self.table(inputs)

        return out

    @classmethod
    def from_config(cls, config):
        config["table"] = tf.keras.layers.deserialize(config["table"])

        return super().from_config(config)

    def get_config(self):
        config = super().get_config()
        config["table"] = tf.keras.layers.serialize(self.table)
        config["combiner"] = self.combiner

        return config


@dataclass
class EmbeddingOptions:
    embedding_dims: Optional[Dict[str, int]] = None
    embedding_dim_default: Optional[int] = 64
    infer_embedding_sizes: bool = False
    infer_embedding_sizes_multiplier: float = 2.0
    infer_embeddings_ensure_dim_multiple_of_8: bool = False
    embeddings_initializers: Optional[
        Union[Dict[str, Callable[[Any], None]], Callable[[Any], None]]
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
            embedding_pre = [Filter(list(feature_config.keys())), AsSparseFeatures()]
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
        schema_copy = copy(schema)

        if tags:
            schema_copy = schema_copy.select_by_tag(tags)

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

        output = cls(
            feature_config,
            schema=schema_copy,
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
        if isinstance(val, tf.SparseTensor):
            out = tf.nn.safe_embedding_lookup_sparse(table_var, val, None, combiner=table.combiner)
        else:
            if output_sequence:
                out = tf.gather(table_var, tf.cast(val, tf.int32))
            else:
                if len(val.shape) > 1:
                    # TODO: Check if it is correct to retrieve only the 1st element
                    # of second dim for non-sequential multi-hot categ features
                    out = tf.gather(table_var, tf.cast(val, tf.int32)[:, 0])
                else:
                    out = tf.gather(table_var, tf.cast(val, tf.int32))
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
