#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

import os
from typing import Dict, Optional, Union

import numpy as np
import tensorflow as tf

import merlin.io
from merlin.models.io import save_merlin_metadata
from merlin.models.tf.core import combinators
from merlin.models.tf.core.base import NoOp
from merlin.models.tf.core.prediction import TopKPrediction
from merlin.models.tf.inputs.base import InputBlockV2
from merlin.models.tf.inputs.embedding import CombinerType, EmbeddingTable
from merlin.models.tf.loader import Loader
from merlin.models.tf.models.base import BaseModel, get_output_schema
from merlin.models.tf.outputs.topk import TopKOutput
from merlin.models.tf.transforms.features import PrepareFeatures
from merlin.models.tf.utils import tf_utils
from merlin.models.tf.utils.batch_utils import TFModelEncode
from merlin.schema import ColumnSchema, Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class Encoder(tf.keras.Model):
    """Block that can be used for prediction and evaluation but not for training

    Parameters
    ----------
    inputs: Union[Schema, tf.keras.layers.Layer]
        The input block or schema.
        When a schema is provided, a default input block will be created.
    *blocks: tf.keras.layers.Layer
        The blocks to use for encoding.
    pre: Optional[tf.keras.layers.Layer]
        A block to use before the main blocks
    post: Optional[tf.keras.layers.Layer]
        A block to use after the main blocks
    prep_features: Optional[bool]
        Whether this block should prepare list and scalar features
        from the dataloader format. By default True.
    """

    def __init__(
        self,
        inputs: Union[Schema, tf.keras.layers.Layer],
        *blocks: tf.keras.layers.Layer,
        pre: Optional[tf.keras.layers.Layer] = None,
        post: Optional[tf.keras.layers.Layer] = None,
        prep_features: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(inputs, Schema):
            input_block = InputBlockV2(inputs)
            self._schema = inputs
        else:
            input_block = inputs
            if not hasattr(inputs, "schema"):
                raise ValueError("inputs must have a schema")
            self._schema = inputs.schema

        self.blocks = [input_block] + list(blocks) if blocks else [input_block]
        self.pre = pre
        self.post = post

        self.prep_features = prep_features

        self._prepare_features = PrepareFeatures(self.schema) if self.prep_features else NoOp()

    def encode(
        self,
        dataset: Union[merlin.io.Dataset, Loader],
        index: Union[str, ColumnSchema, Schema, Tags],
        batch_size: int,
        **kwargs,
    ) -> merlin.io.Dataset:
        """Encodes the given dataset and index.

        Parameters
        ----------
        dataset: Union[merlin.io.Dataset, merlin.models.tf.loader.Loader]
            The dataset to encode.
        index: Union[str, ColumnSchema, Schema, Tags]
            The index to use for encoding.
        batch_size: int
            The batch size for encoding.

        Returns
        -------
        merlin.io.Dataset
            The encoded dataset.
        """
        if isinstance(index, Schema):
            output_schema = index
        elif isinstance(index, ColumnSchema):
            output_schema = Schema([index])
        elif isinstance(index, str):
            output_schema = Schema([self.schema[index]])
        elif isinstance(index, Tags):
            output_schema = self.schema.select_by_tag(index)
        else:
            raise ValueError(f"Invalid index: {index}")

        return self.batch_predict(
            dataset,
            batch_size=batch_size,
            output_schema=output_schema,
            index=index,
            output_concat_func=np.concatenate,
            **kwargs,
        )

    def batch_predict(
        self,
        dataset: Union[merlin.io.Dataset, Loader],
        batch_size: int,
        output_schema: Optional[Schema] = None,
        index: Optional[Union[str, ColumnSchema, Schema, Tags]] = None,
        **kwargs,
    ) -> merlin.io.Dataset:
        """Batched prediction using Dask.

        Parameters
        ----------
        dataset: Union[merlin.io.Dataset, merlin.models.tf.loader.Loader]
            Dataset or Loader to predict on.
        batch_size: int
            Batch size to use for prediction.

        Returns
        -------
        merlin.io.Dataset
        """

        if index:
            if isinstance(index, ColumnSchema):
                index = Schema([index])
            elif isinstance(index, str):
                index = Schema([self.schema[index]])
            elif isinstance(index, Tags):
                index = self.schema.select_by_tag(index)
            elif not isinstance(index, Schema):
                raise ValueError(f"Invalid index: {index}")

            if len(index) != 1:
                raise ValueError("Only one column can be used as index")
            index = index.first.name

        dataset_schema = None
        if hasattr(dataset, "schema"):
            dataset_schema = dataset.schema
            data_output_schema = dataset_schema
            if isinstance(dataset, Loader):
                data_output_schema = dataset.output_schema
            if not set(self.schema.column_names).issubset(set(data_output_schema.column_names)):
                raise ValueError(
                    f"Model schema {self.schema.column_names} does not match dataset schema"
                    + f" {data_output_schema.column_names}"
                )

        loader_transforms = None
        if isinstance(dataset, Loader):
            loader_transforms = dataset.transforms
            batch_size = dataset.batch_size
            dataset = dataset.dataset

        # Check if merlin-dataset is passed
        if hasattr(dataset, "to_ddf"):
            dataset = dataset.to_ddf()

        model_encode = TFModelEncode(
            self,
            batch_size=batch_size,
            loader_transforms=loader_transforms,
            schema=dataset_schema,
            **kwargs,
        )

        encode_kwargs = {}
        if output_schema:
            encode_kwargs["filter_input_columns"] = output_schema.column_names

        # Processing a sample of the dataset with the model encoder
        # to get the output dataframe dtypes
        sample_output = model_encode(dataset.head(), **encode_kwargs)
        output_dtypes = sample_output.dtypes.to_dict()

        predictions = dataset.map_partitions(model_encode, meta=output_dtypes, **encode_kwargs)
        if index:
            predictions = predictions.set_index(index)

        return merlin.io.Dataset(predictions)

    def call(self, inputs, *, targets=None, training=False, testing=False, **kwargs):
        """Calls the model on new inputs and returns the outputs as tensors.

        Parameters
        ----------
        inputs : tensor-like or dict/tuple of tensors.
            Tensors or dict/tuple of tensors representing the input batch.
        targets : tensor-like, optional
            Tensors representing the target data.
        training : bool, optional
            Whether the model is in training mode.
        testing : bool, optional
            Whether the model is in testing mode.

        Returns
        -------
        A tensor or dict of tensors corresponding to the result of calling the layer.
        """
        inputs = self._prepare_features(inputs, targets=targets)
        if isinstance(inputs, tuple):
            inputs, targets = inputs
        return combinators.call_sequentially(
            list(self.to_call),
            inputs=inputs,
            features=inputs,
            targets=targets,
            training=training,
            testing=testing,
            **kwargs,
        )

    def __call__(self, inputs, **kwargs):
        """Overrides the default __call__ method to remove "features" from inputs.

        Parameters
        ----------
        inputs : tensor-like or dict/tuple of tensors.
            Tensors or dict/tuple of tensors representing the input batch.

        Returns
        -------
        A tensor or dict of tensors corresponding to the result of calling the layer.
        """
        # We remove features here since we don't expect them at inference time
        # Inside the `call` method, we will add them back by assuming inputs=features
        if "features" in kwargs:
            kwargs.pop("features")

        return super().__call__(inputs, **kwargs)

    def build(self, input_shape):
        """Creates the variables of the layer.

        Parameters
        ----------
        input_shape: Tuple[int]
            The shape of the input data.
        """
        self._prepare_features.build(input_shape)
        input_shape = self._prepare_features.compute_output_shape(input_shape)

        combinators.build_sequentially(self, list(self.to_call), input_shape=input_shape)
        if not hasattr(self.build, "_is_default"):
            self._build_input_shape = input_shape

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Parameters
        ----------
        input_shape: Tuple[int]
            The shape of the input data.

        Returns
        -------
        Tuple[int]
            The output shape of the layer.
        """
        input_shape = self._prepare_features.compute_output_shape(input_shape)
        return combinators.compute_output_shape_sequentially(list(self.to_call), input_shape)

    def train_step(self, data):
        """Performs a training step.

        Train step method is not implemented and Raises an error as the
        Encoder block is not meant to be trained by itself and can only be
        trained as part of a model.
        """
        raise NotImplementedError(
            "This block is not meant to be trained by itself. ",
            "It can only be trained as part of a model.",
        )

    def fit(self, *args, **kwargs):
        """Fits the model.

        Fit method is not implemented and Raises an error as the Encoder block
        is not meant to be trained by itself and can only be trained as part
        of a model.
        """
        raise NotImplementedError(
            "This block is not meant to be trained by itself. ",
            "It can only be trained as part of a model.",
        )

    def save(
        self,
        export_path: Union[str, os.PathLike],
        include_optimizer=True,
        save_traces=True,
    ) -> None:
        """Saves the model to export_path as a Tensorflow Saved Model.
        Along with merlin model metadata.

        Parameters
        ----------
        export_path : Union[str, os.PathLike]
            Path where model will be saved to
        include_optimizer : bool, optional
            If False, do not save the optimizer state, by default True
        save_traces : bool, optional
            When enabled, will store the function traces for each layer. This
            can be disabled, so that only the configs of each layer are
            stored, by default True
        """
        super().save(
            export_path,
            include_optimizer=include_optimizer,
            save_traces=save_traces,
            save_format="tf",
        )
        input_schema = self.schema
        output_schema = get_output_schema(export_path)
        save_merlin_metadata(export_path, input_schema, output_schema)

    @property
    def to_call(self):
        """Provides the list of blocks to be called during the execution of the model."""
        if self.pre:
            yield self.pre

        for block in self.blocks:
            yield block

        if self.post:
            yield self.post

    @property
    def has_schema(self) -> bool:
        """Returns True as this class does contain a schema."""
        return True

    @property
    def schema(self) -> Schema:
        """Returns the schema of the model."""
        return self._schema

    @property
    def first(self):
        """Returns the first block of the model."""
        return self.blocks[0]

    @property
    def last(self):
        """Returns the last block of the model."""
        return self.blocks[-1]

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Creates a new instance of the class by deserializing.

        Parameters
        ----------
        config: dict
            A dictionary, typically the output of get_config.
        custom_objects: dict, optional
            A dictionary mapping the names of layers to the corresponding
            functions and classes.

        Returns
        -------
        A new instance of Encoder.
        """
        pre = config.pop("pre", None)
        post = config.pop("post", None)
        layers = [
            tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
            for conf in config.values()
        ]

        if pre is not None:
            pre = tf.keras.layers.deserialize(pre, custom_objects=custom_objects)

        if post is not None:
            post = tf.keras.layers.deserialize(post, custom_objects=custom_objects)

        output = Encoder(*layers, pre=pre, post=post)
        output.__class__ = cls

        return output

    def get_config(self):
        """Returns the configuration of the model as a dictionary.

        Returns
        -------
        dict
            The configuration of the model.
        """
        config = tf_utils.maybe_serialize_keras_objects(self, {}, ["pre", "post"])
        for i, layer in enumerate(self.blocks):
            config[i] = tf.keras.utils.serialize_keras_object(layer)

        return config


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class TopKEncoder(Encoder, BaseModel):
    """Block that can be used for top-k prediction & evaluation, initialized
    from a trained retrieval model

    Parameters
    ----------
    query_encoder: Union[Encoder, tf.keras.layers.Layer],
        The layer to use for encoding the query features
    topk_layer: Union[str, tf.keras.layers.Layer, TopKOutput]
        The layer to use for computing the top-k predictions.
        You can also pass the `name` of registered top-k layer.
        The current supported strategies are [`brute-force-topk`]
        By default "brute-force-topk"
    candidates: Union[tf.Tensor, ~merlin.io.Dataset]
        The candidate embeddings to use for the Top-k index.
        You can pass a tensor of pre-trained embeddings
        or a merlin.io.Dataset of pre-trained embeddings, indexed by
        the candidates ids.
        This is required when `topk_layer` is a string
        By default None
    candidate_encoder:  Union[Encoder, tf.keras.layers.Layer],
        The layer to use for encoding the item features
    k: int, Optional
        Number of candidates to return, by default 10
    pre: Optional[tf.keras.layers.Layer]
        A block to use before encoding the input query
        By default None
    post: Optional[tf.keras.layers.Layer]
        A block to use after getting the top-k prediction scores
        By default None
    target: str, optional
        The name of the target. This is required when multiple targets are provided.
        By default None
    """

    def __init__(
        self,
        query_encoder: Union[Encoder, tf.keras.layers.Layer],
        topk_layer: Union[str, tf.keras.layers.Layer, TopKOutput] = "brute-force-topk",
        candidates: Union[tf.Tensor, merlin.io.Dataset] = None,
        candidate_encoder: Union[Encoder, tf.keras.layers.Layer] = None,
        k: int = 10,
        pre: Optional[tf.keras.layers.Layer] = None,
        post: Optional[tf.keras.layers.Layer] = None,
        target: str = None,
        **kwargs,
    ):
        if isinstance(topk_layer, TopKOutput):
            topk_output = topk_layer
        else:
            topk_output = TopKOutput(to_call=topk_layer, candidates=candidates, k=k, target=target)
        self.k = k

        Encoder.__init__(self, query_encoder, topk_output, pre=pre, post=post, **kwargs)
        # The base model is required for the evaluation step:
        BaseModel.__init__(self, **kwargs)

    @classmethod
    def from_candidate_dataset(
        cls,
        query_encoder: Union[Encoder, tf.keras.layers.Layer],
        candidate_encoder: Union[Encoder, tf.keras.layers.Layer],
        dataset: merlin.io.Dataset,
        top_k: int = 10,
        index_column: Optional[Union[str, ColumnSchema, Schema, Tags]] = None,
        **kwargs,
    ):
        """Class method to initialize a TopKEncoder from a dataset of
        raw candidates features.

        Parameters
        ----------
        query_encoder : Union[Encoder, tf.keras.layers.Layer]
            The encoder layer to use for computing the query embeddings.
        candidate_encoder : Union[Encoder, tf.keras.layers.Layer]
            The encoder layer to use for computing the candidates embeddings.
        dataset : merlin.io.Dataset
            Raw candidate features dataset
        index_column : Union[str, ColumnSchema, Schema, Tags], optional
            The column to use as candidates identifiers, this will be used
            for returning the topk ids of candidates with the highest scores.
            If not specified, the candidates indices will be used instead.
            by default None
        top_k : int, optional
            Number of candidates to return, by default 10

        Returns
        -------
        TopKEncoder
            a `TopKEncoder` indexed by the pre-trained embeddings of the candidates
            in the specified `dataset`
        """
        # TODO: Add related unit-test after RetrievalModelV2 is merged
        candidates = cls.encode_candidates(dataset, candidate_encoder)
        topk_output = TopKOutput(
            to_call="topk_layer", candidate_dataset=candidates, k=top_k, **kwargs
        )
        return cls(query_encoder, topk_output, **kwargs)

    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        jit_compile=None,
        k: int = None,
        **kwargs,
    ):
        """Extend the compile method of `BaseModel` to set the threshold `k`
        of the top-k encoder.
        """
        if k is not None:
            self.topk_layer._k = k
            self.k = k
        BaseModel.compile(
            self,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            loss_weights=loss_weights,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            **kwargs,
        )

    @property
    def topk_layer(self):
        return self.blocks[-1].to_call

    def index_candidates(self, candidates, identifiers=None):
        self.topk_layer.index(candidates, identifiers=identifiers)
        return self

    def encode_candidates(
        self,
        dataset: merlin.io.Dataset,
        index_column: Optional[Union[str, ColumnSchema, Schema, Tags]] = None,
        candidate_encoder: Optional[Union[Encoder, tf.keras.layers.Layer]] = None,
        **kwargs,
    ) -> merlin.io.Dataset:
        """Method to generate candidates embeddings

        Parameters
        ----------
        dataset : merlin.io.Dataset
            Raw candidate features dataset
        index_column  : Union[str, ColumnSchema, Schema, Tags], optional
            The column to use as candidates identifiers, this will be used
            for returning the topk ids of candidates with the highest scores.
            If not specified, the candidates indices will be used instead.
            by default None
        candidate_encoder : Union[Encoder, tf.keras.layers.Layer], optional
            The encoder layer to use for computing the candidates embeddings.
            If not specified, the candidate_encoder set in the constructor
            will be used instead.
            by default None
        Returns
        -------
        merlin.io.Dataset
            A merlin dataset of candidates embeddings, indexed by index_column.
        """
        # TODO: Add related unit-test after RetrievalModelV2 is merged
        if not candidate_encoder:
            candidate_encoder = self.candidate_encoder
        assert candidate_encoder is not None, ValueError(
            "You should provide a `candidate_encoder` to compute candidates embeddings"
        )
        return candidate_encoder.encode(dataset=dataset, index=index_column, **kwargs)

    def batch_predict(
        self,
        dataset: Union[merlin.io.Dataset, Loader],
        batch_size: int,
        output_schema: Optional[Schema] = None,
        **kwargs,
    ) -> merlin.io.Dataset:
        """Batched top-k prediction using Dask.

        Parameters
        ----------
        dataset : Union[merlin.io.Dataset, merlin.models.tf.loader.Loader]
            Raw queries features dataset or Loader
        batch_size : int
            The number of queries to process at each prediction step
        output_schema: Schema, optional
            The columns to output from the input dataset
        Returns
        -------
        merlin.io.Dataset
            A merlin dataset with the top-k predictions, the
            candidates identifiers and related scores.
        """
        from merlin.models.tf.utils.batch_utils import TFModelEncode

        loader_transforms = None
        if isinstance(dataset, Loader):
            loader_transforms = dataset.transforms
            batch_size = dataset.batch_size
            dataset = dataset.dataset

        dataset_schema = dataset.schema
        dataset = dataset.to_ddf()

        model_encode = TFModelEncode(
            model=self,
            batch_size=batch_size,
            loader_transforms=loader_transforms,
            schema=dataset_schema,
            output_names=TopKPrediction.output_names(self.k),
            **kwargs,
        )

        encode_kwargs = {}
        if output_schema:
            encode_kwargs["filter_input_columns"] = output_schema.column_names

        # Processing a sample of the dataset with the model encoder
        # to get the output dataframe dtypes
        sample_output = model_encode(dataset.head(), **encode_kwargs)
        output_dtypes = sample_output.dtypes.to_dict()

        predictions = dataset.map_partitions(model_encode, meta=output_dtypes, **encode_kwargs)

        return merlin.io.Dataset(predictions)

    def fit(self, *args, **kwargs):
        """Fit model"""
        raise NotImplementedError(
            "This block is not meant to be trained by itself. ",
            "It can only be trained as part of a model.",
        )


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class EmbeddingEncoder(Encoder):
    """Creates an Encoder from an EmbeddingTable.
    Typically used with RetrievalModelV2.

    Parameters
    ----------
    schema : Union[ColumnSchema, Schema]
        The ColumnSchema of the column for which the
        embedding table needs to be created.
        If a Schema is passed, only the first column
        is considered
    dim : int
        Dimension of the embeddings
    embeddings_initializer : Union[str, tf.keras.layers.Layer], optional
        Initializer for the `embeddings`
        matrix (see `keras.initializers`). By default "uniform"
    embeddings_regularizer : Union[str, tf.keras.layers.Layer], optional
        Regularizer function applied to
        the `embeddings` matrix (see `keras.regularizers`)., by default None
    activity_regularizer : Union[str, tf.keras.layers.Layer], optional
        Sets a layer that applies an update to the cost function based
        input activity, by default None
    embeddings_constraint : Union[str, tf.keras.layers.Layer], optional
        Constraint function applied to
        the `embeddings` matrix (see `keras.constraints`), by default None
    mask_zero : bool, optional
        Whether or not the input value 0 is a special "padding"
        value that should be masked out.
        This is useful when using recurrent layers
        which may take variable length input.
        If this is `True`, then all subsequent layers
        in the model need to support masking or an exception will be raised.
        If mask_zero is set to True, as a consequence, index 0 cannot be
        used in the vocabulary (input_dim should equal size of
        vocabulary + 1), by default False
    input_length : int, optional
        This argument is required if you are going to connect
        `Flatten` then `Dense` layers upstream
        (without it, the shape of the dense outputs cannot be computed),
        by default None
    sequence_combiner : Optional[CombinerType], optional
        A string specifying how to combine embedding results for each
        entry ("mean", "sqrtn" and "sum" are supported) or a layer.
        Default is None (no combiner used), by default None
    trainable : bool, optional
        Whether the layer's variables should be trainable, by default True
    name : str, optional
        String name of the layer, by default None
    dtype : optional
        The dtype of the layer's computations and weights. Can also be a
        `tf.keras.mixed_precision.Policy`, which allows the computation and weight
        dtype to differ. Default of `None` means to use
        `tf.keras.mixed_precision.global_policy()`, which is a float32 policy
        unless set to different value., by default None
    dynamic : bool, optional
        Set this to `True` if your layer should only be run eagerly, and
        should not be used to generate a static computation graph.
        This would be the case for a Tree-RNN or a recursive network,
        for example, or generally for any layer that manipulates tensors
        using Python control flow. If `False`, we assume that the layer can
        safely be used to generate a static computation graph., by default False
    embeddings_l2_batch_regularization : Optional[Union[float, Dict[str, float]]], optional
        Factor for L2 regularization of the embeddings vectors (from the current batch only)
        by default 0.0, by default 0.0
    post : Optional[tf.keras.layers.Layer], optional
        _description_, by default None
    **kwargs: Forwarded Encoder parameters
    """

    def __init__(
        self,
        schema: Union[ColumnSchema, Schema],
        dim: int,
        embeddings_initializer: Optional[Union[str, tf.keras.layers.Layer]] = "uniform",
        embeddings_regularizer: Optional[Union[str, tf.keras.layers.Layer]] = None,
        activity_regularizer: Optional[Union[str, tf.keras.layers.Layer]] = None,
        embeddings_constraint: Optional[Union[str, tf.keras.layers.Layer]] = None,
        mask_zero: bool = False,
        input_length: int = None,
        sequence_combiner: Optional[CombinerType] = None,
        trainable: bool = True,
        name: str = None,
        dtype=None,
        dynamic: bool = False,
        embeddings_l2_batch_regularization: Optional[Union[float, Dict[str, float]]] = 0.0,
        post: Optional[tf.keras.layers.Layer] = None,
        **kwargs,
    ):
        if isinstance(schema, ColumnSchema):
            col = schema
        else:
            col = schema.first
        col_name = col.name

        table = EmbeddingTable(
            dim,
            col,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            input_length=input_length,
            sequence_combiner=sequence_combiner,
            trainable=trainable,
            name=name,
            dtype=dtype,
            dynamic=dynamic,
            l2_batch_regularization_factor=embeddings_l2_batch_regularization,
        )

        super().__init__(table, tf.keras.layers.Lambda(lambda x: x[col_name]), post=post, **kwargs)

    def to_dataset(self, gpu=None) -> merlin.io.Dataset:
        return self.blocks[0].to_dataset(gpu=gpu)
