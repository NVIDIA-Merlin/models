from typing import Optional, Union

import numpy as np
import tensorflow as tf
from packaging import version

import merlin.io
from merlin.models.tf.core import combinators
from merlin.models.tf.inputs.base import InputBlockV2
from merlin.models.tf.inputs.embedding import CombinerType, EmbeddingTable
from merlin.models.tf.utils import tf_utils
from merlin.schema import ColumnSchema, Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class Encoder(tf.keras.Model):
    """Block that can be used for prediction & evaluation but not for training

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

    """

    def __init__(
        self,
        inputs: Union[Schema, tf.keras.layers.Layer],
        *blocks: tf.keras.layers.Layer,
        pre: Optional[tf.keras.layers.Layer] = None,
        post: Optional[tf.keras.layers.Layer] = None,
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

    def encode(
        self,
        dataset: merlin.io.Dataset,
        batch_size: int,
        id_col: Optional[Union[str, ColumnSchema, Schema, Tags]] = None,
        **kwargs,
    ) -> merlin.io.Dataset:
        output_schema = None
        if id_col:
            if isinstance(id_col, Schema):
                output_schema = id_col
            elif isinstance(id_col, ColumnSchema):
                output_schema = Schema([id_col])
            elif isinstance(id_col, str):
                output_schema = Schema([self.schema[id_col]])
            elif isinstance(id_col, Tags):
                output_schema = self.schema.select_by_tag(id_col)
            else:
                raise ValueError(f"Invalid id_col: {id_col}")

        return self.batch_predict(
            dataset,
            batch_size=batch_size,
            output_schema=output_schema,
            output_concat_func=np.concatenate,
            **kwargs,
        )

    def batch_predict(
        self,
        dataset: merlin.io.Dataset,
        batch_size: int,
        output_schema: Optional[Schema] = None,
        **kwargs,
    ) -> merlin.io.Dataset:
        """Batched prediction using Dask.

        Parameters
        ----------
        dataset: merlin.io.Dataset
            Dataset to predict on.
        batch_size: int
            Batch size to use for prediction.

        Returns
        -------
        merlin.io.Dataset

        """
        if hasattr(dataset, "schema"):
            if not set(self.schema.column_names).issubset(set(dataset.schema.column_names)):
                raise ValueError(
                    f"Model schema {self.schema.column_names} does not match dataset schema"
                    + f" {dataset.schema.column_names}"
                )

        # Check if merlin-dataset is passed
        if hasattr(dataset, "to_ddf"):
            dataset = dataset.to_ddf()

        from merlin.models.tf.utils.batch_utils import TFModelEncode

        model_encode = TFModelEncode(self, batch_size=batch_size, **kwargs)
        encode_kwargs = {}
        if output_schema:
            encode_kwargs["filter_input_columns"] = output_schema.column_names
        predictions = dataset.map_partitions(model_encode, **encode_kwargs)

        return merlin.io.Dataset(predictions)

    def call(self, inputs, **kwargs):
        if "features" not in kwargs:
            kwargs["features"] = inputs

        return combinators.call_sequentially(list(self.to_call), inputs=inputs, **kwargs)

    def build(self, input_shape):
        combinators.build_sequentially(self, list(self.to_call), input_shape=input_shape)
        if not hasattr(self.build, "_is_default"):
            self._build_input_shape = input_shape

    def compute_output_shape(self, input_shape):
        return combinators.compute_output_shape_sequentially(list(self.to_call), input_shape)

    def __call__(self, inputs, **kwargs):
        # We remove features here since we don't expect them at inference time
        # Inside the `call` method, we will add them back by assuming inputs=features
        if "features" in kwargs:
            kwargs.pop("features")

        return super().__call__(inputs, **kwargs)

    def train_step(self, data):
        """Train step"""
        raise NotImplementedError(
            "This block is not meant to be trained by itself. ",
            "It can only be trained as part of a model.",
        )

    def fit(self, *args, **kwargs):
        """Fit model"""
        raise NotImplementedError(
            "This block is not meant to be trained by itself. ",
            "It can only be trained as part of a model.",
        )

    def _set_save_spec(self, inputs, args=None, kwargs=None):
        super()._set_save_spec(inputs, args, kwargs)

        # We need to overwrite this in order to fix a Keras-bug in TF<2.9
        if version.parse(tf.__version__) < version.parse("2.9.0"):
            # Keras will interpret kwargs like `features` & `targets` as
            # required args, which is wrong. This is a workaround.
            _arg_spec = self._saved_model_arg_spec
            self._saved_model_arg_spec = ([_arg_spec[0][0]], _arg_spec[1])

    @property
    def to_call(self):
        if self.pre:
            yield self.pre

        for block in self.blocks:
            yield block

        if self.post:
            yield self.post

    @property
    def has_schema(self) -> bool:
        return True

    @property
    def schema(self) -> Schema:
        return self._schema

    @classmethod
    def from_config(cls, config, custom_objects=None):
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
        config = tf_utils.maybe_serialize_keras_objects(self, {}, ["pre", "post"])
        for i, layer in enumerate(self.blocks):
            config[i] = tf.keras.utils.serialize_keras_object(layer)

        return config


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class EmbeddingEncoder(Encoder):
    def __init__(
        self,
        schema: Union[ColumnSchema, Schema],
        dim: int,
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
    ):
        if isinstance(schema, ColumnSchema):
            col = schema
        else:
            col = schema.first
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
        )

        super().__init__(table, tf.keras.layers.Lambda(lambda x: x[col.name]))
