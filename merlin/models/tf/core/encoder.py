from typing import Optional, Union

import tensorflow as tf
from packaging import version

from merlin.models.tf.core import combinators
from merlin.models.tf.inputs.base import InputBlockV2
from merlin.models.tf.utils import tf_utils
from merlin.schema import Schema


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class EncoderBlock(tf.keras.Model):
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

    @property
    def first(self):
        return self.blocks[0]

    @property
    def last(self):
        return self.blocks[-1]

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

        return cls(*layers, pre=pre, post=post)

    def get_config(self):
        config = tf_utils.maybe_serialize_keras_objects(self, {}, ["pre", "post"])
        for i, layer in enumerate(self.blocks):
            config[i] = tf.keras.utils.serialize_keras_object(layer)

        return config
