from typing import Optional, Sequence, Union

import tensorflow as tf
from keras.utils.generic_utils import to_snake_case
from tensorflow.keras.layers import Layer

from merlin.models.tf.core.base import name_fn
from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.core.transformations import LogitsTemperatureScaler
from merlin.models.tf.utils import tf_utils


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class PredictionBlock(Layer):
    def __init__(
        self,
        prediction: Layer,
        default_loss: Union[str, tf.keras.losses.Loss],
        default_metrics: Sequence[tf.keras.metrics.Metric],
        name: Optional[str] = None,
        target: Optional[str] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature=1.0,
        **kwargs,
    ):
        logits_scaler = kwargs.pop("logits_scaler", None)
        self.target = target
        base_name = to_snake_case(self.__class__.__name__)
        self.full_name = name_fn(self.target, base_name) if self.target else base_name

        super().__init__(name=name or self.full_name, **kwargs)
        self.prediction = prediction
        self.default_loss = default_loss
        self.default_metrics = default_metrics
        self.pre = pre
        self.post = post
        if logits_scaler is not None:
            self.logits_scaler = logits_scaler
            self.logits_temperature = logits_scaler.temperature
        else:
            self.logits_temperature = logits_temperature
            if logits_temperature != 1.0:
                self.logits_scaler = LogitsTemperatureScaler(logits_temperature)

    def build(self, input_shape=None):
        """Builds the PredictionBlock.

        Parameters
        ----------
        input_shape : tf.TensorShape, optional
            The input shape, by default None
        """
        if self.pre is not None:
            self.pre.build(input_shape)
            input_shape = self.pre.compute_output_shape(input_shape)

        input_shape = self.prediction.compute_output_shape(input_shape)

        if self.post is not None:
            self.post.build(input_shape)

        self.built = True

    def call(self, inputs, **kwargs):
        return tf_utils.call_layer(self.prediction, inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.pre is not None:
            output_shape = self.pre.compute_output_shape(output_shape)

        output_shape = self.prediction.compute_output_shape(output_shape)

        if self.post is not None:
            output_shape = self.post.compute_output_shape(output_shape)

        return output_shape

    def __call__(self, inputs, *args, **kwargs):
        # call pre
        if self.pre:
            inputs = tf_utils.call_layer(self.pre, inputs, **kwargs)

        # super call
        outputs = super(PredictionBlock, self).__call__(inputs, *args, **kwargs)

        if self.post:
            outputs = tf_utils.call_layer(self.post, inputs, **kwargs)

        if getattr(self, "logits_scaler", None):
            outputs = self.logits_scaler(outputs)

        if kwargs.get("training", False) or kwargs.get("testing", False):
            targets = kwargs.get("targets", {})
            if isinstance(targets, dict) and self.target:
                targets = targets.get(self.target, targets)

            return Prediction(outputs, targets)

        return outputs

    def get_config(self):
        config = super(PredictionBlock, self).get_config()
        config.update(
            {
                "target": self.target,
            }
        )

        objects = [
            "default_metrics",
            "prediction",
            "pre",
            "post",
            "logits_scaler",
        ]

        if isinstance(self.default_loss, str):
            config["default_loss"] = self.default_loss
        else:
            objects.append("default_loss")

        config = tf_utils.maybe_serialize_keras_objects(self, config, objects)

        return config

    @classmethod
    def from_config(cls, config):
        config = tf_utils.maybe_deserialize_keras_objects(
            config,
            {
                "default_metrics": tf.keras.metrics.deserialize,
                "default_loss": tf.keras.losses.deserialize,
                "prediction": tf.keras.layers.deserialize,
                "pre": tf.keras.layers.deserialize,
                "post": tf.keras.layers.deserialize,
                "logits_scaler": tf.keras.layers.deserialize,
            },
        )

        return super().from_config(config)
