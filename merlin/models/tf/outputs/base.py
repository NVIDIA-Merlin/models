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
import sys
import types as python_types
import warnings
from typing import Callable, Optional, Sequence, Union

import tensorflow as tf
from keras.utils import generic_utils
from keras.utils.generic_utils import to_snake_case
from tensorflow.keras.layers import Layer

from merlin.models.tf.core.base import name_fn
from merlin.models.tf.core.combinators import ParallelBlock
from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.transforms.bias import LogitsTemperatureScaler
from merlin.models.tf.utils import tf_utils

MetricsFn = Callable[[], Sequence[tf.keras.metrics.Metric]]

ModelOutputType = Union["ModelOutput", ParallelBlock]


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ModelOutput(Layer):
    """Base-class for prediction blocks.

    Parameters
    ----------
    to_call : Layer
        The layer to call in the forward-pass of the model
    default_loss: Union[str, tf.keras.losses.Loss]
        Default loss to set if the user does not specify one
    get_default_metrics: Callable
        A function returning the list of default metrics to set
        if the user does not specify any
    name: Optional[Text], optional
        Task name, by default None
    target: Optional[str], optional
        Label name, by default None
    pre: Optional[Block], optional
        Optional block to transform predictions before applying the prediction layer,
        by default None
    post: Optional[Block], optional
        Optional block to transform predictions after applying the prediction layer,
        by default None
    logits_temperature: float, optional
        Parameter used to reduce model overconfidence, so that logits / T.
        by default 1.
    """

    def __init__(
        self,
        to_call: Layer,
        default_loss: Union[str, tf.keras.losses.Loss],
        default_metrics_fn: MetricsFn,
        name: Optional[str] = None,
        target: Optional[str] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        **kwargs,
    ):
        logits_scaler = kwargs.pop("logits_scaler", None)
        self.target = target
        self.full_name = self.get_task_name(self.target)

        super().__init__(name=name or self.full_name, **kwargs)
        self.to_call = to_call
        self.default_loss = default_loss
        self.default_metrics_fn = default_metrics_fn
        self.pre = pre
        self.post = post
        if logits_scaler is not None:
            self.logits_scaler = logits_scaler
            self.logits_temperature = logits_scaler.temperature
        else:
            self.logits_temperature = logits_temperature
            if logits_temperature != 1.0:
                self.logits_scaler = LogitsTemperatureScaler(logits_temperature)

    @property
    def task_name(self) -> str:
        return self.full_name

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

        self.to_call.build(input_shape)
        input_shape = self.to_call.compute_output_shape(input_shape)

        if self.post is not None:
            self.post.build(input_shape)

        self.built = True

    def call(self, inputs, training=False, testing=False, **kwargs):
        return tf_utils.call_layer(
            self.to_call, inputs, training=training, testing=testing, **kwargs
        )

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.pre is not None:
            output_shape = self.pre.compute_output_shape(output_shape)

        output_shape = self.to_call.compute_output_shape(output_shape)

        if self.post is not None:
            output_shape = self.post.compute_output_shape(output_shape)

        return output_shape

    def __call__(self, inputs, *args, **kwargs):
        training = kwargs.get("training", False)
        testing = kwargs.get("testing", False)

        # call pre
        if self.pre:
            inputs = tf_utils.call_layer(self.pre, inputs, **kwargs)

        # super call
        outputs = super(ModelOutput, self).__call__(inputs, *args, **kwargs)

        if self.post:
            outputs = tf_utils.call_layer(self.post, outputs, target_name=self.target, **kwargs)

        if getattr(self, "logits_scaler", None):
            if isinstance(outputs, tf.Tensor):
                targets = kwargs.pop("targets", None)
                if isinstance(targets, dict) and self.target in targets:
                    targets = targets[self.target]
                if training or testing:
                    outputs = Prediction(outputs, targets)
            outputs = tf_utils.call_layer(self.logits_scaler, outputs, **kwargs)

        return outputs

    def create_default_metrics(self):
        metrics = self.get_default_metrics()
        for metric in metrics:
            metric._name = self.full_name + "/" + to_snake_case(metric.name)
        return metrics

    def _serialize_function_to_config(self, inputs):
        """function to serialize a callable function,

        Note: This code is adapted from Keras source code of
        the [Lambda layer]
        (https://github.com/keras-team/keras/blob/master/keras/layers/core/lambda_layer.py#L300)

        """
        if isinstance(inputs, python_types.LambdaType):
            output = generic_utils.func_dump(inputs)
            output_type = "lambda"
            module = inputs.__module__
        elif callable(inputs):
            output = inputs.__name__
            output_type = "function"
            module = inputs.__module__
        else:
            raise ValueError("Invalid input for serialization, type: %s " % type(inputs))

        return output, output_type, module

    @classmethod
    def _parse_function_from_config(
        cls, config, func_attr_name, module_attr_name, func_type_attr_name
    ):
        """ "function to de-serialize a callable function,

        Note: This code is adapted from Keras source code of
        the [Lambda layer]
        (https://github.com/keras-team/keras/blob/master/keras/layers/core/lambda_layer.py#L350)

        """
        globs = globals().copy()
        module = config.pop(module_attr_name, None)
        if module in sys.modules:
            globs.update(sys.modules[module].__dict__)
        elif module is not None:
            # Note: we don't know the name of the function if it's a lambda.
            warnings.warn(
                "{} is not loaded, but a Lambda layer uses it. "
                "It may cause errors.".format(module),
                UserWarning,
                stacklevel=2,
            )
        function_type = config.pop(func_type_attr_name)
        if function_type == "function":
            function = generic_utils.deserialize_keras_object(
                config[func_attr_name], printable_module_name="default metrics function"
            )
        elif function_type == "lambda":
            # Unsafe deserialization from bytecode
            function = generic_utils.func_load(config[func_attr_name], globs=globs)
        else:
            supported_types = ["function", "lambda"]
            raise TypeError(
                f"Unsupported value for `function_type` argument. Received: "
                f"function_type={function_type}. Expected one of {supported_types}"
            )
        return function

    def get_config(self):
        config = super(ModelOutput, self).get_config()
        function_config = self._serialize_function_to_config(self.default_metrics_fn)
        config.update(
            {
                "default_metrics_fn": function_config[0],
                "function_type": function_config[1],
                "module": function_config[2],
                "target": self.target,
            }
        )

        objects = [
            "to_call",
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
    def get_task_name(cls, target_name):
        base_name = to_snake_case(cls.__name__)
        return name_fn(target_name, base_name) if target_name else base_name

    @classmethod
    def from_config(cls, config):
        config["default_metrics_fn"] = cls._parse_function_from_config(
            config, "default_metrics_fn", "module", "function_type"
        )

        config = tf_utils.maybe_deserialize_keras_objects(
            config,
            {
                "default_loss": tf.keras.losses.deserialize,
                "to_call": tf.keras.layers.deserialize,
                "pre": tf.keras.layers.deserialize,
                "post": tf.keras.layers.deserialize,
                "logits_scaler": tf.keras.layers.deserialize,
            },
        )

        return super().from_config(config)


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class DotProduct(Layer):
    """Dot-product between queries & items.
    Parameters:
    -----------
    query_name : str, optional
        Identify query tower for query/user embeddings, by default 'query'
    item_name : str, optional
        Identify item tower for item embeddings, by default 'item'
    """

    def __init__(self, query_name: str = "query", item_name: str = "candidate", **kwargs):
        super().__init__(**kwargs)
        self.query_name = query_name
        self.item_name = item_name

    def call(self, inputs, **kwargs):
        return tf.reduce_sum(
            tf.multiply(inputs[self.query_name], inputs[self.item_name]), keepdims=True, axis=-1
        )

    def compute_output_shape(self, input_shape):
        batch_size = tf_utils.calculate_batch_size_from_input_shapes(input_shape)

        return batch_size, 1

    def get_config(self):
        return {
            **super(DotProduct, self).get_config(),
            "query_name": self.query_name,
            "item_name": self.item_name,
        }
