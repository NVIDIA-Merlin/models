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
import abc
import inspect
from enum import Enum
from typing import Optional, Union

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.ops import nn

from merlin.models.config.schema import requires_schema
from merlin.models.tf.blocks.core.base import Block
from merlin.models.tf.blocks.core.tabular import TabularAggregation
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils import tf_utils
from merlin.models.utils.schema_utils import schema_to_tensorflow_metadata_json
from merlin.schema import Schema, Tags

# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter, unexpected-keyword-arg


@TabularAggregation.registry.register("concat")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ConcatFeatures(TabularAggregation):
    """Concatenates tensors along one dimension.

    Parameters
    ----------
    axis : int
        The axis to concatenate along.
    output_dtype : str
        The dtype of the output tensor.
    """

    def __init__(self, axis=-1, output_dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.output_dtype = output_dtype

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        self._expand_non_sequential_features(inputs)
        self._check_concat_shapes(inputs)

        tensors = []
        for name in sorted(inputs.keys()):
            tensors.append(tf.cast(inputs[name], self.output_dtype))

        output = tf.concat(tensors, axis=-1)

        return output

    def compute_output_shape(self, input_shapes):
        agg_dim = sum([i[-1] for i in input_shapes.values()])
        if isinstance(agg_dim, tf.TensorShape):
            raise ValueError(f"Not possible to aggregate, received: {input_shapes}.")
        output_size = self._get_agg_output_size(input_shapes, agg_dim)
        return output_size

    def get_config(self):
        config = super().get_config()
        config["axis"] = self.axis
        config["output_dtype"] = self.output_dtype

        return config


@TabularAggregation.registry.register("stack")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class StackFeatures(TabularAggregation):
    """Stacks tensors along one dimension.

    Parameters
    ----------
    axis : int
        The axis to stack along.
    output_dtype : str
        The dtype of the output tensor.
    """

    def __init__(self, axis=-1, output_dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.output_dtype = output_dtype

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        self._expand_non_sequential_features(inputs)
        self._check_concat_shapes(inputs)

        tensors = []
        for name in sorted(inputs.keys()):
            tensors.append(tf.cast(inputs[name], self.output_dtype))

        return tf.stack(tensors, axis=self.axis)

    def compute_output_shape(self, input_shapes):
        agg_dim = list(input_shapes.values())[0][-1]
        output_size = self._get_agg_output_size(input_shapes, agg_dim, axis=self.axis)

        if len(output_size) == 2:
            output_size = list(output_size)
            if self.axis == -1:
                output_size = [*output_size, len(input_shapes)]
            else:
                output_size.insert(self.axis, len(input_shapes))

        return output_size

    def get_config(self):
        config = super().get_config()
        config["axis"] = self.axis
        config["output_dtype"] = self.output_dtype

        return config


class ElementwiseFeatureAggregation(TabularAggregation):
    def _check_input_shapes_equal(self, inputs):
        all_input_shapes_equal = len(set([tuple(x.shape) for x in inputs.values()])) == 1
        if not all_input_shapes_equal:
            raise ValueError(
                "The shapes of all input features are not equal, which is required for element-wise"
                " aggregation: {}".format({k: v.shape for k, v in inputs.items()})
            )


@TabularAggregation.registry.register("sum")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class Sum(TabularAggregation):
    """Sum tensors along the first dimension."""

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        summed = tf.reduce_sum(list(inputs.values()), axis=0)

        return summed

    def compute_output_shape(self, input_shape):
        batch_size = tf_utils.calculate_batch_size_from_input_shapes(input_shape)
        last_dim = list(input_shape.values())[0][-1]

        return batch_size, last_dim


@TabularAggregation.registry.register("sum-residual")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SumResidual(Sum):
    def __init__(self, activation="relu", shortcut_name="shortcut", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation) if activation else None
        self.shortcut_name = shortcut_name

    def call(self, inputs: TabularData, **kwargs) -> Union[tf.Tensor, TabularData]:
        shortcut = inputs.pop(self.shortcut_name)
        outputs = {}
        for key, val in inputs.items():
            outputs[key] = tf.reduce_sum([inputs[key], shortcut], axis=0)
            if self.activation:
                outputs[key] = self.activation(outputs[key])

        if len(outputs) == 1:
            return list(outputs.values())[0]

        return outputs

    def compute_output_shape(self, input_shape):
        batch_size = tf_utils.calculate_batch_size_from_input_shapes(input_shape)
        last_dim = list(input_shape.values())[0][-1]

        return batch_size, last_dim

    def get_config(self):
        config = super().get_config()
        config["shortcut_name"] = self.shortcut_name
        config["activation"] = tf.keras.activations.serialize(self.activation)

        return config


@TabularAggregation.registry.register("add-left")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class AddLeft(ElementwiseFeatureAggregation):
    """Concat tensors & add it to left_name"""

    def __init__(self, left_name="bias", **kwargs):
        super().__init__(**kwargs)
        self.left_name = left_name
        self.concat = ConcatFeatures()
        self.wide_logit = tf.keras.layers.Dense(1)

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        left = inputs.pop(self.left_name)
        if not left.shape[-1] == 1:
            left = self.wide_logit(left)

        return left + self.concat(inputs)

    def compute_output_shape(self, input_shape):
        batch_size = tf_utils.calculate_batch_size_from_input_shapes(input_shape)
        last_dim = list(input_shape.values())[0][-1]

        return batch_size, last_dim


@TabularAggregation.registry.register("element-wise-sum")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ElementwiseSum(ElementwiseFeatureAggregation):
    """Element-wise sum of all features."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stack = StackFeatures(axis=0)

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        self._expand_non_sequential_features(inputs)
        self._check_input_shapes_equal(inputs)

        return tf.reduce_sum(self.stack(inputs), axis=0)

    def compute_output_shape(self, input_shape):
        batch_size = tf_utils.calculate_batch_size_from_input_shapes(input_shape)
        last_dim = list(input_shape.values())[0][-1]

        return batch_size, last_dim


@TabularAggregation.registry.register("element-wise-sum-item-multi")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
@requires_schema
class ElementwiseSumItemMulti(ElementwiseFeatureAggregation):
    """Element-wise sum of features."""

    def __init__(self, schema=None, **kwargs):
        super().__init__(**kwargs)
        self.stack = StackFeatures(axis=0)
        if schema:
            self.set_schema(schema)
        self.item_id_col_name = None

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        schema: Schema = self.schema  # type: ignore
        item_id_inputs = self.get_item_ids_from_inputs(inputs)
        self._expand_non_sequential_features(inputs)
        self._check_input_shapes_equal(inputs)

        item_id_column = schema.select_by_tag(Tags.ITEM_ID).first.name
        other_inputs = {k: v for k, v in inputs.items() if k != item_id_column}
        # Sum other inputs when there are multiple features.
        if len(other_inputs) > 1:
            other_inputs = tf.reduce_sum(self.stack(other_inputs), axis=0)
        else:
            other_inputs = list(other_inputs.values())[0]
        result = item_id_inputs * other_inputs
        return result

    def compute_output_shape(self, input_shape):
        batch_size = tf_utils.calculate_batch_size_from_input_shapes(input_shape)
        last_dim = list(input_shape.values())[0][-1]

        return batch_size, last_dim

    def get_config(self):
        config = super().get_config()
        if self.schema:
            config["schema"] = schema_to_tensorflow_metadata_json(self.schema)

        return config


class TupleAggregation(TabularAggregation, abc.ABC):
    """Aggregation between two features."""

    def call(self, left: tf.Tensor, right: tf.Tensor, **kwargs) -> tf.Tensor:  # type: ignore
        raise NotImplementedError()

    def super(self):
        return super()


# This is done this way to avoid mypy crashing.
def _tuple_aggregation_call(self, inputs: TabularData, *args, **kwargs):  # type: ignore
    if isinstance(inputs, tf.Tensor):
        left = inputs
        right = args[0]
    else:
        if not len(inputs) == 2:
            raise ValueError(f"Expected 2 inputs, got {len(inputs)}")
        left, right = tuple(inputs.values())
    outputs = self.super().__call__(left, right, **kwargs)  # type: ignore

    return outputs


TupleAggregation.__call__ = _tuple_aggregation_call  # type: ignore


@TabularAggregation.registry.register_with_multiple_names("cosine", "cosine-similarity")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class CosineSimilarity(TupleAggregation):
    """Cosine similarity aggregation"""

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, left: tf.Tensor, right: tf.Tensor, **kwargs) -> tf.Tensor:  # type: ignore
        left = nn.l2_normalize(left, axis=1)
        right = nn.l2_normalize(right, axis=1)
        output = backend.batch_dot(left, right, 1)
        return output


@TabularAggregation.registry.register("elementwise-multiply")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ElementWiseMultiply(TupleAggregation):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, left: tf.Tensor, right: tf.Tensor, **kwargs) -> tf.Tensor:  # type: ignore
        out = tf.keras.layers.Multiply()([left, right])

        return out


def masked_mean(
    input_tensor: tf.Tensor, axis: Optional[int] = None, mask: tf.Tensor = None
) -> tf.Tensor:
    """Computes the mean of the specified axis considering only
    masked values

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor
    axis : int, optional
        axis to compute the mean over, by default None
    mask : tf.Tensor
        Mask tensor with the same shape as input_tensor.
        Only values where mask=True will be considered for average, by default None

    Returns
    -------
    tf.Tensor
        A tensor with the values averaged of the specified axis considering only masked values

    Raises
    ------
    ValueError
        If mask is not provided
    """
    output_tensor = input_tensor
    if mask is not None:
        mask_float = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        output_tensor = tf.divide(
            tf.reduce_sum(tf.multiply(input_tensor, mask_float), axis=axis),
            tf.reduce_sum(mask_float, axis=axis),
        )
    else:
        raise ValueError("The mask is required for masked mean")

    return output_tensor


class SequenceAggregation(Enum):
    MEAN = tf.reduce_mean
    SUM = tf.reduce_sum
    MAX = tf.reduce_max
    MIN = tf.reduce_min
    MASKED_MEAN = masked_mean

    def __str__(self):
        return self.value

    def __eq__(self, o: object) -> bool:
        return str(o) == str(self)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequenceAggregator(Block):
    """Computes the aggregation of elements across dimensions of a 3-D tensor.
    Args:
        combiner:
            tensorflow method to use for aggregation
            Defaults to SequenceAggregation.MEAN
        axis: int
            The dimensions to reduce.
            Defaults to 1
    """

    def __init__(self, combiner=SequenceAggregation.MEAN, axis: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.combiner = combiner

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        assert len(inputs.shape) == 3, "inputs should be a 3-D tensor"
        kwargs = {}
        if (
            "mask" in inspect.signature(self.combiner).parameters
            and "valid_items_mask" in self.context.named_variables
        ):
            kwargs["mask"] = self.context["valid_items_mask"]

        return self.combiner(inputs, axis=self.axis, **kwargs)

    def compute_output_shape(self, input_shape):
        batch_size, _, last_dim = input_shape
        return batch_size, last_dim

    def get_config(self):
        config = super().get_config()
        config = tf_utils.maybe_serialize_keras_objects(
            self, config, {"combiner": tf.keras.layers.serialize}
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = tf_utils.maybe_deserialize_keras_objects(
            config, ["combiner"], tf.keras.layers.deserialize
        )
        return super().from_config(config)
