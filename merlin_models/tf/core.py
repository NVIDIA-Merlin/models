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
import copy
import sys
from collections import defaultdict
from functools import reduce
from typing import Dict, List, Optional, Sequence, Text, Tuple, Type, Union, overload

import six
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import variables as tf_variables

from merlin_models.config.schema import SchemaMixin
from merlin_standard_lib import Registry, RegistryMixin, Schema, Tag
from merlin_standard_lib.utils.doc_utils import docstring_parameter
from merlin_standard_lib.utils.misc_utils import filter_kwargs

from .typing import TabularData, TensorOrTabularData

# from ..features.base import InputBlock
from .utils.tf_utils import (
    ContextMixin,
    LossMixin,
    MetricsMixin,
    ModelContext,
    ModelLikeBlock,
    calculate_batch_size_from_input_shapes,
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
)


class Block(SchemaMixin, ContextMixin, Layer):
    def as_tabular(self, name=None) -> "Block":
        if not name:
            name = self.name

        return SequentialBlock([self, AsTabular(name)], copy_layers=False)

    @classmethod
    def from_layer(cls, layer: tf.keras.layers.Layer) -> "Block":
        layer.__class__ = cls

        return layer  # type: ignore

    def repeat(self, num: int = 1) -> "SequentialBlock":
        repeated = []
        for _ in range(num):
            repeated.append(self.copy())

        return SequentialBlock(repeated)

    def from_inputs(
        self,
        schema: Schema,
        input_block: Optional["InputBlock"] = None,
        post: Optional["TabularTransformationType"] = None,
        aggregation: Optional["TabularAggregationType"] = None,
        **kwargs,
    ) -> "SequentialBlock":
        from merlin_models.tf import TabularFeatures

        input_block = input_block or TabularFeatures
        inputs = input_block.from_schema(schema, post=post, aggregation=aggregation, **kwargs)

        return SequentialBlock([inputs, self])

    def prepare(
        self,
        block=None,
        post: Optional["TabularTransformationType"] = None,
        aggregation: Optional["TabularAggregationType"] = None,
    ) -> "SequentialBlock":
        block = TabularBlock(post=post, aggregation=aggregation) or block

        return SequentialBlock([block, self])

    def repeat_in_parallel(
        self,
        num: int = 1,
        prefix=None,
        names: Optional[List[str]] = None,
        post: Optional["TabularTransformationType"] = None,
        aggregation: Optional["TabularAggregationType"] = None,
        copies=True,
        residual=False,
        **kwargs,
    ) -> "ParallelBlock":
        repeated = {}
        iterator = names if names else range(num)
        if not names and prefix:
            iterator = [f"{prefix}{num}" for num in iterator]
        for name in iterator:
            repeated[str(name)] = self.copy() if copies else self

        if residual:
            repeated["shortcut"] = NoOp()

        return ParallelBlock(repeated, post=post, aggregation=aggregation, **kwargs)

    def connect(
        self, *block: tf.keras.layers.Layer, block_name: Optional[str] = None
    ) -> "SequentialBlock":
        if isinstance(self, SequentialBlock):
            if isinstance(block, (list, tuple)):
                self.layers.extend(block)
            else:
                self.layers.append(block)
            if block_name:
                self.block_name = block_name

            output = self
        elif len(block) == 1 and isinstance(block[0], SequentialBlock):
            block: SequentialBlock = block[0]  # type: ignore
            if isinstance(self, SequentialBlock):
                block.layers = [*self.layers, *block.layers]
            else:
                block.layers = [self, *block.layers]
            if block_name:
                self.block_name = block_name

            if not block.schema:
                block.schema = self.schema

            output = block
        else:
            output = SequentialBlock([self, *block], copy_layers=False, block_name=block_name)

        if isinstance(block[-1], ModelLikeBlock):
            return Model(output)

        return output

    def connect_with_residual(
        self,
        block: tf.keras.layers.Layer,
        activation=None,
    ) -> "SequentialBlock":
        residual_block = ResidualBlock(block, activation=activation)

        if isinstance(self, SequentialBlock):
            self.layers.append(residual_block)

            return self

        return SequentialBlock([self, residual_block], copy_layers=False)

    def connect_with_shortcut(
        self,
        block: tf.keras.layers.Layer,
        shortcut_filter: Optional["Filter"] = None,
        post: Optional["TabularTransformationType"] = None,
        aggregation: Optional["TabularAggregationType"] = None,
        block_outputs_name: Optional[str] = None,
    ) -> "SequentialBlock":
        residual_block = WithShortcut(
            block,
            shortcut_filter=shortcut_filter,
            post=post,
            aggregation=aggregation,
            block_outputs_name=block_outputs_name,
        )

        if isinstance(self, SequentialBlock):
            self.layers.append(residual_block)

            return self

        return SequentialBlock([self, residual_block], copy_layers=False)

    def connect_debug_block(self, append=True):
        if not append:
            return SequentialBlock([Debug(), self])

        return self.apply(Debug())

    def connect_branch(
        self,
        *branches: Union["Block", "PredictionTask"],
        add_rest=False,
        post: Optional["TabularTransformationsType"] = None,
        aggregation: Optional["TabularAggregationType"] = None,
        **kwargs,
    ) -> "SequentialBlock":
        branches = list(branches)

        all_features = []
        for branch in branches:
            if getattr(branch, "set_schema", None):
                branch.set_schema(self.schema)
            if isinstance(branch, SequentialBlock):
                filter_features = branch.filter_features
                if filter_features:
                    all_features.extend(filter_features)

        rest_features = self.schema.remove_by_name(list(set([str(f) for f in all_features])))
        rest_block = None
        if add_rest:
            rest_block = SequentialBlock([Filter(rest_features)])

            if rest_block:
                branches.append(rest_block)

        if all(isinstance(branch, ModelLikeBlock) for branch in branches):
            parallel = ParallelPredictionBlock(
                *branches, post=post, aggregation=aggregation, **kwargs
            )

            return Model(SequentialBlock([self, parallel]))

        return SequentialBlock(
            [self, ParallelBlock(*branches, post=post, aggregation=aggregation, **kwargs)]
        )

    def _add_embedding_table(
        self,
        name=None,
        shape=None,
        dtype=None,
        initializer=None,
        regularizer=None,
        table_name=None,
        trainable=None,
        constraint=None,
        use_resource=None,
        synchronization=tf_variables.VariableSynchronization.AUTO,
        aggregation=tf_variables.VariableAggregation.NONE,
        **kwargs,
    ):
        table_name = table_name or f"{name}/embedding"
        weight = super().add_weight(
            table_name,
            shape,
            dtype,
            initializer,
            regularizer,
            trainable,
            constraint,
            use_resource,
            synchronization,
            aggregation,
            **kwargs,
        )

        self.context.register_variable(table_name, weight)

        return weight

    def select_by_name(self, name: str) -> Optional["Block"]:
        if name == self.name:
            return self

        return None

    def copy(self):
        return self.from_config(self.get_config())

    @classmethod
    def parse_block(cls, input: Union["Block", tf.keras.layers.Layer]) -> "Block":
        if isinstance(input, Block):
            return input

        return cls.from_layer(input)

    def __rrshift__(self, other):
        return right_shift_layer(self, other)


def inputs(
    schema: Schema,
    *block: Block,
    post: Optional["TabularTransformationType"] = None,
    aggregation: Optional["TabularAggregationType"] = None,
    seq: bool = False,
    add_to_context: List[Union[str, Tag]] = None,
    **kwargs,
) -> "Block":
    if seq:
        from merlin_models.tf import TabularSequenceFeatures

        inp_block = TabularSequenceFeatures.from_schema(
            schema, post=post, aggregation=aggregation, **kwargs
        )
    else:
        from merlin_models.tf.block.inputs import TabularFeatures

        inp_block = TabularFeatures(
            schema, aggregation=aggregation, add_to_context=add_to_context, **kwargs
        )

    if not block:
        return inp_block

    return SequentialBlock([inp_block, *block])


def prediction_tasks(
    schema: Schema,
    task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
    task_weight_dict: Optional[Dict[str, float]] = None,
    bias_block: Optional[Layer] = None,
    loss_reduction=tf.reduce_mean,
    **kwargs,
) -> "ParallelPredictionBlock":
    return ParallelPredictionBlock.from_schema(
        schema,
        task_blocks=task_blocks,
        task_weight_dict=task_weight_dict,
        bias_block=bias_block,
        loss_reduction=loss_reduction,
        **kwargs,
    )


def merge(
    *branches: Union["Block", Dict[str, "Block"]],
    post: Optional["TabularTransformationsType"] = None,
    aggregation: Optional["TabularAggregationType"] = None,
    **kwargs,
) -> "ParallelBlock":
    return ParallelBlock(*branches, post=post, aggregation=aggregation, **kwargs)


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class SequentialBlock(Block):
    """The SequentialLayer represents a sequence of Keras layers.
    It is a Keras Layer that can be used instead of tf.keras.layers.Sequential,
    which is actually a Keras Model.  In contrast to keras Sequential, this
    layer can be used as a pure Layer in tf.functions and when exporting
    SavedModels, without having to pre-declare input and output shapes.  In turn,
    this layer is usable as a preprocessing layer for TF Agents Networks, and
    can be exported via PolicySaver.
    Usage::

        c = SequentialLayer([layer1, layer2, layer3])
        output = c(inputs)    # Equivalent to: output = layer3(layer2(layer1(inputs)))
    """

    def __init__(
        self,
        layers,
        filter: Optional[Union[Schema, Tag, List[str], "Filter"]] = None,
        block_name: Optional[str] = None,
        copy_layers: bool = False,
        **kwargs,
    ):
        """Create a composition.

        Parameters
        ----------
        layers:
            A list or tuple of layers to compose.
        **kwargs:
            Arguments to pass to `Keras` layer initializer, including `name`.

        Raises
        ------
        TypeError:
            If any of the layers are not instances of keras `Layer`.
        """
        self.block_name = block_name
        for layer in layers:
            if not isinstance(layer, tf.keras.layers.Layer):
                raise TypeError(
                    "Expected all layers to be instances of keras Layer, but saw: '{}'".format(
                        layer
                    )
                )

        super(SequentialBlock, self).__init__(**kwargs)
        layers = copy.copy(layers) if copy_layers else layers
        if filter:
            if not isinstance(filter, Filter):
                filter = Filter(filter)
            self.layers = [filter, *layers]
        else:
            self.layers = layers

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for layer in self.layers:
            output_shape = layer.compute_output_shape(output_shape)
        return output_shape

    def compute_output_signature(self, input_signature):
        output_signature = input_signature
        for layer in self.layers:
            output_signature = layer.compute_output_signature(output_signature)
        return output_signature

    def build(self, input_shape=None):
        last_layer = None
        for layer in self.layers:
            try:
                layer.build(input_shape)
            except TypeError:
                t, v, tb = sys.exc_info()
                if isinstance(input_shape, dict) and isinstance(last_layer, TabularBlock):
                    v = TypeError(
                        f"Couldn't build {layer}, "
                        f"did you forget to add aggregation to {last_layer}?"
                    )
                six.reraise(t, v, tb)
            input_shape = layer.compute_output_shape(input_shape)
            last_layer = layer
        self.built = True

    def set_schema(self, schema=None):
        for layer in self.layers:
            self._maybe_set_schema(layer, schema)

        return super().set_schema(schema)

    def _set_context(self, context: ModelContext):
        for layer in self.layers:
            if hasattr(layer, "_set_context"):
                layer._set_context(context)

    def _get_name(self):
        return self.block_name if self.block_name else f"{self.__class__.__name__}"

    @property
    def inputs(self):
        first = list(self)[0]
        if is_input_block(first):
            return first

    @property
    def last(self):
        return self.layers[-1]

    @property
    def filter_features(self) -> List[str]:
        if isinstance(self.layers[0], Filter):
            return self.layers[0].feature_names
        elif isinstance(self.layers[0], SequentialBlock):
            return self.layers[0].filter_features

        return []

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        weights = {}
        for layer in self.layers:
            for v in layer.trainable_weights:
                weights[id(v)] = v
        return list(weights.values())

    @property
    def non_trainable_weights(self):
        weights = {}
        for layer in self.layers:
            for v in layer.non_trainable_weights:
                weights[id(v)] = v
        return list(weights.values())

    @property
    def trainable(self):
        return all(layer.trainable for layer in self.layers)

    @trainable.setter
    def trainable(self, value):
        for layer in self.layers:
            layer.trainable = value

    @property
    def losses(self):
        values = set()
        for layer in self.layers:
            values.update(layer.losses)
        return list(values)

    @property
    def regularizers(self):
        values = set()
        for layer in self.layers:
            values.update(layer.regularizers)
        return list(values)

    def call(self, inputs, training=False, **kwargs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                filtered_kwargs = filter_kwargs(kwargs, layer, filter_positional_or_keyword=False)
            else:
                filtered_kwargs = filter_kwargs(
                    dict(training=training), layer, filter_positional_or_keyword=False
                )
            outputs = layer(outputs, **filtered_kwargs)

        return outputs

    def compute_loss(self, inputs, targets, **kwargs):
        outputs, targets = inputs, targets
        for layer in self.layers:
            outputs, targets = layer.compute_loss(outputs, targets, **kwargs)

        return outputs, targets

    def get_config(self):
        config = {}
        for i, layer in enumerate(self.layers):
            config[i] = tf.keras.utils.serialize_keras_object(layer)

        return config

    def __getitem__(self, key):
        return self.layers[key]

    @property
    def is_tabular(self):
        return getattr(self.layers[-1], "is_tabular", False)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        layers = [
            tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
            for conf in config.values()
        ]

        return SequentialBlock(layers)

    def __rrshift__(self, other):
        return right_shift_layer(self, other)

    def __rshift__(self, other):
        # pylint: disable=arguments-out-of-order
        return right_shift_layer(other, self)


tabular_transformation_registry: Registry = Registry.class_registry("tf.tabular_transformations")
tabular_aggregation_registry: Registry = Registry.class_registry("tf.tabular_aggregations")


class TabularTransformation(
    SchemaMixin, tf.keras.layers.Layer, RegistryMixin["TabularTransformation"], abc.ABC
):
    """Transformation that takes in `TabularData` and outputs `TabularData`."""

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        raise NotImplementedError()

    @classmethod
    def registry(cls) -> Registry:
        return tabular_transformation_registry


class TabularAggregation(
    SchemaMixin, tf.keras.layers.Layer, RegistryMixin["TabularAggregation"], abc.ABC
):
    """Aggregation of `TabularData` that outputs a single `Tensor`"""

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        raise NotImplementedError()

    @classmethod
    def registry(cls) -> Registry:
        return tabular_aggregation_registry

    def _expand_non_sequential_features(self, inputs: TabularData) -> TabularData:
        inputs_sizes = {k: v.shape for k, v in inputs.items()}
        seq_features_shapes, sequence_length = self._get_seq_features_shapes(inputs_sizes)

        if len(seq_features_shapes) > 0:
            non_seq_features = set(inputs.keys()).difference(set(seq_features_shapes.keys()))
            for fname in non_seq_features:
                # Including the 2nd dim and repeating for the sequence length
                inputs[fname] = tf.tile(tf.expand_dims(inputs[fname], 1), (1, sequence_length, 1))

        return inputs

    def _get_seq_features_shapes(self, inputs_sizes: Dict[str, tf.TensorShape]):
        seq_features_shapes = dict()
        for fname, fshape in inputs_sizes.items():
            # Saves the shapes of sequential features
            if len(fshape) >= 3:
                seq_features_shapes[fname] = tuple(fshape[:2])

        sequence_length = 0
        if len(seq_features_shapes) > 0:
            if len(set(seq_features_shapes.values())) > 1:
                raise ValueError(
                    "All sequential features must share the same shape in the first two dims "
                    "(batch_size, seq_length): {}".format(seq_features_shapes)
                )

            sequence_length = list(seq_features_shapes.values())[0][1]

        return seq_features_shapes, sequence_length

    def _check_concat_shapes(self, inputs: TabularData):
        input_sizes = {k: v.shape for k, v in inputs.items()}
        if len(set([tuple(v[:-1]) for v in input_sizes.values()])) > 1:
            raise Exception(
                "All features dimensions except the last one must match: {}".format(input_sizes)
            )

    def _get_agg_output_size(self, input_size, agg_dim, axis=-1):
        batch_size = calculate_batch_size_from_input_shapes(input_size)
        seq_features_shapes, sequence_length = self._get_seq_features_shapes(input_size)

        if len(seq_features_shapes) > 0:
            return batch_size, sequence_length, agg_dim

        return batch_size, agg_dim

    def get_values(self, inputs: TabularData) -> List[tf.Tensor]:
        values = []
        for value in inputs.values():
            if type(value) is dict:
                values.extend(self.get_values(value))  # type: ignore
            else:
                values.append(value)

        return values


TabularTransformationType = Union[str, TabularTransformation]
TabularTransformationsType = Union[TabularTransformationType, List[TabularTransformationType]]
TabularAggregationType = Union[str, TabularAggregation]


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class SequentialTabularTransformations(SequentialBlock):
    """A sequential container, modules will be added to it in the order they are passed in.

    Parameters
    ----------
    transformation: TabularTransformationType
        transformations that are passed in here will be called in order.
    """

    def __init__(self, transformation: TabularTransformationsType):
        if isinstance(transformation, list) and len(transformation) == 1:
            transformation = transformation[0]
        if not isinstance(transformation, (list, tuple)):
            transformation = [transformation]
        super().__init__([TabularTransformation.parse(t) for t in transformation])

    def append(self, transformation):
        self.layers.append(TabularTransformation.parse(transformation))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        layers = [
            tf.keras.utils.deserialize_keras_object(conf, custom_objects=custom_objects)
            for conf in config.values()
        ]

        return SequentialTabularTransformations(layers)


TABULAR_MODULE_PARAMS_DOCSTRING = """
    pre: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs when the module is called (so **before** `call`).
    post: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs after the module is called (so **after** `call`).
    aggregation: Union[str, TabularAggregation], optional
        Aggregation to apply after processing the `call`-method to output a single Tensor.

        Next to providing a class that extends TabularAggregation, it's also possible to provide
        the name that the class is registered in the `tabular_aggregation_registry`. Out of the box
        this contains: "concat", "stack", "element-wise-sum" &
        "element-wise-sum-item-multi".
    schema: Optional[DatasetSchema]
        DatasetSchema containing the columns used in this block.
    name: Optional[str]
        Name of the layer.
"""


@docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING)
@tf.keras.utils.register_keras_serializable(package="merlin_models")
class TabularBlock(Block):
    """Layer that's specialized for tabular-data by integrating many often used operations.

    Note, when extending this class, typically you want to overwrite the `compute_call_output_shape`
    method instead of the normal `compute_output_shape`. This because a Block can contain pre- and
    post-processing and the output-shapes are handled automatically in `compute_output_shape`. The
    output of `compute_call_output_shape` should be the shape that's outputted by the `call`-method.

    Parameters
    ----------
    {tabular_module_parameters}
    """

    def __init__(
        self,
        pre: Optional[TabularTransformationsType] = None,
        post: Optional[TabularTransformationsType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.input_size = None
        self.set_pre(pre)
        self.set_post(post)
        self.set_aggregation(aggregation)

        if schema:
            self.set_schema(schema)

    @classmethod
    def from_schema(
        cls, schema: Schema, tags=None, allow_none=True, **kwargs
    ) -> Optional["TabularBlock"]:
        """Instantiate a TabularLayer instance from a DatasetSchema.

        Parameters
        ----------
        schema
        tags
        kwargs

        Returns
        -------
        Optional[TabularModule]
        """
        schema_copy = schema.copy()
        if tags:
            schema_copy = schema_copy.select_by_tag(tags)
            if not schema_copy.column_names and not allow_none:
                raise ValueError(f"No features with tags: {tags} found")

        if not schema_copy.column_names:
            return None

        return cls.from_features(schema_copy.column_names, schema=schema_copy, **kwargs)

    @classmethod
    @docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING, extra_padding=4)
    def from_features(
        cls,
        features: List[str],
        pre: Optional[TabularTransformationsType] = None,
        post: Optional[TabularTransformationsType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        name=None,
        **kwargs,
    ) -> "TabularBlock":
        """
        Initializes a TabularLayer instance where the contents of features will be filtered out

        Parameters
        ----------
        features: List[str]
            A list of feature-names that will be used as the first pre-processing op to filter out
            all other features not in this list.
        {tabular_module_parameters}

        Returns
        -------
        TabularModule
        """
        pre = [Filter(features), pre] if pre else Filter(features)  # type: ignore

        return cls(pre=pre, post=post, aggregation=aggregation, name=name, **kwargs)

    def pre_call(
        self, inputs: TabularData, transformations: Optional[TabularTransformationsType] = None
    ) -> TabularData:
        """Method that's typically called before the forward method for pre-processing.

        Parameters
        ----------
        inputs: TabularData
             input-data, typically the output of the forward method.
        transformations: TabularTransformationsType, optional

        Returns
        -------
        TabularData
        """
        return self._maybe_apply_transformations(
            inputs, transformations=transformations or self.pre
        )

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        return inputs

    def post_call(
        self,
        inputs: TabularData,
        transformations: Optional[TabularTransformationsType] = None,
        merge_with: Union["TabularBlock", List["TabularBlock"]] = None,
        aggregation: Optional[TabularAggregationType] = None,
    ) -> TensorOrTabularData:
        """Method that's typically called after the forward method for post-processing.

        Parameters
        ----------
        inputs: TabularData
            input-data, typically the output of the forward method.
        transformations: TabularTransformationType, optional
            Transformations to apply on the input data.
        merge_with: Union[TabularModule, List[TabularModule]], optional
            Other TabularModule's to call and merge the outputs with.
        aggregation: TabularAggregationType, optional
            Aggregation to aggregate the output to a single Tensor.

        Returns
        -------
        TensorOrTabularData (Tensor when aggregation is set, else TabularData)
        """
        _aggregation: Optional[TabularAggregation] = None
        if aggregation:
            _aggregation = TabularAggregation.parse(aggregation)
        _aggregation = _aggregation or getattr(self, "aggregation", None)

        outputs = inputs
        if merge_with:
            if not isinstance(merge_with, list):
                merge_with = [merge_with]
            for layer_or_tensor in merge_with:
                to_add = layer_or_tensor(inputs) if callable(layer_or_tensor) else layer_or_tensor
                outputs.update(to_add)

        outputs = self._maybe_apply_transformations(
            outputs, transformations=transformations or self.post
        )

        if _aggregation:
            schema = getattr(self, "schema", None)
            _aggregation.set_schema(schema)
            return _aggregation(outputs)

        return outputs

    def __call__(  # type: ignore
        self,
        inputs: TabularData,
        *args,
        pre: Optional[TabularTransformationsType] = None,
        post: Optional[TabularTransformationsType] = None,
        merge_with: Union["TabularBlock", List["TabularBlock"]] = None,
        aggregation: Optional[TabularAggregationType] = None,
        **kwargs,
    ) -> TensorOrTabularData:
        """We overwrite the call method in order to be able to do pre- and post-processing.

        Parameters
        ----------
        inputs: TabularData
            Input TabularData.
        pre: TabularTransformationsType, optional
            Transformations to apply before calling the forward method. If pre is None, this method
            will check if `self.pre` is set.
        post: TabularTransformationsType, optional
            Transformations to apply after calling the forward method. If post is None, this method
            will check if `self.post` is set.
        merge_with: Union[TabularModule, List[TabularModule]]
            Other TabularModule's to call and merge the outputs with.
        aggregation: TabularAggregationType, optional
            Aggregation to aggregate the output to a single Tensor.

        Returns
        -------
        TensorOrTabularData (Tensor when aggregation is set, else TabularData)
        """
        inputs = self.pre_call(inputs, transformations=pre)

        # This will call the `call` method implemented by the super class.
        outputs = super().__call__(inputs, *args, **kwargs)  # noqa

        if isinstance(outputs, dict):
            outputs = self.post_call(
                outputs, transformations=post, merge_with=merge_with, aggregation=aggregation
            )

        return outputs

    def _maybe_apply_transformations(
        self,
        inputs: TabularData,
        transformations: Optional[TabularTransformationsType] = None,
    ) -> TabularData:
        """Apply transformations to the inputs if these are defined.

        Parameters
        ----------
        inputs
        transformations

        Returns
        -------

        """
        if transformations:
            transformations = TabularTransformation.parse(transformations)
            return transformations(inputs)

        return inputs

    def compute_call_output_shape(self, input_shapes):
        return input_shapes

    def compute_output_shape(self, input_shapes):
        if self.pre:
            input_shapes = self.pre.compute_output_shape(input_shapes)

        output_shapes = self._check_post_output_size(self.compute_call_output_shape(input_shapes))

        return output_shapes

    def build(self, input_shapes):
        super().build(input_shapes)
        output_shapes = input_shapes
        if self.pre:
            self.pre.build(input_shapes)
            output_shapes = self.pre.compute_output_shape(input_shapes)

        output_shapes = self.compute_call_output_shape(output_shapes)

        if isinstance(output_shapes, dict):
            if self.post:
                self.post.build(output_shapes)
                output_shapes = self.post.compute_output_shape(output_shapes)
            if self.aggregation:
                schema = getattr(self, "schema", None)
                self.aggregation.set_schema(schema)

                self.aggregation.build(output_shapes)

    def get_config(self):
        config = super(TabularBlock, self).get_config()
        config = maybe_serialize_keras_objects(self, config, ["pre", "post", "aggregation"])

        if self.schema:
            config["schema"] = self.schema.to_json()

        return config

    @property
    def is_tabular(self) -> bool:
        return True

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        if "schema" in config:
            config["schema"] = Schema().from_json(config["schema"])

        return super().from_config(config)

    def _check_post_output_size(self, input_shapes):
        output_shapes = input_shapes

        if isinstance(output_shapes, dict):
            if self.post:
                output_shapes = self.post.compute_output_shape(output_shapes)
            if self.aggregation:
                schema = getattr(self, "schema", None)
                self.aggregation.set_schema(schema)
                output_shapes = self.aggregation.compute_output_shape(output_shapes)

        return output_shapes

    def apply_to_all(self, inputs, columns_to_filter=None):
        if columns_to_filter:
            inputs = Filter(columns_to_filter)(inputs)
        outputs = tf.nest.map_structure(self, inputs)

        return outputs

    def set_schema(self, schema=None):
        self._maybe_set_schema(self.pre, schema)
        self._maybe_set_schema(self.post, schema)
        self._maybe_set_schema(self.aggregation, schema)

        return super().set_schema(schema)

    def set_pre(self, value: Optional[TabularTransformationsType]):
        if value and isinstance(value, SequentialTabularTransformations):
            self._pre: Optional[SequentialTabularTransformations] = value
        elif value and isinstance(value, (tf.keras.layers.Layer, list)):
            self._pre = SequentialTabularTransformations(value)
        else:
            self._pre = None

    @property
    def pre(self) -> Optional[SequentialTabularTransformations]:
        """

        Returns
        -------
        SequentialTabularTransformations, optional
        """
        return self._pre

    @property
    def post(self) -> Optional[SequentialTabularTransformations]:
        """

        Returns
        -------
        SequentialTabularTransformations, optional
        """
        return self._post

    def set_post(self, value: Optional[TabularTransformationsType]):
        if value and isinstance(value, SequentialTabularTransformations):
            self._post: Optional[SequentialTabularTransformations] = value
        elif value and isinstance(value, (tf.keras.layers.Layer, list)):
            self._post = SequentialTabularTransformations(value)
        elif value and isinstance(value, str):
            self._post = TabularTransformation.parse(value)
        else:
            self._post = None

    @property
    def aggregation(self) -> Optional[TabularAggregation]:
        """

        Returns
        -------
        TabularAggregation, optional
        """
        return self._aggregation

    def set_aggregation(self, value: Optional[Union[str, TabularAggregation]]):
        """

        Parameters
        ----------
        value
        """
        if value:
            self._aggregation: Optional[TabularAggregation] = TabularAggregation.parse(value)
        else:
            self._aggregation = None

    def repr_ignore(self):
        return []

    def repr_extra(self):
        return []

    def repr_add(self):
        return []

    @staticmethod
    def calculate_batch_size_from_input_shapes(input_shapes):
        return calculate_batch_size_from_input_shapes(input_shapes)

    def __rrshift__(self, other):
        return right_shift_layer(self, other)


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class Filter(TabularBlock):
    """Transformation that filters out certain features from `TabularData`."

    Parameters
    ----------
    to_include: List[str]
        List of features to include in the result of calling the module
    pop: bool
        Boolean indicating whether to pop the features to exclude from the inputs dictionary.
    """

    @overload
    def __init__(
        self,
        inputs: Schema,
        name=None,
        pop=False,
        exclude=False,
        add_to_context: bool = False,
        **kwargs,
    ):
        ...

    @overload
    def __init__(
        self,
        inputs: Tag,
        name=None,
        pop=False,
        exclude=False,
        add_to_context: bool = False,
        **kwargs,
    ):
        ...

    @overload
    def __init__(
        self,
        inputs: Sequence[str],
        name=None,
        pop=False,
        exclude=False,
        add_to_context: bool = False,
        **kwargs,
    ):
        ...

    def __init__(
        self, inputs, name=None, pop=False, exclude=False, add_to_context: bool = False, **kwargs
    ):
        if isinstance(inputs, Tag):
            self.feature_names = inputs
        else:
            self.feature_names = list(inputs.column_names) if isinstance(inputs, Schema) else inputs
        super().__init__(name=name, **kwargs)
        self.exclude = exclude
        self.pop = pop
        self.add_to_context = add_to_context

    def set_schema(self, schema=None):
        out = super().set_schema(schema)

        if isinstance(self.feature_names, Tag):
            self.feature_names = self.schema.select_by_tag(self.feature_names).column_names

        return out

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        """Filter out features from inputs.

        Parameters
        ----------
        inputs: TabularData
            Input dictionary containing features to filter.

        Returns Filtered TabularData that only contains the feature-names in `self.to_include`.
        -------

        """
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {k: v for k, v in inputs.items() if self.check_feature(k)}
        if self.pop:
            for key in outputs.keys():
                inputs.pop(key)

        if self.add_to_context:
            self.context.tensors.update(outputs)

            return {}

        return outputs

    def compute_call_output_shape(self, input_shape):
        outputs = {k: v for k, v in input_shape.items() if self.check_feature(k)}

        return outputs

    def check_feature(self, feature_name) -> bool:
        if self.exclude:
            return feature_name not in self.feature_names

        return feature_name in self.feature_names

    def get_config(self):
        config = super().get_config()
        config["inputs"] = self.feature_names
        config["exclude"] = self.exclude
        config["pop"] = self.pop

        return config

    # @classmethod
    # def from_config(cls, config):
    #     config = maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
    #     if "schema" in config:
    #         config["schema"] = Schema().from_json(config["schema"])
    #
    #     return cls(config.pop(""), **config)


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ParallelBlock(TabularBlock):
    """Merge multiple layers or TabularModule's into a single output of TabularData.

    Parameters
    ----------
    blocks_to_merge: Union[TabularModule, Dict[str, TabularBlock]]
        TabularBlocks to merge into, this can also be one or multiple dictionaries keyed by the
        name the module should have.
    {tabular_module_parameters}
    """

    def __init__(
        self,
        *inputs: Union[tf.keras.layers.Layer, Dict[str, tf.keras.layers.Layer]],
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ):
        super().__init__(
            pre=pre, post=post, aggregation=aggregation, schema=schema, name=name, **kwargs
        )
        self.strict = strict
        self.parallel_layers: Union[List[TabularBlock], Dict[str, TabularBlock]]
        if all(isinstance(x, dict) for x in inputs):
            to_merge: Dict[str, tf.keras.layers.Layer] = reduce(
                lambda a, b: dict(a, **b), inputs
            )  # type: ignore
            parsed_to_merge: Dict[str, TabularBlock] = {}
            for key, val in to_merge.items():
                if not getattr(val, "is_tabular", False):
                    if not hasattr(val, "as_tabular"):
                        val = SequentialBlock([val, AsTabular(key)], copy_layers=False)
                    else:
                        val = val.as_tabular(key)
                parsed_to_merge[key] = val
            self.parallel_layers = parsed_to_merge
        elif all(isinstance(x, tf.keras.layers.Layer) for x in inputs):
            parsed: List[TabularBlock] = []
            for i, inp in enumerate(inputs):
                if not getattr(inp, "is_tabular", False):
                    if not hasattr(inp, "as_tabular"):
                        val = SequentialBlock([inp, AsTabular(str(i))], copy_layers=False)
                    else:
                        inp = inp.as_tabular(str(i))
                parsed.append(inp)
            self.parallel_layers = parsed
        else:
            raise ValueError(
                "Please provide one or multiple layer's to merge or "
                f"dictionaries of layer. got: {inputs}"
            )

        # Merge schemas if necessary.
        if not schema and all(getattr(m, "schema", False) for m in self.parallel_values):
            if len(self.parallel_values) == 1:
                self.set_schema(self.parallel_values[0].schema)
            else:
                s = reduce(
                    lambda a, b: a + b, [m.schema for m in self.parallel_values]
                )  # type: ignore
                self.set_schema(s)

    @property
    def parallel_values(self) -> List[tf.keras.layers.Layer]:
        if isinstance(self.parallel_layers, dict):
            return list(self.parallel_layers.values())

        return self.parallel_layers

    @property
    def parallel_dict(self) -> Dict[str, tf.keras.layers.Layer]:
        if isinstance(self.parallel_layers, dict):
            return self.parallel_layers

        return {str(i): m for i, m in enumerate(self.parallel_layers)}

    def _set_context(self, context: "ModelContext"):
        for layer in self.parallel_values:
            if hasattr(self, "_set_context"):
                layer._set_context(context)
        super(ParallelBlock, self)._set_context(context)

    def select_by_name(self, name: str) -> Optional["Block"]:
        return self.parallel_dict.get(name)

    def __getitem__(self, key) -> "Block":
        return self.parallel_dict[key]

    def __setitem__(self, key: str, item: "Block"):
        self.parallel_dict[key] = item

    def add_branch(self, name: str, block: "Block") -> "ParallelBlock":
        if isinstance(self.parallel_layers, dict):
            self.parallel_layers[name] = block

        return self

    def apply_to_branch(self, branch_name: str, *block: "Block"):
        if isinstance(self.parallel_layers, dict):
            self.parallel_layers[branch_name] = self.parallel_layers[branch_name].apply(*block)

    def call(self, inputs, **kwargs):
        if self.strict:
            assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {}
        if isinstance(inputs, dict) and all(
            name in inputs for name in list(self.parallel_dict.keys())
        ):
            for name, block in self.parallel_dict.items():
                out = block(inputs[name])
                if isinstance(out, dict):
                    outputs.update(out)
                else:
                    outputs[name] = out
        else:
            for name, layer in self.parallel_dict.items():
                out = layer(inputs)
                if isinstance(out, dict):
                    outputs.update(out)
                else:
                    outputs[name] = out

        return outputs

    def compute_call_output_shape(self, input_shape):
        output_shapes = {}

        for name, layer in self.parallel_dict.items():
            if isinstance(input_shape, dict) and all(
                key in input_shape for key in list(self.parallel_dict.keys())
            ):
                out = layer.compute_output_shape(input_shape[name])
            else:
                out = layer.compute_output_shape(input_shape)
            if isinstance(out, dict):
                output_shapes.update(out)
            else:
                output_shapes[name] = out

        return output_shapes

    # def build(self, input_shape):
    #     if isinstance(input_shape, dict) and all(
    #             name in input_shape for name in list(self.parallel_dict.keys())
    #     ):
    #         for key, block in self.parallel_dict.items():
    #             block.build(input_shape[key])
    #     else:
    #         for layer in self.parallel_values:
    #             layer.build(input_shape)
    #
    #     return super().build(input_shape)

    def get_config(self):
        return maybe_serialize_keras_objects(
            self, super(ParallelBlock, self).get_config(), ["parallel_layers"]
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        if "schema" in config:
            config["schema"] = Schema().from_json(config["schema"])

        parallel_layers = config.pop("parallel_layers")
        inputs = {
            name: tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
            for name, conf in parallel_layers.items()
        }

        return cls(inputs, **config)


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class AsTabular(tf.keras.layers.Layer):
    """Converts a Tensor to TabularData by converting it to a dictionary.

    Parameters
    ----------
    output_name: str
        Name that should be used as the key in the output dictionary.
    name: str
        Name of the layer.
    """

    def __init__(self, output_name: str, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_name = output_name

    def call(self, inputs, **kwargs):
        return {self.output_name: inputs}

    def get_config(self):
        config = super(AsTabular, self).get_config()
        config["output_name"] = self.output_name

        return config

    @property
    def is_tabular(self) -> bool:
        return True


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class NoOp(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class Debug(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


# @tf.keras.utils.register_keras_serializable(package="merlin_models")
# class AddToContext(tf.keras.layers.Layer, ContextMixin):
#     def call(self, inputs, **kwargs):
#         return inputs
#
#     def compute_output_shape(self, input_shape):
#         return input_shape


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class WithShortcut(ParallelBlock):
    def __init__(
        self,
        block: Union[tf.keras.layers.Layer, Block],
        shortcut_filter: Optional[Filter] = None,
        aggregation=None,
        post: Optional[TabularTransformationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        strict: bool = False,
        block_outputs_name: Optional[str] = None,
        **kwargs,
    ):
        block_outputs_name = block_outputs_name or block.name
        shortcut = shortcut_filter if shortcut_filter else NoOp()
        inputs = {block_outputs_name: block, "shortcut": shortcut}
        super().__init__(
            inputs,
            post=post,
            aggregation=aggregation,
            schema=schema,
            name=name,
            strict=strict,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        output = ParallelBlock.from_config(config, **kwargs)
        output.__class__ = cls

        return output


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ResidualBlock(WithShortcut):
    def __init__(
        self,
        block: Union[tf.keras.layers.Layer, Block],
        activation=None,
        post: Optional[TabularTransformationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ):
        from merlin_models.tf.tabular.aggregation import SumResidual

        super().__init__(
            block,
            post=post,
            aggregation=SumResidual(activation=activation),
            schema=schema,
            name=name,
            strict=strict,
            **kwargs,
        )


class DualEncoderBlock(ParallelBlock):
    def __init__(
        self,
        left: Union[TabularBlock, tf.keras.layers.Layer],
        right: Union[TabularBlock, tf.keras.layers.Layer],
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        left_name: str = "left",
        right_name: str = "right",
        name: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ):
        if not getattr(left, "is_tabular", False):
            left = SequentialBlock([left, AsTabular(left_name)])
        if not getattr(right, "is_tabular", False):
            right = SequentialBlock([right, AsTabular(right_name)])

        towers = {left_name: left, right_name: right}

        super().__init__(
            towers,
            pre=pre,
            post=post,
            aggregation=aggregation,
            schema=schema,
            name=name,
            strict=strict,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        output = ParallelBlock.from_config(config, **kwargs)
        output.__class__ = cls

        return output


def call_parallel(self, other, aggregation=None, **kwargs):
    return ParallelBlock(self, other, aggregation=aggregation, **kwargs)


TabularBlock.__add__ = call_parallel


# TabularBlock.merge = call_parallel


def name_fn(name, inp):
    return "/".join([name, inp]) if name else None


MetricOrMetricClass = Union[tf.keras.metrics.Metric, Type[tf.keras.metrics.Metric]]


class Sampler(abc.ABC):
    @abc.abstractmethod
    def sample(self) -> tf.Tensor:
        raise NotImplementedError()


prediction_block_registry: Registry = Registry.class_registry("tf.prediction_blocks")


class PredictionBlock(Block):
    def call(self, inputs, training=True, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            predictions, targets = inputs
        else:
            predictions, targets = inputs, None

        return self.predict(predictions, targets, training=True, **kwargs)

    @abc.abstractmethod
    def predict(self, inputs, targets=None, training=True, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError()


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class PredictionTask(Layer, LossMixin, MetricsMixin, ContextMixin):
    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        metrics: Optional[List[MetricOrMetricClass]] = None,
        pre_call: Optional[PredictionBlock] = None,
        pre_loss: Optional[PredictionBlock] = None,
        task_block: Optional[Layer] = None,
        prediction_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        label_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        name: Optional[Text] = None,
        **kwargs,
    ) -> None:
        """Initializes the task.

        Parameters
        ----------
        loss:
            Loss function. Defaults to BinaryCrossentropy.
        metrics:
            List of Keras metrics to be evaluated.
        prediction_metrics:
            List of Keras metrics used to summarize the predictions.
        label_metrics:
            List of Keras metrics used to summarize the labels.
        loss_metrics:
            List of Keras metrics used to summarize the loss.
        name:
            Optional task name.
        """

        super().__init__(name=name, **kwargs)
        self.target_name = target_name
        self.task_block = task_block
        self._task_name = task_name
        self.pre_call_block = pre_call
        self.pre_loss_block = pre_loss

        create_metrics = self._create_metrics
        self.eval_metrics = create_metrics(metrics) if metrics else []
        self.prediction_metrics = create_metrics(prediction_metrics) if prediction_metrics else []
        self.label_metrics = create_metrics(label_metrics) if label_metrics else []
        self.loss_metrics = create_metrics(loss_metrics) if loss_metrics else []

    def pre_call(self, inputs, **kwargs):
        x = inputs

        if self.task_block:
            x = self.task_block(x)

        if self.pre_call_block:
            x = self.pre_call_block(inputs, **kwargs)

        return x

    def __call__(self, *args, **kwargs):
        inputs = self.pre_call(*args, **kwargs)

        # This will call the `call` method implemented by the super class.
        outputs = super().__call__(inputs, **kwargs)  # noqa

        return outputs

    def build_task(self, input_shape, schema: Schema, body: Block, **kwargs):
        return super().build(input_shape)

    def _create_metrics(self, metrics: List[MetricOrMetricClass]) -> List[tf.keras.metrics.Metric]:
        outputs = []
        for metric in metrics:
            if not isinstance(metric, tf.keras.metrics.Metric):
                metric = metric(name=self.child_name(generic_utils.to_snake_case(metric.__name__)))
            outputs.append(metric)

        return outputs

    @property
    def task_name(self):
        if self._task_name:
            return self._task_name

        base_name = generic_utils.to_snake_case(self.__class__.__name__)

        return name_fn(self.target_name, base_name) if self.target_name else base_name

    def child_name(self, name):
        return name_fn(self.task_name, name)

    @abc.abstractmethod
    def _compute_loss(
        self, predictions, targets, sample_weight=None, training: bool = False, **kwargs
    ) -> tf.Tensor:
        raise NotImplementedError()

    def compute_loss(  # type: ignore
        self,
        predictions,
        targets,
        training: bool = False,
        compute_metrics=True,
        sample_weight: Optional[tf.Tensor] = None,
        **kwargs,
    ) -> tf.Tensor:
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]
        if isinstance(predictions, dict) and self.target_name:
            predictions = predictions[self.task_name]

        if len(targets.shape) == len(predictions.shape) - 1:
            predictions = tf.squeeze(predictions)

        if self.pre_loss_block:
            predictions, targets = self.pre_loss_block(
                predictions, targets, training=training, **kwargs
            )

        # predictions = self(inputs, training=training, **kwargs)
        loss = self._compute_loss(
            predictions, targets, sample_weight=sample_weight, training=training
        )

        if compute_metrics:
            update_ops = self.calculate_metrics(predictions, targets, forward=False, loss=loss)

            update_ops = [x for x in update_ops if x is not None]

            with tf.control_dependencies(update_ops):
                return tf.identity(loss)

        return loss

    def repr_add(self):
        return [("loss", self.loss)]

    def calculate_metrics(self, predictions, targets, sample_weight=None, forward=True, loss=None):
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]

        if forward:
            predictions = self(predictions)

        update_ops = []

        for metric in self.eval_metrics:
            update_ops.append(
                metric.update_state(y_true=targets, y_pred=predictions, sample_weight=sample_weight)
            )

        for metric in self.prediction_metrics:
            update_ops.append(metric.update_state(predictions, sample_weight=sample_weight))

        for metric in self.label_metrics:
            update_ops.append(metric.update_state(targets, sample_weight=sample_weight))

        for metric in self.loss_metrics:
            if not loss:
                loss = self.loss(y_true=targets, y_pred=predictions, sample_weight=sample_weight)
            update_ops.append(metric.update_state(loss, sample_weight=sample_weight))

        return update_ops

    def metric_results(self, mode: str = None):
        return {metric.name: metric.result() for metric in self.metrics}

    def metric_result_dict(self, mode=None):
        return self.metric_results(mode=mode)

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(
            config,
            {
                "pre": tf.keras.layers.deserialize,
                "loss": tf.keras.losses.deserialize,
                "metrics": tf.keras.metrics.deserialize,
                "prediction_metrics": tf.keras.metrics.deserialize,
                "label_metrics": tf.keras.metrics.deserialize,
                "loss_metrics": tf.keras.metrics.deserialize,
            },
        )

        return super().from_config(config)

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(
            self,
            config,
            ["metrics", "prediction_metrics", "label_metrics", "loss_metrics", "loss", "pre"],
        )

        # config["summary_type"] = self.sequence_summary.summary_type
        if self.target_name:
            config["target_name"] = self.target_name
        if self._task_name:
            config["task_name"] = self._task_name

        return config


class ParallelPredictionBlock(ParallelBlock, LossMixin, MetricsMixin):
    def __init__(
        self,
        *prediction_tasks: PredictionTask,
        task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
        task_weights: Optional[List[float]] = None,
        bias_block: Optional[Layer] = None,
        loss_reduction=tf.reduce_mean,
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        **kwargs,
    ):
        self.loss_reduction = loss_reduction

        self.prediction_tasks = prediction_tasks
        self.task_weights = task_weights

        self.bias_block = bias_block
        self.bias_logit = tf.keras.layers.Dense(1)

        # pre = [pre, MaybeCallBody(body)] if pre else MaybeCallBody(body)

        self.prediction_task_dict = {}
        if prediction_tasks:
            for task in prediction_tasks:
                self.prediction_task_dict[task.task_name] = task

        super(ParallelPredictionBlock, self).__init__(self.prediction_task_dict, pre=pre, post=post)

        self._task_weight_dict = defaultdict(lambda: 1.0)
        if task_weights:
            for task, val in zip(prediction_tasks, task_weights):
                self._task_weight_dict[task.task_name] = val

        self._set_task_blocks(task_blocks)

    @classmethod
    def get_tasks_from_schema(cls, schema, task_weight_dict=None):
        task_weight_dict = task_weight_dict or {}

        tasks: List[PredictionTask] = []
        task_weights = []
        from .prediction.classification import BinaryClassificationTask
        from .prediction.regression import RegressionTask

        for binary_target in schema.select_by_tag(Tag.BINARY_CLASSIFICATION).column_names:
            tasks.append(BinaryClassificationTask(binary_target))
            task_weights.append(task_weight_dict.get(binary_target, 1.0))
        for regression_target in schema.select_by_tag(Tag.REGRESSION).column_names:
            tasks.append(RegressionTask(regression_target))
            task_weights.append(task_weight_dict.get(regression_target, 1.0))
        # TODO: Add multi-class classification here. Figure out how to get number of classes

        return task_weights, tasks

    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
        task_weight_dict: Optional[Dict[str, float]] = None,
        bias_block: Optional[Layer] = None,
        loss_reduction=tf.reduce_mean,
        **kwargs,
    ) -> "ParallelPredictionBlock":
        task_weight_dict = task_weight_dict or {}

        task_weights, tasks = cls.get_tasks_from_schema(schema, task_weight_dict)

        return cls(
            *tasks,
            task_blocks=task_blocks,
            task_weights=task_weights,
            bias_block=bias_block,
            loss_reduction=loss_reduction,
            **kwargs,
        )

    @classmethod
    def task_names_from_schema(cls, schema: Schema) -> List[str]:
        _, tasks = cls.get_tasks_from_schema(schema)

        return [task.task_name for task in tasks]

    def _set_task_blocks(self, task_blocks):
        if not task_blocks:
            return

        if isinstance(task_blocks, dict):
            tasks_multi_names = self._prediction_tasks_multi_names()
            for key, task_block in task_blocks.items():
                if key in tasks_multi_names:
                    tasks = tasks_multi_names[key]
                    if len(tasks) == 1:
                        self.prediction_task_dict[tasks[0].task_name].task_block = task_block
                    else:
                        raise ValueError(
                            f"Ambiguous name: {key}, can't resolve it to a task "
                            "because there are multiple tasks that contain the key: "
                            f"{', '.join([task.task_name for task in tasks])}"
                        )
                else:
                    raise ValueError(
                        f"Couldn't find {key} in prediction_tasks, "
                        f"only found: {', '.join(list(self.prediction_task_dict.keys()))}"
                    )
        elif isinstance(task_blocks, Layer):
            for key, val in self.prediction_task_dict.items():
                task_block = task_blocks.from_config(task_blocks.get_config())
                val.task_block = task_block
        else:
            raise ValueError("`task_blocks` must be a Layer or a Dict[str, Layer]")

    def _prediction_tasks_multi_names(self) -> Dict[str, List[PredictionTask]]:
        prediction_tasks_multi_names = {
            name: [val] for name, val in self.prediction_task_dict.items()
        }
        for name, value in self.prediction_task_dict.items():
            name_parts = name.split("/")
            for name_part in name_parts:
                if name_part in prediction_tasks_multi_names:
                    prediction_tasks_multi_names[name_part].append(value)
                else:
                    prediction_tasks_multi_names[name_part] = [value]

        return prediction_tasks_multi_names

    def add_task(self, task: PredictionTask, task_weight=1):
        key = task.target_name
        self.parallel_dict[key] = task
        if task_weight:
            self._task_weight_dict[key] = task_weight

        return self

    def pop_labels(self, inputs: Dict[Text, tf.Tensor]):
        outputs = {}
        for name in self.parallel_dict.keys():
            outputs[name] = inputs.pop(name)

        return outputs

    def call(
        self,
        inputs: Union[TabularData, tf.Tensor],
        training: bool = False,
        bias_outputs=None,
        **kwargs,
    ):
        if isinstance(inputs, dict) and not all(
            name in inputs for name in list(self.parallel_dict.keys())
        ):
            if self.bias_block and not bias_outputs:
                bias_outputs = self.bias_block(inputs)
            inputs = self.body(inputs)

        outputs = super(ParallelPredictionBlock, self).call(inputs, **kwargs)

        if bias_outputs is not None:
            for key in outputs:
                outputs[key] += bias_outputs

        return outputs

    def compute_call_output_shape(self, input_shape):
        if isinstance(input_shape, dict) and not all(
            name in input_shape for name in list(self.parallel_dict.keys())
        ):
            input_shape = self.body.compute_output_shape(input_shape)

        return super().compute_call_output_shape(input_shape)

    def compute_loss(
        self, inputs: Union[tf.Tensor, TabularData], targets, training=False, **kwargs
    ) -> tf.Tensor:
        losses = []

        if isinstance(inputs, dict) and not all(
            name in inputs for name in list(self.parallel_dict.keys())
        ):
            filtered_kwargs = filter_kwargs(
                dict(training=training), self, filter_positional_or_keyword=False
            )
            predictions = self(inputs, **filtered_kwargs)
        else:
            predictions = inputs

        for name, task in self.prediction_task_dict.items():
            loss = task.compute_loss(predictions, targets, training=training, **kwargs)
            losses.append(loss * self._task_weight_dict[name])

        return self.loss_reduction(losses)

    def metric_results(self, mode=None):
        def name_fn(x):
            return "_".join([mode, x]) if mode else x

        metrics = {
            name_fn(name): task.metric_results() for name, task in self.prediction_task_dict.items()
        }

        return _output_metrics(metrics)

    def metric_result_dict(self, mode=None):
        results = {}
        for name, task in self.prediction_task_dict.items():
            results.update(task.metric_results(mode=mode))

        return results

    def reset_metrics(self):
        for task in self.prediction_task_dict.values():
            task.reset_metrics()

    @property
    def task_blocks(self) -> Dict[str, Optional[Layer]]:
        return {name: task.task_block for name, task in self.prediction_task_dict.items()}

    @property
    def task_names(self) -> List[str]:
        return [name for name in self.prediction_task_dict]

    @property
    def metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        outputs = {}
        for name, task in self.parallel_dict.items():
            outputs.update({metric.name: metric for metric in task.metrics})

        return outputs

    def repr_ignore(self) -> List[str]:
        return ["prediction_tasks", "parallel_layers"]

    def _set_context(self, context: "ModelContext"):
        for task in self.prediction_task_dict.values():
            task._set_context(context)
        super(ParallelPredictionBlock, self)._set_context(context)

    @classmethod
    def from_config(cls, config, **kwargs):
        config = maybe_deserialize_keras_objects(
            config, ["body", "prediction_tasks", "task_weights"]
        )

        if "schema" in config:
            config["schema"] = Schema().from_json(config["schema"])

        config["loss_reduction"] = getattr(tf, config["loss_reduction"])

        return cls(**config)

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(
            self, config, ["body", "loss_reduction", "prediction_tasks"]
        )
        if self.task_weights:
            config["task_weights"] = self.task_weights

        return config


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class Model(tf.keras.Model, LossMixin, MetricsMixin):
    def __init__(self, body: Union[ModelLikeBlock, SequentialBlock], **kwargs):
        super(Model, self).__init__(**kwargs)
        if isinstance(body, SequentialBlock) and not isinstance(body.last, ModelLikeBlock):
            raise ValueError("SequentialBlock must have a ModelLikeBlock as last layer")
        self.body = body
        self.context = ModelContext()

    def build(self, input_shapes):
        self.body._set_context(self.context)

        super(Model, self).build(input_shapes)

    def call(self, inputs, **kwargs):
        outputs = self.body(inputs, **kwargs)
        return outputs

    @property
    def loss_block(self) -> ModelLikeBlock:
        return self.body.last if isinstance(self.body, SequentialBlock) else self.body

    def compute_loss(
        self,
        inputs: Union[tf.Tensor, TabularData],
        targets: Union[tf.Tensor, TabularData],
        compute_metrics=True,
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        return self.loss_block.compute_loss(
            inputs, targets, training=training, compute_metrics=compute_metrics, **kwargs
        )

    def calculate_metrics(
        self,
        inputs: Union[tf.Tensor, TabularData],
        targets: Union[tf.Tensor, TabularData],
        mode: str = "val",
        forward=True,
        **kwargs,
    ) -> Dict[str, Union[Dict[str, tf.Tensor], tf.Tensor]]:
        return self.loss_block.calculate_metrics(
            inputs, targets, mode=mode, forward=forward, **kwargs
        )

    def metric_results(self, mode=None):
        return self.loss_block.metric_results(mode=mode)

    def train_step(self, inputs):
        """Custom train step using the `compute_loss` method."""

        with tf.GradientTape() as tape:
            if isinstance(inputs, tuple):
                inputs, targets = inputs
            else:
                targets = None

            predictions = self(inputs, training=True)
            loss = self.compute_loss(predictions, targets, training=True)

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = self.loss_block.metric_result_dict()
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, inputs):
        """Custom test step using the `compute_loss` method."""

        if isinstance(inputs, tuple):
            inputs, targets = inputs
        else:
            targets = None

        predictions = self(inputs, training=True)
        loss = self.compute_loss(predictions, targets, training=False)

        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = self.loss_block.metric_result_dict()
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    @classmethod
    def from_config(cls, config, custom_objects=None):
        body = tf.keras.utils.deserialize_keras_object(config.pop("body"))

        return cls(body, **config)

    def get_config(self):
        return {"body": tf.keras.utils.serialize_keras_object(self.body)}


def is_input_block(block) -> bool:
    return getattr(block, "is_input", False)


def has_input_block(block) -> bool:
    if isinstance(block, SequentialBlock):
        return block.inputs is not None
    return getattr(block, "is_input", False)


class InputBlockMixin:
    @property
    def is_input(self) -> bool:
        return True


class InputBlock(TabularBlock, InputBlockMixin):
    pass


def _output_metrics(metrics):
    if len(metrics) == 1:
        return metrics[list(metrics.keys())[0]]

    return metrics


BlockType = Union[tf.keras.layers.Layer, Block]


def right_shift_layer(self, other):
    if isinstance(other, (list, Tag)):
        left_side = [Filter(other)]
    else:
        left_side = other.layers if isinstance(other, SequentialBlock) else [other]
    right_side = self.layers if isinstance(self, SequentialBlock) else [self]

    return SequentialBlock(left_side + right_side)
