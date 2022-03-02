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
from dataclasses import dataclass
from functools import reduce
from typing import (
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Text,
    Type,
    Union,
    overload,
    runtime_checkable,
)

import six
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import generic_utils

import merlin.io
from merlin.models.config.schema import SchemaMixin
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.models.utils.misc_utils import filter_kwargs
from merlin.models.utils.registry import Registry, RegistryMixin
from merlin.models.utils.schema import (
    schema_to_tensorflow_metadata_json,
    tensorflow_metadata_json_to_schema,
)
from merlin.schema import Schema, Tags

from .typing import TabularData, TensorOrTabularData
from .utils.mixins import LossMixin, MetricsMixin, ModelLikeBlock
from .utils.tf_utils import (
    calculate_batch_size_from_input_shapes,
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
)

block_registry: Registry = Registry.class_registry("tf.blocks")
BlockType = Union["Block", str, Sequence[str]]


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class BlockContext(Layer):
    """BlockContext is part of each block.

    It is used to store/retrieve public variables, and can be used to retrieve features.
    This is created automatically in the model and doesn't need to be created manually.

    """

    def __init__(self, **kwargs):
        feature_names = kwargs.pop("feature_names", [])
        feature_dtypes = kwargs.pop("feature_dtypes", {})
        super(BlockContext, self).__init__(**kwargs)
        self._feature_names = feature_names
        self._feature_dtypes = feature_dtypes

    def add_embedding_weight(self, name, **kwargs):
        table = self.add_weight(name=f"{str(name)}/embedding", **kwargs)

        return table

    def add_features(self, *name):
        self._feature_names = list({*self._feature_names, *name})

    def add_variable(self, variable):
        setattr(self, variable.name, variable)

    def set_dtypes(self, features):
        for feature_name in features:
            feature = features[feature_name]

            if isinstance(feature, tuple):
                dtype = feature[0].dtype
            else:
                dtype = feature.dtype

            self._feature_dtypes[feature_name] = dtype

    def __getitem__(self, item):
        if isinstance(item, Schema):
            if len(item.column_names) > 1:
                raise ValueError("Schema contains more than one column.")
            item = item.column_names[0]
        elif isinstance(item, Tags):
            item = item.value
        else:
            item = str(item)
        return self.named_variables[item]

    def get_embedding(self, item):
        if isinstance(item, Tags):
            item = item.value
        else:
            item = str(item)
        return self.named_variables[f"{item}/embedding"]

    def get_mask(self):
        mask_schema = self.named_variables.get("masking_schema", None)
        if mask_schema is None:
            raise ValueError(
                "The mask schema is not stored, " "please make sure that a MaskingBlock was set"
            )
        return mask_schema

    @property
    def named_variables(self) -> Dict[str, tf.Variable]:
        outputs = {}
        for var in self.variables:
            if var.name.endswith("/embedding:0"):
                name = "/".join(var.name.split("/")[-2:])
            else:
                name = var.name.split("/")[-1]
            outputs[name.replace(":0", "")] = var

        return outputs

    def _merge(self, other: "BlockContext"):
        self.public_variables.update(other.public_variables)
        self._feature_names = list(set(self._feature_names + other._feature_names))

    def build(self, input_shape):
        for feature_name in self._feature_names:
            if feature_name not in self.named_variables:
                shape = input_shape[feature_name]
                dtype = self._feature_dtypes.get(feature_name, tf.float32)

                if len(tuple(shape)) == 2:
                    var = tf.zeros([1, shape[-1]], dtype=dtype)
                    shape = tf.TensorShape([None, shape[-1]])
                elif tuple(shape) != (None,):
                    var = tf.zeros((shape), dtype=dtype)
                else:
                    var = tf.zeros([1], dtype=dtype)

                setattr(
                    self,
                    feature_name,
                    tf.Variable(
                        var,
                        name=feature_name,
                        trainable=False,
                        dtype=dtype,
                        shape=shape,
                    ),
                )

        super(BlockContext, self).build(input_shape)

    def call(self, features, **kwargs):
        for feature_name in self._feature_names:
            self.named_variables[feature_name].assign(features[feature_name])

        return features

    def get_config(self):
        config = super(BlockContext, self).get_config()
        config["feature_names"] = self._feature_names
        config["feature_dtypes"] = self._feature_dtypes

        return config


class ContextMixin:
    @property
    def context(self) -> BlockContext:
        return self._context

    def _set_context(self, context: BlockContext):
        if hasattr(self, "_context"):
            context._merge(self._context)
        self._context = context


class Block(SchemaMixin, ContextMixin, Layer):
    """Core abstraction in Merlin models."""

    registry = block_registry

    def __init__(self, context: Optional[BlockContext] = None, **kwargs):
        super(Block, self).__init__(**kwargs)
        if context:
            self._set_context(context)

    @classmethod
    @tf.autograph.experimental.do_not_convert
    def parse(cls, *block: BlockType) -> "Block":
        if len(block) == 1 and isinstance(block[0], (list, tuple)):
            block = block[0]

        if len(block) == 1:
            output: "Block" = cls.registry.parse(block[0])
        else:
            blocks = [cls.registry.parse(b) for b in block]
            output: "Block" = blocks[0].connect(*blocks[1:])

        return output

    @classmethod
    def from_layer(cls, layer: tf.keras.layers.Layer) -> "Block":
        layer.__class__ = cls

        return layer  # type: ignore

    @classmethod
    def parse_block(cls, input: Union["Block", tf.keras.layers.Layer]) -> "Block":
        if isinstance(input, Block):
            return input

        return cls.from_layer(input)

    def build(self, input_shapes):
        self._maybe_propagate_context(input_shapes)

        return super().build(input_shapes)

    def _maybe_build(self, inputs):
        if getattr(self, "_context", None) and not self.context.built:
            self.context.set_dtypes(inputs)

        super()._maybe_build(inputs)

    def call_targets(self, predictions, targets, training=False, **kwargs) -> tf.Tensor:
        return targets

    def register_features(self, feature_shapes) -> List[str]:
        return []

    def as_tabular(self, name=None) -> "Block":
        if not name:
            name = self.name

        return SequentialBlock([self, AsTabular(name)], copy_layers=False)

    def repeat(self, num: int = 1) -> "SequentialBlock":
        """Repeat the block num times.

        Parameters
        ----------
        num : int
            Number of times to repeat the block.
        """
        repeated = []
        for _ in range(num):
            repeated.append(self.copy())

        return SequentialBlock(repeated)

    def prepare(
        self,
        block: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional["TabularAggregationType"] = None,
    ) -> "SequentialBlock":
        """Transform the inputs of this block.

        Parameters
        ----------
        block: Optional[Block]
            If set, this block will be used to transform the inputs of this block.
        post: Block
            Block to use as post-transformation.
        aggregation: TabularAggregationType
            Aggregation to apply to the inputs.

        """
        block = TabularBlock(post=post, aggregation=aggregation) or block

        return SequentialBlock([block, self])

    def repeat_in_parallel(
        self,
        num: int = 1,
        prefix=None,
        names: Optional[List[str]] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional["TabularAggregationType"] = None,
        copies=True,
        residual=False,
        **kwargs,
    ) -> "ParallelBlock":
        """Repeat the block num times in parallel.

        Parameters
        ----------
        num: int
            Number of times to repeat the block.
        prefix: str
            Prefix to use for the names of the blocks.
        names: List[str]
            Names of the blocks.
        post: Block
            Block to use as post-transformation.
        aggregation: TabularAggregationType
            Aggregation to apply to the inputs.
        copies: bool
            Whether to copy the block or not.
        residual: bool
            Whether to use a residual connection or not.

        """

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
        self,
        *block: Union[tf.keras.layers.Layer, str],
        block_name: Optional[str] = None,
        context: Optional[BlockContext] = None,
    ) -> Union["SequentialBlock", "Model", "RetrievalModel"]:
        """Connect the block to other blocks sequentially.

        Parameters
        ----------
        block: Union[tf.keras.layers.Layer, str]
            Blocks to connect to.
        block_name: str
            Name of the block.
        context: Optional[BlockContext]
            Context to use for the block.

        """

        blocks = [self.parse(b) for b in block]

        for b in blocks:
            if isinstance(b, Block):
                if not b.schema:
                    b.schema = self.schema

        output = SequentialBlock(
            [self, *blocks], copy_layers=False, block_name=block_name, context=context
        )

        if isinstance(blocks[-1], ModelLikeBlock):
            if any(isinstance(b, RetrievalBlock) for b in blocks) or isinstance(
                self, RetrievalBlock
            ):
                return RetrievalModel(output)

            return Model(output)

        return output

    def connect_with_residual(
        self,
        block: Union[tf.keras.layers.Layer, str],
        activation=None,
    ) -> "SequentialBlock":
        """Connect the block to other blocks sequentially with a residual connection.

        Parameters
        ----------
        block: Union[tf.keras.layers.Layer, str]
            Blocks to connect to.
        activation: str
            Activation to use for the residual connection.

        """

        _block = self.parse(block)
        residual_block = ResidualBlock(_block, activation=activation)

        if isinstance(self, SequentialBlock):
            self.layers.append(residual_block)

            return self

        return SequentialBlock([self, residual_block], copy_layers=False)

    def connect_with_shortcut(
        self,
        block: Union[tf.keras.layers.Layer, str],
        shortcut_filter: Optional["Filter"] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional["TabularAggregationType"] = None,
        block_outputs_name: Optional[str] = None,
    ) -> "SequentialBlock":
        """Connect the block to other blocks sequentially with a shortcut connection.

        Parameters
        ----------
        block: Union[tf.keras.layers.Layer, str]
            Blocks to connect to.
        shortcut_filter: Filter
            Filter to use for the shortcut connection.
        post: Block
            Block to use as post-transformation.
        aggregation: TabularAggregationType
            Aggregation to apply to the outputs.
        block_outputs_name: str
            Name of the block outputs.
        """

        _block = self.parse(block) if not isinstance(block, Block) else block
        residual_block = WithShortcut(
            _block,
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
        """Connect the block to a debug block.

        Parameters
        ----------
        append: bool
            Whether to append the debug block to the block or to prepend it.

        """

        if not append:
            return SequentialBlock([Debug(), self])

        return self.connect(Debug())

    def connect_branch(
        self,
        *branches: Union["Block", "PredictionTask", str],
        add_rest=False,
        post: Optional[BlockType] = None,
        aggregation: Optional["TabularAggregationType"] = None,
        **kwargs,
    ) -> Union["SequentialBlock", "Model", "RetrievalModel"]:
        """Connect the block to one or multiple branches.

        Parameters
        ----------
        branches: Union[Block, PredictionTask, str]
            Blocks to connect to.
        add_rest: bool
            Whether to add the rest of the block to the branches.
        post: Block
            Block to use as post-transformation.
        aggregation: TabularAggregationType
            Aggregation to apply to the outputs.

        """
        branches = [self.parse(b) for b in branches]

        all_features = []
        for branch in branches:
            if getattr(branch, "set_schema", None):
                branch.set_schema(self.schema)
            if isinstance(branch, SequentialBlock):
                filter_features = branch.filter_features
                if filter_features:
                    all_features.extend(filter_features)

        if add_rest:
            rest_features = self.schema.without(list(set([str(f) for f in all_features])))
            rest_block = SequentialBlock([Filter(rest_features)])
            branches.append(rest_block)

        if all(isinstance(branch, ModelLikeBlock) for branch in branches):
            parallel = ParallelPredictionBlock(
                *branches, post=post, aggregation=aggregation, **kwargs
            )

            return Model(SequentialBlock([self, parallel]))

        return SequentialBlock(
            [self, ParallelBlock(*branches, post=post, aggregation=aggregation, **kwargs)]
        )

    def select_by_name(self, name: str) -> Optional["Block"]:
        if name == self.name:
            return self

        return None

    def copy(self):
        return self.from_config(self.get_config())

    def _maybe_propagate_context(self, input_shapes):
        if getattr(self, "_context", None) and not self.context.built:
            for module in self.submodules:
                if hasattr(module, "_set_context") and not getattr(module, "context", False):
                    module._set_context(self.context)
                if hasattr(module, "add_features_to_context") and not getattr(
                    module, "_features_registered", False
                ):
                    feature_names = module.add_features_to_context(input_shapes)
                    module._features_registered = True
                    if feature_names:
                        self.context.add_features(*feature_names)
            self._need_to_call_context = True
            self.context.build(input_shapes)

    def __rrshift__(self, other):
        return right_shift_layer(self, other)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
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
        *layers,
        filter: Optional[Union[Schema, Tags, List[str], "Filter"]] = None,
        pre_aggregation: Optional["TabularAggregationType"] = None,
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
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = layers[0]

        self.block_name = block_name

        if pre_aggregation:
            layers = [TabularBlock(aggregation=pre_aggregation), *layers]

        for layer in layers:
            if not isinstance(layer, tf.keras.layers.Layer):
                raise TypeError(
                    "Expected all layers to be instances of keras Layer, but saw: '{}'".format(
                        layer
                    )
                )

        super(SequentialBlock, self).__init__(**kwargs)

        if getattr(layers[0], "schema", None):
            super().set_schema(layers[0].schema)

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
        self._maybe_propagate_context(input_shape)
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

    def _get_name(self):
        return self.block_name if self.block_name else f"{self.__class__.__name__}"

    @property
    def inputs(self):
        first = list(self)[0]
        if isinstance(first, SequentialBlock):
            return first.inputs
        if is_input_block(first):
            return first

    @property
    def first(self):
        return self.layers[0]

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
        if getattr(self, "_need_to_call_context", False):
            self.context(inputs)

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
            outputs, targets = layer.compute_loss(outputs, targets=targets, **kwargs)

        return outputs, targets

    def call_targets(self, predictions, targets, training=False, **kwargs):
        outputs = targets
        for layer in self.layers:
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                outputs, predictions = outputs
            outputs = layer.call_targets(predictions, outputs, training=training, **kwargs)

        return outputs

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


tabular_aggregation_registry: Registry = Registry.class_registry("tf.tabular_aggregations")


class TabularAggregation(
    SchemaMixin, tf.keras.layers.Layer, RegistryMixin["TabularAggregation"], abc.ABC
):
    registry = tabular_aggregation_registry

    """Aggregation of `TabularData` that outputs a single `Tensor`"""

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        raise NotImplementedError()

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

        return tf.TensorShape((batch_size, agg_dim))

    def get_values(self, inputs: TabularData) -> List[tf.Tensor]:
        values = []
        for value in inputs.values():
            if type(value) is dict:
                values.extend(self.get_values(value))  # type: ignore
            else:
                values.append(value)

        return values


TabularAggregationType = Union[str, TabularAggregation]

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
@tf.keras.utils.register_keras_serializable(package="merlin.models")
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
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        is_input: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.input_size = None
        self.set_pre(pre)
        self.set_post(post)
        self.set_aggregation(aggregation)
        self._is_input = is_input

        if schema:
            self.set_schema(schema)

    @property
    def is_input(self) -> bool:
        return self._is_input

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
        schema_copy = copy.copy(schema)
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
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
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
        self, inputs: TabularData, transformations: Optional[BlockType] = None
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
        transformations: Optional[BlockType] = None,
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
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
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
        transformations: Optional[BlockType] = None,
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
            transformations = Block.parse(transformations)
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
            config["schema"] = schema_to_tensorflow_metadata_json(self.schema)

        return config

    @property
    def is_tabular(self) -> bool:
        return True

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        if "schema" in config:
            config["schema"] = tensorflow_metadata_json_to_schema(config["schema"])

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

    def set_pre(self, value: Optional[BlockType]):
        self._pre = Block.parse(value) if value else None

    @property
    def pre(self) -> Optional[Block]:
        """

        Returns
        -------
        SequentialTabularTransformations, optional
        """
        return self._pre

    @property
    def post(self) -> Optional[Block]:
        """

        Returns
        -------
        SequentialTabularTransformations, optional
        """
        return self._post

    def set_post(self, value: Optional[BlockType]):
        self._post = Block.parse(value) if value else None

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


@tf.keras.utils.register_keras_serializable(package="merlin.models")
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
        inputs: Tags,
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
        if isinstance(inputs, Tags):
            self.feature_names = inputs
        else:
            self.feature_names = list(inputs.column_names) if isinstance(inputs, Schema) else inputs
        super().__init__(name=name, **kwargs)
        self.exclude = exclude
        self.pop = pop
        self.add_to_context = add_to_context

    def set_schema(self, schema=None):
        out = super().set_schema(schema)

        if isinstance(self.feature_names, Tags):
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
        if self.add_to_context:
            return {}

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


@tf.keras.utils.register_keras_serializable(package="merlin.models")
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
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
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
                parsed_to_merge[key] = val
            self.parallel_layers = parsed_to_merge
        elif all(isinstance(x, tf.keras.layers.Layer) for x in inputs):
            parsed: List[TabularBlock] = []
            for i, inp in enumerate(inputs):
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

        if getattr(self, "_need_to_call_context", False):
            self.context(inputs)

        outputs = {}
        if isinstance(inputs, dict) and all(
            name in inputs for name in list(self.parallel_dict.keys())
        ):
            for name, block in self.parallel_dict.items():
                out = block(inputs[name])
                if not isinstance(out, dict):
                    out = {name: out}
                outputs.update(out)
        else:
            for name, layer in self.parallel_dict.items():
                out = layer(inputs)
                if not isinstance(out, dict):
                    out = {name: out}
                outputs.update(out)

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

    def build(self, input_shape):
        if isinstance(input_shape, dict) and all(
            name in input_shape for name in list(self.parallel_dict.keys())
        ):
            for key, block in self.parallel_dict.items():
                block.build(input_shape[key])
        else:
            for layer in self.parallel_values:
                layer.build(input_shape)

        return super().build(input_shape)

    def get_config(self):
        return maybe_serialize_keras_objects(
            self, super(ParallelBlock, self).get_config(), ["parallel_layers"]
        )

    @classmethod
    def parse_config(cls, config, custom_objects=None):
        config = maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        if "schema" in config:
            config["schema"] = tensorflow_metadata_json_to_schema(config["schema"])

        parallel_layers = config.pop("parallel_layers")
        inputs = {
            name: tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
            for name, conf in parallel_layers.items()
        }

        return inputs, config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        inputs, config = cls.parse_config(config, custom_objects)

        return cls(inputs, **config)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
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

    def compute_output_shape(self, input_shape):
        return {self.output_name: input_shape}

    def get_config(self):
        config = super(AsTabular, self).get_config()
        config["output_name"] = self.output_name

        return config

    @property
    def is_tabular(self) -> bool:
        return True


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class NoOp(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class Debug(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class WithShortcut(ParallelBlock):
    def __init__(
        self,
        block: Union[tf.keras.layers.Layer, Block],
        shortcut_filter: Optional[Filter] = None,
        aggregation=None,
        post: Optional[BlockType] = None,
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


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ResidualBlock(WithShortcut):
    def __init__(
        self,
        block: Union[tf.keras.layers.Layer, Block],
        activation=None,
        post: Optional[BlockType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ):
        from merlin.models.tf.blocks.aggregation import SumResidual

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
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
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


@dataclass
class EmbeddingWithMetadata:
    embeddings: tf.Tensor
    metadata: Dict[str, tf.Tensor]


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class PredictionTask(Layer, LossMixin, MetricsMixin, ContextMixin):
    """Base-class for prediction tasks.

    Parameters
    ----------
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

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        metrics: Optional[List[MetricOrMetricClass]] = None,
        pre: Optional[Block] = None,
        task_block: Optional[Layer] = None,
        prediction_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        label_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        name: Optional[Text] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.target_name = target_name
        self.task_block = task_block
        self._task_name = task_name
        self.pre = pre

        create_metrics = self._create_metrics
        self.eval_metrics = create_metrics(metrics) if metrics else []
        self.prediction_metrics = create_metrics(prediction_metrics) if prediction_metrics else []
        self.label_metrics = create_metrics(label_metrics) if label_metrics else []
        self.loss_metrics = create_metrics(loss_metrics) if loss_metrics else []

    def pre_call(self, inputs, **kwargs):
        x = inputs

        if self.task_block:
            x = self.task_block(x)

        if self.pre:
            x = self.pre(inputs, **kwargs)

        return x

    def pre_loss(self, predictions, targets, **kwargs):
        targets = self.pre.call_targets(predictions, targets, **kwargs)
        return targets

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

        if isinstance(predictions, dict) and self.target_name and self.task_name in predictions:
            predictions = predictions[self.task_name]

        if self.pre:
            targets = self.pre_loss(predictions, targets, training=training, **kwargs)
            if isinstance(targets, tuple):
                targets, predictions = targets

        if isinstance(targets, tf.Tensor) and len(targets.shape) == len(predictions.shape) - 1:
            predictions = tf.squeeze(predictions)

        loss = self._compute_loss(
            predictions, targets=targets, sample_weight=sample_weight, training=training
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
            ["metrics", "prediction_metrics", "label_metrics", "loss_metrics", "pre"],
        )

        # config["summary_type"] = self.sequence_summary.summary_type
        if self.target_name:
            config["target_name"] = self.target_name
        if self._task_name:
            config["task_name"] = self._task_name

        if "metrics" not in config:
            config["metrics"] = []

        return config


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ParallelPredictionBlock(ParallelBlock, LossMixin, MetricsMixin):
    """Multi-task prediction block.

    Parameters
    ----------
    prediction_tasks: *PredictionTask
        List of tasks to be used for prediction.
    task_blocks: Optional[Union[Layer, Dict[str, Layer]]]
        Task blocks to be used for prediction.
    task_weights : Optional[List[float]]
        Weights for each task.
    bias_block : Optional[Layer]
        Bias block to be used for prediction.
    loss_reduction : Callable
        Reduction function for loss.

    """

    def __init__(
        self,
        *prediction_tasks: PredictionTask,
        task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
        task_weights: Optional[List[float]] = None,
        bias_block: Optional[Layer] = None,
        loss_reduction=tf.reduce_mean,
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        **kwargs,
    ):
        self.loss_reduction = loss_reduction

        self.prediction_tasks = prediction_tasks
        self.task_weights = task_weights

        self.bias_block = bias_block
        self.bias_logit = tf.keras.layers.Dense(1)

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

        for binary_target in schema.select_by_tag(Tags.BINARY_CLASSIFICATION).column_names:
            tasks.append(BinaryClassificationTask(binary_target))
            task_weights.append(task_weight_dict.get(binary_target, 1.0))
        for regression_target in schema.select_by_tag(Tags.REGRESSION).column_names:
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

    def _set_context(self, context: "BlockContext"):
        for task in self.prediction_task_dict.values():
            task._set_context(context)
        super(ParallelPredictionBlock, self)._set_context(context)

    @classmethod
    def from_config(cls, config, **kwargs):
        config = maybe_deserialize_keras_objects(config, ["body", "prediction_tasks"])

        if "schema" in config:
            config["schema"] = tensorflow_metadata_json_to_schema(config["schema"])

        config["loss_reduction"] = getattr(tf, config["loss_reduction"])

        prediction_tasks = config.pop("prediction_tasks", [])

        return cls(*prediction_tasks, **config)

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(
            self, config, ["body", "loss_reduction", "prediction_tasks"]
        )
        if self.task_weights:
            config["task_weights"] = self.task_weights

        return config


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ModelBlock(Block, tf.keras.Model):
    def __init__(self, block: Block, **kwargs):
        super().__init__(**kwargs)
        self.block = block

    def call(self, inputs, **kwargs):
        outputs = self.block(inputs, **kwargs)
        return outputs

    @property
    def schema(self) -> Schema:
        return self.block.schema

    @classmethod
    def from_config(cls, config, custom_objects=None):
        block = tf.keras.utils.deserialize_keras_object(config.pop("block"))

        return cls(block, **config)

    def get_config(self):
        return {"block": tf.keras.utils.serialize_keras_object(self.block)}


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class Model(tf.keras.Model, LossMixin, MetricsMixin):
    def __init__(
        self,
        *blocks: Union[Block, ModelLikeBlock],
        context: Optional[BlockContext] = None,
        **kwargs,
    ):
        super(Model, self).__init__(**kwargs)
        context = context or BlockContext()
        if (
            len(blocks) == 1
            and isinstance(blocks[0], SequentialBlock)
            and isinstance(blocks[0].layers[-1], ModelLikeBlock)
        ):
            self.block = blocks[0]
        else:
            if not isinstance(blocks[-1], ModelLikeBlock):
                raise ValueError("Last block must be able to calculate loss & metrics.")
            self.block = SequentialBlock(blocks, context=context)
        if not getattr(self.block, "_context", None):
            self.block._set_context(context)
        self.context = context

    def call(self, inputs, **kwargs):
        outputs = self.block(inputs, **kwargs)
        return outputs

    # @property
    # def inputs(self):
    #     return self.block.inputs

    @property
    def first(self):
        return self.block.layers[0]

    @property
    def last(self):
        return self.block.layers[-1]

    @property
    def loss_block(self) -> ModelLikeBlock:
        return self.block.last if isinstance(self.block, SequentialBlock) else self.block

    @property
    def schema(self) -> Schema:
        return self.block.schema

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
            tf.assert_rank(
                loss,
                0,
                "The loss tensor should have rank 0. "
                "Check if you are using a tf.keras.losses.Loss with 'reduction' "
                "properly set",
            )
            assert loss.dtype == tf.float32, (
                f"The loss dtype should be tf.float32 but is rather {loss.dtype}. "
                "Ensure that your model output has tf.float32 dtype, as "
                "that should be the case when using mixed_float16 policy "
                "to avoid numerical instabilities."
            )

            regularization_loss = tf.reduce_sum(self.losses)

            total_loss = tf.add_n([loss, regularization_loss])

            if getattr(self.optimizer, "get_scaled_loss", False):
                scaled_loss = self.optimizer.get_scaled_loss(total_loss)

        # If mixed precision (mixed_float16 policy) is enabled
        # (and the optimizer is automatically wrapped by
        #  tensorflow.keras.mixed_precision.LossScaleOptimizer())
        if getattr(self.optimizer, "get_scaled_loss", False):
            scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
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

        predictions = self(inputs, training=False)
        loss = self.compute_loss(predictions, targets, training=False)
        tf.assert_rank(
            loss,
            0,
            "The loss tensor should have rank 0. "
            "Check if you are using a tf.keras.losses.Loss with 'reduction' "
            "properly set",
        )

        # Casting regularization loss to fp16 if needed to match the main loss
        regularization_loss = tf.cast(tf.reduce_sum(self.losses), loss.dtype)

        total_loss = loss + regularization_loss

        metrics = self.loss_block.metric_result_dict()
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        **kwargs,
    ):
        # Check if merlin-dataset is passed
        if hasattr(x, "to_ddf"):
            if not batch_size:
                raise ValueError("batch_size must be specified when using merlin-dataset.")
            from .dataset import Dataset

            x = Dataset(x, batch_size=batch_size, **kwargs)

        return super().fit(
            x,
            y,
            batch_size,
            epochs,
            verbose,
            callbacks,
            validation_split,
            validation_data,
            shuffle,
            class_weight,
            sample_weight,
            initial_epoch,
            steps_per_epoch,
            validation_steps,
            validation_batch_size,
            validation_freq,
            max_queue_size,
            workers,
            use_multiprocessing,
        )

    def batch_predict(
        self, dataset: merlin.io.Dataset, batch_size: int, **kwargs
    ) -> merlin.io.Dataset:
        """Batched prediction using the Dask.

        Parameters
        ----------
        dataset: merlin.io.Dataset
            Dataset to predict on.
        batch_size: int
            Batch size to use for prediction.

        Returns merlin.io.Dataset
        -------

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

        from .prediction.batch import TFModelEncode

        model_encode = TFModelEncode(self, batch_size=batch_size, **kwargs)
        predictions = dataset.map_partitions(model_encode)

        return merlin.io.Dataset(predictions)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        block = tf.keras.utils.deserialize_keras_object(config.pop("block"))

        return cls(block, **config)

    def get_config(self):
        return {"block": tf.keras.utils.serialize_keras_object(self.block)}


@runtime_checkable
class RetrievalBlock(Protocol):
    def query_block(self) -> Block:
        ...

    def item_block(self) -> Block:
        ...


class RetrievalModel(Model):
    """Embedding-based retrieval model."""

    def __init__(
        self,
        *blocks: Union[Block, ModelLikeBlock],
        context: Optional[BlockContext] = None,
        **kwargs,
    ):
        super().__init__(*blocks, context=context, **kwargs)

        if not any(isinstance(b, RetrievalBlock) for b in self.block):
            raise ValueError("Model must contain a `RetrievalBlock`.")

    @property
    def retrieval_block(self) -> RetrievalBlock:
        return next(b for b in self.blocks if isinstance(b, RetrievalBlock))

    def query_embeddings(
        self,
        dataset: merlin.io.Dataset,
        dim: int,
        batch_size=None,
    ) -> merlin.io.Dataset:
        """Export query embeddings from the model.

        Parameters
        ----------
        dataset: merlin.io.Dataset
            Dataset to export embeddings from.
        dim: int
            Dimensionality of the embeddings.
        batch_size: int
            Batch size to use for embedding extraction.

        Returns
        -------
        merlin.io.Dataset
        """
        from merlin.models.tf.prediction.batch import QueryEmbeddings

        get_user_emb = QueryEmbeddings(self, dim=dim, batch_size=batch_size)

        # Check if merlin-dataset is passed
        if hasattr(dataset, "to_ddf"):
            dataset = dataset.to_ddf()

        embeddings = dataset.map_partitions(get_user_emb)

        return merlin.io.Dataset(embeddings)

    def item_embeddings(
        self, dataset: merlin.io.Dataset, dim: int, batch_size=None, **kwargs
    ) -> merlin.io.Dataset:
        """Export item embeddings from the model.

        Parameters
        ----------
        dataset: merlin.io.Dataset
            Dataset to export embeddings from.
        dim: int
            Dimensionality of the embeddings.
        batch_size: int
            Batch size to use for embedding extraction.

        Returns
        -------
        merlin.io.Dataset
        """
        from merlin.models.tf.prediction.batch import ItemEmbeddings

        get_item_emb = ItemEmbeddings(self, dim=dim, batch_size=batch_size)

        # Check if merlin-dataset is passed
        if hasattr(dataset, "to_ddf"):
            dataset = dataset.to_ddf()

        embeddings = dataset.map_partitions(get_item_emb)

        return merlin.io.Dataset(embeddings)


def is_input_block(block: Block) -> bool:
    return block and getattr(block, "is_input", None)


def has_input_block(block: Block) -> bool:
    if isinstance(block, SequentialBlock):
        return block.inputs is not None and is_input_block(block.inputs)
    return is_input_block(block.inputs)


def _output_metrics(metrics):
    if len(metrics) == 1:
        return metrics[list(metrics.keys())[0]]

    return metrics


def right_shift_layer(self, other):
    if isinstance(other, (list, Tags)):
        left_side = [Filter(other)]
    else:
        left_side = other.layers if isinstance(other, SequentialBlock) else [other]
    right_side = self.layers if isinstance(self, SequentialBlock) else [self]

    return SequentialBlock(left_side + right_side)
