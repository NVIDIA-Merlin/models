from __future__ import annotations

import copy
import sys
from functools import reduce
from typing import Dict, List, Optional, Union, TYPE_CHECKING

import six
import tensorflow as tf

from merlin.models.tf.blocks.core.base import (
    Block,
    BlockType,
    NoOp,
    PredictionOutput,
    is_input_block,
    right_shift_layer,
)
from merlin.models.tf.blocks.core.tabular import Filter, TabularAggregationType, TabularBlock
from merlin.models.tf.blocks.core.transformations import AsDenseFeatures, L2Norm
from merlin.models.tf.dataset import DictWithSchema
from merlin.models.tf.utils import tf_utils
from merlin.models.tf.utils.tf_utils import filter_kwargs_layer
from merlin.models.utils import schema_utils
from merlin.models.utils.misc_utils import filter_kwargs, has_kwargs
from merlin.schema import Schema, Tags

if TYPE_CHECKING:
    from merlin.models.tf.models.base import ModelBlock


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

    _has_custom__call__ = True

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
            layers = layers[0]  # type: ignore

        self.block_name = block_name

        if pre_aggregation:
            layers = [TabularBlock(aggregation=pre_aggregation), *layers]  # type: ignore

        for layer in layers:
            if not isinstance(layer, tf.keras.layers.Layer):
                raise TypeError(
                    "Expected all layers to be instances of keras Layer, but saw: '{}'".format(
                        layer
                    )
                )

        if block_name and "name" not in kwargs:
            kwargs["name"] = block_name

        super(SequentialBlock, self).__init__(**kwargs)

        if getattr(layers[0], "has_schema", None):
            super().set_schema(layers[0].schema)

        layers = copy.copy(layers) if copy_layers else layers
        if filter:
            if not isinstance(filter, Filter):
                filter = Filter(filter)
            self.layers = [filter, *layers]
        else:
            self.layers = list(layers)

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
        values, _val_names = [], set()
        for layer in self.layers:
            losses = layer.losses
            for loss in losses:
                if isinstance(loss, tf.Tensor):
                    if loss.ref() not in _val_names:
                        _val_names.add(loss.ref())
                        values.append(loss)
                    else:
                        raise ValueError(f"Loss should be a Tensor, found: {loss}")

        return values

    @property
    def regularizers(self):
        values = set()
        for layer in self.layers:
            values.update(layer.regularizers)
        return list(values)

    def call(
            self,
            inputs,
            targets=None,
            training=False,
            testing=False,
            **kwargs
    ):
        from merlin.models.tf import PredictionTask

        # if getattr(self, "_need_to_call_context", False):
        #     # convert sparse inputs to dense before storing them to the context?
        #     self.context(AsDenseFeatures()(inputs))

        all_kwargs = {"training": training, "testing": testing}

        # TODO fix this
        if "features" not in kwargs and isinstance(inputs, dict):
            context_schema = getattr(self.context, "schema", None)
            schema = kwargs.get("schema", context_schema)

            if schema and len(schema.column_names) == len(inputs):
                features = DictWithSchema(schema, AsDenseFeatures()(inputs))
                all_kwargs["features"] = features

        outputs = inputs

        for i, layer in enumerate(self.layers):
            filtered_kwargs = all_kwargs
            # We need to check if we need to forward kwargs to last layer
            if isinstance(layer, PredictionTask):
                filtered_kwargs = dict(targets=targets, **filtered_kwargs)
            elif i == len(self.layers) - 1:
                filtered_kwargs = dict(**all_kwargs, **kwargs)
                filtered_kwargs = filter_kwargs_layer(filtered_kwargs, layer, cascade_kwargs_if_possible=True)
            else:
                filtered_kwargs = filter_kwargs_layer(
                    filtered_kwargs, layer, cascade_kwargs_if_possible=True
                )
            outputs = layer(outputs, **filtered_kwargs)

        return outputs

    def compute_loss(self, inputs, targets, **kwargs):
        outputs, targets = inputs, targets
        for layer in self.layers:
            outputs, targets = layer.compute_loss(outputs, targets=targets, **kwargs)

        return outputs, targets

    def call_outputs(
        self, outputs: PredictionOutput, training=False, **kwargs
    ) -> "PredictionOutput":
        for layer in self.layers:
            outputs = layer.call_outputs(outputs, training=training, **kwargs)
        return outputs

    def get_config(self):
        config = {"block_name": self.block_name}
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
        block_name = config.pop("block_name", None)
        layers = [
            tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
            for conf in config.values()
        ]

        return SequentialBlock(layers, block_name=block_name)

    def __rrshift__(self, other):
        return right_shift_layer(self, other)

    def __rshift__(self, other):
        # pylint: disable=arguments-out-of-order
        return right_shift_layer(other, self)


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

    _has_custom__call__ = True

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
                parsed.append(inp)  # type: ignore
            self.parallel_layers = parsed
        else:
            raise ValueError(
                "Please provide one or multiple layer's to merge or "
                f"dictionaries of layer. got: {inputs}"
            )

        # Merge schemas if necessary.
        if not schema and all(getattr(m, "_schema", False) for m in self.parallel_values):
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
    def layers(self) -> List[tf.keras.layers.Layer]:
        return self.parallel_values

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

    def call(self, inputs, targets=None, training=False, testing=False, **kwargs):
        if self.strict:
            assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {}
        input_fn = lambda name: inputs      # noqa
        if (
            isinstance(inputs, dict)
            and all(name in inputs for name in list(self.parallel_dict.keys()))
        ):
            block_input_fn = lambda name: inputs[name]
        for name, layer in self.parallel_dict.items():
            filtered_kwargs = dict(training=training, testing=testing, **kwargs)
            filtered_kwargs = filter_kwargs(
                filtered_kwargs, layer, cascade_kwargs_if_possible=True
            )
            from merlin.models.tf import PredictionTask
            if isinstance(layer, PredictionTask):
                filtered_kwargs["targets"] = targets

            out = layer(input_fn(name), **filtered_kwargs)
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
        return tf_utils.maybe_serialize_keras_objects(
            self, super(ParallelBlock, self).get_config(), ["parallel_layers"]
        )

    @classmethod
    def parse_config(cls, config, custom_objects=None):
        config = tf_utils.maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        if "schema" in config:
            config["schema"] = schema_utils.tensorflow_metadata_json_to_schema(config["schema"])

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
class DualEncoderBlock(ParallelBlock):
    QUERY_BRANCH_NAME = "query"
    ITEM_BRANCH_NAME = "item"

    def __init__(
            self,
            query_block: Block,
            item_block: Block,
            normalize: bool = True,
            pre: Optional[BlockType] = None,
            post: Optional[BlockType] = None,
            aggregation: Optional[TabularAggregationType] = None,
            schema: Optional[Schema] = None,
            name: Optional[str] = None,
            strict: bool = False,
            **kwargs,
    ):
        """Prepare the Query and Item towers of a Retrieval block

        Parameters
        ----------
        query_block : Block
            The `Block` instance that combines user features
        item_block : Block
            Optional `Block` instance that combines items features.
        pre : Optional[BlockType], optional
            Optional `Block` instance to apply before the `call` method of the Two-Tower block
        post : Optional[BlockType], optional
            Optional `Block` instance to apply on both outputs of Two-tower model
        aggregation : Optional[TabularAggregationType], optional
            The Aggregation operation to apply after processing the `call` method
            to output a single Tensor.
        schema : Optional[Schema], optional
            The `Schema` object with the input features.
        name : Optional[str], optional
            Name of the layer.
        strict : bool, optional
            If enabled, check that the input of the ParallelBlock instance is a dictionary.
        """
        if normalize:
            query_block = query_block.connect(L2Norm())
            item_block = item_block.connect(L2Norm())

        from merlin.models.tf.models.base import ModelBlock

        self._query_block = ModelBlock(query_block, block_name="QueryBlock")
        self._query_block._name = "query"
        self._item_block = ModelBlock(item_block, block_name="ItemBlock")
        self._item_block._name = "item"

        query_branch = Filter(query_block.schema).connect(self._query_block)
        item_branch = Filter(item_block.schema).connect(self._item_block)

        branches = {
            self.QUERY_BRANCH_NAME: query_branch,
            self.ITEM_BRANCH_NAME: item_branch,
        }

        super().__init__(
            branches,
            pre=pre,
            post=post,
            aggregation=aggregation,
            schema=schema,
            name=name,
            strict=strict,
            **kwargs,
        )

    def query_block(self) -> ModelBlock:
        return self._query_block

    def item_block(self) -> ModelBlock:
        return self._item_block

    @classmethod
    def from_config(cls, config, custom_objects=None):
        inputs, config = cls.parse_config(config, custom_objects)
        output = ParallelBlock(inputs, **config)
        output.__class__ = cls

        return output

    @classmethod
    def create_with_inputs(
            cls,
            schema: Schema,
            query_block: Optional[Block] = None,
            item_block: Optional[Block] = None,
            query_tag=Tags.USER,
            item_tag=Tags.ITEM,
            **kwargs
    ):
        from merlin.models.tf import InputBlock

        query_schema = schema.select_by_tag(query_tag)
        _query_block = InputBlock(
            query_schema, aggregation="concat" if len(query_schema) == 1 else None
        )
        if query_block is not None:
            _query_block = _query_block.connect(query_block)
        item_schema = schema.select_by_tag(item_tag)
        _item_block = InputBlock(
            item_schema, aggregation="concat" if len(item_schema) == 1 else None
        )
        if item_block is not None:
            _item_block = _item_block.connect(item_block)

        return cls(_query_block, _item_block, **kwargs)

    @classmethod
    def create_with_id_inputs(
            cls,
            schema: Schema,
            query_id_tag=Tags.USER_ID,
            item_id_tag=Tags.ITEM_ID,
            **kwargs
    ):
        return cls.create_with_inputs(
            schema,
            query_tag=query_id_tag,
            item_tag=item_id_tag,
            **kwargs
        )


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class WithShortcut(ParallelBlock):
    def __init__(
        self,
        block: Union[tf.keras.layers.Layer, Block],
        shortcut_filter: Optional[Filter] = None,
        shortcut_name: str = "shortcut",
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
        inputs = {block_outputs_name: block, shortcut_name: shortcut}
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
        from merlin.models.tf.blocks.core.aggregation import SumResidual

        super().__init__(
            block,
            post=post,
            aggregation=SumResidual(activation=activation),
            schema=schema,
            name=name,
            strict=strict,
            **kwargs,
        )
