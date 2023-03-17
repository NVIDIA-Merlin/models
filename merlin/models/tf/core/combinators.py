import copy
import sys
from functools import reduce
from typing import Dict, List, Optional, Union

import six
import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.tf.core.base import (
    Block,
    BlockType,
    NoOp,
    PredictionOutput,
    is_input_block,
    right_shift_layer,
)
from merlin.models.tf.core.tabular import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    Filter,
    TabularAggregationType,
    TabularBlock,
)
from merlin.models.tf.utils import tf_utils
from merlin.models.tf.utils.tf_utils import call_layer
from merlin.models.utils import schema_utils
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.schema import Schema, Tags


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
        """Create a sequential composition.

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

        super(SequentialBlock, self).__init__(**kwargs)

        if getattr(layers[0], "has_schema", None):
            super().set_schema(layers[0].schema)

            for layer in layers[1:]:
                if hasattr(layer, "set_schema"):
                    layer.set_schema(layers[0].schema)

        layers = copy.copy(layers) if copy_layers else layers
        if filter:
            if not isinstance(filter, Filter):
                filter = Filter(filter)
            self.layers = [filter, *layers]
        else:
            self.layers = list(layers)

    def compute_output_shape(self, input_shape):
        """Computes the output shape based on the input shape

        Parameters
        ----------
        input_shape : tf.TensorShape
            The input shape

        Returns
        -------
        tf.TensorShape
            The output shape
        """
        return compute_output_shape_sequentially(self.layers, input_shape)

    def compute_output_signature(self, input_signature):
        return compute_output_signature_sequentially(self.layers, input_signature)

    def build(self, input_shape=None):
        """Builds the sequential block

        Parameters
        ----------
        input_shape : tf.TensorShape, optional
            The input shape, by default None
        """
        self._maybe_propagate_context(input_shape)
        build_sequentially(self, self.layers, input_shape)

    def set_schema(self, schema=None):
        for layer in self.layers:
            self._maybe_set_schema(layer, schema)

        return super().set_schema(schema)

    def _get_name(self):
        return self.block_name if self.block_name else f"{self.__class__.__name__}"

    @property
    def inputs(self):
        """Returns the InputBlock, if it is the first
        block within SequenceBlock

        Returns
        -------
        InputBlock
            The input block
        """
        first = list(self)[0]
        if isinstance(first, SequentialBlock):
            return first.inputs
        if is_input_block(first):
            return first

    @property
    def first(self):
        """Returns the first block in the SequenceBlock

        Returns
        -------
        Block
            The first block of SequenceBlock
        """
        return self.layers[0]

    @property
    def last(self):
        """Returns the last block in the SequenceBlock

        Returns
        -------
        Block
            The last block of SequenceBlock
        """
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
        """Returns trainable weights of all layers
        of this block

        Returns
        -------
        List
            List with trainable weights
        """
        if not self.trainable:
            return []
        weights = {}
        for layer in self.layers:
            for v in layer.trainable_weights:
                weights[id(v)] = v
        return list(weights.values())

    @property
    def non_trainable_weights(self):
        """Returns non-trainable weights of all layers
        of this block

        Returns
        -------
        List
            List with non-trainable weights
        """
        weights = {}
        for layer in self.layers:
            for v in layer.non_trainable_weights:
                weights[id(v)] = v
        return list(weights.values())

    @property
    def trainable(self):
        """Returns whether all layer within SequentialBlock are trainable

        Returns
        -------
        bool
            True if any layer within SequentialBlock are trainable, otherwise False
        """
        return any(layer.trainable for layer in self.layers)

    @trainable.setter
    def trainable(self, value):
        """Makes all block layers trainable or not

        Parameters
        ----------
        value : bool
            Sets all layers trainable flag
        """
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
            regularizers = getattr(layer, "regularizers", None)
            if regularizers:
                values.update(regularizers)
        return list(values)

    def call(self, inputs, training=False, **kwargs):
        return call_sequentially(self.layers, inputs, training=training, **kwargs)

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


@docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING)
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ParallelBlock(TabularBlock):
    """Merge multiple layers or TabularModule's into a single output of TabularData.

    Parameters
    ----------
    inputs: Union[tf.keras.layers.Layer, Dict[str, tf.keras.layers.Layer]]
        keras layers to merge into, this can also be one or multiple layers keyed by the
        name the module should have.
    {tabular_module_parameters}
    use_layer_name: use the original name of layers provided in inputs as key-index of the
        parallel branches.
    strict:
        If true, inputs must be a dictionary. Otherwise, an error will be raised.
    automatic_pruning:
        If true, branches with no output will automatically be pruned.
    **kwargs:
        Extra arguments to pass to TabularBlock.
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
        automatic_pruning: bool = True,
        use_layer_name: bool = True,
        **kwargs,
    ):
        super().__init__(
            pre=pre, post=post, aggregation=aggregation, schema=schema, name=name, **kwargs
        )
        self.strict = strict
        self.automatic_pruning = automatic_pruning
        self.parallel_layers: Union[List[TabularBlock], Dict[str, TabularBlock]]
        if isinstance(inputs, tuple) and len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = inputs[0]
        if all(isinstance(x, dict) for x in inputs):
            to_merge: Dict[str, tf.keras.layers.Layer] = reduce(
                lambda a, b: dict(a, **b), inputs
            )  # type: ignore
            parsed_to_merge: Dict[str, TabularBlock] = {}
            for key, val in to_merge.items():
                parsed_to_merge[key] = val
            self.parallel_layers = parsed_to_merge
        elif all(isinstance(x, tf.keras.layers.Layer) for x in inputs):
            if use_layer_name:
                self.parallel_layers = {layer.name: layer for layer in inputs}
            else:
                parsed: List[TabularBlock] = []
                for i, inp in enumerate(inputs):
                    parsed.append(inp)  # type: ignore
                self.parallel_layers = parsed
        else:
            raise ValueError(
                "Please provide one or multiple layer's to merge or "
                f"dictionaries of layer. got: {inputs}"
            )

        if schema:
            for branch in self.parallel_values:
                if not getattr(branch, "has_schema", True):
                    branch.set_schema(schema)

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
    def schema(self):
        if self.has_schema:
            return self._schema

        if all(getattr(m, "_schema", False) for m in self.parallel_values):
            if len(self.parallel_values) == 1:
                return self.parallel_values[0].schema
            else:
                s = reduce(
                    lambda a, b: a + b, [m.schema for m in self.parallel_values]
                )  # type: ignore

                return s

        return None

    @property
    def parallel_values(self) -> List[tf.keras.layers.Layer]:
        if isinstance(self.parallel_layers, dict):
            return list(self.parallel_layers.values())

        return self.parallel_layers

    @property
    def parallel_dict(self) -> Dict[Union[str, int], tf.keras.layers.Layer]:
        if isinstance(self.parallel_layers, dict):
            return self.parallel_layers

        return {i: m for i, m in enumerate(self.parallel_layers)}

    @property
    def layers(self) -> List[tf.keras.layers.Layer]:
        return self.parallel_values

    def select_by_name(self, name: str) -> Optional["Block"]:
        """Select a parallel block by name

        Returns
        -------
        Block
            The block corresponding to the name
        """
        return self.parallel_dict.get(name)

    def select_by_names(self, names: List[str]) -> Optional[List[Block]]:
        """Select a list of parallel blocks by names

        Returns
        -------
        List[Block]
            The blocks corresponding to the names
        """
        blocks = []
        for name in names:
            if name in self.parallel_dict:
                blocks.append(self.parallel_dict.get(name))
            else:
                raise ValueError(f"Given name {name} is not in ParallelBlock {self.name}")
        return blocks

    def select_by_tag(
        self,
        tags: Union[str, Tags, List[Union[str, Tags]]],
    ) -> Optional["ParallelBlock"]:
        """Select layers of parallel blocks by tags.

        This method will return a ParallelBlock instance with all the branches that
        have at least one feature that matches any of the tags provided.

        For example, this method can be useful when a ParallelBlock has both item and
        user features in a two-tower model or DLRM, and we want to select only the item
        or user features.

        >>> all_inputs = InputBlockV2(schema)  # InputBlock is also a ParallelBlock
        >>> item_inputs = all_inputs.select_by_tag(Tags.ITEM)
        ['continuous', 'embeddings']
        >>> item_inputs.schema["continuous"].column_names
        ['item_recency']
        >>> item_inputs.schema["embeddings"].column_names
        ['item_id', 'item_category', 'item_genres']

        Parameters
        ----------
        tags: str or Tags or List[Union[str, Tags]]
             List of tags that describe which blocks to match

        Returns
        -------
        ParallelBlock
        """

        if self.schema is not None and self.schema == self.schema.select_by_tag(tags):
            return self

        if not isinstance(tags, (list, tuple)):
            tags = [tags]

        selected_branches = {}
        selected_schemas = Schema()

        for name, branch in self.parallel_dict.items():
            branch_has_schema = getattr(branch, "has_schema", False)
            if not branch_has_schema:
                continue
            if not hasattr(branch, "select_by_tag"):
                raise AttributeError(
                    f"This ParallelBlock does not support select_by_tag because "
                    f"{branch.__class__} does not support select_by_tag. Consider "
                    "implementing a select_by_tag in an extension of "
                    f"{branch.__class__}."
                )
            selected_branch = branch.select_by_tag(tags)
            if not selected_branch:
                continue
            selected_branches[name] = selected_branch
            selected_schemas += selected_branch.schema

        if not selected_branches:
            return
        return ParallelBlock(
            selected_branches,
            schema=selected_schemas,
            is_input=self.is_input,
            post=self.post,
            pre=self.pre,
            aggregation=self.aggregation,
            strict=self.strict,
            automatic_pruning=self.automatic_pruning,
        )

    def __getitem__(self, key) -> "Block":
        return self.parallel_dict[key]

    def __setitem__(self, key: str, item: "Block"):
        self.parallel_dict[key] = item

    @property
    def first(self) -> "Block":
        return self.parallel_values[0]

    def add_branch(self, name: str, block: "Block") -> "ParallelBlock":
        if isinstance(self.parallel_layers, dict):
            self.parallel_layers[name] = block

        return self

    def apply_to_branch(self, branch_name: str, *block: "Block"):
        if isinstance(self.parallel_layers, dict):
            self.parallel_layers[branch_name] = self.parallel_layers[branch_name].apply(*block)

    def call(self, inputs, **kwargs):
        """The call method for ParallelBlock

        Parameters
        ----------
        inputs : TabularData
            The inputs for the Parallel Block

        Returns
        -------
        TabularData
            Outputs of the ParallelBlock
        """
        if self.strict:
            assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {}

        for name, layer in self.parallel_dict.items():
            layer_inputs = self._maybe_filter_layer_inputs_using_schema(name, layer, inputs)
            out = call_layer(layer, layer_inputs, **kwargs)
            if not isinstance(out, dict):
                out = {name: out}
            outputs.update(out)

        return outputs

    def compute_call_output_shape(self, input_shape):
        output_shapes = {}

        for name, layer in self.parallel_dict.items():
            layer_input_shape = self._maybe_filter_layer_inputs_using_schema(
                name, layer, input_shape
            )
            out = layer.compute_output_shape(layer_input_shape)
            if isinstance(out, dict):
                output_shapes.update(out)
            else:
                output_shapes[name] = out

        return output_shapes

    def build(self, input_shape):
        to_prune = []

        for name, layer in self.parallel_dict.items():
            layer_input_shape = self._maybe_filter_layer_inputs_using_schema(
                name, layer, input_shape
            )
            layer.build(layer_input_shape)
            layer_out_shape = layer.compute_output_shape(layer_input_shape)
            if self.automatic_pruning and layer_out_shape == {}:
                to_prune.append(name)

        if isinstance(self.parallel_layers, dict):
            pruned = {}
            for name, layer in self.parallel_layers.items():
                if name not in to_prune:
                    pruned[name] = layer
            self.parallel_layers = pruned
        else:
            pruned = []
            for layer in self.parallel_layers:
                if layer not in to_prune:
                    pruned.append(layer)
            self.parallel_layers = pruned

        return super().build(input_shape)

    def _maybe_filter_layer_inputs_using_schema(self, name, layer, inputs):
        maybe_schema = getattr(layer, "_schema", None)
        if maybe_schema and isinstance(inputs, dict):
            layer_inputs = {
                k: v
                for k, v in inputs.items()
                if k.replace("__values", "").replace("__offsets", "") in maybe_schema.column_names
            }
        else:
            layer_inputs = inputs

        if isinstance(layer_inputs, dict) and all(
            name in layer_inputs for name in self.parallel_dict
        ):
            layer_inputs = layer_inputs[name]

        return layer_inputs

    def get_config(self):
        config = super(ParallelBlock, self).get_config()
        config.update({"automatic_pruning": self.automatic_pruning})

        return tf_utils.maybe_serialize_keras_objects(self, config, ["parallel_layers"])

    @classmethod
    def parse_config(cls, config, custom_objects=None):
        config = tf_utils.maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        if "schema" in config:
            config["schema"] = schema_utils.tensorflow_metadata_json_to_schema(config["schema"])

        parallel_layers = config.pop("parallel_layers")
        if isinstance(parallel_layers, dict):
            inputs = {
                name: tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
                for name, conf in parallel_layers.items()
            }
        elif isinstance(parallel_layers, (list, tuple)):
            inputs = [
                tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
                for conf in parallel_layers
            ]
        else:
            raise ValueError("Parallel layers need to be a list or a dict")

        return inputs, config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        inputs, config = cls.parse_config(config, custom_objects)

        return cls(inputs, **config)


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
        from merlin.models.tf.core.aggregation import SumResidual

        super().__init__(
            block,
            post=post,
            aggregation=SumResidual(activation=activation),
            schema=schema,
            name=name,
            strict=strict,
            **kwargs,
        )


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class Cond(Layer):
    """Layer to enable conditionally apply layers."""

    def __init__(self, condition: Layer, true: Layer, false: Optional[Layer] = None, **kwargs):
        super(Cond, self).__init__(**kwargs)
        self.condition = condition
        self.true = true
        self.false = false

    def call(self, inputs, **kwargs):
        """Call layers conditionally."""
        condition = call_layer(self.condition, inputs, **kwargs)

        def true_fn():
            return call_layer(self.true, inputs, **kwargs)

        def false_fn():
            if self.false is None:
                return inputs
            return call_layer(self.false, inputs, **kwargs)

        return tf.cond(tf.convert_to_tensor(condition), true_fn, false_fn)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer."""
        true_output_shape = self.true.compute_output_shape(input_shape)

        if self.false:
            false_output_shape = self.false.compute_output_shape(input_shape)
        else:
            false_output_shape = input_shape

        try:
            if isinstance(true_output_shape, dict):
                for key in true_output_shape.keys():
                    true_output_shape[key].assert_is_compatible_with(false_output_shape[key])
            else:
                true_output_shape.assert_is_compatible_with(false_output_shape)
        except ValueError as exc:
            raise ValueError(
                "Both true and false branches must return the same output shape"
            ) from exc

        return true_output_shape

    def get_config(self):
        """Returns the config of the layer as a Python dictionary."""
        config = super(Cond, self).get_config()
        config["condition"] = tf.keras.layers.serialize(self.condition)
        config["true"] = tf.keras.layers.serialize(self.true)
        if self.false:
            config["false"] = tf.keras.layers.serialize(self.false)
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a Cond layer from its config. Returning the instance."""
        condition = tf.keras.layers.deserialize(config.pop("condition"))
        true = tf.keras.layers.deserialize(config.pop("true"))
        false = None
        if "false" in config:
            false = tf.keras.layers.deserialize(config.pop("false"))
        return cls(condition, true, false=false, **config)

    def build(self, input_shape):
        """Creates the variables of the layer."""
        self.condition.build(input_shape)
        self.true.build(input_shape)
        if self.false:
            self.false.build(input_shape)
        return super(Cond, self).build(input_shape)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class MapValues(Layer):
    """Layer to map values of a dictionary of tensors."""

    def __init__(self, layer: Layer, **kwargs):
        super(MapValues, self).__init__(**kwargs)
        self.layer = layer

    def call(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return {key: call_layer(self.layer, value, **kwargs) for key, value in inputs.items()}

        return call_layer(self.layer, inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, dict):
            return {
                key: self.layer.compute_output_shape(value) for key, value in input_shape.items()
            }

        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        config = super(MapValues, self).get_config()
        config["layer"] = tf.keras.layers.serialize(self.layer)
        return config

    @classmethod
    def from_config(cls, config):
        layer = tf.keras.layers.deserialize(config.pop("layer"))
        return cls(layer, **config)


def call_sequentially(layers, inputs, **kwargs):
    """Call layers sequentially."""

    outputs = inputs
    for layer in layers:
        outputs = call_layer(layer, outputs, **kwargs)

    return outputs


def build_sequentially(self, layers, input_shape):
    """Build layers sequentially."""
    last_layer = None
    for layer in layers:
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


def compute_output_signature_sequentially(layers, input_signature):
    """Compute output signature sequentially."""
    output_signature = input_signature
    for layer in layers:
        output_signature = layer.compute_output_signature(output_signature)

    return output_signature


def compute_output_shape_sequentially(layers, input_shape):
    """Compute output shape sequentially."""
    output_shape = input_shape
    for layer in layers:
        output_shape = layer.compute_output_shape(output_shape)

    return output_shape
