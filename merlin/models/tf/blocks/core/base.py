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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Sequence, Type, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.config.schema import SchemaMixin
from merlin.models.tf.typing import TabularData
from merlin.models.utils.registry import Registry
from merlin.schema import Schema, Tags

block_registry: Registry = Registry.class_registry("tf.blocks")
BlockType = Union["Block", str, Sequence[str]]


if TYPE_CHECKING:
    from merlin.models.tf.blocks.core.combinators import (
        Filter,
        ParallelBlock,
        SequentialBlock,
        TabularAggregationType,
    )
    from merlin.models.tf.prediction_tasks.base import PredictionTask


class PredictionOutput(NamedTuple):
    predictions: Union[TabularData, tf.Tensor]
    targets: Union[TabularData, tf.Tensor]
    positive_item_ids: Optional[tf.Tensor] = None
    label_relevant_counts: Optional[tf.Tensor] = None
    valid_negatives_mask: Optional[tf.Tensor] = None
    negative_item_ids: Optional[tf.Tensor] = None

    def copy_with_updates(
        self,
        predictions: Optional[Union[TabularData, tf.Tensor]] = None,
        targets: Optional[Union[TabularData, tf.Tensor]] = None,
        positive_item_ids: Optional[tf.Tensor] = None,
        label_relevant_counts: Optional[tf.Tensor] = None,
        valid_negatives_mask: Optional[tf.Tensor] = None,
        negative_item_ids: Optional[tf.Tensor] = None,
    ):
        """Creates a new instance of PredictionOutput
        allowing to override the attributes for the copy
        """
        output = PredictionOutput(
            predictions=(self.predictions if predictions is None else predictions),
            targets=(self.targets if targets is None else targets),
            positive_item_ids=(
                self.positive_item_ids if positive_item_ids is None else positive_item_ids
            ),
            label_relevant_counts=(
                self.label_relevant_counts
                if label_relevant_counts is None
                else label_relevant_counts
            ),
            valid_negatives_mask=(
                self.valid_negatives_mask if valid_negatives_mask is None else valid_negatives_mask
            ),
            negative_item_ids=(
                self.negative_item_ids if negative_item_ids is None else negative_item_ids
            ),
        )
        return output


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ModelContext(Layer):
    """ModelContext is used to store/retrieve public variables across blocks.

    (This is created automatically in the model and doesn't need to be created manually.)
    """

    def add_embedding_weight(self, name, **kwargs):
        table = self.add_weight(name=f"{str(name)}/embedding", **kwargs)

        return table

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

    def get_config(self):
        config = super(ModelContext, self).get_config()

        return config


class ContextMixin:
    @property
    def context(self) -> ModelContext:
        return self._context

    def _set_context(self, context: ModelContext):
        if hasattr(self, "_context"):
            context._merge(self._context)
        self._context: ModelContext = context


class Block(SchemaMixin, ContextMixin, Layer):
    """Core abstraction in Merlin models."""

    registry = block_registry

    def __init__(self, context: Optional[ModelContext] = None, **kwargs):
        super(Block, self).__init__(**kwargs)
        if context:
            self._set_context(context)

    @classmethod
    @tf.autograph.experimental.do_not_convert
    def parse(cls, *block: BlockType) -> "Block":
        if len(block) == 1 and isinstance(block[0], (list, tuple)):
            block = block[0]  # type: ignore

        output: "Block"
        if len(block) == 1:
            output = cls.registry.parse(block[0])
        else:
            blocks = [cls.registry.parse(b) for b in block]
            output = blocks[0].connect(*blocks[1:])

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

    def call_outputs(
        self, outputs: PredictionOutput, training=False, **kwargs
    ) -> "PredictionOutput":
        return outputs

    def register_features(self, feature_shapes) -> List[str]:
        return []

    def as_tabular(self, name=None) -> "Block":
        from merlin.models.tf.blocks.core.combinators import SequentialBlock
        from merlin.models.tf.blocks.core.tabular import AsTabular

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
        from merlin.models.tf.blocks.core.combinators import SequentialBlock

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
        from merlin.models.tf.blocks.core.combinators import SequentialBlock, TabularBlock

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
        from merlin.models.tf.blocks.core.base import NoOp
        from merlin.models.tf.blocks.core.combinators import ParallelBlock

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
        context: Optional[ModelContext] = None,
    ) -> "SequentialBlock":
        """Connect the block to other blocks sequentially.

        Parameters
        ----------
        block: Union[tf.keras.layers.Layer, str]
            Blocks to connect to.
        block_name: str
            Name of the block.
        context: Optional[ModelContext]
            Context to use for the block.

        """
        from merlin.models.tf.blocks.core.combinators import SequentialBlock

        blocks = [self.parse(b) for b in block]

        for b in blocks:
            if isinstance(b, Block):
                if not b.has_schema and self.has_schema:
                    b._schema = self.schema

        output = SequentialBlock(
            [self, *blocks], copy_layers=False, block_name=block_name, context=context
        )

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
        from merlin.models.tf.blocks.core.combinators import ResidualBlock, SequentialBlock

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
        from merlin.models.tf.blocks.core.combinators import SequentialBlock, WithShortcut

        _block = self.parse(block) if not isinstance(block, Block) else block
        residual_block = WithShortcut(
            _block,
            shortcut_filter=shortcut_filter,
            post=post,
            aggregation=aggregation,
            block_outputs_name=block_outputs_name,
            automatic_pruning=False,
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
        from merlin.models.tf.blocks.core.base import Debug
        from merlin.models.tf.blocks.core.combinators import SequentialBlock

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
    ) -> "SequentialBlock":
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
        from merlin.models.tf.blocks.core.combinators import Filter, ParallelBlock, SequentialBlock

        _branches = [self.parse(b) for b in branches]

        all_features = []
        for branch in _branches:
            if getattr(branch, "set_schema", None):
                branch.set_schema(self.schema)
            if isinstance(branch, SequentialBlock):
                filter_features = branch.filter_features
                if filter_features:
                    all_features.extend(filter_features)

        if add_rest:
            if not self.schema:
                raise ValueError("Schema is required to add rest features.")
            rest_features = self.schema.without(list(set([str(f) for f in all_features])))
            rest_block = SequentialBlock([Filter(rest_features)])
            _branches.append(rest_block)

        return SequentialBlock(
            [self, ParallelBlock(*_branches, post=post, aggregation=aggregation, **kwargs)]
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
            self._need_to_call_context = True
            self.context.build(input_shapes)

    def __rrshift__(self, other):
        return right_shift_layer(self, other)

    def get_config(self):
        config = super().get_config()
        return config


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


def name_fn(name, inp):
    return "/".join([name, inp]) if name else None


MetricOrMetricClass = Union[tf.keras.metrics.Metric, Type[tf.keras.metrics.Metric]]
MetricOrMetrics = Union[Sequence[MetricOrMetricClass], MetricOrMetricClass]


@dataclass
class EmbeddingWithMetadata:
    embeddings: tf.Tensor
    metadata: Dict[str, tf.Tensor]


def is_input_block(block: Block) -> bool:
    is_defined = True if block else False
    return is_defined and getattr(block, "is_input", False)


def has_input_block(block: Block) -> bool:
    from merlin.models.tf.blocks.core.combinators import SequentialBlock

    if isinstance(block, SequentialBlock):
        return block.inputs is not None and is_input_block(block.inputs)
    return is_input_block(block.inputs)


def _output_metrics(metrics):
    if len(metrics) == 1:
        return metrics[list(metrics.keys())[0]]

    return metrics


def right_shift_layer(self, other):
    from merlin.models.tf.blocks.core.combinators import Filter, SequentialBlock

    if isinstance(other, (list, Tags)):
        left_side = [Filter(other)]
    else:
        left_side = other.layers if isinstance(other, SequentialBlock) else [other]
    right_side = self.layers if isinstance(self, SequentialBlock) else [self]

    return SequentialBlock(left_side + right_side)
