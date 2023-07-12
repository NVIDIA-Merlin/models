#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

from typing import Any, Callable, Optional, Sequence, Union

from torch import nn

from merlin.models.torch.block import Block
from merlin.models.torch.inputs.embedding import EmbeddingTables
from merlin.models.torch.router import RouterBlock
from merlin.models.torch.schema import Selection, select, select_union
from merlin.models.torch.transforms.sequences import BroadcastToSequence
from merlin.models.utils.registry import Registry
from merlin.schema import Schema, Tags

Initializer = Callable[["TabularInputBlock"], Any]


class TabularInputBlock(RouterBlock):
    """
    A block for handling tabular input data. This is a special type of block that
    can route data based on specified conditions, as well as perform initialization
    and aggregation operations.

    Example Usage::
        inputs = TabularInputBlock(init="defaults", agg="concat")

    Args:
        init (Optional[Union[str, Initializer]]): An initializer to apply to the block.
            This can be either a string (in which case it should be the name of
            an initializer in the registry), or a callable Initializer function.
        agg (Optional[Union[str, nn.Module]]): An aggregation module to append to the block.
    """

    """
    Registry of initializer functions. Initializers are functions that perform some form of
    initialization operation on a TabularInputBlock instance.
    """
    initializers = Registry("input-initializers")

    def __init__(
        self,
        schema: Optional[Schema] = None,
        init: Optional[Union[str, Initializer]] = None,
        agg: Optional[Union[str, nn.Module]] = None,
    ):
        self.init = init
        super().__init__(schema)
        if agg:
            self.append(Block.parse(agg))

    def initialize_from_schema(self, schema: Schema):
        super().initialize_from_schema(schema)
        self.schema: Schema = self.selectable.schema
        if self.init:
            if isinstance(self.init, str):
                self.init = self.initializers.get(self.init)
                if not self.init:
                    raise ValueError(f"Initializer {self.init} not found.")

            self.init(self)

    @classmethod
    def register_init(cls, name: str):
        """
        Class method to register an initializer function with the given name.

        Example Usage::
            @TabularInputBlock.register_init("defaults")
            def defaults(block: TabularInputBlock):
                block.add_route(Tags.CONTINUOUS)
                block.add_route(Tags.CATEGORICAL, EmbeddingTables())

            inputs = TabularInputBlock(init="defaults")

        Args:
            name (str): The name to assign to the initializer function.

        Returns:
            function: The decorator function used to register the initializer.
        """

        return cls.initializers.register(name)


@TabularInputBlock.register_init("defaults")
def defaults(block: TabularInputBlock, seq_combiner="mean"):
    """
    Default initializer function for a TabularInputBlock.

    This function adds routing for continuous and categorical data, with the categorical
    data being routed through an EmbeddingTables instance.

    Args:
        block (TabularInputBlock): The block to initialize.
    """
    block.add_route(Tags.CONTINUOUS, required=False)
    block.add_route(Tags.CATEGORICAL, EmbeddingTables(seq_combiner=seq_combiner))


@TabularInputBlock.register_init("defaults-no-combiner")
def defaults_no_combiner(block: TabularInputBlock):
    return defaults(block, seq_combiner=None)


@TabularInputBlock.register_init("broadcast-context")
def defaults_broadcast_to_seq(
    block: TabularInputBlock,
    seq_selection: Selection = Tags.SEQUENCE,
    feature_selection: Sequence[Selection] = (Tags.CATEGORICAL, Tags.CONTINUOUS),
):
    context_selection = _not_seq(seq_selection, feature_selection=feature_selection)
    block.add_route(context_selection, TabularInputBlock(init="defaults"), name="context")
    block.add_route(
        seq_selection,
        TabularInputBlock(init="defaults-no-combiner"),
        name="sequence",
    )
    block.append(BroadcastToSequence(context_selection, seq_selection, block.schema))


def stack_context(
    model_dim: int,
    seq_selection: Selection = Tags.SEQUENCE,
    projection_activation=None,
    feature_selection: Sequence[Selection] = (Tags.CATEGORICAL, Tags.CONTINUOUS),
):
    def init_stacked_context(block: TabularInputBlock):
        import merlin.models.torch as mm

        mlp_kwargs = {"units": [model_dim], "activation": projection_activation}
        context_selection = _not_seq(seq_selection, feature_selection=feature_selection)
        context = TabularInputBlock(select(block.schema, context_selection))
        context.add_route(Tags.CATEGORICAL, EmbeddingTables(seq_combiner=None))
        context.add_route(Tags.CONTINUOUS, mm.MLPBlock(**mlp_kwargs))
        context["categorical"].append_for_each(mm.MLPBlock(**mlp_kwargs))
        context.append(mm.Stack(dim=1))

        block.add_route(context.schema, context, name="context")
        block.add_route(
            seq_selection,
            TabularInputBlock(init="defaults-no-combiner", agg=mm.Concat(dim=2)),
            name="sequence",
        )

    return init_stacked_context


def _not_seq(
    seq_selection: Sequence[Selection],
    feature_selection: Sequence[Selection] = (Tags.CATEGORICAL, Tags.CONTINUOUS),
) -> Selection:
    if not isinstance(seq_selection, (tuple, list)):
        seq_selection = (seq_selection,)

    def select_non_seq(schema: Schema) -> Schema:
        seq = select_union(*seq_selection)(schema)
        features = select_union(*feature_selection)(schema)

        return features - seq

    return select_non_seq
