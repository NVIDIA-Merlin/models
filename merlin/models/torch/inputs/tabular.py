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

from typing import Any, Callable, Optional, Union

from torch import nn

from merlin.models.torch.block import Block
from merlin.models.torch.inputs.embedding import EmbeddingTables
from merlin.models.torch.router import RouterBlock
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
    initializers = Registry("initializers")

    def __init__(
        self,
        schema: Schema,
        init: Optional[Union[str, Initializer]] = None,
        agg: Optional[Union[str, nn.Module]] = None,
    ):
        super().__init__(schema)
        self.schema: Schema = self.selectable.schema
        if init:
            if isinstance(init, str):
                init = self.initializers.get(init)
                if not init:
                    raise ValueError(f"Initializer {init} not found.")

            init(self)
        if agg:
            self.append(Block.parse(agg))

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
def defaults(block: TabularInputBlock):
    """
    Default initializer function for a TabularInputBlock.

    This function adds routing for continuous and categorical data, with the categorical
    data being routed through an EmbeddingTables instance.

    Args:
        block (TabularInputBlock): The block to initialize.
    """
    block.add_route(Tags.CONTINUOUS)
    block.add_route(Tags.CATEGORICAL, EmbeddingTables(seq_combiner="mean"))
