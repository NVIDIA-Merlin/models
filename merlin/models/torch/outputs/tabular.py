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

from merlin.models.torch.outputs.classification import BinaryOutput, CategoricalOutput
from merlin.models.torch.outputs.regression import RegressionOutput
from merlin.models.torch.router import RouterBlock
from merlin.models.torch.schema import Selection, select
from merlin.models.utils.registry import Registry
from merlin.schema import Schema, Tags

Initializer = Callable[["TabularOutputBlock"], Any]


class TabularOutputBlock(RouterBlock):
    """
    A block for outputting tabular data. This is a special type of block that
    can route data based on specified conditions, as well as perform initialization
    and aggregation operations.

    Example Usage::
        inputs = TabularOutputBlock(init="defaults")

    Args:
        init (Optional[Union[str, Initializer]]): An initializer to apply to the block.
            This can be either a string (in which case it should be the name of
            an initializer in the registry), or a callable Initializer function.
    """

    """
    Registry of initializer functions. Initializers are functions that perform some form of
    initialization operation on a TabularInputBlock instance.
    """
    initializers = Registry("output-initializers")

    def __init__(
        self,
        schema: Optional[Schema] = None,
        init: Optional[Union[str, Initializer]] = None,
        selection: Optional[Selection] = Tags.TARGET,
    ):
        self.selection = selection
        self.init = init
        super().__init__(schema, prepend_routing_module=False)

    def initialize_from_schema(self, schema: Schema):
        if self.selection:
            schema = select(schema, self.selection)
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
            @TabularOutputBlock.register_init("some-name")
            def defaults(block: TabularOutputBlock):
                block.add_route_for_each([Tags.CONTINUOUS, Tags.REGRESSION], RegressionOutput())

            outputs = TabularOutputBlock(init="some-name")

        Args:
            name (str): The name to assign to the initializer function.

        Returns:
            function: The decorator function used to register the initializer.
        """

        return cls.initializers.register(name)


@TabularOutputBlock.register_init("defaults")
def defaults(block: TabularOutputBlock):
    """
    Default initializer function for a TabularOutputBlock.

    This function adds a route for each of the following tags:
        - Tags.CONTINUOUS/Tags.REGRESSION -> RegressionOutput
        - Tags.BINARY_CLASSIFICATION/Tags.BINARY -> BinaryOutput
        - Tags.MULTI_CLASS_CLASSIFICATION/Tags.CATEGORICAL -> CategoricalOutput

    Args:
        block (TabularOutputBlock): The block to initialize.
    """
    block.add_route_for_each([Tags.CONTINUOUS, Tags.REGRESSION], RegressionOutput(), required=False)
    block.add_route_for_each(BinaryOutput.schema_selection, BinaryOutput(), required=False)
    block.add_route_for_each(
        CategoricalOutput.schema_selection, CategoricalOutput(), required=False
    )
