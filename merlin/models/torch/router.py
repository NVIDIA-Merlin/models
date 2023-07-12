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

from copy import deepcopy
from inspect import isclass
from typing import Optional

from torch import nn

from merlin.models.torch import schema
from merlin.models.torch.block import Block, ParallelBlock
from merlin.models.torch.container import BlockContainerDict
from merlin.schema import Schema


class RouterBlock(ParallelBlock):
    """A block that routes features by selecting them from a selectable object.

    Example usage::

        router = RouterBlock(schema)
        router.add_route(Tags.CONTINUOUS)
        router.add_route(Tags.CATEGORICAL, mm.Embeddings(dim=64))
        router.add_route_for_each(Tags.EMBEDDING, mm.MLPBlock([64, 32]))

    Parameters
    ----------
    selectable : Selectable
        The selectable object from which to select features.

    Attributes
    ----------
    selectable : Selectable
        The selectable object from which to select features.
    """

    def __init__(self, selectable: schema.Selectable, prepend_routing_module: bool = True):
        super().__init__()
        self.prepend_routing_module = prepend_routing_module
        if isinstance(selectable, Schema):
            self.initialize_from_schema(selectable)
        else:
            self.selectable: schema.Selectable = selectable

    def initialize_from_schema(self, schema):
        from merlin.models.torch.inputs.select import SelectKeys

        self.selectable = SelectKeys(schema)

    def add_route(
        self,
        selection: schema.Selection,
        module: Optional[nn.Module] = None,
        name: Optional[str] = None,
        required: bool = True,
    ) -> "RouterBlock":
        """Add a new routing path for a given selection.

        Example usage::

            router.add_route(Tags.CONTINUOUS)

        Example usage with module::

            router.add_route(Tags.CONTINUOUS, MLPBlock([64, 32]]))

        Parameters
        ----------
        selection : Selection
            The selection to apply to the selectable.
        module : nn.Module, optional
            The module to append to the branch after selection.
        name : str, optional
            The name of the branch. Default is the name of the selection.
        required : bool, optional
            Whether the route is required. Default is True.

        Returns
        -------
        RouterBlock
            The router block with the new route added.
        """

        if self.selectable is None:
            raise ValueError(f"{self} has nothing to select from, so cannot add route.")

        routing_module = schema.select(self.selectable, selection)
        if not routing_module:
            if required:
                raise ValueError(f"Selection {selection} not found in {self.selectable}")

            return self

        if module is not None:
            schema.initialize_from_schema(module, routing_module.schema)

            if self.prepend_routing_module:
                if isinstance(module, ParallelBlock):
                    branch = module.prepend(routing_module)
                else:
                    branch = Block(routing_module, module)
            else:
                branch = module
        else:
            if self.prepend_routing_module:
                if not routing_module:
                    return self
                branch = routing_module
            else:
                raise ValueError("Must provide a module.")

        _name: str = name or schema.selection_name(selection)
        if _name in self.branches:
            raise ValueError(f"Branch with name {_name} already exists")
        self.branches[_name] = branch

        return self

    def add_route_for_each(
        self,
        selection: schema.Selection,
        module: nn.Module,
        shared=False,
        required: bool = True,
    ) -> "RouterBlock":
        """Add a new route for each column in a selection.

        Example usage::

            router.add_route_for_each(Tags.EMBEDDING, mm.MLPBlock([64, 32]]))

        Parameters
        ----------
        selection : Selection
            The selections to apply to the selectable.
        module : nn.Module
            The module to append to each branch after selection.
        shared : bool, optional
            Whether to use the same module instance for each selection.

        Returns
        -------
        RouterBlock
            The router block with the new routes added.
        """

        if isinstance(selection, (list, tuple)):
            for sel in selection:
                self.add_route_for_each(sel, module, shared=shared)

            return self

        selected = schema.select(self.selectable.schema, selection)

        for col in selected:
            if shared:
                col_module = module
            else:
                if isclass(module):
                    col_module = module(col)
                elif hasattr(module, "copy"):
                    col_module = module.copy()
                else:
                    col_module = deepcopy(module)

            self.add_route(col, col_module, name=col.name, required=required)

        return self

    def reroute(self) -> "RouterBlock":
        """Create a new nested router block.

        This method is useful for creating hierarchical routing structures.
        For example, you might want to route continuous and categorical features differently,
        and then within each of these categories, route user- and item-features differently.
        This can be achieved by calling `reroute` to create a second level of routing.

        This approach allows for constructing networks with shared computation,
        such as shared embedding tables (like for instance user_genres and item_genres columns).
        This can improve performance and efficiency.

        Example usage::
            router = RouterBlock(selectable)
            # First level of routing: separate continuous and categorical features
            router.add_route(Tags.CONTINUOUS)
            router.add_route(Tags.CATEGORICAL, mm.Embeddings())

            # Second level of routing: separate user- and item-features
            two_tower = router.reroute()
            two_tower.add_route(Tags.USER, mm.MLPBlock([64, 32]))
            two_tower.add_route(Tags.ITEM, mm.MLPBlock([64, 32]))

        Returns
        -------
        RouterBlock
            A new router block with the current block as its selectable.
        """

        return self.__class__(self)

    def replace(self, pre=None, branches=None, post=None, selection=None) -> "RouterBlock":
        if isinstance(branches, dict):
            branches = BlockContainerDict(branches)

        selected = schema.select(self.selectable.schema, selection)
        if not selected:
            return ParallelBlock()

        output = self.__class__(selected)
        output.pre = pre if pre is not None else self.pre
        output.branches = branches if branches is not None else self.branches
        output.post = post if post is not None else self.post

        return output
