from copy import deepcopy
from typing import Dict, List, Optional

import torch
from torch import nn

from merlin.models.torch.block import Block, ParallelBlock
from merlin.models.torch.utils.selection_utils import (
    Selectable,
    Selection,
    select_schema,
    selection_name,
)
from merlin.schema import ColumnSchema, Schema


class RouterBlock(ParallelBlock, Selectable):
    """A block that routes features by selecting them from a selectable object.

    Example usage::

        router = RouterBlock(schema)
        router.add_route(Tags.CONTINUOUS)
        router.add_route(Tags.CATEGORICAL, mm.Embeddings(dim=64))
        router.add_route(Tags.EMBEDDING, mm.MLPBlock([64, 32]))

    Parameters
    ----------
    selectable : Selectable
        The selectable object from which to select features.

    Attributes
    ----------
    selectable : Selectable
        The selectable object from which to select features.
    """

    def __init__(self, selectable: Selectable):
        super().__init__()
        if isinstance(selectable, Schema):
            selectable = SelectKeys(selectable)

        self.selectable: Selectable = selectable

    def add_route(
        self,
        selection: Selection,
        module: Optional[nn.Module] = None,
        name: Optional[str] = None,
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

        Returns
        -------
        RouterBlock
            The router block with the new route added.
        """

        routing_module = self.selectable.select(selection)
        if module is not None:
            if hasattr(module, "setup_schema"):
                module.setup_schema(routing_module.schema)

            if isinstance(module, ParallelBlock):
                branch = module.prepend(routing_module)
            else:
                branch = Block(routing_module, module)
        else:
            branch = routing_module

        _name: str = name or selection_name(selection)
        if _name in self.branches:
            raise ValueError(f"Branch with name {_name} already exists")
        self.branches[_name] = branch

        return self

    def add_route_for_each(
        self, selection: Selection, module: nn.Module, shared=False
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

        selected = select_schema(self.selectable.schema, selection)

        for col in selected:
            col_module = module if shared else deepcopy(module)
            self.add_route(col, col_module, name=col.name)

        return self

    def nested_router(self) -> "RouterBlock":
        """Create a new nested router block.

        This method is useful for creating hierarchical routing structures.
        For example, you might want to route continuous and categorical features differently,
        and then within each of these categories, route user- and item-features differently.
        This can be achieved by calling `nested_router` to create a second level of routing.

        This approach allows for constructing networks with shared computation,
        such as shared embedding tables (like for instance user_genres and item_genres columns).
        This can improve performance and efficiency.

        Example usage::
            router = RouterBlock(selectable)
            # First level of routing: separate continuous and categorical features
            router.add_route(Tags.CONTINUOUS)
            router.add_route(Tags.CATEGORICAL, mm.Embeddings())

            # Second level of routing: separate user- and item-features
            two_tower = router.nested_router()
            two_tower.add_route(Tags.USER, mm.MLPBlock([64, 32]))
            two_tower.add_route(Tags.ITEM, mm.MLPBlock([64, 32]))

        Returns
        -------
        RouterBlock
            A new router block with the current block as its selectable.
        """

        if hasattr(self, "_forward_called"):
            # We don't need to track the schema since we will be using the nested router
            self._handle.remove()

        return RouterBlock(self)

    def select(self, selection: Selection) -> "RouterBlock":
        """Select a subset of the branches based on the provided selection.

        Parameters
        ----------
        selection : Selection
            The selection to apply to the branches.

        Returns
        -------
        RouterBlock
            A new router block with the selected branches.
        """

        selected_branches = {}
        for key, val in self.branches.items():
            if len(val) == 1:
                val = val[0]

            selected_branches[key] = val.select(selection)

        selectable = self.__class__(self.selectable.select(selection))
        for key, val in selected_branches.items():
            selectable.branches[key] = val

        selectable.pre = self.pre
        selectable.post = self.post

        return selectable


class SelectKeys(nn.Module, Selectable):
    """Filter tabular data based on a defined schema.

    Example usage::

        >>> select_keys = mm.SelectKeys(Schema(["user_id", "item_id"]))
        >>> inputs = {
        ...     "user_id": torch.tensor([1, 2, 3]),
        ...     "item_id": torch.tensor([4, 5, 6]),
        ...     "other_key": torch.tensor([7, 8, 9]),
        ... }
        >>> outputs = select_keys(inputs)
        >>> print(outputs.keys())
        dict_keys(['user_id', 'item_id'])

    Parameters
    ----------
    schema : Schema, optional
        The schema to use for selection. Default is None.

    Attributes
    ----------
    col_names : list
        List of column names in the schema.
    """

    def __init__(self, schema: Optional[Schema] = None):
        super().__init__()
        if schema:
            self.setup_schema(schema)

    def setup_schema(self, schema: Schema):
        if isinstance(schema, ColumnSchema):
            schema = Schema([schema])

        super().setup_schema(schema)

        self.col_names: List[str] = schema.column_names

    def select(self, selection: Selection) -> "SelectKeys":
        """Select a subset of the schema based on the provided selection.

        Parameters
        ----------
        selection : Selection
            The selection to apply to the schema.

        Returns
        -------
        SelectKeys
            A new SelectKeys instance with the selected schema.
        """

        return SelectKeys(select_schema(self.schema, selection))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Only keep the inputs that are present in the schema.

        Parameters
        ----------
        inputs : dict
            A dictionary of torch.Tensor objects.

        Returns
        -------
        dict
            A dictionary of torch.Tensor objects after selection.
        """

        outputs = {}

        for key, val in inputs.items():
            _key = key
            if key.endswith("__values"):
                _key = key[: -len("__values")]
            elif key.endswith("__offsets"):
                _key = key[: -len("__offsets")]

            if _key in self.col_names:
                outputs[key] = val

        return outputs
