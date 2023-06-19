from copy import deepcopy
from typing import Optional

from torch import nn

from merlin.models.torch import schema
from merlin.models.torch.block import Block, ParallelBlock
from merlin.models.torch.container import BlockContainerDict
from merlin.schema import Schema


class RouterBlock(ParallelBlock):
    # class RouterBlock(ParallelBlock, Selectable):
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

    def __init__(self, selectable: schema.Selectable):
        super().__init__()
        if isinstance(selectable, Schema):
            from merlin.models.torch.inputs.select import SelectKeys

            selectable = SelectKeys(selectable)

        self.selectable: schema.Selectable = selectable

    def add_route(
        self,
        selection: schema.Selection,
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

        routing_module = schema.select(self.selectable, selection)
        if module is not None:
            schema.setup_schema(module, routing_module.schema)

            if isinstance(module, ParallelBlock):
                branch = module.prepend(routing_module)
            else:
                branch = Block(routing_module, module)
        else:
            branch = routing_module

        _name: str = name or schema.selection_name(selection)
        if _name in self.branches:
            raise ValueError(f"Branch with name {_name} already exists")
        self.branches[_name] = branch

        return self

    def add_route_for_each(
        self, selection: schema.Selection, module: nn.Module, shared=False
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
            col_module = module if shared else deepcopy(module)
            self.add_route(col, col_module, name=col.name)

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

        if hasattr(self, "_forward_called"):
            # We don't need to track the schema since we will be using the nested router
            self._handle.remove()

        return self.__class__(self)

    # def select(self, selection: Selection) -> "RouterBlock":
    #     """Select a subset of the branches based on the provided selection.

    #     Parameters
    #     ----------
    #     selection : Selection
    #         The selection to apply to the branches.

    #     Returns
    #     -------
    #     RouterBlock
    #         A new router block with the selected branches.
    #     """

    #     selected = select(self.selectable, selection)
    #     output = _select_parallel_block(self, selection)
    #     if output:
    #         if isinstance(selected, SelectKeys):
    #             selected_keys = selected
    #         else:
    #             selected_keys = SelectKeys(selected.schema)
    #         if not self.pre or (self.pre and self.pre[0] != selected_keys):
    #             if all(get_pre(self.branches[key]) for key in output.branches):
    #                 for key in output.branches:
    #                     pre = get_pre(self.branches[key])
    #                     if pre and pre[0] != selected_keys:
    #                         set_pre(output.branches[key],
    # BlockContainer(selected_keys, *self.pre))
    #             elif selected_keys not in list(output.branches.modules()):
    #                 output.pre = BlockContainer(selected_keys, *self.pre)

    #     return output

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
