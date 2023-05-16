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
    def __init__(self, selectable: Selectable):
        super().__init__()
        self.selectable = selectable

    def add(
        self,
        selection: Selection,
        module: Optional[nn.Module] = None,
        name: Optional[str] = None,
    ) -> "RouterBlock":
        routing_module = self.selectable.select(selection)
        branch = Block(routing_module)
        if module is not None:
            if isinstance(module, ParallelBlock):
                branch = module.prepend(routing_module)

            if hasattr(module, "setup_schema"):
                module.setup_schema(routing_module.schema)

            branch.append(module)
        else:
            branch = routing_module

        _name: str = name or selection_name(selection)
        if _name in self.branches:
            raise ValueError(f"Branch with name {_name} already exists")
        self.branches[_name] = branch

        return self

    def add_for_each(self, selection: Selection, module: nn.Module, shared=False) -> "RouterBlock":
        if isinstance(selection, (list, tuple)):
            for sel in selection:
                self.add_for_each(sel, module, shared=shared)

            return self

        selected = select_schema(self.selectable.schema, selection)

        for col in selected:
            col_module = module if shared else deepcopy(module)
            if hasattr(col_module, "setup_schema"):
                col_module.setup_schema(selected)

            self.add(col, col_module, name=col.name)

        return self

    def to_router(self) -> "RouterBlock":
        return RouterBlock(self)

    def select(self, selection: Selection) -> "RouterBlock":
        selected_branches = {}
        for key, val in self.branches.items():
            if len(val) == 1:
                val = val[0]

            selected_branches[key] = val.select(selection)

        selectable = self.__class__(self.selectable.select(selection))
        for key, val in selected_branches.items():
            selectable.branches[key] = val

        # TODO: Add pre/post
        return selectable


class SelectKeys(nn.Module, Selectable):
    def __init__(self, schema: Optional[Schema] = None):
        super().__init__()
        if schema:
            self.setup_schema(schema)

    def setup_schema(self, schema: Schema):
        if isinstance(schema, ColumnSchema):
            schema = Schema([schema])

        super().setup_schema(schema)

        self.col_names: List[str] = schema.column_names

    def select(self, selection: Selection) -> Selectable:
        return SelectKeys(select_schema(self.schema, selection))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = {}

        for key, val in inputs.items():
            if key in self.col_names:
                outputs[key] = val

        return outputs
