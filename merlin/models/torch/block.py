from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, Union

import torch
from rich.table import Table
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.container import BlockContainer, BlockContainerDict
from merlin.models.torch.link import Link
from merlin.models.torch.registry import registry
from merlin.schema import ColumnSchema, Schema, Tags


class Block(BlockContainer):
    def __init__(self, *module: nn.Module, name: Optional[str] = None):
        super().__init__(*module, name=name)

    @classmethod
    def from_registry(cls, name) -> "Block":
        if isinstance(name, str):
            if name not in registry:
                raise ValueError(f"Block {name} not found in registry")
            return registry.parse(name)

        raise ValueError(f"Block {name} is not a string")

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        for module in self.values:
            inputs = module(inputs, batch=batch)

        return inputs

    def repeat(self, n: int, name=None) -> "Block":
        repeats = [self.copy() for _ in range(n)]

        return Block(*repeats, name=name)

    def repeat_parallel(
        self,
        n: int,
        # agg=None,
        prefix="",
        # shortcut=False,
        name: Optional[str] = None,
    ) -> "ParallelBlock":
        branches = {}
        for i in range(n):
            branch = self.copy()
            branch.name = f"{prefix}_{i}"
            branches[branch.name] = branch

        return ParallelBlock(branches, name=name)

    def repeat_parallel_like(
        self, input: Union[Mapping[str, Any], List[str]], name: Optional[str] = None
    ) -> "ParallelBlock":
        branches = {}

        if isinstance(input, list):
            _names = input
        else:
            _names = input.keys()

        for name in _names:
            branch = self.copy()
            branch.name = name
            branches[branch.name] = branch

        return ParallelBlock(branches)

    def copy(self) -> "Block":
        return deepcopy(self)


class ParallelBlock(Block):
    def __init__(
        self,
        *inputs: Union[nn.Module, Dict[str, nn.Module]],
        # TODO: Add agg
    ):
        pre = BlockContainer(name="pre")
        branches = BlockContainerDict(*inputs, name="branches")
        post = BlockContainer(name="post")

        super().__init__(pre, branches, post)

        self.pre = pre
        self.branches = branches
        self.post = post

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        for module in self.pre.values:
            inputs = module(inputs, batch=batch)

        # TODO: DO we need to raise an exception when there are no branches?

        outputs = {}
        for name, module in self.branches.items():
            outputs[name] = module(inputs, batch=batch)

        for module in self.post.values:
            outputs = module(outputs, batch=batch)

        return outputs

    def append(self, module: nn.Module, link: Optional[Union[Link, str]] = None):
        self.post.append(module, link=link)

        return self

    def prepend(self, module: nn.Module, link: Optional[Union[Link, str]] = None):
        self.pre.prepend(module, link=link)

        return self

    def append_to(self, name: str, module: nn.Module, link: Optional[Union[Link, str]] = None):
        self.branches[name].append(module, link=link)

        return self

    def prepend_to(self, name: str, module: nn.Module, link: Optional[Union[Link, str]] = None):
        self.branches[name].prepend(module, link=link)

        return self

    def append_for_each(
        self,
        module: nn.Module,
        link: Optional[Union[Link, str]] = None,
        copy=False,  # alternative shared=False
    ):
        self.branches.append_for_each(module, link=link, copy=copy)

        return self

    def prepend_for_each(
        self, module: nn.Module, link: Optional[Union[Link, str]] = None, copy=False
    ):
        self.branches.prepend_for_each(module, link=link, copy=copy)

        return self

    def __getitem__(self, idx: Union[slice, int]):
        if isinstance(idx, str) and idx in self.branches:
            return self.branches[idx]

        return self.values[idx].unwrap()


Selection = Union[Schema, ColumnSchema, Callable[[Schema], Schema], Tags]


def select_schema(schema: Schema, selection: Selection) -> Schema:
    if isinstance(selection, Schema):
        selected = selection
    elif isinstance(selection, ColumnSchema):
        selected = schema[selection.name]
    elif callable(selection):
        selected = selection(schema)
    elif isinstance(selection, Tags):
        selected = schema.select_by_tag(selection)
    else:
        raise ValueError(f"Selection {selection} is not valid")

    return selected


def selection_name(selection: Selection) -> str:
    if isinstance(selection, ColumnSchema):
        return selection.name
    elif isinstance(selection, Tags):
        return selection.value
    elif isinstance(selection, callable):
        return selection.__name__
    elif isinstance(selection, Schema):
        return "_".join(selection.column_names)

    raise ValueError(f"Selection {selection} is not valid")


class Selectable:
    def setup_schema(self, schema: Schema):
        self.schema = schema

        return self

    def select(self, selection: Selection) -> "Selectable":
        raise NotImplementedError()


class RouterBlock(ParallelBlock, Selectable):
    def __init__(self, selectable: Selectable):
        super().__init__()
        self.selectable = selectable

    def add(
        self,
        selection: Selection,
        module: Optional[nn.Module] = None,
        name=None,
        encoder=False
        # TODO agg=None
    ) -> "RouterBlock":
        routing_module = self.selectable.select(selection)
        name = name or selection_name(selection)

        branch = Block(routing_module)
        if module is not None:
            if isinstance(module, ParallelBlock):
                branch = module.prepend(routing_module)

            if hasattr(module, "setup_schema"):
                module.setup_schema(routing_module.schema)

            if encoder:
                # branch.append(ServeableBlock(module))
                raise NotImplementedError()
            else:
                branch.append(module)
        else:
            branch = routing_module

        # TODO: Should we throw an exception if the name already exists?
        self.branches[name] = branch

        return self

    def add_for_each(
        self,
        selection: Selection,
        module: nn.Module,
        shared=False
        # TODO agg=None
    ) -> "RouterBlock":
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

    # def select_by_tag(
    #     self,
    #     tags: Union[str, Tags, List[Union[str, Tags]]],
    # ) -> Optional["ParallelBlock"]:
    #     raise NotImplementedError()

    # def select_by_name(self, names):
    #     raise NotImplementedError()


# Currently called: Filter
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

    # def extra_repr(self) -> str:
    #     return f"keys={self.col_names}"

    def __rich_repr__(self):
        table = Table(title=self._get_name(), title_justify="left")
        table.add_column("Column", justify="left", no_wrap=True)

        for col in self.col_names:
            table.add_row(col)

        return table


# inputs.add(Tags.CONTINUOUS, MLPBlock(100, 100))
# inputs.add(Tag.CATEGORICAL, Embeddings())


class MLPBlock(Block):
    def __init__(self, *hidden: Union[int, List[int]]):
        if len(hidden) == 1 and isinstance(hidden[0], list):
            hidden = hidden[0]

        super().__init__(*[nn.LazyLinear(h) for h in hidden])
        self.hidden = hidden

    def extra_repr(self) -> str:
        return f"hidden={self.hidden}"


class TabularInputBlock(RouterBlock):
    def __init__(self, schema: Schema, init=None, agg=None):
        if isinstance(schema, Schema):
            selectable = SelectKeys(schema)
        else:
            selectable = schema
        super().__init__(selectable)
        if init:
            if init == "defaults":
                init = defaults
            init(self)
        if agg:
            self.append(agg)

    def select(self, selection: Selection) -> "TabularInputBlock":
        selected_branches = {}
        for key, val in self.branches.items():
            if len(val) == 1:
                val = val[0]

            selected_branches[key] = val.select(selection)

        selectable = TabularInputBlock(select_schema(self.selectable.schema, selection))
        for key, val in selected_branches.items():
            selectable.branches[key] = val

        # TODO: Add pre/post
        return selectable


def defaults(inputs: TabularInputBlock):
    inputs.add(Tags.CONTINUOUS)
    inputs.add(Tags.CATEGORICAL, Embeddings())


class OutputBlock(RouterBlock):
    def __init__(self, schema, init=None, agg=None):
        super().__init__(SelectKeys(schema))


class Model(Block):
    def __init__(self, *module: nn.Module, name=None):
        super().__init__(*module, name=name)


TagOrStr = Union[Tags, str]


class ContrastiveOutput(Block):
    def __init__(self, cols: Union[TagOrStr, List[TagOrStr]], schema: Optional[Schema] = None):
        super().__init__()
        self.cols = cols
        self.schema = schema


# TODO: Do we still need the EncoderBlock?

DimFn = Callable[[ColumnSchema], int]


class EmbeddingTable(nn.Module, Selectable):
    def __init__(
        self, dim: Union[int, DimFn] = ..., schema: Optional[Union[ColumnSchema, Schema]] = None
    ):
        super().__init__()
        self.dim = dim
        if schema:
            self.setup_schema(schema)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def setup_schema(self, schema: Schema):
        if isinstance(schema, ColumnSchema):
            schema = Schema([schema])

        self.schema = schema

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.zeros(1, self.dim)


class Embeddings(ParallelBlock, Selectable):
    def __init__(
        self,
        dim: Optional[Union[Dict[str, int], int, DimFn]] = 100,
        schema: Optional[Schema] = None,
        table_cls: Type[nn.Module] = EmbeddingTable,
    ):
        super().__init__()
        self.dim = dim
        self.table_cls = table_cls
        if isinstance(schema, Schema):
            self.setup_schema(schema)

    def setup_schema(self, schema: Schema):
        self.schema = schema

        for col in schema:
            self.branches[col.name] = self.table_cls(self.dim, schema=col)

        return self

    def select(self, selection: Selection) -> "Embeddings":
        # TODO: Fix this
        return Embeddings(self.dim, schema=select_schema(self.schema, selection))

    def __rich_repr__(self):
        table = Table(title=self._get_name(), title_justify="left")
        table.add_column("Column", justify="left", no_wrap=True)
        table.add_column("Num", justify="left", no_wrap=True)
        table.add_column("Dim", justify="left", no_wrap=True)

        for emb_table in self.branches.values():
            features = ", ".join(emb_table[0].schema.column_names)
            num = emb_table[0].schema.first.int_domain.max + 1
            table.add_row(features, str(num), str(emb_table[0].dim))

        return table


class DLRMInteraction(nn.Module):
    ...


class ShortcutConcatContinuous(Link):
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        intermediate_output = self.output(inputs)

        return torch.cat((inputs["continuous"], intermediate_output), dim=1)


class DLRMInputBlock(TabularInputBlock):
    def __init__(
        self,
        schema,
        embedding_dim: Optional[int] = None,
        embeddings: Optional[nn.Module] = None,
        bottom_block: Optional[nn.Module] = None,
        top_block: Optional[nn.Module] = None,
        interaction_block=DLRMInteraction(),
        categorical_selection: Selection = Tags.CATEGORICAL,
        continuous_selection: Selection = Tags.CONTINUOUS,
    ):
        super().__init__(schema)

        interaction_link = None

        # Categorical
        if not embedding_dim or embeddings:
            raise ValueError("Must specify embedding_dim or embeddings")

        if not embeddings:
            embeddings = Embeddings(categorical_selection, dim=embedding_dim)
        self.add(categorical_selection, embeddings)

        # Continuous
        con_schema = self.select(continuous_selection)
        if con_schema:
            if bottom_block is None:
                raise ValueError("Must specify bottom_block if continuous features are present")

            self.add_encoder(continuous_selection, bottom_block, name="continuous")
            interaction_link = ShortcutConcatContinuous()

        # Interaction
        self.append(interaction_block, link=interaction_link)

        if top_block is not None:
            self.append(top_block)


# class ExpertGate(nn.Module):
#     def __init__(self, gate_pre: nn.Module, num_outputs: int, task_experts: Optional[nn.Module] = None):
#         super().__init__()
#         self.gate_pre = gate_pre
#         self.gating = nn.LazyLinear(num_outputs)
#         self.task_experts = task_experts

#     @classmethod
#     def for_each(
#         cls,
#         outputs: ParallelBlock,
#         gate_pre: nn.Module,
#         task_experts: Optional[nn.Module] = None
#     ) -> ParallelBlock:
#         return Block(cls(gate_pre, len(outputs), task_experts)).repeat_parallel_like(outputs)

#     def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
#         gate_logits = self.gating(self.gate_pre(x["shortcut"]))
#         gates = torch.softmax(gate_logits, dim=1).unsqueeze(2)  # (batch_size, num_experts, 1)

#         shared_experts = x["experts"]
#         if self.task_experts is not None:
#             task_experts_outputs = self.task_experts(x["shortcut"])
#             experts = torch.cat([shared_experts, task_experts_outputs], dim=1)
#         else:
#             experts = shared_experts

#         # Expand gates to match expert_outputs shape
#         gates = gates.expand_as(experts)

#         # Multiply and sum along the experts dimension
#         return (experts * gates).sum(dim=1)


# class MixtureOfExpertsBlock(Block):
#     def __init__(
#         self,
#         expert: nn.Module,
#         num_experts: int,
#         outputs: ParallelBlock,
#         pre_gate: nn.Module = MLPBlock([512])
#     ):
#         experts = ParallelBlock({
#             "experts": Block.parse(expert).repeat_parallel(num_experts, agg="stack")
#         }, shortcut=True)

#         # This will output create a ParallelBlock with a gate for each output
#         gates = Block(ExpertGate(pre_gate, len(outputs))).repeat_parallel_like(outputs)

#         super().__init__(experts, gates, outputs)

#     @classmethod
#     def from_schema(
#         cls,
#         schema: Schema,
#         expert: nn.Module,
#         num_experts: int,
#         pre_gate: nn.Module = MLPBlock([512])
#     ):
#         outputs = OutputBlock(schema)

#         return cls(expert, num_experts, outputs, pre_gate)


# class PLEBlock(Block):
#     def __init__(
#         self,
#         expert: nn.Module,
#         num_shared_experts: int,
#         num_task_experts: int,
#         outputs: ParallelBlock,
#         pre_gate: nn.Module = MLPBlock([512])
#     ):
#         shared_experts = ParallelBlock({
#             "experts": Block.parse(expert).repeat_parallel(num_shared_experts, agg="stack")
#         }, shortcut=True)
#         task_experts = Block.parse(expert).repeat_parallel(num_task_experts, agg="stack")
#         gates = ExpertGate.for_each(outputs, pre_gate, task_experts)

#         super().__init__(shared_experts, gates, outputs)


# outputs = OutputBlock(schema) # Block will output a dict of outputs
# outputs.prepend_for_each(MLPBlock([512, 256]), copy=True)

# class BinaryOutput(Block):
#     def __init__(self, pre: Optional[nn.Module] = None, post: Optional[nn.Module] = None):
#         module = nn.Sequential(nn.LazyLinear(1), nn.Sigmoid())

#         super().__init__(module, pre=pre, post=post)
#         self.register_buffer("target", torch.zeros(1, dtype=torch.float32))

#     def forward(self, inputs, batch: Optional[Batch] = None):
#         if self.training and batch is not None and "target" in batch.targets:
#             self.target = batch.targets["target"]

#         return self.module(inputs, batch=batch)

#     def eval(self):
#         # Reset target
#         self.target = torch.zeros(1, dtype=torch.float32)

#         return self.train(False)
