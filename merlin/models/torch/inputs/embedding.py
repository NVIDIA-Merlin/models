import collections
import inspect
from typing import Callable, Dict, Final, Optional, OrderedDict, Type, Union

import torch
from torch import nn

from merlin.models.torch.block import ParallelBlock
from merlin.models.torch.utils.selection_utils import Selectable, Selection, select_schema
from merlin.models.utils.schema_utils import get_embedding_size_from_cardinality
from merlin.schema import ColumnSchema, Schema
from merlin.schema.schema import Domain

DimFn = Callable[[Schema], int]
Combiner = Union[str, nn.Module]


def infer_embedding_dim(schema: Schema, multiplier: float = 2.0, ensure_multiple_of_8: bool = True):
    cardinality = sum(col.int_domain.max for col in schema) + 1

    return get_embedding_size_from_cardinality(
        cardinality, multiplier=multiplier, ensure_multiple_of_8=ensure_multiple_of_8
    )


class EmbeddingTable(nn.Module, Selectable):
    has_module_combiner: Final[bool] = False

    def __init__(
        self,
        dim: Union[int, DimFn] = infer_embedding_dim,
        schema: Optional[Union[ColumnSchema, Schema]] = None,
        combiner: Optional[Combiner] = None,
    ):
        super().__init__()
        self.dim = dim
        self.combiner = combiner
        self.domains: OrderedDict[str, Domain] = collections.OrderedDict()
        self.feature_to_domain: Dict[str, Domain] = {}
        self.has_module_combiner = isinstance(self.combiner, nn.Module)
        self.num_embeddings = 0
        self.setup_schema(schema or Schema())

    def setup_schema(self, schema: Schema):
        if isinstance(schema, ColumnSchema):
            schema = Schema([schema])

        self.schema = Schema()
        for col in schema:
            self.add_feature(col)
        if self.schema:
            self.table = self.create_table()

    def create_table(self) -> nn.Module:
        if not isinstance(self.dim, int):
            self.dim = self.dim(self.schema.first)

        return nn.Embedding(self.num_embeddings, self.dim)

    def insert_rows(self, num_rows: int, start_index: int = 0):
        device = self.table.weight.device
        new_embedding = type(self.table)(num_rows, self.dim).to(device)

        with torch.no_grad():
            # Slice the original weight tensor into two parts: before and after the start index
            weight_before = self.table.weight[:start_index]
            weight_after = self.table.weight[start_index:]

            # Concatenate these parts with the new embedding's weight
            new_weight = torch.cat([weight_before, new_embedding.weight, weight_after])
            new_embedding.weight = torch.nn.Parameter(new_weight)

        self.table = new_embedding

    def append_rows(self, num_rows: int):
        self.insert_rows(num_rows, self.num_embeddings)

        return self

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.schema:
            raise RuntimeError("Table is not initialized, please add features to the table")

        if torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
            if len(inputs) == 2:
                ids, offsets = None, None

                for key in inputs:
                    if key.endswith("__values"):
                        ids = inputs[key]
                    elif key.endswith("__offsets"):
                        offsets = inputs[key]

                if ids is not None and offsets is not None:
                    return self.forward_bag(ids, offsets)

            return self.forward_dict(inputs)

        if ids.dim() > 2:
            ids = torch.squeeze(ids, dim=-1)

        if self.combiner and not self.has_module_combiner:
            out = nn.functional.embedding_bag(ids, offsets=offsets, mode=self.combiner)
        else:
            out = self.table(ids)

        if self.has_module_combiner:
            out = self.combiner(out)

        return out

    def forward_bag(
        self,
        inputs: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ):
        return nn.functional.embedding_bag(inputs, offsets=offsets, mode=self.combiner)

    def forward_dict(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with input as a dictionary.

        When multiple features are passed in, we stack them together
        and do a single pass through the embedding table for efficiency.

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary of input tensors.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of output tensors.
        """

        if len(self.schema) == 1:
            name = self.schema.first.name
            filtered = {key: val for key, val in inputs.items() if key.startswith(name)}

            return {name: self.forward(filtered)}

        names, shapes = [], []
        tensors_by_len = collections.defaultdict(list)
        for feature_name in sorted(self.schema.column_names):
            if feature_name in inputs:
                tensor = inputs[feature_name]
                domain = self.feature_to_domain[feature_name]
                if domain.min > 1:
                    tensor = tensor + domain.min
                tensors_by_len[len(tensor.shape)].append(tensor)
                names.append(feature_name)
                shapes.append(tensor.shape)

        out = {}

        for num_dims, tensors in tensors_by_len.items():
            stacked = self.forward(torch.cat(tensors))
            i = 0
            for name, shape in zip(names, shapes):
                out[name] = stacked[i : i + shape[0]]
                i += shape[0]

        return out

    def add_feature(self, col_schema: ColumnSchema) -> "EmbeddingTable":
        """Add a feature to the table.

        Args:
            col_schema (ColumnSchema): The column schema of the feature to add.

        Returns:
            EmbeddingTable: The updated module with the added feature.
        """
        if not col_schema.int_domain:
            raise ValueError("`col_schema` needs to have an int-domain")

        domain = col_schema.int_domain
        to_add = 0
        if domain.name not in self.domains:
            self.domains[domain.name] = Domain(
                min=domain.min + self.num_embeddings,
                max=domain.max + self.num_embeddings,
                name=domain.name,
            )
            to_add += domain.max

        self.feature_to_domain[col_schema.name] = self.domains[domain.name]
        self.schema += Schema([col_schema])

        if self.num_embeddings == 0:
            self.num_embeddings = 1
        self.num_embeddings += to_add
        if hasattr(self, "table"):
            self.append_rows(to_add)

        return self

    def update_feature(self, col_schema: ColumnSchema) -> "EmbeddingTable":
        """Update a feature in the table.

        Args:
            col_schema (ColumnSchema): The new column schema of the feature.

        Returns:
            EmbeddingTable: The updated module with the updated feature.
        """
        feature_name = col_schema.name

        # Check if the feature to update exists in the table
        if feature_name not in self.feature_to_domain:
            raise ValueError(f"Feature '{feature_name}' not found in the table")

        if not col_schema.int_domain:
            raise ValueError("`new_col_schema` needs to have an int-domain")

        new_domain = col_schema.int_domain

        # Calculate the difference between the new and old max values
        old_max = self.feature_to_domain[feature_name].max
        diff = new_domain.max - old_max

        if diff < 0:
            raise ValueError("New domain cannot have fewer embeddings than the old domain")

        # Update the domain of the feature
        self.domains[new_domain.name] = Domain(
            min=self.feature_to_domain[feature_name].min,
            max=new_domain.max,
            name=new_domain.name,
        )
        for feature in self.feature_to_domain:
            if self.feature_to_domain[feature].name == new_domain.name:
                self.feature_to_domain[feature] = self.domains[new_domain.name]

        # Shift the min and max values of the subsequent domains
        for domain in self.domains.value():
            if domain.min > old_max:
                domain.min += diff
                domain.max += diff

        self.schema[feature_name] = col_schema

        # Update the total number of embeddings
        self.num_embeddings += diff
        if hasattr(self, "table"):
            self.insert_rows(diff, self.feature_to_domain[feature_name].min)

        return self

    def select(self, selection: Selection) -> Selectable:
        selected = select_schema(self.schema, selection)

        if not selected:
            raise ValueError(f"Selection {selection} not found in the table")

        return self

    def contains_multiple_domains(self) -> bool:
        """
        Check if the module contains multiple domains.

        Returns:
            bool: True if the module contains multiple domains, False otherwise.
        """
        return len(self.domains) > 1

    @property
    def table_name(self) -> str:
        """Get the table name.

        Returns:
            str: The name of the table.
        """
        return "_".join([domain.name for domain in self.domains.values()])

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def __bool__(self) -> bool:
        """Check if the module is valid.

        Returns:
            bool: True if the table contains features, False otherwise.
        """
        return bool(self.schema)


class EmbeddingTables(ParallelBlock, Selectable):
    def __init__(
        self,
        dim: Optional[Union[Dict[str, int], int, DimFn]] = 100,
        schema: Optional[Schema] = None,
        table_cls: Type[nn.Module] = EmbeddingTable,
        sequence_combiner: Optional[Combiner] = None,
    ):
        super().__init__()
        self.dim = dim
        self.table_cls = table_cls
        self.sequence_combiner = sequence_combiner
        if isinstance(schema, Schema):
            self.setup_schema(schema)

    def setup_schema(self, schema: Schema):
        self.schema = schema

        for col in schema:
            self.branches[col.name] = self.table_cls(self.dim, schema=col)
            self.branches[col.name] = _forward_kwargs_to_table(col, self.table_cls, self.kwargs)

        return self

    def select(self, selection: Selection) -> "EmbeddingTables":
        # TODO: Fix this
        return EmbeddingTables(self.dim, schema=select_schema(self.schema, selection))


def _forward_kwargs_to_table(col, table_cls, kwargs):
    arg_spec = inspect.getfullargspec(table_cls.__init__)
    supported_kwargs = arg_spec.kwonlyargs
    if arg_spec.defaults:
        supported_kwargs += arg_spec.args[-len(arg_spec.defaults) :]

    table_kwargs = {}
    for key, val in kwargs.items():
        if key in supported_kwargs:
            if isinstance(val, dict):
                if col.name in val:
                    table_kwargs[key] = val[col.name]
            else:
                table_kwargs[key] = val

    return table_kwargs
