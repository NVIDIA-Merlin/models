import collections
import inspect
from typing import Callable, Dict, Optional, OrderedDict, Type, Union

import torch
from torch import nn
from typing_extensions import Self

from merlin.models.torch.combinators import ParallelBlock
from merlin.models.utils.schema_utils import infer_embedding_dim
from merlin.schema import ColumnSchema, Schema
from merlin.schema.schema import Domain


class EmbeddingTableModule(nn.Module):
    """A module representing an embedding table.

    Extend this class to implement a custom embedding table.

    Args:
        dim (int): The embedding dimension.
        schema (Union[ColumnSchema, Schema]): The schema of the input data.

    Attributes:
        dim (int): The embedding dimension.
        schema (Schema): The schema of the input data.
        num_embeddings (int): The total number of embeddings.
        domains (OrderedDict[str, Domain]): The domains of the
            input features.
        feature_to_domain (Dict[str, Domain]): A mapping from feature
            names to their corresponding domains.
    """

    def __init__(self, dim: int, schema: Union[ColumnSchema, Schema]):
        super(EmbeddingTableModule, self).__init__()
        self.dim = dim

        self.num_embeddings: int = 1
        self.domains: OrderedDict[str, Domain] = collections.OrderedDict()
        self.feature_to_domain: Dict[str, Domain] = {}
        self.schema = Schema()

        if isinstance(schema, ColumnSchema):
            self.add_feature(schema)
        elif isinstance(schema, Schema):
            for col_schema in schema:
                self.add_feature(col_schema)
        else:
            raise ValueError("`schema` must be a ColumnSchema or Schema")

    def forward_dict(self, inputs: Dict[str, torch.Tensor]):
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

            return {name: self.forward(inputs[name])}

        tensors, names, shapes = [], [], []
        for feature_name in sorted(self.schema.column_names):
            if feature_name in inputs:
                tensor = inputs[feature_name]
                domain = self.feature_to_domain[feature_name]
                if domain.min > 1:
                    tensor = tensor + domain.min
                tensors.append(tensor)
                names.append(feature_name)
                shapes.append(tensor.shape)

        out = {}
        stacked = self.forward(torch.cat(tensors))
        i = 0
        for name, shape in zip(names, shapes):
            out[name] = stacked[i : i + shape[0]]
            i += shape[0]

        return out

    def add_feature(self, col_schema: ColumnSchema) -> Self:
        """Add a feature to the table.

        Args:
            col_schema (ColumnSchema): The column schema of the feature to add.

        Returns:
            EmbeddingTableModule: The updated module with the added feature.
        """
        if not col_schema.int_domain:
            raise ValueError("`col_schema` needs to have an int-domain")

        domain = col_schema.int_domain
        if domain.name not in self.domains:
            self.domains[domain.name] = Domain(
                min=domain.min + self.num_embeddings,
                max=domain.max + self.num_embeddings,
                name=domain.name,
            )
            self.num_embeddings += domain.max

        self.feature_to_domain[col_schema.name] = self.domains[domain.name]
        self.schema += Schema([col_schema])

        return self

    def select_by_name(self, names) -> Self:
        """Select features by name.

        Args:
            names (List[str]): A list of feature names to select.

        Returns:
            EmbeddingTable: A new embedding table with the selected features.
        """

        return EmbeddingTable(self.dim, self.schema.select_by_name(names))

    def select_by_tag(self, tags) -> Self:
        """
        Select features by tag.

        Args:
            tags (List[str]): A list of tags to select features by.

        Returns:
            EmbeddingTable: A new embedding table with the selected features.
        """

        return EmbeddingTable(self.dim, self.schema.select_by_tag(tags))

    @property
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

    def __bool__(self) -> bool:
        """Check if the module is valid.

        Returns:
            bool: True if the table contains features, False otherwise.
        """
        return bool(self.schema)

    # @classmethod
    # def from_pretrained(
    #     cls,
    #     data: Union[Dataset, DataFrameType],
    #     col_schema: Optional[ColumnSchema] = None,
    #     trainable=True,
    #     **kwargs,
    # ):
    #     raise NotImplementedError()


class EmbeddingTable(EmbeddingTableModule):
    """An embedding table that is backed by a PyTorch embedding table.

    Args:
        dim (int): The embedding dimension.
        schema (Union[ColumnSchema, Schema]): The schema of the input data.
        sequence_combiner (Optional[str]): The sequence combiner
            method (e.g., "mean", "sum", or "sqrtn").
        table (Optional[nn.Modules]): An optional embedding
            table to use.

    """

    def __init__(
        self,
        dim: int,
        schema: Union[ColumnSchema, Schema],
        sequence_combiner: Optional[str] = None,
        # trainable=True,
        # l2_batch_regularization_factor=0.0,
        table=None,
    ):
        super(EmbeddingTable, self).__init__(dim, schema)

        if table is not None:
            self.table = table
        else:
            self.table = nn.Embedding(self.num_embeddings, self.dim)

        self.sequence_combiner = sequence_combiner

    def forward(self, inputs):
        if isinstance(inputs, dict):
            return self.forward_dict(inputs)

        if inputs.dim() > 2:
            inputs = torch.squeeze(inputs, dim=-1)

        out = self.table(inputs)

        if self.sequence_combiner == "mean":
            out = torch.mean(out, dim=1)
        elif self.sequence_combiner == "sum":
            out = torch.sum(out, dim=1)
        elif self.sequence_combiner == "sqrtn":
            out = torch.sum(out, dim=1) / torch.sqrt(torch.sum(torch.square(out), dim=1) + 1e-6)

        return out

    @property
    def weight(self):
        return self.table.weight

    @property
    def input_schema(self) -> Schema:
        return self.schema


class Embeddings(ParallelBlock):
    def __init__(
        self,
        schema: Schema,
        dim: Optional[Union[Dict[str, int], int]] = None,
        infer_dim_fn: Callable[[ColumnSchema], int] = infer_embedding_dim,
        table_cls: Type[nn.Module] = EmbeddingTable,
        pre=None,
        post=None,
        aggregation=None,
    ):
        kwargs = {}
        tables = {}

        for col in schema:
            table_kwargs = _forward_kwargs_to_table(col, table_cls, kwargs)
            table_name = col.int_domain.name or col.name
            if table_name in tables:
                tables[table_name].add_feature(col)
            else:
                tables[table_name] = table_cls(
                    _get_dim(col, dim, infer_dim_fn),
                    col,
                    **table_kwargs,
                )

        super().__init__(tables, pre=pre, post=post, aggregation=aggregation)

    @property
    def input_schema(self) -> Schema:
        return self.schema


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


def _get_dim(col, embedding_dims, infer_dim_fn):
    dim = None
    if isinstance(embedding_dims, dict):
        dim = embedding_dims.get(col.name)
    elif isinstance(embedding_dims, int):
        dim = embedding_dims

    if not dim:
        dim = infer_dim_fn(col)

    return dim
