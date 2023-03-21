import inspect
from typing import Callable, Dict, Optional, Type, Union

import torch
from torch import nn

from merlin.models.torch.core.combinators import ParallelBlock
from merlin.models.utils.schema_utils import infer_embedding_dim
from merlin.schema import ColumnSchema, Schema


class EmbeddingTableBase(nn.Module):
    def __init__(self, dim: int, *col_schemas: ColumnSchema):
        super(EmbeddingTableBase, self).__init__()
        self.dim = dim
        self.features = {}
        if len(col_schemas) == 0:
            raise ValueError("At least one col_schema must be provided to the embedding table.")

        self.col_schema = col_schemas[0]
        for col_schema in col_schemas:
            self.add_feature(col_schema)

    @property
    def schema(self):
        return Schema([col_schema for col_schema in self.features.values()])

    # @classmethod
    # def from_pretrained(
    #     cls,
    #     data: Union[Dataset, DataFrameType],
    #     col_schema: Optional[ColumnSchema] = None,
    #     trainable=True,
    #     **kwargs,
    # ):
    #     raise NotImplementedError()

    @property
    def input_dim(self):
        return self.col_schema.int_domain.max + 1

    @property
    def table_name(self):
        return self.col_schema.int_domain.name or self.col_schema.name

    def add_feature(self, col_schema: ColumnSchema) -> None:
        """Add a feature to the table.

        Adding more than one feature enables the table to lookup and return embeddings
        for more than one feature when called with tabular data (Dict[str, TensorLike]).

        Additional column schemas must have an int domain that matches the existing ones.

        Parameters
        ----------
        col_schema : ColumnSchema
        """
        if not col_schema.int_domain:
            raise ValueError("`col_schema` needs to have an int-domain")

        if (
            col_schema.int_domain.name
            and self.col_schema.int_domain.name
            and col_schema.int_domain.name != self.col_schema.int_domain.name
        ):
            raise ValueError(
                "`col_schema` int-domain name does not match table domain name. "
                f"{col_schema.int_domain.name} != {self.col_schema.int_domain.name} "
            )

        if col_schema.int_domain.max != self.col_schema.int_domain.max:
            raise ValueError(
                "`col_schema.int_domain.max` does not match existing input dim."
                f"{col_schema.int_domain.max} != {self.col_schema.int_domain.max} "
            )

        self.features[col_schema.name] = col_schema


class EmbeddingTable(EmbeddingTableBase):
    def __init__(
        self,
        dim: int,
        *col_schemas: ColumnSchema,
        sequence_combiner: Optional[str] = None,
        # trainable=True,
        table=None,
        l2_batch_regularization_factor=0.0,
    ):
        super(EmbeddingTable, self).__init__(dim, *col_schemas)

        if table is not None:
            self.table = table
        else:
            self.table = nn.Embedding(self.input_dim, self.dim)

        self.sequence_combiner = sequence_combiner
        self.l2_batch_regularization_factor = l2_batch_regularization_factor

    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            out = {}
            for feature_name in self.schema.column_names:
                if feature_name in inputs:
                    out[feature_name] = self._call_table(inputs[feature_name], **kwargs)
        else:
            out = self._call_table(inputs, **kwargs)

        return out

    def _call_table(self, inputs, **kwargs):
        if inputs.dim() > 2:
            inputs = torch.squeeze(inputs, dim=-1)

        out = self.table(inputs)

        if self.sequence_combiner == "mean":
            out = torch.mean(out, dim=1)
        elif self.sequence_combiner == "sum":
            out = torch.sum(out, dim=1)
        elif self.sequence_combiner == "sqrtn":
            out = torch.sum(out, dim=1) / torch.sqrt(torch.sum(torch.square(out), dim=1) + 1e-6)

        if self.l2_batch_regularization_factor > 0:
            self.regularization_loss = self.l2_batch_regularization_factor * torch.sum(
                torch.square(out)
            )

        return out


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
        self.schema = schema


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
