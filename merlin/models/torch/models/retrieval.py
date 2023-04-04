from typing import Callable, Optional, Union

from torch import nn

from merlin.schema import ColumnSchema, Schema, Tags
from merlin.models.torch.blocks.interaction import DotProduct
from merlin.models.torch.inputs.embedding import EmbeddingTable
from merlin.models.torch.inputs.encoder import Encoder
from merlin.models.torch.models.base import RetrievalModel
from merlin.models.torch.outputs.base import ModelOutput
from merlin.models.torch.outputs.contrastive import ContrastiveOutput
from merlin.models.torch.outputs.sampling.popularity import PopularityBasedSampler

DEFAULT_QUERY_TAG = Tags.USER_ID
DEFAULT_CANDIDATE_TAG = Tags.ITEM_ID

ColumnSelectionType = Union[ColumnSchema, Callable[[Schema], ColumnSchema]]


def query_column_fn(schema: Schema, tag=DEFAULT_QUERY_TAG) -> ColumnSchema:
    return schema.select_by_tag(tag).first


def candidate_column_fn(schema: Schema, tag=DEFAULT_CANDIDATE_TAG) -> ColumnSchema:
    return schema.select_by_tag(tag).first


class MatrixFactorizationModel(RetrievalModel):
    def __init__(
        self,
        schema: Schema,
        dim: int,
        *,
        query_column: ColumnSelectionType = query_column_fn,
        candidate_column: ColumnSelectionType = candidate_column_fn,
        output: Optional[Union[nn.Module, ModelOutput]] = None,
        pre=None,
        post=None,
    ):
        query_col = _parse_column_schema(schema, query_column)
        query: Encoder = self._create_encoder(dim, query_col)

        candidate_col = _parse_column_schema(schema, candidate_column)
        candidate: Encoder = self._create_encoder(dim, candidate_col)

        if not output:
            output = ContrastiveOutput(
                candidate_col,
                dot_product=DotProduct(
                    query_name=candidate_col.name, candidate_name=query_col.name
                ),
            )

        super().__init__(
            query=query,
            candidate=candidate,
            output=output,
            pre=pre,
            post=post,
        )

    @classmethod
    def from_tables(
        cls,
        query_table: EmbeddingTable,
        candidate_table: EmbeddingTable,
        output: Optional[Union[nn.Module, ModelOutput]] = None,
        pre=None,
        post=None,
    ) -> "MatrixFactorizationModel":
        if not output:
            output = ContrastiveOutput(
                query_table.schema,
                dot_product=DotProduct(
                    query_name=query_table.schema.name, candidate_name=candidate_table.schema.name
                ),
            )

        model = RetrievalModel(
            query=Encoder(query_table),
            candidate=Encoder(candidate_table),
            output=output,
            pre=pre,
            post=post,
        )

        model.__class__ = cls

        return model

    def _create_encoder(self, dim: int, column: ColumnSchema) -> Encoder:
        return Encoder(EmbeddingTable(dim, column))


class TwoTowerModel(RetrievalModel):
    def __init__(
        self,
        query_tower: nn.Module,
        candidate_tower: nn.Module,
        *,
        output: Optional[Union[nn.Module, ModelOutput]] = None,
        candidate_column: ColumnSelectionType = candidate_column_fn,
        pre=None,
        post=None,
    ):
        candidate_col = _parse_column_schema(candidate_tower.input_schema, candidate_column)
        
        super().__init__(
            query=query_tower,
            candidate=candidate_tower,
            output=output or ContrastiveOutput(candidate_col),
            pre=pre,
            post=post,
        )


class YoutubeDNNRetrievalModel(RetrievalModel):
    def __init__(
        self,
        inputs: Union[Schema, nn.Module],
        top_block: nn.Module,
        output: Optional[Union[nn.Module, ModelOutput]] = None,
        *,
        candidate_column: ColumnSelectionType = candidate_column_fn,
        max_num_samples: int = 100,
        pre=None,
        post=None,
    ):
        schema = inputs if isinstance(inputs, Schema) else inputs.input_schema
        candidate_col = _parse_column_schema(schema, candidate_column)
        
        if not output:
            output = ContrastiveOutput(
                candidate_col,
                negative_samplers=PopularityBasedSampler(max_num_samples),
            )
        
        super().__init__(
            query=Encoder(inputs, top_block),
            output=output,
            pre=pre,
            post=post,
        )


def _parse_column_schema(schema: Schema, column: ColumnSelectionType) -> ColumnSchema:
    if isinstance(column, nn.Module):
        return column

    if callable(column):
        output = column(schema)
    else:
        output = column

    if not isinstance(output, ColumnSchema):
        raise ValueError(f"Column {column} is not a valid ColumnSchema")

    return output
