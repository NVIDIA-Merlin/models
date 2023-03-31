from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torchmetrics import Metric

from merlin.core.dispatch import DataFrameType
from merlin.io import Dataset
from merlin.models.torch.base import Block
from merlin.models.torch.blocks.interaction import DotProduct
from merlin.models.torch.data import register_feature_hook
from merlin.models.torch.inputs.embedding import EmbeddingTable
from merlin.models.torch.outputs.base import ModelOutput
from merlin.models.torch.outputs.classification import (
    CategoricalTarget,
    EmbeddingTablePrediction,
    categorical_output_schema,
)
from merlin.models.torch.outputs.sampling.in_batch import InBatchNegativeSampler  # noqa: F401
from merlin.models.utils.constants import MIN_FLOAT
from merlin.models.utils.protocols import LookUpProtocol
from merlin.schema import ColumnSchema, Schema


class ContrastiveOutput(ModelOutput):
    def __init__(
        self,
        to_call: Union[
            Schema, ColumnSchema, EmbeddingTable, CategoricalTarget, EmbeddingTablePrediction
        ],
        target_name: str = None,
        default_loss=nn.CrossEntropyLoss(),
        default_metrics: Sequence[Metric] = (),
        target: Optional[Union[str, ColumnSchema]] = None,
        negative_samplers="in-batch",
        downscore_false_negatives: bool = True,
        false_negative_score: float = MIN_FLOAT,
        pre=None,
        post=None,
        logits_temperature: float = 1.0,
        dot_product=DotProduct(),
    ):
        _to_call = None
        if to_call is not None:
            if isinstance(to_call, (Schema, ColumnSchema)):
                _to_call = CategoricalTarget(to_call)
                if isinstance(to_call, Schema):
                    to_call = to_call.first
                target_name = target_name or to_call.name
                target = to_call
            elif isinstance(to_call, EmbeddingTable):
                _to_call = EmbeddingTablePrediction(to_call)
                target = _to_call.table.schema.first
                if len(to_call.schema) == 1:
                    target_name = target.name
                else:
                    raise ValueError("Can't infer the target automatically, please provide it.")
            else:
                _to_call = to_call

        super().__init__(
            to_call=_to_call,
            default_loss=default_loss,
            default_metrics=default_metrics,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
        )
        register_feature_hook(self, Schema([self.target_col]))

        if isinstance(negative_samplers, str):
            negative_samplers = [negative_samplers]

        self.negative_samplers = [
            Block.from_registry(s) if isinstance(s, str) else s for s in negative_samplers
        ]
        self.downscore_false_negatives = downscore_false_negatives
        self.false_negative_score = false_negative_score
        self.dot_product = dot_product
        self.keys = [self.dot_product.query_name, self.dot_product.candidate_name]
        self.num_classes = self.to_call.num_classes

    def forward(self, inputs, targets=None, features=None):
        if (
            isinstance(inputs, dict)
            and all(key in inputs for key in self.keys)
            and not isinstance(self.to_call, type(self.dot_product))
        ):
            self.to_call = self.dot_product

        if self.is_in_training or self.is_in_testing:
            if self.has_candidate_weights and targets in [None, {}]:
                return super().forward(inputs, targets=targets)

            return self.contrastive_forward(inputs, targets=targets, features=features)

        return super().forward(inputs, targets=targets)

    def contrastive_forward(
        self, inputs, targets=None, features=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_name = self.keys[0]
        if isinstance(inputs, dict) and query_name in inputs:
            query = inputs[query_name]
        elif isinstance(inputs, torch.Tensor):
            query = inputs
        else:
            raise ValueError("Couldn't infer query embedding")

        positive_id: Optional[torch.Tensor] = None
        if self.has_candidate_weights:
            positive_id = targets
            if isinstance(targets, dict):
                positive_id = targets[self.target_col.name]

            # Check if we need to initialize the embedding table
            if not getattr(self.to_call, "is_initialized", True):
                self.to_call(inputs)

            positive = self.embedding_lookup(positive_id)
        else:
            if isinstance(features, dict):
                positive_id = features.get(self.target_col.name, None)
            positive = inputs[self.keys[1]]

        negative, negative_id = self.sample_negatives(positive, positive_id=positive_id)

        return self.contrastive_outputs(
            query,
            positive,
            negative,
            positive_id=positive_id,
            negative_id=negative_id,
        )

    def sample_negatives(
        self, positive, positive_id=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Samples negative examples for the given positive tensor.

        Args:
            positive (torch.Tensor): Tensor containing positive samples.
            positive_id (torch.Tensor, optional): Tensor containing the IDs
                of positive samples. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing the negative samples
                tensor and the IDs of negative samples.
        """
        outputs, ids = [], []
        for sampler in self.negative_samplers:
            negative = sampler(positive, positive_id=positive_id)
            if isinstance(negative, tuple) and len(negative) == 2:
                negative, negative_ids = negative
                ids.append(negative_ids)
            outputs.append(negative)

        if ids:
            if len(outputs) != len(ids):
                raise RuntimeError("The number of negative samples and ids must be the same")

        negatives = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
        if ids:
            ids = torch.cat(ids, dim=0) if len(ids) > 1 else ids[0]

        return negatives, ids

    def contrastive_outputs(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        positive_id: Optional[torch.Tensor] = None,
        negative_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the contrastive outputs given
            the query tensor, positive tensor, and negative tensor.

        Args:
            query (torch.Tensor): Tensor containing the query data.
            positive (torch.Tensor): Tensor containing the positive data.
            negative (torch.Tensor): Tensor containing the negative data.
            positive_id (torch.Tensor, optional): Tensor containing the IDs of positive samples.
                Defaults to None.
            negative_id (torch.Tensor, optional): Tensor containing the IDs of negative samples.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the output
                tensor and the target tensor.
        """

        # Dot-product for the positive-scores
        positive_scores = torch.sum(query * positive, dim=-1, keepdim=True)
        negative_scores = torch.matmul(query, negative.t())

        if self.downscore_false_negatives:
            if positive_id is None or negative_id is None:
                raise RuntimeError(
                    "Both positive_id and negative_id must be provided "
                    "when downscore_false_negatives is True"
                )
            negative_scores, _ = rescore_false_negatives(
                positive_id, negative_id, negative_scores, self.false_negative_score
            )

        if len(negative_scores.shape) + 1 == len(positive_scores.shape):
            negative_scores = negative_scores.unsqueeze(0)
        output = torch.cat([positive_scores, negative_scores], dim=-1)

        # Ensure the output is always float32 to avoid numerical instabilities
        output = output.to(torch.float32)

        batch_size = output.shape[0]
        num_negatives = output.shape[1] - 1
        target = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=output.dtype),
                torch.zeros(batch_size, num_negatives, dtype=output.dtype),
            ],
            dim=1,
        )

        return output, target

    def embedding_lookup(self, ids: torch.Tensor) -> torch.Tensor:
        return self.to_call.embedding_lookup(torch.squeeze(ids))

    def create_output_schema(self, target: ColumnSchema) -> Schema:
        return categorical_output_schema(target, self.num_classes)

    def to_dataset(self, gpu=None) -> Dataset:
        if not self.has_candidate_weights:
            raise RuntimeError("This model does not have candidate weights")

        return self.to_call.to_dataset(gpu=gpu)

    def to_df(self, gpu=None) -> DataFrameType:
        if not self.has_candidate_weights:
            raise RuntimeError("This model does not have candidate weights")

        return self.to_call.to_df(gpu=gpu)

    @property
    def has_candidate_weights(self) -> bool:
        if isinstance(self.to_call, DotProduct):
            return False

        return isinstance(self.to_call, LookUpProtocol)


def rescore_false_negatives(
    positive_item_ids: torch.Tensor,
    neg_samples_item_ids: torch.Tensor,
    negative_scores: torch.Tensor,
    false_negatives_score: float,
) -> torch.Tensor:
    """Zeroes the logits of accidental negatives.

    Parameters
    ----------
    positive_item_ids : torch.Tensor
        A tensor containing the IDs of the positive items.
    neg_samples_item_ids : torch.Tensor
        A tensor containing the IDs of the negative samples.
    negative_scores : torch.Tensor
        A tensor containing the scores of the negative samples.
    false_negatives_score : float
        The score to assign to false negatives (accidental hits).

    Returns
    -------
    torch.Tensor
        A tensor containing the rescored negative scores.
    torch.Tensor
        A tensor containing a mask representing valid negatives.
    """

    # Removing dimensions of size 1 from the shape of the item ids, if applicable
    positive_item_ids = torch.squeeze(positive_item_ids).to(neg_samples_item_ids.dtype)
    neg_samples_item_ids = torch.squeeze(neg_samples_item_ids)

    # Reshapes positive and negative ids so that false_negatives_mask matches the scores shape
    false_negatives_mask = torch.eq(
        positive_item_ids.unsqueeze(-1), neg_samples_item_ids.unsqueeze(0)
    )

    # Setting a very small value for false negatives (accidental hits) so that it has
    # negligible effect on the loss functions
    negative_scores = torch.where(
        false_negatives_mask,
        torch.ones_like(negative_scores) * false_negatives_score,
        negative_scores,
    )

    valid_negatives_mask = ~false_negatives_mask

    return torch.squeeze(negative_scores), valid_negatives_mask
