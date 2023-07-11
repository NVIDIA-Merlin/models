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
from typing import Dict, Final, Optional, Sequence, Tuple, Union

import torch
import torchmetrics as tm
from torch import nn

from merlin.models.torch import schema
from merlin.models.torch.batch import Batch
from merlin.models.torch.block import Block
from merlin.models.torch.outputs import sampling  # noqa: F401
from merlin.models.torch.outputs.base import ModelOutput
from merlin.models.torch.outputs.classification import (
    CategoricalOutput,
    CategoricalTarget,
    EmbeddingTablePrediction,
    create_retrieval_metrics,
)
from merlin.models.utils.constants import MIN_FLOAT
from merlin.schema import ColumnSchema, Schema


class ContrastiveOutput(ModelOutput):
    """
    A prediction block for a contrastive output.

    Parameters
    ----------
    schema: Union[ColumnSchema, Schema], optional
        The schema defining the column properties. Default is None.
    loss : nn.Module, optional
        The loss function to use for the output model, defaults to
        torch.nn.CrossEntropyLoss.
    metrics : Optional[Sequence[Metric]], optional
        The metrics to evaluate the model output.
    logits_temperature: float, optional
        Parameter used to reduce model overconfidence, so that logits / T.
        by default 1.0
    """

    def __init__(
        self,
        schema: Optional[Union[ColumnSchema, Schema]] = None,
        negative_sampling="in-batch",
        loss: nn.Module = nn.CrossEntropyLoss(),
        metrics: Optional[Sequence[tm.Metric]] = None,
        downscore_false_negatives: bool = True,
        false_negative_score: float = MIN_FLOAT,
        logits_temperature: float = 1.0,
    ):
        if not metrics:
            metrics = create_retrieval_metrics(
                CategoricalOutput.DEFAULT_METRICS_CLS, CategoricalOutput.DEFAULT_K
            )

        super().__init__(
            loss=loss,
            metrics=metrics,
            logits_temperature=logits_temperature,
        )

        if schema:
            self.initialize_from_schema(schema)

        self.init_hook_handle = self.register_forward_pre_hook(self.initialize)
        if not torch.jit.is_scripting():
            if isinstance(negative_sampling, str):
                negative_sampling = [negative_sampling]
            self.negative_sampling = nn.ModuleList((Block.parse(s) for s in negative_sampling))
            self.downscore_false_negatives = downscore_false_negatives
            self.false_negative_score = false_negative_score

    @classmethod
    def with_weight_tying(
        cls,
        block: nn.Module,
        selection: Optional[schema.Selection] = None,
        negative_sampling="popularity",
        loss: nn.Module = nn.CrossEntropyLoss(),
        metrics: Optional[Sequence[tm.Metric]] = None,
        logits_temperature: float = 1.0,
        downscore_false_negatives: bool = True,
        false_negative_score: float = MIN_FLOAT,
    ) -> "CategoricalOutput":
        self = cls(
            loss=loss,
            metrics=metrics,
            logits_temperature=logits_temperature,
            negative_sampling=negative_sampling,
            downscore_false_negatives=downscore_false_negatives,
            false_negative_score=false_negative_score,
        )
        self = self.tie_weights(block, selection)

        return self

    def tie_weights(
        self, block: nn.Module, selection: Optional[schema.Selection] = None
    ) -> "CategoricalOutput":
        prediction = EmbeddingTablePrediction.with_weight_tying(block, selection)
        self.num_classes = prediction.num_classes
        if self:
            self[0] = prediction
        else:
            self.prepend(prediction)
        self.set_to_call(prediction)

        return self

    def initialize_from_schema(self, target: Union[ColumnSchema, Schema]):
        """Set up the schema for the output.

        Parameters
        ----------
        target: Optional[ColumnSchema]
            The schema defining the column properties.
        """
        if not isinstance(target, (ColumnSchema, Schema)):
            raise ValueError(f"Target must be a ColumnSchema or Schema, got {target}.")

        if isinstance(target, Schema) and len(target) > 1:
            if len(target) > 2:
                raise ValueError(f"Schema must have one or two column(s), got {target}.")

            self.set_to_call(DotProduct(*target.column_names))
        else:
            if isinstance(target, Schema):
                target = target.first
            self.set_to_call(CategoricalTarget(target))

        self.prepend(self.to_call)

    def initialize(self, module, inputs):
        if torch.jit.isinstance(inputs[0], Dict[str, torch.Tensor]):
            for i in range(len(module)):
                if isinstance(module[i], CategoricalTarget):
                    module[i] = DotProduct(module[i].target_name)
                    self.set_to_call(module[i])
        elif isinstance(self.to_call, CategoricalTarget):
            # Make sure CategoticalTarget is initialized
            outputs = inputs[0]
            for to_call in module.values:
                outputs = to_call(outputs)

        self.init_hook_handle.remove()  # Clear hook once block is initialized

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ) -> torch.Tensor:
        outputs = inputs
        for module in self.values:
            should_apply_contrastive = False

            can_contrast = (
                not torch.jit.is_scripting()
                and hasattr(module, "requires_batch")
                and hasattr(module.to_wrap, "should_apply_contrastive")
            )
            if can_contrast:
                should_apply_contrastive = module.to_wrap.should_apply_contrastive(batch)

            if should_apply_contrastive:
                if batch is None:
                    raise ValueError("Contrastive output requires a batch")
                _batch = batch if batch is not None else Batch({})

                target_name = module.to_wrap.target_name

                if torch.jit.isinstance(outputs, Dict[str, torch.Tensor]) and hasattr(
                    module.to_wrap, "get_query_name"
                ):
                    query_name: str = module.to_wrap.get_query_name(outputs)
                    outputs = self.contrastive_dot_product(
                        outputs,
                        _batch,
                        target_name=target_name,
                        query_name=query_name,
                    )
                elif torch.jit.isinstance(outputs, torch.Tensor):
                    outputs = self.contrastive_lookup(
                        outputs,
                        _batch,
                        target_name=target_name,
                    )
                else:
                    raise RuntimeError("Couldn't apply contrastive output")
            else:
                outputs = module(outputs, batch=batch)

        return outputs

    @torch.jit.unused
    def contrastive_dot_product(
        self,
        inputs: Dict[str, torch.Tensor],
        batch: Batch,
        target_name: str,
        query_name: str,
    ) -> torch.Tensor:
        query = inputs[query_name]
        positive = inputs[target_name]
        positive_id = None
        if target_name in batch.features:
            positive_id = batch.features[target_name]

        negative, negative_id = self.sample_negatives(positive, positive_id=positive_id)

        return self.contrastive_outputs(
            query,
            positive,
            negative,
            positive_id=positive_id,
            negative_id=negative_id,
        )

    @torch.jit.unused
    def contrastive_lookup(
        self,
        inputs: torch.Tensor,
        batch: Batch,
        target_name: str,
    ) -> torch.Tensor:
        query = inputs
        if len(batch.targets) == 1:
            positive_id = batch.target()
        else:
            positive_id = batch.targets[target_name]

        if not hasattr(self.to_call, "embedding_lookup"):
            raise ValueError("Couldn't infer positive embedding")
        positive = self.embedding_lookup(positive_id)

        negative, negative_id = self.sample_negatives(positive, positive_id=positive_id)

        return self.contrastive_outputs(
            query,
            positive,
            negative,
            positive_id=positive_id,
            negative_id=negative_id,
        )

    @torch.jit.unused
    def sample_negatives(
        self, positive: torch.Tensor, positive_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        for sampler in self.negative_sampling:
            _positive_id: torch.Tensor = positive_id if positive_id is not None else torch.tensor(1)
            negative, negative_id = sampler(positive, positive_id=_positive_id)
            if negative.shape[0] == negative_id.shape[0]:
                ids.append(negative_id)
            outputs.append(negative)

        if ids:
            if len(outputs) != len(ids):
                raise RuntimeError("The number of negative samples and ids must be the same")

        negative_tensor = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
        id_tensor = torch.tensor(1)
        if ids:
            id_tensor = torch.cat(ids, dim=0) if len(ids) > 1 else ids[0]

        return negative_tensor, id_tensor

    @torch.jit.unused
    def contrastive_outputs(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        positive_id: Optional[torch.Tensor] = None,
        negative_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
            torch.Tensor: Tensor containing the contrastive outputs.

            Note, the transformed-targets are stored in
                the `target` attribute (which is a buffer).
        """

        # Dot-product for the positive-scores
        positive_scores = torch.sum(query * positive, dim=-1, keepdim=True)
        negative_scores = torch.matmul(query, negative.t())

        if self.downscore_false_negatives:
            if (
                positive_id is None
                or negative_id is None
                or positive.shape[0] != positive_id.shape[-1]
                or negative.shape[0] != negative_id.shape[-1]
            ):
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

        self.target = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=output.dtype),
                torch.zeros(batch_size, num_negatives, dtype=output.dtype),
            ],
            dim=1,
        )

        return output

    def embedding_lookup(self, ids: torch.Tensor) -> torch.Tensor:
        return self.to_call.embedding_lookup(torch.squeeze(ids))

    def set_to_call(self, to_call: nn.Module):
        self.to_call = to_call
        if not torch.jit.is_scripting() and hasattr(self, "negative_sampling"):
            for sampler in self.negative_sampling:
                if hasattr(sampler, "set_to_call"):
                    sampler.set_to_call(to_call)


@Block.registry.register("dot-product")
class DotProduct(nn.Module):
    """Dot-product between queries & candidates.

    Parameters:
    -----------
    query_name : str, optional
        Identify query tower for query/user embeddings, by default 'query'
    candidate_name : str, optional
        Identify item tower for item embeddings, by default 'candidate'
    """

    is_output_module: Final[bool] = True

    def __init__(
        self,
        candidate_name: str = "candidate",
        query_name: Optional[str] = None,
    ):
        super().__init__()
        self.query_name: Optional[str] = query_name
        self.candidate_name = candidate_name
        self.target_name = self.candidate_name

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if len(inputs) < 2:
            raise RuntimeError(f"DotProduct requires at least two inputs, got: {inputs}")

        candidate = inputs[self.candidate_name]

        if len(inputs) == 2:
            query_name = self.get_query_name(inputs)
        elif self.query_name is not None:
            query_name = self.query_name
        else:
            raise RuntimeError(
                "DotProduct requires query_name to be set when more than ",
                f"two inputs are provided, got: {inputs}",
            )
        query = inputs[query_name]

        # Alternative is: torch.einsum('...i,...i->...', query, item)
        return torch.sum(query * candidate, dim=-1, keepdim=True)

    def get_query_name(self, inputs: Dict[str, torch.Tensor]) -> str:
        if self.query_name is None:
            for key in inputs:
                if key != self.candidate_name:
                    return key
        else:
            return self.query_name

        raise RuntimeError(
            "DotProduct requires query_name to be set when more than two inputs are provided"
        )

    def should_apply_contrastive(self, batch: Optional[Batch]) -> bool:
        return self.training


def rescore_false_negatives(
    positive_item_ids: torch.Tensor,
    neg_samples_item_ids: torch.Tensor,
    negative_scores: torch.Tensor,
    false_negatives_score: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
