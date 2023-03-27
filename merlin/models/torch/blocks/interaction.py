from typing import Tuple

import torch
from torch import nn

from merlin.models.torch.base import Block, registry
from merlin.models.torch.transforms.aggregation import ConcatFeatures, StackFeatures


@registry.register("dot-product")
class DotProduct(Block):
    """Dot-product between queries & candidates.

    Parameters:
    -----------
    query_name : str, optional
        Identify query tower for query/user embeddings, by default 'query'
    candidate_name : str, optional
        Identify item tower for item embeddings, by default 'candidate'
    """

    def __init__(
        self, query_name: str = "query", candidate_name: str = "candidate", pre=None, post=None
    ):
        super().__init__(pre=pre, post=post)
        self.query_name = query_name
        self.candidate_name = candidate_name

    def forward(self, inputs):
        query, candidate = _get_left_and_right(inputs, self.query_name, self.candidate_name)

        # Alternative is: torch.einsum('...i,...i->...', query, item)
        return torch.sum(query * candidate, dim=-1, keepdim=True)


class DLRMInputProcessing(nn.Module):
    def __init__(self, continious_name="continuous", categorical_name="categorical"):
        super().__init__()
        self.continous_name = continious_name
        self.categorical_name = categorical_name
        self.concat = ConcatFeatures()
        self.stack = StackFeatures()

    def forward(self, inputs) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs

        continuous, categorical = _get_left_and_right(
            inputs, self.continous_name, self.categorical_name
        )
        if isinstance(continuous, dict):
            continuous = self.concat(continuous)

        if isinstance(categorical, dict):
            stacked = self.stack({**categorical, **continuous})
        else:
            stacked = torch.cat([continuous, categorical], dim=-1)

        return stacked


@registry.register("dlrm-interaction")
class DLRMInteraction(Block):
    def __init__(self, pre=DLRMInputProcessing(), post=None):
        super().__init__(pre, post)

    def forward(self, inputs):
        # TODO: Cache triu_indices
        triu_indices = torch.triu_indices(inputs.shape[1], inputs.shape[1], offset=1)

        interactions = torch.bmm(inputs, torch.transpose(inputs, 1, 2))
        interactions_flat = interactions[:, triu_indices[0], triu_indices[1]]

        return interactions_flat


def _get_left_and_right(inputs, left_name, right_name) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(inputs, dict):
        try:
            left = inputs[left_name]
            right = inputs[right_name]
        except KeyError as e:
            raise RuntimeError(
                f"Key {e} not found in input dictionary. "
                "Please provide a dictionary with keys "
                f"'{left_name}' and '{right_name}'."
            ) from e
    elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
        left = inputs[0]
        right = inputs[1]
    else:
        raise RuntimeError(
            "Invalid input type. "
            "Expected inputs to be either a dictionary with keys "
            f"'{left_name}' and '{right_name}' or a tuple/list of size 2."
        )

    return left, right
