from typing import Tuple

import torch

from merlin.models.torch.base import Block, registry


@registry.register("dot-product")
class DotProduct(Block):
    """Dot-product between queries & candidates.

    Parameters:
    -----------
    query_name : str, optional
        Identify query tower for query/user embeddings, by default 'query'
    candidate_name : str, optional
        Identify item tower for item embeddings, by default 'item'
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
