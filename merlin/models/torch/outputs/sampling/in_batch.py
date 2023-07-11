from typing import Tuple

import torch
from torch import nn

from merlin.models.torch.block import registry


@registry.register("in-batch")
class InBatchNegativeSampler(nn.Module):
    """PyTorch module that performs in-batch negative sampling."""

    def __init__(self):
        super().__init__()
        self.register_buffer("negative", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("negative_id", torch.zeros(1))

    def forward(
        self, positive: torch.Tensor, positive_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Doing in-batch negative-sampling.

        positive & positive_id are registered as non-persistent buffers

        Args:
            positive (torch.Tensor): Tensor containing positive samples.
            positive_id (torch.Tensor, optional): Tensor containing the IDs of
                positive samples. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the positive
                samples tensor and the positive samples IDs tensor.
        """
        if self.training:
            if torch.jit.isinstance(positive, torch.Tensor):
                self.negative = positive
            if torch.jit.isinstance(positive_id, torch.Tensor):
                self.negative_id = positive_id

        return positive, positive_id
