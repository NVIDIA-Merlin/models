from typing import Optional

import torch
import torch.nn as nn

from merlin.models.torch.data import FeatureMixin


class SamplingProbabilityCorrection(nn.Module, FeatureMixin):
    """Sampling probability correction module.
    Corrects logits based on the candidate sampling probability.

    The sampling probability can be passed when calling the module,
    or it can be propagated as a feature through the Model.

    Parameters
    ----------
    feature_name : str, optional
        The name of the feature that provides the candidate_sampling_probability,
        by default "candidate_sampling_probability".

    """

    def __init__(self, feature_name: str = "candidate_sampling_probability"):
        super().__init__()
        self.feature_name = feature_name

    def forward(
        self, logits: torch.Tensor, candidate_sampling_probability: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Corrects the input logits to account for candidate sampling probability."""

        if candidate_sampling_probability is None:
            probability = self.get_feature(self.feature_name)
        else:
            probability = candidate_sampling_probability

        return logits - torch.log(torch.clamp(probability, min=1e-6, max=1.0))
