import torch
import torch.nn as nn

from merlin.models.torch.data import register_feature_hook
from merlin.schema import Schema


class SamplingProbabilityCorrection(nn.Module):
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
        register_feature_hook(self, Schema([feature_name]))
        self.feature_name = feature_name

    def forward(
        self,
        logits: torch.Tensor,
        features=None,
    ) -> torch.Tensor:
        """Corrects the input logits to account for candidate sampling probability."""

        if isinstance(features, dict):
            probability = features[self.feature_name]
        elif isinstance(features, torch.Tensor):
            probability = features
        else:
            raise RuntimeError("Please provide `candidate_sampling_probability`.")

        return logits - torch.log(torch.clamp(probability, min=1e-6, max=1.0))
