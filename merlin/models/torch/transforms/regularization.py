import torch
import torch.nn as nn

from merlin.models.utils.doc_utils import docstring_parameter

_RMSNORM_REF = """
    ..  [1] Zhang and Sennrich, "Root Mean Square Layer Normalization".
        arXiv preprintarXiv:1910.07467 (2019).
"""


@docstring_parameter(rmsnorm_reference=_RMSNORM_REF)
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization as proposed in [1].

    References
    ----------
    {rmsnorm_reference}
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        rms = tensor.to(torch.float32).square().mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (tensor * rms).to(tensor.dtype) * self.scale
