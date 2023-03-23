from typing import List, Optional, Sequence

from torch import nn

from merlin.models.torch.core.combinators import SequentialBlock


class MLPBlock(SequentialBlock):
    """
    Multi-Layer Perceptron (MLP) Block with custom options for activation, normalization,
    dropout, and pre/post layers.

    Parameters
    ----------
    units : Sequence[int]
        Sequence of integers specifying the dimensions of each linear layer.
    activation : Callable, optional
        Activation function to apply after each linear layer. Default is ReLU.
    normalization : Union[str, nn.Module], optional
        Normalization method to apply after the activation function.
        Supported options are "batch_norm" or any custom `nn.Module`.
        Default is None (no normalization).
    dropout : Optional[float], optional
        Dropout probability to apply after the normalization.
        Default is None (no dropout).
    pre : nn.Module, optional
        An additional layer to add before the main MLP sequence.
        Default is None (no pre-layer).
    post : nn.Module, optional
        An additional layer to add after the main MLP sequence.
        Default is None (no post-layer).

    Raises
    ------
    ValueError
        If the normalization parameter is not supported.
    """

    def __init__(
        self,
        units: Sequence[int],
        activation=nn.ReLU,
        normalization=None,
        dropout: Optional[float] = None,
        pre=None,
        post=None,
    ):
        modules: List[nn.Module] = []

        if not isinstance(units, list):
            units = list(units)

        for dim in units:
            modules.append(nn.LazyLinear(dim))
            if activation is not None:
                modules.append(activation())

            if normalization:
                if normalization == "batch_norm":
                    modules.append(nn.LazyBatchNorm1d())
                elif isinstance(normalization, nn.Module):
                    modules.append(normalization)
                else:
                    raise ValueError(f"Normalization {normalization} not supported")

            if dropout:
                if isinstance(dropout, nn.Module):
                    modules.append(dropout)
                else:
                    modules.append(nn.Dropout(dropout))

        super().__init__(*modules, pre=pre, post=post)
