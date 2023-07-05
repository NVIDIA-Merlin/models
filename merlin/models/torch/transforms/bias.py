import torch
from torch import nn


class LogitsTemperatureScaler(nn.Module):
    """
    A PyTorch Module for scaling logits with a given temperature value.

    This module is useful for implementing temperature scaling in neural networks,
    a technique often used to soften or sharpen the output distribution of a classifier.
    A temperature value closer to 0 makes the output probabilities more extreme
    (either closer to 0 or 1), while a value closer to 1 makes the distribution
    closer to uniform.

    Parameters
    ----------
    temperature : float
        The temperature value used for scaling. Must be a positive float in the range (0.0, 1.0].

    Raises
    ------
    ValueError
        If the temperature value is not a float or is out of the range (0.0, 1.0].
    """

    def __init__(self, temperature: float):
        super().__init__()

        if not isinstance(temperature, float):
            raise ValueError(f"Invalid temperature type: {type(temperature)}")
        if not 0.0 < temperature <= 1.0:
            raise ValueError(
                f"Invalid temperature value: {temperature} ", "Must be in the range (0.0, 1.0]"
            )

        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to the input logits.

        Parameters
        ----------
        logits : torch.Tensor
            The input logits to be scaled.

        Returns
        -------
        torch.Tensor
            The scaled logits.
        """
        return logits / self.temperature
