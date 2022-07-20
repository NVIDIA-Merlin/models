from typing import Optional

import torch


class SequentialBlock(torch.nn.Sequential):
    def __init__(self, *args, pre=None, post=None):
        """Create a composition.

        Parameters
        ----------
        *args:
            A list or tuple of layers to compose.
        """

        super().__init__(*args)

    @property
    def inputs(self):
        from merlin.models.torch import TabularFeatures

        first = list(self)[0]
        if isinstance(first, TabularFeatures):
            return first
