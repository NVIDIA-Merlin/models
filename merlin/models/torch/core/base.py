from typing import Optional

import torch


class Block(torch.nn.Module):
    """Base Block class."""

    def __init__(
        self, pre: Optional[torch.nn.Module] = None, post: Optional[torch.nn.Module] = None
    ) -> None:
        super().__init__()
        self.pre = pre
        self.post = post

    def __call__(self, *args, **kwargs):
        if self.pre is not None:
            x = self.pre(*args, **kwargs)
        x = super().__call__(x)
        if self.post is not None:
            x = self.post(x)
        return x
