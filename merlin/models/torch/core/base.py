from torch import nn


class Block(nn.Module):
    def __init__(self, pre=None, post=None):
        super().__init__()
        self.pre = pre
        self.post = post

    def __call__(self, inputs, *args, **kwargs):
        if self.pre is not None:
            inputs = self.pre(inputs, *args, **kwargs)

        outputs = super().__call__(inputs)

        if self.post is not None:
            outputs = self.post(outputs, *args, **kwargs)

        return outputs

    def forward(self, x):
        return x


class TabularBlock(Block):
    def __init__(self, pre=None, post=None, aggregation=None):
        super().__init__(pre=pre, post=post)
        self.aggregation = aggregation

    def __call__(self, inputs, *args, **kwargs):
        outputs = super().__call__(inputs, *args, **kwargs)

        if self.aggregation is not None:
            outputs = self.aggregation(outputs, *args, **kwargs)

        return outputs

    def forward(self, x):
        return x


class NoOp(nn.Module):
    def forward(self, inputs):
        return inputs
