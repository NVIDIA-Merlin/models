from functools import reduce
from typing import Dict, Union

from torch import nn

from merlin.models.torch.core.aggregation import SumResidual
from merlin.models.torch.core.base import NoOp, TabularBlock
from merlin.models.torch.utils.torch_utils import apply_module


class ParallelBlock(TabularBlock):
    """
    A block that processes inputs in parallel through multiple layers and returns their outputs.

    Parameters
    ----------
    *inputs : Union[nn.Module, Dict[str, nn.Module]]
        Variable length list of PyTorch modules or dictionaries of PyTorch modules.
    pre : Callable, optional
        Preprocessing function to apply on inputs before processing.
    post : Callable, optional
        Postprocessing function to apply on outputs after processing.
    aggregation : Callable, optional
        Aggregation function to apply on outputs.
    """

    def __init__(
        self, *inputs: Union[nn.Module, Dict[str, nn.Module]], pre=None, post=None, aggregation=None
    ):
        super().__init__(pre, post, aggregation)

        if isinstance(inputs, tuple) and len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = inputs[0]

        if all(isinstance(x, dict) for x in inputs):
            self.parallel_dict = reduce(lambda a, b: dict(a, **b), inputs)
        elif all(isinstance(x, nn.Module) for x in inputs):
            self.parallel_dict = {i: m for i, m in enumerate(inputs)}
        else:
            raise ValueError(f"Invalid input. Got: {inputs}")

    def forward(self, inputs, **kwargs):
        """
        Process inputs through the parallel layers.

        Parameters
        ----------
        inputs : Tensor
            Input tensor to process through the parallel layers.
        **kwargs : dict
            Additional keyword arguments for layer processing.

        Returns
        -------
        outputs : dict
            Dictionary containing the outputs of the parallel layers.
        """
        outputs = {}

        for name, module in self.parallel_dict.items():
            module_inputs = inputs  # TODO: Add filtering when adding schema
            out = apply_module(module, module_inputs, **kwargs)
            if not isinstance(out, dict):
                out = {name: out}
            outputs.update(out)

        return outputs


class WithShortcut(ParallelBlock):
    def __init__(
        self,
        input: nn.Module,
        shortcut_filter=None,
        aggregation=None,
        post=None,
        block_outputs_name=None,
        **kwargs,
    ):
        block_outputs_name = block_outputs_name or input.name
        shortcut = shortcut_filter if shortcut_filter else NoOp()
        inputs = {block_outputs_name: input, "shortcut": shortcut}
        super().__init__(
            inputs,
            post=post,
            aggregation=aggregation,
            **kwargs,
        )


class ResidualBlock(WithShortcut):
    def __init__(
        self,
        input: nn.Module,
        activation=None,
        post=None,
        **kwargs,
    ):
        super().__init__(
            input,
            post=post,
            aggregation=SumResidual(activation=activation),
            **kwargs,
        )


class SequentialBlock(nn.Sequential):
    def __init__(self, *args, pre=None, post=None):
        super().__init__(*args)
        self.pre = pre
        self.post = post

    def __call__(self, inputs, *args, **kwargs):
        outputs = super().__call__(inputs)
        return outputs

    def add_with_shortcut(
        self,
        input,
        shortcut_filter=None,
        post=None,
        aggregation=None,
    ) -> "SequentialBlock":
        raise NotImplementedError()

    def add_with_residual(
        self,
        input,
        activation=None,
    ) -> "SequentialBlock":
        raise NotImplementedError()

    def add_branch(
        self,
        *branches,
        add_rest=False,
        post=None,
        aggregation=None,
        **kwargs,
    ) -> "SequentialBlock":
        raise NotImplementedError()

    def repeat(self, num: int = 1) -> "SequentialBlock":
        raise NotImplementedError()

    def repeat_in_parallel(
        self,
        num: int = 1,
        prefix=None,
        names=None,
        post=None,
        aggregation=None,
        copies=True,
        residual=False,
        **kwargs,
    ) -> "ParallelBlock":
        raise NotImplementedError()
