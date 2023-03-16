from functools import reduce
from typing import Dict, List, Union

from torch import nn

from merlin.models.torch.core.aggregation import SumResidual
from merlin.models.torch.core.base import NoOp, TabularBlock
from merlin.models.torch.utils.torch_utils import apply_module


class ParallelBlock(TabularBlock):
    def __init__(
        self, *inputs: Union[nn.Module, Dict[str, nn.Module]], pre=None, post=None, aggregation=None
    ):
        super().__init__(pre, post, aggregation)
        if isinstance(inputs, tuple) and len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = inputs[0]
        if all(isinstance(x, dict) for x in inputs):
            to_merge: Dict[str, nn.Module] = reduce(
                lambda a, b: dict(a, **b), inputs
            )  # type: ignore
            parsed_to_merge: Dict[str, TabularBlock] = {}
            for key, val in to_merge.items():
                parsed_to_merge[key] = val
            self.parallel_layers = parsed_to_merge
        elif all(isinstance(x, nn.Module) for x in inputs):
            # if use_layer_name:
            #     self.parallel_layers = {layer.name: layer for layer in inputs}
            # else:
            parsed: List[TabularBlock] = []
            for inp in inputs:
                parsed.append(inp)  # type: ignore
            self.parallel_layers = parsed
        else:
            raise ValueError(
                "Please provide one or multiple layer's to merge or "
                f"dictionaries of layer. got: {inputs}"
            )

    def forward(self, inputs, **kwargs):
        outputs = {}

        for name, layer in self.parallel_dict.items():
            layer_inputs = self._maybe_filter_layer_inputs_using_schema(name, layer, inputs)
            out = apply_module(layer, layer_inputs, **kwargs)
            if not isinstance(out, dict):
                out = {name: out}
            outputs.update(out)

        return outputs

    @property
    def parallel_values(self) -> List[nn.Module]:
        if isinstance(self.parallel_layers, dict):
            return list(self.parallel_layers.values())

        return self.parallel_layers

    @property
    def parallel_dict(self) -> Dict[Union[str, int], nn.Module]:
        if isinstance(self.parallel_layers, dict):
            return self.parallel_layers

        return {i: m for i, m in enumerate(self.parallel_layers)}


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
