from typing import Optional, Sequence, Union

from torch import nn
from torchmetrics import Metric

from merlin.models.torch.base import Block
from merlin.models.torch.data import register_target_hook
from merlin.models.torch.utils.module_utils import apply
from merlin.schema import ColumnSchema, Schema


class ModelOutput(Block):
    """A ModelOutput block represents the output of a model.
    It handles the target, loss, and metrics computation.

    Parameters
    ----------
    to_call : nn.Module
        The PyTorch module to be applied to the inputs.
    default_loss : Union[str, nn.Module]
        The default loss function to use during training.
    default_metrics : Sequence[Metric], optional, default: ()
        A sequence of Metric objects to be used during evaluation.
    target : Optional[Union[str, ColumnSchema]], optional, default: None
        The target column name or ColumnSchema object.
    pre : Optional[Callable], optional, default: None
        A callable applied to the inputs before applying `to_call`.
    post : Optional[Callable], optional, default: None
        A callable applied to the outputs after applying `to_call`.
    logits_temperature : float, optional, default: 1.0
        The temperature for scaling logits. Not implemented yet.

    """

    def __init__(
        self,
        to_call: nn.Module,
        default_loss: Union[str, nn.Module],
        default_metrics: Sequence[Metric] = (),
        target: Optional[Union[str, ColumnSchema]] = None,
        pre=None,
        post=None,
        logits_temperature: float = 1.0,
        name=None,
    ):
        super().__init__(pre=pre, post=post)
        register_target_hook(self)
        self.to_call = to_call
        self.default_loss = default_loss
        self.default_metrics = default_metrics

        if not isinstance(target, ColumnSchema):
            target = ColumnSchema(target)

        self.target = target.name
        self.target_col = target
        self._name = name or self.target

        if logits_temperature != 1.0:
            raise NotImplementedError("Logits temperature is not implemented yet.")

    def forward(self, inputs, targets=None):
        """
        Apply `self.to_call` module to the inputs and return the output.
        """
        outputs = apply(self.to_call, inputs, targets=targets)

        if targets is not None and not isinstance(outputs, tuple):
            if isinstance(targets, dict):
                _target = targets[self.target]
            else:
                _target = targets

            return outputs, _target

        return outputs

    def create_output_schema(self, target: ColumnSchema) -> Schema:
        """Return the output schema given the target column schema."""
        return Schema([target])

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_schema(self) -> Schema:
        return self.create_output_schema(self.target_col)

    @property
    def is_in_training(self) -> bool:
        return getattr(self, "training", False)

    @property
    def is_in_testing(self) -> bool:
        return getattr(self, "testing", False)
