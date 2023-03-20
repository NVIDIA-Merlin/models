from typing import Optional, Sequence, Union

from torch import nn
from torchmetrics import Metric

from merlin.models.torch.core.base import Block
from merlin.models.torch.utils.torch_utils import apply_module
from merlin.schema import ColumnSchema


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
    ):
        super().__init__(pre=pre, post=post)
        self.to_call = to_call
        self.default_loss = default_loss
        self.default_metrics = default_metrics

        if isinstance(target, ColumnSchema):
            target = target.name
        self.target = target

        if logits_temperature != 1.0:
            raise NotImplementedError("Logits temperature is not implemented yet.")

    def forward(self, inputs, training=False, testing=False, **kwargs):
        """
        Apply `self.to_call` module to the inputs and return the output.
        """
        return apply_module(self.to_call, inputs, training=training, testing=testing, **kwargs)
