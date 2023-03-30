from typing import Optional, Sequence, Union

from torch import nn
from torchmetrics import MeanSquaredError, Metric

import merlin.dtypes as md
from merlin.models.torch.outputs.base import ModelOutput
from merlin.schema import ColumnSchema, Schema


class RegressionOutput(ModelOutput):
    """
    A RegressionOutput block represents the output of a regression task.

    It the target, loss, and metrics computation.

    Parameters
    ----------
    target : Optional[Union[str, ColumnSchema]], optional, default: None
        The target column name or ColumnSchema object.
    to_call : nn.Module, optional, default: nn.LazyLinear(1)
        The PyTorch module to be applied to the inputs.
    default_loss : str, optional, default: "mse"
        The default loss function to use during training.
    default_metrics : Sequence[Metric], optional, default: (MeanSquaredError(),)
        A sequence of Metric objects to be used during evaluation.
    pre : Optional[Callable], optional, default: None
        A callable applied to the inputs before applying `to_call`.
    post : Optional[Callable], optional, default: None
        A callable applied to the outputs after applying `to_call`.
    logits_temperature : float, optional, default: 1.0
        The temperature for scaling logits. Not implemented yet.

    """

    def __init__(
        self,
        target: Optional[Union[str, ColumnSchema]] = None,
        to_call=nn.LazyLinear(1),
        default_loss=nn.MSELoss(),
        default_metrics: Sequence[Metric] = (MeanSquaredError(),),
        pre=None,
        post=None,
        logits_temperature: float = 1,
    ):
        super().__init__(
            to_call=to_call,
            default_loss=default_loss,
            default_metrics=default_metrics,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
        )

    def create_output_schema(self, target: ColumnSchema) -> Schema:
        """Return the output schema given the target column schema."""
        _target = target.with_dtype(md.float32)

        return Schema([_target])
