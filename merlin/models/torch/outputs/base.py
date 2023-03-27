from typing import Optional, Sequence, Union

import torch
from torch import nn
from torchmetrics import Metric

from merlin.models.torch.base import Block
from merlin.models.torch.data import register_target_hook
from merlin.models.torch.utils.module_utils import apply
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
        register_target_hook(self)
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
        return apply(self.to_call, inputs, training=training, testing=testing, **kwargs)

    @property
    def output(self) -> torch.Tensor:
        if self.target is not None:
            attr_name = f"__buffer_target_{self.target}"
        else:
            attr_name = "__buffer_target"

        if not hasattr(self, attr_name):
            return None

        return getattr(self, attr_name)


class DotProduct(nn.Module):
    """Dot-product between queries & items.
    Parameters:
    -----------
    query_name : str, optional
        Identify query tower for query/user embeddings, by default 'query'
    item_name : str, optional
        Identify item tower for item embeddings, by default 'item'
    """

    def __init__(self, query_name: str = "query", item_name: str = "candidate"):
        super().__init__()
        self.query_name = query_name
        self.item_name = item_name

    def forward(self, inputs):
        if isinstance(inputs, dict):
            try:
                query = inputs[self.query_name]
                item = inputs[self.item_name]
            except KeyError as e:
                raise RuntimeError(
                    f"Key {e} not found in input dictionary. "
                    "Please provide a dictionary with keys "
                    f"'{self.query_name}' and '{self.item_name}'."
                ) from e
        elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            query = inputs[0]
            item = inputs[1]
        else:
            raise RuntimeError(
                "Invalid input type. "
                "Expected inputs to be either a dictionary with keys "
                f"'{self.query_name}' and '{self.item_name}' or a tuple/list of size 2."
            )

        # Alternative is: torch.einsum('...i,...i->...', query, item)
        return torch.sum(query * item, dim=-1, keepdim=True)
