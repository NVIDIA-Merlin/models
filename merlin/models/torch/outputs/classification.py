from typing import Optional, Sequence, Union

from torch import nn
from torchmetrics import AUROC, Accuracy, Metric, Precision, Recall

from merlin.models.torch.blocks.mlp import MLPBlock
from merlin.models.torch.outputs.base import ModelOutput
from merlin.schema import ColumnSchema


class BinaryOutput(ModelOutput):
    """
    Binary-classification prediction block.

    Parameters
    ----------
    target: Union[str, Schema], optional
        The name of the target. If a Schema is provided, the target is inferred from the schema.
    pre: Optional[Block], optional
        Optional block to transform predictions before computing the binary logits,
        by default None
    post: Optional[Block], optional
        Optional block to transform the binary logits,
        by default None
    name: str, optional
        The name of the task.
    task_block: Block, optional
        The block to use for the task.
    logits_temperature: float, optional
        Parameter used to reduce model overconfidence, so that logits / T.
        by default 1.
    default_loss: Union[str, tf.keras.losses.Loss], optional
        Default loss to use for binary-classification
        by 'binary_crossentropy'
    default_metrics_fn: Callable
        A function returning the list of default metrics
        to use for binary-classification
    """

    def __init__(
        self,
        target: Optional[Union[str, ColumnSchema]] = None,
        to_call=MLPBlock(1, activation=nn.Sigmoid),
        default_loss=nn.BCELoss(),
        default_metrics: Sequence[Metric] = (
            Accuracy(),
            AUROC(),
            Precision(),
            Recall(),
        ),
        pre=None,
        post=None,
        logits_temperature: float = 1.0,
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
