from typing import Optional, Sequence, Union

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchmetrics import AUROC, Accuracy, Metric, Precision, Recall

from merlin.models.torch.blocks.mlp import MLPBlock
from merlin.models.torch.inputs.embedding import EmbeddingTable
from merlin.models.torch.outputs.base import ModelOutput
from merlin.schema import ColumnSchema, Schema


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
        to_call=MLPBlock([1], activation=nn.Sigmoid),
        default_loss=nn.BCELoss(),
        default_metrics: Sequence[Metric] = (
            Accuracy("binary"),
            AUROC("binary"),
            Precision("binary"),
            Recall("binary"),
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


# "CategoricalTarget", "EmbeddingTablePrediction"
class CategoricalOutput(ModelOutput):
    def __init__(
        self,
        to_call: Union[
            Schema,
            ColumnSchema,
            EmbeddingTable,
        ],
        target_name: str = None,
        pre=None,
        post=None,
        logits_temperature: float = 1.0,
        default_loss=nn.BCELoss(),
        default_metrics: Sequence[Metric] = (),
    ):
        if to_call is not None:
            if isinstance(to_call, (Schema, ColumnSchema)):
                _to_call = CategoricalTarget(to_call)
                if isinstance(to_call, Schema):
                    to_call = to_call.first
                target_name = target_name or to_call.name
            # elif isinstance(to_call, EmbeddingTable):
            #     _to_call = EmbeddingTablePrediction(to_call)
            #     target_name = _to_call.table.col_schema.name
            else:
                _to_call = to_call

        super().__init__(
            to_call=_to_call,
            default_loss=default_loss,
            default_metrics=default_metrics,
            target=target_name,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
        )


class CategoricalTarget(nn.Module):
    def __init__(
        self,
        feature: Union[Schema, ColumnSchema] = None,
        activation=None,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(feature, Schema):
            assert len(feature) == 1, "Schema can have max 1 feature"
            col_schema = feature.first
        else:
            col_schema = feature

        self.schema = col_schema
        self.target_name = col_schema.name
        self.num_classes = col_schema.int_domain.max + 1

        self.linear = nn.LazyLinear(self.num_classes, bias=bias)
        if activation is not None:
            self.activation = activation()

    def forward(self, inputs: torch.Tensor):
        output = self.linear(inputs)
        if hasattr(self, "activation"):
            output = self.activation(output)

        return output

    def embedding_lookup(self, inputs: torch.Tensor):
        return torch.index_select(self.embeddings, 1, inputs)

    @property
    def embeddings(self) -> Parameter:
        return self.linear.weight.t()
