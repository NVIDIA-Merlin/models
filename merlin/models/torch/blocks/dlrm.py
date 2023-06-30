from typing import Dict, Optional

import torch
from torch import nn

from merlin.models.torch.block import Block
from merlin.models.torch.inputs.embedding import EmbeddingTables
from merlin.models.torch.inputs.tabular import TabularInputBlock
from merlin.models.torch.link import Link
from merlin.models.torch.transforms.agg import MaybeAgg, Stack
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.schema import Schema, Tags

_DLRM_REF = """
    References
    ----------
    ..  [1] Naumov, Maxim, et al. "Deep learning recommendation model for
        personalization and recommendation systems." arXiv preprint arXiv:1906.00091 (2019).
"""


@docstring_parameter(dlrm_reference=_DLRM_REF)
class DLRMInputBlock(TabularInputBlock):
    """Input block for DLRM model.

    Parameters
    ----------
    schema : Schema, optional
        The schema to use for selection. Default is None.
    dim : int
        The dimensionality of the output vectors.
    bottom_block : Block
        Block to pass the continuous features to.
        Note that, the output dimensionality of this block must be equal to ``dim``.

    {dlrm_reference}

    Raises
    ------
    ValueError
        If no categorical input is provided in the schema.

    """

    def __init__(self, schema: Schema, dim: int, bottom_block: Block):
        super().__init__(schema)
        self.add_route(Tags.CATEGORICAL, EmbeddingTables(dim, seq_combiner="mean"))
        self.add_route(Tags.CONTINUOUS, bottom_block)

        if "categorical" not in self:
            raise ValueError("DLRMInputBlock must have a categorical input")


@docstring_parameter(dlrm_reference=_DLRM_REF)
class DLRMInteraction(nn.Module):
    """
    This class defines the forward interaction operation as proposed
    in the DLRM
     `paper https://arxiv.org/pdf/1906.00091.pdf`_ [1]_.

    This forward operation performs elementwise multiplication
    of the embeddings followed by a reduction sum.

    {dlrm_reference}

    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "triu_indices"):
            self.register_buffer(
                "triu_indices", torch.triu_indices(inputs.shape[1], inputs.shape[1], offset=1)
            )

        interactions = torch.bmm(inputs, torch.transpose(inputs, 1, 2))
        interactions_flat = interactions[:, self.triu_indices[0], self.triu_indices[1]]

        return interactions_flat


class ShortcutConcatContinuous(Link):
    """
    A shortcut connection that concatenates
    continuous input features and intermediate outputs.

    When there's no continuous input, the intermediate output is returned.
    """

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        intermediate_output = self.output(inputs)

        if "continuous" in inputs:
            return torch.cat((inputs["continuous"], intermediate_output), dim=1)

        return intermediate_output


@docstring_parameter(dlrm_reference=_DLRM_REF)
class DLRMBlock(Block):
    """Builds the DLRM architecture, as proposed in the following
     `paper https://arxiv.org/pdf/1906.00091.pdf`_ [1]_.

    Parameters
    ----------
    schema : Schema, optional
        The schema to use for selection. Default is None.
    dim : int
        The dimensionality of the output vectors.
    bottom_block : Block
        Block to pass the continuous features to.
        Note that, the output dimensionality of this block must be equal to ``dim``.
    top_block : Block, optional
        An optional upper-level block of the model.
    interaction : nn.Module, default=DLRMInteraction()
        Interaction module for DLRM.

    {dlrm_reference}

    Raises
    ------
    ValueError
        If no categorical input is provided in the schema.
    """

    def __init__(
        self,
        schema: Schema,
        dim: int,
        bottom_block: Block,
        top_block: Optional[Block] = None,
        interaction: nn.Module = DLRMInteraction(),
    ):
        super().__init__(DLRMInputBlock(schema, dim, bottom_block))

        # link = ShortcutConcatContinuous() if "continuous" in self[0] else None
        self.append(Block(MaybeAgg(Stack(dim=1)), interaction), link=ShortcutConcatContinuous())

        if top_block:
            self.append(top_block)
