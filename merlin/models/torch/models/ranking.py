from typing import Optional

from torch import nn

from merlin.models.torch.block import Block
from merlin.models.torch.blocks.dlrm import DLRMBlock
from merlin.models.torch.models.base import Model
from merlin.models.torch.outputs.tabular import TabularOutputBlock
from merlin.schema import Schema


class DLRMModel(Model):
    """
    The Deep Learning Recommendation Model (DLRM) as proposed in Naumov, et al. [1]

    Parameters
    ----------
    schema : Schema
        The schema to use for selection.
    dim : int
        The dimensionality of the output vectors.
    bottom_block : Block
        Block to pass the continuous features to.
        Note that, the output dimensionality of this block must be equal to ``dim``.
    top_block : Block, optional
        An optional upper-level block of the model.
    interaction : nn.Module, optional
        Interaction module for DLRM.
        If not provided, DLRMInteraction will be used by default.
    output_block : Block, optional
        The output block of the model, by default None.
        If None, a TabularOutputBlock with schema and default initializations is used.

    Returns
    -------
    Model
        An instance of Model class representing the fully formed DLRM.

    Example usage
    -------------
    >>> model = mm.DLRMModel(
    ...    schema,
    ...    dim=64,
    ...    bottom_block=mm.MLPBlock([256, 64]),
    ...    output_block=BinaryOutput(ColumnSchema("target"))
    ... )
    >>> trainer = pl.Trainer()
    >>> trainer.fit(model, Loader(dataset, batch_size=32))

    References
    ----------
    [1] Naumov, Maxim, et al. "Deep learning recommendation model for
        personalization and recommendation systems." arXiv preprint arXiv:1906.00091 (2019).
    """

    def __init__(
        self,
        schema: Schema,
        dim: int,
        bottom_block: Block,
        top_block: Optional[Block] = None,
        interaction: Optional[nn.Module] = None,
        output_block: Optional[Block] = None,
    ) -> None:
        if output_block is None:
            output_block = TabularOutputBlock(schema, init="defaults")

        dlrm_body = DLRMBlock(
            schema,
            dim,
            bottom_block,
            top_block=top_block,
            interaction=interaction,
        )

        super().__init__(dlrm_body, output_block)
