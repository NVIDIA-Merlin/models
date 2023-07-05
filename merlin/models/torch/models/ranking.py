from typing import Optional

from torch import nn

from merlin.models.torch.block import Block, ParallelBlock
from merlin.models.torch.blocks.cross import CrossBlock
from merlin.models.torch.blocks.dlrm import DLRMBlock
from merlin.models.torch.blocks.mlp import MLPBlock
from merlin.models.torch.inputs.tabular import TabularInputBlock
from merlin.models.torch.models.base import Model
from merlin.models.torch.outputs.tabular import TabularOutputBlock
from merlin.models.torch.transforms.agg import Concat, MaybeAgg
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
    ...    output_block=BinaryOutput(ColumnSchema("target")))
    >>> trainer = pl.Trainer()
    >>> model.initialize(dataloader)
    >>> trainer.fit(model, dataloader)

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


class DCNModel(Model):
    """
    The Deep & Cross Network (DCN) architecture as proposed in Wang, et al. [1]

    Parameters
    ----------
    schema : Schema
        The schema to use for selection.
    depth : int, optional
        Number of cross-layers to be stacked, by default 1
    deep_block : Block, optional
        The `Block` to use as the deep part of the model (typically a `MLPBlock`)
    stacked : bool
        Whether to use the stacked version of the model or the parallel version.
    input_block : Block, optional
        The `Block` to use as the input layer. If None, a default `TabularInputBlock` object
        is instantiated, that creates the embedding tables for the categorical features
        based on the schema. The embedding dimensions are inferred from the features
        cardinality. For a custom representation of input data you can instantiate
        and provide a `TabularInputBlock` instance.

    Returns
    -------
    Model
        An instance of Model class representing the fully formed DCN.

    Example usage
    -------------
    >>> model = mm.DCNModel(
    ...    schema,
    ...    depth=2,
    ...    deep_block=mm.MLPBlock([256, 64]),
    ...    output_block=BinaryOutput(ColumnSchema("target")))
    >>> trainer = pl.Trainer()
    >>> model.initialize(dataloader)
    >>> trainer.fit(model, dataloader)

    References
    ----------
    [1]. Wang, Ruoxi, et al. "DCN V2: Improved deep & cross network and
       practical lessons for web-scale learning to rank systems." Proceedings
       of the Web Conference 2021. 2021. https://arxiv.org/pdf/2008.13535.pdf
    """

    def __init__(
        self,
        schema: Schema,
        depth: int = 1,
        deep_block: Optional[Block] = None,
        stacked: bool = True,
        input_block: Optional[Block] = None,
        output_block: Optional[Block] = None,
    ) -> None:
        if deep_block is None:
            deep_block = MLPBlock([512, 256])

        if input_block is None:
            input_block = TabularInputBlock(schema, init="defaults")

        if output_block is None:
            output_block = TabularOutputBlock(schema, init="defaults")

        if stacked:
            cross_network = Block(CrossBlock.with_depth(depth), deep_block)
        else:
            cross_network = Block(
                ParallelBlock({"cross": CrossBlock.with_depth(depth), "deep": deep_block}),
                MaybeAgg(Concat()),
            )

        super().__init__(input_block, *cross_network, output_block)
