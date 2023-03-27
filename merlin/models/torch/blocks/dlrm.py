from typing import Dict, Optional

from torch import nn

from merlin.models.torch.blocks.interaction import DLRMInteraction
from merlin.models.torch.blocks.mlp import MLPBlock
from merlin.models.torch.combinators import ParallelBlock, SequentialBlock
from merlin.models.torch.inputs.base import TabularInputBlock
from merlin.models.torch.inputs.continuous import Continuous
from merlin.models.torch.inputs.embedding import Embeddings
from merlin.models.torch.transforms.features import Filter
from merlin.schema import Schema


class DLRMBlock(SequentialBlock):
    def __init__(
        self,
        schema: Schema,
        *,
        embedding_dim: Optional[int] = None,
        embeddings: Optional[nn.Module] = None,
        bottom_block: Optional[nn.Module] = None,
        top_block: Optional[nn.Module] = None,
        interaction_block=DLRMInteraction(),
        pre=None,
        post=None
    ):
        schema = schema.excluding_by_tag(TabularInputBlock.TAGS_TO_EXCLUDE)
        if not embedding_dim or embeddings:
            raise ValueError("Must specify embedding_dim or embeddings")

        if not bottom_block:
            bottom_block = MLPBlock([embedding_dim])
        if not embeddings:
            cat_tags = TabularInputBlock.block_to_tag()[Embeddings]
            embeddings = Embeddings(schema.select_by_tag(cat_tags), dim=embedding_dim)

        con_tags = TabularInputBlock.block_to_tag()[Continuous]
        continuous = Continuous(schema.select_by_tag(con_tags), post=bottom_block)
        if continuous.schema:
            inputs = TabularInputBlock(
                schema,
                continuous=Continuous(post=bottom_block),
                embeddings=embeddings,
                flatten=False,
            )
        else:
            inputs = embeddings

        _modules = [inputs, interaction_block]

        if not continuous.schema:
            if top_block:
                _modules.append(top_block)
        else:
            _modules.append(WithContinuousShortcut({"interactions": interaction_block}))

        super().__init__(*_modules, pre=pre, post=post)


class WithContinuousShortcut(ParallelBlock):
    def __init__(self, inputs: Dict[str, nn.Module], pre=None, post=None, aggregation="concat"):
        shortcut_branch = Filter(Schema(["continuous"]))

        super().__init__(
            {**inputs, "continuous": shortcut_branch}, pre=pre, post=post, aggregation=aggregation
        )
