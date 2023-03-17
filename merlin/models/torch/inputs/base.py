from typing import Callable, Dict, Optional, Union

from torch import nn

from merlin.models.torch.core.combinators import ParallelBlock
from merlin.models.torch.inputs.embedding import Embeddings
from merlin.models.torch.transforms.features import Filter
from merlin.schema import Schema, Tags

INPUT_TAG_TO_BLOCK: Dict[Tags, Callable[[Schema], nn.Module]] = {
    Tags.CONTINUOUS: Filter,
    Tags.CATEGORICAL: Embeddings,
}


class TabularInputBlock(ParallelBlock):
    def __init__(
        self,
        schema: Optional[Schema] = None,
        categorical: Union[Tags, nn.Module] = Tags.CATEGORICAL,
        continuous: Union[Tags, nn.Module] = Tags.CONTINUOUS,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        aggregation: Optional[nn.Module] = "concat",
        tag_to_block=INPUT_TAG_TO_BLOCK,
        **branches,
    ):
        # If targets are passed, exclude these from the input block schema
        schema = schema.excluding_by_tag([Tags.TARGET, Tags.BINARY_CLASSIFICATION, Tags.REGRESSION])

        unparsed = {"categorical": categorical, "continuous": continuous, **branches}
        parsed = {}
        for name, branch in unparsed.items():
            if isinstance(branch, nn.Module):
                parsed[name] = branch
            else:
                if not isinstance(schema, Schema):
                    raise ValueError(
                        "If you pass a column-selector as a branch, "
                        "you must also pass a `schema` argument."
                    )
                if branch not in tag_to_block:
                    raise ValueError(f"No default-block provided for {branch}")
                branch_schema: Schema = schema.select_by_tag(branch)
                if branch_schema:
                    parsed[name] = tag_to_block[branch](branch_schema)

        if not parsed:
            raise ValueError("No columns selected for the input block")

        super().__init__(parsed, pre=pre, post=post, aggregation=aggregation)
