from typing import Callable, Dict, Optional, Union

from torch import nn

from merlin.models.torch.combinators import ParallelBlock
from merlin.models.torch.inputs.continuous import Continuous
from merlin.models.torch.inputs.embedding import Embeddings
from merlin.schema import Schema, Tags


class TabularInputBlock(ParallelBlock):
    INPUT_TAG_TO_BLOCK: Dict[Tags, Callable[[Schema], nn.Module]] = {
        Tags.CONTINUOUS: Continuous,
        Tags.CATEGORICAL: Embeddings,
    }
    TAGS_TO_EXCLUDE = [Tags.TARGET, Tags.BINARY_CLASSIFICATION, Tags.REGRESSION]

    def __init__(
        self,
        schema: Optional[Schema] = None,
        categorical: Union[Tags, nn.Module] = Tags.CATEGORICAL,
        continuous: Union[Tags, nn.Module] = Tags.CONTINUOUS,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        aggregation: Optional[Union[nn.Module, str]] = None,
        tag_to_block=INPUT_TAG_TO_BLOCK,
        flatten=True,
        **branches,
    ):
        # If targets are passed, exclude these from the input block schema
        schema = schema.excluding_by_tag(self.TAGS_TO_EXCLUDE)

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

        if flatten:
            flattened = {}
            for key, val in parsed.items():
                if isinstance(val, ParallelBlock):
                    for name, branch in val.items():
                        flattened[f"{key}_{name}"] = branch
                else:
                    flattened[key] = val
            parsed = flattened

        super().__init__(parsed, pre=pre, post=post, aggregation=aggregation)

    @classmethod
    def block_to_tag(cls) -> Dict[nn.Module, Tags]:
        return {value: key for key, value in cls.INPUT_TAG_TO_BLOCK.items()}

    @property
    def input_schema(self) -> Schema:
        return self.schema
