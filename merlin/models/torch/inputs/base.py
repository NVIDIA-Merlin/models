from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union, List

from torch import nn

from merlin.models.torch.combinators import ParallelBlock
from merlin.models.torch.inputs.continuous import Continuous
from merlin.models.torch.inputs.embedding import Embeddings
from merlin.schema import Schema, Tags
from merlin.models.torch.inputs.base import branches


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
        tags_to_exclude=TAGS_TO_EXCLUDE,
        flatten=True,
        **branches,
    ):
        # If targets are passed, exclude these from the input block schema
        schema = schema.excluding_by_tag(tags_to_exclude)

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
                        # TODO: Check for name collisions
                        flattened[f"{key}_{name}"] = branch
                else:
                    flattened[key] = val
            parsed = flattened

        super().__init__(parsed, pre=pre, post=post, aggregation=aggregation)

    @classmethod
    def block_to_tag(cls) -> Dict[nn.Module, Tags]:
        return {value: key for key, value in cls.INPUT_TAG_TO_BLOCK.items()}

    @classmethod
    def filter_schema_for_branch(cls, schema: Schema, branch: nn.Module) -> Schema:
        if branch in cls.block_to_tag():
            return schema.select_by_tag(cls.block_to_tag()[branch])
        else:
            raise ValueError(f"Branch {branch} not found in block_to_tag mapping")

    @property
    def input_schema(self) -> Schema:
        return self.schema


def create_branch(
    selector: Union[Callable[[Schema], Schema], Tags],
    module: Callable[[Schema], nn.Module]
) -> Optional[nn.Module]:
    def branch(schema: Schema):
        if isinstance(selector, Tags):
            selected_schema = schema.select_by_tag(selector)
        else:
            selected_schema = selector(schema)
            
        if not selected_schema:
            return None
        
        return module(selected_schema)
    
    return branch


def default_parsing(
    schema: Optional[Schema],
) -> Dict[str, nn.Module]:
    return {
        "continuous": create_branch(Tags.CONTINUOUS, Continuous)(schema),
        "categorical": create_branch(Tags.CATEGORICAL, Embeddings)(schema),
    }
    
    
from merlin.models.torch.base import Block, Parallel


class FeatureBlock(Block):
    def __init__(
        self, 
        schema: Schema,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        name: Optional[str] = None,
    ):
        super().__init__(Parallel(), pre=pre, post=post, name=name)
        self._schema = schema
        
    def add_features(
        self,
        selector: Union[Callable[[Schema], Schema], Tags], 
        module_cls: Callable[[Schema], nn.Module],
        optional: bool = True,
    ):
        if isinstance(selector, Tags):
            selected = self._schema.select_by_tag(selector)
        else:
            selected = selector(self._schema)
            
        if not selected:
            if optional:
                return self
            else:
                raise ValueError("Schema was empty after selection")
        
        self.block[0][str(selector)] = module_cls(selected)
        
        return self
    
    
# inputs = TabularInputBlock(schema)
# inputs = TabularInputBlock(schema, features="seq-cross-attention")
schema: Schema = Schema()

block = FeatureBlock(schema)
block.add_features(Tags.CONTINUOUS, Continuous)
block.add_features(Tags.CATEGORICAL, Embeddings)
block.append("concat")


class DefaultFeatureBlock(FeatureBlock):
    def __init__(
        self, 
        schema: Schema,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        name: Optional[str] = None,
    ):
        super().__init__(schema, pre=pre, post=post, name=name)
        self.add_features(Tags.CONTINUOUS, Continuous)
        self.add_features(Tags.CATEGORICAL, Embeddings)
        
        
class Broadcast2DTo3D(FeatureBlock):
    def __init__(
        self, 
        schema,
        context=Tags.CONTEXT, 
        sequence=Tags.SEQUENCE,
        features=DefaultFeatureBlock,
    ):
        super().__init__(schema)
        self.add_features(schema, features(schema))
        
        self.append(BroadcastToSequence(context, sequence))



# features is optional we default to the same behaviour as currently
inputs = TabularInputBlock(schema, features=block)

register_feature_block = lambda x: x


BroadcastToSequence = ...


class Broadcast2DTo3D(FeatureBlock):
    def __init__(
        self, 
        schema, 
        context=Tags.CONTEXT, 
        sequence=Tags.SEQUENCE
    ):
        super().__init__(schema)
        self.add_features(schema, FeatureBlock(schema))
        # self.add_features(Tags.Continuous, Continuous)
        # self.add_features(Tags.Categorical, Embeddings)
        
        self.append(BroadcastToSequence(context, sequence))
        
        
        
class TabularInputBlock2(ParallelBlock):
    TAGS_TO_EXCLUDE = [Tags.TARGET, Tags.BINARY_CLASSIFICATION, Tags.REGRESSION]

    def __init__(
        self,
        schema,
        features=DefaultFeatureBlock,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        aggregation: Optional[Union[nn.Module, str]] = None,
        tags_to_exclude=TAGS_TO_EXCLUDE,
        flatten: bool = True,
        **branches,
    ):
        # If targets are passed, exclude these from the input block schema
        schema = schema.excluding_by_tag(tags_to_exclude)

        # TODO: parse if it's a string
        parsed = features(schema)

        if not parsed:
            raise ValueError("No columns selected for the input block")

        if flatten:
            flattened = {}
            for key, val in parsed.items():
                if isinstance(val, ParallelBlock):
                    for name, branch in val.items():
                        # TODO: Check for name collisions
                        flattened[f"{key}_{name}"] = branch
                else:
                    flattened[key] = val
            parsed = flattened

        super().__init__(parsed, pre=pre, post=post, aggregation=aggregation)

    @property
    def input_schema(self) -> Schema:
        return self.schema
        

# InputBlockV2(
#     schema,
#     post=BroadcastToSequence(context_schema, sequence_schema)
# )


config = {
    "config": "default",
    
}

DEFAULT_TO_EXCLUDE  = [Tags.TARGET, Tags.BINARY_CLASSIFICATION, Tags.REGRESSION]

@dataclass
class TabularInputConfig:
    parser: SchemaModuleDict
    flatten: bool = True
    tags_to_exclude: List[Tags] = field(default_factory=lambda: list(DEFAULT_TO_EXCLUDE))