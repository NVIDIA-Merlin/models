from typing import Dict, Optional, Union

import torch

from merlin.models.torch.core.tabular import TabularAggregationType, TabularBlock
from merlin.schema import Schema


class SequentialBlock(torch.nn.Sequential):
    def __init__(self, *args, pre=None, post=None):
        """Create a composition.

        Parameters
        ----------
        *args:
            A list or tuple of layers to compose.
        """

        super().__init__(*args)

    @property
    def inputs(self):
        from merlin.models.torch import TabularFeatures

        first = list(self)[0]
        if isinstance(first, TabularFeatures):
            return first


# TODO
class ParallelBlock(TabularBlock):
    def __init__(
        self,
        *inputs: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
        pre: Optional[torch.nn.Module] = None,
        post: Optional[torch.nn.Module] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        strict: bool = False,
        automatic_pruning: bool = True,
        **kwargs,
    ):
        super().__init__(
            pre=pre, post=post, aggregation=aggregation, schema=schema, name=name, **kwargs
        )
