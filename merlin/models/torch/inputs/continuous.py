from typing import Callable, Union

from merlin.models.torch.transforms.features import Filter
from merlin.models.torch.typing import TabularData
from merlin.schema import Schema, Tags


class Continuous(Filter):
    @classmethod
    def with_selector(
        cls, 
        schema: Schema, 
        selector: Union[Callable[[Schema], Schema], Tags]
    ) -> "Continuous":
        if isinstance(selector, Tags):
            selector = lambda schema: schema.select_by_tag(selector)
        
        return cls(selector(schema))
    
    def forward(self, inputs: TabularData) -> TabularData:
        outputs = {}
        for key, val in super().forward(inputs).items():
            # Add an extra dim to the end of the tensor
            if val.ndim == 1:
                outputs[key] = val.unsqueeze(-1)
            else:
                outputs[key] = val

        return outputs

    @property
    def input_schema(self) -> Schema:
        return self.schema
