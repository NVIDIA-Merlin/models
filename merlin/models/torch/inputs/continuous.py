from merlin.models.torch.transforms.features import Filter
from merlin.models.torch.typing import TabularData
from merlin.schema import Schema


class Continuous(Filter):
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
