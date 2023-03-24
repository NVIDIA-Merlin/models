from typing import Optional

from torch import nn

from merlin.models.torch.data import TabularData, TargetMixin
from merlin.schema import Schema, Tags


class InBatchNegatives(nn.Module, TargetMixin):
    def __init__(
        self,
        schema: Schema,
        n_per_positive: int,
        seed: Optional[int] = None,
        # run_when_testing: bool = True,
        # prep_features: Optional[bool] = False,
    ):
        super().__init__()
        self.n_per_positive = n_per_positive
        self.item_id_col = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        self.schema = schema
        self.seed = seed

    def forward(self, inputs: TabularData, targets=None) -> TabularData:
        raise NotImplementedError

    #     batch_size = inputs[self.item_id_col].shape[0]
    #     sampled_num_negatives = self.n_per_positive * batch_size
