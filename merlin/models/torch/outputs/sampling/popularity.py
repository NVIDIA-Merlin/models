from merlin.models.torch.base import Block, registry


@registry.register("in-batch")
class InBatchNegativeSampler(Block):
    def forward(self, positive, positive_id=None):
        if positive_id is not None:
            self.register_buffer("positive_id", positive_id)
        self.register_buffer("positive", positive)

        return positive, positive_id
