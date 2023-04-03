from merlin.models.torch.base import Block, registry


@registry.register("in-batch")
class InBatchNegativeSampler(Block):
    """PyTorch module that performs in-batch negative sampling."""

    def forward(self, positive, positive_id=None):
        """Doing in-batch negative-sampling.

        positive & positive_id are registered as non-persistent buffers

        Args:
            positive (torch.Tensor): Tensor containing positive samples.
            positive_id (torch.Tensor, optional): Tensor containing the IDs of
                positive samples. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the positive
                samples tensor and the positive samples IDs tensor.
        """
        if positive_id is not None:
            self.register_buffer("negative_id", positive_id, persistent=False)
        self.register_buffer("negative", positive, persistent=False)

        return positive, positive_id
