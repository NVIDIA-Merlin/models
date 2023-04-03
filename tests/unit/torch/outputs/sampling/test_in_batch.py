import torch

from merlin.models.torch.outputs.sampling.in_batch import InBatchNegativeSampler


class TestInBatchNegativeSampler:
    def test_forward(self):
        # Create a sample input tensor
        positive = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        positive_id = torch.tensor([0, 1])

        # Instantiate the InBatchNegativeSampler
        sampler = InBatchNegativeSampler()

        # Call the forward method
        output_positive, output_positive_id = sampler.forward(positive, positive_id)

        # Check if the output_positive and output_positive_id tensors are correct
        assert torch.equal(output_positive, positive)
        assert torch.equal(output_positive_id, positive_id)

        # Call the forward method without positive_id
        output_positive, output_positive_id = sampler.forward(positive)

        # Check if the output_positive tensor is correct and output_positive_id is None
        assert torch.equal(output_positive, positive)
        assert output_positive_id is None
