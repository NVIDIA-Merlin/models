import pytest
import torch

from merlin.models.torch.outputs.sampling.popularity import (
    LogUniformSampler,
    PopularityBasedSampler,
)


class TestLogUniformSampler:
    def test_init(self):
        range_max = 1000
        n_sample = 100
        sampler = LogUniformSampler(n_sample, range_max)

        assert sampler.max_id == range_max
        assert sampler.max_n_samples == n_sample
        assert sampler.dist.size(0) == range_max
        assert sampler.unique_sampling_dist.size(0) == range_max

    def test_sample(self):
        range_max = 1000
        n_sample = 100
        sampler = LogUniformSampler(n_sample, range_max)

        labels = torch.tensor([10, 50, 150])
        neg_samples, true_log_probs, samp_log_probs = sampler.sample(labels)

        assert true_log_probs.size() == labels.size()
        assert samp_log_probs.size()[0] <= 2 * n_sample
        assert neg_samples.size()[0] <= 2 * n_sample

    @pytest.mark.parametrize("range_max, n_sample", [(1000, 100), (5000, 250), (10000, 500)])
    def test_dist_sum(self, range_max, n_sample):
        sampler = LogUniformSampler(n_sample, range_max)

        assert torch.isclose(sampler.dist.sum(), torch.tensor(1.0), atol=1e-6)

    def test_init_exceptions(self):
        with pytest.raises(ValueError, match="n_sample must be a positive integer."):
            LogUniformSampler(-100, 1000)

        with pytest.raises(ValueError, match="max_id must be a positive integer"):
            LogUniformSampler(100, -1000)

    def test_sample_exceptions(self):
        range_max = 1000
        n_sample = 100
        sampler = LogUniformSampler(n_sample, range_max)

        with pytest.raises(TypeError, match="Labels must be a torch.Tensor."):
            sampler.sample([10, 50, 150])

        with pytest.raises(ValueError, match="Labels must be a tensor of dtype long."):
            sampler.sample(torch.tensor([10, 50, 150], dtype=torch.float32))

        with pytest.raises(ValueError, match="Labels must be a tensor of dtype long."):
            sampler.sample(torch.tensor([]))

        with pytest.raises(ValueError, match="All label values must be within the range"):
            sampler.sample(torch.tensor([-1, 50, 150]))

        with pytest.raises(ValueError, match="All label values must be within the range"):
            sampler.sample(torch.tensor([10, 50, 150, 2000]))


class TestPopularityBasedSampler:
    def test_init_defaults(self):
        sampler = PopularityBasedSampler()
        assert sampler.labels.dtype == torch.int64
        assert sampler.labels.shape == torch.Size([1, 1])
        assert sampler.max_num_samples == 10

    def test_init_custom_values(self):
        sampler = PopularityBasedSampler(max_num_samples=20)
        assert sampler.max_num_samples == 20

    def test_forward_raises_runtime_error(self):
        sampler = PopularityBasedSampler()
        with pytest.raises(RuntimeError):
            sampler(torch.tensor([1.0]), torch.tensor([1]))

    def test_forward(self):
        class MockToCall:
            def embedding_lookup(self, ids):
                return torch.tensor([42.0])

            num_classes = 1000

        sampler = PopularityBasedSampler()
        sampler.set_to_call(MockToCall())

        negative, negative_id = sampler(torch.tensor([1.0]), torch.tensor([1]))

        assert isinstance(negative, torch.Tensor)
        assert isinstance(negative_id, torch.Tensor)

    def test_sampler_property(self):
        class MockToCall:
            num_classes = 1000

        sampler = PopularityBasedSampler()
        sampler.set_to_call(MockToCall())

        log_uniform_sampler = sampler.sampler

        assert isinstance(log_uniform_sampler, LogUniformSampler)
        assert log_uniform_sampler.max_id == MockToCall.num_classes
        assert log_uniform_sampler.max_n_samples == sampler.max_num_samples
