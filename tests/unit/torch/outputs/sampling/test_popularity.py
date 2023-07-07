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
        sampler = LogUniformSampler(range_max, n_sample)

        assert sampler.range_max == range_max
        assert sampler.n_sample == n_sample
        assert sampler.dist.size(0) == range_max
        assert sampler.log_q.size(0) == range_max

    def test_sample(self):
        range_max = 1000
        n_sample = 100
        sampler = LogUniformSampler(range_max, n_sample)

        labels = torch.tensor([10, 50, 150])
        true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)

        assert true_log_probs.size() == labels.size()
        assert samp_log_probs.size()[0] <= 2 * n_sample
        assert neg_samples.size()[0] <= 2 * n_sample

    @pytest.mark.parametrize("range_max, n_sample", [(1000, 100), (5000, 250), (10000, 500)])
    def test_dist_sum(self, range_max, n_sample):
        sampler = LogUniformSampler(range_max, n_sample)

        assert torch.isclose(sampler.dist.sum(), torch.tensor(1.0), atol=1e-6)

    def test_init_exceptions(self):
        with pytest.raises(ValueError):
            LogUniformSampler(-1000, 100)

        with pytest.raises(ValueError):
            LogUniformSampler(1000, -100)

    def test_sample_exceptions(self):
        range_max = 1000
        n_sample = 100
        sampler = LogUniformSampler(range_max, n_sample)

        with pytest.raises(TypeError):
            sampler.sample([10, 50, 150])

        with pytest.raises(ValueError):
            sampler.sample(torch.tensor([10, 50, 150], dtype=torch.float32))

        with pytest.raises(ValueError):
            sampler.sample(torch.tensor([]))

        with pytest.raises(ValueError):
            sampler.sample(torch.tensor([-1, 50, 150]))

        with pytest.raises(ValueError):
            sampler.sample(torch.tensor([10, 50, 150, 2000]))


class TestPopularityBasedSampler:
    def test_init_defaults(self):
        sampler = PopularityBasedSampler()
        assert sampler.labels.dtype == torch.int64
        assert sampler.labels.shape == torch.Size([1, 1])
        assert sampler.max_num_samples == 10
        assert sampler.seed is None

    def test_init_custom_values(self):
        sampler = PopularityBasedSampler(max_num_samples=20, seed=42)
        assert sampler.max_num_samples == 20
        assert sampler.seed == 42

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
        sampler.to_call = MockToCall()

        negative, negative_id = sampler(torch.tensor([1.0]), torch.tensor([1]))

        assert isinstance(negative, torch.Tensor)
        assert isinstance(negative_id, torch.Tensor)

    def test_sampler_property(self):
        class MockToCall:
            num_classes = 1000

        sampler = PopularityBasedSampler()
        sampler.to_call = MockToCall()

        log_uniform_sampler = sampler.sampler

        assert isinstance(log_uniform_sampler, LogUniformSampler)
        assert log_uniform_sampler.range_max == MockToCall.num_classes
        assert log_uniform_sampler.n_sample == sampler.max_num_samples
