import pytest
import torch

from merlin.models.torch.models.base import Model
from merlin.models.torch.transforms.bias import SamplingProbabilityCorrection
from merlin.models.torch.transforms.features import Filter
from merlin.schema import Schema


class TestSamplingProbabilityCorrection:
    def test_instantiation(self):
        corr = SamplingProbabilityCorrection()
        assert corr.feature_name == "candidate_sampling_probability"

        corr = SamplingProbabilityCorrection("custom_feature_name")
        assert corr.feature_name == "custom_feature_name"

    def test_forward(self):
        corr = SamplingProbabilityCorrection()

        logits = torch.randn(5, 10)
        candidate_sampling_probability = torch.rand(5, 1)

        # Test with provided candidate_sampling_probability
        output = corr(logits, candidate_sampling_probability)
        expected_output = logits - torch.log(
            torch.clamp(candidate_sampling_probability, min=1e-6, max=1.0)
        )
        assert torch.allclose(output, expected_output)

        with pytest.raises(RuntimeError, match="No feature buffers found."):
            corr(logits)

    def test_forward_feature_propagated(self):
        sampling_feature_name = "candidate_sampling_probability"
        corr = SamplingProbabilityCorrection(sampling_feature_name)

        data = {sampling_feature_name: torch.rand(5, 1), "other_feature": torch.rand(5, 10)}

        input_module = Filter(Schema(list(data.keys())), aggregation="concat")
        model = Model(input_module, corr)

        out = model(data)
        assert torch.allclose(out, corr(input_module(data), data[sampling_feature_name]))
