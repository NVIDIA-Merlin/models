import torch

from merlin.models.torch.transforms.regularization import RMSNorm


class TestRMSNorm:
    def test_init(self):
        eps = 2e-5
        rms_norm = RMSNorm(8, eps=eps)
        assert isinstance(rms_norm.scale, torch.nn.Parameter)
        assert rms_norm.eps == eps

    def test_forward(self):
        rms_norm = RMSNorm(8)
        inputs = torch.randn(2, 4, 8)
        outputs = rms_norm(inputs)
        assert inputs.size() == outputs.size()
