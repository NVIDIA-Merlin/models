import torch
from torch import nn

from merlin.models.torch.link import Link, Residual, Shortcut, ShortcutConcat
from merlin.models.torch.utils import module_utils


class TestResidual:
    def test_forward(self):
        input_tensor = torch.randn(1, 3, 64, 64)
        conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        residual = Residual(conv)

        output_tensor = module_utils.module_test(residual, input_tensor)
        expected_tensor = input_tensor + conv(input_tensor)

        assert torch.allclose(output_tensor, expected_tensor)

    def test_from_registry(self):
        residual = Link.parse("residual")

        assert isinstance(residual, Residual)


class TestShortcut:
    def test_forward(self):
        input_tensor = torch.randn(1, 3, 64, 64)
        conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        shortcut = Shortcut(conv)

        output_dict = module_utils.module_test(shortcut, input_tensor)

        assert "output" in output_dict
        assert "shortcut" in output_dict
        assert torch.allclose(output_dict["output"], conv(input_tensor))
        assert torch.allclose(output_dict["shortcut"], input_tensor)

    def test_from_registry(self):
        shortcut = Link.parse("shortcut")

        assert isinstance(shortcut, Shortcut)


class TestShortcutConcat:
    def test_forward(self):
        input_tensor = torch.randn(1, 3, 64, 64)
        conv = nn.Conv2d(
            3, 10, kernel_size=3, padding=1
        )  # Output channels are different for concatenation
        shortcut_concat = ShortcutConcat(conv)

        output_tensor = module_utils.module_test(shortcut_concat, input_tensor)
        expected_tensor = torch.cat((input_tensor, conv(input_tensor)), dim=1)

        assert torch.allclose(output_tensor, expected_tensor)

    def test_from_registry(self):
        shortcut_concat = Link.parse("shortcut-concat")

        assert isinstance(shortcut_concat, ShortcutConcat)
