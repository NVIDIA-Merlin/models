import torch

import merlin.models.torch as mm
from merlin.models.torch.blocks.llama import LlamaTransformer, LlamaAttentionHead
from merlin.models.torch.utils import module_utils


class TestLlamaBlock:
    def setup_method(self):
        self.llama_config = mm.LlamaConfig(
            max_seq_length=64,
            vocab_size=100,
            num_layers=1,
            num_heads=1,
            embedding_dim=16,
        )
        self.llama = mm.LlamaBlock(self.llama_config)
        self.input_dict = {
            "token": torch.tensor([[1, 3, 36, 2, 10]]),
            "position": torch.tensor([0, 1, 2, 3, 4]),
        }

    def test_forward(self):
        assert "position" in self.input_dict
        out = self.llama(self.input_dict)
        assert isinstance(out, torch.Tensor)
        assert out.shape[:-1] == self.input_dict["token"].shape
        assert out.shape[-1] == self.llama_config.padded_vocab_size

    def test_forward_without_position(self):
        self.input_dict.pop("position")
        assert "position" not in self.input_dict
        out = self.llama(self.input_dict)
        assert isinstance(out, torch.Tensor)
        assert out.shape[:-1] == self.input_dict["token"].shape
        assert out.shape[-1] == self.llama_config.padded_vocab_size

    def test_forward_tensor(self):
        assert "position" in self.input_dict
        out = self.llama(self.input_dict["token"])
        assert isinstance(out, torch.Tensor)
        assert out.shape[:-1] == self.input_dict["token"].shape
        assert out.shape[-1] == self.llama_config.padded_vocab_size

    def test_forward_torchscript(self):
        assert "position" in self.input_dict
        out = module_utils.module_test(self.llama, self.input_dict)
        assert isinstance(out, torch.Tensor)
        assert out.shape[:-1] == self.input_dict["token"].shape
        assert out.shape[-1] == self.llama_config.padded_vocab_size

    def test_reset_cache(self):
        _ = self.llama(self.input_dict)
        assert all(h.attention.kv_cache is not None for h in self.llama.transformer.heads)
        self.llama.reset_cache()
        assert all(h.attention.kv_cache is None for h in self.llama.transformer.heads)


class TestLlamaTransformer:
    def setup_method(self):
        self.llama_config = mm.LlamaConfig(
            max_seq_length=64,
            vocab_size=100,
            num_layers=1,
            num_heads=1,
            embedding_dim=16,
        )
        self.transformer = LlamaTransformer(self.llama_config)
        self.input_dict = {
            "token": torch.tensor([[1, 3, 36, 2, 10]]),
            "position": torch.tensor([0, 1, 2, 3, 4]),
        }

    def test_forward(self):
        assert "position" in self.input_dict
        out = self.transformer(self.input_dict)
        assert isinstance(out, torch.Tensor)
        assert out.shape[:-1] == self.input_dict["token"].shape
        assert out.shape[-1] == self.llama_config.embedding_dim

    def test_forward_without_position(self):
        self.input_dict.pop("position")
        assert "position" not in self.input_dict
        out = self.transformer(self.input_dict)
        assert isinstance(out, torch.Tensor)
        assert out.shape[:-1] == self.input_dict["token"].shape
        assert out.shape[-1] == self.llama_config.embedding_dim

    def test_forward_tensor(self):
        assert "position" in self.input_dict
        out = self.transformer(self.input_dict["token"])
        assert isinstance(out, torch.Tensor)
        assert out.shape[:-1] == self.input_dict["token"].shape
        assert out.shape[-1] == self.llama_config.embedding_dim

    def test_forward_torchscript(self):
        assert "position" in self.input_dict
        out = module_utils.module_test(self.transformer, self.input_dict)
        assert isinstance(out, torch.Tensor)
        assert out.shape[:-1] == self.input_dict["token"].shape
        assert out.shape[-1] == self.llama_config.embedding_dim

    def test_reset_cache(self):
        _ = self.transformer(self.input_dict)
        assert all(h.attention.kv_cache is not None for h in self.transformer.heads)
        self.transformer.reset_cache()
        assert all(h.attention.kv_cache is None for h in self.transformer.heads)


class TestLlamaAttentionHead:
    def test_forward(self):
        batch_size = 2
        embedding_dim = 16
        max_seq_length = 64
        attn_head = LlamaAttentionHead(
            num_heads=1,
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
        )
        inputs = torch.randn(batch_size, max_seq_length, embedding_dim)
        outputs = attn_head(inputs)
        assert outputs.size() == inputs.size()
