import torch
from torch import nn

import merlin.models.torch as mm
from merlin.models.torch.blocks.peft import LoRA
from merlin.models.torch.utils import module_utils


class TestLoRA:
    def setup_method(self):
        # create a simple linear layer to use in tests
        self.linear_module = nn.Linear(10, 5)
        self.lora_linear = LoRA(
            self.linear_module, r=2, lora_alpha=1.0, lora_dropout=0.1, merge_weights=True
        )
        self.embedding_module = nn.Embedding(10, 5)
        self.lora_embedding = LoRA(
            self.embedding_module, r=2, lora_alpha=1.0, lora_dropout=0.1, merge_weights=True
        )

    def test_init_linear(self):
        # test whether lora module was initialized correctly
        assert self.lora_linear.module == self.linear_module
        assert self.lora_linear.r == 2
        assert self.lora_linear.lora_alpha == 1.0
        assert isinstance(self.lora_linear.lora_dropout, nn.Dropout)
        assert self.lora_linear.merge_weights
        assert not self.lora_linear.merged
        assert self.lora_linear.in_features == self.linear_module.in_features
        assert self.lora_linear.out_features == self.linear_module.out_features
        assert not self.lora_linear.module.weight.requires_grad

    def test_init_embedding(self):
        # test whether lora module was initialized correctly
        assert self.lora_embedding.module == self.embedding_module
        assert self.lora_embedding.r == 2
        assert self.lora_embedding.lora_alpha == 1.0
        assert isinstance(self.lora_embedding.lora_dropout, nn.Dropout)
        assert self.lora_embedding.merge_weights
        assert not self.lora_embedding.merged
        assert self.lora_embedding.in_features == self.embedding_module.num_embeddings
        assert self.lora_embedding.out_features == self.embedding_module.embedding_dim
        assert not self.lora_embedding.module.weight.requires_grad

    def test_walk(self):
        block = mm.Block(nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 2))
        lora_block = LoRA.walk(block, r=2, lora_alpha=1.0, lora_dropout=0.0, merge_weights=True)

        # test whether the apply function correctly replaced Linear layers with LoRA modules
        assert isinstance(lora_block[0], LoRA)
        assert isinstance(lora_block[2], LoRA)
        assert isinstance(lora_block[1], nn.ReLU)

    def test_reset_parameters(self):
        self.lora_linear.reset_parameters()

        # check if reset correctly
        assert torch.sum(self.lora_linear.lora_B).item() == 0
        assert self.lora_linear.lora_A.data is not None  # filled with some initialization

    def test_linear_train_and_eval_mode(self):
        self.lora_linear.train()

        # test if module is in training mode
        assert self.lora_linear.training

        self.lora_linear.eval()

        # test if module is in eval mode
        assert not self.lora_linear.training

    def test_linear_forward(self):
        x = torch.randn(1, 10)
        out = module_utils.module_test(self.lora_linear, x)

        # test output shape
        assert out.shape == (1, 5)

    def test_embedding_train_and_eval_mode(self):
        self.lora_embedding.train()

        # test if module is in training mode
        assert self.lora_embedding.training

        self.lora_embedding.eval()

        # test if module is in eval mode
        assert not self.lora_embedding.training

    def test_embedding_forward(self):
        x = torch.tensor([1, 2, 3])
        out = module_utils.module_test(self.lora_embedding, x)

        # test output shape
        assert out.shape == (3, 5)
