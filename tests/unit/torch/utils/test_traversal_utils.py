from torch import nn

import merlin.models.torch as mm
from merlin.models.torch.utils.traversal_utils import find, leaf
from merlin.schema import Schema, Tags


class TestFind:
    def test_find_base(self):
        class SomeBlock(nn.Module):
            ...

        class OtherBlock(nn.Module):
            ...

        some_block = SomeBlock()
        other_block = OtherBlock()

        assert len(find(some_block, SomeBlock)) == 1
        assert len(find(some_block, lambda x: x == some_block)) == 1
        assert len(find(some_block, OtherBlock)) == 0
        assert len(find(some_block, lambda x: x == other_block)) == 0

    def test_module_list(self):
        class ToFind(nn.Module):
            ...

        class NotToFind(nn.Module):
            ...

        block = nn.ModuleList(
            [
                ToFind(),
                NotToFind(),
                ToFind(),
                NotToFind(),
                ToFind(),
            ]
        )
        assert len(find(block, ToFind)) == 3

    def test_nested(self):
        class ToFind(nn.Module):
            ...

        class NotToFind(nn.Module):
            ...

        block = mm.ParallelBlock(
            {
                "a": NotToFind(),
                "b": mm.ParallelBlock(
                    {
                        "c": NotToFind(),
                        "d": ToFind(),
                        "e": mm.ParallelBlock(
                            {
                                "f": ToFind(),
                            }
                        ),
                    }
                ),
                "g": ToFind(),
            }
        )
        assert len(find(block, ToFind)) == 3


class TestLeaf:
    def test_simple(self):
        lin = nn.Linear(10, 20)
        assert leaf(lin) == lin

    def test_nested(self):
        model = nn.Sequential(nn.Linear(10, 20))
        assert isinstance(leaf(model), nn.Linear)
        assert isinstance(leaf(nn.Sequential(model)), nn.Linear)

    def test_custom_module(self):
        class CustomModule(nn.Module):
            def __init__(self):
                super(CustomModule, self).__init__()
                self.layer = nn.Linear(10, 20)

        model = CustomModule()
        assert isinstance(leaf(model), CustomModule)

    def test_sequential(self):
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(10, 30))
        assert leaf(model).out_features == 30

    def test_embedding(self, user_id_col_schema):
        input_block = mm.TabularInputBlock(Schema([user_id_col_schema]), init="defaults")
        user_emb = input_block.select(Tags.USER_ID).leaf()

        assert isinstance(user_emb, mm.EmbeddingTable)
