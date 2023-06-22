from torch import nn

import merlin.models.torch as mm
from merlin.models.torch.utils.traversal_utils import find


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
