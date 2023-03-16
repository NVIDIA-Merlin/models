import torch
import torch.nn as nn

from merlin.models.torch.core.base import Block, TabularBlock


def test_block_no_pre_post():
    block = Block()
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    outputs = block(inputs)

    assert torch.equal(inputs, outputs)


def test_block_pre():
    pre = nn.Linear(2, 3)
    block = Block(pre=pre)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    outputs = block(inputs)
    expected_outputs = pre(inputs)

    assert torch.equal(outputs, expected_outputs)


def test_block_post():
    post = nn.Linear(2, 3)
    block = Block(post=post)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    outputs = block(inputs)
    expected_outputs = post(inputs)

    assert torch.equal(outputs, expected_outputs)


def test_block_pre_post():
    pre = nn.Linear(2, 3)
    post = nn.Linear(3, 4)
    block = Block(pre=pre, post=post)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    outputs = block(inputs)
    expected_outputs = pre(inputs)
    expected_outputs = post(expected_outputs)

    assert torch.equal(outputs, expected_outputs)


def test_tabular_block_no_pre_post_aggregation():
    block = TabularBlock()
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    outputs = block(inputs)

    assert torch.equal(inputs, outputs)


def test_tabular_block_aggregation():
    aggregation = nn.Linear(2, 3)
    block = TabularBlock(aggregation=aggregation)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    outputs = block(inputs)
    expected_outputs = aggregation(inputs)

    assert torch.equal(outputs, expected_outputs)
