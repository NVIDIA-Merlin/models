#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch

import merlin.models.torch as ml


def test_base_block(torch_tabular_features):
    block = torch_tabular_features >> ml.MLPBlock([64, 32])

    embedding_block = block.get_children_by_class_name(list(block), "EmbeddingFeatures")[0]

    assert isinstance(embedding_block, ml.EmbeddingFeatures)


def test_sequential_block(torch_tabular_features):
    block = ml.SequentialBlock(
        torch_tabular_features,
        ml.MLPBlock([64, 32]),
        ml.Block(torch.nn.Dropout(0.5), [None, 32]),
    )

    output_size = block.output_size()
    assert list(output_size) == [-1, 32]

    embedding_block = block.get_children_by_class_name(list(block), "EmbeddingFeatures")[0]
    assert isinstance(embedding_block, ml.EmbeddingFeatures)


def test_sequential_block_with_output_size(torch_tabular_features):
    block = ml.SequentialBlock(
        torch_tabular_features,
        ml.MLPBlock([64, 32]),
        torch.nn.Dropout(0.5),
        output_size=[None, 32],
    )

    output_size = block.output_size()
    assert list(output_size) == [None, 32]


def test_sequential(torch_tabular_features):
    inputs = torch_tabular_features
    block = torch.nn.Sequential(*ml.build_blocks(inputs, ml.MLPBlock([64, 32])))
    block2 = torch.nn.Sequential(inputs, ml.MLPBlock([64, 32]).to_module(inputs))

    assert isinstance(block, torch.nn.Sequential)
    assert isinstance(block2, torch.nn.Sequential)
