import numpy as np
import pytest

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.schema import Tags


@pytest.mark.parametrize("name_branches", [True, False])
def test_parallel_block_pruning(music_streaming_data: Dataset, name_branches: bool):
    music_streaming_data.schema = music_streaming_data.schema.remove_by_tag(Tags.CONTINUOUS)

    continuous_block = mm.Filter(music_streaming_data.schema.select_by_tag(Tags.CONTINUOUS))
    embedding_block = mm.EmbeddingFeatures.from_schema(
        music_streaming_data.schema.select_by_tag(Tags.CATEGORICAL)
    )

    if name_branches:
        branches = {"continuous": continuous_block, "embedding": embedding_block}
    else:
        branches = [continuous_block, embedding_block]

    input_block = mm.ParallelBlock(branches, schema=music_streaming_data.schema)

    features = mm.sample_batch(music_streaming_data, batch_size=10, include_targets=False)

    outputs = input_block(features)

    assert len(outputs) == 7  # There are 7 categorical features
    assert continuous_block not in input_block.parallel_values


def test_parallel_block_serialization(music_streaming_data: Dataset):
    unknown_filter = mm.Filter(["none"])
    block = mm.ParallelBlock(mm.Filter(["position"]), unknown_filter, automatic_pruning=False)
    block_copy = block.from_config(block.get_config())

    assert not block_copy.automatic_pruning
    assert unknown_filter not in block_copy.parallel_values

    features = mm.sample_batch(music_streaming_data, batch_size=10, include_targets=False)

    outputs_1 = block(features)
    outputs_2 = block_copy(features)

    for key in outputs_1:
        np.testing.assert_array_equal(outputs_2[key].numpy(), outputs_1[key].numpy())
    assert len(outputs_1) == len(outputs_2) == 1
