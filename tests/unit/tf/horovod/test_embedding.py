import importlib

import numpy as np
import pytest
import tensorflow as tf

from merlin.core.dispatch import HAS_GPU
from merlin.models.tf.distributed.embedding import SOKEmbedding
from merlin.schema import ColumnSchema, Tags


@pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
@pytest.mark.skipif(importlib.util.find_spec("sparse_operation_kit") is None, reason="needs sok")
class TestSOKEmbedding:
    sample_column_schema = ColumnSchema(
        "item_id",
        dtype=np.int32,
        properties={"domain": {"min": 0, "max": 10, "name": "item_id"}},
        tags=[Tags.CATEGORICAL],
    )

    def test_sok_embedding_basic(self):
        embedding = SOKEmbedding(16, self.sample_column_schema, vocab_sizes=[10])
        inputs = [tf.ragged.constant([[0, 1, 0], [1, 0]])]
        combiners = ["sum"]
        outputs = embedding(inputs, combiners)
        assert outputs[0].shape == (2, 16)

    def test_sok_embedding_pretrained(self):
        weights = {}
        indices = np.array([0, 1, 2])
        values = np.arange(3 * 16).reshape(3, 16)
        weights["indice"] = indices
        weights["values"] = values
        embedding = SOKEmbedding.from_pretrained(
            16, vocab_sizes=[10], data=[weights], name="item_id"
        )
        inputs = [tf.ragged.constant([[0, 1, 0], [1, 0]])]
        combiners = ["sum"]
        outputs = embedding(inputs, combiners)
        assert outputs[0].shape == (2, 16)

    def test_sok_embedding_config(self):
        embedding = SOKEmbedding(16, self.sample_column_schema, vocab_sizes=[10], name="item_id")
        config = embedding.get_config()
        _ = SOKEmbedding.from_config(config)
