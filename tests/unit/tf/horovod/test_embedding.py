import numpy as np
import pytest
import tensorflow as tf

from merlin.core.dispatch import HAS_GPU
from merlin.models.tf.distributed.embedding import SOKEmbedding
from merlin.schema import ColumnSchema, Tags


@pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
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
        weights = np.random.rand(10, 16)
        embedding = SOKEmbedding.from_pretrained(16, [weights])
        inputs = [tf.ragged.constant([[0, 1, 0], [1, 0]])]
        combiners = ["sum"]
        outputs = embedding(inputs, combiners)
        assert outputs[0].shape == (2, 16)

    def test_sok_embedding_config(self):
        embedding = SOKEmbedding(16, self.sample_column_schema, vocab_sizes=[10])
        config = embedding.get_config()
        _ = SOKEmbedding.from_config(config)