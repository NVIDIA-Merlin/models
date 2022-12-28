import numpy as np
import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.schema import ColumnSchema, Tags

hvd = pytest.importorskip("horovod.tensorflow")
sok = pytest.importorskip("sparse_operation_kit")


class TestSOKEmbedding:
    sample_column_schema = ColumnSchema(
        "item_id",
        dtype=np.int32,
        properties={"domain": {"min": 0, "max": 10, "name": "item_id"}},
        tags=[Tags.CATEGORICAL],
    )

    def test_raises_with_invalid_schema(self):
        column_schema = ColumnSchema("item_id")
        with pytest.raises(ValueError) as exc_info:
            mm.EmbeddingTable(16, column_schema)
        assert "needs to have an int-domain" in str(exc_info.value)

    @pytest.mark.parametrize("dim", [16, 32])
    def test_sok_dynamic_variables(self, dim):
        hvd.init()
        sok.init()

        rows = [65536 * 10, 65536]
        cols = [128, 4]
        initial_vals = [13, 17]

        # sok variables
        sok_vars = [
            sok.DynamicVariable(dimension=cols[i], initializer=str(initial_vals[i]))
            for i in range(len(cols))
        ]
        local_indices = []
        for row in rows:
            local_size = row // hvd.size()
            if hvd.rank() < row % hvd.size():
                local_size += 1
            indices = np.arange(local_size) * hvd.size() + hvd.rank()
            indices = tf.convert_to_tensor(indices, dtype=tf.int64)
            local_indices.append(indices)
        out1 = []
        for i in range(len(sok_vars)):
            out1.append(tf.nn.embedding_lookup(sok_vars[i], local_indices[i]))

        tf_vars = [
            tf.Variable(tf.constant(initial_vals[i], shape=[rows[i], cols[i]], dtype=tf.float32))
            for i in range(len(rows))
        ]
        out2 = []
        for i, v in enumerate(tf_vars):
            out2.append(tf.nn.embedding_lookup(v, local_indices[i]))

        # Check results
        diff = 0
        for i in range(len(out1)):
            length = out1[i] ** 2 + out2[i] ** 2 + 1e-8
            diff = diff + tf.reduce_sum((out1[i] - out2[i]) ** 2 / length)
        print("[SOK INFO] diff:", diff)
        assert diff < 1e-6

    @pytest.mark.parametrize("dim", [16, 32])
    def test_distributed_variables(self, dim):
        hvd.init()
        sok.init()

        rows = [65536 * 10, 65536]
        cols = [128, 4]

        # initial value of embedding table
        weights = []
        for i in range(len(rows)):
            weight = np.random.rand(rows[i], cols[i]).astype(np.float32)
            weight = tf.convert_to_tensor(weight, dtype=tf.float32)
            # make sure the weight is same on each rank
            weight = hvd.allreduce(weight)
            weights.append(weight)

        # sok variables
        sok_vars = [sok.Variable(w) for w in weights]
        local_indices = []
        for row in rows:
            local_size = row // hvd.size()
            if hvd.rank() < row % hvd.size():
                local_size += 1
            indices = np.arange(local_size) * hvd.size() + hvd.rank()
            indices = tf.convert_to_tensor(indices, dtype=tf.int64)
            local_indices.append(indices)

        out1 = sok_vars
        tf_vars = [tf.Variable(w) for w in weights]
        out2 = []
        for i, v in enumerate(tf_vars):
            out2.append(tf.nn.embedding_lookup(v, local_indices[i]))

        # Check results
        diff = 0
        for i in range(len(out1)):
            length = out1[i] ** 2 + out2[i] ** 2 + 1e-8
            diff = diff + tf.reduce_sum((out1[i] - out2[i]) ** 2 / length)
        print("[SOK INFO] diff:", diff)
        assert diff < 1e-6
