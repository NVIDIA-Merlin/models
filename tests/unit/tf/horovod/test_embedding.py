import numpy as np
import tensorflow as tf

import merlin.models.tf as mm
from merlin.models.tf.distributed.backend import hvd, hvd_installed, dmp_installed
from merlin.schema import ColumnSchema, Tags


def generate_inputs(table_sizes, domain_max, global_batch_size):
    global_inputs = [
        tf.random.uniform(shape=[global_batch_size], minval=0, maxval=domain_max, dtype=tf.int64)
        for size in table_sizes
    ]
    for t in global_inputs:
        hvd.broadcast(t, root_rank=0)
    local_batch_size = global_batch_size // hvd.size()
    size = hvd.size()
    rank = hvd.rank()
    inputs = [
        t[rank * local_batch_size : (rank + 1) * local_batch_size] for t in global_inputs
    ]

    return inputs


def test_distributed_embedding_basic():
    assert hvd_installed is True
    assert dmp_installed is True

    dim = 2
    domain_max = 10
    column_schema = ColumnSchema(
        "item_id",
        dtype=np.int32,
        properties={"domain": {"min": 0, "max": domain_max, "name": "item_id"}},
        tags=[Tags.CATEGORICAL],
    )
    table_sizes = [3, 4]
    global_batch_size = 8

    inputs = generate_inputs(table_sizes, domain_max, global_batch_size)
    table = mm.DistributedEmbeddingTable(dim, table_sizes, column_schema)
    outputs = table(inputs)

    assert outputs[0].shape == (4, 2)
    assert outputs[1].shape == (4, 2)
