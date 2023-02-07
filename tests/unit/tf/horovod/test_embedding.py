import numpy as np
import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.models.tf.utils import testing_utils
from merlin.schema import ColumnSchema, Schema, Tags

hvd = pytest.importorskip("horovod.tensorflow.keras")
dmp = pytest.importorskip("distributed_embeddings.python.layers.dist_model_parallel")


def generate_inputs(input_dims, global_batch_size):
    global_inputs = [
        tf.random.uniform(shape=[global_batch_size], minval=0, maxval=dim, dtype=tf.int64)
        for dim in input_dims
    ]
    for t in global_inputs:
        hvd.broadcast(t, root_rank=0)
    local_batch_size = global_batch_size // hvd.size()
    rank = hvd.rank()
    inputs = [t[rank * local_batch_size : (rank + 1) * local_batch_size] for t in global_inputs]
    return inputs


def test_distributed_embeddings_basic(embedding_dim=4, global_batch_size=8):
    column_schema_0 = ColumnSchema(
        "col0",
        dtype=np.int32,
        properties={"domain": {"min": 0, "max": 10, "name": "col0"}},
        tags=[Tags.CATEGORICAL],
    )
    column_schema_1 = ColumnSchema(
        "col1",
        dtype=np.int32,
        properties={"domain": {"min": 0, "max": 20, "name": "col1"}},
        tags=[Tags.CATEGORICAL],
    )
    schema = Schema([column_schema_0, column_schema_1])

    inputs = generate_inputs([10, 20], global_batch_size)
    table = mm.DistributedEmbeddings(schema, embedding_dim)
    outputs = table(inputs)

    assert len(outputs) == 2
    assert outputs[0].shape == (global_batch_size // hvd.size(), embedding_dim)
    assert outputs[1].shape == (global_batch_size // hvd.size(), embedding_dim)


def test_dlrm_model_with_embeddings(music_streaming_data, batch_size=8, embedding_dim=4):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_id", "user_age", "click"]
    )
    schema = music_streaming_data.schema

    ddf = music_streaming_data.to_ddf().repartition(npartitions=hvd.size())
    train = Dataset(ddf, schema=schema)

    train_loader = mm.Loader(
        train,
        schema=train.schema,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = mm.DLRMModel(
        schema,
        embeddings=mm.DistributedEmbeddings(
            schema.select_by_tag(Tags.CATEGORICAL), dim=embedding_dim
        ),
        bottom_block=mm.MLPBlock([embedding_dim]),
        prediction_tasks=mm.BinaryOutput("click"),
    )

    testing_utils.model_test(model, train_loader, run_eagerly=True)
