import importlib
import os
import random

import cupy
import numpy as np
import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.datasets.advertising.criteo.dataset import default_criteo_transform
from merlin.io.dataset import Dataset
from merlin.models.tf.distributed.backend import hvd
from merlin.models.utils.example_utils import workflow_fit_transform
from merlin.schema.tags import Tags

# Seed with system randomness (or a static seed)
os.environ["TF_CUDNN_DETERMINISTIC"] = str(hvd.rank())
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
cupy.random.seed(None)


def seed_fn():
    """
    Generate consistent dataloader shuffle seeds across workers

    Reseeds each worker's dataloader each epoch to get fresh a shuffle
    that's consistent across workers.
    """
    min_int, max_int = tf.int32.limits
    max_rand = max_int // hvd.size()

    # Generate a seed fragment on each worker
    seed_fragment = cupy.random.randint(0, max_rand).get()

    # Aggregate seed fragments from all Horovod workers
    seed_tensor = tf.constant(seed_fragment)
    reduced_seed = hvd.allreduce(
        seed_tensor,
        name="shuffle_seed",
        op=hvd.mpi_ops.Sum,
    )

    return reduced_seed % max_rand


@pytest.mark.skipif(
    importlib.util.find_spec("horovod") is None, reason="This unit test requires horovod"
)
def test_import():
    from merlin.models.tf.distributed.backend import hvd

    assert hvd is not None


@pytest.mark.timeout(180)
def test_horovod_multigpu(criteo_data, tmpdir, batch_size):
    """
    This test needs to be executed with `horovodrun`:
    $ horovodrun -np 2 hvd_wrapper.sh python -m pytest tests/unit/tf/distributed/test_block.py
    """
    criteo_data.to_ddf().to_parquet(os.path.join(tmpdir, "train"))
    criteo_data.to_ddf().to_parquet(os.path.join(tmpdir, "valid"))
    workflow = default_criteo_transform()
    workflow_fit_transform(
        workflow,
        os.path.join(tmpdir, "train", "*.parquet"),
        os.path.join(tmpdir, "valid", "*.parquet"),
        os.path.join(tmpdir, "processed"),
    )

    train = Dataset(os.path.join(tmpdir, "processed", "train", "*.parquet"))
    ddf = train.to_ddf().repartition(npartitions=hvd.size())
    train = Dataset(ddf, schema=train.schema)

    train_loader = mm.Loader(
        train,
        schema=train.schema,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed_fn=seed_fn,
    )

    target_column = train.schema.select_by_tag(Tags.TARGET).column_names[0]

    model = mm.DLRMModel(
        train.schema,
        embedding_dim=16,
        bottom_block=mm.MLPBlock([32, 16]),
        top_block=mm.MLPBlock([32, 16]),
        prediction_tasks=mm.BinaryClassificationTask(target_column),
    )

    opt = tf.keras.optimizers.Adagrad(learning_rate=0.03)
    model.compile(optimizer=opt, run_eagerly=False, metrics=[tf.keras.metrics.AUC()])

    model.fit(
        train_loader,
        batch_size=batch_size,
    )
