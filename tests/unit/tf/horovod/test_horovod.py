import os
import importlib
import subprocess
import time
import random

import cupy
import numpy as np
import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.datasets.synthetic import generate_data
from merlin.datasets.advertising.criteo.dataset import default_criteo_transform
from merlin.models.utils.example_utils import workflow_fit_transform
from merlin.io.dataset import Dataset
from merlin.schema.tags import Tags
from merlin.models.tf.distributed.backend import hvd


def test_import():
    from merlin.models.tf.distributed.backend import hvd
    horovod_found = importlib.util.find_spec("horovod")

    assert (horovod_found and hvd is not None) or (not horovod_found and hvd is None)


@pytest.mark.timeout(180)
def test_horovod_multigpu(criteo_data, tmpdir, batch_size=11, learning_rate=0.03):
    """
    This test needs to be executed with `horovodrun`:
    $ horovodrun -np 2 hvd_wrapper.sh python -m pytest tests/unit/tf/horovod/test_horovod.py
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
    )

    target_column = train.schema.select_by_tag(Tags.TARGET).column_names[0]

    model = mm.DLRMModel(
        train.schema,
        embedding_dim=16,
        bottom_block=mm.MLPBlock([32, 16]),
        top_block=mm.MLPBlock([32, 16]),
        prediction_tasks=mm.BinaryClassificationTask(target_column),
    )

    opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    model.compile(optimizer=opt, run_eagerly=False, metrics=[tf.keras.metrics.AUC()])
    
    model.fit(
        train_loader,
        batch_size=batch_size,
    )
