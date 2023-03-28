import os

import pytest
import tensorflow as tf
from packaging import version
from tensorflow.keras.utils import set_random_seed

import merlin.models.tf as mm
from merlin.datasets.advertising.criteo.dataset import default_criteo_transform
from merlin.io.dataset import Dataset
from merlin.models.tf.distributed.backend import hvd, hvd_installed
from merlin.models.utils.example_utils import workflow_fit_transform
from merlin.schema.tags import Tags

# Set seed to make tests deterministic. In real training, it is not
# necessary to set the random seed.
set_random_seed(42)


@pytest.mark.parametrize("custom_distributed_optimizer", [True, False])
def test_horovod_multigpu_dlrm(
    criteo_data,
    tmpdir,
    custom_distributed_optimizer,
    batch_size=11,
    learning_rate=0.03,
):
    """
    This test should work on CPU, single GPU, and multiple GPUs.
    However, for distributed training on multiple GPUs, it needs to be
    executed with `horovodrun`:

    $ horovodrun -np 2 hvd_wrapper.sh python -m pytest tests/unit/tf/horovod/test_horovod.py

    Even with multiple GPUs, if you simply run pytest without horovodrun, it
    will use only one GPU.
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

    # As a workaround for nvtabular producing different numbers of batches in
    # workers, we repartition the dataset in multiples of hvd.size().
    # https://github.com/NVIDIA-Merlin/models/issues/765
    train = Dataset(os.path.join(tmpdir, "processed", "train", "*.parquet"))
    ddf = train.to_ddf().repartition(npartitions=2)
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

    if version.parse(tf.__version__) < version.parse("2.11.0"):
        keras_optimizers = tf.keras.optimizers
    else:
        keras_optimizers = tf.keras.optimizers.legacy

    if custom_distributed_optimizer:
        # Test for a case when the user uses a custom DistributedOptimizer.
        # With a custom hvd.DistributedOptimzer, users have to adjust the learning rate.
        opt = keras_optimizers.Adagrad(learning_rate=learning_rate * hvd.size())
        opt = hvd.DistributedOptimizer(opt, compression=hvd.Compression.fp16)
    else:
        opt = keras_optimizers.Adagrad(learning_rate=learning_rate)

    model.compile(optimizer=opt, run_eagerly=False, metrics=[tf.keras.metrics.AUC()])

    # model.fit() will hang or terminate with error if all workers don't have
    # the same number of batches.
    losses = model.fit(
        train_loader,
        batch_size=batch_size,
    )

    model.save(tmpdir)

    if hvd_installed:
        assert model.optimizer.learning_rate == learning_rate * hvd.size()
    else:
        assert model.optimizer.learning_rate == learning_rate

    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])

    # Check the steps in each worker to check that the dataset is distributed
    # across workers. If this works correctly in a multi-gpu setting, the steps
    # should decrease with more workers, e.g., steps = 9 in each worker with
    # 1 GPU, steps = 4 in each worker with 2 GPUS, steps = 3 in each worker
    # with 3 GPUS, and so on.
    if hvd_installed:
        assert losses.params["steps"] == 9 // hvd.size()
    else:
        assert losses.params["steps"] == 9

    saved_model = "saved_model.pb"
    if hvd_installed and hvd.rank() == 0:
        assert saved_model in os.listdir(tmpdir)
    if hvd_installed and hvd.rank() != 0:
        assert saved_model not in os.listdir(tmpdir)


def test_horovod_multigpu_two_tower(
    music_streaming_data, tmpdir, batch_size=11, learning_rate=0.03
):
    """
    This test should work on CPU, single GPU, and multiple GPUs.
    However, for distributed training on multiple GPUs, it needs to be
    executed with `horovodrun`:

    $ horovodrun -np 2 hvd_wrapper.sh python -m pytest tests/unit/tf/horovod/test_horovod.py

    Even with multiple GPUs, if you simply run pytest without horovodrun, it
    will use only one GPU.
    """
    # As a workaround for nvtabular producing different numbers of batches in
    # workers, we repartition the dataset in multiples of hvd.size().
    # https://github.com/NVIDIA-Merlin/models/issues/765
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_genres"]
    )
    ddf = music_streaming_data.to_ddf().repartition(npartitions=2)
    train = Dataset(ddf, schema=music_streaming_data.schema)

    train_loader = mm.Loader(
        train,
        schema=train.schema,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = mm.TwoTowerModel(music_streaming_data.schema, query_tower=mm.MLPBlock([2]))
    model.compile(optimizer="adam", run_eagerly=False)

    # model.fit() will hang or terminate with error if all workers don't have
    # the same number of batches.
    losses = model.fit(train_loader, batch_size=batch_size, epochs=2)

    model.save(tmpdir)

    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])

    if hvd_installed:
        assert losses.params["steps"] == 9 // hvd.size()  # 9 steps per epoch; 2 epochs
    else:
        assert losses.params["steps"] == 9

    saved_model = "saved_model.pb"
    if hvd_installed and hvd.rank() != 0:
        assert saved_model not in os.listdir(tmpdir)
    else:
        assert saved_model in os.listdir(tmpdir)
