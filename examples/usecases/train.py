
import argparse
import os
import random
from pathlib import Path

# we can control how much memory to give tensorflow with this environment variable
# IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
# TF will have claimed all free GPU memory
os.environ["TF_MEMORY_ALLOCATION"] = "0.3"  # fraction of free memory

import cupy
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.schema.tags import Tags
#from merlin.models.tf.distributed.backend import hvd

hvd.init()

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


def train():
    base_dir = Path(args.base_dir)
    train = Dataset(base_dir / "train" / "*.parquet")
    ddf = train.to_ddf().repartition(npartitions=hvd.size())
    train = Dataset(ddf, schema=train.schema)

    train_loader = mm.Loader(
        train,
        schema=train.schema,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        seed_fn=seed_fn,
    )

    target_column = train.schema.select_by_tag(Tags.TARGET).column_names[0]

    model = mm.DLRMModel(
        train.schema,
        embedding_dim=args.embedding_dim,
        bottom_block=mm.MLPBlock([128, 64]),
        top_block=mm.MLPBlock([128, 64, 32]),
        prediction_tasks=mm.BinaryClassificationTask(target_column),
    )

    opt = tf.keras.optimizers.Adagrad(learning_rate=args.learning_rate)
    model.compile(optimizer=opt, run_eagerly=False, metrics=[tf.keras.metrics.AUC()])

    
    model.fit(
        train_loader,
        batch_size=args.batch_size,
    )

    if hvd.rank() == 0:
        model.save(base_dir)
        print(f"Training complete. Model saved to {base_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./data", help="Input directory")
    parser.add_argument("--batch_size", type=int, default=16 * 1024)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--embedding_dim", type=int, default=64)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train()
