
import os

MPI_SIZE = int(os.getenv("OMPI_COMM_WORLD_SIZE"))
MPI_RANK = int(os.getenv("OMPI_COMM_WORLD_RANK"))

os.environ["CUDA_VISIBLE_DEVICES"] = str(MPI_RANK)

import nvtabular as nvt
from nvtabular.ops import *

from merlin.models.utils.example_utils import workflow_fit_transform
from merlin.schema.tags import Tags

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser(
    description='Hyperparameters for model training'
)
parser.add_argument(
    '--batch-size', 
    type=str,
    help='Batch-Size per GPU worker'
)
parser.add_argument(
    '--path', 
    type=str,
    help='Directory with training and validation data'
)
args = parser.parse_args()

# define train and valid dataset objects
#train = Dataset(os.path.join(args.path, "train", "part_" + str(MPI_RANK) + ".parquet"))
#valid = Dataset(os.path.join(args.path, "valid", "part_" + str(MPI_RANK) + ".parquet"))

train = Dataset(os.path.join(args.path, "train", "*.parquet"))
valid = Dataset(os.path.join(args.path, "valid", "*.parquet"))

ddf = train.to_ddf().repartition(npartitions=2)
train = Dataset(ddf, schema=train.schema)
    
ddf = valid.to_ddf().repartition(npartitions=2)
valid = Dataset(ddf, schema=valid.schema)

# define schema object
target_column = train.schema.select_by_tag(Tags.TARGET).column_names[0]

train_loader = mm.Loader(
    train,
    schema=train.schema,
    batch_size=int(args.batch_size),
    shuffle=True,
    drop_last=True,
)

valid_loader = mm.Loader(
    valid,
    schema=valid.schema,
    batch_size=int(args.batch_size),
    shuffle=False,
    drop_last=True,
)

print("Number batches: " + str(len(train_loader)))

model = mm.DLRMModel(
    train.schema,
    embedding_dim=16,
    bottom_block=mm.MLPBlock([32, 16]),
    top_block=mm.MLPBlock([32, 16]),
    prediction_tasks=mm.BinaryOutput(target_column),
)

opt = tf.keras.optimizers.Adagrad(learning_rate=0.01)
model.compile(optimizer=opt, run_eagerly=False, metrics=[tf.keras.metrics.AUC()])
losses = model.fit(
    train_loader
)

print(model.evaluate(valid, batch_size=int(args.batch_size), return_dict=True))
