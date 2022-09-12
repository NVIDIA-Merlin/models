import numpy as np

from merlin.dag import BaseOperator, Graph
from merlin.dag.executors import DataFrameLike, LocalExecutor
from merlin.datasets.entertainment import get_movielens

# merlin-core
from merlin.io import Dataset
from merlin.models.tf.data_augmentation.negative_sampling import UniformNegativeSampling

# merlin-models
from merlin.models.tf.dataset import BatchedDataset


def test_layer_transform():

    # load a dataset
    train, valid = get_movielens(variant="ml-100k")
    input_schema = train.schema
    input_schema = input_schema.without(["rating", "title"])

    # TODO: Figure out why the original data and schema dtypes don't match
    # correct userId_count dtype
    input_schema["userId_count"] = input_schema["userId_count"].with_dtype(np.float32)

    # keep only positives
    train_df = train.to_ddf().compute()
    train = Dataset(train_df[train_df["rating_binary"] == 1])
    train.schema = input_schema

    # sampling transform to create negatives from a batch with only positives
    # for the purposes of this example it could be any function that accepts two arguments (x, y)
    # and returns a tuple (x, y)
    sampling_layer = UniformNegativeSampling(input_schema, 5, seed=42, return_tuple=True)

    batched_dataset = BatchedDataset(train, batch_size=10)
    example_batch = next(batched_dataset)
    # example_batch is a 2-tuple containing x (input features) and y (targets)
    # where x and y are Dict[str, TensorLike]

    x_dtypes = {col: value.numpy().dtype for col, value in example_batch[0].items()}
    x = DataFrameLike(
        example_batch[0],
        dtypes=x_dtypes,
    )

    # TODO: Figure out why the original data and schema dtypes don't match
    # correct userId_count dtype
    input_schema["userId_count"] = input_schema["userId_count"].with_dtype(np.float64)
    x_schema = input_schema.without(["rating_binary", "title"])
    y_schema = input_schema.select_by_name("rating_binary")

    y = {"y": example_batch[1]}
    y = DataFrameLike(
        y,
        dtypes={col: value.numpy().dtype for col, value in y.items()},
    )

    class LayerTransform(BaseOperator):
        def __init__(self, layer):
            self.layer = layer

        def transform(self, column_selector, data):
            transformed = self.layer(data)
            return transformed.outputs

    node = [] >> LayerTransform(sampling_layer)
    graph = Graph(node)

    executor = LocalExecutor()
    executor.transform_multi(
        (x, y),
        (x_schema, y_schema),
        [graph.output_node],
    )
