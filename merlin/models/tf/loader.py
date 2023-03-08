#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging
import os
from typing import Optional, Protocol, Union

import dask.dataframe as dd
import numpy as np
import tensorflow as tf

import merlin.dataloader.tensorflow
from merlin.core.dispatch import HAS_GPU
from merlin.dataloader.tf_utils import get_dataset_schema_from_feature_columns
from merlin.io import Dataset
from merlin.models.loader.backend import _augment_schema
from merlin.models.tf.distributed.backend import hvd, hvd_installed
from merlin.models.utils.schema_utils import select_targets
from merlin.schema import Schema, Tags

LOG = logging.getLogger("merlin.models")

# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter,unexpected-keyword-arg,redundant-keyword-arg


dd_engine = {
    "parquet": dd.read_parquet,
    "csv": dd.read_csv,
    "df": dd.DataFrame,
}


def _validate_dataset(paths_or_dataset, batch_size, buffer_size, engine, device, reader_kwargs):
    # TODO: put this in parent class and allow
    # torch dataset to leverage as well?

    # if a dataset was passed, just return it
    if hasattr(paths_or_dataset, "schema"):
        return paths_or_dataset

    # otherwise initialize a dataset
    # from paths or glob pattern
    if isinstance(paths_or_dataset, str):
        files = tf.io.gfile.glob(paths_or_dataset)
        parent, file = os.path.split(paths_or_dataset)
        _is_empty_msg = f"Couldn't find file pattern {file} in directory {parent}"
    else:
        # TODO: some checking around attribute
        # error here?
        files = list(paths_or_dataset)
        _is_empty_msg = "paths_or_dataset list must contain at least one filename"

    assert isinstance(files, list)
    if len(files) == 0:
        raise ValueError(_is_empty_msg)

    if not engine:
        # default engine is parquet
        engine = "parquet"

    cpu = device and "cpu" in device

    return Dataset(files, engine=engine, cpu=cpu)


def _validate_schema(feature_columns, cat_names, cont_names, label_names, schema=None):
    _uses_feature_columns = feature_columns is not None
    _uses_explicit_schema = (cat_names is not None) or (cont_names is not None)

    cat_tag_names = (
        schema.select_by_tag([Tags.CATEGORICAL]).excluding_by_tag(Tags.TARGET).column_names
        if schema
        else []
    )
    cont_tag_names = (
        schema.select_by_tag([Tags.CONTINUOUS]).excluding_by_tag(Tags.TARGET).column_names
        if schema
        else []
    )
    label_names = label_names or []
    _uses_dataset_schema = cat_tag_names or cont_tag_names

    if _uses_feature_columns and _uses_explicit_schema:
        raise ValueError(
            "Passed `feature_column`s and explicit column names, must be one or the other"
        )
    elif _uses_feature_columns:
        cat_names, cont_names = get_dataset_schema_from_feature_columns(feature_columns)

        return cat_names, cont_names, label_names
    elif _uses_explicit_schema:
        cat_names = cat_names or []
        cont_names = cont_names or []

        return cat_names, cont_names, label_names
    elif _uses_dataset_schema:
        label_names = label_names or select_targets(schema).column_names

        return cat_tag_names, cont_tag_names, label_names
    else:
        raise ValueError(
            "Must either pass a list of TensorFlow `feature_column`s "
            "or explicit `cat_name` and `cont_name` column name lists."
        )


def _get_schema(dataset):
    if hasattr(dataset, "schema"):
        return dataset.schema
    return None


class SchemaAwareTransform(Protocol):
    def __call__(self, *args, **kwargs):
        ...

    def compute_output_schema(self, schema: Schema):
        ...


class Loader(merlin.dataloader.tensorflow.Loader):
    """
    Override class to customize data loading for backward compatibility with
    older NVTabular releases.

    In most cases, when you use the `merlin.io.Dataset` class, data loading
    for model training and evaluation is performed automatically by Merlin Models.

    Infinite generator used to asynchronously iterate through CSV or Parquet
    dataframes on GPU by leveraging an NVTabular `Dataset`. Applies preprocessing
    via NVTabular `Workflow` objects and outputs tabular dictionaries of TensorFlow
    Tensors via `dlpack <https://github.com/dmlc/dlpack>`_. Useful for training tabular models
    built in Keras and trained via
    `tf.keras.Model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_.

    The data loading scheme is implemented by loading, preprocessing, and
    batching data in an asynchronous thread. The amount of randomness in
    shuffling is controlled by the `buffer_size` and `parts_per_chunk`
    kwargs. At load time, sub-chunks of data with size controlled by
    `buffer_size` are loaded from random partitions in the dataset,
    and `parts_per_chunk` of them are concatenated into a single chunk,
    shuffled, and split into batches. This means that each chunk has
    `buffer_size*parts_per_chunk` rows, and due to the asynchronous
    nature of the dataloader that means there are, including the batch
    being processed by your network, `3*buffer_size*parts_per_chunk`
    rows of data in GPU memory at any given time. This means that
    for a fixed memory budget, using more `parts_per_chunk` will
    come at the expense of smaller `buffer_size`, increasing the number
    of reads and reducing throughput. The goal should be to maximize the
    total amount of memory utilized at once without going OOM and with
    the fewest number of reads to meet your epoch-level randomness needs.

    An important thing to note is that TensorFlow's default behavior
    is to claim all GPU memory for itself at initialziation time,
    which leaves none for NVTabular to load or preprocess data.
    As such, we attempt to configure TensorFlow to restrict
    its memory allocation on a given GPU using the environment variables
    `TF_MEMORY_ALLOCATION` and `TF_VISIBLE_DEVICE`. If `TF_MEMORY_ALLOCATION < 1`,
    it will be assumed that this refers to a fraction of free GPU
    memory on the given device. Otherwise, it will refer to an explicit
    allocation amount in MB. `TF_VISIBLE_DEVICE` should be an integer GPU
    index.

    Iterator output is of the form `(dict(features), list(labels))`,
    where each element of the features dict is a
    `feature_name: feature_tensor`  and each elemtn of the labels
    list is a tensor, and all tensors are of shape `(batch_size, 1)`.
    Note that this means vectorized continuous and multi-hot categorical
    features are not currently supported.
    The underlying NVTabular `Dataset` object is stored in the `data`
    attribute, and should be used for updating NVTabular `Workflow`
    statistics::

        workflow = nvt.Workflow(...)
        dataset = KerasSequenceLoader(...)
        workflow.update_stats(dataset.data.to_iter(), record_stats=True)

    Parameters
    -------------
    paths_or_dataset: str or list(str)
        Either a string representing a file pattern (see `tf.glob` for
        pattern rules), a list of filenames to be iterated through, or
        a Dataset object, in which case `buffer_size`, `engine`, and
        `reader_kwargs` will be ignored
    batch_size: int
        Number of samples to yield at each iteration
    label_names: list(str)
        Column name of the target variable in the dataframe specified by
        `paths_or_dataset`
    feature_columns: list(tf.feature_column) or None
        A list of TensorFlow feature columns representing the inputs
        exposed to the model to be trained. Columns with parent columns
        will climb the parent tree, and the names of the columns in the
        unique set of terminal columns will be used as the column names.
        If left as None, must specify `cat_names` and `cont_names`
    cat_names: list(str) or None
        List of categorical column names. Ignored if `feature_columns` is
        specified
    cont_names: list(str) or None
        List of continuous column names. Ignored if `feature_columns` is
        specified
    engine: {'csv', 'parquet', None}, default None
        String specifying the type of read engine to use. If left as `None`,
        will try to infer the engine type from the file extension.
    shuffle: bool, default True
        Whether to shuffle chunks of batches before iterating through them.
    buffer_size: float or int
        If `0 <  buffer_size < 1`, `buffer_size` will refer to the fraction of
        total GPU memory to occupy with a buffered chunk. If `1 < buffer_size <
        batch_size`, the number of rows read for a buffered chunk will
        be equal to `int(buffer_size*batch_size)`. Otherwise, if `buffer_size >
        batch_size`, `buffer_size` rows will be read in each chunk (except for
        the last chunk in a dataset, which will, in general, be smaller).
        Larger chunk sizes will lead to more efficiency and randomness,
        but require more memory.
    device: None
        Which GPU device to load from. Ignored for now
    parts_per_chunk: int
        Number of dataset partitions with size dictated by `buffer_size`
        to load and concatenate asynchronously. More partitions leads to
        better epoch-level randomness but can negatively impact throughput
    reader_kwargs: dict
        extra kwargs to pass when instantiating the underlying
        `nvtabular.Dataset`
    sparse_list : list(str) or None
        list with column names of columns that should be represented as sparse tensors
    sparse_max : dict
        dictionary of key: column_name + value: integer representing max sequence length for column
    sparse_as_dense : bool
        bool value to activate transforming sparse tensors to dense
    """

    def __init__(
        self,
        paths_or_dataset,
        batch_size,
        label_names=None,
        feature_columns=None,
        cat_names=None,
        cont_names=None,
        engine=None,
        shuffle=True,
        seed_fn=None,
        buffer_size=0.1,
        device=None,
        parts_per_chunk=1,
        reader_kwargs=None,
        global_size=None,
        global_rank=None,
        drop_last=False,
        sparse_names=None,
        sparse_max=None,
        sparse_as_dense=False,
        schema=None,
    ):
        dataset = _validate_dataset(
            paths_or_dataset, batch_size, buffer_size, engine, device, reader_kwargs
        )
        if schema:
            dataset.schema = schema

        cat_names = cat_names or (
            dataset.schema.select_by_tag(Tags.CATEGORICAL)
            .excluding_by_tag(Tags.TARGET)
            .column_names
            if _get_schema(dataset)
            else []
        )
        cont_names = cont_names or (
            dataset.schema.select_by_tag(Tags.CONTINUOUS).excluding_by_tag(Tags.TARGET).column_names
            if _get_schema(dataset)
            else []
        )
        label_names = label_names or (
            dataset.schema.select_by_tag(Tags.TARGET).column_names if _get_schema(dataset) else []
        )

        cat_names, cont_names, label_names = _validate_schema(
            feature_columns, cat_names, cont_names, label_names, schema=dataset.schema
        )

        dataset.schema = _augment_schema(
            dataset.schema,
            cat_names,
            cont_names,
            label_names,
            sparse_names,
            sparse_max,
            sparse_as_dense,
        )

        device = "cpu" if not HAS_GPU else device
        if hvd_installed and hvd.size() > 1:
            device = hvd.local_rank()
            global_size = global_size or hvd.size()
            global_rank = global_rank or hvd.rank()
            seed_fn = seed_fn or get_default_hvd_seed_fn()

        super().__init__(
            dataset,
            batch_size,
            shuffle=shuffle,
            seed_fn=seed_fn,
            parts_per_chunk=parts_per_chunk,
            device=device,
            global_size=global_size,
            global_rank=global_rank,
            drop_last=drop_last,
        )

        # Override these parameters after initializing the parent dataloader
        # class since the new dataloader will use sparse tensors for list
        # columns by default, but sparse tensors were disabled by default
        # and were optional in the old version of merlin.loader.
        self.sparse_names = sparse_names or []
        self.sparse_max = sparse_max or {}
        self.sparse_as_dense = sparse_as_dense

    @property
    def output_schema(self) -> Schema:
        output_schema = super().output_schema
        for map_fn in self._map_fns:
            if hasattr(map_fn, "compute_output_schema"):
                output_schema = map_fn.compute_output_schema(output_schema)
            else:
                raise ValueError(
                    f"Couldn't infer schema from transform {map_fn}. "
                    "Please implement the `compute_output_schema` method on "
                    "the transform layer."
                )

        return output_schema

    @property
    def has_transforms(self) -> bool:
        """Returns True if Loader has transforms or map functions.

        Returns
        -------
        bool
            True if Loader has transforms or map functions, otherwise False.
        """
        return len(self._map_fns) > 0 or self.transforms is not None


KerasSequenceValidater = (
    KerasSequenceValidator
) = merlin.dataloader.tensorflow.KerasSequenceValidater


def sample_batch(
    dataset_or_loader: Union[Dataset, Loader],
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = False,
    include_targets: Optional[bool] = True,
    prepare_features: Optional[bool] = True,
):
    """Util function to generate a batch of input tensors from a merlin.io.Dataset instance

    Parameters
    ----------
    data: merlin.io.dataset
        A Dataset object.
    batch_size: int
        Number of samples to return.
    shuffle: bool
        Whether to sample a random batch or not, by default False.
    include_targets: bool
        Whether to include the targets in the returned batch, by default True.
    prepare_features: bool
        Whether to prepare features from dataloader for the model, by default False.
        If enabled, it converts multi-hot/list features to dense or ragged based on the schema.
        It also ensures that scalar features are converted to 2D (batch size, 1).
        P.s. The features are automatically prepared by InputBlockV2 if it is used
    Returns:
    -------
    batch: Dict[tf.tensor]
        dictionary of input tensors.
    """

    from merlin.models.tf.transforms.features import PrepareFeatures

    if isinstance(dataset_or_loader, Dataset):
        if not batch_size:
            raise ValueError("Either use 'Loader' or specify 'batch_size'")
        loader = Loader(dataset_or_loader, batch_size=batch_size, shuffle=shuffle)
    else:
        loader = dataset_or_loader

    batch = loader.peek()
    # batch could be of type Prediction, so we can't unpack directly
    inputs, targets = batch[0], batch[1]

    if prepare_features:
        pf = PrepareFeatures(loader.schema)
        if targets is not None:
            inputs, targets = pf(inputs, targets=targets)
        else:
            inputs = pf(inputs)

    if not include_targets:
        return inputs
    return inputs, targets


def get_default_hvd_seed_fn(seed=None):
    """
    Generate consistent dataloader shuffle seeds across workers
    Reseeds each worker's dataloader each epoch to get fresh a shuffle
    that's consistent across workers.
    """
    if HAS_GPU:
        import cupy

        cupy.random.seed(seed)
    else:
        np.random.seed(seed)

    if hvd_installed:
        import horovod
    else:
        raise ImportError("'horovod' is required to use this function.")

    def _seed_fn():
        min_int, max_int = tf.int32.limits
        max_rand = max_int // horovod.tensorflow.keras.size()
        # Generate a seed fragment on each worker
        if HAS_GPU:
            seed_fragment = cupy.random.randint(0, max_rand).get()
        else:
            seed_fragment = np.random.randint(0, max_rand)
        # Aggregate seed fragments from all Horovod workers
        seed_tensor = tf.constant(seed_fragment)
        reduced_seed = hvd.allreduce(
            seed_tensor, name="shuffle_seed", op=horovod.tensorflow.mpi_ops.Sum
        )

        return reduced_seed % max_rand

    return _seed_fn
