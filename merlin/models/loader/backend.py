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
from collections import OrderedDict

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = np

from merlin.core.dispatch import HAS_GPU, annotate, make_df, pull_apart_list
from merlin.loader.loader_base import LoaderBase
from merlin.schema import Tags


def _get_dataset_schema(dataset):
    return dataset.schema if hasattr(dataset, "schema") else None


# TODO: implement as metaclass and assign methods to children
# to avoid having to do Dataset.<method> calls?
class DataLoader(LoaderBase):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        cat_names=None,
        cont_names=None,
        label_names=None,
        seed_fn=None,
        parts_per_chunk=1,
        device=None,
        global_size=None,
        global_rank=None,
        drop_last=False,
        sparse_names=None,
        sparse_max=None,
        sparse_as_dense=False,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed_fn=seed_fn,
            parts_per_chunk=parts_per_chunk,
            global_size=global_size,
            global_rank=global_rank,
            drop_last=drop_last,
        )

        self.data = dataset
        self.schema = _get_dataset_schema(dataset)

        self.indices = cp.arange(self.data.npartitions)
        self.device = (device or 0) if HAS_GPU else "cpu"

        self.sparse_names = sparse_names or []
        self.sparse_max = sparse_max or {}
        self.sparse_as_dense = sparse_as_dense

        self.cat_names = cat_names or (
            self.schema.select_by_tag(Tags.CATEGORICAL).excluding_by_tag(Tags.TARGET).column_names
            if self.schema
            else []
        )
        self.cont_names = cont_names or (
            self.schema.select_by_tag(Tags.CONTINUOUS).excluding_by_tag(Tags.TARGET).column_names
            if self.schema
            else []
        )
        self.label_names = label_names or (
            self.schema.select_by_tag(Tags.TARGET).column_names if self.schema else []
        )

        if not self.cat_names and not self.cont_names:
            raise ValueError(
                "Neither Categorical or Continuous columns were found by the dataloader. "
                "You must either specify the cat_names, cont_names and "
                "label_names properties or supply a schema.pbtxt file in dataset directory."
            )

        self.__buff = None
        self.__buff_len = None
        self._batch_itr = None
        self._workers = None

    @annotate("make_tensors", color="darkgreen", domain="merlin_dataloader")
    def make_tensors(self, gdf, use_nnz=False):
        split_idx = self._get_segment_lengths(len(gdf))

        # map from big chunk to framework-specific tensors
        chunks = self._create_tensors(gdf)

        # if we have any offsets, calculate nnzs up front
        if len(chunks) == 4:
            offsets = chunks[-1]
            if use_nnz:
                nnzs = offsets[1:] - offsets[:-1]
            chunks = chunks[:-1]

        # split them into batches and map to the framework-specific output format
        batches = [[] for _ in range(len(split_idx))]
        offset_idx = 0
        for chunk in chunks:
            lists = None
            if isinstance(chunk, tuple):
                chunk, lists = chunk

            if len(split_idx) > 1 and chunk is not None:
                chunk = self._split_fn(chunk, split_idx)
            else:
                chunk = [chunk for _ in split_idx]

            if lists is not None:
                num_list_columns = len(lists)

                # grab the set of offsets and nnzs corresponding to
                # the list columns from this chunk
                chunk_offsets = offsets[:, offset_idx : offset_idx + num_list_columns]
                if use_nnz:
                    chunk_nnzs = nnzs[:, offset_idx : offset_idx + num_list_columns]
                offset_idx += num_list_columns

                # split them into batches, including an extra 1 on the offsets
                # so we know how long the very last element is
                batch_offsets = self._split_fn(chunk_offsets, split_idx + [1])
                if use_nnz and len(split_idx) > 1:
                    batch_nnzs = self._split_fn(chunk_nnzs, split_idx)
                elif use_nnz:
                    batch_nnzs = [chunk_nnzs]
                else:
                    batch_nnzs = [None] * (len(batch_offsets) - 1)

                # group all these indices together and iterate through
                # them in batches to grab the proper elements from each
                # values tensor
                chunk = zip(chunk, batch_offsets[:-1], batch_offsets[1:], batch_nnzs)

            for n, c in enumerate(chunk):
                if isinstance(c, tuple):
                    c, off0s, off1s, _nnzs = c
                    offsets_split_idx = [1 for _ in range(num_list_columns)]
                    off0s = self._split_fn(off0s, offsets_split_idx, axis=1)
                    off1s = self._split_fn(off1s, offsets_split_idx, axis=1)
                    if use_nnz:
                        _nnzs = self._split_fn(_nnzs, offsets_split_idx, axis=1)

                    # TODO: does this need to be ordereddict?
                    batch_lists = {}
                    for k, (column_name, values) in enumerate(lists.items()):
                        off0, off1 = off0s[k], off1s[k]
                        if use_nnz:
                            nnz = _nnzs[k]

                        # need to grab scalars for TF case
                        if len(off0.shape) == 1:
                            start, stop = off0[0], off1[0]
                        elif len(off0.shape) == 2:
                            start, stop = off0[0, 0], off1[0, 0]
                        else:
                            print(off0, off1)
                            raise ValueError
                        value = values[int(start) : int(stop)]
                        index = off0 - start if not use_nnz else nnz
                        batch_lists[column_name] = (value, index)
                    c = (c, batch_lists)

                batches[n].append(c)
        return (self._handle_tensors(*batch) for batch in batches)

    @annotate("_create_tensors", color="darkgreen", domain="merlin_dataloader")
    def _create_tensors(self, gdf):
        """
        Breaks a dataframe down into the relevant
        categorical, continuous, and label tensors.
        Can be overrideen
        """
        workflow_nodes = (self.cat_names, self.cont_names, self.label_names)
        dtypes = (self._LONG_DTYPE, self._FLOAT32_DTYPE, self._FLOAT32_DTYPE)
        tensors = []
        offsets = make_df(device=self.device)
        for column_names, dtype in zip(workflow_nodes, dtypes):
            if len(column_names) == 0:
                tensors.append(None)
                continue
            if hasattr(column_names, "column_names"):
                column_names = column_names.column_names

            gdf_i = gdf[column_names]
            gdf.drop(columns=column_names, inplace=True)

            scalars, lists = self._separate_list_columns(gdf_i)

            x = None
            if scalars:
                # should always return dict column_name: values, offsets (optional)
                x = self._to_tensor(gdf_i[scalars])
            if lists:
                list_tensors = OrderedDict()
                for column_name in lists:
                    column = gdf_i.pop(column_name)
                    leaves, col_offsets = pull_apart_list(column)
                    if isinstance(leaves[0], list):

                        leaves, nest_offsets = pull_apart_list(leaves)
                        col_offsets = nest_offsets.iloc[col_offsets[:]]
                    offsets[column_name] = col_offsets.reset_index(drop=True)
                    list_tensors[column_name] = self._to_tensor(leaves)
                x = x, list_tensors
            tensors.append(x)

        if not offsets.empty:
            offsets_tensor = self._to_tensor(offsets)
            if len(offsets_tensor.shape) == 1:
                offsets_tensor = offsets_tensor[:, None]
            tensors.append(offsets_tensor)
        del gdf, offsets

        return tensors

    @annotate("_handle_tensors", color="darkgreen", domain="merlin_dataloader")
    def _handle_tensors(self, cats, conts, labels):
        X = {}
        for tensor, names in zip([cats, conts], [self.cat_names, self.cont_names]):
            lists = {}
            if isinstance(tensor, tuple):
                tensor, lists = tensor
            names = [i for i in names if i not in lists]

            # now add in any scalar tensors
            if len(names) > 1:
                tensors = self._tensor_split(tensor, len(names), axis=1)
                lists.update(zip(names, tensors))
            elif len(names) == 1:
                lists[names[0]] = tensor
            X.update(lists)

        for column_name in X:
            if column_name in self.sparse_names:
                if column_name not in self.sparse_max:
                    raise ValueError(
                        f"Did not convert {column_name} to sparse due to missing sparse_max entry"
                    )
                X[column_name] = self._to_sparse_tensor(X[column_name], column_name)

        # TODO: use dict for labels as well?
        # would require output layers to match naming
        if len(self.label_names) > 1:
            labels = self._tensor_split(labels, len(self.label_names), axis=1)
        return X, labels
