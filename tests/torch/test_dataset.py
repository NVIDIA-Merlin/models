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
import numpy as np
import pandas as pd
import pytest
import torch

from merlin.core.dispatch import HAS_GPU, make_df
from merlin.io.dataset import Dataset

import merlin.models.torch.dataset as torch_dataloader  # noqa isort:skip


def test_shuffling():
    num_rows = 10000
    batch_size = 10000

    df = pd.DataFrame({"a": np.asarray(range(num_rows)), "b": np.asarray([0] * num_rows)})

    train_dataset = torch_dataloader.Dataset(
        Dataset(df), conts=["a"], labels=["b"], batch_size=batch_size, shuffle=True
    )

    batch = next(iter(train_dataset))
    first_batch = batch[0]["a"].cpu()
    in_order = torch.arange(0, batch_size)

    assert (first_batch != in_order).any()
    assert (torch.sort(first_batch).values == in_order).all()


@pytest.mark.parametrize("batch_size", [10, 9, 8])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("num_rows", [100])
def test_torch_drp_reset(tmpdir, batch_size, drop_last, num_rows):
    df = make_df(
        {
            "cat1": [1] * num_rows,
            "cat2": [2] * num_rows,
            "cat3": [3] * num_rows,
            "label": [0] * num_rows,
            "cont3": [3.0] * num_rows,
            "cont2": [2.0] * num_rows,
            "cont1": [1.0] * num_rows,
        }
    )
    cat_names = ["cat3", "cat2", "cat1"]
    cont_names = ["cont3", "cont2", "cont1"]
    label_name = ["label"]

    data_itr = torch_dataloader.Dataset(
        Dataset(df),
        cats=cat_names,
        conts=cont_names,
        labels=label_name,
        batch_size=batch_size,
        drop_last=drop_last,
        device="cpu",
    )

    all_len = len(data_itr) if drop_last else len(data_itr) - 1
    all_rows = 0
    df_cols = df.columns.to_list()
    for idx, chunk in enumerate(data_itr):
        all_rows += len(chunk[0]["cat1"])
        if idx < all_len:
            for col in df_cols:
                if col in chunk[0].keys():
                    if HAS_GPU:
                        assert (list(chunk[0][col].cpu().numpy()) == df[col].values_host).all()
                    else:
                        assert (list(chunk[0][col].cpu().numpy()) == df[col].values).all()

    if drop_last and num_rows % batch_size > 0:
        assert num_rows > all_rows
    else:
        assert num_rows == all_rows


@pytest.mark.parametrize("sparse_dense", [False, True])
def test_sparse_tensors(sparse_dense):
    # create small Dataset, add values to sparse_list
    df = make_df(
        {
            "spar1": [[1, 2, 3, 4], [4, 2, 4, 4], [1, 3, 4, 3], [1, 1, 3, 3]],
            "spar2": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14], [15, 16]],
        }
    )
    spa_lst = ["spar1", "spar2"]
    spa_mx = {"spar1": 5, "spar2": 6}
    batch_size = 2
    data_itr = torch_dataloader.Dataset(
        Dataset(df),
        cats=spa_lst,
        conts=[],
        labels=[],
        batch_size=batch_size,
        sparse_names=spa_lst,
        sparse_max=spa_mx,
        sparse_as_dense=sparse_dense,
    )
    for batch in data_itr:
        feats, labs = batch
        for col in spa_lst:
            feature_tensor = feats[col]
            if not sparse_dense:
                assert list(feature_tensor.shape) == [batch_size, spa_mx[col]]
                assert feature_tensor.is_sparse
            else:
                assert feature_tensor.shape[1] == spa_mx[col]
                assert not feature_tensor.is_sparse
