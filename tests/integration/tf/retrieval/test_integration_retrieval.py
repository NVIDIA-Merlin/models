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

import os

import pytest

from merlin.io import Dataset
from tests.common.tf.retrieval import retrieval_tests_common

STANDARD_CI_LASTFM_PREPROC_DATA_PATH = "/raid/data/lastfm/preprocessed"
STANDARD_CI_WANDB_PROJECT = "merlin-ci"


@pytest.fixture
def train_eval_datasets_last_fm():
    data_path = os.getenv("CI_LASTFM_PREPROC_DATA_PATH", STANDARD_CI_LASTFM_PREPROC_DATA_PATH)
    train_ds = Dataset(os.path.join(data_path, "train/*.parquet"), part_size="500MB")
    eval_ds = Dataset(os.path.join(data_path, "valid/*.parquet"), part_size="500MB")
    return train_ds, eval_ds


@pytest.mark.parametrize("version", [1, 2])
def test_integration_train_eval_two_tower(train_eval_datasets_last_fm, version):
    train_ds, eval_ds = train_eval_datasets_last_fm
    metrics = retrieval_tests_common.train_eval_two_tower_for_lastfm(
        train_ds,
        eval_ds,
        train_epochs=1,
        train_steps_per_epoch=None,
        eval_steps=2000,
        train_batch_size=4096,
        eval_batch_size=512,
        log_to_wandb=True,
        wandb_project=os.getenv("CI_WANDB_PROJECT", STANDARD_CI_WANDB_PROJECT),
        retrieval_api_version=version,
    )
    assert metrics["loss-final"] > 0.0
    assert metrics["recall_at_100-final"] > 0.0
    assert metrics["runtime_sec-final"] > 0.0
    assert metrics["avg_examples_per_sec-final"] > 0.0


@pytest.mark.parametrize("version", [1, 2])
def test_integration_train_eval_mf(train_eval_datasets_last_fm, version):
    train_ds, eval_ds = train_eval_datasets_last_fm
    metrics = retrieval_tests_common.train_eval_mf_for_lastfm(
        train_ds,
        eval_ds,
        train_epochs=1,
        train_steps_per_epoch=None,
        eval_steps=2000,
        train_batch_size=4096,
        eval_batch_size=512,
        log_to_wandb=True,
        wandb_project=os.getenv("CI_WANDB_PROJECT", STANDARD_CI_WANDB_PROJECT),
        retrieval_api_version=version,
    )
    assert metrics["loss-final"] > 0.0
    assert metrics["recall_at_100-final"] > 0.0
    assert metrics["runtime_sec-final"] > 0.0
    assert metrics["avg_examples_per_sec-final"] > 0.0
