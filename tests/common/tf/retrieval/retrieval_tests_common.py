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

from typing import Callable, Optional

import fiddle as fdl

from merlin.io import Dataset
from tests.common.tf.retrieval.retrieval_config import config_retrieval_train_eval_runner
from tests.common.tf.tests_utils import extract_hparams_from_config


def set_lastfm_two_tower_hparams_config(runner_cfg: fdl.Config):
    runner_cfg.model.two_tower_activation = "selu"
    runner_cfg.model.two_tower_mlp_layers = "64"
    runner_cfg.model.two_tower_dropout = 0.2
    runner_cfg.model.two_tower_embedding_sizes_multiplier = 4.0
    runner_cfg.model.logits_temperature = 0.6
    runner_cfg.model.l2_reg = 2e-06
    runner_cfg.model.embeddings_l2_reg = 1e-05
    runner_cfg.model.logq_correction_factor = 0.7

    runner_cfg.model.samplers.neg_sampling = "inbatch"

    runner_cfg.loss.loss = "categorical_crossentropy"
    runner_cfg.loss.xe_label_smoothing = 0.3

    runner_cfg.optimizer.lr = 0.02
    runner_cfg.optimizer.lr_decay_rate = 0.97
    runner_cfg.optimizer.lr_decay_steps = 50
    runner_cfg.optimizer.optimizer = "adam"

    runner_cfg.metrics.topk_metrics_cutoffs = "10,50,100"

    runner_cfg.train_batch_size = 4096
    runner_cfg.eval_batch_size = 512

    runner_cfg.train_epochs = 20
    runner_cfg.train_steps_per_epoch = None
    runner_cfg.eval_steps = 5000

    runner_cfg.callbacks.train_batch_size = runner_cfg.train_batch_size


def set_lastfm_mf_hparams_config(runner_cfg: fdl.Config):
    runner_cfg.model.mf_dim = 64
    runner_cfg.model.logits_temperature = 1.4
    runner_cfg.model.embeddings_l2_reg = 3e-07
    runner_cfg.model.logq_correction_factor = 0.9

    runner_cfg.model.samplers.neg_sampling = "inbatch"

    runner_cfg.loss.loss = "categorical_crossentropy"
    runner_cfg.loss.xe_label_smoothing = 0.0

    runner_cfg.optimizer.lr = 0.005
    runner_cfg.optimizer.lr_decay_rate = 0.98
    runner_cfg.optimizer.lr_decay_steps = 50
    runner_cfg.optimizer.optimizer = "adam"

    runner_cfg.metrics.topk_metrics_cutoffs = "10,50,100"

    runner_cfg.train_batch_size = 4096
    runner_cfg.eval_batch_size = 512

    runner_cfg.train_epochs = 20
    runner_cfg.train_steps_per_epoch = None
    runner_cfg.eval_steps = 5000


def train_eval_two_tower(
    train_ds: Dataset,
    eval_ds: Dataset,
    train_epochs: int = 1,
    train_steps_per_epoch: Optional[int] = None,
    eval_steps: Optional[int] = 2000,
    train_batch_size: int = 512,
    eval_batch_size: int = 512,
    topk_metrics_cutoffs: str = "10,50,100",
    log_to_wandb: bool = False,
    wandb_project: str = None,
    config_callback: Callable = None,
    retrieval_api_version: int = 1,
):
    runner_cfg = config_retrieval_train_eval_runner(
        train_ds,
        eval_ds,
        model_type="two_tower",
        log_to_wandb=log_to_wandb,
        wandb_project=wandb_project,
        retrieval_api_version=retrieval_api_version,
    )

    if config_callback:
        config_callback(runner_cfg)

    runner_cfg.train_epochs = train_epochs
    runner_cfg.train_steps_per_epoch = train_steps_per_epoch
    runner_cfg.eval_steps = eval_steps
    runner_cfg.train_batch_size = train_batch_size
    runner_cfg.eval_batch_size = eval_batch_size
    runner_cfg.metrics.topk_metrics_cutoffs = topk_metrics_cutoffs

    hparams = extract_hparams_from_config(runner_cfg)

    runner = fdl.build(runner_cfg)
    metrics = runner.run(hparams)
    return metrics


def train_eval_mf(
    train_ds: Dataset,
    eval_ds: Dataset,
    train_epochs: int = 1,
    train_steps_per_epoch: Optional[int] = None,
    eval_steps: Optional[int] = 2000,
    train_batch_size: int = 512,
    eval_batch_size: int = 512,
    topk_metrics_cutoffs: str = "10,50,100",
    log_to_wandb: bool = False,
    wandb_project: str = None,
    config_callback: Callable = None,
    retrieval_api_version: int = 1,
):
    runner_cfg = config_retrieval_train_eval_runner(
        train_ds,
        eval_ds,
        model_type="mf",
        log_to_wandb=log_to_wandb,
        wandb_project=wandb_project,
        retrieval_api_version=retrieval_api_version,
    )

    if config_callback:
        config_callback(runner_cfg)

    runner_cfg.train_epochs = train_epochs
    runner_cfg.train_steps_per_epoch = train_steps_per_epoch
    runner_cfg.eval_steps = eval_steps
    runner_cfg.train_batch_size = train_batch_size
    runner_cfg.eval_batch_size = eval_batch_size
    runner_cfg.metrics.topk_metrics_cutoffs = topk_metrics_cutoffs

    runner_cfg.callbacks.train_batch_size = runner_cfg.train_batch_size

    hparams = extract_hparams_from_config(runner_cfg)

    runner = fdl.build(runner_cfg)
    metrics = runner.run(hparams)
    return metrics


def train_eval_two_tower_for_lastfm(
    train_ds: Dataset,
    eval_ds: Dataset,
    train_epochs: int = 1,
    train_steps_per_epoch: Optional[int] = None,
    eval_steps: Optional[int] = 2000,
    train_batch_size: int = 512,
    eval_batch_size: int = 512,
    topk_metrics_cutoffs: str = "10,50,100",
    log_to_wandb: bool = False,
    wandb_project: str = None,
    retrieval_api_version: int = 1,
):
    return train_eval_two_tower(
        train_ds,
        eval_ds,
        train_epochs,
        train_steps_per_epoch,
        eval_steps,
        train_batch_size,
        eval_batch_size,
        topk_metrics_cutoffs,
        log_to_wandb,
        wandb_project,
        config_callback=set_lastfm_two_tower_hparams_config,
        retrieval_api_version=retrieval_api_version,
    )


def train_eval_mf_for_lastfm(
    train_ds: Dataset,
    eval_ds: Dataset,
    train_epochs: int = 1,
    train_steps_per_epoch: Optional[int] = None,
    eval_steps: Optional[int] = 2000,
    train_batch_size: int = 512,
    eval_batch_size: int = 512,
    topk_metrics_cutoffs: str = "10,50,100",
    log_to_wandb: bool = False,
    wandb_project: str = None,
    retrieval_api_version: int = 1,
):
    return train_eval_mf(
        train_ds,
        eval_ds,
        train_epochs,
        train_steps_per_epoch,
        eval_steps,
        train_batch_size,
        eval_batch_size,
        topk_metrics_cutoffs,
        log_to_wandb,
        wandb_project,
        config_callback=set_lastfm_mf_hparams_config,
        retrieval_api_version=retrieval_api_version,
    )
