import logging
import os

import fiddle as fdl

from tests.integration.tf.retrieval.retrieval_config import config_retrieval_train_eval_runner
from tests.integration.tf.utils import extract_hparams_from_config

STANDARD_CI_LASTFM_PREPROC_DATA_PATH = "/raid/data/lastfm/preprocessed"
STANDARD_CI_WANDB_PROJECT = "merlin-ci"


def test_train_eval_two_tower():
    runner_cfg = config_retrieval_train_eval_runner(
        model_type="two_tower",
        data_path=os.getenv("CI_LASTFM_PREPROC_DATA_PATH", STANDARD_CI_LASTFM_PREPROC_DATA_PATH),
        log_to_wandb=True,
        wandb_project=os.getenv("CI_WANDB_PROJECT", STANDARD_CI_WANDB_PROJECT),
    )

    # Hparams based on https://wandb.ai/nvidia-merlin/retrieval_models/runs/3a9ja3ms/overview
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
    runner_cfg.eval_batch_size = 1024

    # Changed to speed up execution
    runner_cfg.train_epochs = 3
    runner_cfg.eval_steps = 5000
    # runner_cfg.train_steps_per_epoch = 10

    runner_cfg.callbacks.train_batch_size = runner_cfg.train_batch_size

    hparams = extract_hparams_from_config(runner_cfg)

    runner = fdl.build(runner_cfg)
    metrics = runner.run(hparams)
    logging.info(f"FINAL METRICS: {metrics}")


def test_train_eval_mf():
    runner_cfg = config_retrieval_train_eval_runner(
        model_type="mf",
        data_path=os.getenv("CI_LASTFM_PREPROC_DATA_PATH", STANDARD_CI_LASTFM_PREPROC_DATA_PATH),
        log_to_wandb=True,
        wandb_project=os.getenv("CI_WANDB_PROJECT", STANDARD_CI_WANDB_PROJECT),
    )

    # Hparams based on https://wandb.ai/nvidia-merlin/retrieval_models/runs/2q6d7r7j
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
    runner_cfg.eval_batch_size = 1024

    # Changed to speed up execution
    runner_cfg.train_epochs = 3
    runner_cfg.eval_steps = 5000
    # runner_cfg.train_steps_per_epoch = 10

    runner_cfg.callbacks.train_batch_size = runner_cfg.train_batch_size

    hparams = extract_hparams_from_config(runner_cfg)

    runner = fdl.build(runner_cfg)
    metrics = runner.run(hparams)
    logging.info(f"FINAL METRICS: {metrics}")
