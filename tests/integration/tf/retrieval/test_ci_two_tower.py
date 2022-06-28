import os

import fiddle as fdl

from tests.integration.tf.retrieval.retrieval_config import config_retrieval_train_eval_runner

# STANDARD_CI_PATH = "/raid/data/lastfm/preproc_retrieval"
STANDARD_CI_PATH = "/mnt/nvme0n1/datasets/lastfm_1b/lfm1b_B_preprocessed_v01"


# TODO: Create an example of command line script passing hparams as arguments


def test_train_eval_two_tower():
    runner_cfg = config_retrieval_train_eval_runner(
        model_type="two_tower",
        data_path=os.getenv("IT_RETRIEVAL_INPUT_PATH", STANDARD_CI_PATH),
    )
    runner_cfg.metrics.topk_metrics_cutoffs = "10,50,100"
    runner_cfg.optimizer.lr = 1e-5
    runner = fdl.build(runner_cfg)
    runner.run()


def test_train_eval_mf():
    runner_cfg = config_retrieval_train_eval_runner(
        model_type="mf",
        data_path=os.getenv("IT_RETRIEVAL_INPUT_PATH", STANDARD_CI_PATH),
    )
    runner_cfg.loss = "bpr-max"
    runner = fdl.build(runner_cfg)
    runner.run()
