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

import fiddle as fdl

from merlin.io import Dataset
from tests.common.tf.retrieval.retrieval_utils import (
    RetrievalTrainEvalRunner,
    WandbLogger,
    filter_schema,
    get_callbacks,
    get_dual_encoder_model,
    get_item_frequencies,
    get_loss,
    get_metrics,
    get_optimizer,
    get_samplers,
    get_youtube_dnn_model,
)


def config_retrieval_train_eval_runner(
    train_ds: Dataset, eval_ds: Dataset, model_type: str, log_to_wandb: bool, wandb_project: str
):
    def make_model(schema, train_ds, model_type):
        if model_type == "youtubednn":
            model = fdl.Config(get_youtube_dnn_model, schema)
        else:
            samplers = fdl.Config(get_samplers, schema)
            items_frequencies = fdl.Config(get_item_frequencies, schema, train_ds)
            model = fdl.Config(
                get_dual_encoder_model, schema, samplers, items_frequencies, model_type
            )
        return model

    wandb_logger_cfg = fdl.Config(WandbLogger, enabled=log_to_wandb, wandb_project=wandb_project)

    schema_cfg = fdl.Config(filter_schema, schema=train_ds.schema)

    model_cfg = make_model(schema_cfg, train_ds, model_type=model_type)
    optimizer = fdl.Config(get_optimizer)
    metrics = fdl.Config(get_metrics)
    loss = fdl.Config(get_loss)
    callbacks = fdl.Config(get_callbacks, wandb_logger=wandb_logger_cfg)

    runner_cfg = fdl.Config(
        RetrievalTrainEvalRunner,
        wandb_logger=wandb_logger_cfg,
        model_type=model_type,
        schema=schema_cfg,
        train_ds=train_ds,
        eval_ds=eval_ds,
        model=model_cfg,
        optimizer=optimizer,
        metrics=metrics,
        loss=loss,
        callbacks=callbacks,
    )

    return runner_cfg
