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
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras import regularizers
from tensorflow.keras.utils import set_random_seed

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.models.tf.outputs.base import DotProduct
from merlin.models.tf.transforms.bias import PopularityLogitsCorrection
from merlin.models.utils import schema_utils
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.schema.tags import Tags


def filter_schema(schema, filter_dataset_features=""):
    schema = schema.select_by_tag([Tags.ITEM_ID, Tags.USER_ID, Tags.ITEM, Tags.USER])

    if filter_dataset_features:
        schema = schema.select_by_name(filter_dataset_features.split(","))

    return schema


def get_dataset(schema, data_path, dataset):
    ds = Dataset(os.path.join(data_path, f"{dataset}/*.parquet"), part_size="500MB", schema=schema)
    return ds


def get_samplers(schema, neg_sampling="inbatch", cached_negatives_capacity=16):
    if neg_sampling == "inbatch":
        samplers = [mm.InBatchSampler()]
    elif neg_sampling == "cached_crossbatch":
        samplers = [
            mm.CachedCrossBatchSampler(
                capacity=cached_negatives_capacity,
                ignore_last_batch_on_sample=False,
            )
        ]
    elif neg_sampling == "cached_uniform":
        item_id_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        samplers = [
            mm.CachedUniformSampler(
                capacity=cached_negatives_capacity,
                ignore_last_batch_on_sample=False,
                item_id_feature_name=item_id_name,
            )
        ]
    elif neg_sampling == "inbatch+cached_crossbatch":
        samplers = [
            mm.InBatchSampler(),
            mm.CachedCrossBatchSampler(
                capacity=cached_negatives_capacity,
                ignore_last_batch_on_sample=True,
            ),
        ]
    elif neg_sampling == "inbatch+cached_uniform":
        item_id_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        samplers = [
            mm.InBatchSampler(),
            mm.CachedUniformSampler(
                capacity=cached_negatives_capacity,
                ignore_last_batch_on_sample=True,
                item_id_feature_name=item_id_name,
            ),
        ]
    else:
        raise Exception(f"Invalid neg_sampling option: {neg_sampling}")
    return samplers


def get_samplers_v2(schema, neg_sampling="inbatch", cached_negatives_capacity=16):
    if neg_sampling == "inbatch":
        samplers = [mm.InBatchSamplerV2()]
    else:
        raise Exception(f"Invalid neg_sampling option: {neg_sampling}")
    return samplers


def get_item_id_cardinality(schema):
    item_id_feature = schema.select_by_tag(Tags.ITEM_ID)
    item_id_feature_name = item_id_feature.column_names[0]

    cardinalities = schema_utils.categorical_cardinalities(schema)
    item_id_cardinality = cardinalities[item_id_feature_name]
    return item_id_cardinality


def get_item_frequencies(schema, train_ds):
    item_id_cardinality = get_item_id_cardinality(schema)

    item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]

    item_frequencies_df = (
        train_ds.to_ddf().groupby(item_id_feature_name).size().to_frame("freq").compute()
    )
    assert len(item_frequencies_df) <= item_id_cardinality
    assert item_frequencies_df.index.max() < item_id_cardinality

    # Completing the missing item ids and filling freq with 0
    item_frequencies_df = item_frequencies_df.reindex(np.arange(0, item_id_cardinality)).fillna(0)
    assert len(item_frequencies_df) == item_id_cardinality

    item_frequencies_df = item_frequencies_df.sort_index()
    item_frequencies_df["dummy"] = 1
    item_frequencies_df["expected_id"] = item_frequencies_df["dummy"].cumsum() - 1
    assert (
        item_frequencies_df.index == item_frequencies_df["expected_id"]
    ).all(), f"The item id feature ({item_id_feature_name}) "
    f"should be contiguous from 0 to {item_id_cardinality-1}"

    item_frequencies = tf.convert_to_tensor(item_frequencies_df["freq"].values.tolist())

    return item_frequencies


def get_metrics(topk_metrics_cutoffs="50,100"):
    topk_metrics = []
    for cutoff in topk_metrics_cutoffs.split(","):
        cutoff = int(cutoff)
        topk_metrics.append(mm.RecallAt(cutoff))
        topk_metrics.append(mm.MRRAt(cutoff))
        topk_metrics.append(mm.NDCGAt(cutoff))

    topk_metrics_aggregator = mm.TopKMetricsAggregator(*topk_metrics)

    return [topk_metrics_aggregator]


def get_embeddings_initializer(emb_init_distr="truncated_normal", emb_init_range=0.05):
    if emb_init_distr == "uniform":
        emb_init = tf.keras.initializers.RandomUniform(
            minval=-emb_init_range, maxval=emb_init_range
        )
    elif emb_init_distr == "truncated_normal":
        emb_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=emb_init_range)
    else:
        raise ValueError("Invalid embedding initialization option")
    return emb_init


def get_loss(loss="categorical_crossentropy", xe_label_smoothing=0.0, bprmax_reg=1.0):
    if loss == "categorical_crossentropy":
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=xe_label_smoothing,
        )
    elif loss == "bpr-max":
        loss = mm.losses.BPRmaxLoss(reg_lambda=bprmax_reg)
    return loss


def get_dual_encoder_model(
    schema,
    samplers,
    items_frequencies,
    model_type,
    emb_init_distr="truncated_normal",
    emb_init_range=0.05,
    logq_correction_factor=0.0,
    logits_temperature=0.1,
    mf_dim=128,
    embeddings_l2_reg=0.0,
    cosine_similarity_logits=False,
    two_tower_mlp_layers="64",
    item_id_emb_size=64,
    user_id_emb_size=64,
    two_tower_activation="relu",
    two_tower_mlp_init="he_normal",
    two_tower_dropout=0.0,
    l2_reg=1e-5,
    two_tower_embedding_sizes_multiplier=2.0,
):
    post_logits = None
    if logq_correction_factor > 0:
        post_logits = PopularityLogitsCorrection(
            items_frequencies, reg_factor=logq_correction_factor, schema=schema
        )

    retrieval_task = mm.ItemRetrievalTask(
        samplers=samplers,
        schema=schema,
        logits_temperature=logits_temperature,
        post_logits=post_logits,
        store_negative_ids=True,
    )

    emb_init = get_embeddings_initializer(emb_init_distr, emb_init_range)

    if model_type == "mf":
        model = mm.MatrixFactorizationModel(
            schema,
            dim=mf_dim,
            prediction_tasks=retrieval_task,
            embeddings_initializers=emb_init,
            embeddings_l2_reg=embeddings_l2_reg,
            l2_normalization=cosine_similarity_logits,
        )
    elif model_type == "two_tower":
        layers_dims = list([int(v.strip()) for v in two_tower_mlp_layers.split(",")])

        embedding_dims = {}
        if item_id_emb_size:
            item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
            item_id_domain = schema_utils.categorical_domains(schema)[item_id_feature_name]
            embedding_dims[item_id_domain] = item_id_emb_size
        if user_id_emb_size:
            user_id_feature_name = schema.select_by_tag(Tags.USER_ID).column_names[0]
            user_id_domain = schema_utils.categorical_domains(schema)[user_id_feature_name]
            embedding_dims[user_id_domain] = user_id_emb_size

        model = mm.TwoTowerModel(
            schema,
            query_tower=mm.MLPBlock(
                layers_dims,
                activation=two_tower_activation,
                kernel_initializer=two_tower_mlp_init,
                no_activation_last_layer=True,
                dropout=two_tower_dropout,
                kernel_regularizer=regularizers.l2(l2_reg),
                bias_regularizer=regularizers.l2(l2_reg),
            ),
            item_tower=mm.MLPBlock(
                layers_dims,
                activation=two_tower_activation,
                kernel_initializer=two_tower_mlp_init,
                no_activation_last_layer=True,
                dropout=two_tower_dropout,
                kernel_regularizer=regularizers.l2(l2_reg),
                bias_regularizer=regularizers.l2(l2_reg),
            ),
            embedding_options=mm.EmbeddingOptions(
                embedding_dims=embedding_dims,
                infer_embedding_sizes=True,
                infer_embedding_sizes_multiplier=two_tower_embedding_sizes_multiplier,
                infer_embeddings_ensure_dim_multiple_of_8=True,
                embeddings_initializers=emb_init,
                embeddings_l2_reg=embeddings_l2_reg,
            ),
            prediction_tasks=retrieval_task,
            l2_normalization=cosine_similarity_logits,
        )
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    return model


def get_dual_encoder_model_v2(
    schema,
    samplers,
    items_frequencies,
    model_type,
    emb_init_distr="truncated_normal",
    emb_init_range=0.05,
    logq_correction_factor=0.0,
    logits_temperature=0.1,
    mf_dim=128,
    embeddings_l2_reg=0.0,
    cosine_similarity_logits=False,
    two_tower_mlp_layers="64",
    item_id_emb_size=64,
    user_id_emb_size=64,
    two_tower_activation="relu",
    two_tower_mlp_init="he_normal",
    two_tower_dropout=0.0,
    l2_reg=0.0,
    two_tower_embedding_sizes_multiplier=2.0,
):
    post_logits = None
    if logq_correction_factor > 0:
        post_logits = PopularityLogitsCorrection(
            items_frequencies, reg_factor=logq_correction_factor, schema=schema
        )

    output = mm.ContrastiveOutput(
        to_call=DotProduct(),
        logits_temperature=logits_temperature,
        post=post_logits,
        negative_samplers=samplers,
        schema=schema.select_by_tag(Tags.ITEM_ID),
        store_negative_ids=True,
    )

    emb_init = get_embeddings_initializer(emb_init_distr, emb_init_range)

    if model_type == "mf":
        model = mm.MatrixFactorizationModelV2(
            schema,
            dim=mf_dim,
            outputs=output,
            embeddings_initializers=emb_init,
            embeddings_l2_batch_regularization=embeddings_l2_reg,
            post=mm.L2Norm() if cosine_similarity_logits else None,
        )

    elif model_type == "two_tower":
        layers_dims = list([int(v.strip()) for v in two_tower_mlp_layers.split(",")])

        embedding_dims = {}
        if item_id_emb_size:
            item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
            item_id_domain = schema_utils.categorical_domains(schema)[item_id_feature_name]
            embedding_dims[item_id_domain] = item_id_emb_size
        if user_id_emb_size:
            user_id_feature_name = schema.select_by_tag(Tags.USER_ID).column_names[0]
            user_id_domain = schema_utils.categorical_domains(schema)[user_id_feature_name]
            embedding_dims[user_id_domain] = user_id_emb_size

        # define query tower
        user_schema = schema.select_by_tag(Tags.USER)
        query_inputs = mm.InputBlockV2(
            user_schema,
            categorical=mm.Embeddings(
                user_schema.select_by_tag(Tags.CATEGORICAL),
                dim=embedding_dims,
                infer_dim_fn=partial(
                    schema_utils.infer_embedding_dim,
                    multiplier=two_tower_embedding_sizes_multiplier,
                    ensure_multiple_of_8=True,
                ),
                l2_batch_regularization_factor=embeddings_l2_reg,
                embeddings_initializer=emb_init,
            ),
        )
        query_block = mm.MLPBlock(
            layers_dims,
            activation=two_tower_activation,
            kernel_initializer=two_tower_mlp_init,
            no_activation_last_layer=True,
            dropout=two_tower_dropout,
            kernel_regularizer=regularizers.l2(l2_reg),
            bias_regularizer=regularizers.l2(l2_reg),
        )
        query = mm.Encoder(
            query_inputs,
            query_block,
            post=mm.L2Norm() if cosine_similarity_logits else None,
        )

        # define candidate tower
        candidate_schema = schema.select_by_tag(Tags.ITEM)
        candidate_inputs = mm.InputBlockV2(
            candidate_schema,
            categorical=mm.Embeddings(
                candidate_schema.select_by_tag(Tags.CATEGORICAL),
                dim=embedding_dims,
                infer_dim_fn=partial(
                    schema_utils.infer_embedding_dim,
                    multiplier=two_tower_embedding_sizes_multiplier,
                    ensure_multiple_of_8=True,
                ),
                l2_batch_regularization_factor=embeddings_l2_reg,
                embeddings_initializer=emb_init,
            ),
        )
        candidate_block = mm.MLPBlock(
            layers_dims,
            activation=two_tower_activation,
            kernel_initializer=two_tower_mlp_init,
            no_activation_last_layer=True,
            dropout=two_tower_dropout,
            kernel_regularizer=regularizers.l2(l2_reg),
            bias_regularizer=regularizers.l2(l2_reg),
        )
        candidate = mm.Encoder(
            candidate_inputs,
            candidate_block,
            post=mm.L2Norm() if cosine_similarity_logits else None,
        )

        model = mm.TwoTowerModelV2(
            query,
            candidate,
            outputs=output,
        )

    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    return model


def get_youtube_dnn_model(
    schema,
    item_id_emb_size=32,
    emb_init_distr="truncated_normal",
    emb_init_range=0.05,
    two_tower_mlp_layers="64",
    youtubednn_sampled_softmax_n_candidates=100,
    youtubednn_sampled_softmax=True,
    logits_temperature=0.1,
    two_tower_embedding_sizes_multiplier=2.0,
    embeddings_l2_reg=0.0,
    two_tower_activation="relu",
    two_tower_mlp_init="he_normal",
    two_tower_dropout=0.0,
    l2_reg=0.0,
    cosine_similarity_logits=False,
):
    # Keeping only categorical features and removing the user id (keeping only seq features)
    # schema_selected = schema.select_by_tag(Tags.CATEGORICAL).remove_by_tag(Tags.USER_ID)
    schema_selected = schema.remove_by_tag(Tags.USER_ID)

    embedding_dims = {}
    if item_id_emb_size:
        item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        # item_id_domain = schema_utils.categorical_domains(schema)[item_id_feature_name]
        embedding_dims[item_id_feature_name] = item_id_emb_size
    else:
        raise Exception(
            "You must specify --item_id_emb_size for YouTubeDNN "
            "as the last layer dim of the MLP must match the item id embedding dim."
        )

    emb_init = get_embeddings_initializer(emb_init_distr, emb_init_range)

    layers_dims = []

    layers_dims = list([int(v.strip()) for v in two_tower_mlp_layers.split(",") if v != ""])
    # Appends the top MLP layer with the same size as the item id embedding
    layers_dims.append(item_id_emb_size)

    # TODO: Update test to use YoutubeDNNRetrievalModelV2, as this one is deprecated
    model = mm.YoutubeDNNRetrievalModel(
        schema=schema_selected,
        num_sampled=youtubednn_sampled_softmax_n_candidates,
        sampled_softmax=youtubednn_sampled_softmax,
        logits_temperature=logits_temperature,
        embedding_options=mm.EmbeddingOptions(
            embedding_dims=embedding_dims,
            infer_embedding_sizes=True,
            infer_embedding_sizes_multiplier=two_tower_embedding_sizes_multiplier,
            infer_embeddings_ensure_dim_multiple_of_8=True,
            embeddings_initializers=emb_init,
            embeddings_l2_reg=embeddings_l2_reg,
        ),
        top_block=mm.MLPBlock(
            layers_dims,
            activation=two_tower_activation,
            kernel_initializer=two_tower_mlp_init,
            no_activation_last_layer=True,
            dropout=two_tower_dropout,
            kernel_regularizer=regularizers.l2(l2_reg),
            bias_regularizer=regularizers.l2(l2_reg),
        ),
        l2_normalization=cosine_similarity_logits,
    )
    return model


def get_optimizer(
    lr=1e-4,
    lr_decay_rate=0.99,
    lr_decay_steps=100,
    optimizer="adam",
    opt_clip_norm=None,
    opt_clip_value=None,
):
    lerning_rate = lr
    if lr_decay_rate:
        lerning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=lr_decay_steps,
            decay_rate=lr_decay_rate,
            staircase=True,
        )

    if optimizer == "adam":
        opt = tf.keras.optimizers.Adam(
            learning_rate=lerning_rate,
            clipnorm=opt_clip_norm,
            clipvalue=opt_clip_value,
        )
    elif optimizer == "adagrad":
        opt = tf.keras.optimizers.Adagrad(
            learning_rate=lerning_rate,
            clipnorm=opt_clip_norm,
            clipvalue=opt_clip_value,
        )
    else:
        raise ValueError("Invalid optimizer")

    return opt


class ExamplesPerSecondCallback(tf.keras.callbacks.Callback):
    """ExamplesPerSecond callback.
    This callback records the average_examples_per_sec and
    current_examples_per_sec during training.
    """

    def __init__(self, batch_size, every_n_steps=1, log_as_print=True, wandb_logger=None):
        self.log_as_print = log_as_print
        self.wandb_logger = wandb_logger
        self._batch_size = batch_size
        self._every_n_steps = every_n_steps
        super(ExamplesPerSecondCallback, self).__init__()

    def on_train_begin(self, logs=None):
        self._first_batch = True
        self._epoch_steps = 0
        self._train_batches_average_examples_per_sec = []
        # self._train_start_time = time.time()
        # self._last_recorded_time = time.time()

    def on_train_end(self, logs=None):
        average_examples_per_sec = self.get_avg_examples_per_sec()
        self._train_batches_average_examples_per_sec.append(average_examples_per_sec)

    def get_train_batches_mean_of_avg_examples_per_sec(self):
        if len(self._train_batches_average_examples_per_sec) > 0:
            return np.mean(self._train_batches_average_examples_per_sec)
        else:
            return 0.0

    def get_avg_examples_per_sec(self):
        current_time = time.time()
        average_examples_per_sec = self._batch_size * (
            self._epoch_steps / (current_time - self._epoch_start_time)
        )
        return average_examples_per_sec

    def on_train_batch_end(self, batch, logs=None):
        # Discards the first batch, as it is used to compile the
        # graph and affects the average
        if self._first_batch:
            self._epoch_steps = 0
            self._first_batch = False
            self._epoch_start_time = time.time()
            self._last_recorded_time = time.time()
            return

        """Log the examples_per_sec metric every_n_steps."""
        self._epoch_steps += 1
        current_time = time.time()

        if self._epoch_steps % self._every_n_steps == 0:
            average_examples_per_sec = self.get_avg_examples_per_sec()
            current_examples_per_sec = self._batch_size * (
                self._every_n_steps / (current_time - self._last_recorded_time)
            )

            if self.log_as_print:
                logging.info(
                    f"[Examples/sec - Epoch step: {self._epoch_steps}] "
                    f"current: {current_examples_per_sec:.2f}, avg: {average_examples_per_sec:.2f}"
                )

            self.wandb_logger.log(
                {
                    "current_examples_per_sec": current_examples_per_sec,
                    "average_examples_per_sec": average_examples_per_sec,
                }
            )

            self._last_recorded_time = current_time  # Update last_recorded_time


def get_callbacks(train_batch_size=16, metrics_log_frequency=10, wandb_logger=None):
    callbacks = [
        ExamplesPerSecondCallback(
            train_batch_size, every_n_steps=metrics_log_frequency, wandb_logger=wandb_logger
        )
    ]

    if wandb_logger:
        wandb_callback = wandb_logger.get_callback(metrics_log_frequency=metrics_log_frequency)
        if wandb_callback:
            callbacks.append(wandb_callback)

    return callbacks


class WandbLogger:
    def __init__(self, enabled, wandb_project="", config={}):
        self.enabled = enabled
        if self.enabled:
            wandb.init(project=wandb_project, config=config)

    def config(self, config={}):
        if self.enabled:
            wandb.config.update(config)

    def log(self, metrics):
        if self.enabled:
            wandb.log(metrics)

    def get_callback(self, metrics_log_frequency, save_model=False, save_graph=False):
        callback = None
        if self.enabled:
            callback = wandb.keras.WandbCallback(
                log_batch_frequency=metrics_log_frequency,
                save_model=save_model,
                save_graph=save_graph,
            )
        return callback

    def teardown(self, exit_code=0):
        wandb.finish(exit_code=exit_code)


@dataclass
class RetrievalTrainEvalRunner:
    wandb_logger: Any = None
    model_type: str = None
    schema: Any = None
    train_ds: Any = None
    eval_ds: Any = None
    model: Any = None
    optimizer: Any = None
    metrics: Any = None
    loss: Any = None
    callbacks: Any = None
    # hparams for actions
    random_seed: int = 42
    train_epochs: int = 1
    train_steps_per_epoch: Optional[int] = None
    train_batch_size: int = 128
    train_metrics_steps: int = 100
    eval_steps: int = 100
    eval_batch_size: int = 128

    def run(self, hparams):
        start_time = time.time()

        set_random_seed(self.random_seed)

        self.wandb_logger.config(hparams)

        # Marks W&B execution as failed by default
        exit_code = 1
        try:
            self.model.compile(
                run_eagerly=False, optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
            )

            self.model.fit(
                self.train_ds,
                epochs=self.train_epochs,
                steps_per_epoch=self.train_steps_per_epoch,
                batch_size=self.train_batch_size,
                shuffle=True,
                drop_last=True,
                # validation_data=eval_ds,
                # validation_steps=args.validation_steps,
                callbacks=self.callbacks,
                train_metrics_steps=self.train_metrics_steps,
            )

            eval_kwargs = {}
            if self.model_type != "youtubednn":
                eval_kwargs["item_corpus"] = self.train_ds

            # Evaluate on train set
            train_metrics = self.model.evaluate(
                self.train_ds,
                steps=self.eval_steps,
                batch_size=self.eval_batch_size,
                return_dict=True,
                callbacks=self.callbacks,
                **eval_kwargs,
            )
            train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}

            # Evaluate on valid set
            eval_metrics = self.model.evaluate(
                self.eval_ds,
                steps=self.eval_steps,
                batch_size=self.eval_batch_size,
                return_dict=True,
                callbacks=self.callbacks,
                **eval_kwargs,
            )

            final_metrics = {**train_metrics, **eval_metrics}

            final_time = time.time()
            final_metrics["runtime_sec"] = final_time - start_time

            examples_per_sec_callback = list(
                [x for x in self.callbacks if isinstance(x, ExamplesPerSecondCallback)]
            )[0]
            avg_examples_per_sec = (
                examples_per_sec_callback.get_train_batches_mean_of_avg_examples_per_sec()
            )
            final_metrics["avg_examples_per_sec"] = avg_examples_per_sec

            final_metrics = {f"{k}-final": v for k, v in final_metrics.items()}
            self.wandb_logger.log(final_metrics)

            # Marks W&B execution as successfully finished
            exit_code = 0
        finally:
            self.wandb_logger.teardown(exit_code=exit_code)

        return final_metrics


@dataclass
class RetrievalTrainEvalRunnerV2:
    wandb_logger: Any = None
    model_type: str = None
    schema: Any = None
    train_ds: Any = None
    eval_ds: Any = None
    model: Any = None
    optimizer: Any = None
    metrics: Any = None
    loss: Any = None
    callbacks: Any = None
    # hparams for actions
    random_seed: int = 42
    train_epochs: int = 1
    train_steps_per_epoch: Optional[int] = None
    train_batch_size: int = 128
    train_metrics_steps: int = 100
    eval_steps: int = 100
    eval_batch_size: int = 128

    def run(self, hparams):
        start_time = time.time()

        set_random_seed(self.random_seed)

        self.wandb_logger.config(hparams)

        # Marks W&B execution as failed by default
        exit_code = 1
        try:
            self.model.compile(
                run_eagerly=False, optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
            )

            self.model.fit(
                self.train_ds,
                epochs=self.train_epochs,
                steps_per_epoch=self.train_steps_per_epoch,
                batch_size=self.train_batch_size,
                shuffle=True,
                drop_last=True,
                # validation_data=eval_ds,
                # validation_steps=args.validation_steps,
                callbacks=self.callbacks,
                train_metrics_steps=self.train_metrics_steps,
            )

            # get encoder for top-k evaluation
            max_cutoff = self.metrics[0].k
            item_features = self.schema.select_by_tag(Tags.ITEM).column_names
            item_dataset = self.train_ds.to_ddf()[item_features].drop_duplicates().compute()
            item_dataset = Dataset(item_dataset)
            item_dataset = unique_rows_by_features(item_dataset, Tags.ITEM, Tags.ITEM_ID)
            recommender = self.model.to_top_k_encoder(
                item_dataset, batch_size=self.train_batch_size, k=max_cutoff
            )
            recommender.compile(run_eagerly=False, metrics=self.metrics)
            item_id_name = self.schema.select_by_tag(Tags.ITEM_ID).column_names[0]

            # Evaluate on train set
            train_loader = mm.Loader(
                self.train_ds,
                batch_size=self.eval_batch_size,
                shuffle=False,
            ).map(mm.ToTarget(self.train_ds.schema, item_id_name))

            train_metrics = recommender.evaluate(
                train_loader,
                steps=self.eval_steps,
                return_dict=True,
                callbacks=self.callbacks,
            )
            train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}

            # Evaluate on valid set
            eval_loader = mm.Loader(
                self.eval_ds,
                batch_size=self.eval_batch_size,
                shuffle=False,
            ).map(mm.ToTarget(self.eval_ds.schema, item_id_name))
            eval_metrics = recommender.evaluate(
                eval_loader,
                batch_size=self.eval_batch_size,
                return_dict=True,
                callbacks=self.callbacks,
            )

            final_metrics = {**train_metrics, **eval_metrics}

            final_time = time.time()
            final_metrics["runtime_sec"] = final_time - start_time

            examples_per_sec_callback = list(
                [x for x in self.callbacks if isinstance(x, ExamplesPerSecondCallback)]
            )[0]
            avg_examples_per_sec = (
                examples_per_sec_callback.get_train_batches_mean_of_avg_examples_per_sec()
            )
            final_metrics["avg_examples_per_sec"] = avg_examples_per_sec

            final_metrics = {f"{k}-final": v for k, v in final_metrics.items()}
            self.wandb_logger.log(final_metrics)

            # Marks W&B execution as successfully finished
            exit_code = 0
        finally:
            self.wandb_logger.teardown(exit_code=exit_code)

        return final_metrics
