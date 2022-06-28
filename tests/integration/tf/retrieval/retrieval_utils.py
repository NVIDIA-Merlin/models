import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import cupy
import dllogger as DLLogger
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras import regularizers
from tensorflow.keras.utils import set_random_seed

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.models.tf.blocks.core.transformations import PopularityLogitsCorrection
from merlin.models.utils import schema_utils
from merlin.schema.io.tensorflow_metadata import TensorflowMetadata
from merlin.schema.tags import Tags


def get_schema(data_path="", filter_dataset_features=""):
    schema = TensorflowMetadata.from_proto_text_file(
        os.path.join(data_path, "train/")
    ).to_merlin_schema()
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

    item_frequencies = tf.convert_to_tensor(cupy.asnumpy(item_frequencies_df["freq"].values))

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
    model_type="two_tower",
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


def get_youtube_dnn_model(
    schema,
    max_seq_length,
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

    model = mm.YoutubeDNNRetrievalModel(
        schema=schema_selected,
        max_seq_length=max_seq_length,
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

    def __init__(self, batch_size, every_n_steps=1, log_as_print=True, log_to_wandb=False):
        self.log_as_print = log_as_print
        self.log_to_wandb = log_to_wandb
        self._batch_size = batch_size
        self._every_n_steps = every_n_steps
        super(ExamplesPerSecondCallback, self).__init__()

    def on_train_begin(self, logs=None):
        self._first_batch = True
        self._epoch_steps = 0
        # self._train_start_time = time.time()
        # self._last_recorded_time = time.time()

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
            average_examples_per_sec = self._batch_size * (
                self._epoch_steps / (current_time - self._epoch_start_time)
            )
            current_examples_per_sec = self._batch_size * (
                self._every_n_steps / (current_time - self._last_recorded_time)
            )

            if self.log_as_print:
                logging.info(
                    f"[Examples/sec - Epoch step: {self._epoch_steps}] "
                    f"current: {current_examples_per_sec:.2f}, avg: {average_examples_per_sec:.2f}"
                )

            if self.log_to_wandb:
                wandb.log(
                    {
                        "current_examples_per_sec": current_examples_per_sec,
                        "average_examples_per_sec": average_examples_per_sec,
                    }
                )

            self._last_recorded_time = current_time  # Update last_recorded_time


def get_callbacks(train_batch_size=16, metrics_log_frequency=1, log_to_wandb=False):
    callbacks = [ExamplesPerSecondCallback(train_batch_size, every_n_steps=metrics_log_frequency)]

    if log_to_wandb:

        wandb_callback = wandb.keras.WandbCallback(
            log_batch_frequency=metrics_log_frequency,
            save_model=False,
            save_graph=False,
        )
        callbacks.append(wandb_callback)

    return callbacks


def config_wandb(wandb_project, args):
    wandb.init(project=wandb_project, config=args)


def config_dllogger(output_path, args):
    from dllogger import JSONStreamBackend, StdOutBackend, Verbosity

    DLLOGGER_FILENAME = os.path.join(output_path, "log.json")
    DLLogger.init(
        backends=[
            StdOutBackend(verbosity=Verbosity.DEFAULT),
            JSONStreamBackend(
                Verbosity.VERBOSE,
                DLLOGGER_FILENAME,
            ),
        ]
    )
    DLLogger.log(step="PARAMETER", data=args)
    DLLogger.flush()


def config_loggers(output_path="./", log_to_wandb=False, wandb_project="", args={}):
    if log_to_wandb:
        config_wandb(wandb_project, args)
    config_dllogger(output_path, args)


def log_final_metrics(metrics_results, log_to_wandb=False):
    metrics_results = {f"{k}-final": v for k, v in metrics_results.items()}

    if log_to_wandb:
        wandb.log(metrics_results)
        wandb.finish()

    DLLogger.log(step=(), data=metrics_results)
    DLLogger.flush()


@dataclass
class RetrievalTrainEvalRunner:
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
    log_to_wandb: bool = False
    wandb_project: str = "merlin-ci"

    def run(self):
        set_random_seed(self.random_seed)

        # TODO: Find a way to get a dictionary with all hparams to log to W&B
        config_loggers(log_to_wandb=self.log_to_wandb, wandb_project=self.wandb_project, args={})

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

        log_final_metrics(
            metrics_results={**train_metrics, **eval_metrics}, log_to_wandb=self.log_to_wandb
        )
