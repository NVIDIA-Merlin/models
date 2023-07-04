import logging
import math
import os

import cupy
import tensorflow as tf
from parse_argument import parse_arguments
from run_logging import WandbLogger, get_callbacks

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.models.tf.core.tabular import TabularBlock
from merlin.models.tf.transforms.sequence import (
    SequenceMaskLast,
    SequenceMaskRandom,
    SequencePredictLast,
    SequencePredictNext,
    SequencePredictRandom,
)
from merlin.models.utils.schema_utils import categorical_cardinalities
from merlin.schema.io.tensorflow_metadata import TensorflowMetadata
from merlin.schema.tags import Tags
from merlin.models.utils import schema_utils
from merlin.models.tf.transforms.bias import PopularityLogitsCorrection

# set logger
info_logger = logging.getLogger(__name__)


# Create equivalent class of T4Rec's TabularDroupout
class TabularDropout(TabularBlock):
    """
    Applies dropout transformation.
    """

    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def forward(self, inputs, training=False):
        outputs = {key: self.dropout(val, training=training) for key, val in inputs.items()}
        return outputs


# Create equivalent class of T4Rec's 'layer-norm'
class TabularNorm(TabularBlock):
    """
    Applies layr-norm transformation.
    """

    def __init__(self):
        super().__init__()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def forward(self, inputs):
        outputs = {key: self.layer_norm(val) for key, val in inputs.items()}
        return outputs


def get_embeddings_initilizer(args):
    if args.emb_init_distribution == "normal":
        initilizer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=args.emb_init_std)
    elif args.emb_init_distribution == "truncated_normal":
        initilizer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=args.emb_init_std)
    else:
        initilizer = "uniform"

    return initilizer


def get_input_block(schema, args):
    post = None
    if args.input_dropout > 0:
        post = TabularDropout(args.input_dropout)

    if args.feature_normalization:
        if post is None:
            post = TabularNorm()
        else:
            post = post.connect(TabularNorm())

    input_block = mm.InputBlockV2(
        schema,
        categorical=mm.Embeddings(
            schema.select_by_tag(Tags.CATEGORICAL),
            dim=args.embedding_dim,
            embeddings_initializer=get_embeddings_initilizer(args),
            sequence_combiner=None,
        ),
        post=post,
    )
    return input_block


def get_output_block(schema, args, input_block=None):
    candidate = schema.select_by_tag(Tags.ITEM_ID)
    if not candidate:
        raise ValueError(f"The schema should contain a feature tagged as `{Tags.ITEM_ID}`")

    cardinalities = categorical_cardinalities(candidate)
    candidate = candidate.first
    num_classes = cardinalities[candidate.name]

    if args.weight_tying:
        # TODO add check for input_block that contains the target feature
        candidate_table = input_block["categorical"][candidate.properties["domain"]["name"]]
        to_call = candidate_table
    else:
        to_call = candidate

    if args.sampled_softmax:
        outputs = mm.ContrastiveOutput(
            to_call=to_call,
            logits_temperature=args.logits_temperature,
            negative_samplers=mm.PopularityBasedSamplerV2(
                max_num_samples=args.num_negatives,
                max_id=num_classes - 1,
                min_id=args.min_sampled_id,
            ),
            logq_sampling_correction=args.logq_correction,
        )
    else:
        outputs = mm.CategoricalOutput(
            to_call=to_call,
            logits_temperature=args.logits_temperature,
        )
    return outputs


def get_sequential_block(args):
    kwargs = {}
    if args.model_type == "lstm":
        return tf.keras.layers.LSTM(args.d_model, return_sequences=False)
    if args.model_type == "gpt2":
        block = mm.GPT2Block
    if args.model_type == "bert":
        block = mm.BertBlock
    if args.model_type == "xlnet":
        block = mm.XLNetBlock
        kwargs = {
            "attn_type": args.xlnet_attn_type,
        }
    if args.model_type == "albert":
        block = mm.AlbertBlock
        num_hidden_groups = args.num_hidden_groups
        if num_hidden_groups == -1:
            num_hidden_groups = args.n_layer
        kwargs = {
            "num_hidden_groups": num_hidden_groups,
            "inner_group_num": args.inner_group_num,
        }

    return block(
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        hidden_act=args.transformer_activation,
        initializer_range=args.transformer_initializer_range,
        layer_norm_eps=args.transformer_layer_norm_eps,
        dropout=args.transformer_dropout,
        **kwargs,
    )


def get_sequence_transforms(schema, args, transformer=None):
    """Set the sequential task for training and evaluation"""
    pre_fit, pre_eval = None, None
    target = schema.select_by_tag(Tags.ITEM_ID).first.name
    sequence_schema = schema.select_by_tag(Tags.SEQUENCE)
    if args.training_task == "masked":
        pre_fit = SequenceMaskRandom(
            schema=sequence_schema,
            target=target,
            masking_prob=args.masking_probability,
            transformer=transformer,
        )
        if args.evaluation_task == "last":
            pre_eval = SequenceMaskLast(sequence_schema, target=target, transformer=transformer)
        elif args.evaluation_task == "random":
            pre_eval = SequenceMaskRandom(
                sequence_schema,
                target=target,
                masking_prob=args.masking_probability,
                transformer=transformer,
            )
        else:
            raise ValueError(
                f"{args.evaluation_task} not supported for masked training"
            )  # TODO define better error message

    if args.training_task == "causal":
        pre_fit = SequencePredictNext(sequence_schema, target=target, transformer=transformer)
        if args.evaluation_task == "last":
            pre_eval = SequencePredictLast(sequence_schema, target=target, transformer=transformer)
        elif args.evaluation_task == "random":
            pre_eval = SequencePredictRandom(
                sequence_schema, target=target, transformer=transformer
            )
        elif args.evaluation_task == "all":
            pre_eval = SequencePredictNext(sequence_schema, target=target, transformer=transformer)
        else:
            raise ValueError(
                f"{args.evaluation_task} not supported for causal training"
            )  # TODO define better error message

    if args.training_task == "last":
        pre_fit = SequencePredictLast(sequence_schema, target=target)
        if args.evaluation_task == "last":
            pre_eval = SequencePredictLast(sequence_schema, target=target)
        elif args.evaluation_task == "random":
            pre_eval = SequencePredictRandom(sequence_schema, target=target)
        else:
            raise ValueError(
                f"{args.evaluation_task} not supported for 'last' training"
            )  # TODO define better error message

    if args.training_task == "random":
        pre_fit = SequencePredictRandom(sequence_schema, target=target)
        if args.evaluation_task == "last":
            pre_eval = SequencePredictLast(sequence_schema, target=target)
        elif args.evaluation_task == "random":
            pre_eval = SequencePredictRandom(sequence_schema, target=target)
        else:
            raise ValueError(
                f"{args.evaluation_task} not supported for 'random' training"
            )  # TODO define better error message

    return pre_fit, pre_eval


def get_metrics(args):
    return mm.TopKMetricsAggregator.default_metrics(top_ks=[int(k) for k in args.top_ks.split(",")])


def get_datasets(args):
    train_ds = Dataset(os.path.join(args.train_path, "*.parquet"), part_size="500MB")
    eval_ds = Dataset(os.path.join(args.eval_path, "*.parquet"), part_size="500MB")

    return train_ds, eval_ds


def log_final_metrics(logger, metrics_results):
    if logger:
        metrics_results = {f"{k}-final": v for k, v in metrics_results.items()}
        logger.log(metrics_results)


def get_optimizer(args, total_steps=None):
    from transformers.optimization_tf import AdamWeightDecay

    learning_rate = args.lr
    if args.lr_decay_rate:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            args.lr,
            decay_steps=args.lr_decay_steps,
            decay_rate=args.lr_decay_rate,
            staircase=True,
        )

    if args.optimizer == "adam":
        opt = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        )
    elif args.optimizer == "adagrad":
        opt = tf.keras.optimizers.Adagrad(
            learning_rate=learning_rate,
        )
    elif args.optimizer == "adamw":
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_steps,
            power=1,
        )

        opt = AdamWeightDecay(
            learning_rate=lr_schedule,
            weight_decay_rate=args.weight_decay,
        )
    else:
        raise ValueError("Invalid optimizer")

    return opt


def main(args):
    # load train / eval data
    train_ds, eval_ds = get_datasets(args)

    if args.log_to_wandb:
        logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, config=args)
        logger.setup()

    # load schema, if specified
    if args.schema_path:
        schema = TensorflowMetadata.from_proto_text_file(
            args.schema_path, file_name="schema.pbtxt"
        ).to_merlin_schema()
    else: 
        schema = train_ds.schema

    if args.side_information_features == "":
        schema_model = schema.select_by_tag([Tags.ITEM_ID])
    else:
        # TODO add filter step
        schema_model = schema

    train_ds.schema = schema_model
    eval_ds.schema = schema_model

    # get input block
    input_block = get_input_block(train_ds.schema, args)

    # get transformer block
    transformer_block = get_sequential_block(args)

    # get output block
    output_block = get_output_block(train_ds.schema, args, input_block=input_block)

    # Define the session encoder
    if args.weight_tying:
        # project tranformer's output to same dimension as target
        projection = mm.MLPBlock(
            [output_block.to_call.table.dim],
            no_activation_last_layer=True,
        )
        session_encoder = mm.Encoder(
            input_block,
            mm.MLPBlock([args.d_model], no_activation_last_layer=True),
            transformer_block,
            projection,
        )

    else:
        session_encoder = mm.Encoder(
            input_block,
            mm.MLPBlock([args.d_model], no_activation_last_layer=True),
            transformer_block,
        )

    model = mm.RetrievalModelV2(query=session_encoder, output=output_block)

    # get optimizer
    steps_per_epoch = math.floor(train_ds.compute().shape[0] / args.train_batch_size)
    total_steps = steps_per_epoch * args.epochs
    optimizer = get_optimizer(args, total_steps=total_steps)

    # get loss
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=args.label_smoothing
    )

    # get metrics
    metrics = get_metrics(args)

    # compile the model
    model.compile(optimizer, run_eagerly=False, metrics=metrics, loss=loss)

    # get callbacks
    callbacks = get_callbacks(args)

    # get sequence transforms
    pre_fit, pre_eval = get_sequence_transforms(schema_model, args, transformer=transformer_block)

    # start training
    info_logger.info("Starting to train the model")
    model.fit(
        train_ds,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        steps_per_epoch=args.train_steps_per_epoch,
        callbacks=callbacks,
        train_metrics_steps=args.train_metrics_steps,
        pre=pre_fit,
    )

    info_logger.info("Starting to evlaluate the model on train data")
    train_metrics = model.evaluate(
        train_ds,
        batch_size=args.eval_batch_size,
        return_dict=True,
        callbacks=callbacks,
        pre=pre_eval,
    )
    train_metrics = {"train_" + k: v for k, v in train_metrics.items()}

    # start evaluation
    info_logger.info("Starting to evlaluate the model on eval data")
    eval_metrics = model.evaluate(
        eval_ds,
        batch_size=args.eval_batch_size,
        return_dict=True,
        callbacks=callbacks,
        pre=pre_eval,
    )

    # count number of parameters:
    eval_metrics["total_parameters"] = model.count_params()

    info_logger.info(f"EVALUATION METRICS: {eval_metrics}")

    if args.save_topk_predictions:
        target = schema_model.select_by_tag(Tags.ITEM_ID).first
        sequence_schema = schema_model.select_by_tag(Tags.LIST)
        max_k = max([int(k) for k in args.top_ks.split(",")])
        topk_model = model.to_top_k_encoder(k=max_k)
        topk_model.compile(run_eagerly=False, metrics=metrics)
        # Check the evaluation scores
        loader = mm.Loader(eval_ds, batch_size=args.eval_batch_size)
        metrics = topk_model.evaluate(loader, return_dict=True, pre=pre_eval)
        for k in args.top_ks.split(","):
            eval_metrics[f"top-{k}_recall"] = metrics[f"recall_at_{k}"]
            eval_metrics[f"top-{k}_ndcg"] = metrics[f"ndcg_at_{k}"]

        # Get topk predictions
        # Extract last item by applying the SequencePredictLast transform to dataloader
        loader = mm.Loader(eval_ds, batch_size=args.eval_batch_size, shuffle=False).map(
            mm.SequencePredictLast(sequence_schema, target)
        )
        predictions = topk_model.predict(loader)

        data = eval_ds.to_ddf().compute().to_pandas()
        data["topk_indices"] = list(predictions.identifiers)
        data["topk_scores"] = list(predictions.scores)

        data.to_parquet(
            os.path.join(args.output_path, f"mm_top_{max_k}_predictions_task_{args.training_task}"),
            row_group_size=10000,
        )

    log_final_metrics(logger, eval_metrics)
    log_final_metrics(logger, train_metrics)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
