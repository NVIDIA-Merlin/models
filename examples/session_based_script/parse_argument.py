import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_int_list_arg(value):
    # Used because autobench can't provide empty string ("") as argument
    if value == "None":
        value = ""

    alist = list([int(v.strip()) for v in value.split(",") if v != ""])
    return alist


def parse_arguments():
    parser = build_arg_parser()
    args = parser.parse_args()
    return args


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Session-based transformer models")

    # Inputs
    parser.add_argument("--train_path", default="/data/train/", help="")
    parser.add_argument("--eval_path", default="/data/eval/", help="")
    parser.add_argument("--schema_path", default="/data/eval/", help="")
    parser.add_argument("--side_information_features", default="", type=str, help="")  # TODO
    # Outputs
    parser.add_argument("--output_path", default="/results/", help="")
    parser.add_argument("--save_trained_model_path", default=None, help="")

    # Task
    parser.add_argument(
        "--task",
        default="multi_class_classification",
        choices=["multi_class_classification", "contrastive_classification"],
        help="",
    )

    # InputBlock args
    # Embeddings args
    parser.add_argument("--embeddings_l2_reg", default=0.0, type=float, help="")
    parser.add_argument("--embedding_sizes_multiplier", default=2.0, type=float, help="")
    parser.add_argument(
        "--emb_init_distribution",
        default="uniform",
        choices=["normal", "truncated_normal", "uniform"],
        help="",
    )
    parser.add_argument(
        "--emb_init_std",
        default=0.05,
        type=float,
        help="Standard deviation of the random values to generate.",
    )
    parser.add_argument("--embedding_dim", default=None, type=int, help="")

    # ## features args
    parser.add_argument(
        "--feature_normalization",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enables layer norm for each feature individually, before their aggregation.",
    )
    parser.add_argument(
        "--input_dropout",
        default=0.00,
        type=float,
        help="The dropout probability of the input embeddings, before being aggregated",
    )
    parser.add_argument(
        "--aggregation",
        default="concat",
        choices=["concat", "element-wise-sum-item-multi"],
        help="",
    )

    # MLPs args
    parser.add_argument("--mlp_init", type=str, default="glorot_uniform", help="")
    parser.add_argument("--l2_reg", default=1e-5, type=float, help="")
    parser.add_argument("--dropout", default=0.00, type=float, help="")

    # hyperparams for the transformer block
    parser.add_argument(
        "--model_type",
        default="xlnet",
        choices=["xlnet", "bert", "albert", "roberta", "gpt", "lstm"],
        help="",
    )
    parser.add_argument(
        "--xlnet_attn_type",
        default="bi",
        choices=["uni", "bi"],
        help="",
    )
    parser.add_argument(
        "--d_model",
        default=256,
        type=int,
        help="size of hidden states (or internal states) for Transformers",
    )
    parser.add_argument("--n_layer", default=12, type=int, help="number of layers for Transformers")
    parser.add_argument(
        "--n_head", default=4, type=int, help="number of attention heads for Transformers"
    )
    parser.add_argument(
        "--transformer_layer_norm_eps",
        default=1e-12,
        type=float,
        help="The epsilon used by the layer normalization layers for Transformers",
    )
    parser.add_argument(
        "--transformer_initializer_range",
        default=0.02,
        type=float,
        help="The standard deviation of the truncated_normal_initializer for "
        "initializing all weight matrices for Transformers",
    )
    parser.add_argument(
        "--transformer_activation",
        default="gelu",
        choices=["gelu", "selu", "relu", "swish"],
        help="The non-linear activation function (function or string) in Transformers. ",
    )
    parser.add_argument(
        "--transformer_dropout",
        default=0.0,
        type=float,
        help="The dropout probability for all fully connected layers in the embeddings, "
        "encoder, and decoders for Transformers",
    )
    parser.add_argument(
        "--summary_type",
        default="last",
        choices=["first", "last", "mean"],
        help="How to summarize the vector representation of the sequence",
    )
    # ## args for ALBERT
    parser.add_argument(
        "--num_hidden_groups",
        default=4,
        type=int,
        help="(ALBERT) Number of groups for the hidden layers, parameters in the same"
        "group are shared.",
    )
    parser.add_argument(
        "--inner_group_num",
        default=1,
        type=int,
        help="(ALBERT) Number of groups for the hidden layers, parameters in the same"
        "group are shared.",
    )

    # hyperparams for the output block
    parser.add_argument(
        "--weight_tying",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Evaluate metrics only on predictions for the last item of the sequence "
        "(rather then evaluation for all next-item predictions).",
    )
    parser.add_argument(
        "--logits_temperature",
        default=1.0,
        type=float,
        help="used to reduce model overconfidence, so that "
        "softmax(logits / T). Value 1.0 reduces to regular softmax.",
    )
    parser.add_argument(
        "--label_smoothing",
        default=0.0,
        type=float,
        help="Applies label smoothing using as alpha this parameter value. "
        "It helps overconfidence of models and calibration of the predictions.",
    )
    # ## Sampled softmwax negatives
    parser.add_argument(
        "--num_negatives",
        default=100,
        type=int,
        help="Number of negative samples to use for sampled softmax.",
    )
    parser.add_argument(
        "--min_sampled_id",
        default=1,
        type=int,
        help="The minimum id value to be sampled with sampled softmax. "
        "Useful to ignore the first categorical encoded ids,"
        " which are usually reserved for <nulls>, OOV",
    )
    # TODO add log-q correction

    # Training / Eval tasks args
    parser.add_argument(
        "--masking_probability",
        default=0.2,
        type=float,
        help="Ratio of tokens to mask (set as target) from an original sequence.",
    )
    parser.add_argument(
        "--training_task",
        default="causal",
        type=str,
        choices=["causal", "masked", "last", "random"],
        help="Training approach ",
    )

    parser.add_argument(
        "--evaluation_task",
        default="last",
        type=str,
        choices=["all", "last", "random"],
        help="Evaluation approach ",
    )

    # hyperparams for training
    # ## optimizer
    parser.add_argument("--lr", default=1e-4, type=float, help="")
    parser.add_argument("--lr_decay_rate", default=None, type=float, help="")
    parser.add_argument("--lr_decay_steps", default=None, type=int, help="")
    parser.add_argument(
        "--learning_rate_schedule",
        default=None,
        type=str,
        choices=[
            "constant_with_warmup",
            "linear_with_warmup",
            "cosine_with_warmup",
        ],
        help="Learning Rate schedule .",
    )
    parser.add_argument("--weight_decay", type=float, help="")
    parser.add_argument(
        "--optimizer", default="adam", choices=["adagrad", "adam", "adamw"], help=""
    )
    # ## loader
    parser.add_argument("--train_batch_size", default=1024, type=int, help="")
    parser.add_argument("--eval_batch_size", default=1024, type=int, help="")
    parser.add_argument("--epochs", default=1, type=int, help="")
    parser.add_argument("--train_metrics_steps", default=50, type=int, help="")
    parser.add_argument("--validation_steps", default=0, type=int, help="")
    parser.add_argument("--train_steps_per_epoch", type=int, help="")
    # ## evaluation
    parser.add_argument(
        "--top_ks",
        default="10,20",
        type=str,
        help="The list of top-k thresholds to use for ranking metrics calculation",
    )

    parser.add_argument("--random_seed", default=42, type=int, help="")

    # Logging
    parser.add_argument("--metrics_log_frequency", default=50, type=int, help="")
    parser.add_argument(
        "--log_to_tensorboard",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="",
    )

    parser.add_argument(
        "--log_to_wandb",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="",
    )

    parser.add_argument("--wandb_project", default="mm-session-based-api", help="")
    parser.add_argument("--wandb_entity", default="merlin-research", help="")
    parser.add_argument("--wandb_exp_group", default="", help="")
    parser.add_argument(
        "--save_topk_predictions",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="",
    )

    return parser
