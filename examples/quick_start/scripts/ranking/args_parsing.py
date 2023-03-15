import argparse
from enum import Enum


class Task(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"


class MtlArgsPrefix(Enum):
    POS_CLASS_WEIGHT_ARG_PREFIX = "mtl_pos_class_weight_"
    LOSS_WEIGHT_ARG_PREFIX = "mtl_loss_weight_"


INT_LIST_ARGS = ["mlp_layers", "expert_mlp_layers", "tower_layers"]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def parse_dynamic_args(dyn_args):
    dyn_args_dict = dict([arg.replace("--", "").split("=") for arg in dyn_args])
    return dyn_args_dict


def parse_int_list_arg(value):
    # Used because autobench can't provide empty string ("") as argument
    if value == "None":
        value = ""

    alist = list([int(v.strip()) for v in value.split(",") if v != ""])
    return alist


def parse_arguments():
    parser = build_arg_parser()
    args, dynamic_args = parser.parse_known_args()
    dynamic_args = parse_dynamic_args(dynamic_args)

    unknown_args = list(
        [
            arg
            for arg in dynamic_args
            if not any([arg.startswith(prefix.value) for prefix in MtlArgsPrefix])
        ]
    )
    if len(unknown_args) > 0:
        raise ValueError(f"Unrecognized arguments: {unknown_args}")

    new_args = AttrDict({**args.__dict__, **dynamic_args})

    # Parsing str args that contains lists of ints
    for a in INT_LIST_ARGS:
        new_args[a] = parse_int_list_arg(new_args[a])

    # logging.info(f"ARGUMENTS: {new_args}")

    return new_args


def build_arg_parser():
    parser = argparse.ArgumentParser(description="MTL & STL models")

    # Inputs
    parser.add_argument("--train_path", default="/data/train/", help="")
    parser.add_argument("--eval_path", default="/data/eval/", help="")
    # Outputs
    parser.add_argument("--output_path", default="/results/", help="")
    parser.add_argument("--save_trained_model_path", default=None, help="")

    # Tasks
    parser.add_argument(
        "--tasks", help="",
    )
    parser.add_argument(
        "--tasks_sample_space", default="", help="",
    )
    parser.add_argument(
        "--ignore_tasks", default="", help="",
    )

    # Model
    parser.add_argument(
        "--model",
        default="mlp",
        choices=["mmoe", "cgc", "ple", "dcn", "dlrm", "mlp", "wide_n_deep", "deepfm",],
        help="",
    )

    parser.add_argument(
        "--activation", default="relu", choices=["tanh", "selu", "relu", "elu", "swish"], help="",
    )

    parser.add_argument("--mlp_init", type=str, default="glorot_uniform", help="")
    parser.add_argument("--l2_reg", default=1e-5, type=float, help="")
    parser.add_argument("--embeddings_l2_reg", default=0.0, type=float, help="")

    # Embeddings args
    parser.add_argument("--embedding_sizes_multiplier", default=2.0, type=float, help="")

    # MLPs args
    parser.add_argument("--dropout", default=0.00, type=float, help="")

    # hyperparams for STL models
    parser.add_argument("--mlp_layers", default="128,64,32", type=str, help="")
    parser.add_argument("--stl_positive_class_weight", default=1.0, type=float, help="")

    # DCN
    parser.add_argument("--dcn_interacted_layer_num", default=1, type=int, help="")

    # DLRM & DeepFM
    parser.add_argument("--embeddings_dim", default=128, type=int, help="")

    # Wide&Deep
    parser.add_argument("--wnd_hashed_cross_num_bins", default=10000, type=int, help="")
    parser.add_argument("--wnd_wide_l2_reg", default=1e-5, type=float, help="")

    # hyperparams for experts
    parser.add_argument("--expert_mlp_layers", default="64", type=str, help="")
    parser.add_argument("--expert_dcn_interacted_layer_num", default=1, type=int, help="")

    # MMOE
    parser.add_argument("--mmoe_num_mlp_experts", default=4, type=int, help="")
    parser.add_argument("--mmoe_num_dcn_experts", default=0, type=int, help="")
    parser.add_argument("--mmoe_num_dlrm_experts", default=0, type=int, help="")

    # CGC and PLE
    parser.add_argument("--cgc_num_task_mlp_experts", default=1, type=int, help="")
    parser.add_argument("--cgc_num_shared_mlp_experts", default=2, type=int, help="")
    parser.add_argument("--cgc_num_task_dcn_experts", default=0, type=int, help="")
    parser.add_argument("--cgc_num_shared_dcn_experts", default=0, type=int, help="")
    parser.add_argument("--ple_num_layers", default=1, type=int, help="")

    # hyperparams for multi-task (MMOE, CGC, PLE)
    parser.add_argument("--gate_dim", default=64, type=int, help="")

    parser.add_argument("--mtl_gates_softmax_temperature", default=1.0, type=float, help="")

    parser.add_argument(
        "--use_task_towers", type=str2bool, nargs="?", const=True, default=False, help="",
    )

    parser.add_argument("--tower_layers", default="64", type=str, help="")

    # hyperparams for training
    parser.add_argument("--lr", default=1e-4, type=float, help="")
    parser.add_argument("--lr_decay_rate", default=0.98, type=float, help="")
    parser.add_argument("--lr_decay_steps", default=100, type=int, help="")
    parser.add_argument("--train_batch_size", default=1024, type=int, help="")
    parser.add_argument("--eval_batch_size", default=1024, type=int, help="")
    parser.add_argument("--epochs", default=1, type=int, help="")
    parser.add_argument("--optimizer", default="adam", choices=["adagrad", "adam"], help="")

    parser.add_argument("--train_metrics_steps", default=50, type=int, help="")
    parser.add_argument("--metrics_log_frequency", default=50, type=int, help="")
    parser.add_argument("--validation_steps", default=0, type=int, help="")

    parser.add_argument("--random_seed", default=42, type=int, help="")
    parser.add_argument("--train_steps_per_epoch", type=int, help="")

    # In-batch negatives
    parser.add_argument("--in_batch_negatives_train", default=0, type=int, help="")
    parser.add_argument("--in_batch_negatives_eval", default=0, type=int, help="")

    # Logging
    parser.add_argument(
        "--log_to_tensorboard", type=str2bool, nargs="?", const=True, default=False, help="",
    )

    parser.add_argument(
        "--log_to_wandb", type=str2bool, nargs="?", const=True, default=True, help="",
    )

    parser.add_argument("--wandb_project", default="mm_quick_start", help="")
    parser.add_argument("--wandb_entity", default="merlin-research", help="")
    parser.add_argument("--wandb_exp_group", default="", help="")

    return parser
