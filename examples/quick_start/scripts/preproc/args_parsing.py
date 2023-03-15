import argparse


def build_arg_parser():
    parser = argparse.ArgumentParser(description="MTL & STL models")

    # Inputs / Outputs
    parser.add_argument("--input_data_path", default="/data/ranking/", help="")
    parser.add_argument("--input_data_format", default="csv", choices=["csv", "parquet"], help="")
    parser.add_argument("--csv_sep", default=",", help="")
    parser.add_argument("--csv_na_values", default=None, help="")

    parser.add_argument("--use_cols", default=None, help="")

    parser.add_argument("--output_path", default="/results/", help="")
    parser.add_argument("--output_num_partitions", default=10, type=int, help="")
    parser.add_argument(
        "--persist_intermediate_files",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="",
    )

    parser.add_argument("--categorical_features", default="", help="")
    parser.add_argument("--continuous_features", default="", help="")

    parser.add_argument("--user_features", default="", help="")
    parser.add_argument("--item_features", default="", help="")
    parser.add_argument("--binary_classif_targets", default="", help="")
    parser.add_argument("--regression_targets", default="", help="")

    parser.add_argument("--user_id_feature", default="", help="")
    parser.add_argument("--item_id_feature", default="", help="")
    parser.add_argument("--timestamp_feature", default="", help="")
    parser.add_argument("--session_id_feature", default="", help="")

    # parser.add_argument("--groupby_feature", default="", help="")

    parser.add_argument("--to_int32", default="", help="")
    parser.add_argument("--to_int16", default="", help="")
    parser.add_argument("--to_int8", default="", help="")
    parser.add_argument("--to_float32", default="", help="")

    parser.add_argument("--min_user_freq", default=None, type=int, help="")
    parser.add_argument("--max_user_freq", default=None, type=int, help="")
    parser.add_argument("--min_item_freq", default=None, type=int, help="")
    parser.add_argument("--max_item_freq", default=None, type=int, help="")
    parser.add_argument("--num_max_rounds_filtering", default=5, type=int, help="")

    parser.add_argument("--filter_query", default=None, type=str, help="")

    parser.add_argument(
        "--dataset_split_strategy",
        default=None,
        type=str,
        choices=["random", "random_by_user", "temporal"],
        help="",
    )

    parser.add_argument("--random_split_eval_perc", default=None, type=float, help="")
    parser.add_argument("--temporal_timestamp_split", default=None, type=int, help="")

    parser.add_argument("--visible_gpu_devices", default="0", type=str, help="")
    parser.add_argument("--gpu_device_spill_frac", default=0.7, type=float, help="")

    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_list_arg(v):
    if v is None or v == "":
        return []
    return v.split(",")


def parse_arguments():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Parsing list args
    args.categorical_features = parse_list_arg(args.categorical_features)
    args.continuous_features = parse_list_arg(args.continuous_features)

    args.binary_classif_targets = parse_list_arg(args.binary_classif_targets)
    args.regression_targets = parse_list_arg(args.regression_targets)

    args.user_features = parse_list_arg(args.user_features)
    args.item_features = parse_list_arg(args.item_features)
    args.to_int32 = parse_list_arg(args.to_int32)
    args.to_int16 = parse_list_arg(args.to_int16)
    args.to_int8 = parse_list_arg(args.to_int8)
    args.to_float32 = parse_list_arg(args.to_float32)

    if args.filter_query:
        args.filter_query = args.filter_query.replace('"', "")

    return args
