import os
import shutil
from os import path

import nvtabular as nvt
from numba import config
from nvtabular import ops

# Get dataframe library - cudf or pandas
from nvtabular.dispatch import get_lib

df_lib = get_lib()


def movielens_convert_etl(local_filename):
    """this funct does the preliminary preprocessing on movielens dataset
    and converts the csv files to parquet files and saves to disk. Then,
    using NVTabular, it does feature engineering on the parquet files
    and saves the processed files to disk."""
    local_filename = os.path.abspath(local_filename)
    movies = df_lib.read_csv(os.path.join(local_filename, "ml-25m/movies.csv"))
    movies["genres"] = movies["genres"].str.split("|")
    movies.to_parquet(os.path.join(local_filename, "movies_converted.parquet"))
    ratings = df_lib.read_csv(os.path.join(local_filename, "ml-25m", "ratings.csv"))
    # shuffle the dataset
    ratings = ratings.sample(len(ratings), replace=False)
    # split the train_df as training and validation data sets.
    num_valid = int(len(ratings) * 0.2)
    train = ratings[:-num_valid]
    valid = ratings[-num_valid:]
    train.to_parquet(os.path.join(local_filename, "train.parquet"))
    valid.to_parquet(os.path.join(local_filename, "valid.parquet"))

    # Avoid Numba warnings
    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    # NVTabular pipeline
    movies = df_lib.read_parquet(os.path.join(local_filename, "movies_converted.parquet"))
    joined = ["userId", "movieId"] >> ops.JoinExternal(movies, on=["movieId"])
    cat_features = joined >> ops.Categorify()
    label = nvt.ColumnSelector(["rating"])

    # Columns to apply to
    cats = nvt.ColumnSelector(["movieId"])

    # Target Encode movieId column
    te_features = cats >> ops.TargetEncoding(label, kfold=5, p_smooth=20)
    te_features_norm = te_features >> ops.NormalizeMinMax()

    # count encode `userId`
    count_logop_feat = (
        ["userId"] >> ops.JoinGroupby(cont_cols=["rating"], stats=["count"]) >> ops.LogOp()
    )
    feats_item = cat_features["movieId"] >> ops.AddMetadata(tags=["item_id", "item"])
    feats_user = cat_features["userId"] >> ops.AddMetadata(tags=["user_id", "user"])
    feats_genres = cat_features["genres"] >> ops.AddMetadata(tags=["item"])

    feats_target = (
        nvt.ColumnSelector(["rating"])
        >> ops.LambdaOp(lambda col: (col > 3).astype("int32"))
        >> ops.AddMetadata(tags=["binary_classification", "target"])
        >> nvt.ops.Rename(name="rating_binary")
    )
    target_orig = (
        ["rating"]
        >> ops.LambdaOp(lambda col: col.astype("float32"))
        >> ops.AddMetadata(tags=["regression", "target"])
    )
    workflow = nvt.Workflow(
        feats_item
        + feats_user
        + feats_genres
        + te_features_norm
        + count_logop_feat
        + target_orig
        + feats_target
        + joined["title"]
    )
    train_dataset = nvt.Dataset([os.path.join(local_filename, "train.parquet")])
    valid_dataset = nvt.Dataset([os.path.join(local_filename, "valid.parquet")])
    if path.exists(os.path.join(local_filename, "train")):
        shutil.rmtree(os.path.join(local_filename, "train"))
    if path.exists(os.path.join(local_filename, "valid")):
        shutil.rmtree(os.path.join(local_filename, "valid"))

    workflow.fit(train_dataset)
    workflow.transform(train_dataset).to_parquet(
        output_path=os.path.join(local_filename, "train"),
        out_files_per_proc=1,
        shuffle=False,
    )
    workflow.transform(valid_dataset).to_parquet(
        output_path=os.path.join(local_filename, "valid"),
        out_files_per_proc=1,
        shuffle=False,
    )
    # Save the workflow
    workflow.save(os.path.join(local_filename, "workflow"))
