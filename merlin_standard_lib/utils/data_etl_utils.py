import os
import shutil
from os import path

import numpy as np
import nvtabular as nvt
import pandas as pd
from numba import config
from nvtabular import ops

# Get dataframe library - cudf or pandas
from nvtabular.dispatch import get_lib
from nvtabular.utils import download_file

df_lib = get_lib()


def movielens_download_etl(local_filename, name="ml-25m", outputdir=None):
    """this funct does the preliminary preprocessing on movielens dataset
    and converts the csv files to parquet files and saves to disk. Then,
    using NVTabular, it does feature engineering on the parquet files
    and saves the processed files to disk."""
    local_filename = os.path.abspath(local_filename)
    if outputdir is None:
        outputdir = local_filename
    if name == "ml-25m":
        print("downloading movielens 25M..")
        download_file(
            "http://files.grouplens.org/datasets/movielens/ml-25m.zip",
            os.path.join(local_filename, "ml-25m.zip"),
        )
        movies = df_lib.read_csv(os.path.join(local_filename, name, "movies.csv"))
        movies["genres"] = movies["genres"].str.split("|")
        movies.to_parquet(os.path.join(local_filename, name, "movies_converted.parquet"))
        ratings = df_lib.read_csv(os.path.join(local_filename, name, "ratings.csv"))
        # shuffle the dataset
        ratings = ratings.sample(len(ratings), replace=False)
        # split the train_df as training and validation data sets.
        num_valid = int(len(ratings) * 0.2)
        train = ratings[:-num_valid]
        valid = ratings[-num_valid:]
        train.to_parquet(os.path.join(local_filename, name, "train.parquet"))
        valid.to_parquet(os.path.join(local_filename, name, "valid.parquet"))

        # Avoid Numba warnings
        config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

        print("starting data preprocessing..")

        # NVTabular pipeline
        movies = df_lib.read_parquet(os.path.join(local_filename, name, "movies_converted.parquet"))
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

    elif name == "ml-100k":
        print("downloading movielens 100K..")
        download_file(
            "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
            os.path.join(local_filename, "ml-100k.zip"),
        )
        print("starting ETL..")
        ratings = pd.read_csv(
            os.path.join(local_filename, "ml-100k/u.data"),
            names=["userId", "movieId", "rating", "timestamp"],
            sep="\t",
        )
        user_features = pd.read_csv(
            os.path.join(local_filename, "ml-100k/u.user"),
            names=["userId", "age", "gender", "occupation", "zip_code"],
            sep="|",
        )
        user_features.to_parquet(os.path.join(local_filename, "ml-100k/user_features.parquet"))
        cols = [
            "movieId",
            "title",
            "release_date",
            "video_release_date",
            "imdb_URL",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Childrens",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film_Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]

        genres_ = [
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Childrens",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film_Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]

        movies = pd.read_csv(os.path.join(local_filename, "ml-100k/u.item"), names=cols, sep="|")
        for col in genres_:
            movies[col] = movies[col].replace(1, col)
            movies[col] = movies[col].replace(0, np.nan)
        s = movies[genres_]
        s.notnull()
        movies["genres"] = s.notnull().dot(s.columns + ",").str[:-1]
        movies_converted = movies[
            ["movieId", "title", "release_date", "video_release_date", "imdb_URL", "genres"]
        ]
        movies_converted.to_parquet(
            os.path.join(local_filename, "ml-100k/movies_converted.parquet")
        )
        train = pd.read_csv(
            os.path.join(local_filename, "ml-100k/ua.base"),
            names=["userId", "movieId", "rating", "timestamp"],
            sep="\t",
        )
        valid = pd.read_csv(
            os.path.join(local_filename, "ml-100k/ua.test"),
            names=["userId", "movieId", "rating", "timestamp"],
            sep="\t",
        )
        train = train.merge(user_features, on="userId", how="left")
        train = train.merge(movies_converted, on="movieId", how="left")
        valid = valid.merge(user_features, on="userId", how="left")
        valid = valid.merge(movies_converted, on="movieId", how="left")
        train.to_parquet(os.path.join(local_filename, "ml-100k/train.parquet"))
        valid.to_parquet(os.path.join(local_filename, "ml-100k/valid.parquet"))
        print("starting data preprocessing..")

        cat_features = [
            "userId",
            "movieId",
            "gender",
            "occupation",
            "zip_code",
            "imdb_URL",
            "genres",
        ] >> nvt.ops.Categorify()

        cont_names = ["age"]
        boundaries = {"age": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]}
        age_bucket = cont_names >> ops.Bucketize(boundaries) >> ops.AddMetadata(tags=["user"])

        label = nvt.ColumnSelector(["rating"])

        # Target Encode movieId column
        te_features = ["movieId"] >> ops.TargetEncoding(label, kfold=5, p_smooth=20)

        te_features_norm = te_features >> ops.NormalizeMinMax()

        # count encode `userId`
        count_logop_feat = (
            ["userId"] >> ops.JoinGroupby(cont_cols=["rating"], stats=["count"]) >> ops.LogOp()
        )

        feats_item = cat_features["movieId"] >> ops.AddMetadata(tags=["item_id", "item"])
        feats_user = cat_features["userId"] >> ops.AddMetadata(tags=["user_id", "user"])
        feats_genres = cat_features["genres", "imdb_URL"] >> ops.AddMetadata(tags=["item"])
        user_features = cat_features["gender", "zip_code"] >> ops.AddMetadata(tags=["user"])

        feats_target = (
            nvt.ColumnSelector(["rating"])
            >> ops.LambdaOp(lambda col: (col > 3).astype("int32"))
            >> ops.AddMetadata(tags=["binary_classification", "target"])
            >> nvt.ops.Rename(name="rating_binary")
        )
        target_orig = ["rating"] >> ops.AddMetadata(tags=["regression", "target"])

        workflow = nvt.Workflow(
            feats_item
            + feats_user
            + feats_genres
            + te_features_norm
            + count_logop_feat
            + user_features
            + target_orig
            + feats_target
            + age_bucket
            + ["title"]
        )

    else:
        raise ValueError("Unknown dataset name.")

    train_dataset = nvt.Dataset([os.path.join(local_filename, name, "train.parquet")])
    valid_dataset = nvt.Dataset([os.path.join(local_filename, name, "valid.parquet")])

    if path.exists(os.path.join(local_filename, name, "train")):
        shutil.rmtree(os.path.join(local_filename, name, "train"))
    if path.exists(os.path.join(local_filename, name, "valid")):
        shutil.rmtree(os.path.join(local_filename, name, "valid"))

    workflow.fit(train_dataset)
    workflow.transform(train_dataset).to_parquet(
        output_path=os.path.join(outputdir, name, "train"),
        out_files_per_proc=1,
        shuffle=False,
    )
    workflow.transform(valid_dataset).to_parquet(
        output_path=os.path.join(outputdir, name, "valid"),
        out_files_per_proc=1,
        shuffle=False,
    )
    # Save the workflow
    workflow.save(os.path.join(outputdir, name, "workflow"))
    print("saving the workflow..")
