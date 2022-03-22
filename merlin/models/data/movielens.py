import logging
import os
import shutil
from os import path

import numpy as np
import nvtabular as nvt
import pandas as pd
from nvtabular import ops

import merlin.io

# Get dataframe library - cuDF or pandas
from merlin.core.dispatch import get_lib
from merlin.core.utils import download_file
from merlin.schema import Tags

df_lib = get_lib()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_movielens(path=None, variant="ml-25m", user_sessions=False, max_length=None):
    """Gets the movielens dataset for use with merlin-models

    This function will return a tuple of train/test merlin.io.Dataset objects for the
    movielens dataset. This will download the movielens dataset locally if needed,
    and run a ETL pipeline with NVTabular to make this dataset ready for use with
    merlin-models.

    Parameters
    ----------
    path : str
        The path to download the files locally to. If not set will default to
        the 'merlin-models-data` directory in your home folder
    variant : "ml-1m" "ml-25m" or "ml-100k"
        Which variant of the movielens dataset to use. Must be either "ml-1m", "ml-25m" or "ml-100k"
    user_sessions: Bool
        If enabled, group the movielens data into a set of daily user sessions.
        Currently, this option is only supported for "ml-1m" variant.
        Defaults to False

    Returns
    -------
    tuple
        A tuple consisting of a merlin.io.Dataset for the training dataset and validation dataset
    """
    if path is None:
        path = os.environ.get(
            "INPUT_DATA_DIR", os.path.expanduser("~/merlin-models-data/movielens/")
        )

    variant_path = os.path.join(path, variant)
    if not os.path.exists(variant_path):
        os.makedirs(variant_path)
        if not user_sessions:
            movielens_download_etl(path, variant)
        else:
            assert (
                variant == "ml-1m"
            ), "We are currently supporting only the 'ML-1m' variant for session data"
            movielens_session_download_etl(path, variant, max_length=max_length)

    train = merlin.io.Dataset(os.path.join(variant_path, "train"), engine="parquet")
    valid = merlin.io.Dataset(os.path.join(variant_path, "valid"), engine="parquet")
    return train, valid


def movielens_download_etl(local_filename, name="ml-25m", outputdir=None):
    """This funct does the preliminary preprocessing on movielens dataset
    and converts the csv files to parquet files and saves to disk. Then,
    using NVTabular, it does feature engineering on the parquet files
    and saves the processed files to disk.

    Parameters
    ----------
    local_filename : str
        path for downloading the raw dataset in and storing the converted
        parquet files.
    name : str
        The name of the Movielens dataset. Currently Movielens 25M and
        Movielens 100k datasets are supported.
    outputdir : str, optional
        path for saving the processed parquet files generated from NVTabular
        workflow. If not provided, local_filename is used as outputdir.
    """
    local_filename = os.path.abspath(local_filename)
    if outputdir is None:
        outputdir = local_filename
    if name == "ml-25m":
        download_file(
            "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
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

        logger.info("starting ETL..")

        # NVTabular pipeline
        movies = df_lib.read_parquet(os.path.join(local_filename, name, "movies_converted.parquet"))
        joined = ["userId", "movieId"] >> ops.JoinExternal(movies, on=["movieId"])
        cat_features = joined >> ops.Categorify(dtype="int32")
        label = nvt.ColumnSelector(["rating"])

        # Columns to apply to
        cats = nvt.ColumnSelector(["movieId"])

        # Target Encode movieId column
        te_features = cats >> ops.TargetEncoding(label, kfold=5, p_smooth=20)
        te_features_norm = te_features >> ops.NormalizeMinMax()

        # count encode `userId`
        count_logop_feat = (
            ["userId"] >> ops.JoinGroupby(cont_cols=["movieId"], stats=["count"]) >> ops.LogOp()
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
        download_file(
            "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
            os.path.join(local_filename, "ml-100k.zip"),
        )
        logger.info("starting ETL..")
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

        movies = pd.read_csv(
            os.path.join(local_filename, "ml-100k/u.item"), names=cols, sep="|", encoding="latin1"
        )
        for col in genres_:
            movies[col] = movies[col].replace(1, col)
            movies[col] = movies[col].replace(0, np.nan)
        s = movies[genres_]
        s.notnull()
        movies["genres"] = s.notnull().dot(s.columns + ",").str[:-1]
        movies_converted = movies[
            ["movieId", "title", "release_date", "video_release_date", "genres", "imdb_URL"]
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

        cat_features = [
            "userId",
            "movieId",
            "gender",
            "occupation",
            "zip_code",
            "genres",
        ] >> nvt.ops.Categorify(dtype="int32")

        cont_names = ["age"]
        boundaries = {"age": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]}
        age_bucket = cont_names >> ops.Bucketize(boundaries) >> ops.AddMetadata(tags=["user"])

        label = nvt.ColumnSelector(["rating"])

        # Target Encode movieId column
        te_features = ["movieId"] >> ops.TargetEncoding(label, kfold=5, p_smooth=20)

        te_features_norm = te_features >> ops.NormalizeMinMax()

        # count encode `userId`
        count_logop_feat = (
            ["userId"] >> ops.JoinGroupby(cont_cols=["movieId"], stats=["count"]) >> ops.LogOp()
        )

        feats_item = cat_features["movieId"] >> ops.AddMetadata(tags=["item_id", "item"])
        feats_user = cat_features["userId"] >> ops.AddMetadata(tags=["user_id", "user"])
        feats_genres = cat_features["genres"] >> ops.AddMetadata(tags=["item"])
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
    elif name == "ml-1m":
        download_file(
            "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            os.path.join(local_filename, "ml-1m.zip"),
        )

        users = pd.read_csv(
            os.path.join(local_filename, "ml-1m/users.dat"),
            sep="::",
            names=["userId", "gender", "age", "occupation", "zipcode"],
        )
        ratings = pd.read_csv(
            os.path.join(local_filename, "ml-1m/ratings.dat"),
            sep="::",
            names=["userId", "movieId", "rating", "timestamp"],
        )
        movies = pd.read_csv(
            os.path.join(local_filename, "ml-1m/movies.dat"),
            names=["movieId", "title", "genres"],
            sep="::",
        )
        movies["genres"] = movies["genres"].str.split("|")
        movies.to_parquet(os.path.join(local_filename, name, "movies_converted.parquet"))
        users.to_parquet(os.path.join(local_filename, name, "users_converted.parquet"))
        ratings = ratings.sample(len(ratings), replace=False)
        # split the train_df as training and validation data sets.
        num_valid = int(len(ratings) * 0.2)
        train = ratings[:-num_valid]
        valid = ratings[-num_valid:]
        train.to_parquet(os.path.join(local_filename, name, "train.parquet"))
        valid.to_parquet(os.path.join(local_filename, name, "valid.parquet"))

        logger.info("starting ETL..")

        movies = df_lib.read_parquet(os.path.join(local_filename, name, "movies_converted.parquet"))
        users = df_lib.read_parquet(os.path.join(local_filename, name, "users_converted.parquet"))
        joined = (
            ["userId", "movieId"]
            >> ops.JoinExternal(movies, on=["movieId"])
            >> ops.JoinExternal(users, on=["userId"])
        )
        cat_features = joined >> ops.Categorify(dtype="int32")
        label = nvt.ColumnSelector(["rating"])

        # Columns to apply to
        cats = nvt.ColumnSelector(["movieId", "userId"])

        # Target Encode movieId column
        te_features = cats + joined[
            ["age", "gender", "occupation", "zipcode"]
        ] >> ops.TargetEncoding(label, kfold=5, p_smooth=20)
        te_features_norm = te_features >> ops.NormalizeMinMax()

        # count encode `userId`
        count_logop_feat = (
            ["userId"] >> ops.JoinGroupby(cont_cols=["movieId"], stats=["count"]) >> ops.LogOp()
        )
        feats_item = cat_features["movieId"] >> ops.AddMetadata(tags=["item_id", "item"])
        feats_userId = cat_features["userId"] >> ops.AddMetadata(tags=["user_id", "user"])
        feats_genres = cat_features["genres"] >> ops.AddMetadata(tags=["item"])
        feats_te_user = (
            te_features_norm[
                [
                    "TE_userId_rating",
                    "TE_age_rating",
                    "TE_gender_rating",
                    "TE_occupation_rating",
                    "TE_zipcode_rating",
                ]
            ]
            >> ops.AddMetadata(tags=["user"])
        )
        feats_te_item = te_features_norm[["TE_movieId_rating"]] >> ops.AddMetadata(tags=["item"])
        feats_user = joined[["age", "gender", "occupation", "zipcode"]] >> ops.AddMetadata(
            tags=["item"]
        )

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
            cat_features
            + te_features_norm
            + feats_te_user
            + feats_te_item
            + feats_item
            + feats_userId
            + feats_genres
            + feats_target
            + target_orig
        )
    else:
        raise ValueError(
            "Unknown dataset name. Only Movielens 25M, 1M and 100k datasets are supported."
        )

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

    logger.info("saving the workflow..")


def movielens_session_download_etl(local_filename, name="ml-1m", outputdir=None, max_length=None):
    """This funct does the preliminary preprocessing on movielens dataset
    and converts the csv files to parquet files and saves to disk. Then,
    using NVTabular, it does feature engineering on the parquet files to
    generate daily user sessions and saves the processed files to disk.
    Parameters
    ----------
    local_filename : str
        path for downloading the raw dataset in and storing the converted
        parquet files.
    name : str
        The name of the Movielens dataset. Currently Movielens 1M is supported.
        Default "ml-1m""
    outputdir : str, optional
        path for saving the processed parquet files generated from NVTabular
        workflow. If not provided, local_filename is used as outputdir.
        Default to None
    max_length: int
        If set, truncate the sessions to last `max_length` interacted movies.
        Default to None
    """
    local_filename = os.path.abspath(local_filename)
    if outputdir is None:
        outputdir = local_filename
    # Download data
    download_file(
        "https://files.grouplens.org/datasets/movielens/%s.zip" % name,
        os.path.join(local_filename, "%s.zip" % name),
    )
    if name == "ml-1m":

        # Convert .dat files to parquet format
        users = pd.read_csv(
            os.path.join(local_filename, "ml-1m/users.dat"),
            sep="::",
            names=["userId", "gender", "age", "occupation", "zipcode"],
        )
        users.to_parquet(os.path.join(local_filename, name, "users.parquet"))
        movies = pd.read_csv(
            os.path.join(local_filename, "ml-1m/movies.dat"),
            names=["movieId", "title", "genres"],
            sep="::",
        )
        movies["genres"] = movies["genres"].str.split("|")
        movies.to_parquet(os.path.join(local_filename, name, "movies.parquet"))
        ratings = pd.read_csv(
            os.path.join(local_filename, "ml-1m/ratings.dat"),
            sep="::",
            names=["userId", "movieId", "rating", "timestamp"],
        )
        ratings.to_parquet(os.path.join(local_filename, name, "ratings.parquet"))
        # Get global variable needed for nvt workflow
        min_day = ratings["timestamp"].min() // 86400

        logger.info("starting ETL..")
        # NVTabular pipeline
        # join with movies and users features
        joined = (
            ["userId", "movieId", "rating", "timestamp"]
            >> ops.JoinExternal(movies, on=["movieId"])
            >> ops.JoinExternal(users, on=["userId"])
        )
        # encode categorical features
        cat_features = joined >> ops.Categorify(dtype="int32")

        # Get day column from timestamps
        def create_day_column(ts, df):
            day = ts // 86400
            day = day - min_day
            return day

        day_feature = (
            ["timestamp"] >> ops.LambdaOp(create_day_column) >> ops.Rename(f=lambda x: "day")
        )
        # add tags
        feats_item = cat_features["movieId"] >> ops.AddMetadata(tags=[Tags.ITEM_ID, Tags.ITEM])
        feats_userId = cat_features["userId"] >> ops.AddMetadata(tags=[Tags.USER_ID, Tags.USER])
        feats_genres = cat_features["genres"] >> ops.AddMetadata(tags=[Tags.ITEM, Tags.SEQUENCE])
        feats_user = cat_features["gender", "age", "occupation", "zipcode"] >> ops.AddMetadata(
            tags=Tags.USER
        )
        # group by user and day
        features = (
            cat_features + day_feature + feats_item + feats_genres + feats_userId + feats_user
        )
        groupby_features = features >> nvt.ops.Groupby(
            groupby_cols=["userId", "day"],
            sort_cols=["timestamp"],
            aggs={
                "movieId": ["list", "count"],
                "timestamp": ["list"],
                "genres": ["list"],
                "gender": ["first"],
                "age": ["first"],
                "occupation": ["first"],
                "zipcode": ["first"],
            },
            name_sep="-",
        )
        if max_length:
            # Truncate sessions to the last interacted `max_length` movies
            groupby_features_list = groupby_features["movieId-list"] >> ops.AddMetadata(
                tags=[Tags.SEQUENCE]
            )
            groupby_features_truncated = (
                groupby_features_list
                >> nvt.ops.ListSlice(-max_length, pad=True)
                >> nvt.ops.Rename(postfix="_truncated")
            )
            # Apply workflow
            workflow = nvt.Workflow(groupby_features + groupby_features_truncated)
        else:
            workflow = nvt.Workflow(groupby_features)
        train_dataset = nvt.Dataset([os.path.join(local_filename, name, "ratings.parquet")])
        workflow.fit(train_dataset)
        user_sessions = workflow.transform(train_dataset)
        logger.info("saving the workflow..")
        # save the workflow
        workflow.save(os.path.join(local_filename, name, "workflow"))
        # Save the processed data and correspondin schema
        user_sessions.to_parquet(
            output_path=os.path.join(local_filename, name),
            out_files_per_proc=1,
            shuffle=False,
        )

        # Post-processing to create train / validation sets
        user_sessions_gdf = user_sessions.compute()

        # flatten the nested list of genres
        def flatten_genres(g):
            if len(g) > 0:
                return np.concatenate(g).ravel()
            else:
                return []

        user_sessions_gdf["genres-list"] = (
            user_sessions_gdf["genres-list"].to_pandas().map(flatten_genres)
        )
        # split data by `day`
        max_day = user_sessions_gdf.day.max()
        valid = user_sessions_gdf[user_sessions_gdf.day > max_day - 200]
        train = user_sessions_gdf[user_sessions_gdf.day <= max_day - 200]

        # save train and validation
        os.makedirs(os.path.join(local_filename, name, "train"), exist_ok=True)
        train.to_parquet(os.path.join(local_filename, name, "train", "train.parquet"))
        os.makedirs(os.path.join(local_filename, name, "valid"), exist_ok=True)
        valid.to_parquet(os.path.join(local_filename, name, "valid", "valid.parquet"))
