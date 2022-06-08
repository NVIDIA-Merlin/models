import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

import merlin.io

# Get dataframe library - cuDF or pandas
from merlin.core.dispatch import get_lib
from merlin.core.utils import download_file
from merlin.datasets import BASE_PATH
from merlin.models.utils.example_utils import workflow_fit_transform
from merlin.models.utils.nvt_utils import require_nvt

df_lib = get_lib()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    import nvtabular as nvt

    Workflow = nvt.Workflow
except ImportError:
    Workflow = None


VARIANTS = {"ml-25m", "ml-1m", "ml-100k"}


def validate_variant(variant: str):
    if variant not in VARIANTS:
        raise ValueError("MovieLens dataset variant not supported. " f"Must be one of {VARIANTS}")


def get_movielens(
    path: Union[str, Path] = None,
    variant="ml-25m",
    overwrite: bool = False,
    transformed_name: str = "transformed",
    nvt_workflow: Optional[Workflow] = None,
    **kwargs,
):
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
    variant :  str
        The variant of the movielens dataset to use.
        Must be one of "ml-25m", "ml-1m" or "ml-100k"

    Returns
    -------
    tuple
        A tuple consisting of a merlin.io.Dataset for the training dataset and validation dataset
    """
    validate_variant(variant)
    require_nvt()

    if path is None:
        p = Path(BASE_PATH) / "movielens"
    else:
        p = Path(path)

    raw_path = p / variant
    if not raw_path.exists():
        download_movielens(p, variant)

    nvt_path = raw_path / transformed_name
    train_path, valid_path = nvt_path / "train", nvt_path / "valid"
    nvt_path_exists = train_path.exists() and valid_path.exists()
    if not nvt_path_exists or overwrite:
        transform_movielens(
            raw_path, nvt_path, nvt_workflow=nvt_workflow, variant=variant, **kwargs
        )

    train = merlin.io.Dataset(str(train_path), engine="parquet")
    valid = merlin.io.Dataset(str(valid_path), engine="parquet")

    return train, valid


def download_movielens(path: Union[str, Path], variant: str = "ml-25m"):
    """Downloads the movielens dataset to the specified path

    Parameters
    ----------
    path : str
        The path to download the files locally to. If not set will default to
        the 'merlin-models-data` directory in your home folder
    variant :  str
        The variant of the movielens dataset to use.
        Must be one of "ml-25m", "ml-1m" or "ml-100k"
    """
    download_file(
        f"http://files.grouplens.org/datasets/movielens/{variant}.zip",
        os.path.join(path, f"{variant}.zip"),
    )


def transform_movielens(
    raw_data_path: Union[str, Path],
    output_path: Union[str, Path],
    nvt_workflow: Optional[Workflow] = None,
    variant: str = "ml-25m",
    **kwargs,
):
    """
    Transforms the movielens dataset to be ready for use with merlin-models

    Parameters
    ----------
    raw_data_path: Union[str, Path]
        The path to the raw data
    output_path: Union[str, Path]
        The path to save the transformed data
    nvt_workflow: Optional[Workflow]
        The NVTabular workflow to use for the transformation.
        If not set, will use the default.
    variant: str
        The variant of the movielens dataset to use.
        Must be one of "ml-25m", "ml-1m" or "ml-100k"
    """

    if nvt_workflow:
        _nvt_workflow = nvt_workflow
    else:
        if variant == "ml-25m":
            _nvt_workflow = default_ml25m_transformation(**locals())
        elif variant == "ml-1m":
            _nvt_workflow = default_ml1m_transformation(**locals())
        elif variant == "ml-100k":
            _nvt_workflow = default_ml100k_transformation(**locals())
        else:
            raise ValueError(
                "Unknown dataset name. Only Movielens 25M, 1M and 100k datasets are supported."
            )

    workflow_fit_transform(
        _nvt_workflow,
        os.path.join(raw_data_path, "train.parquet"),
        os.path.join(raw_data_path, "valid.parquet"),
        str(output_path),
    )


def default_ml25m_transformation(raw_data_path: str, **kwargs):
    from nvtabular import ops

    movies = df_lib.read_csv(os.path.join(raw_data_path, "movies.csv"))
    movies["genres"] = movies["genres"].str.split("|")
    movies.to_parquet(os.path.join(raw_data_path, "movies_converted.parquet"))
    ratings = df_lib.read_csv(os.path.join(raw_data_path, "ratings.csv"))
    # shuffle the dataset
    ratings = ratings.sample(len(ratings), replace=False)
    # split the train_df as training and validation data sets.
    num_valid = int(len(ratings) * 0.2)
    train = ratings[:-num_valid]
    valid = ratings[-num_valid:]
    train.to_parquet(os.path.join(raw_data_path, "train.parquet"))
    valid.to_parquet(os.path.join(raw_data_path, "valid.parquet"))

    logger.info("starting ETL..")

    # NVTabular pipeline
    movies = df_lib.read_parquet(os.path.join(raw_data_path, "movies_converted.parquet"))
    joined = ["userId", "movieId"] >> ops.JoinExternal(movies, on=["movieId"])
    cat_features = joined >> ops.Categorify(dtype="int32")
    label = nvt.ColumnSelector(["rating"])

    # Columns to apply to
    cats = nvt.ColumnSelector(["movieId"])

    # Target Encode movieId column
    te_features = cats >> ops.TargetEncoding(label, kfold=5, p_smooth=20)
    te_features_norm = te_features >> ops.Normalize() >> ops.TagAsItemFeatures()

    # count encode `userId`
    count_logop_feat = (
        ["userId"]
        >> ops.JoinGroupby(cont_cols=["movieId"], stats=["count"])
        >> ops.LogOp()
        >> ops.TagAsUserFeatures()
    )
    feats_item = cat_features["movieId"] >> ops.AddMetadata(tags=["item_id", "item"])
    feats_user = cat_features["userId"] >> ops.AddMetadata(tags=["user_id", "user"])
    feats_genres = cat_features["genres"] >> ops.ValueCount() >> ops.TagAsItemFeatures()

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
    return nvt.Workflow(
        feats_item
        + feats_user
        + feats_genres
        + te_features_norm
        + count_logop_feat
        + target_orig
        + feats_target
        + joined["title"]
    )


def default_ml1m_transformation(raw_data_path: str, **kwargs):
    from nvtabular import ops

    users = pd.read_csv(
        os.path.join(raw_data_path, "users.dat"),
        sep="::",
        names=["userId", "gender", "age", "occupation", "zipcode"],
    )
    ratings = pd.read_csv(
        os.path.join(raw_data_path, "ratings.dat"),
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        os.path.join(raw_data_path, "movies.dat"),
        names=["movieId", "title", "genres"],
        sep="::",
        encoding="latin1",
    )
    movies["genres"] = movies["genres"].str.split("|")
    movies.to_parquet(os.path.join(raw_data_path, "movies_converted.parquet"))
    users.to_parquet(os.path.join(raw_data_path, "users_converted.parquet"))
    ratings = ratings.sample(len(ratings), replace=False)
    # split the train_df as training and validation data sets.
    num_valid = int(len(ratings) * 0.2)
    train = ratings[:-num_valid]
    valid = ratings[-num_valid:]
    train.to_parquet(os.path.join(raw_data_path, "train.parquet"))
    valid.to_parquet(os.path.join(raw_data_path, "valid.parquet"))

    logger.info("starting ETL..")

    movies = df_lib.read_parquet(os.path.join(raw_data_path, "movies_converted.parquet"))
    users = df_lib.read_parquet(os.path.join(raw_data_path, "users_converted.parquet"))
    joined = (
        ["userId", "movieId"]
        >> ops.JoinExternal(movies, on=["movieId"])
        >> ops.JoinExternal(users, on=["userId"])
    )

    cat = lambda: nvt.ops.Categorify(dtype="int32")  # noqa

    cat_features = joined >> cat()
    label = nvt.ColumnSelector(["rating"])

    # Columns to apply to
    cats = nvt.ColumnSelector(["movieId", "userId"])

    # Target Encode movieId column
    te_features = cats + joined[["age", "gender", "occupation", "zipcode"]] >> ops.TargetEncoding(
        label, kfold=5, p_smooth=20
    )
    te_features_norm = te_features >> ops.Normalize()

    # count encode `userId`
    # count_logop_feat = (
    #     ["userId"] >> ops.JoinGroupby(cont_cols=["movieId"], stats=["count"]) >> ops.LogOp()
    # )
    feats_item = cat_features["movieId"] >> ops.AddMetadata(tags=["item_id", "item"])
    feats_userId = cat_features["userId"] >> ops.AddMetadata(tags=["user_id", "user"])
    feats_genres = cat_features["genres"] >> ops.ValueCount() >> ops.TagAsItemFeatures()
    feats_te_user = te_features_norm[
        [
            "TE_userId_rating",
            "TE_age_rating",
            "TE_gender_rating",
            "TE_occupation_rating",
            "TE_zipcode_rating",
        ]
    ] >> ops.AddMetadata(tags=["user"])
    feats_te_item = te_features_norm[["TE_movieId_rating"]] >> ops.AddMetadata(tags=["item"])
    # feats_user = joined[["age", "gender", "occupation", "zipcode"]] >> ops.AddMetadata(
    #     tags=["item"]
    # )

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

    return nvt.Workflow(
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


def default_ml100k_transformation(raw_data_path: str, **kwargs):
    from nvtabular import ops

    logger.info("starting ETL..")
    # ratings = pd.read_csv(
    #     os.path.join(raw_data_path, "u.data"),
    #     names=["userId", "movieId", "rating", "timestamp"],
    #     sep="\t",
    # )
    user_features = pd.read_csv(
        os.path.join(raw_data_path, "u.user"),
        names=["userId", "age", "gender", "occupation", "zip_code"],
        sep="|",
    )
    user_features.to_parquet(os.path.join(raw_data_path, "user_features.parquet"))
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
        "Childrens",  # noqa
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
        os.path.join(raw_data_path, "u.item"), names=cols, sep="|", encoding="latin1"
    )
    for col in genres_:
        movies[col] = movies[col].replace(1, col)
        movies[col] = movies[col].replace(0, np.nan)
    s = movies[genres_]
    movies["genres"] = s.notnull().dot(s.columns + ",").str[:-1]
    movies_converted = movies[
        ["movieId", "title", "release_date", "video_release_date", "genres", "imdb_URL"]
    ]
    movies_converted.to_parquet(os.path.join(raw_data_path, "movies_converted.parquet"))
    train = pd.read_csv(
        os.path.join(raw_data_path, "ua.base"),
        names=["userId", "movieId", "rating", "timestamp"],
        sep="\t",
    )
    valid = pd.read_csv(
        os.path.join(raw_data_path, "ua.test"),
        names=["userId", "movieId", "rating", "timestamp"],
        sep="\t",
    )
    train = train.merge(user_features, on="userId", how="left")
    train = train.merge(movies_converted, on="movieId", how="left")
    valid = valid.merge(user_features, on="userId", how="left")
    valid = valid.merge(movies_converted, on="movieId", how="left")

    train.to_parquet(os.path.join(raw_data_path, "train.parquet"))
    valid.to_parquet(os.path.join(raw_data_path, "valid.parquet"))

    cat = lambda: nvt.ops.Categorify(dtype="int32")  # noqa

    cont_names = ["age"]
    boundaries = {"age": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]}
    age_bucket = cont_names >> ops.Bucketize(boundaries) >> cat() >> ops.AddMetadata(tags=["user"])

    label = nvt.ColumnSelector(["rating"])

    # Target Encode movieId column
    te_features = ["movieId"] >> ops.TargetEncoding(label, kfold=5, p_smooth=20)

    te_features_norm = te_features >> ops.Normalize()

    # count encode `userId`
    count_logop_feat = (
        ["userId"] >> ops.JoinGroupby(cont_cols=["movieId"], stats=["count"]) >> ops.LogOp()
    )

    feats_item = ["movieId"] >> cat() >> ops.TagAsItemID()
    feats_user = ["userId"] >> cat() >> ops.TagAsUserID()
    feats_genres = ["genres"] >> cat() >> ops.ValueCount() >> ops.TagAsItemFeatures()
    user_features = ["gender", "zip_code"] >> cat() >> ops.TagAsUserFeatures()

    feats_target = (
        nvt.ColumnSelector(["rating"])
        >> ops.LambdaOp(lambda col: (col > 3).astype("int32"))
        >> ops.AddMetadata(tags=["binary_classification", "target"])
        >> nvt.ops.Rename(name="rating_binary")
    )
    target_orig = ["rating"] >> ops.AddMetadata(tags=["regression", "target"])

    return nvt.Workflow(
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
