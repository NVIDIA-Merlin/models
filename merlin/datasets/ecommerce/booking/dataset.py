import os
import urllib
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import merlin.io
from merlin.core.dispatch import get_lib
from merlin.dag import ColumnSelector
from merlin.datasets import BASE_PATH
from merlin.models.utils.example_utils import workflow_fit_transform
from merlin.models.utils.nvt_utils import require_nvt
from merlin.schema import Tags

try:
    import nvtabular as nvt

    Workflow = nvt.Workflow
except ImportError:
    Workflow = None


_FILES = ["ground_truth.csv", "test_set.csv", "train_set.csv"]
_DATA_URL = "https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main/"


def get_booking(
    path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    transformed_name: str = "transformed",
    nvt_workflow: Optional[Workflow] = None,
    **kwargs,
) -> Tuple[merlin.io.Dataset, merlin.io.Dataset]:
    """Dataset for the WSDM '21 challenge organized by booking.com.

    The goal of this challenge is to use a dataset based on
    millions of real anonymized accommodation reservations
    to come up with a strategy for making the best recommendation
    for their next destination in real-time.

    Parameters
    ----------
    path (Union[str, Path]): Path to save the dataset.
    overwrite (bool, optional): Whether or not to overwrite the dataset,
        if already downloaded. Defaults to False.
    transformed_name (str, optional): Name of folder to put the transformed dataset in.
        Defaults to "transformed".
    nvt_workflow (Optional[Workflow], optional): NVTabular workflow, pass this in
        if you want to customize the default workflow. Defaults to `default_booking_transformation`.

    Returns
    -------
        train: merlin.io.Dataset
            Training dataset.
        valid: merlin.io.Dataset
            Test dataset.
    """
    require_nvt()

    if path is None:
        p = Path(BASE_PATH) / "booking"
    else:
        p = Path(path)

    raw_path = p
    if not (
        _check_path(raw_path / "train", check_schema=False)
        and _check_path(raw_path / "test", check_schema=False)
    ):
        download_booking(p)

    nvt_path = raw_path / transformed_name
    train_path, valid_path = nvt_path / "train", nvt_path / "valid"
    nvt_path_exists = _check_path(train_path) and _check_path(valid_path)
    if not nvt_path_exists or overwrite:
        transform_booking(raw_path, nvt_path, nvt_workflow=nvt_workflow, **kwargs)

    _check_path(train_path, strict=True)
    _check_path(valid_path, strict=True)
    train = merlin.io.Dataset(str(train_path), engine="parquet")
    valid = merlin.io.Dataset(str(valid_path), engine="parquet")

    return train, valid


def download_booking(path: Path):
    """Automatically download the booking dataset.

    Parameters
    ----------
    path (Path): output-path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for file in _FILES:
        local_filename = str(path / file)
        url = os.path.join(_DATA_URL, file)
        desc = f"downloading {os.path.basename(local_filename)}"
        with tqdm(unit="B", unit_scale=True, desc=desc) as progress:

            def report(chunk, chunksize, total):
                if not progress.total:
                    progress.reset(total=total)
                progress.update(chunksize)

            urllib.request.urlretrieve(url, local_filename, reporthook=report)

    preprocess_booking(path)


def preprocess_booking(
    path: Path,
):
    path = Path(path)
    train = get_lib().read_csv(path / "train_set.csv")
    test = get_lib().read_csv(path / "test_set.csv")

    train["checkin"] = get_lib().to_datetime(train["checkin"], format="%Y-%m-%d")
    train["checkout"] = get_lib().to_datetime(train["checkout"], format="%Y-%m-%d")
    train["timestamp"] = train["checkout"].astype("int64")
    test["checkin"] = get_lib().to_datetime(test["checkin"], format="%Y-%m-%d")
    test["checkout"] = get_lib().to_datetime(test["checkout"], format="%Y-%m-%d")
    test["timestamp"] = test["checkout"].astype("int64")

    (path / "train").mkdir(exist_ok=True)
    (path / "test").mkdir(exist_ok=True)

    train.to_parquet(path / "train/data.parquet")
    test.to_parquet(path / "test/data.parquet")


def _get_cycled_feature_value_sin(col, max_value=7):
    value_scaled = (col + 0.000001) / max_value
    value_sin = np.sin(2 * np.pi * value_scaled)
    return value_sin


def _get_cycled_feature_value_cos(col, max_value=7):
    value_scaled = (col + 0.000001) / max_value
    value_cos = np.cos(2 * np.pi * value_scaled)
    return value_cos


def transform_booking(
    data: Union[str, Path, Tuple[merlin.io.Dataset, merlin.io.Dataset]],
    output_path: Union[str, Path],
    nvt_workflow=None,
    **kwargs,
):
    """Transform the booking dataset.

    Parameters
    ----------
    data (Union[str, Path, Tuple[merlin.io.Dataset, merlin.io.Dataset]]):
        Raw training and test data.
    output_path (Union[str, Path]): Path to save the transformed dataset.
    nvt_workflow (Optional[Workflow], optional): NVTabular workflow, pass this in
        if you want to customize the default workflow. Defaults to `default_booking_transformation`.
    """
    nvt_workflow = nvt_workflow or default_booking_transformation(**locals())

    if isinstance(data, (str, Path)):
        _train = merlin.io.Dataset(str(Path(data) / "train"), engine="parquet")
        _valid = merlin.io.Dataset(str(Path(data) / "test"), engine="parquet")
    elif (
        isinstance(data, tuple)
        and len(data) == 2
        and all(isinstance(x, merlin.io.Dataset) for x in data)
    ):
        _train, _valid = data
    else:
        raise ValueError("data must be a path or a tuple of train and valid datasets")

    workflow_fit_transform(nvt_workflow, _train, _valid, str(output_path), **kwargs)


def default_booking_transformation(**kwargs):
    """Default transformation for the booking dataset.

    Returns:
        Workflow: NVTabular workflow
    """
    cat = lambda: nvt.ops.Categorify(start_index=1)  # noqa: E731

    df_season = get_lib().DataFrame(
        {"month": range(1, 13), "season": ([0] * 3) + ([1] * 3) + ([2] * 3) + ([3] * 3)}
    )

    month = (
        ["checkin"]
        >> nvt.ops.LambdaOp(lambda col: col.dt.month.astype("int64"))
        >> nvt.ops.Rename(name="month")
    )

    weekday_checkin = (
        ["checkin"]
        >> nvt.ops.LambdaOp(lambda col: col.dt.weekday)
        >> nvt.ops.Rename(name="weekday_checkin")
    )

    weekday_checkout = (
        ["checkout"]
        >> nvt.ops.LambdaOp(lambda col: col.dt.weekday)
        >> nvt.ops.Rename(name="weekday_checkout")
    )

    is_weekend = (
        weekday_checkin
        >> nvt.ops.LambdaOp(lambda col: col.isin([5, 6]).astype(int))
        >> nvt.ops.Rename(name="is_weekend")
    )

    length_feature = (
        ["checkout"]
        >> nvt.ops.LambdaOp(lambda col, df: (col - df["checkin"]).dt.days, dependency=["checkin"])
        >> nvt.ops.Rename(name="length")
    )

    season = month >> nvt.ops.JoinExternal(
        df_season,
        on="month",
        how="left",
    )

    weekday_sin = (
        weekday_checkout
        >> (lambda col: _get_cycled_feature_value_sin(col + 1, 7))
        >> nvt.ops.Rename(name="dayofweek_sin")
        >> nvt.ops.AddTags([Tags.CONTINUOUS])
    )
    weekday_cos = (
        weekday_checkout
        >> (lambda col: _get_cycled_feature_value_cos(col + 1, 7))
        >> nvt.ops.Rename(name="dayofweek_cos")
        >> nvt.ops.AddTags([Tags.CONTINUOUS])
    )

    context_cat_features = ["device_class", "affiliate_id"] + season >> cat()
    seq_cat_features = (
        ["booker_country", "hotel_country"]
        + month
        + weekday_checkin
        + weekday_checkout
        + weekday_sin
        + weekday_cos
        + is_weekend
        + length_feature
        >> cat()
        >> nvt.ops.AddTags(tags=[Tags.SEQUENCE])
    )

    cityid = ["city_id"] >> cat() >> nvt.ops.AddTags(tags=[Tags.ITEM_ID])
    session_id = (
        ["utrip_id"]
        >> cat()
        >> nvt.ops.AddTags(tags=[Tags.SESSION_ID])
        >> nvt.ops.Rename(name="session_id")
    )
    user_id = ["user_id"] >> cat() >> nvt.ops.AddTags(tags=[Tags.USER_ID])

    features = (
        ColumnSelector(["timestamp", "utrip_id"])
        + seq_cat_features
        + cityid
        + context_cat_features
        + session_id
        + user_id
    )

    grouped = (
        features
        >> nvt.ops.Groupby(
            groupby_cols=["utrip_id"],
            sort_cols=["timestamp"],
            aggs={
                "user_id": "first",
                "session_id": "first",
                "device_class": "first",
                "booker_country": "list",
                "hotel_country": "list",
                "month": "list",
                "is_weekend": "list",
                "weekday_checkin": "list",
                "weekday_checkout": "list",
                "length_feature": "first",
                "city_id": "list",
            },
        )
        >> nvt.ops.Rename(_remove_list_and_first_from_name)
        >> nvt.ops.ValueCount()
    )

    return grouped


def _remove_list_and_first_from_name(name):
    if name.endswith("_list"):
        return name[:-5]

    if name.endswith("_first"):
        return name[:-6]

    return name


def _check_path(path, strict=False, check_schema=True):
    if not isinstance(path, (str, Path)):
        if strict:
            raise ValueError("path must be a string or a Path object")

        return False

    if not Path(path).exists():
        if strict:
            raise ValueError(f"path {path} does not exist")

        return False

    if not len(list(Path(path).glob("*.parquet"))):
        if strict:
            raise ValueError(f"path {path} does not contain any parquet files")

        return False

    if check_schema:
        if not len(list(Path(path).glob("schema.*"))):
            if strict:
                raise ValueError(f"path {path} does not contain a schema")

            return False

    return True
