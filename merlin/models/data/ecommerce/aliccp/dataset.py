import os
import pickle
import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from nvtabular import ops as nvt_ops
from tqdm import tqdm

import merlin.io
from merlin.core.dispatch import get_lib
from merlin.models.utils.example_utils import workflow_fit_transform
from merlin.schema import Tags


def get_aliccp(path, **kwargs) -> Tuple[merlin.io.Dataset, merlin.io.Dataset]:
    path_fn = lambda x: os.path.join(path, x)  # noqa
    if not os.path.exists(path_fn("raw")):
        os.makedirs(path_fn("raw"))
        prepare_alliccp(path, output_dir=path_fn("raw"), **kwargs)

    t_path_fn = lambda x: os.path.join(path_fn("transformed"), x)  # noqa
    if not (os.path.exists(t_path_fn("train")) and os.path.exists(t_path_fn("valid"))):
        transform_aliccp(path_fn("raw"), path_fn("transformed"))

    train = merlin.io.Dataset(t_path_fn("train"), engine="parquet")
    valid = merlin.io.Dataset(t_path_fn("valid"), engine="parquet")

    return train, valid


def prepare_alliccp(
    data_dir: Union[str, Path],
    convert_train: bool = True,
    convert_test: bool = True,
    file_size: int = 10000,
    max_num_rows: Optional[int] = None,
    pickle_common_features=True,
    output_dir: Optional[Union[str, Path]] = None,
):
    """
    Convert Ali-CPP data to parquet files.

    To download the raw the Ali-CCP training and test datasets visit
    [tianchi.aliyun.com](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408#1).

    Parameters
    ----------
    data_dir: Union[str, Path]
        Directory to load the raw data from.
    convert_train: bool
        Whether to convert the training data.
    convert_test: bool
        Whether to convert the test data.
    file_size: int
        Number of rows to write to each parquet file.
    max_num_rows: int, optional
        Maximum number of rows to read from the raw data.
    pickle_common_features: bool
        Whether to pickle the common features.
        When enabled it will make the conversion faster if it would be run again.
    output_dir: Union[str, Path], optional
        Directory to write the converted data to.
        If not specified the data will be written to the same directory as the raw data.

    Returns
    -------
    data_dir
    """

    if convert_train:
        _convert_data(
            str(data_dir),
            file_size,
            is_train=True,
            max_num_rows=max_num_rows,
            pickle_common_features=pickle_common_features,
            output_dir=output_dir,
        )
    if convert_test:
        _convert_data(
            str(data_dir),
            file_size,
            is_train=False,
            max_num_rows=max_num_rows,
            pickle_common_features=pickle_common_features,
            output_dir=output_dir,
        )

    return data_dir


def transform_aliccp(
    raw_data_path: Union[str, Path], output_path: Union[str, Path], add_target_encoding=True
):
    user_id = ["user_id"] >> nvt_ops.Categorify(dtype="int32") >> nvt_ops.TagAsUserID()
    item_id = ["item_id"] >> nvt_ops.Categorify(dtype="int32") >> nvt_ops.TagAsItemID()

    item_features = (
        ["item_category", "item_shop", "item_brand"]
        >> nvt_ops.Categorify(dtype="int32")
        >> nvt_ops.TagAsItemFeatures()
    )

    user_features = (
        [
            "user_shops",
            "user_profile",
            "user_group",
            "user_gender",
            "user_age",
            "user_consumption_2",
            "user_is_occupied",
            "user_geography",
            "user_intentions",
            "user_brands",
            "user_categories",
        ]
        >> nvt_ops.Categorify(dtype="int32")
        >> nvt_ops.TagAsUserFeatures()
    )

    targets = ["click", "conversion"] >> nvt_ops.AddMetadata(
        tags=[str(Tags.BINARY_CLASSIFICATION), "target"]
    )
    outputs = user_id + item_id + item_features + user_features + targets

    if add_target_encoding:
        continuous = (
            ["user_id", "item_id"]
            >> nvt_ops.TargetEncoding(["click"], kfold=1, p_smooth=20)
            >> nvt_ops.Normalize()
        )
        outputs += continuous

    workflow_fit_transform(
        outputs,
        glob(os.path.join(raw_data_path, "train", "train_*")),
        glob(os.path.join(raw_data_path, "test", "test_*")),
        str(output_path),
    )


@dataclass
class _Feature:
    name: str
    id: str
    tags: List[Union[str, Tags]]
    description: str


class _Features:
    def __init__(self):
        self.features: List[_Feature] = [
            # User
            _Feature("user_id", "101", [Tags.USER_ID, Tags.USER], "User ID"),
            _Feature(
                "user_categories",
                "109_14",
                [Tags.USER],
                "User historical behaviors of category ID and count",
            ),
            _Feature(
                "user_shops",
                "110_14",
                [Tags.USER],
                "User historical behaviors of shop ID and count",
            ),
            _Feature(
                "user_brands",
                "127_14",
                [Tags.USER],
                "User historical behaviors of brand ID and count",
            ),
            _Feature(
                "user_intentions",
                "150_14",
                [Tags.USER],
                "User historical behaviors of intention node ID and count",
            ),
            _Feature("user_profile", "121", [Tags.USER], "Categorical ID of User Profile"),
            _Feature("user_group", "122", [Tags.USER], "Categorical group ID of User Profile"),
            _Feature("user_gender", "124", [Tags.USER], "Users Gender ID"),
            _Feature("user_age", "125", [Tags.USER], "Users Age ID"),
            _Feature("user_consumption_1", "126", [Tags.USER], "Users Consumption Level Type I"),
            _Feature("user_consumption_2", "127", [Tags.USER], "Users Consumption Level Type II"),
            _Feature(
                "user_is_occupied", "128", [Tags.USER], "Users Occupation: whether or not to work"
            ),
            _Feature("user_geography", "129", [Tags.USER], "Users Geography Informations"),
            # Item
            _Feature("item_id", "205", [Tags.ITEM, Tags.ITEM_ID], "Item ID"),
            _Feature(
                "item_category", "206", [Tags.ITEM], "Category ID to which the item belongs to"
            ),
            _Feature("item_shop", "207", [Tags.ITEM], "Shop ID to which item belongs to"),
            _Feature(
                "item_intention", "210", [Tags.ITEM], "Intention node ID which the item belongs to"
            ),
            _Feature("item_brand", "216", [Tags.ITEM], "Brand ID of the item"),
            # User-Item
            _Feature(
                "user_item_categories",
                "508",
                ["user_item"],
                "The combination of features with 109_14 and 206",
            ),
            _Feature(
                "user_item_shops",
                "509",
                ["user_item"],
                "The combination of features with 110_14 and 207",
            ),
            _Feature(
                "user_item_brands",
                "702",
                ["user_item"],
                "The combination of features with 127_14 and 216",
            ),
            _Feature(
                "user_item_intentions",
                "853",
                ["user_item"],
                "The combination of features with 150_14 and 210",
            ),
            # Context
            _Feature("position", "301", [Tags.CONTEXT], "A categorical expression of position"),
        ]

    @property
    def by_id(self):
        return {feature.id: feature.name for feature in self.features}


def _convert_common_features(common_path, pickle_path=None):
    common = {}

    with open(common_path, "r") as common_features:
        for csv_line in tqdm(common_features, desc="Reading common features..."):
            line = csv_line.strip().split(",")
            kv = np.array(re.split("[]", line[2]))
            keys = kv[range(0, len(kv), 3)]
            values = kv[range(1, len(kv), 3)]
            common[line[0]] = dict(zip(keys, values))

        if pickle_path:
            with open(pickle_path, "wb") as handle:
                pickle.dump(common, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return common


# TODO: Optimize this function, right now it's too slow.
def _convert_data(
    data_dir,
    file_size,
    is_train=True,
    pickle_common_features=True,
    max_num_rows=None,
    output_dir=None,
):
    data_type = "train" if is_train else "test"
    output_dir = output_dir or data_dir

    common_path = os.path.join(data_dir, data_type, f"common_features_{data_type}.csv")
    path = os.path.join(data_dir, data_type, f"sample_skeleton_{data_type}.csv")
    common_features_path = os.path.join(data_dir, data_type, "common_features.pickle")

    common = {}
    if pickle_common_features:
        if os.path.exists(common_features_path):
            with open(common_features_path, "rb") as f:
                common = pickle.load(f)

    if not common:
        pickle_path = common_features_path if pickle_common_features else None
        common = _convert_common_features(common_path, pickle_path)

    current = []
    by_id = _Features().by_id

    with open(path, "r") as skeleton:
        for i, csv_line in tqdm(enumerate(skeleton), desc="Processing data..."):
            if max_num_rows and i >= max_num_rows:
                break

            line = csv_line.strip().split(",")
            if line[1] == "0" and line[2] == "1":
                continue
            kv = np.array(re.split("[]", line[5]))
            key = kv[range(0, len(kv), 3)]
            value = kv[range(1, len(kv), 3)]
            feat_dict = dict(zip(key, value))
            feat_dict.update(common[line[3]])
            feat_dict["click"] = int(line[1])
            feat_dict["conversion"] = int(line[2])

            current.append(feat_dict)

            if i > 0 and i % file_size == 0:
                df = get_lib().DataFrame(current)
                cols = []
                for col in list(df.columns):
                    if col == "click" or col == "conversion":
                        cols.append(col)
                    else:
                        cols.append(by_id[col])

                df.columns = cols

                out_dir = os.path.join(str(output_dir), data_type)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                index = int((i / file_size) - 1)
                df.to_parquet(
                    os.path.join(out_dir, f"{data_type}_{index}.parquet"),
                    overwrite=True,
                )
                current = []


def _raw_transform(raw_data_path: Union[str, Path], output_path: Union[str, Path]):
    user_id = ["user_id"] >> nvt_ops.Categorify() >> nvt_ops.TagAsUserID()
    item_id = ["item_id"] >> nvt_ops.Categorify() >> nvt_ops.TagAsItemID()
    targets = ["click"] >> nvt_ops.AddMetadata(tags=[str(Tags.BINARY_CLASSIFICATION), "target"])

    add_feat = [
        "user_item_categories",
        "user_item_shops",
        "user_item_brands",
        "user_item_intentions",
        "item_category",
        "item_shop",
        "item_brand",
    ] >> nvt_ops.Categorify()

    outputs = user_id + item_id + targets + add_feat

    workflow_fit_transform(
        outputs,
        glob(os.path.join(raw_data_path, "train", "train_*")),
        glob(os.path.join(raw_data_path, "test", "test_*")),
        str(output_path),
    )
