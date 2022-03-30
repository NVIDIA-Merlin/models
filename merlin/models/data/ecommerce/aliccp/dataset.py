import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from merlin.core.dispatch import get_lib
from merlin.schema import Tags


def convert_alliccp(
    data_dir: Union[str, Path],
    convert_train: bool = True,
    convert_test: bool = True,
    file_size: int = 10000,
    max_num_rows: Optional[int] = None,
    pickle_common_features=True,
):
    """
    Convert Ali-CPP data to parquet files.

    To download the raw the Ali-CCP training and test datasets visit
    [tianchi.aliyun.com](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408#1).

    Parameters
    ----------
    data_dir:
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
        )
    if convert_test:
        _convert_data(
            str(data_dir),
            file_size,
            is_train=False,
            max_num_rows=max_num_rows,
            pickle_common_features=pickle_common_features,
        )

    return data_dir


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


def _convert_data(
    data_dir, file_size, is_train=True, pickle_common_features=True, max_num_rows=None
):
    data_type = "train" if is_train else "test"

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

                index = int((i / file_size) - 1)
                df.to_parquet(
                    os.path.join(data_dir, data_type, f"{data_type}_{index}.parquet"),
                    overwrite=True,
                )
                current = []
