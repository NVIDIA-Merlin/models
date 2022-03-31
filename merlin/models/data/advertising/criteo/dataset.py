import glob
import os
import shutil
from pathlib import Path
from typing import Union

import numpy as np
import nvtabular as nvt
from nvtabular import ops as nvt_ops

import merlin.io
from merlin.core.utils import download_file
from merlin.models.data import BASE_PATH
from merlin.schema import Tags


def get_criteo(path=None, num_days=2):
    if path is None:
        path = os.path.join(BASE_PATH, "criteo")

    raw_data_path = os.path.join(path, "orig")
    maybe_download(raw_data_path, num_days)

    variant_path = os.path.join(path, f"{num_days}-days")
    variant_path_fn = lambda x: os.path.join(variant_path, x)  # noqa
    if not os.path.exists(variant_path):
        os.makedirs(variant_path_fn("raw"))
        prepare_criteo(raw_data_path, variant_path_fn("raw"), num_days)

    if not (os.path.exists(variant_path_fn("train")) and os.path.exists(variant_path_fn("valid"))):
        transform_criteo(variant_path_fn("raw"), variant_path)

    train = merlin.io.Dataset(os.path.join(variant_path, "train"), engine="parquet")
    valid = merlin.io.Dataset(os.path.join(variant_path, "valid"), engine="parquet")

    return train, valid


def maybe_download(destination: Union[str, Path], num_days: int):
    if num_days < 2 or num_days > 23:
        raise ValueError(
            str(num_days)
            + " is not supported. A minimum of 2 days are "
            + "required and a maximum of 24 (0-23 days) are available"
        )

    # Create input dir if not exists
    if not os.path.exists(destination):
        os.makedirs(destination)

    # Iterate over days
    for i in range(0, num_days):
        file = os.path.join(destination, "day_" + str(i) + ".gz")
        # Download file, if there is no .gz, .csv or .parquet file
        if not (
            os.path.exists(file)
            or os.path.exists(
                file.replace(".gz", ".parquet").replace("crit_orig", "converted/criteo/")
            )
            or os.path.exists(file.replace(".gz", ""))
        ):
            download_file(
                "http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_"
                + str(i)
                + ".gz",
                file,
            )


def prepare_criteo(
    raw_data_path: Union[str, Path],
    output_path: Union[str, Path],
    num_days: int,
    part_mem_fraction=0.1,
    client=None,
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Specify column names
    cont_names = ["I" + str(x) for x in range(1, 14)]
    cat_names = ["C" + str(x) for x in range(1, 27)]
    cols = ["label"] + cont_names + cat_names

    # Specify column dtypes. Note that "hex" means that
    # the values will be hexadecimal strings that should
    # be converted to int32
    dtypes = {}
    dtypes["label"] = np.int32
    for x in cont_names:
        dtypes[x] = np.int32
    for x in cat_names:
        dtypes[x] = "hex"

    # Create an NVTabular Dataset from a CSV-file glob
    file_list = glob.glob(os.path.join(str(raw_data_path), "day_*[!.gz]"))
    file_list = sorted(file_list)[:num_days]
    dataset = nvt.Dataset(
        file_list,
        engine="csv",
        names=cols,
        part_mem_fraction=part_mem_fraction,
        sep="\t",
        dtypes=dtypes,
        client=client,
    )

    dataset.to_parquet(str(output_path), preserve_files=True)

    return output_path


def transform_criteo(
    raw_data_path: Union[str, Path], output_path: Union[str, Path], num_buckets=10000000
):
    continuous = ["I" + str(x) for x in range(1, 14)]
    categorical = ["C" + str(x) for x in list(range(1, 21)) + list(range(22, 27))]
    # It's a reasonable guess that C21 is the item-id col.
    # This is only used to generate synthetic data
    item_id = ["C21"] >> nvt_ops.AddMetadata(tags=[Tags.ITEM_ID])
    targets = ["label"] >> nvt_ops.AddMetadata(tags=["target", Tags.BINARY_CLASSIFICATION])

    categorify_op = nvt_ops.Categorify(max_size=num_buckets)
    cat_features = categorical + item_id >> categorify_op
    cont_features = (
        continuous >> nvt_ops.FillMissing() >> nvt_ops.Clip(min_value=0) >> nvt_ops.Normalize()
    )
    features = cat_features + cont_features + targets

    workflow = nvt.Workflow(features)

    file_list = glob.glob(os.path.join(str(raw_data_path), "day_*[!.gz]"))
    train_dataset = nvt.Dataset(file_list[:-1])
    valid_dataset = nvt.Dataset(file_list[-1])

    if os.path.exists(os.path.join(output_path, "train")):
        shutil.rmtree(os.path.join(output_path, "train"))
    if os.path.exists(os.path.join(output_path, "valid")):
        shutil.rmtree(os.path.join(output_path, "valid"))

    workflow.fit(train_dataset)
    workflow.transform(train_dataset).to_parquet(
        output_path=os.path.join(output_path, "train"),
        out_files_per_proc=1,
        shuffle=False,
    )
    workflow.transform(valid_dataset).to_parquet(
        output_path=os.path.join(output_path, "valid"),
        out_files_per_proc=1,
        shuffle=False,
    )
    workflow.save(os.path.join(output_path, "workflow"))

    return output_path
