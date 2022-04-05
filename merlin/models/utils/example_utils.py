import shutil
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from merlin.io import Dataset


def workflow_fit_transform(
    outputs,
    train: Union[str, Path, List[str], Dataset],
    valid: Union[str, Path, List[str], Dataset],
    output_path: Union[str, Path],
    workflow_name: Optional[str] = None,
    save_workflow: bool = True,
):
    """fits and transforms datasets applying NVT workflow"""
    import nvtabular as nvt

    if not isinstance(outputs, nvt.Workflow):
        workflow = nvt.Workflow(outputs)
    else:
        workflow = outputs

    _train = train if isinstance(train, Dataset) else Dataset(train)
    _valid = valid if isinstance(valid, Dataset) else Dataset(valid)

    train_path, valid_path = Path(output_path) / "train", Path(output_path) / "valid"
    if train_path.exists():
        shutil.rmtree(train_path)
    if valid_path.exists():
        shutil.rmtree(valid_path)

    workflow.fit(_train)
    workflow.transform(_train).to_parquet(str(train_path))
    workflow.transform(_valid).to_parquet(str(valid_path))

    if save_workflow:
        _name = Path(output_path) / workflow_name if workflow_name else "workflow"
        workflow.save(str(_name))


def save_results(model_name, model):
    """a funct to save valudation accurracy results in a text file"""
    with open("results.txt", "a") as f:
        f.write(model_name)
        f.write("\n")
        for key, value in model.history.history.items():
            if "val_auc" in key:
                f.write("%s:%s\n" % (key, value[0]))


def create_bar_chart(text_file_name, models_name):
    import matplotlib.pyplot as plt

    """a func to plot barcharts via parsing the  accurracy results in a text file"""
    auc = []
    with open(text_file_name, "r") as infile:
        for line in infile:
            if "auc" in line:
                data = [line.rstrip().split(":")]
                key, value = zip(*data)
                auc.append(float(value[0]))

    X_axis = np.arange(len(models_name))

    plt.title("Models' accuracy metrics comparison", pad=20)
    plt.bar(X_axis - 0.2, auc, 0.4, label="AUC")

    plt.xticks(X_axis, models_name)
    plt.xlabel("Models")
    plt.ylabel("AUC")
    plt.show()
