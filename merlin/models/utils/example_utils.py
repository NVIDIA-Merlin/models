import shutil
from pathlib import Path
from typing import List, Optional, Union

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


def plot_keras_history(
    history, keys: Optional[Union[str, List[str]]] = None, add_loss: bool = True
):
    import matplotlib.pyplot as plt

    if not keys:
        keys = list(history.history.keys())

    has_val = False
    if any(k.startswith("val_") for k in keys):
        has_val = True
        keys = [k for k in keys if not k.startswith("val_")]
        if not add_loss:
            keys = [k for k in keys if "loss" not in k]

    def _plot(key):
        plt.plot(history.history[key])
        if has_val:
            plt.plot(history.history[f"val_{key}"])
        plt.title(key)
        plt.ylabel(key)
        plt.xlabel("epoch")
        plt.legend(["train", "valid"] if has_val else ["train"], loc="upper left")
        plt.show()

    for key in keys:
        _plot(key)


def save_results(model_name, model):
    """a funct to save validation accuracy results in a text file"""
    with open("results.txt", "a") as f:
        f.write(model_name)
        f.write("\n")
        for key, value in model.history.history.items():
            if "val_auc" in key:
                f.write("%s:%s\n" % (key, value[0]))
