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


def save_results(model_name, model):
    """a funct to save validation accuracy results in a text file"""
    with open("results.txt", "a") as f:
        f.write(model_name)
        f.write("\n")
        for key, value in model.history.history.items():
            if "val_auc" in key:
                f.write("%s:%s\n" % (key, value[0]))
