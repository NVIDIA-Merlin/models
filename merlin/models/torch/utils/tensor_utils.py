import torch
from torch.utils.dlpack import to_dlpack

from merlin.core.dispatch import DataFrameType


def tensor_to_df(tensor: torch.Tensor, index=None, gpu=None) -> DataFrameType:
    if gpu is None:
        try:
            import cudf  # noqa: F401
            import cupy

            gpu = True
        except ImportError:
            gpu = False

    if gpu:
        tensor_cupy = cupy.fromDlpack(to_dlpack(torch.as_tensor(tensor)))
        df = cudf.DataFrame(tensor_cupy)
        df.columns = [str(col) for col in list(df.columns)]
        if not index:
            index = cudf.RangeIndex(0, tensor.shape[0])
        df.set_index(index)
    else:
        import pandas as pd

        df = pd.DataFrame(tensor.detach().numpy())
        df.columns = [str(col) for col in list(df.columns)]
        if not index:
            index = pd.RangeIndex(0, tensor.shape[0])
        df.set_index(index)

    return df
