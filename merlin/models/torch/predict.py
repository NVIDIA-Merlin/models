import re
from typing import Dict, Optional, Union, overload

import numpy as np
import pandas as pd
import torch
from torch import nn

import merlin.table
from merlin.core.dispatch import DataFrameType, concat_columns, get_lib
from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.predict import ModelEncode
from merlin.models.torch.batch import Batch, Sequence
from merlin.schema import Schema
from merlin.table import TensorTable

OUT_KEY = "output"


class EncodedBatch(Batch):
    def __init__(
        self,
        features: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
        predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
        sequences: Optional[Sequence] = None,
    ):
        super().__init__(features, targets, sequences)

        default_key = "output"

        if isinstance(predictions, torch.Tensor):
            _predictions = {default_key: predictions}
        elif torch.jit.isinstance(predictions, Dict[str, torch.Tensor]):
            _predictions = predictions
        else:
            raise ValueError("Predictions must be a tensor or a dictionary of tensors")

        self.predictions: Dict[str, torch.Tensor] = _predictions

        # TODO: Add to_tables etc.

    def to_df(self):
        if len(self.features) == 1:
            output_dict = {"feature": self.features["default"]}
        else:
            output_dict = {**self.features}

        if len(self.targets) == 1:
            output_dict.update({"target": self.targets["default"]})
        else:
            output_dict.update(self.targets)

        if len(self.predictions) == 1:
            output_dict.update(self.predictions)
        else:
            output_dict.update({f"output_{key}": val for key, val in self.predictions.items()})

        return TensorTable(output_dict).to_df()


def encode(module: nn.Module, loader: Loader, index=None):  # Of type Selection
    dataset = loader.dataset.to_ddf()
    output_df = dataset.map_partitions(Encoder(module), batch_size=loader.batch_size)


class Predictor:
    def __init__(self, module: nn.Module):
        self.module = module

    @classmethod
    def from_loader(cls, module, loader):
        dataset = loader.dataset.to_ddf()

        predictor = cls(module)
        output_df = dataset.map_partitions(predictor.batched, batch_size=loader.batch_size)

        return output_df

    def __call__(self, inputs, targets=None) -> EncodedBatch:
        if not isinstance(inputs, dict):
            ...

        predictions = self.module(inputs)

        return EncodedBatch(inputs, targets, predictions)

    def call_df(self, partition, batch_size):
        output_df = None
        for batch in Loader(partition, batch_size=batch_size):
            if output_df is None:
                output_df = self(batch).to_df()
            else:
                output_df = concat_columns([output_df, self(batch).to_df()])

        return output_df


class Encoder:
    def __init__(self, module):
        self.module = module

    def __call__(self, partition, batch_size=None):
        if not batch_size:
            return self.forward(partition)

        outputs = []
        for batch in Loader(partition, batch_size=batch_size):
            outputs.append(self.forward(batch))

        output_df = concat_columns(outputs)

        # TODO: Add index column
        # TODO: Optionally add input columns

        return output_df

    def forward(self, batch):
        out = self.module(batch[0])

        if isinstance(out, dict):
            table = TensorTable(out)
        else:
            table = TensorTable({OUT_KEY: out})

        return table.to_df()


# def partition_mapper(partition):
#     outputs = []
#     part_loader = Loader(partition, batch_size=loader.batch_size)

#     for batch in part_loader:
#         out = module(batch[0])

#         if isinstance(out, dict):
#             table = TensorTable(out)
#         else:
#             table = TensorTable({OUT_KEY: out})

#         outputs.append(table.to_df())

#     output_df = concat_columns(outputs)

#     # TODO: Add index column
#     # TODO: Optionally add input columns

#     return output_df


class ModuleEncoder:
    def __init__(self, module: nn.Module, loader: Loader):
        self.module = module
        self.loader = loader


class TabularModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_forward_pre_hook(self._hook)


# loader = Loader(..., batch_size=128)


# model(ddf)
# model(df)

# BatchPredictor(model)(loader)


def batch_predict(
    module: nn.Module,
    dataset: Dataset,
    batch_size: int,
    module_output_schema: Optional[Schema] = None,
    add_inputs: bool = True,
    index=None,
) -> Dataset:
    """
    Predict the outputs for a given dataset in batches using a PyTorch module (model).

    Parameters
    ----------
    module : nn.Module
        The PyTorch module (model) to be used for prediction.
    dataset : Dataset
        The dataset to be predicted.
    batch_size : int
        The size of the batches to be used for prediction.
    module_output_schema : Optional[Schema], default None
        The schema of the output produced by the module. If None, the default schema is used.
    add_inputs : bool, default True
        Whether to concatenate the input dataset with the module's output.
    index : str or None, default None
        The name of the column in the input dataset to be added to the output dataframe.

    Returns
    -------
    Dataset
        The output dataset after prediction.
    """

    if hasattr(module, "output_schema"):
        module_output_schema = module.output_schema
    elif module_output_schema is None:
        raise ValueError(
            "module_output_schema must be provided if the model does not have an output_schema attribute."
        )

    data_iterator_func = ModelEncode.create_data_iterator_func(Loader, batch_size=batch_size)
    encoder = ModelEncode(
        module,
        output_schema=module_output_schema,
        data_iterator_func=data_iterator_func,
        model_encode_func=module_encode,
    )

    output = encoder.encode_dataset(dataset, index=index, add_inputs=add_inputs)

    return output


def module_encode(
    module: nn.Module,
    inputs,
) -> DataFrameType:
    """Encode the inputs using the model.

    This function handles the cases when the model outputs a NamedTuple or a dict.

    Parameters
    ----------
    module : nn.Module
        The PyTorch model to be used for encoding.
    inputs : iterable
        The inputs to be encoded.

    Returns
    -------
    DataFrameType
        The encoded output in the form of a DataFrame.

    Raises
    ------
    ValueError
        If the model outputs a type that is not handled.
    """

    # TODO: How to handle list outputs?

    features = inputs[0] if isinstance(inputs, tuple) else inputs
    batch = Batch(*inputs) if isinstance(inputs, tuple) else Batch(inputs)

    module.eval()
    module.to(batch.device())
    with torch.no_grad():
        model_outputs = module(features)

    # handle when the model outputs a NamedTuple
    if hasattr(model_outputs, "to_df"):
        return model_outputs.to_df()
    elif hasattr(model_outputs, "_asdict"):
        model_outputs = model_outputs._asdict()

    if isinstance(model_outputs, dict):
        return get_lib().DataFrame({key: encode_output(val) for key, val in model_outputs.items()})

    return encode_output(model_outputs)


def encode_output(output: torch.Tensor) -> np.ndarray:
    """
    Convert the output tensor from a PyTorch model to a numpy array.
    This function handles the case when the output tensor has a
    shape of (N, 1) by squeezing it to (N,).

    Parameters
    ----------
    output : torch.Tensor
        The output tensor from a PyTorch model.

    Returns
    -------
    np.ndarray
        The output converted to a numpy array.
    """

    if len(output.shape) == 2 and output.shape[1] == 1:
        output = torch.squeeze(output)

    return output.cpu().numpy()
