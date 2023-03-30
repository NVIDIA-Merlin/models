import numpy as np
import torch
from torch import nn

from merlin.core.dispatch import get_lib
from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.batch import ModelEncode
from merlin.models.torch.data import get_device
from merlin.schema import Schema


def batch_predict(
    module: nn.Module,
    module_output_schema: Schema,
    dataset: Dataset,
    batch_size: int,
    add_inputs: bool = True,
    index=None,
) -> Dataset:
    data_iterator_func = ModelEncode.create_data_iterator_func(Loader, batch_size=batch_size)
    encoder = ModelEncode(
        module,
        output_schema=module_output_schema,
        data_iterator_func=data_iterator_func,
        model_encode_func=model_encode,
    )

    output = encoder.encode_dataset(dataset, index=index, add_inputs=add_inputs)

    return output


def model_encode(model, batch):
    # TODO: How to handle list outputs?

    model.eval()
    model.to(get_device(batch[0]))
    with torch.no_grad():
        model_outputs = model(batch[0])

    # handle when the model outputs a NamedTuple
    if hasattr(model_outputs, "to_df"):
        return model_outputs.to_df()
    elif hasattr(model_outputs, "_asdict"):
        model_outputs = model_outputs._asdict()

    if isinstance(model_outputs, dict):
        return get_lib().DataFrame({key: encode_output(val) for key, val in model_outputs.items()})

    return encode_output(model_outputs)


def encode_output(output: torch.Tensor) -> np.ndarray:
    if len(output.shape) == 2 and output.shape[1] == 1:
        output = torch.squeeze(output)

    return output.cpu().numpy()
