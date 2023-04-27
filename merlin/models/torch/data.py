from typing import Dict, Optional, Union

import torch

from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.torch.typing import TabularData
from merlin.schema import Schema


def sample_batch(
    dataset_or_loader: Union[Dataset, Loader],
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = False,
    include_targets: Optional[bool] = True,
) -> TabularData:
    """Util function to generate a batch of input tensors from a merlin.io.Dataset instance

    Parameters
    ----------
    data: merlin.io.dataset
        A Dataset object.
    batch_size: int
        Number of samples to return.
    shuffle: bool
        Whether to sample a random batch or not, by default False.
    include_targets: bool
        Whether to include the targets in the returned batch, by default True.

    Returns:
    -------
    batch: Dict[torch.Tensor]
        dictionary of input tensors.
    """

    if isinstance(dataset_or_loader, Dataset):
        if not batch_size:
            raise ValueError("Either use 'Loader' or specify 'batch_size'")
        loader = Loader(dataset_or_loader, batch_size=batch_size, shuffle=shuffle)
    else:
        loader = dataset_or_loader

    batch = loader.peek()
    # batch could be of type Prediction, so we can't unpack directly
    inputs, targets = batch[0], batch[1]

    if not include_targets:
        return inputs

    return inputs, targets


def get_device(data):
    if isinstance(data, torch.Tensor):
        device = data.device
    elif isinstance(data, tuple):
        device = data[0].device
    elif isinstance(data, dict):
        for d in data.values():
            if isinstance(d, torch.Tensor):
                device = d.device
                break
    else:
        raise ValueError(f"Unsupported data type {type(data)}")

    return device


def initialize(module, data: Loader):
    if isinstance(data, (Loader, Dataset)):
        module.double()  # TODO: Put in data-loader PR to standardize on float-32
        batch = sample_batch(data, batch_size=1, shuffle=False, include_targets=False)
    else:
        batch = data

    module.to(get_device(batch))
    return module(batch)


@torch.jit.script
class TabularSequence:
    """
    A PyTorch scriptable class representing a sequence of tabular data.

    Attributes:
        lengths (Dict[str, torch.Tensor]): A dictionary mapping the feature names to their
            corresponding sequence lengths.
        masks (Dict[str, torch.Tensor]): A dictionary mapping the feature names to their
            corresponding masks. Default is an empty dictionary.

    Examples:
        >>> lengths = {'feature1': torch.tensor([4, 5]), 'feature2': torch.tensor([3, 7])}
        >>> masks = {'feature1': torch.tensor([[1, 0], [1, 1]]), 'feature2': torch.tensor([[1, 1], [1, 0]])}    # noqa: E501
        >>> tab_seq = TabularSequence(lengths, masks)
    """

    def __init__(
        self,
        lengths: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.lengths: Dict[str, torch.Tensor] = lengths
        _masks = {}
        if masks is not None:
            _masks = masks
        self.masks: Dict[str, torch.Tensor] = _masks

    def __contains__(self, name: str) -> bool:
        return name in self.lengths


@torch.jit.script
class TabularBatch:
    """
    A PyTorch scriptable class representing a batch of tabular data.

    Attributes:
        features (Dict[str, torch.Tensor]): A dictionary mapping feature names to their
            corresponding feature values.
        targets (Dict[str, torch.Tensor]): A dictionary mapping target names to their
            corresponding target values. Default is an empty dictionary.
        sequences (Optional[TabularSequence]): An optional instance of the TabularSequence class
            representing sequence lengths and masks for the batch.

    Examples:
        >>> features = {'feature1': torch.tensor([1, 2]), 'feature2': torch.tensor([3, 4])}
        >>> targets = {'target1': torch.tensor([0, 1])}
        >>> tab_batch = TabularBatch(features, targets)
    """

    def __init__(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
        sequences: Optional[TabularSequence] = None,
    ):
        self.features: Dict[str, torch.Tensor] = features
        if targets is None:
            _targets = {}
        else:
            _targets = targets
        self.targets: Dict[str, torch.Tensor] = _targets
        self.sequences: Optional[TabularSequence] = sequences

    def replace(
        self,
        features: Optional[Dict[str, torch.Tensor]] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        sequences: Optional[TabularSequence] = None,
    ) -> "TabularBatch":
        return TabularBatch(
            features=features if features is not None else self.features,
            targets=targets if targets is not None else self.targets,
            sequences=sequences if sequences is not None else self.sequences,
        )

    # def select(self, schema: Schema) -> "TabularBatch":
    #     _features = {}
    #     col_names = schema.column_names

    #     for name, val in self.features.items():
    #         if name in col_names:
    #             _features[name] = val

    #     return TabularBatch(
    #         _features, self.targets, self.sequences
    #     )

    def __bool__(self) -> bool:
        return bool(self.features)
