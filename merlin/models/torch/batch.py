#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Dict, Optional, Union

import torch

from merlin.dataloader.torch import Loader
from merlin.io import Dataset


@torch.jit.script
class Sequence:
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
        >>> seq = Sequence(lengths, masks)
    """

    def __init__(
        self,
        lengths: Union[torch.Tensor, Dict[str, torch.Tensor]],
        masks: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
    ):
        if isinstance(lengths, torch.Tensor):
            _lengths = {"default": lengths}
        elif torch.jit.isinstance(lengths, Dict[str, torch.Tensor]):
            _lengths = lengths
        else:
            raise ValueError("Lengths must be a tensor or a dictionary of tensors")
        self.lengths: Dict[str, torch.Tensor] = _lengths

        if masks is None:
            _masks = {}
        elif isinstance(masks, torch.Tensor):
            _masks = {"default": masks}
        elif torch.jit.isinstance(masks, Dict[str, torch.Tensor]):
            _masks = masks
        else:
            raise ValueError("Masks must be a tensor or a dictionary of tensors")
        self.masks: Dict[str, torch.Tensor] = _masks

    def __contains__(self, name: str) -> bool:
        return name in self.lengths

    def length(self, name: str = "default") -> torch.Tensor:
        """Retrieves a length tensor from a sequence by name.

        Args:
            name (str, optional): The name of the feature. Defaults to "default".

        Returns:
            torch.Tensor: The length tensor of the specified feature.

        Raises:
            ValueError: If the Sequence object has multiple lengths and
                no feature name is specified.
        """

        if name in self.lengths:
            return self.lengths[name]

        raise ValueError("Batch has multiple lengths, please specify a feature name")

    def mask(self, name: str = "default") -> torch.Tensor:
        """Retrieves a mask tensor from a sequence by name.

        Args:
            name (str, optional): The name of the feature. Defaults to "default".

        Returns:
            torch.Tensor: The mask tensor of the specified feature.

        Raises:
            ValueError: If the Sequence object has multiple masks and
                no feature name is specified.
        """
        if name in self.masks:
            return self.masks[name]

        raise ValueError("Batch has multiple masks, please specify a feature name")

    def device(self) -> torch.device:
        """Retrieves the device of the tensors in the Sequence object.

        Returns:
            torch.device: The device of the tensors.

        Raises:
            ValueError: If the Sequence object is empty.
        """
        for d in self.lengths.values():
            if isinstance(d, torch.Tensor):
                return d.device

        raise ValueError("Sequence is empty")


@torch.jit.script
class Batch:
    """
    A PyTorch scriptable class representing a batch of data.

    Attributes:
        features (Dict[str, torch.Tensor]): A dictionary mapping feature names to their
            corresponding feature values.
        targets (Dict[str, torch.Tensor]): A dictionary mapping target names to their
            corresponding target values. Default is an empty dictionary.
        sequences (Optional[Sequence]): An optional instance of the Sequence class
            representing sequence lengths and masks for the batch.

    Examples:
        >>> features = {'feature1': torch.tensor([1, 2]), 'feature2': torch.tensor([3, 4])}
        >>> targets = {'target1': torch.tensor([0, 1])}
        >>> batch = Batch(features, targets)
    """

    def __init__(
        self,
        features: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        sequences: Optional[Sequence] = None,
    ):
        default_key = "default"

        if isinstance(features, torch.Tensor):
            _features = {default_key: features}
        elif torch.jit.isinstance(features, Dict[str, torch.Tensor]):
            _features = features
        else:
            raise ValueError("Features must be a tensor or a dictionary of tensors")

        self.features: Dict[str, torch.Tensor] = _features

        if isinstance(targets, torch.Tensor):
            targets = {default_key: targets}

        if targets is None:
            _targets = {}
        elif torch.jit.isinstance(targets, Dict[str, torch.Tensor]):
            _targets = targets
        else:
            raise ValueError("Targets must be a tensor or a dictionary of tensors")
        self.targets: Dict[str, torch.Tensor] = _targets
        self.sequences: Optional[Sequence] = sequences

    @staticmethod
    @torch.jit.ignore
    def sample_from(
        dataset_or_loader: Union[Dataset, Loader],
        batch_size: int = 32,
        shuffle: Optional[bool] = False,
    ) -> "Batch":
        """Sample a batch from a dataset or a loader.

        Example usage::
            dataset = merlin.io.Dataset(...)
            batch = Batch.sample_from(dataset)

        Parameters
        ----------
        dataset_or_loader: merlin.io.dataset
            A Dataset object or a Loader object.
        batch_size: int, default=32
            Number of samples to return.
        shuffle: bool
            Whether to sample a random batch or not, by default False.

        Returns:
        -------
        features: Dict[torch.Tensor]
            dictionary of feature tensors.
        targets: Dict[torch.Tensor]
            dictionary of target tensors.
        """

        return sample_batch(dataset_or_loader, batch_size, shuffle)

    def replace(
        self,
        features: Optional[Dict[str, torch.Tensor]] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        sequences: Optional[Sequence] = None,
    ) -> "Batch":
        """
        Create a new `Batch` instance, replacing specified attributes with new values.

        Parameters
        ----------
        features : Optional[Dict[str, torch.Tensor]]
            A dictionary of tensors representing the features of the batch. Default is None.
        targets : Optional[Dict[str, torch.Tensor]]
            A dictionary of tensors representing the targets of the batch. Default is None.
        sequences : Optional[Sequence]
            An instance of the Sequence class representing sequence lengths and masks for the
            batch. Default is None.

        Returns
        -------
        Batch
            A new Batch object with replaced attributes.
        """

        return Batch(
            features=features if features is not None else self.features,
            targets=targets if targets is not None else self.targets,
            sequences=sequences if sequences is not None else self.sequences,
        )

    def feature(self, name: str = "default") -> torch.Tensor:
        """Retrieve a feature tensor from the batch by name.

        Parameters
        ----------
        name : str
            The name of the feature tensor to return. Default is "default".

        Returns
        -------
        torch.Tensor
            The feature tensor of the specified name.

        Raises
        ------
        ValueError
            If the specified name does not exist in the features attribute.
        """

        if name in self.features:
            return self.features[name]

        raise ValueError("Batch has multiple features, please specify a feature name")

    def target(self, name: str = "default") -> torch.Tensor:
        """Retrieve a target tensor from the batch by name.

        Parameters
        ----------
        name : str
            The name of the target tensor to return. Default is "default".

        Returns
        -------
        torch.Tensor
            The target tensor of the specified name.

        Raises
        ------
        ValueError
            If the specified name does not exist in the targets attribute.
        """

        if name in self.targets:
            return self.targets[name]

        raise ValueError("Batch has multiple target, please specify a target name")

    def __bool__(self) -> bool:
        return bool(self.features)

    def device(self) -> torch.device:
        """Retrieves the device of the tensors in the Batch object.

        Returns:
            torch.device: The device of the tensors.

        Raises:
            ValueError: If the Batch object is empty.
        """
        for d in self.features.values():
            if isinstance(d, torch.Tensor):
                return d.device

        raise ValueError("Batch is empty")


def sample_batch(
    data: Union[Dataset, Loader],
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = False,
) -> Batch:
    """Util function to generate a batch of input tensors from a merlin.io.Dataset instance

    Parameters
    ----------
    data: merlin.io.dataset
        A Dataset object.
    batch_size: int
        Number of samples to return.
    shuffle: bool
        Whether to sample a random batch or not, by default False.

    Returns
    -------
    features: Dict[torch.Tensor]
        dictionary of feature tensors.
    targets: Dict[torch.Tensor]
        dictionary of target tensors.
    """

    if isinstance(data, Dataset):
        if not batch_size:
            raise ValueError("Either use 'Loader' or specify 'batch_size'")
        loader = Loader(data, batch_size=batch_size, shuffle=shuffle)
    elif isinstance(data, Loader):
        loader = data
    else:
        raise ValueError(f"Expected Dataset or Loader instance, got: {data}")

    batch = loader.peek()
    # batch could be of type Prediction, so we can't unpack directly
    inputs, targets = batch[0], batch[1]

    return Batch(inputs, targets)


def sample_features(
    data: Union[Dataset, Loader],
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = False,
) -> Dict[str, torch.Tensor]:
    """Util function to generate a dict of feature tensors from a merlin.io.Dataset instance

    Parameters
    ----------
    data: merlin.io.dataset
        A Dataset object.
    batch_size: int
        Number of samples to return.
    shuffle: bool
        Whether to sample a random batch or not, by default False.

    Returns
    -------
    features: Dict[torch.Tensor]
        dictionary of feature tensors.
    """

    return sample_batch(data, batch_size, shuffle).features
