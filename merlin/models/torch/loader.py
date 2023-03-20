from typing import Optional, Union

from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.torch.typing import TabularData


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
