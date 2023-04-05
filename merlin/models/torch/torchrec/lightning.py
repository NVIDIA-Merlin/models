import logging
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.utilities.types import _PATH

logger: logging.Logger = logging.getLogger(__name__)


class TorchrecStrategy(ParallelStrategy):
    """
    Lightning Trainer takes care of the operations that are related to DDP.
    However, our models are parallelization aware, which are not fully compatible to the
    given accelerators and strategies provided by Lightning.
    The torchrec accelerator and strategies bypasses the corresponding logic in Lightning.

    Credits:
    https://github.com/facebookresearch/recipes/blob/main/torchrecipes/rec/accelerators/torchrec.py
    """

    def __init__(self) -> None:
        super().__init__()
        logger.info("Creating torchrec strategy")

    def broadcast(self, obj: object, src: int = 0) -> object:
        if dist.is_initialized:
            if isinstance(obj, torch.Tensor):
                dist.broadcast(obj, src)
                return obj
            else:
                object_list = [obj]
                dist.broadcast_object_list(object_list=object_list, src=src)
                return object_list[0]
        else:
            raise AssertionError(
                "Broadcast called in torchrec strategy w/o initializing distributed"
            )

    @property
    def root_device(self) -> torch.device:
        rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")
        return device

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        filepath: _PATH,
        # pyre-ignore[2]: Parameter `storage_options` has type `None` but type `Any` is specified.
        storage_options: Optional[Any] = None,
    ) -> None:
        self.checkpoint_io.save_checkpoint(
            checkpoint=checkpoint, path=filepath, storage_options=storage_options
        )

    # pyre-ignore[3]
    def batch_to_device(
        self,
        # pyre-ignore[2]
        batch: Any,
        device: Optional[torch.device] = None,
        dataloader_idx: Optional[int] = None,
    ) -> Any:
        if self.lightning_module:
            return batch.to(self.lightning_module.device)

    def barrier(self, name: Optional[str] = None) -> None:
        if dist.is_initialized:
            dist.barrier()
        else:
            raise AssertionError(
                "All gather called in torchrec strategy w/o initializing distributed"
            )

    def all_gather(
        self,
        tensor: torch.Tensor,
        # pyre-ignore[2]: Parameter `group` has type `None` but type `Any` is specified.
        group: Optional[Any] = None,
        sync_grads: bool = False,
    ) -> torch.Tensor:
        if dist.is_initialized:
            dist.all_gather(tensor, group, sync_grads)
            return tensor
        else:
            raise AssertionError(
                "All gather called in torchrec strategy w/o initializing distributed"
            )

    # pyre-ignore[3]: Return type must be specified as type other than `Any`.
    def reduce(
        self,
        # pyre-ignore[2]: Parameter `tensor` must have a type other than `Any`.
        tensor: Union[Any, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Union[Any, torch.Tensor]:
        if dist.is_initialized:
            dist.all_reduce(tensor)
            return tensor
        else:
            raise AssertionError("Reduce called in torchrec strategy w/o initializing distributed")

    def model_to_device(self) -> None:
        pass

    def teardown(self) -> None:
        return None
