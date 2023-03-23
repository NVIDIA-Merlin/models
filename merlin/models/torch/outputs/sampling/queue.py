from typing import Sequence

import torch
from torch import nn


class FIFOQueue(nn.Module):
    def __init__(
        self,
        capacity: int,
        dims: Sequence[int] = [],
        persistent: bool = False,
    ):
        super().__init__()

        self.shape = [capacity] + list(dims) if dims else capacity
        self.capacity = capacity
        self.dims = dims
        self.persistent = persistent

        self.reset()

    def reset(self):
        self._upsert_buffer("storage", torch.zeros(0), persistent=self.persistent)
        self._upsert_buffer("first_ptr", torch.tensor(0), persistent=self.persistent)
        self._upsert_buffer("next_ptr", torch.tensor(0), persistent=self.persistent)
        self._upsert_buffer("empty", torch.tensor(True), persistent=self.persistent)
        self._upsert_buffer("full", torch.tensor(False), persistent=self.persistent)

    def enqueue(self, x: torch.Tensor):
        if len(x.shape) == 1:
            if not list(x.shape) == self.shape[-1:]:
                raise RuntimeError(
                    "Invalid shape for enqueueing. "
                    + f"Expected shape: {self.dims}, received shape: {list(x.shape)}"
                )
            num_vals = 1
        else:
            if not list(x.shape[1:]) == self.shape[-len(self.shape) + 1 :]:
                raise RuntimeError(
                    "Invalid shape for enqueueing. ",
                    f"Expected shape: {self.shape[-len(self.shape) + 1:]}, ",
                    f"received shape: {list(x.shape[1:])}",
                )
            num_vals = int(x.shape[0])
            if num_vals > self.capacity:
                x = x[-self.capacity :]
                num_vals = self.capacity

        next_start = self.next_ptr
        next_end = next_start + num_vals

        _x = x

        if next_end.item() > self.capacity:
            extra_items = next_end - self.capacity
            self.storage = torch.cat([self.storage, x[: num_vals - extra_items]])
            next_start = 0
            next_end = extra_items
            self.full = torch.tensor(True)
            _x = x[num_vals - extra_items :]

        if self.empty.item():
            self.storage = _x
            self.empty = torch.tensor(False)
        elif self.full.item():
            self.storage[next_start:next_end] = _x
        else:
            self.storage = torch.stack([self.storage, _x])

        if self.full.item() or (
            next_start.item() < self.first_ptr.item() and next_end.item() >= self.first_ptr.item()
        ):
            self.first_ptr = next_end
            self.full = torch.tensor(True)

        self.next_ptr = next_end

        return self

    def dequeue(self, n: int = 1) -> torch.Tensor:
        """Dequeues many examples from the queue

        Parameters
        ----------
        n : int
            Number of examples to sample from the queue

        Returns
        -------
        torch.Tensor
            A tensor with N examples, being the first dim equal to N

        Raises
        ------
        IndexError
            The queue is empty
        ValueError
            The number of elements to dequeue must be greater than 0
        """
        if self.empty.item():
            raise IndexError("The queue is empty")
        if n <= 0:
            raise ValueError("The number of elements to dequeue must be greater than 0.")

        _n = min(n, self.storage.shape[0])
        to_return = self.storage[:_n]
        if _n == self.capacity:
            self.reset()
        else:
            self.storage = self.storage[_n:]
            self.full = torch.tensor(False)

        return to_return

    def list_all(self) -> torch.Tensor:
        """
        Returns all items in the queue, sorted by the
        order they were added (FIFO)

        Returns
        -------
        torch.Tensor
            Returns a tensor with all examples added to the queue
        """
        if self.first_ptr < self.next_ptr:
            return self.storage[self.first_ptr : self.next_ptr]
        elif self.first_ptr == self.next_ptr and not self.full.item():
            return self.storage[0:0]  # Returns empty Tensor when queue is empty
        else:
            return torch.cat(
                [self.storage[self.first_ptr :], self.storage[: self.next_ptr]],
                dim=0,
            )

    @property
    def size(self) -> int:
        """Returns the number of examples added to the queue
        Returns
        -------
        int
            The number of examples added to the queue
        """
        if self.first_ptr < self.next_ptr:
            return self.next_ptr - self.first_ptr
        elif self.full.item():
            return self.capacity
        elif self.first_ptr == self.next_ptr:
            return 0
        else:
            return self.capacity - self.first_ptr + self.next_ptr

    def _upsert_buffer(self, name: str, value: torch.Tensor, persistent=False):
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            self.register_buffer(name, value, persistent=persistent)
