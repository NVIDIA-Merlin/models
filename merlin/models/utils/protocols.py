from typing import Protocol, runtime_checkable


@runtime_checkable
class LookUpProtocol(Protocol):
    """Protocol for embedding lookup layers"""

    def embeddings(self):
        ...

    def embedding_lookup(self, inputs, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        ...
