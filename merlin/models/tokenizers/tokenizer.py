from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    """
    Base class for all tokenizers.
    """

    def __call__(self, string: str):
        return self.encode(string)

    @abstractmethod
    def decode(self, tokens: List[int]):
        ...

    @abstractmethod
    def encode(self, string: str):
        ...
