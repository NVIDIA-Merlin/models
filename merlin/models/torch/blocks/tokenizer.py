import os
from pathlib import Path
from typing import Optional

import torch
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


class SentencePieceTokenizer:
    """Tokenizer for LLaMA.

    Example usage
    -------------
    >> tokenizer_path = Path("llama/tokenizer.model")
    >> tokenizer = SentencePieceTokenizer(tokenizer_path)
    >> tokenizer.encode("Hello, my name is", bos=True, eos=False)
    tensor([    1, 15043, 29892,   590,  1024,   338], dtype=torch.int32)
    """

    def __init__(self, model_path: Path) -> None:
        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            raise ImportError(
                "`sentencepiece` is required to use this feature. "
                "Install it with `pip install sentencepiece`."
            )

        self.processor = SentencePieceProcessor(model_file=str(model_path))
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        self.pad_id = self.processor.pad_id()

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(
        self,
        string: str,
        bos: bool = True,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        tokens = self.processor.encode(string)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.processor.decode(tokens.tolist())

    @staticmethod
    def train(input: str, destination: str, vocab_size=32000) -> None:
        model_prefix = os.path.join(destination, "tokenizer")
        SentencePieceTrainer.Train(input=input, model_prefix=model_prefix, vocab_size=vocab_size)
