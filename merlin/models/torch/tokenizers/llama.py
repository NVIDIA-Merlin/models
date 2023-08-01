from pathlib import Path
from typing import Optional, Union

import torch

from merlin.models.tokenizers.sentencepiece import SentencePieceTokenizer, require_sentencepiece


class LlamaTokenizer(SentencePieceTokenizer):
    def __init__(self, path: Union[str, Path]) -> None:
        require_sentencepiece()

        from sentencepiece import SentencePieceProcessor

        if isinstance(path, Path):
            path = str(path)
        processor = SentencePieceProcessor(model_file=str(path))

        super().__init__(processor=processor)

    def endode(
        self,
        string: str,
        bos: bool = True,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        tokens = super().encode(
            string=string,
            bos=bos,
            eos=eos,
            max_length=max_length,
            pad=pad,
        )
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.processor.decode(tokens.tolist())
