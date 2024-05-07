from typing import List

from merlin.models.tokenizers.tokenizer import Tokenizer


class SentencePieceTokenizer(Tokenizer):
    """Tokenizer using SentencePiece [1].

    References
    ----------
    [1] https://github.com/google/sentencepiece
    """

    def __init__(self, *, processor: "SentencePieceTrainer") -> None:  # noqa: F821
        require_sentencepiece()

        self.processor = processor
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        self.pad_id = self.processor.pad_id()

    def encode(
        self,
        string: str,
        bos: bool = False,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
    ) -> List[int]:
        tokens = self.processor.encode(string)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self.processor.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()


def require_sentencepiece() -> None:
    try:
        from sentencepiece import SentencePieceProcessor, SentencePieceTrainer  # noqa: F401
    except ImportError:
        raise ImportError(
            "This requires `sentencepiece`. Install it with `pip install sentencepiece`."
        )
