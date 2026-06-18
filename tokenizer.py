"""Unified tokenizer interface over tiktoken and Hugging Face backends.

The rest of the codebase depends only on the :class:`Tokenizer` protocol and the
:func:`build_tokenizer` factory, so swapping a tokenizer is a config change.
"""

from typing import Protocol

from config import Config


class Tokenizer(Protocol):
    """Minimal interface tokenizer implementations must abide by."""

    eos_id: int
    vocab_size: int

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids without adding special tokens."""
        ...

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode many texts at once, parallelized across cores."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text."""
        ...


class _TiktokenTokenizer:
    def __init__(self, name: str) -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(name)
        self.eos_id = self._enc.encode_single_token("<|endoftext|>")
        self.vocab_size = self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self._enc.encode_ordinary_batch(texts)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


class _HuggingFaceTokenizer:
    def __init__(self, name: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(name)
        self.eos_id = self._tok.eos_token_id
        self.vocab_size = self._tok.vocab_size

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self._tok(texts, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


def build_tokenizer(config: Config) -> Tokenizer:
    """Build the tokenizer selected by ``config.tokenizer_backend``."""
    if config.tokenizer_backend == "tiktoken":
        return _TiktokenTokenizer(config.tokenizer_name)
    if config.tokenizer_backend == "huggingface":
        return _HuggingFaceTokenizer(config.tokenizer_name)
    raise ValueError(f"Unknown tokenizer_backend: {config.tokenizer_backend!r}")
