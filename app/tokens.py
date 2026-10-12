from functools import cache
from typing import Literal, get_args

import tiktoken

Encoding = Literal["cl100k_base", "o200k_base", "p50k_base", "r50k_base"]
ENCODINGS: tuple[str, ...] = get_args(Encoding)
DEFAULT_ENCODING: Encoding = "cl100k_base"


@cache
def get_encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding: str) -> int:
    # disallowed_special=() treats strings like "<|endoftext|>" as plain text instead of raising.
    return len(get_encoding(encoding).encode(text, disallowed_special=()))
