"""Tree-sitter grammars for the code strategies.

Grammars are downloaded ahead of time by `chunker warmup` (run during the Docker build) for the
languages in CODE_LANGUAGES. Requests for other languages are rejected rather than letting the
libraries download at request time: Chonkie's code chunker fetches all ~380 grammars (~500 MB)
when 19 or fewer are installed, and any missing grammar on demand.
"""

from collections.abc import Iterable

import tree_sitter_language_pack as tslp

from app.errors import NotConfiguredError

CHONKIE_MIN_GRAMMARS = 20


def installed() -> list[str]:
    return sorted(tslp.downloaded_languages())


def install(languages: Iterable[str]) -> None:
    tslp.download(list(languages))


def require(language: str, *, minimum_installed: int = 1) -> None:
    languages = installed()
    if language not in languages:
        available = ", ".join(languages) or "none"
        raise NotConfiguredError(
            f"No {language!r} grammar is installed on this server (installed: {available}). "
            "Add it to CODE_LANGUAGES and run `chunker warmup`."
        )
    if len(languages) < minimum_installed:
        raise NotConfiguredError(
            f"This strategy needs at least {minimum_installed} grammars installed; "
            "run `chunker warmup` with the default CODE_LANGUAGES."
        )
