from collections.abc import Iterator
from typing import Any

import pytest
import uvicorn

from app import __main__ as cli
from app.config import get_settings
from app.engines import grammars
from app.tokens import ENCODINGS


@pytest.fixture
def uvicorn_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append({"app": app, **kwargs}))
    return calls


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir("/")  # ignore any local .env file
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_serve_is_the_default_command(uvicorn_calls: list[dict[str, Any]]) -> None:
    cli.main([])

    [call] = uvicorn_calls
    assert call["app"] == "app.main:app"
    assert (call["host"], call["port"], call["workers"], call["reload"]) == (
        "127.0.0.1",
        8000,
        1,
        False,
    )


def test_serve_reads_environment(
    monkeypatch: pytest.MonkeyPatch, uvicorn_calls: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("WORKERS", "4")

    cli.main(["serve"])

    [call] = uvicorn_calls
    assert (call["host"], call["port"], call["workers"]) == ("0.0.0.0", 9000, 4)


def test_serve_flags_override_environment(
    monkeypatch: pytest.MonkeyPatch, uvicorn_calls: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("PORT", "9000")

    cli.main(["serve", "--port", "7000", "--reload"])

    [call] = uvicorn_calls
    assert (call["port"], call["reload"]) == (7000, True)


def test_warmup_loads_encodings_and_grammars(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []
    installed: list[list[str]] = []
    monkeypatch.setattr(cli, "get_encoding", loaded.append)
    monkeypatch.setattr(grammars, "install", lambda languages: installed.append(list(languages)))
    monkeypatch.setenv("CODE_LANGUAGES", "python, go")

    cli.main(["warmup"])

    assert loaded == list(ENCODINGS)
    assert installed == [["python", "go"]]


def test_warmup_reports_unknown_languages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_encoding", lambda name: None)
    monkeypatch.setenv("CODE_LANGUAGES", "not-a-language")

    with pytest.raises(SystemExit, match="CODE_LANGUAGES"):
        cli.main(["warmup"])
