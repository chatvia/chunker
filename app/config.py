from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.webhooks import decode_secret

DEFAULT_CODE_LANGUAGES = (
    "bash", "c", "cpp", "csharp", "css", "dart", "go", "html", "java", "javascript", "json",
    "kotlin", "lua", "markdown", "php", "python", "ruby", "rust", "scala", "sql", "swift", "tsx",
    "typescript", "yaml",
)  # fmt: skip


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables and an optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: PositiveInt = 8000
    workers: PositiveInt = 1
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    api_key: SecretStr | None = Field(
        default=None, description="When set, /v1 endpoints require `Authorization: Bearer <key>`."
    )
    max_request_mb: PositiveInt = 50

    # Any OpenAI-compatible embeddings API: OpenAI, Mistral, Ollama, vLLM, TEI, LM Studio, ...
    embeddings_base_url: str = "https://api.openai.com/v1"
    embeddings_api_key: SecretStr | None = None
    embeddings_model: str | None = None
    embeddings_dimensions: PositiveInt | None = None
    embeddings_batch_size: PositiveInt = 64
    embeddings_timeout: PositiveFloat = 60.0
    embeddings_max_retries: NonNegativeInt = 2
    allow_request_embeddings: bool = Field(
        default=True,
        description="Accept `embeddings` in requests when EMBEDDINGS_MODEL is unset.",
    )

    code_languages: Annotated[list[str], NoDecode] = Field(
        default=list(DEFAULT_CODE_LANGUAGES),
        description="Tree-sitter grammars installed by `chunker warmup` for the code strategies.",
    )

    webhook_secret: SecretStr | None = Field(
        default=None,
        description="Standard Webhooks signing secret: base64, optionally prefixed with `whsec_`.",
    )
    webhook_timeout: PositiveFloat = 30.0
    webhook_max_attempts: PositiveInt = 3
    webhook_verify_tls: bool = True

    @field_validator("code_languages", mode="before")
    @classmethod
    def _split_languages(cls, value: object) -> object:
        if isinstance(value, str):
            return [language.strip() for language in value.split(",") if language.strip()]
        return value

    @field_validator("webhook_secret")
    @classmethod
    def _validate_webhook_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None:
            decode_secret(value.get_secret_value())
        return value

    @property
    def max_request_bytes(self) -> int:
        return self.max_request_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
