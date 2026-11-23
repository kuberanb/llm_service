from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    OPENAI_API_KEY: str

    PRIMARY_MODEL: str = "gpt-6.1-terra"
    FALLBACK_MODEL: str = "gpt-6-luna"

    LLM_TIMEOUT: float = 30

    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()