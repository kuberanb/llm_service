

from pydantic_settings import BaseSettings,SettingsConfigDict

class settings(BaseSettings):

    OPEN_API_KEY: str

    PRIMARY_MODEL: str = "gpt-4.1-mini"
    FALLBACK_MODEL: str = "gpt-4.1-nano"

    REQUEST_TIMEOUT: int = 30

    MAX_RETRIES:int = 3

    REDIS_URL: str = "redis://localhost:6379"

    RATE_LIMIT: int = 10
    RATE_LIMIT_WINDOW: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = settings()       