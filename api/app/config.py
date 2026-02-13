from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = 'llm-quality-api'
    app_env: str = 'dev'
    app_port: int = 8000

    database_url: str = 'postgresql+psycopg2://llm:llm@postgres:5432/llm_intel'
    redis_url: str = 'redis://redis:6379/0'

    mlflow_tracking_uri: str = 'http://mlflow:5000'
    langsmith_tracing: bool = True


settings = Settings()
