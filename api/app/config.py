from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = 'llm-quality-api'
    app_env: str = 'dev'
    app_port: int = 8000

    database_url: str = 'postgresql+psycopg2://llm:llm@postgres:5432/llm_intel'
    redis_url: str = 'redis://redis:6379/0'

    mlflow_tracking_uri: str = 'http://mlflow:5000'
    mlflow_experiment_name: str = 'llm-quality-intel'
    langsmith_tracing: bool = True
    langsmith_project: str = 'llm-quality-intel'

    # Cost model defaults (USD per 1K tokens).
    input_cost_per_1k_tokens_usd: float = 0.0015
    output_cost_per_1k_tokens_usd: float = 0.0020


settings = Settings()
