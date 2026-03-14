from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "admin"
    postgres_password: str = "password123"
    postgres_db: str = "knowledge_graph"

    # APIs
    anthropic_api_key: str = ""
    together_api_key: str = ""
    github_token: str = ""
    twitter_bearer_token: str = ""

    # App
    environment: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()