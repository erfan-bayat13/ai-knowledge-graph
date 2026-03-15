# CHANGED (Phase 1): added semantic_scholar_api_key, openalex_email for enrichment APIs
# CHANGED: removed unused postgres and twitter settings

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Neo4j
    neo4j_uri:      str = "bolt://localhost:7687"
    neo4j_user:     str = "neo4j"
    neo4j_password: str = "password123"

    # External enrichment APIs (Phase 1)
    semantic_scholar_api_key: str = ""   # optional — unlocks higher S2 rate limits
    openalex_email:           str = "erfanbyt13@gmail.com"   # optional — OpenAlex polite pool User-Agent
    openalex_api_key:         str = ""   # optional — OpenAlex API key (not required, but may improve rate limits)

    # LLM (fallback judge + on-demand summaries)
    together_api_key:  str = ""
    anthropic_api_key: str = ""

    # GitHub (post-MVP)
    github_token: str = ""

    # App
    environment: str = "development"
    log_level:   str = "INFO"

@lru_cache()
def get_settings() -> Settings:
    return Settings()