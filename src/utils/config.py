# Without this file, every module in your project would need to call os.getenv("QDRANT_URL"), os.getenv("TAVILY_API_KEY") etc. scattered across dozens of files. That creates three problems:

# If a key name changes in .env, you have to hunt through every file to update it
# No type safety — os.getenv() always returns a raw string, so a threshold like "0.7" stays a string unless you manually convert it everywhere
# No validation — if a key is missing, the error only surfaces when that specific line runs, not at startup
# config.py centralises all of that into one place. Every other file just calls get_config("qdrant") and gets back a clean, typed object.

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import model_validator

load_dotenv()


# Nvidia is used to create vector embeddings (in phase 2) and is used to generate answers (in phase 5).
class NvidiaConfig(BaseSettings):
    api_key: str = os.getenv("NVIDIA_API_KEY", "")
    embedding_model_id: str = os.getenv(
        "NVIDIA_EMBEDDING_MODEL_ID",
        "nvidia/nv-embedqa-e5-v5"
    )
    llm_model_id: str = os.getenv(
        "NVIDIA_LLM_MODEL_ID",
        "google/gemma-4-31b-it"
    )
    base_url: str = "https://integrate.api.nvidia.com/v1"

    @model_validator(mode="after")
    def validate_nvidia_config(self):
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is not set")
        return self

class QdrantConfig(BaseSettings):
    url: str = os.getenv("QDRANT_URL","")
    api_key: str = os.getenv("QDRANT_API_KEY","")
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "")
    vector_size: int = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))

    @model_validator(mode="after")
    def validate_qdrant_config(self):
        if not self.url:
            raise ValueError("QDRANT_URL is not set")
        if not self.api_key:
            raise ValueError("QDRANT_API_KEY is not set")
        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not set")
        return self

# Tavily is a search engine built specifically for AI applications. Unlike Google/Bing, Tavily returns clean, structured text results (not HTML pages) that LLMs can directly read and process.

class TavilyConfig(BaseSettings):
    api_key: str = os.getenv("TAVILY_API_KEY","")
    search_depth: str = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")
    max_results: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

    @model_validator(mode="after")
    def check_required(self):
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is missing from .env")
        return self

class CRAGConfig(BaseSettings):
    domain: str = os.getenv("DOMAIN", "general")
    retrieval_threshold: float = float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.7"))
    web_search_threshold: float = float(os.getenv("WEB_SEARCH_THRESHOLD", "0.6"))
    verification_enabled: bool = os.getenv("ANSWER_VERIFICATION_ENABLED", "true").lower() == "true"
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    timeout: int = int(os.getenv("TIMEOUT_SECONDS", "30"))

# Global config instance
config = {
    "nvidia": NvidiaConfig(),
    "qdrant": QdrantConfig(),
    "tavily": TavilyConfig(),
    "crag": CRAGConfig()
}

def get_config(service: str):
    if service not in config:
        raise KeyError(
            f"Unknown service: '{service}'. "
            f"Available services: {list(config.keys())}"
        )
    return config[service]