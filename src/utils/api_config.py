from dataclasses import dataclass
import os
from typing import Dict

from openai import OpenAI

# Default OpenAI-compatible API configuration. Override these with environment
# variables before running entity extraction or counterfactual generation.
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "YOUR_LLM_BASE_URL")
DEFAULT_API_KEY = os.getenv("LLM_API_KEY", "YOUR_LLM_API_KEY")
DEFAULT_MODEL = "medgemma-27b"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.1


@dataclass(frozen=True)
class ModelApiConfig:
    base_url: str
    api_key: str
    model: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens_param: str = "max_tokens"


# 以“模型名字”为键的配置表（可按需扩展）
MODEL_CONFIGS: Dict[str, ModelApiConfig] = {
    "medgemma-27b": ModelApiConfig(
        base_url=DEFAULT_BASE_URL,
        api_key=DEFAULT_API_KEY,
        model="medgemma-27b",
        max_tokens=64,
        temperature=0.1,
        max_tokens_param="max_tokens",
    ),
    "gemini-3-flash-preview": ModelApiConfig(
        base_url=os.getenv("GEMINI_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.getenv("GEMINI_API_KEY", DEFAULT_API_KEY),
        model="gemini-3-flash-preview",
        max_tokens=64,
        temperature=0.1,
        max_tokens_param="max_tokens",
    ),
    "gpt-4.1-mini": ModelApiConfig(
        base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY),
        model="gpt-4.1-mini",
        max_tokens=64,
        temperature=1,
        max_tokens_param="max_completion_tokens",
    ),
    "grok-4-fast": ModelApiConfig(
        base_url=os.getenv("GROK_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.getenv("GROK_API_KEY", DEFAULT_API_KEY),
        model="grok-4-fast",
        max_tokens=64,
        temperature=0.1,
        max_tokens_param="max_tokens",
    ),
    "claude-haiku-4-5": ModelApiConfig(
        base_url=os.getenv("CLAUDE_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.getenv("CLAUDE_API_KEY", DEFAULT_API_KEY),
        model="claude-haiku-4-5",
        max_tokens=64,
        temperature=0.1,
        max_tokens_param="max_tokens",
    ),
}


def get_model_config(model_name: str) -> ModelApiConfig:
    return MODEL_CONFIGS.get(
        model_name,
        ModelApiConfig(
            base_url=DEFAULT_BASE_URL,
            api_key=DEFAULT_API_KEY,
            model=model_name,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens_param="max_tokens",
        ),
    )


def create_client_from_model(model_name: str) -> OpenAI:
    cfg = get_model_config(model_name)
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


def create_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)
