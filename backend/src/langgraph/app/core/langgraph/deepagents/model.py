from venv import logger
from langchain_anthropic import ChatAnthropic
from src.langgraph.app.core.config import (
    Environment,
    settings,
)

def get_default_model():

        # Development - we can use lower speeds for cost savings
        if settings.ENVIRONMENT == Environment.DEVELOPMENT:
            # New, correct line using an f-string
            logger.info(f"LLM initialized: model={settings.ANTHROPIC_MODEL}, environment={settings.ENVIRONMENT.value}")
            if settings.ANTHROPIC_MODEL != "claude-sonnet-4-20250514":
                raise ValueError("'claude-sonnet-4-20250514' is recommended as default model in this configuration.")
            return ChatAnthropic(model_name=settings.ANTHROPIC_MODEL, max_tokens=64000, temperature=0)

        # Production - use higher quality settings
        elif settings.ENVIRONMENT == Environment.PRODUCTION:
            logger.info(f"LLM initialized: model={settings.ANTHROPIC_MODEL}, environment={settings.ENVIRONMENT.value}")
            if settings.ANTHROPIC_MODEL != "claude-sonnet-4-20250514":
                raise ValueError("'claude-sonnet-4-20250514' is recommended as default model in this configuration.")
            return ChatAnthropic(model_name=settings.ANTHROPIC_MODEL, max_tokens=64000, temperature=0)
        