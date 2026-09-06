"""OpenAI model presets; explicit GPT model IDs are accepted too."""

from enum import StrEnum


class Model(StrEnum):
    GPT_5_6_LUNA = "gpt-5.6-luna"
    GPT_5_6_TERRA = "gpt-5.6-terra"
    GPT_6_ASTRA = "gpt-6-astra"
    MOCK = "mock"


def required_env_var(model: Model | str) -> str | None:
    name = str(model).removeprefix("openai/")
    if name == "mock":
        return None
    if name.startswith("gpt-"):
        return "OPENAI_API_KEY"
    raise ValueError(f"Expected an OpenAI GPT model ID, got {name}")
