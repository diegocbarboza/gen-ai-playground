"""API module to manage model instances and parameters."""

import os
import json
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from orchestrator import Orchestrator

load_dotenv()

available_models = []

with open("models.json", "r", encoding="utf-8") as f:
    available_models = json.load(f)


def get_model_list():
    """Returns a list of available model names."""
    return [model["name"] for model in available_models]


def get_model_parameters(index: int):
    """Gets the model parameters based on the index provided."""
    return available_models[index]


def get_model_instance(index: int, temperature: float = 0.7, max_tokens: int = 1024):
    """Returns an instance of the model based on the index provided."""
    model = available_models[index]

    if model["provider"] == "nvidia":
        return ChatNVIDIA(
            base_url = model["base_url"],
            model=model["name"],
            api_key = os.environ["NVIDIA_API_KEY"],
            max_completion_tokens=max_tokens,
            temperature=temperature
        )

    print("No model instance found")
    return None


def call_orchestrator(
    model_index: int,
    messages: list[BaseMessage],
    temperature: float = 0.7,
    max_tokens: int = 1024
):
    """Function to call the orchestrator with the selected model and user input."""
    model = get_model_instance(model_index, temperature, max_tokens)
    orchestrator = Orchestrator(model)
    return orchestrator.invoke(messages)
