"""API module to manage model instances and parameters."""

import os
import json
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

available_models = []

with open("data/models.json", "r", encoding="utf-8") as f:
    available_models = json.load(f)


def get_model_list():
    """Returns a list of available model names."""
    return [model["name"] for model in available_models]


def get_model_parameters(index: int):
    """Gets the model parameters based on the index provided."""
    return available_models[index]


def add_model(model_parameters):
    """Adds a new model to the available models list."""
    available_models.append(model_parameters)
    save_models_to_file()


def update_model(model_index, model_parameters):
    """Updates the model parameters based on the index provided"""
    available_models[model_index] = {**available_models[model_index], **model_parameters}
    save_models_to_file()


def delete_model(model_index):
    """Deletes the model based on the index provided."""
    del available_models[model_index]
    save_models_to_file()


def save_models_to_file():
    """Saves the current available models to the JSON file."""
    with open("models.json", "w", encoding="utf-8") as f:
        json.dump(available_models, f)


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
    elif model["provider"] == "google":
        return ChatGoogleGenerativeAI(
            model=model["name"],
            api_key=os.environ["GOOGLE_API_KEY"],
            max_output_tokens=max_tokens,
            temperature=temperature
        )

    print("No model instance found")
    return None
