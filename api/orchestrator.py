"""API module to manage the orchestrator."""

from langchain_core.messages import BaseMessage

from api.models import get_model_instance
from services.orchestrator import Orchestrator

def call_orchestrator(
    model_index: int,
    messages: list[BaseMessage],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    selected_agents: list[str] = [],
):
    """Function to call the orchestrator with the selected model and user input."""
    model = get_model_instance(model_index, temperature, max_tokens)
    orchestrator = Orchestrator(model)
    return orchestrator.invoke(messages, selected_agents=selected_agents)
