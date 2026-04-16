"""Module to create an agent."""

from typing import TypedDict, List
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END


class State(TypedDict):
    """Agent state."""
    messages: List[BaseMessage]


def create_agent(
    name: str,
    version: str,
    model: BaseChatModel,
    instructions: str
) -> StateGraph:
    """Function to create the agent."""

    def invoke(state: State):
        """Node to invoke the agent."""

        prompt = [SystemMessage(content=instructions)] + state["messages"]
        model_response = model.invoke(prompt)

        return {
            "messages": state["messages"] + [model_response]
        }

    graph = StateGraph(State)
    graph.add_node("invoke", invoke)
    graph.set_entry_point("invoke")
    graph.add_edge("invoke", END)
    agent = graph.compile()

    print(f"Agent created (name: {name}, version: {version})")

    return agent
