"""Orchestrator class to manage interactions between different models, tools and agents."""
import os
from typing import TypedDict, List
from dotenv import load_dotenv
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
#from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

tracer_provider = register(
    space_id=os.environ.get("ARIZE_SPACE_ID"),
    api_key=os.environ.get("ARIZE_API_KEY"),
    project_name=os.environ.get("ARIZE_PROJECT_NAME", "default"),
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city"""
    return f"The weather in {city} is sunny, 5°C."

class State(TypedDict):
    """Agent state."""
    messages: List[BaseMessage]
    tool_results: List[str]

class Orchestrator:
    """Class to orchestrate interactions between different models, tools and agents."""

    def __init__(self, model: BaseChatModel):
        self.model = model


    async def invoke(self, messages: list[BaseMessage]):
        """Invoke the orchestrator with the given user input."""
        graph = StateGraph(State)

        graph.add_node("tool_decider", self.tool_decider)
        graph.add_node("tool_executor", ToolNode([get_weather]))
        graph.add_node("final_answer", self.final_answer)

        graph.set_entry_point("tool_decider")

        graph.add_conditional_edges(
            "tool_decider",
            tools_condition,
            {
                "tools": "tool_executor",
                END: "final_answer",
            }
        )

        graph.add_edge("tool_executor", "final_answer")
        graph.add_edge("final_answer", END)

        app = graph.compile()

        # https://docs.langchain.com/oss/python/langgraph/streaming
        for event in app.stream({
            "messages": messages,
            "tool_results": []
        }, stream_mode="messages"):
            yield event


    def tool_decider(self, state: State):
        """Node that decides if will use tools."""
        llm_with_tools = self.model.bind_tools([get_weather])

        response = llm_with_tools.invoke(state["messages"])

        return {
            "messages": state["messages"] + [response]
        }


    def final_answer(self, state: State):
        """Node to generate final answer."""
        tool_context = [
            m.content for m in state["messages"]
            if m.type == "tool"
        ]

        if len(tool_context) == 0:
            response = self.model.invoke([state["messages"][0]])
        else:
            context = "\n".join(str(tool_context))
            prompt = [
                SystemMessage(
                    content=(
                        "Answer the user clearly. "
                        "If a task was successful, confirm it."
                        "You may use the following context if relevant:\n"
                        f"{context}"
                        "Always answer in portuguese."
                    )
                )
            ] + state["messages"]
            response = self.model.invoke(prompt)

        return {
            "messages": state["messages"] + [response]
        }
