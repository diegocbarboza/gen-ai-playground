import os
import json
#from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing import TypedDict, List
from langchain_core.language_models import BaseChatModel 
import asyncio

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, MessagesState, START, END

from langgraph.prebuilt import ToolNode, tools_condition

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

    
    async def invoke(self, text):
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
        for event in app.stream({
            "messages": [HumanMessage(content=text)],
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
