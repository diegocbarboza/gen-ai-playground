"""Orchestrator class to manage interactions between different models, tools and agents."""
import os
import re
from typing import TypedDict, List
from dotenv import load_dotenv
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
#from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import Tool, tool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from api.agents import get_agent_list, get_agent_parameters
from services.create_agent import create_agent

load_dotenv()

os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"
tracer_provider = register()

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city"""
    return f"The weather in {city} is sunny, 5°C."

def make_tool(model: BaseChatModel, name: str, agent_parameters):
    # Sanitize tool name: Replace invalid chars with underscores, ensure max length of 128
    # Valid chars: alphanumeric (a-z, A-Z, 0-9), underscores (_), dots (.), colons (:), or dashes (-)
    sanitized_name = re.sub(r'[^a-zA-Z0-9_.:-]', '_', name)[:128]

    def dynamic_tool(prompt: str) -> str:
        """Tool function that runs the agent with the given prompt and returns the response."""
        agent = create_agent(
            name=sanitized_name,
            version=agent_parameters["version"],
            model=model,
            instructions=agent_parameters["prompt"]
        )

        return agent.invoke({
            "messages": [HumanMessage(content=prompt)],
        })

    return Tool(
        name=sanitized_name,
        description=agent_parameters["description"],
        func=dynamic_tool,
    )

class State(TypedDict):
    """Agent state."""
    messages: List[BaseMessage]
    tool_results: List[str]

class Orchestrator:
    """Class to orchestrate interactions between different models, tools and agents."""

    def __init__(self, model: BaseChatModel):
        self.model = model
        self.tools = []

    async def invoke(self, messages: list[BaseMessage], selected_agents: list[str] = None):
        """Invoke the orchestrator with the given user input."""
    
        #self.tools = [get_weather]
        available_agents = get_agent_list()
        for agent in selected_agents:
            agent_params = get_agent_parameters(available_agents.index(agent))
            self.tools.append(make_tool(self.model, agent, agent_params))

        graph = StateGraph(State)

        graph.add_node("tool_decider", self.tool_decider)
        graph.add_node("tool_executor", ToolNode(self.tools))
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
        llm_with_tools = self.model.bind_tools(self.tools)

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
            response = self.model.invoke(state["messages"])
        else:
            context = "\n".join(str(tool_context))
            prompt = [
                SystemMessage(
                    content=(
                        "Answer the user clearly. "
                        "If a task was successful, confirm it."
                        "You may use the following context if relevant:\n"
                        f"{context}"
                    )
                )
            ] + state["messages"]
            response = self.model.invoke(prompt)

        return {
            "messages": state["messages"] + [response]
        }
