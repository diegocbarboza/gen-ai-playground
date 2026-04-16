"""API module to manage agent instances and parameters."""

import json

available_agents = []

with open("data/agents.json", "r", encoding="utf-8") as f:
    available_agents = json.load(f)


def get_agent_list():
    """Returns a list of available agent names."""
    agents = [f"{agent['name']} (v{agent['version']})" for agent in available_agents]
    agents.sort()
    return agents


def get_agent_parameters(index: int):
    """Gets the agent parameters based on the index provided."""
    return available_agents[index]


def add_agent(agent_parameters):
    """Adds a new agent to the available agents list."""
    available_agents.append(agent_parameters)
    save_agents_to_file()


def update_agent(agent_index, agent_parameters):
    """Updates the agent parameters based on the index provided"""
    available_agents[agent_index] = {**available_agents[agent_index], **agent_parameters}
    save_agents_to_file()


def delete_agent(agent_index):
    """Deletes the agent based on the index provided."""
    del available_agents[agent_index]
    save_agents_to_file()


def save_agents_to_file():
    """Saves the current available agents to the JSON file."""
    with open("data/agents.json", "w", encoding="utf-8") as f:
        json.dump(available_agents, f, indent=4)
