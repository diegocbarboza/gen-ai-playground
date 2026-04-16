"""Settings screen."""

import streamlit as st
from api.agents import add_agent, delete_agent, get_agent_list, get_agent_parameters, update_agent

def edit_agent(agent_parameters):
    """Function to edit or add an agent."""

    name = st.text_input("Name", value=agent_parameters.get("name", "") if agent_parameters else "")
    version = st.text_input("Version", value=agent_parameters.get("version", "") if agent_parameters else "")
    description = st.text_input("Description", value=agent_parameters.get("description", "") if agent_parameters else "")
    prompt = st.text_area("Prompt", value=agent_parameters.get("prompt", "") if agent_parameters else "")

    return {
        "name": name,
        "version": version,
        "description": description,
        "prompt": prompt,
    }


st.header("Agents")
if st.button("➕ Add Agent"):
    st.session_state["edit_agent"] = "<<<NEW_AGENT>>>"

if st.session_state.get("edit_agent", "") == "<<<NEW_AGENT>>>":
    agent_parameters = edit_agent(None)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💾 Save New Agent"):
            add_agent(agent_parameters)
            st.session_state["edit_agent"] = None
            st.rerun()
    with col2:
        if st.button("❌ Cancel"):
            st.session_state["edit_agent"] = ""
            st.rerun()

st.markdown("---")

available_agents = get_agent_list()
for agent in available_agents:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader(agent)
    with col2:
        params = get_agent_parameters(available_agents.index(agent))
        if st.button("✏️ Edit", key=f"edit_{agent}"):
            if "edit_agent" not in st.session_state:
                st.session_state["edit_agent"] = agent
            else:
                st.session_state["edit_agent"] = agent if st.session_state["edit_agent"] != agent else ""
            st.rerun()

    if "edit_agent" in st.session_state and st.session_state["edit_agent"] == agent:
        agent_parameters = edit_agent(get_agent_parameters(available_agents.index(st.session_state["edit_agent"])))
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("💾 Save Agent"):
                update_agent(available_agents.index(st.session_state["edit_agent"]), agent_parameters)
                st.session_state["edit_agent"] = ""
                st.rerun()
        with col2:
            if st.button("🗑️ Delete Agent"):
                delete_agent(available_agents.index(st.session_state["edit_agent"]))
                st.session_state["edit_agent"] = ""
                st.rerun()
        with col3:
            if st.button("❌ Cancel"):
                st.session_state["edit_agent"] = ""
                st.rerun()
    else:
        parameters = get_agent_parameters(available_agents.index(agent))
        st.markdown(f"**Version**: {parameters['version']}")
        st.markdown(f"**Description**: {parameters['description']}")
        st.markdown(f"**Prompt**: {parameters['prompt']}")

    st.markdown("---")

st.page_link("main.py", label="⬅ Back to Main")
