"""Settings screen."""

import streamlit as st
from api import add_model, update_model, delete_model, get_model_parameters, get_model_list

def edit_model(model_parameters):
    """Function to edit or add a model."""

    name = st.text_input("Name", value=model_parameters["name"] if model_parameters else "")
    provider = st.text_input("Provider", value=model_parameters["provider"] if model_parameters else "")
    url = st.text_input("URL", value=model_parameters["base_url"] if model_parameters else "")
    api_key = st.text_input("API Key", value=model_parameters["api_key"] if model_parameters else "", type="password")
    temperature_min = st.text_input("Temperature Min", value=model_parameters["temperature_min"] if model_parameters else 0)
    temperature_max = st.text_input("Temperature Max", value=model_parameters["temperature_max"] if model_parameters else 2)
    temperature_default = st.text_input("Temperature Default", value=model_parameters["temperature_default"] if model_parameters else 0.7)
    max_completion_tokens = st.text_input("Max Completion Tokens", value=model_parameters["max_completion_tokens"] if model_parameters else 1024)

    return {
        "name": name,
        "provider": provider,
        "base_url": url,
        "api_key": api_key,
        "temperature_min": float(temperature_min),
        "temperature_max": float(temperature_max),
        "temperature_default": float(temperature_default),
        "max_completion_tokens": int(max_completion_tokens)
    }


def show_settings():
    """Function to show the settings screen."""
    st.header("Settings")

    st.markdown("---")

    st.header("Models")
    if st.button("➕ Add Model"):
        st.session_state["edit_model"] = "<<<NEW_MODEL>>>"

    if st.session_state.get("edit_model", "") == "<<<NEW_MODEL>>>":
        model_parameters = edit_model(None)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("💾 Save New Model"):
                add_model(model_parameters)
                st.session_state["edit_model"] = ""
                st.rerun()
        with col2:
            if st.button("❌ Cancel"):
                st.session_state["edit_model"] = ""
                st.rerun()

    available_models = get_model_list()
    for model in available_models:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader(model)
        with col2:
            if st.button("✏️ Edit", key=f"edit_{model}"):
                if "edit_model" not in st.session_state:
                    st.session_state["edit_model"] = model
                else:
                    st.session_state["edit_model"] = model if st.session_state["edit_model"] != model else ""
                st.rerun()

        if "edit_model" in st.session_state and st.session_state["edit_model"] == model:
            model_parameters = edit_model(get_model_parameters(available_models.index(st.session_state["edit_model"])))
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("💾 Save Model"):
                    update_model(available_models.index(st.session_state["edit_model"]), model_parameters)
                    st.session_state["edit_model"] = ""
                    st.rerun()
            with col2:
                if st.button("🗑️ Delete Model"):
                    delete_model(available_models.index(st.session_state["edit_model"]))
                    st.session_state["edit_model"] = ""
                    st.rerun()
            with col3:
                if st.button("❌ Cancel"):
                    st.session_state["edit_model"] = ""
                    st.rerun()
        else:
            parameters = get_model_parameters(available_models.index(model))
            st.markdown(f"**Provider**: {parameters['provider']}")
            st.markdown(f"**Base URL**: {parameters['base_url']}")
            st.markdown(f"**Max Completion Tokens**: {parameters['max_completion_tokens']}")
            st.markdown(f"**Temperature Range**: {parameters['temperature_min']} - {parameters['temperature_max']} | Default: {parameters['temperature_default']}")

        st.markdown("---")

    if st.button("Back"):
        st.session_state["show_settings"] = False
        st.rerun()
