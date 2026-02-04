"""Main app file for the GenAI Playground"""
import asyncio
import streamlit as st
from api import get_model_parameters, get_model_list, get_model_instance, call_orchestrator
from langchain_core.messages import AIMessageChunk

async def main():
    """Main function to run the Streamlit app."""

    st.title("GenAI Playground")

    # Sidebar: model settings
    with st.sidebar:
        # Conversation controls
        if st.button("New conversation"):
            st.session_state["messages"] = []

        st.markdown("---")

        # Settings controls
        st.header("Settings")
        if "model" not in st.session_state:
            st.session_state["model"] = get_model_parameters(0)["name"]

        if "temperature" not in st.session_state:
            st.session_state["temperature"] = get_model_parameters(0)["temperature_default"]

        if "max_completion_tokens" not in st.session_state:
            st.session_state["max_completion_tokens"] = get_model_parameters(0)["max_completion_tokens"]

        available_models = get_model_list()
        model_index = available_models.index(st.session_state["model"])

        model_input = st.selectbox("Model", options=available_models, index=model_index)
        model_parameters = get_model_parameters(available_models.index(model_input))
        temperature_input = st.slider("Temperature",
                                    min_value=model_parameters["temperature_min"],
                                    max_value=model_parameters["temperature_max"],
                                    value=st.session_state["temperature"],
                                    step=0.01)
        max_completion_tokens_input = st.slider("Max Tokens",
                                    min_value=0,
                                    max_value=model_parameters["max_completion_tokens"],
                                    value=st.session_state["max_completion_tokens"],
                                    step=1)

        st.session_state["model"] = model_input
        st.session_state["temperature"] = float(temperature_input)
        st.session_state["max_completion_tokens"] = int(max_completion_tokens_input)


    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # apply sidebar settings to the API model instance
            model_index = available_models.index(st.session_state["model"])
            stream = call_orchestrator(model_index, 
                                       prompt,
                                       temperature=st.session_state["temperature"],
                                       max_tokens=st.session_state["max_completion_tokens"])

            placeholder = st.empty()
            reasoning_placeholder = st.empty()
            metadata_placeholder = st.empty()
            full_response = ""
            full_reasoning = ""
            usage_metadata = None

            async for _chunk in stream:
                for chunk in _chunk:
                    
                    if isinstance(chunk, AIMessageChunk):
                        if chunk.content:
                            full_response += chunk.content
                            placeholder.markdown(full_response)
                        elif 'reasoning_content' in chunk.additional_kwargs:
                            full_reasoning += chunk.additional_kwargs['reasoning_content']
                            with reasoning_placeholder.container():
                                with st.expander("🧠 Model Reasoning", expanded=False):
                                    st.markdown(full_reasoning)
                        elif chunk.usage_metadata:
                            usage_metadata = chunk.usage_metadata
                            model = {'model': st.session_state["model"]}
                            with metadata_placeholder.container():
                                with st.expander("📊 Usage Details", expanded=True):
                                    st.caption(f"✨ General: {model}")
                                    st.caption(f"🧾 Usage: {usage_metadata}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

if __name__ == "__main__":
    asyncio.run(main())
