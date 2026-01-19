import streamlit as st
from agent import RealEstateAssistant

st.set_page_config(
    page_title="🏠 Real Estate Assistant",
    page_icon="🏢",
    layout="centered"
)

st.title("🏠 AI Real Estate Assistant")
st.caption("Ask about buying flats in any Pune city")

# Session state initialization
if "assistant" not in st.session_state:
    st.session_state.assistant = RealEstateAssistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("e.g. Show me flats in Pune")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for token in st.session_state.assistant.chat(user_input):
            full_response += token
            placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
