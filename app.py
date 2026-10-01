import streamlit as st

from src.rag import rag_chain


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🩺 Medical RAG Chatbot")


# --------------------------------------------------
# Initialize chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input("Ask a medical question...")


# --------------------------------------------------
# Process question
# --------------------------------------------------

if question:

    # ----------------------------------------------
    # Build previous conversation history
    # ----------------------------------------------

    chat_history = ""

    for message in st.session_state.messages:

        chat_history += (
            f'{message["role"]}: '
            f'{message["content"]}\n'
        )


    # ----------------------------------------------
    # Display user message
    # ----------------------------------------------

    with st.chat_message("user"):
        st.write(question)


    # ----------------------------------------------
    # Save user message
    # ----------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # ----------------------------------------------
    # Generate response
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching medical documents..."):

            response = rag_chain.invoke({
                "input": question,
                "history": chat_history
            })

        st.write(response)


    # ----------------------------------------------
    # Save assistant response
    # ----------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })