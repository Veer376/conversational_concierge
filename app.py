import streamlit as st
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any

# Import LangChain components
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

# Import our custom modules
from src.tools import search_web, weather_forecast
from src.utils import (
    process_document,
    save_document_index,
    load_document_index,
    retrieve_document_context,
)
from src.agent import graph
from src.schema import State

# Set page config
st.set_page_config(page_title="Conversation Concierge", page_icon="💬", layout="wide")

# Initialize session state for chat messages (UI display) if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for langchain messages (actual message objects) if it doesn't exist
if "langchain_messages" not in st.session_state:
    st.session_state.langchain_messages = []

# Initialize session state for uploaded documents if it doesn't exist
if "documents" not in st.session_state:
    # Load existing document index
    st.session_state.documents = load_document_index()


def handle_document_upload(file):
    """
    Process and index the uploaded document.
    """
    # Read file content
    content = file.getvalue()

    # Process the document
    doc_info = process_document(content, file.name)

    # Add to session state
    doc_id = doc_info["id"]
    st.session_state.documents[doc_id] = doc_info

    # Save updated index to disk
    save_document_index(st.session_state.documents)

    return doc_id


def process_message_stream(user_input):
    """
    Process a user message through the LangGraph agent and stream the response.
    """
    # Create a human message from the user input
    human_message = HumanMessage(content=user_input)

    # Add the message to our langchain messages list
    st.session_state.langchain_messages.append(human_message)

    # Create the initial state for the graph
    state = State(
        messages=st.session_state.langchain_messages,
    )

    # Stream the response from the graph using "values" mode like in runner.py
    for event in graph.stream(state, stream_mode="values"):
        message = event.get("messages")[-1]
        st.session_state.langchain_messages.append(message)
        yield message


# Sidebar for document upload
with st.sidebar:
    st.title("Document Management")

    uploaded_file = st.file_uploader(
        "Upload a document", type=["txt", "pdf", "docx", "csv"]
    )

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                doc_id = handle_document_upload(uploaded_file)
                st.success(f"Document '{uploaded_file.name}' processed and indexed!")

    # Display list of uploaded documents
    if st.session_state.documents:
        st.subheader("Uploaded Documents")
        for doc_id, doc_info in st.session_state.documents.items():
            st.text(f"• {doc_info['filename']}")

# Main chat interface
st.title("Conversation Concierge 💬")
st.markdown("Upload documents and chat about them!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "tool":
            # Display tool calls in a different format
            st.info(f"🛠️ **Tool Call**: {message['tool_name']}")
            st.code(message["content"])
        else:
            st.write(message["content"])

# Get user input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Create a container for tool calls
        tool_container = st.container()

        # Stream the response
        for chunk in process_message_stream(user_input):
            if isinstance(chunk, AIMessage):
                # Handle AI response
                full_response = chunk.content
                response_placeholder.write(full_response)

                # Add to session state
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            elif isinstance(chunk, ToolMessage):
                # Handle tool calls
                tool_name = chunk.name if hasattr(chunk, "name") else "Unknown Tool"
                tool_content = chunk.content

                # Display tool call in a special way
                with tool_container:
                    st.info(f"🛠️ **Tool Call**: {tool_name}")
                    st.code(tool_content)

                # Add to session state
                st.session_state.messages.append(
                    {"role": "tool", "tool_name": tool_name, "content": tool_content}
                )
