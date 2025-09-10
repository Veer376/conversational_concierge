from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Annotated

# Type annotations for langchain/langgraph
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, START, add_messages


@dataclass
class State:
    """State class for the conversation agent."""

    messages: Annotated[List[BaseMessage], add_messages]


@dataclass
class Document:
    """Document class for storing document metadata and content."""

    id: str
    filename: str
    upload_time: str
    content: str
    chunks: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
