# Import LangChain components
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model

# Import LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

# Import local modules
from src.schema import State
from src.tools import search_web, weather_forecast, search_documents

import os
from dotenv import load_dotenv

load_dotenv()
from langchain_community.utilities import OpenWeatherMapAPIWrapper


weather = OpenWeatherMapAPIWrapper(
    openweathermap_api_key="6a82d2fafe4833df415fddf6af80365e"
)

# Check API key at module level
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY environment variable not set. Please set it in your .env file."
    )

# Set the environment variable for Google API
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize tools
tools = [search_web, weather.run, search_documents]


def llm_processor(state: State):
    """LLM Processor for handling chat messages with tool capabilities including document search."""

    system_content = """
You are a helpful, tool-first AI assistant that always grounds answers in external data.

- Do not answer factual or time-sensitive questions from internal memory alone. Use tools to fetch up-to-date information.
- Never say that you don't have access to information or you can't exactlyl see without searching documents or the web. First search then only reach to the conclusion that you can't answer.
- If you think that a question can be answered using the web, prefer using the parallel tools call to search the documents as well as the web.
- When using web or document tools, include concise citations and, when available, source dates or retrieval timestamps for key facts.
- Mark any inference that isn’t directly supported by tool output (e.g., “inferred from X”).
- Keep responses concise, honest, and user-facing. Avoid hallucination.
- Respect privacy and safety: do not fetch or expose private data unless it was explicitly provided in the conversation.

Have a gen-z, curious, lightly skeptical, and forward-thinking tone.
"""


    system_message = SystemMessage(content=system_content)

    apikey = os.getenv("GOOGLE_API_KEY")
    if not apikey:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.8,
        api_key=apikey,
        disable_streaming=True,
    )
    response = model.bind_tools(tools).invoke([system_message] + state.messages)

    return {"messages": [response]}


def router(state: State):
    """Router node - simplified for now."""
    # Just pass through the state
    last_message = state.messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"

    return "end"


# Define the tool node
tool_node = ToolNode(tools=tools)

# Create a branching graph with router
graph = (
    StateGraph(State)
    .add_node("llm_processor", llm_processor)
    .add_node("tool_node", tool_node)
    .add_edge(START, "llm_processor")
    .add_conditional_edges(
        "llm_processor", router, {"tool_node": "tool_node", "end": END}
    )
    .add_edge("tool_node", "llm_processor")
    .compile()
)
