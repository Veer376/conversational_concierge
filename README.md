# Conversation Concierge
A lightweight LangGraph conversational agent for a Napa Valley wine business — integrated into a Streamlit UI. The agent answers from provided docs, performs web searches and weather checks, and demonstrates tool calls with a simple front-end. Fast, practical, and built so you can iterate.

https://github.com/user-attachments/assets/1560b4e8-636e-49ef-8196-6a7848d73cd5


## Features
- Chat interface built with Streamlit
- Document upload and processing
- Document indexing and storage
- Available tools - web search, weather_forecast.

## Getting Started


### Installation
```bash
# Prerequisites
- Python 3.12 or higher
- Required package manager [uv](https://docs.astral.sh/uv/getting-started/installation/)

# Clone the repository
git clone <repository-url>
cd conversation-concierge

uv init
# activate the virtual env
uv sync

# run the application
streamlit run app.py
```

## Report
### Approach 
I started by mapping the problem requirements: the agent must answer from docs, do web search, and fetch weather. I sketched a minimal LangGraph pipeline with three tool nodes — a retriever for documents, a web-search tool, and a weather tool — then created a small Streamlit UI to let users chat and see the tool calls. My aim was to keep the pipeline modular so tools can be toggled or swapped easily.

### Solution 
- Built a document retriever that indexes local docs and returns context to the LLM.
- Added a web-search tool (simple search API wrapper) and a weather tool (calls a weather API and returns structured text).
- Wired everything in LangGraph: the agent uses the retriever first, then can call web or weather tools as required, and composes a final answer.
- Made a Streamlit front-end that sends user messages to the LangGraph endpoint (or runs the pipeline locally), displays the LLM response, and shows each tool call and result in a small side panel.
  
### Challenges
- Choosing the right web/Search API & rate limits — I struggled to find an API that gave reliable, quick results without complex auth. I solved this by choosing a simple search wrapper for prototyping and adding a fallback to cached results for repeated queries.
- Syncing UI and pipeline events — showing tool calls live in Streamlit needed careful ordering. I solved it by logging tool calls from LangGraph and streaming updates to the UI (or polling short-interval updates) so the user could see the sequence of events.

### Experiments/Improvements
- UI/UX: Add animations when tool calls happen (e.g., animated spinner + “Tool X is fetching…”), and animate tool-call entries as they appear.
- Multi-source verification: Cross-check web search results with multiple search providers before answering to reduce hallucination.
- Better caching & fallback: Implement smarter caching for web and weather calls, and local offline mode with cached doc snippets.
- Tool debugging view: Expand the tool panel to include raw tool inputs/outputs and confidence scores from the LLM.
- Latency improvements: Batch retriever queries and fine-tune prompt templates to reduce token usage and speed up responses.
- Observability: Langfuse or Langsmith can be used to observe the agent performance.
