# Conversation Concierge
Agent which can answer questions from your documents and using other tools.


https://github.com/user-attachments/assets/1560b4e8-636e-49ef-8196-6a7848d73cd5


## Features
- Chat interface built with Streamlit
- Document upload and processing
- Document indexing and storage
- Available tools - web search, weather_forecast.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Required package manager [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd conversation-concierge

# Install dependencies
uv sync

# run the application
streamlit run app.py
```

## Usage

1. Open the application in your web browser
2. Upload documents using the sidebar uploader
3. Click "Process Document" to index the document
4. Start chatting in the main interface
5. The assistant will respond with context from your documents when relevant

## Project Structure

- `app.py`: Streamlit chat UI and document processing
- `main.py`: Main entry point
- `src/agent.py`: Agent logic (to be implemented)
- `src/tools.py`: Utility tools
- `src/types.py`: Type definitions
- `processed_docs/`: Directory for storing processed documents and indices
