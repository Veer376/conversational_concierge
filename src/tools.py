"""
This module contains tools that can be used by the conversation agent
to interact with external services and fetch information.
"""

import requests
from typing import Dict, Any, List, Optional
from src.utils import retrieve_document_context


def search_web(query: str) -> Dict[str, Any]:
    """
    Perform a web search using SerpAPI and returns clean, structured results.

    Args:
        query: The search query string

    Returns:
        Dictionary containing:
        - answer_box data (title, snippet, price, exchange, stock, currency) when available
        - organic_results snippets (snippet, snippet2, snippet3)
        - search_metadata for reference
    """
    try:
        response = requests.get(
            f"https://serpapi.com/search.json?engine=google&q={query}&api_key=a57ce980c9195fe924ac2ebb4fbecc42346203211415d47769e484467c412ad4"
        )
        response.raise_for_status()
        data = response.json()

        # Extract answer_box data (for structured results like stocks, weather, etc.)
        answer_box = data.get("answer_box", {})

        # Extract organic results (general search results)
        organic_results = data.get("organic_results", [])

        return [answer_box] + organic_results

    except Exception as e:
        return {"error": str(e)}


def weather_forecast(location: str) -> str:
    """
    Placeholder for weather forecast functionality.
    In a real implementation, this would use a weather API.
    """
    return f"Weather forecast for {location}: Sunny with a chance of rain"


def search_documents(query: str) -> str:
    """
    Search through documents uploaded by the user. e.g. Resumes, PDFs, etc.
    """
    try:
        # Use the retrieve_document_context function to get relevant documents from vector store
        retrieved_chunks = retrieve_document_context(
            query, {}, top_k=3  # Empty dict since documents param is not used
        )

        if retrieved_chunks:
            result = f"Found {len(retrieved_chunks)} relevant document chunks:\n\n"
            for i, chunk in enumerate(retrieved_chunks):
                result += f"--- Document Chunk {i+1} (from {chunk.get('filename', 'Unknown')}) ---\n"
                result += f"{chunk.get('content', '')}\n\n"
            return result
        else:
            return "No relevant documents found for your query. Please make sure you have uploaded documents or try a different search term."

    except Exception as e:
        return f"Error searching documents: {str(e)}"
