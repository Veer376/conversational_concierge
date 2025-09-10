"""Test runner for the conversation concierge graph."""

import asyncio
import logging
from langchain_core.messages import HumanMessage
from src.agent import graph
from src.schema import State
from pretty_message import pretty_message

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conversation_concierge.runner")


def main():
    """Main function to test the conversation concierge graph."""

    try:

        prompt = "tell me if aryaveer resume is good enough for the AIML role"

        # Create initial state
        initial_state = State(messages=[HumanMessage(content=prompt)])

        print("🚀 Starting conversation concierge test...")
        print("=" * 50)
        
        # Stream the graph execution
        for event in graph.stream(initial_state, stream_mode="values"):
            pretty_message(event.get("messages")[-1])
            
        print("=" * 50)
        print("✅ Graph execution completed!")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        print(f"❌ Error: {e}")


def sync_main():
    """Synchronous wrapper for the main function."""
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        print("🛑 Stopped by user")


if __name__ == "__main__":
    sync_main()
