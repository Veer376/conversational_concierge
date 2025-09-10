#!/usr/bin/env python3
"""
Demo script to test the vector database integration with Google embeddings.
This script shows how to:
1. Initialize the vector store with Google's text-embedding-004 model
2. Add sample documents
3. Perform semantic search using Google embeddings
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    initialize_vector_store,
    process_document,
    retrieve_document_context,
    get_document_stats,
    save_document_index,
    load_document_index,
)


def demo_vector_database():
    """Demonstrate the vector database functionality."""
    print("🚀 Vector Database Demo")
    print("=" * 50)

    # Initialize vector store
    print("1. Initializing vector store...")
    initialize_vector_store()

    # Add sample documents
    print("\n2. Adding sample documents...")

    sample_docs = [
        {
            "filename": "ai_basics.txt",
            "content": """
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines that can perform tasks that typically require human intelligence. 
            AI includes machine learning, deep learning, natural language processing, and computer vision.
            Modern AI systems use neural networks and large datasets to learn patterns and make predictions.
            """,
        },
        {
            "filename": "python_programming.txt",
            "content": """
            Python is a high-level programming language known for its simplicity and readability.
            It's widely used in data science, web development, automation, and artificial intelligence.
            Python has extensive libraries like NumPy, Pandas, TensorFlow, and PyTorch that make it
            ideal for machine learning and data analysis tasks.
            """,
        },
        {
            "filename": "vector_databases.txt",
            "content": """
            Vector databases are specialized databases designed to store and search high-dimensional vectors.
            They are essential for similarity search, recommendation systems, and retrieval-augmented generation (RAG).
            Popular vector databases include FAISS, Pinecone, Chroma, and Qdrant. These databases use
            techniques like approximate nearest neighbor search to efficiently find similar vectors.
            """,
        },
    ]

    documents = load_document_index()

    for doc in sample_docs:
        print(f"   Adding: {doc['filename']}")
        file_content = doc["content"].encode("utf-8")
        doc_info = process_document(file_content, doc["filename"])
        documents[doc_info["id"]] = doc_info

    # Save the updated document index
    save_document_index(documents)

    # Show stats
    print("\n3. Vector Database Stats:")
    stats = get_document_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test semantic search
    print("\n4. Testing semantic search...")
    test_queries = [
        "What is machine learning?",
        "How to use Python for data science?",
        "Tell me about similarity search",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = retrieve_document_context(query, documents, top_k=2)

        if results:
            for i, result in enumerate(results):
                print(f"   Result {i+1} (Score: {result['relevance_score']:.3f}):")
                print(f"   From: {result['filename']}")
                print(f"   Content: {result['content'][:100]}...")
        else:
            print("   No results found")

    print("\n✅ Demo completed!")
    print("\nNow you can use the conversation agent with document retrieval!")
    print("Your documents are stored in the vector database and will be automatically")
    print("retrieved when users ask questions related to the content.")


if __name__ == "__main__":
    demo_vector_database()
