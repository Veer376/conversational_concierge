import os
import pickle
from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime
import asyncio
import fitz  # PyMuPDF for PDF processing
import io

# Vector database imports
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument

# Constants
DOCS_DIR = "processed_docs"
INDEX_FILE = os.path.join(DOCS_DIR, "document_index.pkl")
VECTOR_STORE_DIR = os.path.join(DOCS_DIR, "faiss_index")
CHUNKS_DIR = os.path.join(DOCS_DIR, "chunks")

# Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Initialize embedding model (cached for performance)
_embedding_model = None


def get_embedding_model():
    """Get or initialize the Google embedding model."""
    global _embedding_model
    if _embedding_model is None:
        # Ensure an event loop exists for async operations
        try:
            # Try to get the current event loop
            asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists, create one for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Get Google API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        _embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
            task_type="retrieval_document",
        )
    return _embedding_model


def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """
    Extract text from various file types including PDF, TXT, etc.
    """
    file_extension = filename.lower().split(".")[-1] if "." in filename else ""

    if file_extension == "pdf":
        try:
            # Use PyMuPDF to extract text from PDF
            pdf_stream = io.BytesIO(file_content)
            pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")

            text_content = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text_content += page.get_text() + "\n"

            pdf_document.close()

            if text_content.strip():
                return text_content.strip()
            else:
                return f"PDF file '{filename}' appears to contain no extractable text (possibly scanned images)"

        except Exception as e:
            return f"Error extracting text from PDF '{filename}': {str(e)}"

    else:
        # For text files and other formats
        try:
            return file_content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                # Try different encodings
                for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                    try:
                        return file_content.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                # If all encodings fail
                return f"Could not decode text from file '{filename}' - unsupported encoding or binary file"
            except Exception as e:
                return f"Error processing file '{filename}': {str(e)}"


def process_document(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    Process a document and extract its content.
    Now includes:
    1. Text extraction and chunking (supports PDF, TXT, and other text files)
    2. Embedding generation
    3. Storage in vector database
    """
    # Generate a unique ID for the document
    doc_id = str(uuid.uuid4())

    # Extract text based on file type
    content = extract_text_from_file(file_content, filename)

    # Chunk the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_text(content)

    # Create LangChain documents for vector store
    documents = []
    for i, chunk in enumerate(chunks):
        doc = LangChainDocument(
            page_content=chunk,
            metadata={
                "doc_id": doc_id,
                "filename": filename,
                "chunk_id": f"{doc_id}_{i}",
                "upload_time": datetime.now().isoformat(),
            },
        )
        documents.append(doc)

    # Add to vector store
    add_documents_to_vector_store(documents)

    # Create document metadata
    doc_info = {
        "id": doc_id,
        "filename": filename,
        "upload_time": datetime.now().isoformat(),
        "content": content,
        "chunk_count": len(chunks),
        "chunks": [
            {"chunk_id": f"{doc_id}_{i}", "content": chunk}
            for i, chunk in enumerate(chunks)
        ],
    }

    return doc_info


def save_document_index(documents: Dict[str, Dict[str, Any]]) -> None:
    """
    Save the document index to disk.
    """
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(documents, f)


def load_document_index() -> Dict[str, Dict[str, Any]]:
    """
    Load the document index from disk.
    """
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def get_vector_store() -> Optional[FAISS]:
    """
    Load the FAISS vector store from disk.
    """
    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    pkl_path = os.path.join(VECTOR_STORE_DIR, "index.pkl")

    if os.path.exists(index_path) and os.path.exists(pkl_path):
        try:
            embeddings = get_embedding_model()
            return FAISS.load_local(
                VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    return None


def save_vector_store(vector_store: FAISS) -> None:
    """
    Save the FAISS vector store to disk.
    """
    try:
        vector_store.save_local(VECTOR_STORE_DIR)
    except Exception as e:
        print(f"Error saving vector store: {e}")


def add_documents_to_vector_store(documents: List[LangChainDocument]) -> None:
    """
    Add documents to the vector store.
    """
    if not documents:
        return

    embeddings = get_embedding_model()

    # Load existing vector store or create new one
    vector_store = get_vector_store()

    if vector_store is None:
        # Create new vector store
        vector_store = FAISS.from_documents(documents, embeddings)
    else:
        # Add to existing vector store
        vector_store.add_documents(documents)

    # Save the updated vector store
    save_vector_store(vector_store)


def retrieve_document_context(
    query: str, documents: Dict[str, Dict[str, Any]], top_k: int = 3
) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieve relevant document contexts based on a query using semantic search with Google embeddings.

    Args:
        query: The search query
        documents: Document metadata (not used in vector search but kept for compatibility)
        top_k: Number of top relevant chunks to return

    Returns:
        List of relevant document chunks with metadata
    """
    # Load the vector store
    vector_store = get_vector_store()

    if vector_store is None:
        print("No vector store found. Please add documents first.")
        return None

    try:
        # Ensure an event loop exists for async operations
        try:
            # Try to get the current event loop
            asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists, create one for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # For query embeddings, we can create a separate embedding instance optimized for queries
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        query_embedder = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
            task_type="retrieval_query",  # Optimized for queries
        )

        # Get query embedding
        query_embedding = query_embedder.embed_query(query)

        # Perform similarity search using the query embedding
        search_results = vector_store.similarity_search_by_vector(
            query_embedding, k=top_k
        )

        # Format results with scores
        relevant_chunks = []
        for i, doc in enumerate(search_results):
            chunk_info = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": 1.0
                - (
                    i * 0.1
                ),  # Approximate scoring since similarity_search_by_vector doesn't return scores
                "doc_id": doc.metadata.get("doc_id"),
                "filename": doc.metadata.get("filename"),
                "chunk_id": doc.metadata.get("chunk_id"),
            }
            relevant_chunks.append(chunk_info)

        return relevant_chunks

    except Exception as e:
        print(f"Error during document retrieval: {e}")
        # Fallback to regular similarity search
        try:
            search_results = vector_store.similarity_search(query, k=top_k)
            relevant_chunks = []
            for i, doc in enumerate(search_results):
                chunk_info = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 1.0 - (i * 0.1),
                    "doc_id": doc.metadata.get("doc_id"),
                    "filename": doc.metadata.get("filename"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                }
                relevant_chunks.append(chunk_info)
            return relevant_chunks
        except Exception as fallback_error:
            print(f"Fallback search also failed: {fallback_error}")
            return None


def initialize_vector_store() -> None:
    """
    Initialize an empty vector store if none exists.
    """
    if get_vector_store() is None:
        embeddings = get_embedding_model()
        # Create an empty vector store with a dummy document
        dummy_doc = LangChainDocument(
            page_content="Initialization document",
            metadata={"type": "init", "doc_id": "init"},
        )
        vector_store = FAISS.from_documents([dummy_doc], embeddings)
        save_vector_store(vector_store)
        print("Initialized new vector store")


def get_document_stats() -> Dict[str, Any]:
    """
    Get statistics about the vector store and documents.
    """
    vector_store = get_vector_store()
    documents = load_document_index()

    stats = {
        "total_documents": len(documents),
        "vector_store_exists": vector_store is not None,
        "total_chunks": (
            sum(doc.get("chunk_count", 0) for doc in documents.values())
            if documents
            else 0
        ),
    }

    if vector_store:
        try:
            stats["vector_store_size"] = vector_store.index.ntotal
        except:
            stats["vector_store_size"] = "Unknown"

    return stats
