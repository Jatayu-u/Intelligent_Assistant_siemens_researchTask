import argparse
import os
import shutil
from uuid import uuid4
import faiss

# Import necessary modules for document loading and processing
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema.document import Document
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Define constants for data and index paths
DATA_PATH = "data"
FAISS_PATH = "faiss_index_semantic_hnsw"
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
# Fetch the API keys from environment variables
api_key = os.getenv("HUGGINGFACE_API_KEY")  # HuggingFace API key
groq_api_key = os.getenv("GROQ_API_KEY")    # Groq API key (if needed)

# Function to initialize and return the embedding model
def get_embedding_function():
    # Use the HuggingFaceInferenceAPIEmbeddings with a specified model
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    return embeddings

# Main function for handling the script's operations
def main():
    parser = argparse.ArgumentParser()
    # Add arguments to reset the index and choose indexing type
    parser.add_argument("--reset", action="store_true", help="Reset the FAISS index.")
    parser.add_argument(
        "--use_hnsw", action="store_true", help="Use HNSW indexing instead of FlatL2."
    )
    args = parser.parse_args()

    # Reset the FAISS index if the --reset flag is provided
    if args.reset:
        print("✨ Clearing FAISS Index")
        clear_faiss_index()

    # Load documents, split them into chunks, and add to FAISS
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_faiss(chunks, args.use_hnsw)

# Function to load documents from the specified data path
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# Function to split documents into semantic chunks
def split_documents(documents: list[Document]):
    # Initialize the semantic chunker with the embedding function
    embedding_function = get_embedding_function()
    text_splitter = SemanticChunker(embedding_function)

    # Split each document into semantically meaningful chunks
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.create_documents([doc.page_content]))

    # Reattach metadata to chunks for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata = documents[i // len(chunks)].metadata

    return chunks

# Function to add document chunks to the FAISS vector store
def add_to_faiss(chunks: list[Document], use_hnsw: bool):
    embeddings = get_embedding_function()
    dimension = len(embeddings.embed_query("sample query"))  # Get the embedding dimension

    # Initialize the FAISS index based on the selected indexing type
    if use_hnsw:
        print("Using HNSW Indexing...")
        index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW Index
    else:
        print("Using FlatL2 Indexing...")
        index = faiss.IndexFlatL2(dimension)  # FlatL2 Index

    # Load or create the FAISS vector store
    if os.path.exists(FAISS_PATH):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(
            FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("Creating new FAISS index...")
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    # Assign unique IDs to each document chunk
    chunks_with_ids = calculate_chunk_ids(chunks)
    new_chunks = [chunk for chunk in chunks_with_ids]
    new_ids = [str(uuid4()) for _ in range(len(new_chunks))]

    # Add the chunks to the FAISS vector store
    print(f"👉 Adding new documents: {len(new_chunks)}")
    vector_store.add_documents(documents=new_chunks, ids=new_ids)

    # Save the updated FAISS index
    vector_store.save_local(FAISS_PATH)
    print("✅ FAISS index updated and saved!")

# Function to calculate unique IDs for document chunks
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    # Generate unique IDs for chunks based on their source and page
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

# Function to clear the FAISS index directory
def clear_faiss_index():
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)  # Remove the directory and its contents
        print("FAISS index cleared successfully.")

# Entry point of the script
if __name__ == "__main__":
    main()
