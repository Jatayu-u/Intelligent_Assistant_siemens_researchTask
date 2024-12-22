# Import necessary libraries for argument parsing, file operations, and FAISS indexing
import argparse
import os
import shutil
from uuid import uuid4
import faiss

# Import libraries for handling PDF documents, text splitting, embeddings, and vector stores
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Define paths for data and FAISS index storage
DATA_PATH = "data"
FAISS_PATH = "faiss_index"

# Fetch the API key from environment variables
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Function to initialize the HuggingFace embedding model
def get_embedding_function():
    # Use the HuggingFaceInferenceAPIEmbeddings model for generating embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    return embeddings

# Main function to handle argument parsing and orchestrate the workflow
def main():
    # Set up argument parser for CLI options
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the FAISS index.")
    parser.add_argument(
        "--use_hnsw", action="store_true", help="Use HNSW indexing instead of FlatL2."
    )
    args = parser.parse_args()

    # Handle the reset argument to clear the FAISS index
    if args.reset:
        print("✨ Clearing FAISS Index")
        clear_faiss_index()

    # Load documents from the specified directory
    documents = load_documents()

    # Split the loaded documents into smaller chunks
    chunks = split_documents(documents)

    # Add the split chunks to the FAISS index
    add_to_faiss(chunks, args.use_hnsw)

# Function to load documents from a directory containing PDF files
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# Function to split large documents into smaller chunks for indexing
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,          # Maximum size of each chunk
        chunk_overlap=80,        # Overlap size between consecutive chunks
        length_function=len,     # Function to determine chunk size
        is_separator_regex=False # Whether the separator is a regex
    )
    return text_splitter.split_documents(documents)

# Function to add document chunks to the FAISS index
def add_to_faiss(chunks: list[Document], use_hnsw: bool):
    embeddings = get_embedding_function()  # Get embedding model
    dimension = len(embeddings.embed_query("sample query"))  # Get embedding dimension

    # Choose indexing type based on user input
    if use_hnsw:
        print("Using HNSW Indexing...")
        index = faiss.IndexHNSWFlat(dimension, 32)  # Hierarchical Navigable Small World Index
    else:
        print("Using FlatL2 Indexing...")
        index = faiss.IndexFlatL2(dimension)  # Flat L2 distance index

    # Load an existing FAISS vector store or create a new one
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

    # Generate unique IDs for each document chunk
    chunks_with_ids = calculate_chunk_ids(chunks)
    new_chunks = [chunk for chunk in chunks_with_ids]
    new_ids = [str(uuid4()) for _ in range(len(new_chunks))]

    # Add the new document chunks to the vector store
    print(f"👉 Adding new documents: {len(new_chunks)}")
    vector_store.add_documents(documents=new_chunks, ids=new_ids)

    # Save the updated FAISS index to disk
    vector_store.save_local(FAISS_PATH)
    print("✅ FAISS index updated and saved!")

# Function to calculate unique IDs for each document chunk
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")  # Get document source metadata
        page = chunk.metadata.get("page")      # Get document page metadata
        current_page_id = f"{source}:{page}"   # Combine source and page for unique ID

        # Increment the chunk index if it's part of the same page, otherwise reset
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Assign a unique ID to the chunk
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

# Function to clear the existing FAISS index by deleting the directory
def clear_faiss_index():
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)  # Delete the FAISS index directory
        print("FAISS index cleared successfully.")

# Entry point of the script
if __name__ == "__main__":
    main()
