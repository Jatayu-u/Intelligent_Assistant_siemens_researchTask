# Importing necessary libraries
import streamlit as st
import os
import faiss
from uuid import uuid4
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil

# Constants for paths and API keys
FAISS_PATH = "faiss_index_hnsw"  # Path to save or load the FAISS index
DATA_PATH = "data"  # Directory for storing data

# Fetch API keys from environment variables for embedding and model access
api_key = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Prompt template for querying the RAG model
PROMPT_TEMPLATE = """
You are an expert in the Indian Constitution and legal domain. Your role is to answer questions with the utmost accuracy and detail, providing logical reasoning and references to specific articles, sections, amendments, or precedents of the Indian Constitution. 

Your response must include:
1. **The final answer** to the question in a clear, concise manner.
2. **Step-by-step reasoning** explaining how you arrived at the answer.
3. **Key references** (specific articles, constitutional provisions, case laws, amendments, or any relevant legal documents) that were used to reach the conclusion.
4. **Why each reference was important** in answering the question logically and conclusively.

The reasoning should follow a clear progression, showing your expertise and understanding of the Indian Constitution and how the context provided aligns with the question.

---

Context:
{context}

---

Based on the above context, answer the following question comprehensively: 
{question}

---

Format your response as follows:
1. **Final Answer**: [Your final conclusion or answer, must be comprehensive including all details ]
2. **Logical Reasoning (Step-by-Step)**: 
    - Step 1: [Key considerations and reasoning]
    - Step 2: [Further refinements and conclusions based on references]
3. **References and Justification**:
    - Reference 1: [Key article, section, or case law] - Why it was relevant.
    - Reference 2: [Additional supporting legal source] - Why it was necessary.
"""

# Function to load embeddings using HuggingFaceInference API
def get_embedding_function():
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    return embeddings

# Function to load or create a FAISS index for vector storage
def load_or_create_faiss_index(embedding_function):
    try:
        # Attempt to load an existing FAISS index
        vector_store = FAISS.load_local(
            FAISS_PATH, embedding_function, allow_dangerous_deserialization=True
        )
        print("Loaded existing FAISS index.")
    except:
        # Create a new FAISS index if none exists
        print("No FAISS index found. Creating a new one.")
        index = faiss.IndexFlatL2(embedding_function.embedding_dimension)
        vector_store = FAISS(
            index, embedding_function, InMemoryDocstore({}), {}
        )
    return vector_store

# Function to load and quantize the model for inference
def load_quantized_model(model_name="gemma:2b"):
    model = Ollama(model=model_name)
    return model

# Function to load and preprocess documents from uploaded PDFs
def load_documents_from_pdf(uploaded_files):
    # Create a temporary directory for storing uploaded PDFs
    temp_pdf_path = "temp_pdfs"
    if not os.path.exists(temp_pdf_path):
        os.makedirs(temp_pdf_path)

    # Save the uploaded files locally
    for file in uploaded_files:
        file_path = os.path.join(temp_pdf_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    # Load documents from the temporary directory
    document_loader = PyPDFDirectoryLoader(temp_pdf_path)
    documents = document_loader.load()
    return documents

# Function to split documents into manageable chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Maximum size of each text chunk
        chunk_overlap=80,  # Overlap between consecutive chunks for context retention
        length_function=len,  # Function to measure text length
        is_separator_regex=False,  # Treat separators as plain text
    )
    return text_splitter.split_documents(documents)

# Function to query the RAG system for answers
def query_rag(query_text: str):
    # Initialize FAISS vector store with embeddings
    embedding_function = get_embedding_function()
    db = load_or_create_faiss_index(embedding_function)

    # Perform similarity search to retrieve relevant documents
    results = db.similarity_search_with_score(query_text, k=5)

    # Extract the context from the retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Load and invoke the model with the generated prompt
    model = load_quantized_model()
    response_text = model.invoke(prompt, batch_size=8)

    # Collect document sources and metadata for reference
    sources = [
        {
            "id": doc.metadata.get("id", "Unknown"),
            "content": doc.page_content,
            "score": round(score, 4),
        }
        for doc, score in results
    ]

    # Format the response with sources for clarity
    formatted_response = f"Response:\n{response_text}\n\nSources:"
    for i, source in enumerate(sources, 1):
        formatted_response += f"\n{i}. [ID: {source['id']} | SIM: {source['score']}] {source['content']}"

    # Save the updated FAISS index
    db.save_local(FAISS_PATH)

    return response_text, formatted_response

# Streamlit-based user interface
def main():
    st.title("Legal Assistant")  # Title of the application

    # File uploader for PDF documents
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)

    if uploaded_files:
        # Process uploaded files and display summary
        documents = load_documents_from_pdf(uploaded_files)
        chunks = split_documents(documents)
        st.write(f"Uploaded {len(documents)} documents with {len(chunks)} chunks.")

        # Input field for user query
        query_text = st.text_input("Ask a Question about the Indian Constitution:")

        if query_text:
            st.write("Retrieving your answer...")
            response, formatted_response = query_rag(query_text)
            
            # Display the answer and sources
            st.subheader("Answer")
            st.write(response)
            st.subheader("Detailed Answer with Sources")
            st.write(formatted_response)

# Run the Streamlit app
if __name__ == "__main__":
    main()
