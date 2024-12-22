import os
import shutil
from uuid import uuid4
import faiss
import streamlit as st
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Paths and API keys
DATA_PATH = "data"  # Path where PDF documents will be stored
FAISS_PATH = "faiss_index"  # Path to store the FAISS vector index

# Fetch the API key from environment variables
api_key = os.getenv("HUGGINGFACE_API_KEY")  # Hugging Face API key for embeddings
groq_api_key = os.getenv("GROQ_API_KEY")  # Groq API key for the language model

# Prompt template for the question-answering system
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

# Function to initialize the embedding model
def get_embedding_function():
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    return embeddings

# Function to load documents from the specified directory
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# Function to split documents into manageable chunks for processing
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Max chunk size
        chunk_overlap=80,  # Overlap between chunks
        length_function=len,  # Function to measure chunk length
        is_separator_regex=False,  # Separator type
    )
    return text_splitter.split_documents(documents)

# Function to load or create a FAISS vector index
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

# Function to query the retrieval-augmented generation (RAG) system
def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = load_or_create_faiss_index(embedding_function)

    # Perform similarity search in the vector store
    results = db.similarity_search_with_score(query_text, k=5)

    # Extract context from the search results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Query the language model
    model = ChatGroq(
        temperature=0.7,  # Model response creativity level
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        streaming=True,  # Stream responses for better user experience
        verbose=True,  # Enable detailed logging
    )
    response_text = model.invoke(prompt)

    # Collect metadata and content of the source documents
    sources = [
        {
            "id": doc.metadata.get("id", "Unknown"),
            "content": doc.page_content,
            "score": round(score, 4),
        }
        for doc, score in results
    ]

    # Format the response and sources
    formatted_response = f"**Response:**\n{response_text}\n\n**Sources:**"
    for i, source in enumerate(sources, 1):
        formatted_response += f"\n{i}. [ID: {source['id']} | SIM: {source['score']}] {source['content']}"

    # Save the updated FAISS index
    db.save_local(FAISS_PATH)

    return formatted_response

# Main function to run the Streamlit application
def main():
    st.title("Legal Intelligent Assistant")  # App title

    # Upload PDF files
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            for uploaded_file in uploaded_files:
                # Save uploaded files to the data directory
                file_path = os.path.join(DATA_PATH, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success("PDFs processed successfully.")

        # User input for querying
        question = st.text_input("Ask a question:")
        if question:
            with st.spinner("Retrieving your answer..."):
                # Process query and display results
                response, formatted_response = query_rag(question)
                
                # Display response and sources
                st.subheader("Answer")
                st.write(response)
                st.subheader("Detailed Answer with Sources")
                st.write(formatted_response)

# Entry point for the application
if __name__ == "__main__":
    main()
