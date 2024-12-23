import argparse
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.schema import Document
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
# Constants
FAISS_INDEX_PATH = "faiss_index"  # Path to save/load the FAISS index
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

# Fetch the API keys from environment variables
api_key = os.getenv("HUGGINGFACE_API_KEY")  # API key for HuggingFace embeddings
groq_api_key = os.getenv("GROQ_API_KEY")    # API key for Groq model

# Function to create and return an embedding model
def get_embedding_function():
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    return embeddings

# Entry point of the program
def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")  # Accepts a query as input
    args = parser.parse_args()
    query_text = args.query_text  # Extract the query text
    query_rag(query_text)         # Process the query using the RAG approach

# Function to load or create a FAISS index
def load_or_create_faiss_index(embedding_function):
    try:
        # Attempt to load an existing FAISS index from disk
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embedding_function, allow_dangerous_deserialization=True
        )
        print("Loaded existing FAISS index.")
    except:
        # If the index doesn't exist, create a new one
        print("No FAISS index found. Creating a new one.")
        index = faiss.IndexFlatL2(embedding_function.embedding_dimension)  # FlatL2 index
        vector_store = FAISS(
            index, embedding_function, InMemoryDocstore({}), {}  # Initialize a new vector store
        )
    return vector_store

# Function to process a query using RAG (Retrieval-Augmented Generation)
def query_rag(query_text: str):
    # Initialize the embedding function
    embedding_function = get_embedding_function()
    
    # Load or create the FAISS index
    db = load_or_create_faiss_index(embedding_function)

    # Perform a similarity search in the vector store for the query
    results = db.similarity_search_with_score(query_text, k=5)

    # Combine the retrieved documents into a single context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Format the prompt with the retrieved context and the query
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Invoke the model with the formatted prompt
    model = Ollama(model='gemma:2b')
    response_text = model.invoke(prompt)

    # Collect sources with their metadata and similarity scores
    sources = [
        {
            "id": doc.metadata.get("id", "Unknown"),
            "content": doc.page_content,
            "score": round(score, 4),
        }
        for doc, score in results
    ]

    # Format the response along with sources for display
    formatted_response = f"Response:\n{response_text}\n\nSources:"
    for i, source in enumerate(sources, 1):
        formatted_response += f"\n{i}. [ID: {source['id']} | SIM: {source['score']}] {source['content']}"

    # Print the formatted response
    print(formatted_response)

    # Save the updated FAISS index to disk
    db.save_local(FAISS_INDEX_PATH)

    return response_text

if __name__ == "__main__":
    main()
 
