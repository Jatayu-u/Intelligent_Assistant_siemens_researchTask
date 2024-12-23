# Import necessary libraries for handling command-line arguments, FAISS vector store, and model integration
import argparse
import faiss
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.schema import Document
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from torch.quantization import quantize_dynamic

# Allow duplicate library loading for compatibility on some systems
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define the path to store the FAISS index
FAISS_INDEX_PATH = "faiss_index"

# Fetch API keys for HuggingFace and Groq from environment variables
api_key = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")



# Set the number of threads for FAISS
os.environ["OMP_NUM_THREADS"] = "4"  # Adjust based on your CPU cores
faiss.omp_set_num_threads(4)

# Define the prompt template for the model, focusing on answering questions about the Indian Constitution
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

# Function to get the embedding model from HuggingFace
def get_embedding_function():
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    return embeddings

# Function to load or dynamically quantize the language model for inference
def load_quantized_model(model_name="gemma:2b"):
    model = Ollama(model=model_name)
    # Apply dynamic quantization to improve inference speed while maintaining accuracy
    # model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model

# Main function to handle command-line input and process the query
def main():
    # Create a command-line interface (CLI) to accept query text as input
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

# Function to load an existing FAISS index or create a new one if none exists
def load_or_create_faiss_index(embedding_function):
    try:
        # Attempt to load an existing FAISS index from disk
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embedding_function, allow_dangerous_deserialization=True
        )
        print("Loaded existing FAISS index.")
    except:
        # Create a new FAISS index if loading fails
        print("No FAISS index found. Creating a new one.")
        index = faiss.IndexFlatL2(embedding_function.embedding_dimension)
        vector_store = FAISS(
            index, embedding_function, InMemoryDocstore({}), {}
        )
    return vector_store

# Function to perform retrieval-augmented generation (RAG) for a given query
def query_rag(query_text: str):
    # Initialize the FAISS vector store
    embedding_function = get_embedding_function()
    db = load_or_create_faiss_index(embedding_function)

    # Perform similarity search on the vector store with the input query
    results = db.similarity_search_with_score(query_text, k=5)

    # Combine the retrieved context into a single text block for the model
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Load and optimize the language model
    model = load_quantized_model()

    # Use the model to generate a response with dynamic batching for efficiency
    response_text = model.invoke(prompt, batch_size=8)  # Adjust batch size as needed

    # Collect source metadata and content for reference
    sources = [
        {
            "id": doc.metadata.get("id", "Unknown"),
            "content": doc.page_content,
            "score": round(score, 4),
        }
        for doc, score in results
    ]

    # Format the response and sources for display
    formatted_response = f"Response:\n{response_text}\n\nSources:"
    for i, source in enumerate(sources, 1):
        formatted_response += f"\n{i}. [ID: {source['id']} | SIM: {source['score']}] {source['content']}"

    # Print the response and sources
    print(formatted_response)

    # Save the updated FAISS index for future queries
    db.save_local(FAISS_INDEX_PATH)

    return response_text

# Entry point for the script
if __name__ == "__main__":
    main()
