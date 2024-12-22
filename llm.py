import argparse
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.schema import Document
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq

FAISS_INDEX_PATH = "faiss_index"
import os

# Fetch the API key for Hugging Face and Groq from environment variables
api_key = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Template for the prompt used by the language model, structured for legal domain expertise
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

# Function to load or initialize the embedding model using Hugging Face's API
def get_embedding_function():
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    return embeddings

# Main function for the CLI application
def main():
    # Parse command-line arguments for the query text
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

# Function to load an existing FAISS index or create a new one
def load_or_create_faiss_index(embedding_function):
    try:
        # Attempt to load a pre-existing FAISS index
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embedding_function, allow_dangerous_deserialization=True
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

# Function to query the system using RAG (retrieval-augmented generation)
def query_rag(query_text: str):
    # Initialize the embedding function and load the FAISS vector store
    embedding_function = get_embedding_function()
    db = load_or_create_faiss_index(embedding_function)

    # Perform a similarity search in the vector store for the input query
    results = db.similarity_search_with_score(query_text, k=5)

    # Compile the retrieved documents into context text
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Query the language model (Groq in this case) with the generated prompt
    model = ChatGroq(
        temperature=0.7,
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        streaming=True,
        verbose=True,
    )

    # Generate the response from the model
    response_text = model.invoke(prompt)

    # Collect and format the metadata and content of the sources
    sources = [
        {
            "id": doc.metadata.get("id", "Unknown"),
            "content": doc.page_content,
            "score": round(score, 4),
        }
        for doc, score in results
    ]

    # Format the final response for display
    formatted_response = f"Response:\n{response_text}\n\nSources:"
    for i, source in enumerate(sources, 1):
        formatted_response += f"\n{i}. [ID: {source['id']} | SIM: {source['score']}] {source['content']}"

    print(formatted_response)

    # Save the updated FAISS index for future queries
    db.save_local(FAISS_INDEX_PATH)

    return response_text

# Entry point for the script
if __name__ == "__main__":
    main()
