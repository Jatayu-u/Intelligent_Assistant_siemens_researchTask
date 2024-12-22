import argparse
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.schema import Document
from langchain_groq import ChatGroq
import os

# Constants for the FAISS index storage path and environment variables for API keys
FAISS_INDEX_PATH = "faiss_index"

# Fetch API keys securely from environment variables
api_key = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Define a prompt template for the LLM with instructions for answering legal domain queries
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

---

Format your response as follows:
1. **Final Answer**: [Your final conclusion or answer]
2. **Logical Reasoning (Step-by-Step)**: 
    - Step 1: [Initial observation or analysis]
    - Step 2: [Key considerations and reasoning]
    - Step 3: [Further refinements and conclusions based on references]
3. **References and Justification**:
    - Reference 1: [Key article, section, or case law] - Why it was relevant.
    - Reference 2: [Additional supporting legal source] - Why it was necessary.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:{agent_scratchpad}
"""

# Function to initialize the embedding function using HuggingFaceInferenceAPI
def get_embedding_function():
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    return embeddings

# Load an existing FAISS index or create a new one if it doesn't exist
def load_or_create_faiss_index(embedding_function):
    try:
        # Attempt to load the FAISS index from the specified path
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embedding_function, allow_dangerous_deserialization=True
        )
        print("Loaded existing FAISS index.")
    except:
        # If the index does not exist, create a new one
        print("No FAISS index found. Creating a new one.")
        index = faiss.IndexFlatL2(embedding_function.embedding_dimension)
        vector_store = FAISS(
            index, embedding_function, InMemoryDocstore({}), {}
        )
    return vector_store

# Perform similarity search in the FAISS index and retrieve context and sources
def faiss_tool(query_text: str, embedding_function):
    db = load_or_create_faiss_index(embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    # Combine the content of the retrieved documents into a single context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    # Create a list of sources with metadata and similarity scores
    sources = [
        {
            "id": doc.metadata.get("id", "Unknown"),
            "content": doc.page_content,
            "score": round(score, 4),
        }
        for doc, score in results
    ]
    return context_text, sources, db

# Main function to handle the Retrieval-Augmented Generation (RAG) query workflow
def query_rag(query_text: str):
    # Get the embedding function
    embedding_function = get_embedding_function()
    # Retrieve relevant context and sources from the FAISS vector store
    context_text, sources, db = faiss_tool(query_text, embedding_function)

    # Define a helper function for FAISS search tool usage
    def faiss_search_tool(input_text: str) -> str:
        context, _, _ = faiss_tool(input_text, embedding_function)
        return context

    # Define the tools available for the LLM agent
    tools = [
        Tool(
            name="FAISS Search",
            func=faiss_search_tool,
            description="Use this tool to retrieve relevant context from the FAISS vector store."
        )
    ]

    # Initialize the LLM for question answering
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        streaming=True,
        verbose=True,
    )
    # Use the defined prompt template for guiding the LLM
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # Create a reactive agent using the LLM and tools
    agent = create_react_agent(llm, tools, prompt)

    # Create an agent executor to handle queries
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # Define inputs for the agent
    inputs = {
        "question": query_text,
        "context": context_text,
        "agent_scratchpad": ""
    }

    # Invoke the agent with the provided inputs
    response = agent_executor.invoke(inputs)
    # Format the response and sources for output
    formatted_response = f"Response:\n{response['output']}\n\nSources:"
    for i, source in enumerate(sources, 1):
        formatted_response += f"\n{i}. [ID: {source['id']} | SIM: {source['score']}] {source['content']}"

    # Print the formatted response
    print(formatted_response)
    # Save the updated FAISS index to disk
    db.save_local(FAISS_INDEX_PATH)

    return response

# Entry point of the script for handling command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)
