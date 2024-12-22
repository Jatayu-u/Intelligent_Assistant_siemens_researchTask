from typing import List, Dict
import math
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
FAISS_INDEX_PATH = "faiss_index"

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

# Function to calculate Precision@k
def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_at_k = retrieved[:k]
    relevant_at_k = [doc for doc in retrieved_at_k if doc in relevant]
    return len(relevant_at_k) / k

# Function to calculate Recall@k
def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_at_k = retrieved[:k]
    relevant_at_k = [doc for doc in retrieved_at_k if doc in relevant]
    return len(relevant_at_k) / len(relevant) if relevant else 0.0

# Function to calculate nDCG@k
def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        relevance = 1 if doc in relevant else 0
        dcg += relevance / math.log2(i + 2)  # Discounted gain

    # Calculate ideal DCG (IDCG)
    ideal_relevance = [1] * min(len(relevant), k)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))

    return dcg / idcg if idcg > 0 else 0.0

# Modified query_rag function to include metrics
def query_rag_with_metrics(query_text: str, ground_truth: List[str]):
    # Prepare the FAISS vector store
    embedding_function = get_embedding_function()
    db = load_or_create_faiss_index(embedding_function)

    # Search the vector store
    results = db.similarity_search_with_score(query_text, k=5)

    # Extract retrieved document IDs
    retrieved_ids = [doc.metadata.get("id", "Unknown") for doc, _ in results]
    print(retrieved_ids)
    # Calculate metrics
    k = 5
    precision = precision_at_k(retrieved_ids, ground_truth, k)
    recall = recall_at_k(retrieved_ids, ground_truth, k)
    ndcg = ndcg_at_k(retrieved_ids, ground_truth, k)

    print(f"Precision@{k}: {precision}")
    print(f"Recall@{k}: {recall}")
    print(f"nDCG@{k}: {ndcg}")

    # Generate the response (original behavior)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # model = ChatGroq(
    #     temperature=0.7,
    #     groq_api_key=groq_api_key,
    #     model_name="mixtral-8x7b-32768",
    #     streaming=True,
    #     verbose=True,
    # )
    model = Ollama(model='gemma:2b')

    

    response_text = model.invoke(prompt)
    print("Response:\n", response_text)

    # Save the FAISS index for future use
    db.save_local(FAISS_INDEX_PATH)

    return response_text

# Example usage
if __name__ == "__main__":
    # Define ground truth for a sample query
    ground_truth_docs1 = ['data\\CONSTITUTION.pdf:0:1', 'data\\CONSTITUTION.pdf:0:0', 'data\\CONSTITUTION.pdf:13:2', 'data\\CONSTITUTION.pdf:1:3', 'data\\CONSTITUTION.pdf:8:0']

    # Run the query and evaluate metrics
    main_query_text1 = "What is the significance of Article 21 in the Indian Constitution?"
    query_rag_with_metrics(main_query_text1, ground_truth_docs1)

    # Example 2
    ground_truth_docs2 = ['data\\CONSTITUTION.pdf:0:1', 'data\\CONSTITUTION.pdf:0:0', 'data\\CONSTITUTION.pdf:13:2', 'data\\CONSTITUTION.pdf:1:3', 'data\\CONSTITUTION.pdf:8:0']
    main_query_text2 = "What does Article 15(3) of the Constitution permit?"
    query_rag_with_metrics(main_query_text2, ground_truth_docs2)

    # Example 3
    ground_truth_docs3 = ['data\\CONSTITUTION.pdf:11:1', 'data\\CONSTITUTION.pdf:14:0', 'data\\CONSTITUTION.pdf:9:1', 'data\\CONSTITUTION.pdf:3:1', 'data\\CONSTITUTION.pdf:11:0']
    main_query_text3 = "Which article allows provisions for socially and educationally backward classes?"
    query_rag_with_metrics(main_query_text3, ground_truth_docs3)

    # Example 4
    ground_truth_docs4 = ['data\\CONSTITUTION.pdf:4:1', 'data\\CONSTITUTION.pdf:11:0', 'data\\CONSTITUTION.pdf:4:2', 'data\\CONSTITUTION.pdf:13:0', 'data\\CONSTITUTION.pdf:10:3']
    main_query_text4 = "What does Article 16 ensure?"
    query_rag_with_metrics(main_query_text4, ground_truth_docs4)

    # Example 5
    ground_truth_docs5 = ['data\\CONSTITUTION.pdf:11:1', 'data\\CONSTITUTION.pdf:8:0', 'data\\CONSTITUTION.pdf:15:2', 'data\\CONSTITUTION.pdf:3:1', 'data\\CONSTITUTION.pdf:14:0']
    main_query_text5 = "What does Article 16 permit regarding backward classes?"
    query_rag_with_metrics(main_query_text5, ground_truth_docs5)

    # Example 6
    ground_truth_docs6 = ['data\\CONSTITUTION.pdf:4:1', 'data\\CONSTITUTION.pdf:11:0', 'data\\CONSTITUTION.pdf:8:0', 'data\\CONSTITUTION.pdf:4:2', 'data\\CONSTITUTION.pdf:3:2']
    main_query_text6 = "What does the Double Jeopardy clause state?"
    query_rag_with_metrics(main_query_text6, ground_truth_docs6)

    # Example 7
    ground_truth_docs7 = ['data\\CONSTITUTION.pdf:7:0', 'data\\CONSTITUTION.pdf:15:3', 'data\\CONSTITUTION.pdf:6:3', 'data\\CONSTITUTION.pdf:3:4', 'data\\CONSTITUTION.pdf:8:2']
    main_query_text7 = "What does Article 17 abolish?"
    query_rag_with_metrics(main_query_text7, ground_truth_docs7)

    # Example 8
    ground_truth_docs8 = ['data\\CONSTITUTION.pdf:15:2', 'data\\CONSTITUTION.pdf:11:1', 'data\\CONSTITUTION.pdf:8:0', 'data\\CONSTITUTION.pdf:14:0', 'data\\CONSTITUTION.pdf:9:1']
    main_query_text8 = "Which article guarantees six freedoms?"
    query_rag_with_metrics(main_query_text8, ground_truth_docs8)

    # Example 9
    ground_truth_docs9 = ['data\\CONSTITUTION.pdf:5:3', 'data\\CONSTITUTION.pdf:6:0', 'data\\CONSTITUTION.pdf:7:3', 'data\\CONSTITUTION.pdf:8:0', 'data\\CONSTITUTION.pdf:11:1']
    main_query_text9 = "What does Article 18 prohibit?"
    query_rag_with_metrics(main_query_text9, ground_truth_docs9)

    # Example 10
    ground_truth_docs10 = ['data\\CONSTITUTION.pdf:9:1', 'data\\CONSTITUTION.pdf:8:0', 'data\\CONSTITUTION.pdf:11:1', 'data\\CONSTITUTION.pdf:15:2', 'data\\CONSTITUTION.pdf:12:0']
    main_query_text10 = "What does freedom of speech include?"
    query_rag_with_metrics(main_query_text10, ground_truth_docs10)

    # Example 11
    ground_truth_docs11 = ['data\\CONSTITUTION.pdf:6:0', 'data\\CONSTITUTION.pdf:6:2', 'data\\CONSTITUTION.pdf:5:3', 'data\\CONSTITUTION.pdf:6:1', 'data\\CONSTITUTION.pdf:9:1']
    main_query_text11 = "Which case upheld the freedom of silence?"
    query_rag_with_metrics(main_query_text11, ground_truth_docs11)

    # Example 12
    ground_truth_docs12 = ['data\\CONSTITUTION.pdf:6:2', 'data\\CONSTITUTION.pdf:6:0', 'data\\CONSTITUTION.pdf:7:2', 'data\\CONSTITUTION.pdf:6:1', 'data\\CONSTITUTION.pdf:7:3']
    main_query_text12 = "Which case declared pre-censorship invalid?"
    query_rag_with_metrics(main_query_text12, ground_truth_docs12)

    # Example 13
    ground_truth_docs13 = ['data\\CONSTITUTION.pdf:6:2', 'data\\CONSTITUTION.pdf:12:0', 'data\\CONSTITUTION.pdf:6:0', 'data\\CONSTITUTION.pdf:5:1', 'data\\CONSTITUTION.pdf:12:1']
    main_query_text13 = "What is the principal behind the Right to Know?"
    query_rag_with_metrics(main_query_text13, ground_truth_docs13)

    # Example 14
    ground_truth_docs14 = ['data\\CONSTITUTION.pdf:6:1', 'data\\CONSTITUTION.pdf:7:3', 'data\\CONSTITUTION.pdf:11:0', 'data\\CONSTITUTION.pdf:11:1', 'data\\CONSTITUTION.pdf:10:3']
    main_query_text14 = "What did the court decide in Prabhu Datt vs Union of India?"
    query_rag_with_metrics(main_query_text14, ground_truth_docs14)

    # Example 15
    ground_truth_docs15 = ['data\\CONSTITUTION.pdf:6:1', 'data\\CONSTITUTION.pdf:13:2', 'data\\CONSTITUTION.pdf:7:2', 'data\\CONSTITUTION.pdf:2:2', 'data\\CONSTITUTION.pdf:1:3']
    main_query_text15 = "What does Article 23 prohibit?"
    query_rag_with_metrics(main_query_text15, ground_truth_docs15)

    # Example 16
    ground_truth_docs16 = ['data\\CONSTITUTION.pdf:9:1', 'data\\CONSTITUTION.pdf:8:0', 'data\\CONSTITUTION.pdf:11:1', 'data\\CONSTITUTION.pdf:15:2', 'data\\CONSTITUTION.pdf:12:0']
    main_query_text16 = "What age restriction does Article 24 impose on child labour?"
    query_rag_with_metrics(main_query_text16, ground_truth_docs16)

    # Example 17
    ground_truth_docs17 = ['data\\CONSTITUTION.pdf:11:1', 'data\\CONSTITUTION.pdf:9:1', 'data\\CONSTITUTION.pdf:8:0', 'data\\CONSTITUTION.pdf:15:2', 'data\\CONSTITUTION.pdf:11:0']
    main_query_text17 = "What does Article 25 ensure?"
    query_rag_with_metrics(main_query_text17, ground_truth_docs17)

    # Example 18
    ground_truth_docs18 = ['data\\CONSTITUTION.pdf:9:2', 'data\\CONSTITUTION.pdf:10:0', 'data\\CONSTITUTION.pdf:9:1', 'data\\CONSTITUTION.pdf:10:1', 'data\\CONSTITUTION.pdf:11:1']
    main_query_text18 = "What does Article 26 grant to religious denominations?"
    query_rag_with_metrics(main_query_text18, ground_truth_docs18)

    # Example 19
    ground_truth_docs19 = ['data\\CONSTITUTION.pdf:11:1', 'data\\CONSTITUTION.pdf:9:1', 'data\\CONSTITUTION.pdf:8:0', 'data\\CONSTITUTION.pdf:15:2', 'data\\CONSTITUTION.pdf:12:0']
    main_query_text19 = "What does Article 27 prohibit?"
    query_rag_with_metrics(main_query_text19, ground_truth_docs19)

    # Example 20
    ground_truth_docs20 = ['data\\CONSTITUTION.pdf:9:1', 'data\\CONSTITUTION.pdf:10:0', 'data\\CONSTITUTION.pdf:10:1', 'data\\CONSTITUTION.pdf:11:0', 'data\\CONSTITUTION.pdf:9:2']
    main_query_text20 = "What does Article 28 provide regarding religious instruction?"
    query_rag_with_metrics(main_query_text20, ground_truth_docs20)



