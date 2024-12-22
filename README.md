# Intelligent_Assistant_siemens_researchTask

Step 1: Clone the respository in the code editor of ypour choice using the command git clone.
Step2: Install all the necessary libraries in the enviroment using the command pip install -r requirements.txt.
Step3: Create a folder called data and add constitution.pdf or any other pdf in it.
Step4: Ensure that you have ollama downloaded in your system, run ollama pull gemma:2b in your command line. After the installation is complete run ollama serve to serve ollama locally.
Step 4: Create a .env file and add your Hugging face and groq api key in it for generation of embeddings and using llms.

The files app.py and app_slm.py are streamlit application using llms and optimized slm respectively. Run these files through streamlit run <script_name>

The files rag.py and sematic_rag.py implements chunking, indexing and embedding creating for the pdfs stored in data folder use the following commands
python run rag.py
python run rag.py --use_hnsw

python run sematic_rag.py
python run sematic_rag.py --use_hnsw

The files llm,slm, optimized_slm and agent.py adds an llm mistral 7b, slm gemma 2b, optimization and react agent respectivily to the vector database. Run these by python <script_name> "your query or question" in the command line.

The files test_rag is a pytest file and uses an llm to judge the responses of the rag system created. Run this file by using pytest command in the cli.

The file test_retrival tests the retrival accuracy of the system by using metrices like precision@k, recall@k and ncds@K FOR 20 TEST CASES.



Agent.py


FAISS Index Setup:

Ensure the FAISS index file (faiss_index) is present in the working directory if previously created.
If it doesn't exist, the script will automatically create a new FAISS index during execution.
Run the Script:

Execute the script from the command line by passing the query as an argument:
bash
Copy code
python Agent.py "Your question here"
Retrieve Relevant Context:

The script will use the FAISS index to find the most relevant documents and construct a context for the query.
Agent-Based Query Execution:

The Retrieval-Augmented Generation (RAG) workflow will trigger the agent, which integrates:
FAISS Search for retrieving context.
LLM (ChatGroq) to answer the query with step-by-step reasoning, references, and logical conclusions.
View the Output:

The script will display the final answer, logical reasoning, and references. Additionally, it will list the sources used with similarity scores.

Summary of What This Code Does:
This script is designed to answer queries in the legal domain, specifically focusing on the Indian Constitution. It employs a Retrieval-Augmented Generation (RAG) approach to ensure that responses are accurate, contextually relevant, and logically sound. The workflow involves:

Retrieving Context: Using a FAISS vector store to find the most relevant documents based on the input query.
Answering Queries: A specialised prompt guides the Large Language Model (LLM) to provide structured and comprehensive responses.
Agent-Oriented Execution: The agent handles reasoning, decision-making, and answering through a sequence of actions (thought-process logging).

app.py

Summary of the Code:
This app.py file implements a Retrieval-Augmented Generation (RAG) system with a focus on answering questions about the Indian Constitution. It utilises uploaded PDF documents, processes them into manageable text chunks, and creates a vector database for efficient retrieval. When a user asks a question, it retrieves relevant information using FAISS and responds with a detailed answer, complete with logical reasoning and legal references using llm called using groq mistral 7b in this case.

app_slm.py

Summary of the Code:
This app.py file implements a Retrieval-Augmented Generation (RAG) system with a focus on answering questions about the Indian Constitution. It utilises uploaded PDF documents, processes them into manageable text chunks, and creates a vector database for efficient retrieval. When a user asks a question, it retrieves relevant information using FAISS and responds with a detailed answer, complete with logical reasoning and legal references using slm called using gemma:2b in this case.


llm.py

Summary of What This Code Does
The code creates a Streamlit-powered legal question-answering system. It uses retrieval-augmented generation (RAG) to fetch relevant documents from a FAISS vector store and generates precise answers using the Groq language model. The answers are tailored for the Indian Constitution and legal domain with references to specific articles and case laws.

optimized_slm.py

 It combines document retrieval using FAISS with language generation from a dynamically optimized Ollama model. It retrieves the most relevant documents, processes them into a contextual prompt, and generates a detailed, step-by-step legal response.

 slm.py


The code creates a Streamlit-powered legal question-answering system. It uses retrieval-augmented generation (RAG) to fetch relevant documents from a FAISS vector store and generates precise answers using the ollama gemma:2b language model. The answers are tailored for the Indian Constitution and legal domain with references to specific articles and case laws.

rag.py


Command-line Arguments:

The script allows two main command-line arguments:
--reset: Clears the existing FAISS index.
--use_hnsw: Option to use HNSW indexing instead of FlatL2 for FAISS.
Run the Script:

Execute the script with desired arguments. For example, python script.py --use_hnsw or python script.py --reset.
Document Processing:

The script loads PDFs from the data folder, splits them into smaller chunks, and stores them in the FAISS index.
FAISS Indexing:

The script will either load an existing FAISS index or create a new one based on the presence of the faiss_index directory.
Summary of What This Code Does:
This Python script processes PDF documents, splits them into smaller chunks, and stores them in a FAISS index. It leverages embeddings generated by HuggingFace’s API and supports two indexing methods: FlatL2 (standard) and HNSW (Hierarchical Navigable Small World). The script also allows for resetting the FAISS index and re-indexing the documents.

Technologies and Processes Used:
FAISS: A vector store library used for efficient similarity search and indexing.
HuggingFace Embeddings: Embeddings generated using HuggingFace’s sentence-transformers for semantic search.
Document Chunking: PDFs are split into smaller text chunks for better handling and indexing.
Argument Parsing: CLI arguments allow flexibility for resetting the index and choosing the indexing method.
UUIDs: Unique identifiers are generated for document chunks to track them in the FAISS index.

sematic_rag.py

Run the Script: Execute the script from the command line, passing any desired arguments. For example:

To reset the FAISS index: python script.py --reset
To use HNSW indexing: python script.py --use_hnsw
Observe the Output: The script will:

Load the documents from the specified directory (data).
Split the documents into semantic chunks based on the content.
Add these chunks to the FAISS index, choosing either FlatL2 or HNSW indexing as specified.
Save the FAISS index to the faiss_index_semantic_hnsw directory.
Access the FAISS Index: After the script runs, the FAISS index will be available in the faiss_index_semantic_hnsw folder, ready for queries.

Summary of What This Code Does:
This code processes a collection of PDF documents to enable efficient querying of the content using semantic search. It does so by:

Document Loading: It loads PDF documents from a specified directory.
Semantic Chunking: It splits the documents into semantically meaningful chunks using the SemanticChunker. This allows the content to be indexed based on the actual meaning rather than just text segmentation.
Indexing with FAISS: It adds the document chunks to a FAISS vector store, either using the FlatL2 or HNSW indexing methods, depending on user input. This enables fast nearest neighbor search on the document chunks.
Customizable Aspects:
Chunking Method: You can adjust the chunk size and the overlap of chunks to control the granularity of the semantic chunks.
Indexing Type: You can choose between FlatL2 or HNSW indexing, depending on the desired trade-off between speed and memory usage.
Embedding Model: The embedding model used for creating document embeddings can be modified by changing the model name in the get_embedding_function function.
Resetting the Index: If you want to clear the existing FAISS index, you can use the --reset argument.

test_retrival

Customizable Querying:
Change the Query Text: Modify main_query_text1, main_query_text2, etc., to ask different legal questions.
Change Ground Truth Documents: Modify ground_truth_docs1, ground_truth_docs2, etc., to match expected documents for evaluation.
Metrics Calculation: Precision@k, Recall@k, and nDCG@k are used to evaluate the quality of retrieved results.
Run the Script: The script will process each query, evaluate the metrics, and generate the response using the defined prompt template.
Customizable Parts:
FAISS Index Path: Modify FAISS_INDEX_PATH to change where the FAISS index is stored.
Embedding Function: You can replace the Hugging Face model with a different embedding model.
Prompt Template: The legal domain prompt is customizable to answer different types of questions.
Ground Truth: Adjust the ground truth documents for testing based on the expected result.
Tests and Metrics:
The tests used here include:

Precision at K: Measures the proportion of relevant documents retrieved among the top K results.
Recall at K: Evaluates how many of the relevant documents are retrieved in the top K results.
nDCG at K: Calculates the normalized discounted cumulative gain, which accounts for the position of relevant documents.

test_rag

t combines evaluation metrics like precision, recall, F1 score, ROUGE, and BERTScore to assess the model's performance in terms of accuracy and relevance.

Key Technologies and Process:
RAG Model: This method involves querying a model with questions and validating responses against predefined expected answers.
Custom Evaluation: Uses ChatGroq for prompt-based evaluation to determine if the model's answer is correct.
Metrics: Precision, Recall, F1 Score, ROUGE, and BERTScore are used to evaluate response quality.
Groq API: A custom model is invoked via Groq for evaluating the response’s alignment with the expected answer.
Customizable Parts:
FAISS_INDEX_PATH: Path to the FAISS index can be modified to point to different directories or datasets for improved search capabilities.
EVAL_PROMPT: This template for evaluating responses can be altered based on the structure of questions or expected answer formats.
Model Settings: You can adjust parameters like temperature in the ChatGroq model, and modify the model_name to switch between different Groq models for better results.
Test Cases: The questions and expected responses in the tests can be customized to evaluate the model on different domains or sets of queries.
Metrics Evaluation: If additional metrics are required, the code can be extended to compute more advanced or domain-specific measures.

 



