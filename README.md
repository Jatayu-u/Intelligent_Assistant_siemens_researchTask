

# Intelligent Assistant - Siemens Research Task

This repository implements a Retrieval-Augmented Generation (RAG) system with various tools and techniques for document retrieval, question answering, and evaluation. It supports different indexing strategies, embeddings, and language models to optimise performance and accuracy.

---

## Table of Contents
1. [Getting Started](###getting-started)
2. [Usage Instructions](###usage-instructions)
    - [Running Applications](###running-applications)
    - [Indexing and Chunking](###indexing-and-chunking)
    - [Query Execution](###query-execution)
3. [Testing and Evaluation](###testing-and-evaluation)
    - [Running the Tests](###running-the-tests)
    - [Verifying Results](###verifying-results)
    - [Retrieval Metrics](###retrieval-metrics)
4. [Script Summaries](###script-summaries)
    - [app.py and app_slm.py](###apppy-and-app_slmpy)
    - [llm.py, slm.py, and optimized_slm.py](###llmpy-slmpy-and-optimized_slmpy)
    - [rag.py and semantic_rag.py](###ragpy-and-semantic_ragpy)
    - [test_rag.py](###test_ragpy)
    - [test_retrieval.py](###test_retrievalpy)
5. [Technologies Used](###technologies-used)
    - [FAISS](###faiss)
    - [Hugging Face](###hugging-face)
    - [Ollama](###ollama)
    - [Streamlit](###streamlit)
    - [Python](###python)
    - [Pytest](###pytest)
6. [Customization Options](##customization-options)
    - [Chunking](###chunking)
    - [Indexing Type](###indexing-type)


---

## Getting Started

Follow these steps to set up and run the repository:

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Intelligent_Assistant_siemens_researchTask```


2. Step 2: Install Dependencies
   
Install the necessary libraries by running:

```python
pip install -r requirements.txt 
```
3. Prepare the Data

Create a folder named data in the project directory and add your PDF files (e.g., constitution.pdf) to this folder.

4. Set Up Ollama

Ensure Ollama is installed on your system. Run the following commands:

Pull the required model:
```python
ollama pull gemma:2b
```
Serve Ollama locally:
```python
ollama serve
```
5. Configure API Keys
Create a .env file in the root directory and add your Hugging Face and Groq API keys:
```
env
HUGGINGFACE_API_KEY=<your_huggingface_api_key>
GROQ_API_KEY=<your_groq_api_key>
```

6. Create the vector databas by running the rag ans sematic rag files and ave them into different folders by changing the FILE_PATH option.

## Usuage Instructions

### Running Applications

Streamlit Applications
app.py: Streamlit app using an LLM (groq mistral 7b) for RAG-based question answering.
app_slm.py: Streamlit app using an SLM (ollama gemma:2b) for optimized RAG.

Run the applications:

```python
streamlit run app.py
streamlit run app_slm.py
```
### Running Scripts

Chunking, Indexing, and Embedding Creation
Use the following scripts for processing PDFs in the data folder:

Standard RAG rag.py:

```python

python rag.py
python rag.py --use_hnsw

```
Semantic RAG sematic_rag.py:

```python

python run_sematic_rag.py
python run_sematic_rag.py --use_hnsw
```

### Query Execution
The following scripts integrate various models with the vector database:

llm.py: Adds mistral 7b LLM.
slm.py: Adds gemma:2b SLM.
optimized_slm.py: Adds an optimized SLM.
agent.py: Executes RAG with a React agent.

Run queries:

```python

python <script_name>.py "Your question here"
```
## Testing and Evaluation

## `test_rag.py` 
The `test_rag.py` script uses **pytest** with 45 test cases to evaluate the Retrieval-Augmented Generation (RAG) system. This ensures that the system's accuracy, efficiency, and overall performance are thoroughly tested.

### Running the Tests:
To run the test cases for the RAG system, use the following command:
```python
pytest test_rag.py
```
## `test_retrieval.py`
The `test_retrieval.py` script evaluates the retrieval accuracy of the system using various metrics. It focuses on **Precision@K**, **Recall@K**, and **nDCG@K**, and includes 20 test cases to validate the retrieval process.

### Running the Tests:
To execute the retrieval evaluation tests, run the following command:

```python
python test_retrival.py
```
## Verifying Results

### Indexing and Chunking Strategies
To create indexes using **FAISS** for document chunking, use the following commands:

- To run the `rag.py` script:

```python
python rag.py
```
- To run the `run_rag.py` script:
```python
python sematic_rag.py

```
By default, FlatL2 indexing is used. To switch to HNSW indexing, add the --use_hnsw flag to the command:

```python
python sematic_rag.py --use_hnsw
```

## SLM vs Optimized SLM

To compare the performance of the standard **SLM** and **Optimized SLM**, modify the `test_rag.py` script to switch between the models.

Update the import in `test_rag.py` as follows to use **Optimized SLM**:

```python
from slm/optimized_slm import query_rag
```
Then, run the tests with:
```python
pytest test_rag.py
```
## SLM vs LLM

To compare the performance between **SLM** and **LLM**, modify the `test_rag.py` script to switch models.

Update the import in `test_rag.py` to use **LLM**:

Modify test_rag.py:
```python
from llm/slm import query_rag
```
Then, execute the tests with:
```python
pytest test_rag.py
```

## Retrieval Metrics

To evaluate the retrieval metrics, run the `test_retrieval.py` script and compute the following average metrics:

- **Precision@K**: Measures the proportion of relevant items among the top K retrieved results.
- **Recall@K**: Measures the proportion of relevant items that were retrieved in the top K results.
- **nDCG@K**: Normalized Discounted Cumulative Gain at rank K, evaluating the ranked relevance of retrieved results.

Run the following command to compute and display the retrieval metrics:
```python
python test_retrival.py
```

## Script Summaries


## `app.py` and `app_slm.py`
Both scripts implement Retrieval-Augmented Generation (RAG) systems with the following functionalities:

- **Process Uploaded PDF Documents**: Converts PDF documents into text chunks for easier processing.
- **FAISS Vector Database**: Creates and manages a FAISS vector database to store text chunks and enable efficient similarity search.
- **User Query Responses**: Handles user queries and provides detailed responses based on the processed documents using RAG techniques.

---

## `llm.py`, `slm.py`, and `optimized_slm.py`
These scripts focus on integrating various models with the vector database, enhancing the RAG system's ability to generate high-quality responses:

- **Model Integration**: 
  - `mistral 7b`
  - `gemma:2b`
  - `optimized gemma:2b`
- **Response Generation**: 
  - These models generate responses with step-by-step reasoning and references to support the answers, enhancing clarity and trustworthiness.

---

## `rag.py` and `semantic_rag.py`
These scripts are responsible for:

- **PDF Processing**: Converting PDF documents into manageable text chunks.
- **FAISS Index Creation**: 
  - Creates FAISS indexes for efficient similarity search using either `FlatL2` or `HNSW` indexing methods, optimizing for search performance and accuracy.
- **CLI Support**: Offers flexibility through command-line interface (CLI) arguments such as:
  - `--reset`: Resets the system's state (e.g., vector database).
  - `--use_hnsw`: Specifies the use of the HNSW indexing method for faster retrieval.

---

## `test_rag.py`
This script evaluates the overall accuracy and performance of the RAG system using various evaluation metrics:

- **Metrics Used**:
  - **Precision**: Measures the proportion of relevant results among the retrieved results.
  - **Recall**: Measures the proportion of relevant results that were retrieved.
  - **F1 Score**: The harmonic mean of Precision and Recall.

---

## `test_retrieval.py`
This script focuses on measuring retrieval effectiveness, particularly how well the vector database retrieves relevant information:

- **Metrics Used**:
  - **Precision@K**: Measures the proportion of relevant items in the top K retrieved results.
  - **Recall@K**: Measures the proportion of relevant items that were retrieved within the top K results.
  - **nDCG@K**: Normalized Discounted Cumulative Gain at rank K, a metric that evaluates the ranked relevance of retrieved results.



## Technologies Used


## FAISS
- **Purpose**: Efficient similarity search and indexing.
- **Description**: FAISS (Facebook AI Similarity Search) is used for fast nearest neighbor search, helping with large-scale similarity searches in high-dimensional spaces.

## Hugging Face
- **Purpose**: Embedding generation.
- **Description**: Hugging Face provides pre-trained models for generating embeddings, which are then used to represent documents or texts in vector space for similarity search and other NLP tasks.

## Ollama
- **Purpose**: Serving and using Large Language Models (SLMs) locally.
- **Description**: Ollama allows the use of large language models locally for inference, helping in integrating models without relying on cloud-based solutions, ensuring privacy and faster processing.

## Streamlit
- **Purpose**: Building user-friendly applications.
- **Description**: Streamlit is used for creating interactive and intuitive applications with minimal effort, particularly useful for displaying results, visualizations, and working with machine learning models.

## Python
- **Purpose**: Core language for scripting and applications.
- **Description**: Python serves as the backbone for scripting, data processing, and handling the logic behind the application. Its rich ecosystem supports libraries for machine learning, web development, and more.

## Pytest
- **Purpose**: Automated testing.
- **Description**: Pytest is used to write and run automated tests, ensuring code quality and functionality across different components of the project.

---

## Customization Options

### Chunking
- **Description**: Customize the document processing by adjusting the chunk size and overlap to optimize for different document structures and performance needs.
  - **Chunk Size**: Define how large each chunk of text should be.
  - **Chunk Overlap**: Adjust how much of the text overlaps between chunks to ensure smooth transitions in processing.

### Indexing Type
- **Description**: Choose between different indexing methods to optimize the similarity search based on performance needs.
  - **FlatL2**: A simple and direct method for indexing using L2 distance. Best for small datasets or when high precision is needed.
  - **HNSW (Hierarchical Navigable Small World)**: A more advanced method that balances search speed and accuracy, ideal for large datasets.

 



