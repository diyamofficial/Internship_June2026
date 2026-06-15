# RAG Phase 2 – Retrieval Augmented Generation Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built using LangChain, LangGraph, Groq LLMs, embeddings, and ChromaDB for context-aware question answering and evaluation.

---
## Pipeline Architecture

![Pipeline Diagram](pipeline.png)

## Features

* Document ingestion and preprocessing
* Text chunking for efficient retrieval
* Embedding generation using Google Generative AI Embeddings
* Vector storage and similarity search with ChromaDB
* Retrieval-Augmented Generation workflow
* LangGraph-based pipeline orchestration
* Groq LLM integration for response generation
* Wikipedia-based document retrieval
* LLM-as-a-Judge evaluation for answer relevancy and faithfulness assessment

---

## Technologies Used

* Python
* LangChain
* LangGraph
* Groq API
* ChromaDB
* Google Generative AI Embeddings
* Jupyter Notebook / Google Colab

---

## Repository Structure

```text
Phase2_RAG.ipynb     -> Main notebook containing the complete RAG workflow
README.md            -> Project documentation
requirements.txt     -> Required dependencies
.gitignore           -> Ignored files and folders
```

---

## Workflow Overview

1. Load and preprocess documents
2. Split documents into chunks
3. Generate vector embeddings
4. Store embeddings in ChromaDB
5. Retrieve relevant context based on user queries
6. Generate responses using Groq LLM
7. Evaluate generated responses using LLM-based judging

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_LINK>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Set the following environment variables or Colab secrets:

* `GROQ_API_KEY`
* `GOOGLE_API_KEY`

---

## Running the Project

Open the notebook:

```text
Phase2_RAG.ipynb
```

Run all cells sequentially to execute the complete RAG pipeline.

---

## Evaluation

The project includes an LLM-as-a-Judge evaluation approach to assess:

* Answer Relevancy
* Faithfulness
* Context Alignment

---

## Author

Diya Mathews


