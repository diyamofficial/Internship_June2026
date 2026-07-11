import pickle
import chromadb
from rank_bm25 import BM25Okapi
import streamlit as st

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBED_MODEL = "models/gemini-embedding-001"

embedder = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL,
    google_api_key=st.secrets["GOOGLE_API_KEY"]
)


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    return client.get_collection(
        "rag_chunks"
    )


collection = load_chroma()


# ============================================================
# LOAD CHUNKS
# ============================================================

@st.cache_resource
def load_chunks():
    with open("all_chunks.pkl", "rb") as f:
        return pickle.load(f)

all_chunks = load_chunks()


# ============================================================
# BM25 INDEX
# ============================================================

@st.cache_resource
def load_bm25():

    tokenized_chunks = [
        chunk["text"].lower().split()
        for chunk in all_chunks
    ]

    return BM25Okapi(tokenized_chunks)


bm25 = load_bm25()


# ============================================================
# GROQ LLM
# ============================================================

llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile",
    temperature=0.1
)


# ============================================================
# PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """
You are a helpful AI assistant.

Use ONLY the information provided in the context below.

If the answer cannot be found in the context,
respond exactly:

I don't have enough information in the provided context to answer that.

Context:
{context}

Question:
{question}

Answer:
"""


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def retrieve(question, k=5):

    # ---------- Vector Search ----------
    query_vector = embedder.embed_query(question)

    vector_results = collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )

    vector_chunks = [
        {
            "text": doc,
            "metadata": meta,
            "distance": dist
        }
        for doc, meta, dist in zip(
            vector_results["documents"][0],
            vector_results["metadatas"][0],
            vector_results["distances"][0]
        )
    ]

    # ---------- BM25 Search ----------
    tokenized_query = question.lower().split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_chunks = [
        {
            "text": all_chunks[i]["text"],
            "metadata": all_chunks[i]["metadata"],
            "distance": None
        }
        for i in top_indices
    ]

    # ---------- Merge Results ----------
    merged = {}

    for chunk in vector_chunks + bm25_chunks:
        merged[chunk["text"]] = chunk

    return list(merged.values())[:min(k, len(merged))]


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(
        f"[{i+1}] {chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def rag_chat(question):

    retrieved_chunks = retrieve(
        question,
        k=5
    )

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    return answer, retrieved_chunks
