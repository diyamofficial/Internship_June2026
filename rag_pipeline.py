import pickle
import chromadb
from rank_bm25 import BM25Okapi

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq


# -------------------------------
# Load Embedding Model
# -------------------------------

EMBED_MODEL = "models/gemini-embedding-001"

embedder = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL
)


# -------------------------------
# Load ChromaDB
# -------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "rag_chunks"
)


# -------------------------------
# Load Chunks
# -------------------------------

with open("all_chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)


# -------------------------------
# Build BM25
# -------------------------------

tokenized_chunks = [
    chunk["text"].lower().split()
    for chunk in all_chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# -------------------------------
# Load Groq
# -------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1
)


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


def retrieve(question, k=5):

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

    merged = {}

    for chunk in vector_chunks + bm25_chunks:
        merged[chunk["text"]] = chunk

    return list(merged.values())[:k]


def generate_answer(question, chunks):

    context = "\n\n".join(
        f"[{i+1}] {chunk['text']}"
        for i, chunk in enumerate(chunks)
    )

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    return response.content


def rag_chat(question):

    chunks = retrieve(question)

    answer = generate_answer(
        question,
        chunks
    )

    return answer, chunks
