import streamlit as st

from rag_pipeline import rag_chat

st.set_page_config(
    page_title="Wikipedia RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Wikipedia RAG Assistant")

st.markdown(
    """
    Hybrid Search • ChromaDB • Gemini Embeddings • Groq LLM
    """
)

with st.sidebar:

    st.header("Project Info")

    st.markdown("""
    **Knowledge Base**
    - Wikipedia Documents

    **Search**
    - Vector Search
    - BM25 Search

    **Models**
    - Gemini Embeddings
    - Llama 3.3 70B

    **Evaluation**
    - RAGAS Tested
    """)

question = st.text_input(
    "Ask a question"
)

if st.button("Generate Answer"):

    if question:

        with st.spinner("Searching knowledge base..."):

            answer, chunks = rag_chat(
                question
            )

        st.subheader("🧠 Answer")

        st.write(answer)

        st.subheader("📚 Retrieved Chunks")

        for i, chunk in enumerate(chunks):

            with st.expander(
                f"Chunk {i+1}"
            ):
                st.write(chunk["text"])

        st.subheader("⚡ Statistics")

        st.info(
            f"""
Retrieved Chunks: {len(chunks)}

Embedding Model:
Gemini Embedding 001

LLM:
Llama 3.3 70B

Search:
Hybrid (Vector + BM25)
"""
        )
