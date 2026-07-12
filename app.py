import streamlit as st
from rag_pipeline import rag_chat

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="InsightRAG",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 InsightRAG")

    st.markdown("""
    ### Hybrid Knowledge Assistant

    Built using:

    - Gemini Embeddings
    - ChromaDB
    - BM25 Retrieval
    - Groq Llama 3.3 70B

    ### Features

    ✅ Hybrid Search

    ✅ Retrieval-Augmented Generation

    ✅ Source Transparency

    ✅ Context-Based Answers

    ✅ RAGAS Evaluated
    """)

    st.divider()

    st.markdown("### Example Questions")

    st.markdown("""
    - What is Retrieval-Augmented Generation?
    - Explain ChromaDB.
    - What are embeddings?
    - How does hybrid retrieval work?
    - Explain BM25 ranking.
    """)

    st.divider()

    with st.expander("⚙️ How It Works"):

        st.markdown("""
        ```text
        User Question
              ↓
        Hybrid Retrieval
        (Vector + BM25)
              ↓
        Relevant Chunks
              ↓
        Groq Llama 3.3
              ↓
        Final Answer
        ```
        """)

# ============================================================
# HEADER
# ============================================================

st.title("🤖 InsightRAG")

st.markdown(
    """
    **Hybrid Knowledge Assistant powered by ChromaDB, Gemini Embeddings, BM25 Retrieval, and Groq Llama 3.3**
    """
)

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the knowledge base..."
)

# ============================================================
# PROCESS QUERY
# ============================================================

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            answer, chunks = rag_chat(question)

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.divider()

    st.subheader("📊 Retrieval Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Chunks Retrieved",
        len(chunks)
    )

    col2.metric(
        "Embedding Model",
        "Gemini"
    )

    col3.metric(
        "LLM",
        "Llama 3.3"
    )

    st.divider()

    st.subheader("📚 Retrieved Context")

    for i, chunk in enumerate(chunks):

        with st.expander(
            f"Chunk {i+1}"
        ):

            st.markdown(chunk["text"])

            if chunk.get("metadata"):
                st.caption(
                    f"Metadata: {chunk['metadata']}"
                )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "InsightRAG • Hybrid Retrieval (Vector + BM25) • ChromaDB • Gemini Embeddings • Groq Llama 3.3"
)
