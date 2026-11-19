import streamlit as st
import requests
import chromadb

# ── Configuration ──────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "company_docs"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"
N_RESULTS = 3

# ── Initialize ChromaDB (cached across reruns) ──────────────────
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)

# ── Embedding helper ───────────────────────────────────────────
def get_embedding(text):
    response = requests.post(OLLAMA_EMBED_URL, json={
        "model": EMBED_MODEL,
        "prompt": text
    })
    response.raise_for_status()
    return response.json()["embedding"]

# ── Retrieval ──────────────────────────────────────────────────
def retrieve_context(query, n_results=N_RESULTS):
    collection = get_collection()
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "distance": dist
        })
    return chunks

# ── Build RAG system prompt ────────────────────────────────────
RELEVANCE_THRESHOLD = 0.55  # Lab default was 1.2; tuned to 0.55 from observed cosine distances (Meridian questions 0.26-0.46, out-of-scope 0.62-0.67)

def build_system_prompt(chunks):
    # Only use chunks that are close enough to the question
    relevant_chunks = [c for c in chunks if c["distance"] < RELEVANCE_THRESHOLD]

    if not relevant_chunks:
        return """You are a helpful HR assistant for Meridian Analytics Group.

The user's question does not appear to match any content in the company documents.
Politely explain that you can only answer questions about Meridian company policies,
HR, expenses, and data governance, and suggest they contact HR directly."""

    context_text = "\n\n---\n\n".join([
        f"Source: {c['source']}\n{c['text']}"
        for c in relevant_chunks
    ])
    return f"""You are a helpful HR and company policy assistant for Meridian Analytics Group.

Answer the user's question using ONLY the context provided below. If the answer is not
in the context, say "I don't have information about that in the company documents."

Always cite the source document when you provide an answer.

CONTEXT:
{context_text}"""

# ── Page setup ─────────────────────────────────────────────────
st.set_page_config(page_title="Meridian HR Assistant", page_icon="📋")
st.title("📋 Meridian Analytics Group — HR Assistant")
st.caption("Ask me anything about company policies, time off, expenses, and data governance.")

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    n_chunks = st.slider("Chunks to retrieve", 1, 5, N_RESULTS)
    show_context = st.toggle("Show retrieved context", value=True)

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(f"Model: {CHAT_MODEL}")
    st.caption(f"Embedding: {EMBED_MODEL}")
    st.caption(f"Vector DB: ChromaDB (local)")

# ── Session state ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display chat history ───────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "chunks" in message and show_context:
            with st.expander("📄 Retrieved context"):
                for i, chunk in enumerate(message["chunks"]):
                    st.markdown(f"**Chunk {i+1}** | `{chunk['source']}` | Distance: `{chunk['distance']:.4f}`")
                    st.text(chunk["text"][:400] + "..." if len(chunk["text"]) > 400 else chunk["text"])
                    st.divider()

# ── Handle new input ───────────────────────────────────────────
if prompt := st.chat_input("Ask about company policies..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                chunks = retrieve_context(prompt, n_results=n_chunks)
            except Exception as e:
                st.error(f"Vector search failed: {e}")
                st.stop()

        if show_context:
            with st.expander("📄 Retrieved context"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}** | `{chunk['source']}` | Distance: `{chunk['distance']:.4f}`")
                    st.text(chunk["text"][:400] + "..." if len(chunk["text"]) > 400 else chunk["text"])
                    st.divider()

        system_prompt = build_system_prompt(chunks)

        with st.spinner("Generating answer..."):
            try:
                messages_payload = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                response = requests.post(
                    OLLAMA_CHAT_URL,
                    json={
                        "model": CHAT_MODEL,
                        "messages": messages_payload,
                        "stream": False
                    }
                )
                response.raise_for_status()
                reply = response.json()["message"]["content"]
            except requests.exceptions.ConnectionError:
                reply = "❌ Cannot connect to Ollama. Is it running? Try: `ollama serve`"
            except Exception as e:
                reply = f"❌ Error: {str(e)}"

        st.markdown(reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "chunks": chunks
    })