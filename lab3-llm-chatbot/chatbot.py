import streamlit as st
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a senior business analyst assistant helping MSIS graduate
students understand business analysis concepts, requirements gathering, process modeling,
and stakeholder communication. When students ask vague questions, guide them to be more
specific. Use real-world examples from industries like healthcare, finance, and retail.
Keep answers practical and concise."""

st.set_page_config(page_title="Local LLM Chat", page_icon="🤖")
st.title("🤖 Local LLM Chatbot")
st.caption(f"Powered by {MODEL} — running on a GitHub Codespaces VM (no model API calls)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("🗑️ Clear conversation"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages_payload += st.session_state.messages

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={"model": MODEL, "messages": messages_payload, "stream": False}
                )
                response.raise_for_status()
                reply = response.json()["message"]["content"]
            except requests.exceptions.ConnectionError:
                reply = "❌ Cannot connect to Ollama. Run `ollama serve &` in the terminal."
            except Exception as e:
                reply = f"❌ Error: {str(e)}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})