import streamlit as st
import requests, os, yaml
from datetime import datetime

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

API_URL = config["api_url"]
UPLOAD_DIR = config["upload_dir"]

# Session state setup
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- LOGIN ---
if not st.session_state.authenticated:
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        for u in config["users"]:
            if user == u["username"] and pwd == u["password"]:
                st.session_state.authenticated = True
                st.success("Logged in successfully!")
                # st.experimental_rerun()
        st.error("Invalid credentials")
    st.stop()

# --- MAIN UI ---
st.sidebar.title("📜 Chat History")
for idx, msg in enumerate(st.session_state.messages):
    role, text = msg
    st.sidebar.markdown(f"**{role}:** {text[:40]}...")

st.title("💬 My Internal LLM")

# File upload
from ingest import ingest_doc

# File upload
st.subheader("📂 Upload Documents")
uploaded = st.file_uploader("Upload a text file", type=["txt"])
if uploaded:
    try:
        # Read with utf-8 and fallback to latin-1 if error
        try:
            text = uploaded.read().decode("utf-8")
        except UnicodeDecodeError:
            text = uploaded.read().decode("latin-1")

        chunks_added = ingest_doc(uploaded.name, text)
        st.success(f"Ingested {uploaded.name} with {chunks_added} chunks ✅")
    except Exception as e:
        st.error(f"Failed to ingest {uploaded.name}: {e}")

# Chat
st.subheader("🤖 Chat with LLM")
query = st.text_input("Ask a question:")
if st.button("Send", use_container_width=True) and query:
    try:
        res = requests.get(API_URL, params={"q": query}, timeout=60)
        answer = res.json()["answer"]
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    st.session_state.messages.append(("You", query))
    st.session_state.messages.append(("AI", answer))

# Display conversation
for role, msg in st.session_state.messages:
    if role == "You":
        st.markdown(f"🧑 **{role}:** {msg}")
    else:
        st.markdown(f"🤖 **{role}:** {msg}")
