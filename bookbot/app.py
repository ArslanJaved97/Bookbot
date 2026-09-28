import os

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_TOKEN"] = "hf_dummy"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tempfile
from pathlib import Path

import streamlit as st

from bot import BookBot

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="BookBot",
    page_icon="📚",
    layout="wide",
)

# ------------------------------------------------------------------
# Load bot once (models are heavy)
# ------------------------------------------------------------------
@st.cache_resource
def get_bot():
    return BookBot()


bot = get_bot()

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------------
# Sidebar — library management
# ------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Library")

    books = bot.store.list_books()
    if books:
        for b in books:
            st.markdown(f"**{b['title']}**")
            st.caption(f"{b['pages']} pages · {b['chunks']} chunks")
    else:
        st.info("No books yet. Upload one below.")

    st.divider()
    st.subheader("Add a book")
    uploaded = st.file_uploader("Drop a PDF here", type=["pdf"])
    if uploaded is not None:
        save_path = Path("books") / uploaded.name
        save_path.parent.mkdir(exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner(f"Ingesting {uploaded.name}..."):
            bot.add_pdf(save_path)
        st.success(f"Added: {uploaded.name}")
        st.rerun()

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Passages to return", 1, 15, 5)

    book_options = ["All books"] + [b["title"] for b in books]
    book_choice = st.selectbox("Search in", book_options)
    book_id = None
    if book_choice != "All books":
        for b in books:
            if b["title"] == book_choice:
                book_id = b["id"]
                break

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------
# Main area — chat
# ------------------------------------------------------------------
st.title("BookBot")
st.caption("Offline · semantic search · no filters")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask a question about your books...")

if query:
    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Bot answer
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            results = bot.retriever.search(query, top_k=top_k, book_id=book_id)

        if not results:
            answer = "No matching content found in your library."
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        else:
            lines = []
            for i, r in enumerate(results, 1):
                score = r.get("rerank_score", r.get("rrf_score", 0.0))
                snippet = r["text"].strip().replace("\n", " ")
                if len(snippet) > 1200:
                    snippet = snippet[:1200].rsplit(" ", 1)[0] + "..."
                lines.append(
                    f"**[{i}] {r['book']} — page {r['page']}**  (score {score:.2f})\n\n{snippet}"
                )
            answer = "\n\n---\n\n".join(lines)
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )