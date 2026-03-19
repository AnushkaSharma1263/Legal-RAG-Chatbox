# app.py — Entry point & Chat page
# Run with:  streamlit run app.py

import streamlit as st
import os, tempfile, json, datetime

st.set_page_config(
    page_title="Legal RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.main-header {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem; color: #0f1f3d;
    letter-spacing: -0.4px; margin-bottom: 0;
}
.sub-header { color: #64748b; font-size: 0.97rem; margin-top: 0.25rem; margin-bottom: 1.4rem; }

.chat-user {
    background: #eef2ff; border-left: 4px solid #4f46e5;
    padding: 0.85rem 1.1rem; border-radius: 0 8px 8px 0;
    margin: 0.75rem 0; font-weight: 600; color: #1e1b4b;
}
.chat-bot {
    background: #f0fdf4; border-left: 4px solid #16a34a;
    padding: 0.85rem 1.1rem; border-radius: 0 8px 8px 0;
    margin: 0.75rem 0; color: #14532d; line-height: 1.65;
}
.source-card {
    background: #fffbeb; border: 1px solid #fcd34d;
    border-radius: 8px; padding: 0.7rem 1rem;
    margin: 0.3rem 0; font-size: 0.83rem; color: #78350f;
}
.score-bar-wrap { background:#e5e7eb; border-radius:999px; height:6px; width:100%; margin-top:4px; }
.score-bar      { background:#4f46e5; border-radius:999px; height:6px; }

.badge-ready { display:inline-block; padding:.2rem .7rem; border-radius:999px;
               font-size:.75rem; font-weight:600; background:#dcfce7; color:#166534; }
.badge-warn  { display:inline-block; padding:.2rem .7rem; border-radius:999px;
               font-size:.75rem; font-weight:600; background:#fef9c3; color:#854d0e; }

#MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("indexed",      False),
    ("chat_history", []),
    ("doc_names",    []),
    ("chunk_count",  0),
    ("built_at",     ""),
    ("top_k",        4),
    ("min_score",    0.25),
    ("temperature",  0.1),
    ("stream_mode",  True),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Cached VectorStore ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_vs():
    """Load embedding model once; auto-restore saved index if present."""
    from src.vector_store import VectorStore
    vs = VectorStore()
    if vs.load():
        meta_path = "data/index_meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            st.session_state.indexed     = True
            st.session_state.doc_names   = meta.get("doc_names", [])
            st.session_state.chunk_count = meta.get("chunk_count", 0)
            st.session_state.built_at    = meta.get("built_at", "")
    return vs


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Legal RAG Chatbot")
    st.markdown("---")
    st.markdown("### 📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "PDF files (IPC, contracts, judgments…)",
        type=["pdf"], accept_multiple_files=True,
    )

    col_build, col_add = st.columns(2)
    build_btn = col_build.button("🔨 Build",  use_container_width=True, type="primary",
                                  help="Re-index all uploaded files from scratch")
    add_btn   = col_add.button(  "➕ Add",    use_container_width=True,
                                  help="Add new files to the existing index")

    # ── Build fresh index ──────────────────────────────────────────────────
    if build_btn:
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Indexing documents…"):
                from src.document_loader import load_pdf
                from src.text_chunker    import split_into_chunks
                vs = get_vs()
                all_chunks, doc_names = [], []
                with tempfile.TemporaryDirectory() as tmp:
                    for uf in uploaded_files:
                        p = os.path.join(tmp, uf.name)
                        open(p, "wb").write(uf.read())
                        chunks = split_into_chunks(load_pdf(p))
                        all_chunks.extend(chunks)
                        doc_names.append(uf.name)
                if all_chunks:
                    vs.build_index(all_chunks)
                    vs.save()
                    st.session_state.update({
                        "indexed":      True,
                        "doc_names":    doc_names,
                        "chunk_count":  len(all_chunks),
                        "built_at":     vs.meta.get("built_at", ""),
                        "chat_history": [],
                    })
                    st.success(f"✅ {len(all_chunks)} chunks indexed from {len(doc_names)} file(s)!")
                else:
                    st.error("No text could be extracted from the uploaded PDFs.")

    # ── Add to existing index ──────────────────────────────────────────────
    if add_btn:
        if not uploaded_files:
            st.warning("Upload PDFs to add.")
        elif not st.session_state.indexed:
            st.warning("Build an index first, then use Add.")
        else:
            with st.spinner("Adding documents to index…"):
                from src.document_loader import load_pdf
                from src.text_chunker    import split_into_chunks
                vs = get_vs()
                new_chunks, new_names = [], []
                with tempfile.TemporaryDirectory() as tmp:
                    for uf in uploaded_files:
                        if uf.name in st.session_state.doc_names:
                            continue
                        p = os.path.join(tmp, uf.name)
                        open(p, "wb").write(uf.read())
                        new_chunks.extend(split_into_chunks(load_pdf(p)))
                        new_names.append(uf.name)
                if new_chunks:
                    combined = vs.chunks + new_chunks
                    vs.build_index(combined)
                    vs.save()
                    st.session_state.doc_names  += new_names
                    st.session_state.chunk_count = len(combined)
                    st.session_state.built_at    = vs.meta.get("built_at", "")
                    st.success(f"➕ Added {len(new_chunks)} chunks from {len(new_names)} file(s).")
                else:
                    st.info("No new documents found (all already indexed).")

    st.markdown("---")

    # ── Status panel ──────────────────────────────────────────────────────
    if st.session_state.indexed:
        st.markdown(
            f'<span class="badge-ready">● {st.session_state.chunk_count} chunks ready</span>',
            unsafe_allow_html=True,
        )
        for name in st.session_state.doc_names:
            st.caption(f"📄 {name}")
        if st.session_state.built_at:
            st.caption(f"🕒 Built: {st.session_state.built_at}")
    else:
        st.markdown('<span class="badge-warn">● No index loaded</span>',
                    unsafe_allow_html=True)
        st.caption("Upload PDFs and click Build.")

    st.markdown("---")

    # ── Quick settings (also editable on Settings page) ───────────────────
    with st.expander("⚙️ Quick Settings", expanded=False):
        st.session_state.top_k      = st.slider("Results per query (top-k)", 1, 8,
                                                  st.session_state.top_k)
        st.session_state.min_score  = st.slider("Min relevance score", 0.0, 0.9,
                                                  st.session_state.min_score, step=0.05)
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0,
                                                   st.session_state.temperature, step=0.05)
        st.session_state.stream_mode = st.toggle("Streaming responses",
                                                   value=st.session_state.stream_mode)

    st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # Export chat as JSON
    if c2.button("💾 Export", use_container_width=True, help="Export chat to JSON"):
        if st.session_state.chat_history:
            export_data = [
                {"question": e["question"], "answer": e["answer"],
                 "timestamp": e.get("timestamp", ""),
                 "sources": [{"source": s["source"], "page": s["page"],
                               "score": s["score"]} for s in e.get("sources", [])]}
                for e in st.session_state.chat_history
            ]
            st.download_button(
                "⬇️ Download JSON",
                json.dumps(export_data, indent=2),
                file_name=f"legal_chat_{datetime.date.today()}.json",
                mime="application/json",
            )
        else:
            st.info("No chat to export yet.")

    st.caption("Groq API · FAISS · SentenceTransformers")


# ── Main: header ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">⚖️ Legal Document Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ask questions grounded strictly in your uploaded legal documents. '
    'No hallucination — every answer cites its source.</p>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.info("👈 Upload PDFs in the sidebar → Build → then ask your question below.")
else:
    for entry in st.session_state.chat_history:
        st.markdown(f'<div class="chat-user">🧑‍💼 {entry["question"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bot">⚖️ {entry["answer"]}</div>',
                    unsafe_allow_html=True)

        if entry.get("sources"):
            with st.expander(f"📑 {len(entry['sources'])} source excerpt(s)", expanded=False):
                for src in entry["sources"]:
                    pct = int(src["score"] * 100)
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>📄 {src["source"]}</strong> — Page {src["page"]}'
                        f'<div class="score-bar-wrap">'
                        f'<div class="score-bar" style="width:{pct}%"></div></div>'
                        f'<small>Relevance score: {src["score"]:.2f}</small><br><br>'
                        f'{src["text"][:450]}{"…" if len(src["text"])>450 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        st.markdown("")


# ── Input bar ─────────────────────────────────────────────────────────────────
st.markdown("---")
col_q, col_btn = st.columns([5, 1])
with col_q:
    user_question = st.text_input(
        "question", label_visibility="collapsed",
        placeholder="e.g. What is the punishment under IPC Section 302?",
    )
with col_btn:
    ask_btn = st.button("Ask ➜", use_container_width=True, type="primary")


# ── Query handler ─────────────────────────────────────────────────────────────
if ask_btn and user_question.strip():
    if not st.session_state.indexed:
        st.error("Build the knowledge base first (upload PDFs → Build).")
    else:
        vs        = get_vs()
        top_k     = st.session_state.top_k
        min_sc    = st.session_state.min_score
        temp      = st.session_state.temperature
        do_stream = st.session_state.stream_mode

        with st.spinner("🔍 Searching documents…"):
            chunks = vs.search(user_question, top_k=top_k, min_score=min_sc)

        if do_stream:
            from src.llm_client import ask_llm_stream as ask_grok_stream
            st.markdown(f'<div class="chat-user">🧑‍💼 {user_question}</div>',
                        unsafe_allow_html=True)
            answer_box  = st.empty()
            full_answer = ""
            for token in ask_grok_stream(user_question, chunks, temperature=temp):
                full_answer += token
                answer_box.markdown(
                    f'<div class="chat-bot">⚖️ {full_answer}▌</div>',
                    unsafe_allow_html=True,
                )
            answer_box.markdown(
                f'<div class="chat-bot">⚖️ {full_answer}</div>',
                unsafe_allow_html=True,
            )
        else:
            from src.llm_client import ask_llm as ask_grok
            with st.spinner("⚖️ Consulting Groq…"):
                full_answer = ask_grok(user_question, chunks, temperature=temp)
            st.markdown(f'<div class="chat-user">🧑‍💼 {user_question}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bot">⚖️ {full_answer}</div>',
                        unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "question":  user_question,
            "answer":    full_answer,
            "sources":   chunks,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        st.rerun()

elif ask_btn:
    st.warning("Please type a question first.")
