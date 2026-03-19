# pages/2_⚙️_Settings.py
# Persistent settings panel for RAG and model parameters

import streamlit as st
import os, json

st.set_page_config(page_title="Settings — Legal RAG", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.section-title { font-size:1.1rem; font-weight:600; color:#0f1f3d; margin-bottom:0.3rem; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("# ⚙️ Settings")
st.markdown("Tune retrieval and generation parameters. Changes apply immediately to the Chat page.")
st.markdown("---")

# ── Retrieval settings ────────────────────────────────────────────────────────
st.markdown('<p class="section-title">🔍 Retrieval</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    top_k = st.slider(
        "Top-K results per query",
        min_value=1, max_value=10,
        value=st.session_state.get("top_k", 4),
        help="How many document chunks are passed as context to Grok. "
             "More = richer context but slower and uses more tokens.",
    )
with col2:
    min_score = st.slider(
        "Minimum relevance score (0–1)",
        min_value=0.0, max_value=0.95,
        value=st.session_state.get("min_score", 0.25),
        step=0.05,
        help="Chunks below this cosine similarity are dropped. "
             "Raise to get higher-confidence sources only; lower to widen recall.",
    )

st.caption(
    f"ℹ️  With these settings the chatbot will retrieve up to **{top_k}** chunks "
    f"with a similarity score ≥ **{min_score:.2f}**."
)

st.markdown("---")

# ── Generation settings ───────────────────────────────────────────────────────
st.markdown('<p class="section-title">🤖 Generation (Groq API)</p>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.0,
        value=st.session_state.get("temperature", 0.1),
        step=0.05,
        help="0 = deterministic/factual. 1 = creative/varied. "
             "Keep low for legal accuracy.",
    )
with col4:
    model_choice = st.selectbox(
        "Groq Model",
        options=[
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
        ],
        index=0,
        help=(
            "llama-3.1-8b-instant — fastest & free tier friendly. "
            "llama-3.1-70b-versatile — best quality. "
            "mixtral-8x7b-32768 — 32k context window."
        ),
    )

stream_mode = st.toggle(
    "Enable streaming responses",
    value=st.session_state.get("stream_mode", True),
    help="Show the answer token-by-token as it arrives (feels faster).",
)

st.markdown("---")

# ── Chunking settings ─────────────────────────────────────────────────────────
st.markdown('<p class="section-title">✂️ Chunking (applied on next Build)</p>',
            unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=200, max_value=1500,
        value=st.session_state.get("chunk_size", 500),
        step=50,
        help="Smaller chunks = more precise retrieval. "
             "Larger chunks = more context per chunk.",
    )
with col6:
    chunk_overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=0, max_value=400,
        value=st.session_state.get("chunk_overlap", 100),
        step=25,
        help="Overlap prevents context loss at chunk boundaries. "
             "Typically 10–20% of chunk_size.",
    )

st.markdown("---")

# ── Index management ──────────────────────────────────────────────────────────
st.markdown('<p class="section-title">🗄️ Index Management</p>', unsafe_allow_html=True)

col7, col8 = st.columns(2)

index_exists = os.path.exists("data/faiss.index")
with col7:
    if index_exists:
        meta_path = "data/index_meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            st.success(
                f"✅ Saved index found\n\n"
                f"**Docs:** {', '.join(meta.get('doc_names', []))}\n\n"
                f"**Chunks:** {meta.get('chunk_count', '?')}\n\n"
                f"**Built:** {meta.get('built_at', '?')}"
            )
        else:
            st.success("✅ Saved index found on disk.")
    else:
        st.warning("No saved index found. Build one on the Chat page.")

with col8:
    if st.button("🗑️ Delete Saved Index", use_container_width=True,
                 disabled=not index_exists,
                 help="Removes the FAISS index from disk. You'll need to rebuild."):
        for f in ["data/faiss.index", "data/chunks.pkl", "data/index_meta.json"]:
            if os.path.exists(f):
                os.remove(f)
        st.session_state.indexed     = False
        st.session_state.doc_names   = []
        st.session_state.chunk_count = 0
        st.session_state.built_at    = ""
        st.success("Index deleted. Reload the page and rebuild.")

st.markdown("---")

# ── Apply button ──────────────────────────────────────────────────────────────
if st.button("💾 Save Settings", type="primary"):
    st.session_state.top_k        = top_k
    st.session_state.min_score    = min_score
    st.session_state.temperature  = temperature
    st.session_state.stream_mode  = stream_mode
    st.session_state.chunk_size   = chunk_size
    st.session_state.chunk_overlap = chunk_overlap

    # Patch the Grok model live
    import src.grok_client as gc
    gc.GROQ_MODEL = model_choice

    st.success("✅ Settings saved — they apply to all future queries on the Chat page.")
