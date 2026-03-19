# pages/1_📊_Analytics.py
# Shows chat statistics: query count, avg confidence, top sources, score histogram

import streamlit as st
import json
from collections import Counter

st.set_page_config(page_title="Analytics — Legal RAG", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1.1rem 1.4rem;
    text-align: center;
}
.metric-val  { font-size: 2rem; font-weight: 700; color: #0f1f3d; }
.metric-lbl  { font-size: 0.82rem; color: #64748b; margin-top: 0.2rem; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 Analytics")
st.markdown("Insights from your current chat session.")
st.markdown("---")

history = st.session_state.get("chat_history", [])

if not history:
    st.info("No chat history yet. Ask some questions on the Chat page first.")
    st.stop()

# ── Aggregate data ────────────────────────────────────────────────────────────
total_queries   = len(history)
all_scores      = [s["score"] for e in history for s in e.get("sources", [])]
avg_score       = sum(all_scores) / len(all_scores) if all_scores else 0
source_counter  = Counter(
    s["source"] for e in history for s in e.get("sources", [])
)
page_counter    = Counter(
    f'{s["source"]} p.{s["page"]}' for e in history for s in e.get("sources", [])
)
unanswered      = sum(
    1 for e in history
    if "don't have enough information" in e["answer"].lower()
)

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics = [
    (c1, str(total_queries),       "Total Questions Asked"),
    (c2, f"{avg_score:.2f}",       "Avg Relevance Score"),
    (c3, str(len(source_counter)), "Unique Source Documents"),
    (c4, str(unanswered),          "Unanswered Questions"),
]
for col, val, lbl in metrics:
    col.markdown(
        f'<div class="metric-card"><div class="metric-val">{val}</div>'
        f'<div class="metric-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Source frequency ──────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("### 📄 Most Referenced Documents")
    if source_counter:
        max_v = max(source_counter.values())
        for doc, count in source_counter.most_common(10):
            pct = int(count / max_v * 100)
            st.markdown(f"**{doc}** — {count} hit(s)")
            st.progress(pct / 100)
    else:
        st.caption("No sources yet.")

with col_r:
    st.markdown("### 📑 Most Cited Pages")
    if page_counter:
        for page_ref, count in page_counter.most_common(8):
            st.markdown(f"- `{page_ref}` — cited **{count}** time(s)")
    else:
        st.caption("No pages cited yet.")

st.markdown("---")

# ── Score distribution ────────────────────────────────────────────────────────
st.markdown("### 📈 Relevance Score Distribution")
if all_scores:
    # Build a simple histogram with 10 buckets
    buckets = [0] * 10
    for s in all_scores:
        idx = min(int(s * 10), 9)
        buckets[idx] += 1
    max_b = max(buckets) or 1
    labels = [f"{i/10:.1f}–{(i+1)/10:.1f}" for i in range(10)]

    cols = st.columns(10)
    for i, (col, label, count) in enumerate(zip(cols, labels, buckets)):
        height_pct = int(count / max_b * 100)
        col.markdown(
            f'<div style="text-align:center;">'
            f'<div style="background:#4f46e5;border-radius:4px 4px 0 0;'
            f'height:{max(height_pct,2)}px;width:100%;"></div>'
            f'<small style="font-size:0.65rem;color:#64748b">{label}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )
else:
    st.caption("No scores recorded yet.")

st.markdown("---")

# ── Full Q&A log ──────────────────────────────────────────────────────────────
st.markdown("### 🗒️ Full Session Log")
for i, entry in enumerate(reversed(history), 1):
    ts = entry.get("timestamp", "")
    with st.expander(f"Q{total_queries - i + 1}: {entry['question'][:80]}…  {ts}", expanded=False):
        st.markdown(f"**Answer:** {entry['answer']}")
        srcs = entry.get("sources", [])
        if srcs:
            st.markdown("**Sources:** " + " · ".join(
                f"`{s['source']} p.{s['page']}` ({s['score']:.2f})" for s in srcs
            ))

# ── Download full session ─────────────────────────────────────────────────────
import datetime
export = [
    {"question": e["question"], "answer": e["answer"],
     "timestamp": e.get("timestamp",""),
     "sources": [{"source":s["source"],"page":s["page"],"score":s["score"]}
                 for s in e.get("sources",[])]}
    for e in history
]
st.download_button(
    "⬇️ Export Full Session (JSON)",
    json.dumps(export, indent=2),
    file_name=f"legal_session_{datetime.date.today()}.json",
    mime="application/json",
)
