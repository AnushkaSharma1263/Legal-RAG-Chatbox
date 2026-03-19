# src/text_chunker.py
# Splits long page text into overlapping smaller chunks for better retrieval

def split_into_chunks(pages: list[dict], chunk_size: int = 500, overlap: int = 100,
                      _use_session: bool = True) -> list[dict]:
    """
    Accepts chunk_size / overlap from caller, or auto-reads from Streamlit
    session_state if _use_session=True and Streamlit is running.
    """
    try:
        if _use_session:
            import streamlit as st
            chunk_size = st.session_state.get("chunk_size",  chunk_size)
            overlap    = st.session_state.get("chunk_overlap", overlap)
    except Exception:
        pass  # Not running inside Streamlit — use provided defaults
    """
    Split page texts into smaller overlapping chunks.

    Why overlap? So that context isn't lost at chunk boundaries.

    Args:
        pages:         List of page dicts from document_loader.
        chunk_size:    Target character length per chunk.
        overlap:       Number of characters shared between consecutive chunks.
        _use_session:  If True, auto-reads chunk_size/overlap from Streamlit
                       session_state (Settings page values take precedence).

    Returns:
        List of chunk dicts: [{"chunk_id": int, "text": str, "source": str, "page": int}]
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        source = page["source"]
        page_num = page["page"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:  # Skip empty chunks
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": source,
                    "page": page_num
                })
                chunk_id += 1

            # Move forward by (chunk_size - overlap) to create overlap
            start += chunk_size - overlap

    print(f"✂️  Total chunks created: {len(chunks)}")
    return chunks
