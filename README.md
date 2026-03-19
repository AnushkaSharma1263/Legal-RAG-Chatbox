# ⚖️ Legal RAG Chatbot

A beginner-friendly **Retrieval-Augmented Generation (RAG)** chatbot for legal documents.  
Upload IPC sections, contracts, or any legal PDF and ask questions — answers are sourced **only from your documents**. No hallucination.

---

## 🗂️ Folder Structure

```
legal-rag-chatbot/
├── app.py                  # Streamlit UI (main entry point)
├── test_pipeline.py        # CLI test script (no UI needed)
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .gitignore
├── data/
│   └── documents/          # ← Put your PDF files here
└── src/
    ├── __init__.py
    ├── document_loader.py  # Step 1: Load & parse PDFs
    ├── text_chunker.py     # Step 2: Split text into chunks
    ├── vector_store.py     # Step 3 & 4: Embeddings + FAISS index
    └── grok_client.py      # Step 5 & 6: Grok API + answer generation
```

---

## 🚀 How to Run Locally

### 1. Clone / Download the project

```bash
git clone <repo-url>
cd legal-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90MB) on first run. This is a one-time download.

### 4. Set up your Grok API key

```bash
cp .env.example .env
```

Open `.env` and replace `your_grok_api_key_here` with your actual key from [https://console.x.ai/](https://console.x.ai/).

```
XAI_API_KEY=xai-xxxxxxxxxxxxxxxxxxxx
```

### 5. Add your legal PDFs

Place any PDF files (IPC sections, contracts, court judgments, etc.) into the `data/documents/` folder.

### 6. Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🖥️ How to Use

1. **Upload PDFs** using the sidebar file uploader
2. **Click "Build Knowledge Base"** — this processes and indexes your documents
3. **Type your legal question** in the input box at the bottom
4. **Click "Ask ➜"** — the chatbot will:
   - Search for relevant document chunks
   - Send them to Grok API
   - Return a grounded answer with source references

---

## 🧪 Test Without UI

You can also test the pipeline from the terminal:

```bash
# Add PDFs to data/documents/ first, then:
python test_pipeline.py
```

---

## ⚙️ Configuration

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | `text_chunker.py` | 500 chars | Size of each text chunk |
| `overlap` | `text_chunker.py` | 100 chars | Overlap between chunks |
| `top_k` | `vector_store.py` | 4 | Number of chunks retrieved per query |
| `GROK_MODEL` | `grok_client.py` | `grok-3-mini` | Grok model to use |
| `temperature` | `grok_client.py` | 0.1 | Lower = more factual answers |

---

## 🔧 How It Works (RAG Pipeline)

```
PDF Files
    ↓  [document_loader.py]
Extract text page by page
    ↓  [text_chunker.py]
Split into 500-char overlapping chunks
    ↓  [vector_store.py]
Encode chunks → embeddings (SentenceTransformers)
Store in FAISS index
    ↓
User asks a question
    ↓  [vector_store.py]
Encode question → embedding
Search FAISS → Top 4 similar chunks
    ↓  [grok_client.py]
Send: System Prompt + Context Chunks + Question → Grok API
    ↓
Return answer + source citations → Streamlit UI
```

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `sentence-transformers` | Text embeddings (all-MiniLM-L6-v2) |
| `faiss-cpu` | Fast vector similarity search |
| `PyMuPDF` | PDF text extraction |
| `openai` | Grok API client (OpenAI-compatible) |
| `python-dotenv` | Load API key from .env |

---

## ❓ FAQ

**Q: The chatbot says "I don't have enough information" — why?**  
A: The answer wasn't found in your uploaded documents. Try uploading more relevant PDFs.

**Q: Can I use a different LLM instead of Grok?**  
A: Yes! In `grok_client.py`, change the `base_url` and `api_key` to any OpenAI-compatible API (OpenAI, Together AI, Groq, etc.).

**Q: How many PDFs can it handle?**  
A: Comfortably handles 10–50 PDFs on a normal laptop. For larger collections, consider using `faiss-gpu` or a persistent vector DB like ChromaDB.

---

## 📄 License

MIT — free to use and modify.
