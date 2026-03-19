# test_pipeline.py
# Run this script to test the RAG pipeline WITHOUT the UI.
# Usage: python test_pipeline.py
# Make sure you have at least one PDF in data/documents/ first.

from src.document_loader import load_all_pdfs
from src.text_chunker import split_into_chunks
from src.vector_store import VectorStore
from src.llm_client import ask_llm as ask_grok

DOCS_FOLDER = "data/documents"

def main():
    print("=" * 60)
    print("  Legal RAG Chatbot — Pipeline Test")
    print("=" * 60)

    # Step 1: Load PDFs
    print("\n[Step 1] Loading PDFs...")
    pages = load_all_pdfs(DOCS_FOLDER)
    if not pages:
        print("❌ No PDFs found. Add some PDFs to data/documents/ first.")
        return

    # Step 2: Chunk text
    print("\n[Step 2] Splitting into chunks...")
    chunks = split_into_chunks(pages, chunk_size=500, overlap=100)

    # Step 3: Build FAISS index
    print("\n[Step 3] Building FAISS vector index...")
    vs = VectorStore()
    vs.build_index(chunks)

    # Step 4: Interactive Q&A loop
    print("\n[Step 4] Ready! Type your legal question (or 'quit' to exit).\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("\n🔍 Searching relevant chunks...")
        results = vs.search(question, top_k=3)

        print(f"   Found {len(results)} relevant chunk(s).")
        print("\n🤖 Asking Groq...\n")
        answer = ask_grok(question, results)

        print(f"Answer:\n{answer}")
        print("\nSources:")
        for r in results:
            print(f"  - {r['source']} | Page {r['page']} | Score: {r['score']:.2f}")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
