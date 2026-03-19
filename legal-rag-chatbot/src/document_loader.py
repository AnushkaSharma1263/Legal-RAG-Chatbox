# src/document_loader.py
# Handles loading and parsing PDF files into raw text chunks

import fitz  # PyMuPDF
import os
from pathlib import Path


def load_pdf(file_path: str) -> list[dict]:
    """
    Load a PDF and return a list of page dicts with text and metadata.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of dicts: [{"page": int, "text": str, "source": str}]
    """
    pages = []
    doc = fitz.open(file_path)
    file_name = Path(file_path).name

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        # Skip empty pages
        if text:
            pages.append({
                "page": page_num + 1,
                "text": text,
                "source": file_name
            })

    doc.close()
    return pages


def load_all_pdfs(folder_path: str) -> list[dict]:
    """
    Load all PDFs from a folder.

    Args:
        folder_path: Path to folder containing PDF files.

    Returns:
        Combined list of page dicts from all PDFs.
    """
    all_pages = []
    folder = Path(folder_path)

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDF files found in: {folder_path}")
        return []

    for pdf_file in pdf_files:
        print(f"📄 Loading: {pdf_file.name}")
        pages = load_pdf(str(pdf_file))
        all_pages.extend(pages)
        print(f"   ✅ Loaded {len(pages)} pages")

    print(f"\n📚 Total pages loaded: {len(all_pages)}")
    return all_pages
