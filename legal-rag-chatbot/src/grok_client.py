# src/llm_client.py
# Sends context + question to the Groq API and returns the answer.
# Groq is OpenAI-API-compatible, ultra-fast (LPU inference).
# Supports both regular and streaming responses.

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Groq uses an OpenAI-compatible REST API
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Default model — fast and free-tier friendly.
# Other good options: "llama-3.1-70b-versatile", "mixtral-8x7b-32768"
GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a precise legal assistant. Your job is to answer legal questions
ONLY based on the provided document excerpts.

Rules you must follow:
1. Base your answer STRICTLY on the context provided — do not add outside knowledge.
2. If the context does not contain enough information to answer the question,
   respond with exactly: "I don't have enough information in the uploaded documents to answer this question."
3. Never fabricate laws, section numbers, case names, or legal interpretations.
4. Always cite the source document name and page number when giving an answer.
5. Structure longer answers with clear numbered points where appropriate.
6. Be clear, concise, and factual. Avoid unnecessary jargon.
"""


def _build_messages(question: str, context_chunks: list[dict]) -> list[dict]:
    """Build the messages list for the Groq API call."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"--- Excerpt {i} | Source: {chunk['source']} | Page: {chunk['page']} ---\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_parts)

    user_message = f"""Here are the relevant excerpts from the legal documents:

{context_text}

---

Question: {question}

Answer strictly based on the excerpts above. Cite source and page number."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]


def ask_llm(question: str, context_chunks: list[dict],
            temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """
    Send the question + context to Groq and return the full answer string.

    Args:
        question:       The user's legal question.
        context_chunks: Relevant chunks from FAISS search.
        temperature:    Sampling temperature (0.0–1.0). Lower = more factual.
        max_tokens:     Maximum tokens in the response.

    Returns:
        Answer string, or a formatted error message.
    """
    if not context_chunks:
        return "I don't have enough information in the uploaded documents to answer this question."

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=_build_messages(question, context_chunks),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error calling Groq API: {str(e)}"


def ask_llm_stream(question: str, context_chunks: list[dict],
                   temperature: float = 0.1, max_tokens: int = 1024):
    """
    Generator: yields text tokens one-by-one for live streaming in the UI.

    Usage:
        for token in ask_llm_stream(q, chunks):
            print(token, end="", flush=True)
    """
    if not context_chunks:
        yield "I don't have enough information in the uploaded documents to answer this question."
        return

    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=_build_messages(question, context_chunks),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as e:
        yield f"❌ Error calling Groq API: {str(e)}"


# Keep old names as aliases so existing imports don't break
ask_grok        = ask_llm
ask_grok_stream = ask_llm_stream
