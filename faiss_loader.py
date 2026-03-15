# faiss_loader.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# ── Single consistent path ────────────────────────────────────────────────────
DB_PATH = "faiss_index"

def load_faiss_tutor(k: int = 4, temperature: float = 0.3):
    """
    Loads FAISS vector store and returns a tutor callable.
    Uses HuggingFace embeddings (free, no server needed)
    and Groq LLM (free API, fast).
    """

    # ── Embeddings (free, runs on CPU, no API key needed) ─────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Build absolute path so it works on any server ─────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, DB_PATH)

    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"FAISS index not found at: {full_path}\n"
            "Run build_vector_db.py first to create it."
        )

    # ── Load FAISS index ──────────────────────────────────────────────────────
    vectorstore = FAISS.load_local(
        full_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # ── Load Groq LLM (free API) ──────────────────────────────────────────────
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("groq_api_key"),
        temperature=temperature,
    )

    # ── Tutor callable ────────────────────────────────────────────────────────
    def tutor(query: str):
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""You are an AI Tutor on an automated learning platform.

Rules:
- Use document context when relevant
- If context is incomplete, explain using general knowledge
- Never refuse to answer
- Teach clearly and progressively

Document Context:
{context}

Question:
{query}

Answer with:
1. Direct answer
2. Explanation
3. Example (if useful)
4. Learning note (if beyond the document)
"""
        return llm.invoke(prompt)

    return tutor