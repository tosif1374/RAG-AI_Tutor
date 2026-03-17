import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

DB_PATH = "faiss_index"

def load_faiss_tutor(k: int = 6, temperature: float = 0.2):

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, DB_PATH)

    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"FAISS index not found at: {full_path}\n"
            "Run build_vector_db.py first to create it."
        )

    vectorstore = FAISS.load_local(
        full_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
    )

    def tutor(query: str):

        greetings = ["hi", "hello", "hey", "hlo", "sup", "yo", "good morning", "greetings"]
        if query.strip().lower() in greetings:
            return "👋 Hello! I'm your AI Tutor. Ask me anything about ML, DL, NLP, or LLMs!"

        ml_keywords = [
            "machine learning", "deep learning", "nlp", "neural", "model", "training",
            "transformer", "llm", "embedding", "classification", "regression",
            "clustering", "gradient", "algorithm", "dataset", "overfitting",
            "backpropagation", "attention", "encoder", "decoder", "loss", "epoch",
            "bias", "variance", "feature", "vector", "bayes", "tree", "forest",
            "cnn", "rnn", "lstm", "bert", "gpt", "rag", "fine-tune", "tokenization"
        ]
        if not any(kw in query.lower() for kw in ml_keywords):
            return "⚠️ I specialize only in ML, DL, NLP & LLMs. Please ask a relevant question!"

        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        system_prompt = """You are an expert AI Tutor specializing in Machine Learning, Deep Learning, NLP, and LLMs.
You teach like a senior professor — clear, detailed, structured, and engaging.

STRICT RULES:
- ONLY answer questions related to ML, DL, NLP, LLMs, and AI concepts
- Use the provided context when relevant; if incomplete, use your knowledge
- Never hallucinate — if unsure, say "I'm not confident about this, please verify"
- Use proper Markdown: headings, bullets, numbered steps, code blocks
- For code concepts, always include a working code example

RESPONSE STRUCTURE:
1. Direct Answer — one clear sentence
2. Detailed Explanation — thorough breakdown
3.  Math / Algorithm Steps — step by step with formulas (if applicable)
4.  Real-World Example or Analogy
5.  Key Takeaways — 3 to 5 bullet points"""

        full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = llm.invoke(full_prompt)
        return response.content

    return tutor