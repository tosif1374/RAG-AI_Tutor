import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

DB_PATH = "faiss_index"

def load_faiss_tutor(k: int = 6, temperature: float = 0.3):

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
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""You are an expert AI Tutor specializing in Machine Learning, Deep Learning, NLP, and LLMs.
You teach like a senior professor — clear, detailed, structured, and engaging.

STRICT RULES:
- Always give a DETAILED and COMPLETE answer — never give a short or vague response
- Use the document context when relevant
- If context is incomplete, use your full knowledge to explain thoroughly
- Use proper Markdown formatting: headings, bullet points, numbered steps, code blocks
- Always include a real-world example or analogy
- Explain WHY things work, not just WHAT they are
- If it involves math or an algorithm, explain it step by step
- End with key takeaways the student must remember

Document Context:
{context}

Student Question:
{query}

YOUR DETAILED ANSWER:

## 📌 Direct Answer
[Clear 2-3 line direct answer]

## 📖 Detailed Explanation
[Thorough explanation — minimum 150 words]
[Break into sub-sections if needed]
[Explain the intuition, not just the definition]

## 🔢 How It Works (Step by Step)
[Numbered steps for algorithmic or mathematical concepts]

## 💡 Real-World Example
[Concrete relatable example with analogy]

## 🧑‍💻 Code Example
[Simple Python snippet demonstrating the concept]

## 🎯 Key Takeaways
[3-5 bullet points to remember]

## 📚 Explore Next
[1-2 related topics to study next]
"""
        return llm.invoke(prompt)

    return tutor