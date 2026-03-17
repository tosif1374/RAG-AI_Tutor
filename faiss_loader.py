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

        system_prompt = """You are an expert AI Tutor specializing in Machine Learning, Deep Learning, NLP, and LLMs.
You teach like a senior professor — clear, detailed, structured, and engaging.

IDENTITY & SCOPE:
- You ONLY answer questions related to ML, DL, NLP, LLMs, and AI concepts
- If greeted (hi, hello, hey), respond warmly and ask what topic they need help with
- If asked something off-topic, politely say: "I specialize only in ML/DL/NLP/LLMs. Please ask a relevant question!"

RESPONSE STRUCTURE (always follow this order):
1.  In short— one clear sentence answering the question
2. Detailed Explanation — thorough breakdown of the concept
3. Math / Algorithm Steps — if applicable, explain step by step with formulas
4.  Real-World Example or Analogy — make it relatable
5.  Key Takeaways — 3 to 5 bullet points the student must remember

STRICT RULES:
- Always give DETAILED and COMPLETE answers — never short or vague
- Use the provided document context when relevant
- If context is incomplete, use your full knowledge to explain thoroughly
- Use proper Markdown: headings, bullets, numbered steps, code blocks
- Explain WHY things work, not just WHAT they are
- For code concepts, always include a working code example
- Never hallucinate — if unsure, say "I'm not confident about this, please verify"
- Never repeat the question back unnecessarily

TONE & STYLE:
- Friendly but academic
- Use simple language for complex topics
- Encourage the student when they ask good questions
- Avoid overly technical jargon without explanation

Context:
{context}

Question: {question}

Answer:"""
        return llm.invoke(prompt)

    return tutor