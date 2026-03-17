# 🤖 AI Learning Tutor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-green?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-orange?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-LLM-purple?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit)

**RAG-powered chatbot for Machine Learning, Deep Learning, NLP & LLMs**

</div>

---

## ✨ Features

- 🧠 **RAG Pipeline** — Retrieves context from documents before answering
- 🗂️ **FAISS Vector Store** — Fast similarity search over knowledge base
- 🦙 **Ollama LLM** — Runs fully local, no API key needed
- 🛡️ **Hallucination Guard** — Filters greetings & off-topic queries
- 💬 **Streamlit UI** — Clean chat interface with conversation history

---

## 📁 Project Structure

```
chatbot_llm/
├── app.py               # Streamlit UI — main entry point
├── build_vector_db.py   # Chunks & embeds docs into FAISS
├── faiss_loader.py      # Loads FAISS index + RAG chain
├── chat_rag.py          # Prompt template & LLM chain
├── faiss_index/         # Saved vector index
└── requirement.txt      # Dependencies
```

---

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/tosif1374t/chatbot_llm.git
cd chatbot_llm
python -m venv venv && venv\Scripts\activate

# 2. Install dependencies
pip install -r requirement.txt

# 3. Start Ollama
ollama pull llama3 && ollama serve

# 4. Build vector DB
python build_vector_db.py

# 5. Run app
streamlit run app.py
```

---

## ⚙️ Configuration

```python
# faiss_loader.py
load_faiss_tutor(
    llm_model="llama3",   # swap model here
    temperature=0.2,       # lower = less hallucination
    k=4                    # retrieved chunks count
)
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Ollama (Llama3 / Mistral) |
| Vector Store | FAISS |
| Framework | LangChain |
| UI | Streamlit |

---

## 👨‍💻 Author

**Tosif** — B.Tech CSE (AI & ML), Jaipur National University  
🔗 [@tosif1374t](https://github.com/tosif1374t)

---

<div align="center"><i>Built with ❤️ for AI/ML learners</i></div>
