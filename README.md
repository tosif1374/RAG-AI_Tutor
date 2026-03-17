# 🤖 DPChatbot — AI Learning Tutor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-green?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-orange?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM-red?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?style=for-the-badge&logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b?style=for-the-badge&logo=streamlit)

**RAG-powered AI Tutor for Machine Learning, Deep Learning, NLP & LLMs**

</div>

---

## ✨ Features

- 🧠 **RAG Pipeline** — Retrieves context from documents before answering
- 🗂️ **FAISS Vector Store** — Fast similarity search over knowledge base
- 🤗 **HuggingFace Embeddings** — High-quality sentence embeddings
- ⚡ **Groq LLM** — Ultra-fast inference via Groq API
- 🛡️ **Hallucination Guard** — Filters greetings & off-topic queries
- ☁️ **Deployment Ready** — Procfile included for cloud deployment

---

## 📁 Project Structure

```
DPCHATBOT/
├── app.py                        # Streamlit UI — main entry point
├── build_vector_db.py            # Chunks & embeds docs into FAISS
├── faiss_loader.py               # Loads FAISS index + RAG chain
├── faiss_index/                  # Saved FAISS vector index
├── Machine-Learning-Systems.pdf  # Knowledge base document
├── Procfile                      # Deployment config (Heroku/Render)
├── requirements.txt              # Python dependencies
└── .streamlit/
    └── config.toml               # Streamlit theme config
```

---

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/tosif1374t/dpchatbot.git
cd dpchatbot
python -m venv venv && venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key in .env
GROQ_API_KEY=your_groq_api_key_here

# 4. Build vector DB
python build_vector_db.py

# 5. Run app
streamlit run app.py
```

> 🔑 Get your free Groq API key at [console.groq.com](https://console.groq.com)

---

## ☁️ Deploy on Render / Heroku

The `Procfile` is already configured for deployment:

```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Just connect your GitHub repo to [Render](https://render.com) or [Heroku](https://heroku.com) and add `GROQ_API_KEY` as an environment variable.

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq (Llama3 / Mixtral) |
| Embeddings | HuggingFace Sentence Transformers |
| Vector Store | FAISS |
| Framework | LangChain |
| UI | Streamlit |
| Deployment | Streamlit|

---

## 👨‍💻 Author

**Tosif** — B.Tech CSE (AI & ML), Jaipur National University  
🔗 [@tosif1374t](https://github.com/tosif1374t)

---

<div align="center"><i>Built with ❤️ for AI/ML learners</i></div>
