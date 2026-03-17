✨ Features

🧠 RAG Pipeline — Retrieves context from documents before answering
🗂️ FAISS Vector Store — Fast similarity search over knowledge base
🦙 Ollama LLM — Runs fully local, no API key needed
🛡️ Hallucination Guard — Filters greetings & off-topic queries
💬 Streamlit UI — Clean chat interface with conversation history


📁 Project Structure
chatbot_llm/
├── app.py               # Streamlit UI — main entry point
├── build_vector_db.py   # Chunks & embeds docs into FAISS
├── faiss_loader.py      # Loads FAISS index + RAG chain
├── chat_rag.py          # Prompt template & LLM chain
├── faiss_index/         # Saved vector index
└── requirement.txt      # Dependencies

🚀 Quick Start
bash# 1. Clone & setup
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

⚙️ Configuration
python# faiss_loader.py
load_faiss_tutor(
    llm_model="llama3",   # swap model here
    temperature=0.2,       # lower = less hallucination
    k=4                    # retrieved chunks count
)

🧰 Tech Stack
LayerTechnologyLLMOllama (Llama3 / Mistral)Vector StoreFAISSFrameworkLangChainUIStreamlit

👨‍💻 Author
Tosif — B.Tech CSE (AI & ML), Jaipur National University
🔗 @tosif1374t