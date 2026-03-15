# app.py — ChatGPT style UI
import streamlit as st
from faiss_loader import load_faiss_tutor
import time

st.set_page_config(
    page_title="AI Learning Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
.stApp {background-color: #0f172a;}
.user-bubble {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 75%;
    margin-left: auto;
    font-size: 15px;
    line-height: 1.6;
}
.assistant-bubble {
    background: #1e293b;
    color: #e2e8f0;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 80%;
    font-size: 15px;
    line-height: 1.7;
    border: 1px solid #334155;
}
.label {color: #64748b; font-size: 12px; margin-bottom: 2px;}
[data-testid="stSidebar"] {background-color: #0f172a; border-right: 1px solid #1e293b;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_tutor():
    return load_faiss_tutor()

with st.sidebar:
    st.markdown("## 🎓 AI Learning Tutor")
    st.divider()
    st.markdown("**Powered by**")
    st.markdown(" LangChain + FAISS")
    st.markdown(" Groq LLaMA 3.3 70B")
    st.markdown("HuggingFace Embeddings")
    st.divider()
    st.markdown("**Knowledge Base**")
    st.markdown("• Machine Learning Systems PDF")
    st.markdown("• TDS Articles (ML/DL/NLP/LLM)")
    st.divider()
    if st.button(" Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    total_q = len([m for m in st.session_state.get("messages",[]) if m["role"]=="user"])
    st.metric("Questions Asked", total_q)

st.markdown("### 🎓 AI Learning Tutor")
st.markdown("*Ask anything about Machine Learning, Deep Learning, NLP & LLMs*")

with st.spinner("Loading AI model..."):
    tutor = get_tutor()

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.markdown("#### 💡 Suggested Questions")
    cols = st.columns(2)
    suggestions = [
        "What is backpropagation?",
        "Explain transformer architecture",
        "What is RAG in LLMs?",
        "How does LSTM work?",
        "What is attention mechanism?",
        "Explain gradient descent",
    ]
    for i, s in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(s, use_container_width=True, key=f"s{i}"):
                st.session_state.messages.append({"role":"user","content":s})
                st.rerun()

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="label" style="text-align:right">You</div><div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="label">🎓 AI Tutor</div><div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Ask about ML, DL, NLP, LLMs..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.markdown(f'<div class="label" style="text-align:right">You</div><div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        start = time.time()
        response = tutor(prompt)
        latency = time.time() - start
        answer = response.content if hasattr(response,"content") else str(response)

    st.markdown(f'<div class="label">🎓 AI Tutor</div><div class="assistant-bubble">{answer}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#475569;font-size:12px;text-align:right">⏱ {latency:.2f}s</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role":"assistant","content":answer})