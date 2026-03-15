# app.py
import streamlit as st
import time
from faiss_loader import load_faiss_tutor

st.set_page_config(page_title="AI Learning Tutor", page_icon="🎓", layout="wide")

st.title("🎓 AI Learning Tutor")
st.markdown("Ask any question")

@st.cache_resource
def get_tutor():
    return load_faiss_tutor()

with st.spinner("Loading AI Tutor..."):
    tutor = get_tutor()

st.success("Tutor ready! Ask your question below.")

query = st.text_input("Ask a question:", placeholder="e.g. What is backpropagation?")

if query:
    with st.spinner("Thinking..."):
        start = time.time()
        answer = tutor(query)
        latency = time.time() - start

    st.markdown("### Answer")
    st.write(answer.content if hasattr(answer, "content") else answer)
    st.markdown(f"⏱ **Response Time:** `{latency:.2f} seconds`")