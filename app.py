import streamlit as st
from faiss_loader import load_faiss_tutor
import time

st.set_page_config(
    page_title="AI Learning Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme definitions ─────────────────────────────────────────────────────────
THEMES = {
    "Light": {
        "app_bg":        "#f9fafb",
        "sidebar_bg":    "#f3f4f6",
        "sidebar_border":"#e5e7eb",
        "text":          "#111827",
        "text_muted":    "#6b7280",
        "user_bubble":   "#e5e7eb",
        "user_text":     "#111827",
        "bot_bubble":    "#ffffff",
        "bot_text":      "#111827",
        "bot_border":    "#e5e7eb",
        "bot_shadow":    "rgba(0,0,0,0.06)",
        "input_bg":      "#ffffff",
        "input_border":  "#d1d5db",
        "input_focus":   "#6366f1",
        "btn_bg":        "#ffffff",
        "btn_text":      "#374151",
        "btn_border":    "#d1d5db",
        "btn_hover_bg":  "#f3f4f6",
        "metric_bg":     "#ffffff",
        "metric_border": "#e5e7eb",
        "hr":            "#e5e7eb",
        "scroll_track":  "#f3f4f6",
        "scroll_thumb":  "#d1d5db",
        "latency_color": "#9ca3af",
    },
    "Dark": {
        "app_bg":        "#0f172a",
        "sidebar_bg":    "#1e293b",
        "sidebar_border":"#334155",
        "text":          "#e2e8f0",
        "text_muted":    "#64748b",
        "user_bubble":   "#1e3a5f",
        "user_text":     "#e2e8f0",
        "bot_bubble":    "#1e293b",
        "bot_text":      "#e2e8f0",
        "bot_border":    "#334155",
        "bot_shadow":    "rgba(0,0,0,0.3)",
        "input_bg":      "#1e293b",
        "input_border":  "#475569",
        "input_focus":   "#6366f1",
        "btn_bg":        "#1e293b",
        "btn_text":      "#94a3b8",
        "btn_border":    "#334155",
        "btn_hover_bg":  "#0f172a",
        "metric_bg":     "#1e293b",
        "metric_border": "#334155",
        "hr":            "#334155",
        "scroll_track":  "#1e293b",
        "scroll_thumb":  "#475569",
        "latency_color": "#475569",
    },
}

# ── Init session state ────────────────────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "theme"      not in st.session_state: st.session_state.theme      = "Light"

T = THEMES[st.session_state.theme]

# ── Inject CSS with current theme ─────────────────────────────────────────────
st.markdown(f"""
<style>
#MainMenu, header, footer {{visibility: hidden;}}

.stApp {{
    background-color: {T['app_bg']};
    color: {T['text']};
}}

[data-testid="stSidebar"] {{
    background-color: {T['sidebar_bg']};
    border-right: 1px solid {T['sidebar_border']};
}}
[data-testid="stSidebar"] * {{
    color: {T['text']} !important;
}}

.user-bubble {{
    background-color: {T['user_bubble']};
    color: {T['user_text']};
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 75%;
    margin-left: auto;
    font-size: 15px;
    line-height: 1.6;
}}

.assistant-bubble {{
    background-color: {T['bot_bubble']};
    color: {T['bot_text']};
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 82%;
    font-size: 15px;
    line-height: 1.8;
    border: 1px solid {T['bot_border']};
    box-shadow: 0 1px 3px {T['bot_shadow']};
}}

.label       {{ color: {T['text_muted']}; font-size: 12px; margin-bottom: 2px; }}
.label-right {{ color: {T['text_muted']}; font-size: 12px; margin-bottom: 2px; text-align: right; }}

.stChatInput textarea {{
    background-color: {T['input_bg']} !important;
    color: {T['text']} !important;
    border: 1.5px solid {T['input_border']} !important;
    border-radius: 12px !important;
    font-size: 15px !important;
}}
.stChatInput textarea:focus {{
    border-color: {T['input_focus']} !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}}

.stButton > button {{
    background-color: {T['btn_bg']};
    color: {T['btn_text']};
    border: 1px solid {T['btn_border']};
    border-radius: 8px;
    font-size: 13px;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    background-color: {T['btn_hover_bg']};
    border-color: #6366f1;
    color: #6366f1;
}}

[data-testid="stMetric"] {{
    background-color: {T['metric_bg']};
    border-radius: 10px;
    padding: 10px 14px;
    border: 1px solid {T['metric_border']};
}}

hr {{ border-color: {T['hr']}; }}

::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {T['scroll_track']}; }}
::-webkit-scrollbar-thumb {{ background: {T['scroll_thumb']}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# ── Load tutor ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_tutor():
    return load_faiss_tutor()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 AI Learning Tutor")
    st.divider()

    # Theme toggle
    current = st.session_state.theme
    other   = "Dark" if current == "Light" else "Light"
    icon    = "🌙" if current == "Light" else "☀️"
    if st.button(f"{icon}  Switch to {other} Mode", use_container_width=True):
        st.session_state.theme = other
        st.rerun()

    st.divider()
    st.markdown("**Powered by**")
    st.markdown(" LangChain + FAISS")
    st.markdown(" Groq LLaMA 3.3 70B")
    st.markdown("HuggingFace Embeddings")
    st.divider()
    st.markdown("** Knowledge Base**")
    st.markdown("• Machine Learning Systems PDF")
    st.markdown("• TDS Articles (ML/DL/NLP/LLM)")
    st.divider()
    if st.button("  Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    total_q = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Questions Asked", total_q)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("### 🎓 AI Learning Tutor")
st.markdown("*Ask anything about Machine Learning, Deep Learning, NLP & LLMs*")
st.divider()

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading AI model..."):
    tutor = get_tutor()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="label-right">You</div>'
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="label">🎓 AI Tutor</div>'
            f'<div class="assistant-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about ML, DL, NLP, LLMs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    st.markdown(
        f'<div class="label-right">You</div>'
        f'<div class="user-bubble">{prompt}</div>',
        unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        start    = time.time()
        response = tutor(prompt)
        latency  = time.time() - start
        answer   = response.content if hasattr(response, "content") else str(response)

    st.markdown(
        f'<div class="label">🎓 AI Tutor</div>'
        f'<div class="assistant-bubble">{answer}</div>',
        unsafe_allow_html=True)

    st.markdown(
        f'<div style="color:{T["latency_color"]};font-size:12px;text-align:right;">'
        f'⏱ {latency:.2f}s</div>',
        unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})