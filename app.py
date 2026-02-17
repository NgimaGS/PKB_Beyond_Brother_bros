import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from knowledge_base import KnowledgeBase
import time

# --- PROFESSIONAL PAGE CONFIG ---
st.set_page_config(
    page_title="NPL Engine // Workspace",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL UI CSS (SAAS STANDARD) ---
st.markdown("""
    <style>
    /* Import Inter Font - The gold standard for UI */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Reset & Base Styles */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0e1117; /* Deep Slate Blue-Grey */
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e0e6ed;
        line-height: 1.6;
    }

    /* Typography Hierarchy */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #f8fafc;
        letter-spacing: -0.01em;
    }

    p, div, span, label {
        font-weight: 400;
        color: #cbd5e1;
    }

    /* Layout Cleaning */
    footer { visibility: hidden; }
    .block-container { padding: 3rem 1rem !important; max-width: 1000px !important; margin: 0 auto; }

    /* Sidebar - Professional Ingestion Hub */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
        width: 320px !important;
    }

    /* Result Cards - Crisp & Technical */
    .nexus-card {
        background: #1f2937; /* Gray 800 */
        border: 1px solid #374151;
        border-radius: 12px; /* Tighter radius for professional feel */
        padding: 1.25rem;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s;
    }
    .nexus-card:hover {
        border-color: #4b5563;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transform: translateY(-2px);
    }

    /* Metric Tags */
    .match-tag {
        background: rgba(16, 185, 129, 0.1); /* Emerald Tint */
        color: #34d399; /* Emerald 400 */
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    /* Metadata Labels */
    .meta-label {
        font-size: 0.75rem;
        color: #94a3b8; /* Slate 400 */
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Primary Action Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px !important;
        background: #2563eb !important; /* Royal Blue */
        color: white !important;
        font-weight: 500 !important;
        border: 1px solid #1d4ed8 !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.875rem;
        transition: all 0.2s;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .stButton>button:hover {
        background: #1d4ed8 !important;
        border-color: #1e40af !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    .stButton>button:active { transform: scale(0.98); }

    /* Chat Input Field */
    .stChatFloatingInputContainer { background-color: transparent !important; }
    .stTextInput>div>div>input {
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 10px !important;
        color: #f3f4f6 !important;
        font-size: 0.95rem;
        padding: 0.75rem;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .stTextInput>div>div>input:focus {
        border-color: #3b82f6 !important; /* Blue 500 */
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #4b5563; }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session State
if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_syncing" not in st.session_state:
    st.session_state.is_syncing = False

# --- SIDEBAR: INGESTION HUB ---
with st.sidebar:
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="font-size: 20px; margin-bottom: 5px; color: white;">💠 NPL</h2>
            <p style="font-size: 11px; color: #64748b; font-weight: 500; letter-spacing: 0.05em;">ENTERPRISE KNOWLEDGE CORE</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload Knowledge Base", accept_multiple_files=True, label_visibility="collapsed")

    if st.button("Initialize System"):
        if uploaded_files:
            st.session_state.is_syncing = True
            # Clear previous state
            st.session_state.kb.documents_metadata = []
            st.session_state.kb.cleaning_report = []
            st.session_state.kb.file_contents = {}
            st.session_state.messages = []

            st.markdown("### Process Log")
            bar = st.progress(0)
            status_text = st.empty()

            total = len(uploaded_files)
            for idx, f in enumerate(uploaded_files):
                status_text.markdown(
                    f"<span style='color:#94a3b8; font-size: 13px;'>Processing: <b>{f.name}</b></span>",
                    unsafe_allow_html=True)
                content = ""
                if f.name.endswith(".pdf"):
                    reader = PdfReader(f)
                    for i, page in enumerate(reader.pages):
                        p_text = page.extract_text() or ""
                        st.session_state.kb.process_text(f.name, p_text, i + 1)
                else:
                    content = f.read().decode("utf-8")
                    st.session_state.kb.process_text(f.name, content, 1)

                bar.progress((idx + 1) / total)
                time.sleep(0.05)

            status_text.markdown(
                "<span style='color:#34d399; font-size: 13px; font-weight: 500;'>✔ Vector Index Built Successfully</span>",
                unsafe_allow_html=True)
            st.session_state.kb.build_index()

            # Welcome Message with Stats
            welcome_html = f"""
            <div style="background: rgba(37, 99, 235, 0.1); border: 1px solid rgba(37, 99, 235, 0.2); border-radius: 8px; padding: 12px;">
                <p style="color: #60a5fa; margin: 0; font-size: 14px; font-weight: 500;">
                    System Online. Index contains <b>{len(st.session_state.kb.documents_metadata)}</b> semantic units across <b>{len(uploaded_files)}</b> documents.
                </p>
            </div>
            """
            st.session_state.messages.append({"role": "assistant", "content": welcome_html, "type": "stats"})

            st.session_state.is_syncing = False
            st.rerun()

    # Show Active Files
    if st.session_state.kb.file_contents:
        st.markdown("<div style='margin-top: 30px; border-top: 1px solid #30363d; padding-top: 20px;'>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase;'>Active Sources</p>",
            unsafe_allow_html=True)
        for f in st.session_state.kb.file_contents.keys():
            st.markdown(
                f"<div style='display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'><span style='color: #2563eb;'>•</span> <span style='font-size: 13px; color: #cbd5e1;'>{f}</span></div>",
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- MAIN WORKSPACE ---

# If documents are loaded and we aren't syncing, show the chat
if st.session_state.kb.file_contents and not st.session_state.is_syncing:

    # Header
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h3 style="margin-bottom: 0.5rem;">Semantic Research Assistant</h3>
            <p style="font-size: 14px; color: #94a3b8;">Query your knowledge base with high-dimensional vector precision.</p>
        </div>
    """, unsafe_allow_html=True)

    # Chat Container
    chat_box = st.container(height=600, border=False)
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
                if msg.get("type") == "stats":
                    with st.expander("View Index Analytics"):
                        st.dataframe(pd.DataFrame(st.session_state.kb.cleaning_report), width='stretch',
                                     hide_index=True)
                        st.bar_chart(st.session_state.kb.get_top_keywords_df().set_index('Keyword'), color="#2563eb")

    # Chat Input & Logic
    if prompt := st.chat_input("Enter your research query..."):
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get Results
        results = st.session_state.kb.search(prompt)

        if results:
            response_html = f"<p style='margin-bottom: 12px; color: #e2e8f0;'>Top semantic matches for: <b>{prompt}</b></p>"
            for res in results:
                response_html += f"""
                <div class="nexus-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                        <div style="display: flex; flex-direction: column;">
                            <span class="meta-label" style="color: #60a5fa;">{res['file']}</span>
                            <span class="meta-label" style="font-size: 0.7rem; margin-top: 2px;">Page {res['page']}</span>
                        </div>
                        <span class="match-tag">{res['score']}</span>
                    </div>
                    <p style="font-size: 0.95rem; color: #d1d5db; line-height: 1.6; margin: 0;">"{res['text']}"</p>
                </div>"""
            st.session_state.messages.append({"role": "assistant", "content": response_html})
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": "No correlations found exceeding the relevance threshold."})

        # Rerun to update chat
        st.rerun()

else:
    # Landing View
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.markdown("<div style='height: 20vh;'></div>", unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 60px; margin-bottom: 20px;">💠</div>
                <h1 style="color: white; font-weight: 700;">Nexus Engine</h1>
                <p style="color: #94a3b8; font-size: 16px; margin-top: 10px;">
                    Enterprise-grade semantic retrieval system.<br>
                    Initialize the vector core via the sidebar to begin.
                </p>
            </div>
        """, unsafe_allow_html=True)