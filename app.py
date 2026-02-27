import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from knowledge_base import KnowledgeBase
import time
import os
import subprocess

# --- PROFESSIONAL PAGE CONFIG ---
st.set_page_config(
    page_title="NPL Engine // Workspace",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CORE DATA PROCESSING ---
def process_single_file(file_obj, filename=None, full_path=None):
    """
    NLP Orchestration: Orchestrates the ingestion of different file types.
    This bridge converts raw binary/text files into processed semantic units in the KnowledgeBase.
    """
    fname = filename if filename else file_obj.name
    
    if fname.endswith(".pdf"):
        # PDF Parsing: Extracting raw text from specific pages
        reader = PdfReader(file_obj)
        for i, page in enumerate(reader.pages):
            p_text = page.extract_text() or ""
            # Sends raw text to the NLP pipeline in knowledge_base.py
            st.session_state.kb.process_text(fname, p_text, i + 1, full_path=full_path)
    elif fname.endswith(".md"):
        # Markdown Parsing: Reading raw text content
        if hasattr(file_obj, 'read'):
            content = file_obj.read().decode("utf-8")
        else:
            with open(file_obj, 'r', encoding='utf-8') as f:
                content = f.read()
        st.session_state.kb.process_text(fname, content, 1, full_path=full_path)
    return fname

# ... (CSS section stays as is)

# --- PROFESSIONAL UI CSS (SAAS STANDARD) ---
# ... (rest of CSS snippet)
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

    /* Layout Expansion & Scroll Optimization */
    footer { visibility: hidden; }
    .block-container { 
        padding-top: 4rem !important; 
        padding-bottom: 1rem !important; 
        max-width: 1400px !important; 
        margin: 0 auto; 
    }
    
    /* Ensure only the chat container scrolls by preventing document-level overflow if possible */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden;
    }

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
        border-radius: 10px;
        padding: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s;
    }
    .nexus-card:hover {
        border-color: #3b82f6;
    }

    /* Metric Tags */
    .match-tag {
        background: rgba(16, 185, 129, 0.1); /* Emerald Tint */
        color: #34d399; /* Emerald 400 */
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
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
else:
    # Safely re-initialize if the existing object is from an older version of the class
    import inspect
    sig = inspect.signature(st.session_state.kb.process_text)
    if "full_path" not in sig.parameters:
        st.session_state.kb = KnowledgeBase()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_syncing" not in st.session_state:
    st.session_state.is_syncing = False
if "is_searching" not in st.session_state:
    st.session_state.is_searching = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# --- SIDEBAR: MINIMAL BRANDING ---
with st.sidebar:
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="font-size: 20px; margin-bottom: 5px; color: white;">💠 Nexus</h2>
            <p style="font-size: 11px; color: #64748b; font-weight: 500; letter-spacing: 0.05em;">PRECISION RETRIEVAL CORE</p>
        </div>
        <div style="margin-top: auto; border-top: 1px solid #30363d; padding-top: 20px;">
            <p style="font-size: 10px; color: #4b5563;">SYSTEM VERSION: 2.1.0-TABBED</p>
        </div>
    """, unsafe_allow_html=True)

# --- GLOBAL HEADER ---
num_docs = len(st.session_state.kb.file_contents) if st.session_state.kb.file_contents else 0
st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1rem; margin-top: -1rem;">
        <h2 style="margin-bottom: 0.25rem; font-size: 24px; color: white;">Nexus Semantic Assistant</h2>
        <p style="font-size: 13px; color: #64748b; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;">
            Engine Status: Online • Active Intelligence: {num_docs} Documents
        </p>
    </div>
""", unsafe_allow_html=True)

# --- TABBED WORKSPACE ---
tab_research, tab_analytics, tab_settings = st.tabs(["💠 Research Hub", "📊 Vector Analytics", "⚙️ System Settings"])

with tab_settings:
    st.markdown("### ⚙️ Knowledge Ingestion")
    st.markdown("<p class='meta-label'>Manual Upload</p>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True, label_visibility="visible", help="Upload PDF or Markdown files to be processed and indexed by the semantic engine.")
    
    st.markdown("<p class='meta-label' style='margin-top:15px;'>Mount Local Directory</p>", unsafe_allow_html=True)
    directory_path = st.text_input("Directory Path", placeholder="C:\\path\\to\\notes", label_visibility="visible", help="Path to a folder on your computer. The engine will recursively scan for docs.")

    if st.button("Initialize System", help="Clears the current engine state and builds a new semantic index from your sources."):
        # Resetting the Engine state for a fresh ingestion
        files_to_process = []
        if uploaded_files:
            for f in uploaded_files:
                files_to_process.append({'obj': f, 'name': f.name, 'full_path': None})
        if directory_path:
            # Recursive Directory Traversal
            if os.path.isdir(directory_path):
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        if file.endswith((".pdf", ".md")):
                            full_path = os.path.join(root, file)
                            files_to_process.append({'obj': full_path, 'name': file, 'full_path': full_path})
            else:
                st.error("Invalid directory path.")

        if files_to_process:
            st.session_state.is_syncing = True
            st.session_state.kb.documents_metadata = []
            st.session_state.kb.cleaning_report = []
            st.session_state.kb.file_contents = {}
            st.session_state.messages = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Sequence-Based Ingestion: Processing each document through the NLP pipeline
            total = len(files_to_process)
            for idx, item in enumerate(files_to_process):
                status_text.markdown(f"<span style='color:#94a3b8; font-size: 13px;'>Processing: {item['name']}</span>", unsafe_allow_html=True)
                process_single_file(item['obj'], item['name'], full_path=item['full_path'])
                progress_bar.progress((idx + 1) / total)
                time.sleep(0.01)

            # INDEX BUILDING: Calculations start here (TF-IDF Matrix creation)
            st.session_state.kb.build_index()
            
            # Semantic Statistics for the user
            welcome_html = f"""
            <div style="background: rgba(37, 99, 235, 0.1); border: 1px solid rgba(37, 99, 235, 0.2); border-radius: 8px; padding: 12px; margin-bottom: 20px;">
                <p style="color: #60a5fa; margin: 0; font-size: 14px; font-weight: 500;">
                    ✔ System Online. Index contains <b>{len(st.session_state.kb.documents_metadata)}</b> semantic units across <b>{len(files_to_process)}</b> documents.
                </p>
            </div>
            """
            st.session_state.messages.append({"role": "assistant", "content": welcome_html, "type": "stats"})
            st.session_state.is_syncing = False
            st.success("System Initialized! Head to the Research Hub to start querying.")

# --- TAB 2: VECTOR ANALYTICS ---
with tab_analytics:
    st.markdown("### 📊 Vector Analytics")
    if not st.session_state.kb.file_contents:
        st.info("No data indexed yet. Use the System Settings tab to ingest documents.")
    else:
        # FEATURE IMPORTANCE [TOPIC MODELLING]: Showing words that define the knowledge base
        st.markdown("<p class='meta-label'>Top Keywords by Importance</p>", unsafe_allow_html=True)
        st.bar_chart(st.session_state.kb.get_top_keywords_df().set_index('Keyword'), color="#2563eb", height=300)
        st.caption("The bar chart represents 'Feature Importance'—words that define the unique themes of your documents.")

        st.markdown("<p class='meta-label' style='margin-top: 30px;'>Cleaning & Normalization Report</p>", unsafe_allow_html=True)
        # Scrollable Data Report
        st.dataframe(pd.DataFrame(st.session_state.kb.cleaning_report), use_container_width=True, height=400, hide_index=True)
        st.caption("The report displays the results of the Preprocessing Pipeline (Noise Removal & Lemmatization).")
        

# --- TAB 1: RESEARCH HUB ---
with tab_research:
    if not st.session_state.kb.file_contents:
        # Landing View
        st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 60px; margin-bottom: 20px;">💠</div>
                <h1 style="color: white; font-weight: 700;">Nexus Engine</h1>
                <p style="color: #94a3b8; font-size: 16px; margin-top: 10px;">
                    Enterprise-grade semantic retrieval system.<br>
                    Go to <b>System Settings</b> to initialize the vector core.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Render Chat History - Balanced height to prevent page-level scaling
        chat_box = st.container(height=650, border=False)
        with chat_box:
            for m_idx, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    if msg.get("type") == "results":
                        # Persistent Result Rendering
                        st.markdown(msg["content"], unsafe_allow_html=True)
                        data = msg["data"]
                        sorted_files = msg["sorted_files"]
                        
                        for f_idx, fname in enumerate(sorted_files[:3]):
                            file_chunks = data[fname]
                            avg_score = sum(r['score'] for r in file_chunks) / len(file_chunks)
                            full_path = file_chunks[0].get('full_path')
                            
                            # Unified Nexus Card Container
                            with st.container(border=True):
                                # Header Component
                                st.markdown(f"""
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                        <div style="display: flex; flex-direction: column;">
                                            <span class="meta-label" style="color: #60a5fa; font-size: 0.85rem;">{fname}</span>
                                            <span class="meta-label" style="font-size: 0.7rem;">{len(file_chunks)} relevant segments</span>
                                        </div>
                                        <span class="match-tag" style="background: rgba(59, 130, 246, 0.1); color: #60a5fa; border-color: rgba(59, 130, 246, 0.2);">{int(avg_score * 100)}% Match</span>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Actions Component
                                if full_path:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button(f"📄 Open File", key=f"open_f_{m_idx}_{f_idx}", help=f"Open {fname} in your default application.", use_container_width=True):
                                            try:
                                                os.startfile(full_path)
                                            except Exception as e:
                                                st.error(f"Error: {e}")
                                    with col2:
                                        if st.button(f"📁 Open Folder", key=f"open_d_{m_idx}_{f_idx}", help=f"Open containing folder and highlight {fname}.", use_container_width=True):
                                            try:
                                                norm_path = os.path.normpath(full_path)
                                                subprocess.run(['explorer', '/select,', norm_path])
                                            except Exception as e:
                                                st.error(f"Error: {e}")
                                else:
                                    st.caption("💡 Manually uploaded file - Native opening unavailable.")
                                
                                # Details Component
                                with st.expander("🔍 View Content Segments", expanded=False):
                                    for res in file_chunks:
                                        st.markdown(f"""
                                            <div style="margin-bottom: 15px; border-bottom: 1px solid #374151; padding-bottom: 10px;">
                                                <span class="meta-label" style="font-size: 0.7rem; display: block; margin-bottom: 4px;">Page {res['page']} // {int(res['score']*100)}% Match</span>
                                                <p style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.5; margin: 0;">"{res['text']}"</p>
                                            </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(msg["content"], unsafe_allow_html=True)

        # SEMANTIC SEARCH INTERFACE [QUESTION-ANSWERING HUB]
        if st.session_state.is_searching and st.session_state.pending_query:
            with st.status("💠 Nexus Engine Correlating Vectors...", expanded=True) as status:
                prompt = st.session_state.pending_query
                results = st.session_state.kb.search(prompt, top_n=50)

                if results:
                    grouped_results = {}
                    for res in results:
                        fname = res['file']
                        if fname not in grouped_results: grouped_results[fname] = []
                        grouped_results[fname].append(res)
                    
                    sorted_files = sorted(grouped_results.keys(), 
                                        key=lambda f: sum(r['score'] for r in grouped_results[f]) / len(grouped_results[f]), 
                                        reverse=True)

                    header_text = f"<p style='margin-bottom: 12px; color: #e2e8f0;'>Top semantic matches for: <b>{prompt}</b></p>"
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "type": "results",
                        "content": header_text,
                        "data": grouped_results,
                        "sorted_files": sorted_files
                    })
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "No relevant correlations found."})
                
                status.update(label="Engine Correlation Complete", state="complete", expanded=False)
            
            st.session_state.is_searching = False
            st.session_state.pending_query = None
            st.rerun()
        else:
            if prompt := st.chat_input("Enter your research query..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.is_searching = True
                st.session_state.pending_query = prompt
                st.rerun()
