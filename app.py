"""
NLP Engine — Principal Orchestration Hub
========================================

Architecture Rationale:
-----------------------
This is the central entry point of the application. It follows a 'Controller' 
pattern in an MVC-lite architecture:
1.  **State Management**: Orchestrates the Streamlit session persistence.
2.  **UI Rendering**: Leverages 'ui_components.py' to maintain a high-fidelity SaaS aesthetic.
3.  **Task Orchestration**: Coordinates between the 'KnowledgeBase' (Retrieval) 
    and 'OllamaService' (Generation).
4.  **Ingestion Bridge**: Uses 'file_processor.py' to handle diverse data streams 
    (PDF, CSV, Excel, MD).

Structure:
- **Phase 1: Environment Setup**: Configuration, CSS injection, and state init.
- **Phase 2: Management Sidebar**: Engine toggles, status monitors, and active sources.
- **Phase 3: Categorized System Settings**: Modular zones for General/ML/DL parameters.
- **Phase 4: Analytics Dashboard**: Vector importance and normalization reports.
- **Phase 5: Research Hub (The Heart)**: Interactive QA, RAG flow, and Token Reporting.
"""

import streamlit as st
import pandas as pd
import time
import os
import subprocess

# Modular Core Imports
from core.knowledge_base import KnowledgeBase
from core.llm_service import OllamaService
from utils.ui_components import inject_custom_css, render_header, render_sidebar_branding, render_token_report
from utils.file_processor import process_single_file

# --- PHASE 1: CONFIGURATION & STATE ---
st.set_page_config(
    page_title="NLP Engine // Workspace",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Global Aesthetic (CSS)
inject_custom_css()

# Initialize Persistent Sessions
if "llm" not in st.session_state: st.session_state.llm = OllamaService()
if "kb" not in st.session_state: st.session_state.kb = KnowledgeBase()
if "messages" not in st.session_state: st.session_state.messages = []
if "is_syncing" not in st.session_state: st.session_state.is_syncing = False
if "is_searching" not in st.session_state: st.session_state.is_searching = False
if "pending_query" not in st.session_state: st.session_state.pending_query = None

# --- PHASE 2: MANAGEMENT SIDEBAR ---
with st.sidebar:
    render_sidebar_branding()

    # 2.1 Engine Intelligence Mode
    st.markdown("<p class='meta-label'>Core Intelligence Mode</p>", unsafe_allow_html=True)
    engine_choice = st.radio("Engine Selection", ["Machine Learning", "Deep Learning"], 
                             index=0 if st.session_state.kb.engine_mode == "Machine Learning" else 1,
                             label_visibility="collapsed")
    
    if engine_choice != st.session_state.kb.engine_mode:
        st.session_state.kb.engine_mode = engine_choice
        st.warning(f"Engine switched. Re-build index to sync vectors.")

    # 2.2 Neural Server Heartbeat
    st.session_state.llm.reset_status()
    ollama_ok = st.session_state.llm.is_available()
    status_color = "#34d399" if ollama_ok else "#f87171"
    status_text = "Online" if ollama_ok else "Offline"
    st.markdown(f"""
        <div style="margin-top: 10px; margin-bottom: 20px; padding: 12px; background: #1f2937; border-radius: 8px; border: 1px solid #374151;">
            <p style="font-size: 11px; margin: 0; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Neural Server</p>
            <p style="font-size: 14px; margin: 4px 0 0 0; font-weight: 600; color: {status_color};">● {status_text}</p>
        </div>
    """, unsafe_allow_html=True)

    # 2.3 Knowledge Base Dashboard (Redesigned Table View)
    if st.session_state.kb.file_contents:
        st.markdown("<p class='meta-label' style='margin-top:20px;'>Active Knowledge Base</p>", unsafe_allow_html=True)
        kb_files = list(st.session_state.kb.file_contents.keys())
        
        # New tabular view for better utility
        for fname in kb_files:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"<span style='font-size: 12px; color: #cbd5e1;'>{fname[:22]}...</span>", unsafe_allow_html=True)
            with c2:
                # LLM Actions available only when server is up
                if engine_choice == "Deep Learning" and ollama_ok:
                    if st.button("📝", key=f"sum_{fname}", help=f"LLM Summary for {fname}"):
                        with st.spinner("Analyzing..."):
                            doc_text = st.session_state.kb.get_document_text(fname)
                            result = st.session_state.llm.summarize_text(doc_text, fname)
                            st.session_state.messages.append({"role": "assistant", "content": f"### 📝 {fname} Summary\n{result}", "type": "summary"})
                            st.rerun()

    st.markdown("""<div style="margin-top: auto; border-top: 1px solid #30363d; padding-top: 20px;">
        <p style="font-size: 10px; color: #4b5563;">V3.5 // REFACTORED</p></div>""", unsafe_allow_html=True)

# Main Application Header
num_docs = len(st.session_state.kb.file_contents)
render_header(num_docs, engine_choice, status_text)

# --- PHASE 3 & 4: CATEGORIZED SETTINGS & ANALYTICS ---
tab_research, tab_analytics, tab_settings = st.tabs(["💠 Research Hub", "📊 Vector Analytics", "⚙️ System Settings"])

with tab_settings:
    st.markdown("### ⚙️ System Configuration")
    
    # 3.1 GENERAL SECTION
    with st.expander("📂 General Ingestion Settings", expanded=True):
        colS1, colS2 = st.columns(2)
        with colS1:
            uploaded_files = st.file_uploader("Upload Knowledge", accept_multiple_files=True, type=["pdf", "md", "txt", "csv", "xlsx"])
        with colS2:
            directory_path = st.text_input("Local Directory Path", placeholder="C:\\Users\\...")

    # 3.2 MACHINE LEARNING SECTION
    with st.expander("📊 Machine Learning (TF-IDF) Parameters", expanded=False):
        # Use getattr to prevent errors with stale session state objects
        ml_limit = getattr(st.session_state.kb, 'ml_top_n', 5)
        st.session_state.kb.ml_top_n = st.slider("Max Results (Top N)", 1, 20, ml_limit, 
                                                help="Limits the number of retrieved document segments for keyword search.")
        st.info("Uses sparse matrix normalization and word-frequency vectors.")

    # 3.3 DEEP LEARNING SECTION
    with st.expander("🧠 Deep Learning (Neural) Parameters", expanded=False):
        st.write(f"Active Model: **{st.session_state.llm.model_name}**")
        st.write(f"Embedding Engine: **{st.session_state.llm.embedding_model}**")
        st.info("Uses 1024-dimensional dense vectors for contextual understanding.")

    # Ingestion Orchestration
    if st.button("Initialize Engine & Build Index", use_container_width=True):
        files = []
        if uploaded_files:
            for f in uploaded_files: files.append({'obj': f, 'name': f.name, 'full_path': None})
        if directory_path and os.path.isdir(directory_path):
            for r, _, fs in os.walk(directory_path):
                for f in fs:
                    if f.endswith((".pdf", ".md", ".txt", ".csv", ".xlsx")):
                        p = os.path.join(r, f)
                        files.append({'obj': p, 'name': f, 'full_path': p})

        if files:
            st.session_state.is_syncing = True
            st.session_state.kb.__init__(engine_mode=engine_choice) # Hard Reset
            prog = st.progress(0)
            st_text = st.empty()
            for idx, item in enumerate(files):
                st_text.markdown(f"<span style='color:#94a3b8; font-size: 13px;'>Ingesting: {item['name']}</span>", unsafe_allow_html=True)
                process_single_file(item['obj'], st.session_state.kb, item['name'], item['full_path'])
                prog.progress((idx + 1) / len(files))
            
            st_text.markdown("<span style='color:#2563eb; font-size: 13px;'>Building Vector Core...</span>", unsafe_allow_html=True)
            st.session_state.kb.build_index(st.session_state.llm if engine_choice == "Deep Learning" else None)
            st.session_state.is_syncing = False
            st.rerun()

with tab_analytics:
    st.markdown("### 📊 Vector Analytics")
    if not st.session_state.kb.file_contents:
        st.info("Ingest documents to view semantic distributions.")
    else:
        st.markdown("<p class='meta-label'>Top Topic Distributions</p>", unsafe_allow_html=True)
        keywords_df = st.session_state.kb.get_top_keywords_df()
        if not keywords_df.empty:
            st.bar_chart(keywords_df.set_index('Keyword'), color="#2563eb", height=300)
        st.markdown("<p class='meta-label' style='margin-top: 30px;'>NLP Pipeline Report</p>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.kb.cleaning_report), use_container_width=True, height=300, hide_index=True)

# --- PHASE 5: RESEARCH HUB ---
with tab_research:
    if not st.session_state.kb.file_contents:
        st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center;'><div style='font-size: 60px;'>💠</div><h1>Nexus Hub</h1><p>Ingest knowledge to begin your semantic research.</p></div>", unsafe_allow_html=True)
    else:
        # 5.1 Chat Display
        chat_box = st.container(height=650, border=False)
        with chat_box:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    # Inline Results for Retrieval-only search
                    if msg.get("type") == "results":
                        for fname, chunks in msg["data"].items():
                            with st.container(border=True):
                                avg_score = sum(r['score'] for r in chunks) / len(chunks)
                                st.markdown(f"<div style='display:flex;justify-content:space-between;'><span class='meta-label'>{fname}</span><span class='match-tag'>{int(avg_score*100)}% Match</span></div>", unsafe_allow_html=True)
                                with st.expander("View Segments", expanded=False):
                                    for r in chunks: st.markdown(f"<p style='font-size:0.85rem;'>({r['page']}) {r['text']}</p>", unsafe_allow_html=True)

        # 5.2 Input Command Handler
        if st.session_state.is_searching and st.session_state.pending_query:
            query = st.session_state.pending_query
            
            # --- HELP COMMANDS ---
            if query.lower() in ["/help", "/examples"]:
                if engine_choice == "Machine Learning":
                    help_msg = "### 🔦 ML Search Tips\nUse specific keywords like **'Revenue 2024'** or **'Protocol X'**. Avoid natural questions as this engine matches literal vector intersections."
                else:
                    help_msg = "### 🧠 DL RAG Tips\nAsk natural questions like **'What are the key risks mentioned in file X?'** or **'Summarize the conclusion of page 4'**."
                st.session_state.messages.append({"role": "assistant", "content": help_msg})
                st.session_state.is_searching = False
                st.rerun()

            # --- SEARCH EXECUTION ---
            with st.status("💠 Processing Semantic Hub...", expanded=True) as status:
                if engine_choice == "Deep Learning" and ollama_ok:
                    # NEURAL RAG FLOW
                    ctx = st.session_state.kb.get_context_for_query(query, st.session_state.llm)
                    if ctx:
                        with chat_box:
                            with st.chat_message("assistant"):
                                stream = st.session_state.llm.generate_rag_response(query, ctx, st.session_state.messages)
                                full_res = st.write_stream(stream)
                        stats = st.session_state.llm.get_last_stats()
                        st.session_state.messages.append({"role": "assistant", "content": full_res, "type": "rag", "stats": stats})
                        # Post-Response Token Analysis
                        render_token_report(stats['input_tokens'], stats['output_tokens'], stats['total_tokens'])
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "No context found."})
                else:
                    # STATISTICAL RETRIEVAL FLOW
                    res = st.session_state.kb.search(query, top_n=st.session_state.kb.ml_top_n)
                    if res:
                        grouped = {}
                        for r in res: 
                            if r['file'] not in grouped: grouped[r['file']] = []
                            grouped[r['file']].append(r)
                        st.session_state.messages.append({"role": "assistant", "type": "results", "content": f"Vector correlations for: **{query}**", "data": grouped})
                    else: st.session_state.messages.append({"role": "assistant", "content": "No matches found."})
                
                status.update(label="Query Complete", state="complete")
            st.session_state.is_searching = False
            st.session_state.pending_query = None
            st.rerun()
        else:
            # 5.3 Chat Input Box
            if prompt := st.chat_input("Enter your research query (or type /help)..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.is_searching = True
                st.session_state.pending_query = prompt
                st.rerun()
