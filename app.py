"""
NLP Engine — Principal Orchestration Hub
========================================

Architecture Rationale:
-----------------------
This is the central entry point of the application. It follows a 'Controller' 
pattern in an MVC-lite architecture:
1.  **State Management**: Orchestrates the Streamlit session persistence.
2.  **UI Rendering**: Leverages 'ui_components.py' for a consistent SaaS aesthetic.
3.  **Task Orchestration**: Bridges the 'KnowledgeBase' and 'OllamaService'.
4.  **Ingestion Bridge**: Coordinates multi-format parsing via 'file_processor.py'.

Streamlit Execution Model (Mental Model for Devs):
Streamlit re-runs the entire script from top to bottom on every user interaction.
To prevent the application from losing data (like your indexed vectors) or
re-instantiating heavy objects (like the KnowledgeBase), we bury them in 
`st.session_state`. This ensures they 'survive' the re-run cycle.
"""


import streamlit as st
import pandas as pd
import time
import os
import subprocess
import traceback
import re
from datetime import datetime

# Modular Core Imports
import plotly.express as px
from core.knowledge_base import KnowledgeBase
from core.llm_service import OllamaService
from core.config_manager import ConfigManager
from core.identity_manager import IdentityManager
from core.image_service import LocalImageService
from utils.ui_components import inject_custom_css, render_header, render_sidebar_branding, render_token_report, get_plotly_template
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

# --- PHASE 1: PERSISTENT STATE REGISTRY ---
# We initialize our 'Core Engines' here. If they already exist in the state, 
# Streamlit will skip these initializations, preserving the active index.

if "llm" not in st.session_state: 
    # The OllamaService handles all LLM inference and embeddings.
    st.session_state.llm = OllamaService()

if "kb" not in st.session_state: 
    # The KnowledgeBase handles vector search and document chunking.
    st.session_state.kb = KnowledgeBase()

if "img_service" not in st.session_state:
    # Manages local Stable Diffusion generation
    st.session_state.img_service = LocalImageService()

if "messages" not in st.session_state: 
    # Stores the chat history for the research session.
    st.session_state.messages = []

if "config" not in st.session_state: 
    # ConfigManager handles saving/loading settings.json to disk.
    st.session_state.config = ConfigManager()
    
    # Push stored preferences into the live engines.
    st.session_state.llm.model_nickname = st.session_state.config.get("model_nickname")
    st.session_state.llm.base_url = st.session_state.config.get("ollama_host")
    st.session_state.kb.engine_mode = st.session_state.config.get("engine_mode")
    
    # --- PROACTIVE COLD-START RECOVERY ---
    # If a previous index exists on disk, we auto-load it on first launch.
    if os.path.exists("data/index/metadata.json") and not st.session_state.kb.documents_metadata:
        if st.session_state.kb.load_from_disk():
            st.toast("✅ Persistent Knowledge Loaded", icon="🧬")

elif not hasattr(st.session_state.config, 'get'):
    # Force re-init if the object is stale/broken
    st.session_state.config = ConfigManager()

# UI State Flags
if "confirm_clear_err" not in st.session_state: st.session_state.confirm_clear_err = False
if "is_indexing" not in st.session_state: st.session_state.is_indexing = False

# --- STATE MIGRATION & BACKWARD COMPATIBILITY ---
# Developer Note: As the project evolves, we add new attributes to the 
# KnowledgeBase or OllamaService classes. Since these objects are persisted 
# in the session state, older sessions might "break" if they lack a new 
# attribute. We "hot-patch" them here to ensure zero-crash sessions.

if not hasattr(st.session_state.kb, 'spatial_granularity'):
    st.session_state.kb.spatial_granularity = "Segments"
if not hasattr(st.session_state.kb, 'documents_spatial'):
    st.session_state.kb.documents_spatial = []
if not hasattr(st.session_state.kb, 'indexing_errors'):
    st.session_state.kb.indexing_errors = []

# --- INTELLIGENT THRESHOLDING ---
if "neural_threshold" not in st.session_state:
    # Set intelligent default based on active model dimensions
    dims = st.session_state.llm.get_embedding_dimension()
    if dims <= 384: st.session_state.neural_threshold = 0.25 # Light (minilm)
    elif dims <= 768: st.session_state.neural_threshold = 0.30 # Standard (nomic)
    else: st.session_state.neural_threshold = 0.35 # Dense (mxbai/gemma)

st.session_state.kb.neural_threshold = st.session_state.neural_threshold

if not hasattr(st.session_state.llm, 'model_nickname'):
    st.session_state.llm.model_nickname = st.session_state.llm.model_name

if "is_syncing" not in st.session_state: st.session_state.is_syncing = False
if "is_searching" not in st.session_state: st.session_state.is_searching = False
if "pending_query" not in st.session_state: st.session_state.pending_query = None
if "view_level" not in st.session_state: st.session_state.view_level = "Universe"
if "focus_cluster" not in st.session_state: st.session_state.focus_cluster = None

# --- REMOVED: OLD SYNC HUB ---

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
    
    # Model Mismatch Check (Neural Only)
    if engine_choice == "Deep Learning" and hasattr(st.session_state.kb, 'index_embedding_model'):
        idx_model = st.session_state.kb.index_embedding_model
        curr_model = st.session_state.llm.embedding_model
        
        # Enhanced math comparison
        idx_dim = getattr(st.session_state.kb, 'index_embedding_dimension', 0)
        curr_dim = st.session_state.llm.get_embedding_dimension()
        
        if idx_model and idx_model != curr_model:
            dim_warning = f"\nMath: `{idx_dim}d` vs `{curr_dim}d`" if idx_dim != curr_dim else ""
            st.error(f"⚠️ **Model Mismatch**\nIndex: `{idx_model}`\nActive: `{curr_model}`{dim_warning}\nPlease re-index to sync.")

    # 2.2 Neural Server Heartbeat
    ollama_ok = True  # Default for ML mode
    status_text = "N/A"
    status_color = "#94a3b8"
    
    if engine_choice == "Deep Learning":
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

    # --- MOVED: 2.3 Galaxy Explorer / Drill-Down Hub ---
    # Use existing metadata based on current granularity
    is_doc_mode = getattr(st.session_state.kb, 'spatial_granularity', "Segments") == "Documents"
    source_list = st.session_state.kb.documents_spatial if is_doc_mode else st.session_state.kb.documents_metadata
    
    if source_list:
        st.markdown("<p class='meta-label' style='margin-top:20px;'>Galaxy Explorer</p>", unsafe_allow_html=True)
        all_clusters = sorted(list(set(m.get('cluster', 0) for m in source_list)))
        
        # Universe Reset
        if st.session_state.view_level != "Universe":
            if st.button("🌌 Back to Universe", width="stretch"):
                st.session_state.view_level = "Universe"
                st.session_state.focus_cluster = None
                st.rerun()
        
        # Focus Selection with Semantic Naming (with safety fallback)
        cluster_options = []
        has_topics_method = hasattr(st.session_state.kb, 'get_cluster_topics')
        
        for c in all_clusters:
            if has_topics_method and not st.session_state.is_indexing:
                topics = st.session_state.kb.get_cluster_topics(c, top_n=2)
                name = f"Galaxy {c}: {', '.join(topics)}"
            else:
                tag = " (Syncing...)" if st.session_state.is_indexing else " (Sync Required)"
                name = f"Galaxy {c}{tag}"
            cluster_options.append({"id": c, "name": name})
        
        selected_name = st.selectbox("Focus Galaxy", ["Global View"] + [o['name'] for o in cluster_options], 
                                index=0 if st.session_state.focus_cluster is None else 
                                [o['id'] for o in cluster_options].index(st.session_state.focus_cluster) + 1)
        
        if selected_name == "Global View":
            if st.session_state.view_level != "Universe":
                st.session_state.view_level = "Universe"
                st.session_state.focus_cluster = None
                st.rerun()
        else:
            # Extract ID from selected option name
            c_id = next(o['id'] for o in cluster_options if o['name'] == selected_name)
            if st.session_state.focus_cluster != c_id:
                st.session_state.view_level = "Galaxy"
                st.session_state.focus_cluster = c_id
                st.rerun()

        # --- MOVED: 2.4 Spatial Granularity Toggle ---
        st.markdown("<p class='meta-label' style='margin-top:20px;'>Visualization Scale</p>", unsafe_allow_html=True)
        current_gran = getattr(st.session_state.kb, 'spatial_granularity', "Segments")
        gran_choice = st.radio("Granularity", ["📄 Documents", "🧩 Segments"], 
                               index=0 if current_gran == "Documents" else 1,
                               horizontal=True, key="sidebar_gran_toggle", label_visibility="collapsed")
        
        clean_gran = "Documents" if "Documents" in gran_choice else "Segments"
        if clean_gran != current_gran:
            st.session_state.kb.spatial_granularity = clean_gran
            with st.spinner("Re-scaling Universe..."):
                st.session_state.kb._generate_3d_spatial_data()
            st.session_state.view_level = "Universe"
            st.session_state.focus_cluster = None
            st.rerun()

    # 2.5 Knowledge Base Dashboard (Files List)
    if st.session_state.kb.file_contents:
        st.markdown("<p class='meta-label' style='margin-top:20px;'>Active Knowledge Base</p>", unsafe_allow_html=True)
        manifest = st.session_state.kb.get_file_manifest() if hasattr(st.session_state.kb, 'get_file_manifest') else sorted(list(st.session_state.kb.file_contents.keys()))
        
        for fname in manifest:
            # Full path for tooltip if available
            full_path = ""
            # Search metadata for full path
            for m in st.session_state.kb.documents_metadata:
                if m['file'] == fname:
                    full_path = m.get('full_path', 'No path available')
                    break
            
            # Parallax scrolling effect on hover
            st.markdown(f"""
                <div class="parallax-wrapper" title="{full_path}">
                    <div class="parallax-text">{fname}</div>
                </div>
            """, unsafe_allow_html=True)
            
            if engine_choice == "Deep Learning" and ollama_ok:
                if st.button("📝 Summarize", key=f"sum_{fname}", help=f"LLM Summary for {fname}", use_container_width=True):
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

# --- GLOBAL SYSTEM HUBS (Sync & Errors) ---
# This ensures critical alerts are visible regardless of the active tab.

# 1. Sync Hub: Fixes session state issues before they cause crashes
if not hasattr(st.session_state.kb, 'get_cluster_topics') or not hasattr(st.session_state.kb, 'clear_previous_index'):
    with st.container(border=True):
        st.error("🚨 ENGINE SYNCHRONIZATION REQUIRED")
        st.markdown("Your session is using an older version of the Intelligence Engine. Please synchronize to enable new Galaxy features and Progressive Ingestion.")
        if st.button("🚀 Sync Intelligence Engine Now", width="stretch", type="primary"):
            with st.spinner("Synchronizing Vectors..."):
                old_kb = st.session_state.kb
                new_kb = KnowledgeBase()
                # Migrate data
                new_kb.file_contents = getattr(old_kb, 'file_contents', {})
                new_kb.documents_metadata = getattr(old_kb, 'documents_metadata', [])
                new_kb.documents_spatial = getattr(old_kb, 'documents_spatial', [])
                new_kb.tfidf_matrix = getattr(old_kb, 'tfidf_matrix', None)
                new_kb.vectorizer = getattr(old_kb, 'vectorizer', None)
                new_kb.embeddings = getattr(old_kb, 'embeddings', None)
                new_kb.engine_mode = getattr(old_kb, 'engine_mode', "Machine Learning")
                new_kb.spatial_granularity = getattr(old_kb, 'spatial_granularity', "Segments")
                new_kb.documents_matrix_agg = getattr(old_kb, 'documents_matrix_agg', None)
                st.session_state.kb = new_kb
                st.rerun()
    st.markdown("---")

# 2. Global Error Hub: Persistent visibility for ingestion issues
if hasattr(st.session_state.kb, 'indexing_errors') and st.session_state.kb.indexing_errors:
    with st.container(border=True):
        st.markdown(f"### ⚠️ Ingestion Report: {len(st.session_state.kb.indexing_errors)} Errors")
        full_report = "[\n" + ",\n".join(st.session_state.kb.indexing_errors) + "\n]"
        with st.expander("View Technical Tracebacks", expanded=False):
            st.code(full_report, language="text")
        
        c_clear, c_copy = st.columns([1, 1])
        with c_clear:
            with st.popover("✨ Clear Progress Logs", use_container_width=True):
                st.markdown("### 💠 Ingestion Pulse Control")
                st.write("This will permanently clear all ingestion tracebacks and internal error reports from the current session.")
                if st.button("Confirm Reset", type="primary", use_container_width=True, key="pop_confirm_clear"):
                    st.session_state.kb.indexing_errors = []
                    if hasattr(st.session_state.kb, 'ingestion_log'):
                        st.session_state.kb.ingestion_log = None
                    st.rerun()
        with c_copy:
            # Highlighting the native copy button
            st.button("ℹ️ Use Copy in Expander", use_container_width=True, disabled=True, help="Use the copy button in the 'View Technical Tracebacks' section above.")

# --- NEW: DETAILED INGESTION ANALYTICS (PULLED OUT OF ERROR BLOCK) ---
if hasattr(st.session_state.kb, 'ingestion_log') and st.session_state.kb.ingestion_log:
    with st.expander("📂 Knowledge Scoping Report", expanded=False):
        log = st.session_state.kb.ingestion_log
        colL1, colL2, colL3 = st.columns(3)
        colL1.metric("Total Items Found", log['total_found'])
        colL2.metric("Skipped (Format)", len(log['skipped']))
        colL3.metric("Unreachable (Cloud)", len(log['unreachable']))
        
        if log['unreachable']:
            st.error("### ❌ Unreachable Files (Hydration Issues)")
            st.write("These files exist but could not be read. If using Google Drive, please open the folder in Windows Explorer to 'hydrate' them.")
            st.code("\n".join(log['unreachable']), language="text")
        
        if log['skipped']:
            st.warning("### ⚠️ Unsupported Formats")
            st.write("These files were ignored because their format is not yet supported (.docx, .pptx, etc).")
            st.code("\n".join(log['skipped']), language="text")

# --- PHASE 3 & 4: CATEGORIZED SETTINGS & ANALYTICS ---
tab_research, tab_analytics, tab_studio, tab_settings = st.tabs(["💠 Research Hub", "📊 Vector Analytics", "🎨 Image Studio", "⚙️ System Settings"])

with tab_settings:
    st.markdown("### ⚙️ System Configuration")
    
    # --- RELOCATED: 3.0 PROCESS ORCHESTRATION (THE "DASHBOARD") ---
    if "is_indexing" not in st.session_state: st.session_state.is_indexing = False
    
    if not st.session_state.is_indexing:
        if st.button("🚀 Initialize Engine & Build Index", width="stretch", type="primary"):
            st.session_state.is_indexing = True
            st.session_state.kb.stop_requested = False
            st.session_state.kb_already_cleared = False
            # REMOVED: st.rerun() - Let the script flow naturally into the processor below.

    else:
        # Stop Indexing Button (Disabled if threshold reached)
        stop_disabled = getattr(st.session_state, 'kb_already_cleared', False)
        if st.button("🛑 Stop Indexing", width="stretch", type="secondary", disabled=stop_disabled):
            st.session_state.kb.stop_requested = True
            st.session_state.is_indexing = False
            st.rerun()
        if stop_disabled:
            st.info("Commit threshold reached. Stopping is no longer possible.")

    # --- TOP-ANCHORED PROGRESS SLOTS ---
    prog_placeholder = st.empty()
    status_placeholder = st.empty()


    # Note: Process Orchestration loop relocated here for immediate UI visibility.
    # --- RELOCATED: 3.1 INPUT SOURCES (Required for indexing loop) ---
    with st.expander("📂 General Ingestion Settings", expanded=True):
        colS1, colS2 = st.columns(2)
        with colS1:
            uploaded_files = st.file_uploader("Upload Knowledge", accept_multiple_files=True, 
                                              type=["pdf", "md", "txt", "csv", "xlsx", "docx", "pptx"],
                                              key="uploaded_files_input")
        with colS2:
            directory_path = st.text_input("Local Directory Path", placeholder="C:\\Users\\...",
                                           key="directory_path_input")
            
            # --- NEW: PROACTIVE PATH VALIDATION ---
            if st.button("🔍 Test Path Visibility", use_container_width=True):
                test_p = st.session_state.directory_path_input.strip().strip('"').strip("'")
                if test_p:
                    if os.path.isdir(test_p):
                        t_folders, t_files, t_supported, t_unreachable = 0, 0, 0, 0
                        supported_ext = (".pdf", ".md", ".txt", ".csv", ".xlsx", ".docx", ".pptx")
                        with st.status(f"Scanning `{test_p}`...", expanded=True):
                            for r, ds, fs in os.walk(test_p):
                                t_folders += 1
                                t_files += len(fs)
                                for f in fs:
                                    if f.lower().endswith(supported_ext):
                                        t_supported += 1
                                        # Hydration Check (Strict)
                                        try:
                                            full_p = os.path.join(r, f)
                                            with open(full_p, 'rb') as tmp:
                                                tmp.read(1)
                                        except Exception:
                                            t_unreachable += 1
                                            t_supported -= 1

                        if t_supported > 0:
                            st.success(f"### ✅ Path Verified\n"
                                       f"- **Folders Scanned**: {t_folders}\n"
                                       f"- **Files Seen**: {t_files}\n"
                                       f"- **Ready to Index**: {t_supported}\n"
                                       f"- **Unreachable (Cloud)**: {t_unreachable}")
                        elif t_unreachable > 0:
                            st.error(f"### ⚠️ Drive Hydration Required\n"
                                     f"Found **{t_unreachable}** potential matches, but they are all **'Cloud-Only'**. \n\n"
                                     "Please right-click the folder in Windows Explorer and select **'Always keep on this device'** to fix this.")
                        else:
                            st.warning(f"### 🔍 Scan Complete: 0 Matches\n"
                                       f"- **Folders Scanned**: {t_folders}\n"
                                       f"- **Total Files Seen**: {t_files}\n"
                                       "No files matching supported extensions were found.")
                    else:
                        st.error(f"### ❌ Path Not Found\nThe directory either doesn't exist or is inaccessible: `{test_p}`")
                else:
                    st.warning("Please enter a path first.")


        
        st.markdown("---")
        colL1, colL2, colL3 = st.columns(3)
        with colL1:
            log_path = st.text_input("Default Log Path", st.session_state.config.get("log_path"), 
                                     help="Directory where persistent error logs are saved.")
            if log_path != st.session_state.config.get("log_path"):
                st.session_state.config.save({"log_path": log_path})
        with colL2:
            log_pattern = st.text_input("Log Naming Pattern", st.session_state.config.get("log_pattern"), 
                                        help="Use strftime symbols like %Y-%m-%d for timestamps.")
            if log_pattern != st.session_state.config.get("log_pattern"):
                st.session_state.config.save({"log_pattern": log_pattern})
        with colL3:
            threshold = st.slider("KB Clear Threshold (%)", 0, 100, st.session_state.config.get("clearing_threshold"),
                                  help="The old KB will be purged once new indexing reaches this percentage.")
            if threshold != st.session_state.config.get("clearing_threshold"):
                st.session_state.config.save({"clearing_threshold": threshold})

        # --- PERSISTENT STORAGE MANAGEMENT ---
        st.markdown("---")
        st.markdown("#### 💾 Persistent Storage")
        if os.path.exists("data/index"):
            c_save, c_wipe, c_load = st.columns(3)
            with c_save:
                if st.button("💾 Force Save", use_container_width=True):
                    if st.session_state.kb.save_to_disk():
                        st.toast("✅ Index Saved", icon="💾")
            with c_wipe:
                if st.button("🗑️ Wipe Saved", use_container_width=True, type="secondary"):
                    import shutil
                    if os.path.exists("data/index"):
                        shutil.rmtree("data/index")
                        st.session_state.kb.documents_metadata = []
                        st.session_state.kb.file_contents = {}
                        st.toast("🔥 Persistence Wiped", icon="🗑️")
                        st.rerun()
            with c_load:
                if st.button("🧬 Load from Disk", use_container_width=True):
                    if st.session_state.kb.load_from_disk():
                        st.toast("🧬 Knowledge Restored", icon="✅")
                        st.rerun()
        else:
            st.info("No persistent index found at data/index/. It will be created after your first successful build.")

        # --- MOVED: INGESTION CONSTRAINTS ---
        st.markdown("---")
        st.markdown("#### 📏 Ingestion Constraints")
        c_act, c_sz = st.columns([1, 2])
        with c_act:
            is_active = st.toggle("Limit File Size", st.session_state.config.get("ingestion_size_limit_active"),
                                  help="If enabled, files larger than the threshold will be skipped.")
            if is_active != st.session_state.config.get("ingestion_size_limit_active"):
                st.session_state.config.save({"ingestion_size_limit_active": is_active})
        with c_sz:
            size_mb = st.number_input("Max File Size (MB)", 1, 1000, st.session_state.config.get("ingestion_size_limit_mb"),
                                      disabled=not is_active, help="Maximum allowed file size in Megabytes.")
            if size_mb != st.session_state.config.get("ingestion_size_limit_mb"):
                st.session_state.config.save({"ingestion_size_limit_mb": size_mb})
        if is_active:
            st.info(f"Note: Files over **{size_mb} MB** will be ignored during the scan.")


        






        


    # 3.2 MACHINE LEARNING SECTION

    # 3.2 MACHINE LEARNING SECTION
    with st.expander("📊 Machine Learning (TF-IDF) Parameters", expanded=False):
        # Use getattr to prevent errors with stale session state objects
        ml_limit = getattr(st.session_state.kb, 'ml_top_n', 5)
        st.session_state.kb.ml_top_n = st.slider("Max Results (Top N)", 1, 20, ml_limit, 
                                                help="Limits the number of retrieved document segments for keyword search.")
        st.info("Uses sparse matrix normalization and word-frequency vectors.")

    # 3.3 DEEP LEARNING SECTION
    with st.expander("🧠 Deep Learning (Neural) Parameters", expanded=False):
        colA1, colA2 = st.columns(2)
        with colA1:
            st.write(f"Active Model: **{st.session_state.llm.model_name}**")
            st.write(f"Embedding Engine: **{st.session_state.llm.embedding_model}**")
            
            # 3.3.1 Chat Model with Refresh
            chat_models = st.session_state.llm.get_chat_models()
            col_chat, col_refresh1 = st.columns([4, 1])
            with col_chat:
                if chat_models:
                    new_model = st.selectbox("Switch Chat Model", chat_models, 
                                            index=chat_models.index(st.session_state.llm.model_name) if st.session_state.llm.model_name in chat_models else 0)
                    if new_model != st.session_state.llm.model_name:
                        if st.session_state.llm.set_model(new_model):
                            st.success(f"Model switched to {new_model}")
                            st.rerun()
            with col_refresh1:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄", key="ref_chat", help="Refresh Chat Models & Check Heartbeat"):
                    if st.session_state.llm.is_available():
                        st.session_state.llm.reset_status()
                        st.toast("Ollama Active: Models Refreshed", icon="✅")
                    else:
                        st.toast("Ollama Offline!", icon="❌")
                    st.rerun()
                
            # 3.3.2 Embedding Model with Refresh
            embed_models = st.session_state.llm.get_embedding_models()
            col_embed, col_refresh2 = st.columns([4, 1])
            with col_embed:
                if embed_models:
                    new_embed = st.selectbox("Switch Embedding Model", embed_models,
                                            index=embed_models.index(st.session_state.llm.embedding_model) if st.session_state.llm.embedding_model in embed_models else 0)
                    if new_embed != st.session_state.llm.embedding_model:
                        if st.session_state.llm.set_embedding_model(new_embed):
                            # Auto-Scale Threshold based on NEW model dimensions
                            new_dims = st.session_state.llm.get_embedding_dimension()
                            if new_dims <= 384: st.session_state.neural_threshold = 0.25
                            elif new_dims <= 768: st.session_state.neural_threshold = 0.35
                            else: st.session_state.neural_threshold = 0.45
                            st.session_state.kb.neural_threshold = st.session_state.neural_threshold
                            
                            st.warning("Embedding engine switched. You MUST re-index the Knowledge Base to synchronize vectors.")
                            st.rerun()

                else:
                    st.error("No embedding models found. Please install nomic-embed-text or mxbai-embed-large.")
            with col_refresh2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄", key="ref_embed", help="Refresh Embedding Models & Check Heartbeat"):
                   if st.session_state.llm.is_available():
                        st.session_state.llm.get_embedding_dimension() # Force probe
                        st.toast("Ollama Active: Embeddings Refreshed", icon="✅")
                   else:
                        st.toast("Ollama Offline!", icon="❌")
                   st.rerun()
        with colA2:
            host = st.text_input("Ollama Host", st.session_state.config.get("ollama_host"))
            if host != st.session_state.config.get("ollama_host"):
                # Simple validation attempt
                old_host = st.session_state.llm.base_url
                st.session_state.llm.base_url = host
                if st.session_state.llm.is_available():
                    st.session_state.config.save({"ollama_host": host})
                    st.success("Host updated and validated.")
                else:
                    st.session_state.llm.base_url = old_host
                    st.error("Validation failed. Reverting to previous host.")
        
            
        dims = st.session_state.llm.get_embedding_dimension()
        label = f"{dims}-dimensional" if dims > 0 else "Neural"
        st.info(f"🧠 Uses {label} dense vectors for contextual understanding.")
        
        # 3.3.3 Similarity Threshold Tuning
        st.session_state.neural_threshold = st.slider("Neural Similarity Threshold", 0.05, 0.95, 
                                                      st.session_state.neural_threshold, 
                                                      help="Higher = stricter matches. Lower = broad contextual reach. Auto-scales when switching models.")
        st.session_state.kb.neural_threshold = st.session_state.neural_threshold

        


        # --- NEW: NEURAL CACHE MANAGEMENT ---
        st.markdown("---")
        st.markdown("#### 🧠 Neural Cache Management")
        st.markdown("<p style='font-size: 14px; color: #94a3b8;'>Isolated caches prevent 'Neural Dimension Mismatches' when switching models. Clear these if you encounter consistency errors.</p>", unsafe_allow_html=True)
        
        c_cache_info, c_cache_purge = st.columns([2, 1])
        with c_cache_info:
            import glob
            cache_files = glob.glob("data/.cache_*.json")
            cache_count = len(cache_files)
            if cache_count > 0:
                total_size_kb = sum(os.path.getsize(f) for f in cache_files) / 1024
                st.write(f"📁 {cache_count} model-specific caches found ({total_size_kb:.1f} KB)")
            else:
                st.write("✨ Cache is currently clean.")
        
        with c_cache_purge:
            if st.button("🧹 Purge Caches", use_container_width=True, help="Deletes ALL model-specific neural caches. This will force a full re-embedding on the next index build."):
                import glob
                for f in glob.glob("data/.cache_*.json"):
                    os.remove(f)
                # Also remove old legacy cache if exists
                if os.path.exists("data/.neural_cache.json"):
                    os.remove("data/.neural_cache.json")
                st.toast("Neural Cache Purged", icon="🧹")
                st.rerun()


        # --- NEW: AGENT IDENTITY EDITOR ---
        st.markdown("---")
        st.markdown("#### 🧬 Agent Identity & Persona")
        id_mgr = IdentityManager()
        current_agent_md = id_mgr.load_config()
        
        # Identity sections in a form
        with st.form("identity_editor"):
            st.markdown("<p style='font-size: 12px; color: #94a3b8;'>Manage the core DNA of the Intelligence Engine via AGENT.md</p>", unsafe_allow_html=True)
            new_identity_md = st.text_area("Agent Source (AGENT.md)", value=current_agent_md, height=300)
            
            if st.form_submit_button("Apply Identity Changes", width="stretch"):
                if id_mgr.is_diff(new_identity_md):
                    # Trigger confirmation (simulated since st.dialog is newer, using a nested state or simpler confirmation)
                    st.session_state.pending_agent_md = new_identity_md
                    st.session_state.confirm_id_save = True
                else:
                    st.toast("No changes detected.", icon="ℹ️")
        
        if getattr(st.session_state, 'confirm_id_save', False):
            st.warning("⚠️ Are you sure you want to overwrite the engine's core instructions?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, Commit Changes"):
                    id_mgr.save_config(st.session_state.pending_agent_md)
                    st.success("AGENT.md updated. Reloading engine...")
                    st.session_state.confirm_id_save = False
                    st.rerun()
            with c2:
                if st.button("Cancel"):
                    st.session_state.confirm_id_save = False
                    st.rerun()

        with st.expander("📚 View Prompt Engineering References", expanded=False):
            ref_path = "docs/deep_learning.md"
            if os.path.exists(ref_path):
                with open(ref_path, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            else:
                st.info("Reference notes will appear here after the first persona configuration.")

        # --- NEW: VISUAL INTELLIGENCE & IMAGE GENERATION ---
        st.markdown("---")
        st.markdown("#### 🎨 Visual Intelligence (Stable Diffusion)")
        st.markdown("<p style='font-size: 14px; color: #94a3b8;'>Manage local image generation models and visual indexing capabilities.</p>", unsafe_allow_html=True)
        
        img_active = st.session_state.config.get("image_gen_active")
        new_img_active = st.toggle("Enable Image Generation", value=img_active, help="Unlock /image command in chat and the Image Studio tab.")
        
        if new_img_active != img_active:
            st.session_state.config.save({"image_gen_active": new_img_active})
            st.rerun()

        if new_img_active:
            st.info("💡 Image generation is active. Ensure you have downloaded at least one model below.")
            
            c_mod, c_dev = st.columns(2)
            with c_mod:
                current_model_id = st.session_state.config.get("image_model_id")
                # Predefined recommended models
                model_options = LocalImageService.RECOMMENDED_MODELS
                selected_label = next((k for k, v in model_options.items() if v == current_model_id), "Custom")
                
                new_label = st.selectbox("Active Image Model", list(model_options.keys()), 
                                        index=list(model_options.keys()).index(selected_label) if selected_label in model_options else 0)
                
                if model_options[new_label] != current_model_id:
                    st.session_state.config.save({"image_model_id": model_options[new_label]})
                    st.rerun()
            
            with c_dev:
                current_dev = st.session_state.config.get("image_device")
                new_dev = st.selectbox("Processing Device", ["cuda", "cpu"], 
                                      index=0 if current_dev == "cuda" else 1,
                                      help="GPU (cuda) is significantly faster for image generation.")
                if new_dev != current_dev:
                    st.session_state.config.save({"image_device": new_dev})
                    st.rerun()

            # --- MODEL DOWNLOADER HUB ---
            st.markdown("##### 📥 Model Downloader")
            for label, m_id in model_options.items():
                is_down = st.session_state.img_service.is_model_downloaded(m_id)
                col1, col2 = st.columns([3, 1])
                col1.write(f"**{label}** (`{m_id}`)")
                if is_down:
                    col2.success("Installed")
                else:
                    if col2.button("Download", key=f"dl_{m_id}", use_container_width=True):
                        with st.spinner(f"Downloading {label} (~2-5GB)..."):
                            if st.session_state.img_service.download_model(m_id):
                                st.success(f"{label} ready!")
                                st.rerun()
                            else:
                                st.error("Download failed.")
            
            st.markdown("---")
            st.markdown("##### 👁️ Neural Vision (Llava)")
            vision_model = st.session_state.config.get("vision_model")
            st.write(f"Active Vision Model: `{vision_model}`")
            st.info("Used for 'captioning' generated images during indexing into the Galaxy.")


    # 3.4 Process Orchestration
    if "is_indexing" not in st.session_state: st.session_state.is_indexing = False
    
    if st.session_state.is_indexing:
        # Use direct widget variables if available, fallback to state
        current_uploaded = st.session_state.get("uploaded_files_input")
        current_path = st.session_state.get("directory_path_input")

        # Execute Indexing Pipeline
        limit_active = st.session_state.config.get("ingestion_size_limit_active")
        limit_mb = st.session_state.config.get("ingestion_size_limit_mb")
        
        files = []
        skipped_files = []
        unreachable_files = []
        supported_ext = (".pdf", ".md", ".txt", ".csv", ".xlsx", ".docx", ".pptx")
        scan_stats = {"folders": 0, "total_files": 0, "supported_files": 0}

        def gather_files(base_path, target_list, skip_list, unreach_list):
            base_path = base_path.strip().strip('"').strip("'")
            if not os.path.exists(base_path): return
            try:
                for r, ds, fs in os.walk(base_path):
                    scan_stats["folders"] += 1
                    scan_stats["total_files"] += len(fs)
                    for f in fs:
                        full_p = os.path.join(r, f)
                        if limit_active:
                            try:
                                f_size_mb = os.path.getsize(full_p) / (1024 * 1024)
                                if f_size_mb > limit_mb:
                                    skip_list.append(f"{f} (Large: {f_size_mb:.1f}MB)")
                                    continue
                            except Exception: pass
                        if f.lower().endswith(supported_ext):
                            try:
                                with open(full_p, 'rb') as tmp: tmp.read(1)
                                target_list.append({'obj': full_p, 'name': f, 'full_path': full_p})
                                scan_stats["supported_files"] += 1
                            except Exception as e:
                                unreach_list.append(f"{f} (Hydration Error: {str(e)})")
            except Exception as e:
                st.error(f"Directory Scan Error: {str(e)}")

        if current_uploaded:
            for f in current_uploaded: 
                if limit_active:
                    f_size_mb = f.size / (1024 * 1024)
                    if f_size_mb > limit_mb:
                        skipped_files.append(f"{f.name} (Large: {f_size_mb:.1f}MB)")
                        continue
                files.append({'obj': f, 'name': f.name, 'full_path': None})
        if current_path:
            clean_path = current_path.strip().strip('"').strip("'")
            if os.path.isdir(clean_path): gather_files(clean_path, files, skipped_files, unreachable_files)
        
        vault_path = "vault"
        if os.path.exists(vault_path): gather_files(vault_path, files, skipped_files, unreachable_files)

        if files:
            st.session_state.kb.documents_metadata = []
            st.session_state.kb.file_contents = {}
            st.session_state.kb.stop_requested = False 
            st.session_state.kb.indexing_errors = [] 
            
            # Use placeholders at the top
            prog = prog_placeholder.progress(0)
            live_err_placeholder = st.empty() # Still keep this near the bottom for details
            
            for idx, item in enumerate(files):
                if st.session_state.kb.stop_requested: break
                try:
                    process_single_file(item['obj'], st.session_state.kb, item['name'], item['full_path'])
                except Exception as e:
                    err_msg = f"{len(st.session_state.kb.indexing_errors) + 1}. {idx + 1}/{len(files)} {item['full_path'] or item['name']} - {str(e)}"
                    st.session_state.kb.indexing_errors.append(err_msg)
                
                err_count = len(st.session_state.kb.indexing_errors)
                status_color = "#ef4444" if err_count > 0 else "#2563eb"
                prog_msg = f"Step 1/2: Ingesting {item['name']} ({idx+1}/{len(files)})"
                if err_count > 0: prog_msg += f" | {err_count} Errors"
                
                status_placeholder.markdown(f"<p style='color:{status_color}; font-size: 14px; font-weight: 600;'>{prog_msg}</p>", unsafe_allow_html=True)
                prog.progress((idx + 1) / len(files))
            
            if not st.session_state.kb.stop_requested:
                status_placeholder.markdown("<p style='color:#8b5cf6; font-size: 14px; font-weight: 600;'>Step 2/2: Building Semantic Galaxy (Vector Core Calculation)...</p>", unsafe_allow_html=True)
                try:
                    st.session_state.kb.build_index(st.session_state.llm if engine_choice == "Deep Learning" else None)
                    st.session_state.kb.save_to_disk()
                except Exception as e:
                    st.error(f"Vector Core Build Failed: {str(e)}")
            
            st.session_state.is_indexing = False
            st.rerun()
        else:
            status_placeholder.warning("No files found to index.")
            st.session_state.is_indexing = False

    # Persistent Error Report
    if hasattr(st.session_state.kb, 'indexing_errors') and st.session_state.kb.indexing_errors:

        st.markdown("---")
        st.markdown(f"### ⚠️ Ingestion Report: {len(st.session_state.kb.indexing_errors)} Errors")
        full_report = "[\n" + ",\n".join(st.session_state.kb.indexing_errors) + "\n]"
        st.code(full_report, language="text")
        
        # Action Buttons
        c_clear, c_copy, c_exp, c_ask = st.columns(4)
        with c_clear:
            if not st.session_state.confirm_clear_err:
                if st.button("✨ Clear", use_container_width=True, key="clear_err"):
                    st.session_state.confirm_clear_err = True
                    st.rerun()
            else:
                if st.button("⚠️ Confirm?", use_container_width=True, key="confirm_clear_err_btn", type="primary"):
                    st.session_state.kb.indexing_errors = []
                    st.session_state.confirm_clear_err = False
                    st.rerun()
                if st.button("Cancel", use_container_width=True, key="cancel_clear_err"):
                    st.session_state.confirm_clear_err = False
                    st.rerun()

        with c_copy:
            if st.button("📋 Copy Log", use_container_width=True, key="copy_err"):
                st.info("Manual copy supported in the code block above.")
        with c_exp:
            log_pattern = st.session_state.config.get("log_pattern")
            log_filename = datetime.now().strftime(log_pattern) if "%" in log_pattern else log_pattern
            if not log_filename.endswith(".log"): log_filename += ".log"
            st.download_button("📤 Export", data=full_report, file_name=log_filename, mime="text/plain", use_container_width=True)
        with c_ask:
            model_nick = st.session_state.llm.model_nickname
            if st.button(f"🤖 Ask {model_nick}", use_container_width=True, key="ask_err"):
                recent_errors = st.session_state.kb.indexing_errors[-3:]
                diagnose_query = f"I encountered these errors during indexing. Analyze and suggest fixes:\n\n" + "\n".join(recent_errors)
                st.session_state.messages.append({"role": "user", "content": diagnose_query})
                st.session_state.is_searching = True
                st.session_state.pending_query = diagnose_query
                st.rerun()
        st.markdown("---")


with tab_analytics:
    st.markdown("### 📊 Vector Analytics")
    
    # Check for empty metadata based on current granularity
    has_data = (st.session_state.kb.spatial_granularity == "Segments" and st.session_state.kb.documents_metadata) or \
               (st.session_state.kb.spatial_granularity == "Documents" and st.session_state.kb.documents_spatial)

    if not has_data:
        st.info("Ingest documents and build the index to view the knowledge architecture.")
    else:
        # Determine Visualization Data
        if st.session_state.view_level == "Universe":
            st.markdown(f"#### 🌌 Universe View ({st.session_state.kb.spatial_granularity})")
            df = pd.DataFrame(st.session_state.kb.documents_metadata if st.session_state.kb.spatial_granularity == "Segments" else st.session_state.kb.documents_spatial)
            
            # Fetch Global Universe Statistics
            galaxy_stats = st.session_state.kb.get_universe_stats()
            title = f"Global Semantic Distribution ({st.session_state.kb.spatial_granularity})"
        else:
            # Fetch Galaxy Statistics (with safety fallback for stale sessions)
            if hasattr(st.session_state.kb, 'get_cluster_stats'):
                galaxy_stats = st.session_state.kb.get_cluster_stats(st.session_state.focus_cluster)
                topics_str = " & ".join(galaxy_stats['topics'])
            else:
                galaxy_stats = None
                topics_str = f"Galaxy {st.session_state.focus_cluster}"
            
            st.markdown(f"#### 🛰️ Detailed Scan: {topics_str}")
            with st.spinner(f"Refining {topics_str} coordinates..."):
                df = st.session_state.kb.get_cluster_spatial_data(st.session_state.focus_cluster)
            
            # Strict UI-level safety: If in Documents mode, ensure we only have one row per file
            if st.session_state.kb.spatial_granularity == "Documents" and not df.empty:
                df = df.drop_duplicates(subset=['file'])

            title = f"Localized Map: {topics_str}"

        # Render 3D Spatial Inspector
        if not df.empty and 'x' in df.columns:
            display_df = df.copy()
            
            # Map columns for consistency across granularities
            if 'Snippet' not in display_df.columns: 
                display_df['Snippet'] = display_df['text'].str.slice(0, 150)
            if 'segments' not in display_df.columns: display_df['segments'] = 1
            if 'page' not in display_df.columns: display_df['page'] = df['page'] if 'page' in df.columns else "N/A"
            
            # Semantic Mapping for Hover
            unique_clusters = display_df['cluster'].unique()
            cluster_map = {}
            for c in unique_clusters:
                if hasattr(st.session_state.kb, 'get_cluster_topics'):
                    topics = st.session_state.kb.get_cluster_topics(c, top_n=2)
                    cluster_map[c] = f"{' & '.join(topics)}"
                else:
                    cluster_map[c] = f"Galaxy {c}"
            display_df['galaxy_name'] = display_df['cluster'].map(cluster_map)
            
            fig = px.scatter_3d(
                display_df, x='x', y='y', z='z',
                color='cluster',
                # Using custom_data to pass values (x, file, page, Snippet, galaxy_name) to the JS hover listener
                custom_data=['x', 'file', 'page', 'Snippet', 'galaxy_name'],
                template=get_plotly_template(),
                color_continuous_scale='Viridis'
            )
            fig.update_traces(marker=dict(size=6 if st.session_state.kb.spatial_granularity == "Documents" else 4, 
                                         opacity=0.8, line=dict(width=0)))
            fig.update_layout(height=700)
            
            # PASS UNIVERSE OR GALAXY STATS TO THE INSPECTOR
            from utils.ui_components import render_spatial_inspector
            render_spatial_inspector(fig, galaxy_stats=galaxy_stats)
        else:
            st.warning("Insufficient spatial data to render map.")

        if engine_choice == "Machine Learning":
            st.markdown("<p class='meta-label' style='margin-top: 30px;'>Statistical Importance (Global)</p>", unsafe_allow_html=True)
            keywords_df = st.session_state.kb.get_top_keywords_df()
            if not keywords_df.empty:
                st.bar_chart(keywords_df.set_index('Keyword'), color="#2563eb", height=200)

        
        st.markdown("<p class='meta-label' style='margin-top: 30px;'>NLP Pipeline Report</p>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.kb.cleaning_report), width="stretch", height=200, hide_index=True)

with tab_studio:
    st.markdown("### 🎨 Image Studio")
    
    if not st.session_state.config.get("image_gen_active"):
        st.warning("Image Generation is disabled. Enable it in System Settings to use this tab.")
    elif not st.session_state.img_service.is_model_downloaded():
        st.error(f"Selected model `{st.session_state.config.get('image_model_id')}` is not downloaded. Visit System Settings.")
    else:
        # Sidebar for parameters
        col_gen, col_gal = st.columns([1, 2])
        
        with col_gen:
            st.markdown("#### 🔬 Parameters")
            studio_prompt = st.text_area("Positive Prompt", placeholder="An astronaut riding a horse in hyper-realistic style...", height=100)
            studio_neg = st.text_area("Negative Prompt", placeholder="blurry, distorted, low quality...", height=68)
            
            c1, c2 = st.columns(2)
            with c1:
                studio_steps = st.slider("Steps", 1, 50, 20)
                studio_seed = st.number_input("Seed", value=-1, help="-1 for random")
            with c2:
                studio_cfg = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
                
            if st.button("🚀 Generate High Detail", width="stretch", type="primary"):
                with st.spinner("Rendering..."):
                    seed = None if studio_seed == -1 else studio_seed
                    path, filename = st.session_state.img_service.generate(
                        studio_prompt, 
                        negative_prompt=studio_neg,
                        steps=studio_steps,
                        guidance_scale=studio_cfg,
                        seed=seed
                    )
                    if path:
                        st.image(path, caption="Studio Render Result")
                        st.session_state.messages.append({"role": "assistant", "content": f"🎨 Studio Render: **{studio_prompt}**", "type": "image", "path": path, "filename": filename})
                        st.success(f"Saved as {filename}")
                    else:
                        st.error(f"Render Failed: {filename}")
        
        with col_gal:
            st.markdown("#### 🖼️ History & Gallery")
            img_dir = "data/images"
            if os.path.exists(img_dir):
                files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")], reverse=True)
                if files:
                    # Render in a grid
                    cols = st.columns(3)
                    for idx, f in enumerate(files[:12]): # Show last 12
                        path = os.path.join(img_dir, f)
                        with cols[idx % 3]:
                            st.image(path, use_container_width=True)
                            if st.button("🧬 Index", key=f"gal_idx_{f}", use_container_width=True):
                                with st.spinner("Analyzing..."):
                                    desc = st.session_state.llm.describe_image(path)
                                    st.session_state.kb.process_image_asset(f, desc, path)
                                    st.success(f"Indexed {f}")
                else:
                    st.info("No images generated yet.")
            else:
                st.info("Gallery directory not found.")

# --- PHASE 5: RESEARCH HUB ---

# --- FRAGMENTED CHAT HUB (Zero-Flicker Orchestration) ---
# Developer Note (Why Fragments?):
# Streamlit usually refreshes the ENTIRE page when any input changes. 
# Inside this @st.fragment, only the content within THIS function is refreshed
# when the user chats. This prevents the sidebar/tabs from flickering or 
# disabling during a long LLM generation.

@st.fragment
def render_chat_hub(engine_choice, ollama_ok):
    """
    Principal interface for real-time document research.
    Coordinates between user input, vector retrieval, and LLM streaming.
    """
    # 5.1 Chat Display
    chat_box = st.container(height=650, border=False)
    
    # Guidance Mode: If the KB is empty, we guide the user instead of searching.
    is_empty_kb = not st.session_state.kb.file_contents

    
    with chat_box:
        if is_empty_kb and not st.session_state.messages:
            st.markdown(f"""
                <div style='text-align: center; margin-top: 50px;'>
                    <div style='font-size: 60px;'>💠</div>
                    <h1>{st.session_state.llm.model_nickname} is Ready</h1>
                    <p style='color: #94a3b8;'>The intelligence engine is active, but your Knowledge Base is currently empty.</p>
                    <p style='color: #94a3b8; font-size: 14px;'>Ask me how to get started or initialize your data in the <b>System Settings</b> tab.</p>
                </div>
            """, unsafe_allow_html=True)
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
                
                # Render historical images
                if msg.get("type") == "image" and os.path.exists(msg.get("path", "")):
                    st.image(msg["path"])

                # Inline Results for Retrieval-only search
                if msg.get("type") == "results":
                    for fname, chunks in msg["data"].items():
                        with st.container(border=True):
                            avg_score = sum(r['score'] for r in chunks) / len(chunks)
                            st.markdown(f"<div style='display:flex;justify-content:space-between;'><span class='meta-label'>{fname}</span><span class='match-tag'>{int(avg_score*100)}% Match</span></div>", unsafe_allow_html=True)
                            with st.expander("View Segments", expanded=False):
                                for c in chunks:
                                    st.markdown(f"**Score: {c['score']:.2f} (Page {c.get('page','?')})**")
                                    st.write(c['text'])
                                    st.markdown("---")

        # 5.2 Input Command Handler
        if st.session_state.is_searching and st.session_state.pending_query:
            query = st.session_state.pending_query
            
            # --- IMAGE GENERATION COMMAND ---
            if query.lower().startswith("/image "):
                prompt = query[7:].strip()
                if not st.session_state.config.get("image_gen_active"):
                    st.session_state.messages.append({"role": "assistant", "content": "⚠️ **Image Generation is disabled.** Enable it in System Settings."})
                elif not st.session_state.img_service.is_model_downloaded():
                    st.session_state.messages.append({"role": "assistant", "content": f"⚠️ **Model not found.** Please download `{st.session_state.config.get('image_model_id')}` in System Settings."})
                else:
                    with st.chat_message("assistant"):
                        with st.spinner(f"🎨 Visualizing: '{prompt}'..."):
                            path, filename = st.session_state.img_service.generate_quick(prompt)
                            if path and os.path.exists(path):
                                st.image(path, caption=f"Generated: {prompt}")
                                st.session_state.messages.append({"role": "assistant", "content": f"🎨 Generated: **{prompt}**", "type": "image", "path": path, "filename": filename})
                                
                                # Indexing UI (inline)
                                if st.button("🧬 Index into Galaxy", key=f"index_{filename}"):
                                    with st.spinner("Analyzing with Neural Vision (Llava)..."):
                                        desc = st.session_state.llm.describe_image(path)
                                        st.session_state.kb.process_image_asset(filename, desc, path)
                                        st.success("Image indexed into the 3D map!")
                                        st.rerun()
                            else:
                                st.error(f"Generation failed: {filename}")
                
                st.session_state.is_searching = False
                st.session_state.pending_query = None
                st.rerun()

            # --- HELP COMMANDS ---
            if query.lower() in ["/help", "/examples"]:
                if engine_choice == "Machine Learning":
                    help_msg = "### 🔦 ML Search Tips\nUse specific keywords like **'Revenue 2024'** or **'Protocol X'**. Avoid natural questions as this engine matches literal vector intersections."
                else:
                    help_msg = "### 🧠 DL RAG Tips\nAsk natural questions like **'What are the key risks mentioned in file X?'** or **'Summarize the conclusion of page 4'**.\n\n**Visuals**: Use `/image [prompt]` to generate local visuals (requires activation)."
                st.session_state.messages.append({"role": "assistant", "content": help_msg})
                st.session_state.is_searching = False
                st.rerun()

            # --- SEARCH EXECUTION ---
            with st.status("💠 Processing Semantic Hub...", expanded=True) as status:
                if engine_choice == "Deep Learning" and ollama_ok:
                    # 1. RETRIEVAL: Pull 'Ground Truth' from the KnowledgeBase.
                    try:
                        ctx = st.session_state.kb.get_context_for_query(query, st.session_state.llm) if not is_empty_kb else None
                    except ValueError as ve:
                        st.error(str(ve))
                        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ **Search Blocked**: {str(ve)}"})
                        st.session_state.is_searching = False
                        st.rerun()
                    
                    # 2. SYSTEM GUIDANCE: Triggered if no files are indexed.
                    if is_empty_kb:
                        ctx = "SYSTEM_GUIDANCE_MODE: The user's knowledge base is empty. Instead of searching, guide the user on how to use the 'System Settings' tab to upload files (PDF, CSV, etc.) and build an index. Be encouraging."

                    if ctx or is_empty_kb:
                        with chat_box:
                            with st.chat_message("assistant"):
                                # 3. IDENTITY: Load the agent's persona (from AGENT.md).
                                agent_id_mgr = IdentityManager()
                                agent_context = agent_id_mgr.load_config()
                                # 4. MANIFEST: Tell the LLM which files exist for citation support.
                                manifest = st.session_state.kb.get_file_manifest() if hasattr(st.session_state.kb, 'get_file_manifest') else sorted(list(st.session_state.kb.file_contents.keys()))
                                
                                # 5. GENERATION: Stream the RAG-grounded response.
                                stream = st.session_state.llm.generate_rag_response(query, ctx, st.session_state.messages, 
                                                                                  agent_context=agent_context, 
                                                                                  file_manifest=manifest)
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
            st.rerun() # Local fragment rerun to update history
        else:
            # 5.3 Chat Input Box
            if prompt := st.chat_input("Enter your research query (or type /help)..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.is_searching = True
                st.session_state.pending_query = prompt
                st.rerun() # Local fragment rerun to trigger search block above

with tab_research:
    render_chat_hub(engine_choice, ollama_ok)

