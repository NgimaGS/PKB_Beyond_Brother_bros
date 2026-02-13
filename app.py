import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from knowledge_base import KnowledgeBase

st.set_page_config(page_title="NLP Midterm Project", layout="wide")
st.title("📚 Personal AI Knowledge Dashboard")

if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR: DATA PIPELINE ---
with st.sidebar:
    st.header("⚙️ Data Engineering")
    files = st.file_uploader("Upload PDF or Markdown", accept_multiple_files=True)

    if st.button("🚀 Process & Index"):
        if files:
            with st.status("Cleaning, Chunking, and Vectorizing...", expanded=True) as status:
                st.session_state.kb.documents = []
                st.session_state.kb.cleaning_report = []
                st.session_state.kb.file_chunk_counts = {}

                for f in files:
                    text = ""
                    if f.name.endswith(".pdf"):
                        reader = PdfReader(f)
                        text = " ".join([p.extract_text() or "" for p in reader.pages])
                    else:
                        text = f.read().decode("utf-8")
                    st.session_state.kb.process_text(f.name, text)

                st.session_state.kb.build_index()
                status.update(label="✅ Ready!", state="complete")
        else:
            st.warning("Please upload documents first.")

# --- UI: ANALYTICS & CLEANING REPORT ---
if st.session_state.kb.tfidf_matrix is not None:
    st.subheader("🧹 Preprocessing & Cleaning Report")
    with st.expander("View Data Reduction Metrics"):
        st.table(pd.DataFrame(st.session_state.kb.cleaning_report))
        st.info(
            "**Decision Log:** We used *Sentence-Aware Punctuation Logic* to ensure chunks represent complete thoughts.")

    st.subheader("Visual Analytics")
    tab1, tab2 = st.tabs(["Feature Importance", "Corpus Distribution"])

    with tab1:
        st.write("#### Top 10 Mathematically Significant Keywords")
        kw_df = st.session_state.kb.get_top_keywords_df(10)
        st.bar_chart(kw_df.set_index('Keyword'))
        st.caption("These words have the highest average TF-IDF weights across your dataset.")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### File Contribution")
            chunk_df = pd.DataFrame(list(st.session_state.kb.file_chunk_counts.items()), columns=['File', 'Chunks'])
            st.bar_chart(chunk_df.set_index('File'))
        with col2:
            st.write("#### Chunk Length Variability")
            st.line_chart([len(d) for d in st.session_state.kb.documents])
            st.caption("Shows how punctuation logic naturally varies chunk sizes.")

# --- CHAT RETRIEVAL INTERFACE ---
st.divider()
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

if prompt := st.chat_input("Query your knowledge base..."):
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        results = st.session_state.kb.search(prompt)
        if results:
            score, text = results[0]
            response = f"**Similarity Score: {score:.4f}**\n\n{text}"
            with st.expander("📊 View Secondary Matches"):
                for s, t in results[1:]:
                    st.write(f"**Score {s:.4f}:** {t}")
        else:
            response = "No relevant matches found."

        st.markdown(response)
        st.session_state.chat_history.append(("assistant", response))