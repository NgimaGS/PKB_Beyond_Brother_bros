# app.py
import streamlit as st
from PyPDF2 import PdfReader

# Import YOUR custom machine learning backend
from knowledge_base import KnowledgeBase

# ==========================================
# THE STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="My AI Notes", page_icon="📚")
st.title("Personal Knowledge Base")
st.caption("Upload your PDFs or Markdown notes and ask questions about them!")

# Initialize session state (keeps data alive when the page refreshes)
if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR: FILE UPLOADER ---
with st.sidebar:
    st.header("1. Upload Knowledge")
    uploaded_files = st.file_uploader("Upload PDFs or Text", accept_multiple_files=True)

    if st.button("Process & Index Files"):
        if uploaded_files:
            with st.spinner("Extracting text and building math vectors..."):
                # Reset documents if re-indexing
                st.session_state.kb.documents = []

                for file in uploaded_files:
                    text = ""
                    if file.name.endswith(".pdf"):
                        reader = PdfReader(file)
                        for page in reader.pages:
                            text += page.extract_text() + " "
                    else:
                        text = file.read().decode("utf-8")

                    # Send text to your backend
                    st.session_state.kb.process_text(text)

                # Build the mathematical space
                st.session_state.kb.build_index()
                st.success(f"Success! Created {len(st.session_state.kb.documents)} text chunks.")
        else:
            st.warning("Please upload files first.")

# --- MAIN AREA: CHAT INTERFACE ---
st.header("2. Chat with your Notes")

# Display previous messages
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# Handle new user input
if user_input := st.chat_input("Ask a question about your documents..."):
    # 1. Show user message
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Generate and show assistant response
    with st.chat_message("assistant"):
        # Call your backend search function
        results = st.session_state.kb.search(user_input)

        if not results:
            response = "I couldn't find any relevant information. Have you uploaded and indexed your files?"
            st.markdown(response)
        else:
            best_score, best_text = results[0]

            # Threshold to ensure it's actually a relevant match
            if best_score > 0.05:
                response = f"**Best Match (Similarity Score: {best_score:.2f})**\n\n> {best_text}"
                st.markdown(response)

                # Midterm Grading Feature: Expandable "Show Math Work"
                with st.expander("Show Math Breakdown"):
                    st.write("Calculated via manual Dot Product / Magnitudes.")
                    for i, (score, chunk) in enumerate(results):
                        st.caption(f"Rank {i + 1} | Score: {score:.4f} | Snippet: {chunk[:50]}...")
            else:
                response = "I couldn't find a strong mathematical match for that query in your documents."
                st.markdown(response)

        st.session_state.chat_history.append(("assistant", response))