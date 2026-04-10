# 🚀 Developer Onboarding Guide

Welcome to the **Nexus Engine** (PKB Beyond Brother Bros). This project is designed as both a professional-grade research tool and an educational resource for modern NLP.

## 🧠 The Mental Model

Before writing code, understand the **Hybrid Philosophy**:
*   **Machine Learning (Statistical)**: We use **TF-IDF**. It's fast, local, and doesn't need a GPU. It's the "keyword search" that tells you *exactly* which words matched.
*   **Deep Learning (Neural)**: We use **Ollama + Dense Embeddings**. It understands *meaning*. If you search for "automobile," it finds documents about "cars."

The system is built so you can switch between these instantly.

---

## 🛠️ Environmental Setup

### 1. The Python Stack
We use Python 3.10+.
```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# Windows: venv\Scripts\activate
# Unix: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 2. The Local AI Stack (Ollama)
This project requires [Ollama](https://ollama.com/) to be running on your local machine.
1. Download and install Ollama.
2. Pull the default models:
```bash
ollama pull llama3.1:8b            # The "Brain" (LLM)
ollama pull mxbai-embed-large      # The "Senses" (Embeddings)
ollama pull nomic-embed-text       # Optional: High speed embeddings
```

### 3. Verification
Run the app locally:
```bash
streamlit run app.py
```
Check the **"System Health"** section in the sidebar. If "Ollama Connection" is green, you are ready.

---

## 🏗️ Architecture for Newcomers

We follow a strict **Separation of Concerns**:

| Layer | Responsibility | File Location |
| :--- | :--- | :--- |
| **Orchestration** | UI Layout, State Management, User Input | `app.py` |
| **Intelligence** | Vector Math, Search, LLM Chat, Token Stats | `core/` |
| **Logistics** | File Parsing, GDrive hydration, UI CSS | `utils/` |
| **Persistence** | Vector Caching, User Settings | `data/` |

---

## 🎭 Customizing Your Agent

The assistant's "soul" is stored in [AGENT.md](file:///c:/Users/Shaine/Documents/GitHub/PKB_Beyond_Brother_bros/AGENT.md). 
You can modify this markdown file directly or via the UI to:
1.  **Change Identity**: Give the engine a new name and persona.
2.  **Adjust Boundaries**: Define what the AI should and shouldn't talk about.
3.  **Prompt Engineering**: Add specific instructions on how it should cite sources or summarize documents.

---

## 🛠️ Implementing Your First Feature

Ready to contribute? Here is the standard workflow for adding a new "Research Tool":
1.  **Logic**: Add a new method in `core/llm_service.py` (for AI tasks) or `core/knowledge_base.py` (for data tasks).
2.  **UI**: Add a new control (button/slider) in `app.py`. Wrap it in an `@st.fragment` if it modifies the query view.
3.  **State**: Register any new persistent settings in `ConfigManager.DEFAULT_CONFIG`.

---

## 🧹 Resetting the Engine

If you find yourself in an inconsistent state or want to clear all data:
1.  **Clear Index**: Delete the contents of `data/index/`.
2.  **Clear Config**: Delete `data/settings.json`. The app will recreate it with defaults on the next run.
3.  **Clear Cache**: Delete any `.cache_*.json` files in `data/` to force a re-generation of neural clusters.

*   **Rationale Blocks**: Every new module should start with an explanation of *why* it exists.
*   **Fail Fast**: Check for the user's data or server connectivity *before* starting heavy computation.
*   **Pedagogy First**: If you write complex math (like a custom cosine similarity), add a comment explaining the theory.

---

## 🆘 Troubleshooting

*   **"No Context Found"**: The Similarity Threshold might be too high. Check `app.py`'s `neural_threshold` logic.
*   **UI Flickering**: Ensure new interactive elements are wrapped in an `@st.fragment` if they update frequently.
*   **GDrive Files Missing**: Google Drive often stores files in the cloud until they are opened. Use the **"Test Path Visibility"** button in the UI to verify the files are actually readable on disk.
