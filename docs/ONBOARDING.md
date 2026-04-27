# 🚀 Developer Onboarding Guide

Welcome to the **Nexus Engine** (PKB Beyond Brother Bros). This project is designed as both a professional-grade research tool and an educational resource for modern NLP.

## 🧠 The Mental Model

Before writing code, understand the **Hybrid Philosophy**:
*   **Machine Learning (Statistical)**: We use **TF-IDF**. It's fast, local, and doesn't need a GPU. It's the "keyword search" that tells you *exactly* which words matched.
*   **Deep Learning (Neural)**: We use **Ollama + Dense Embeddings**. It understands *meaning*. If you search for "automobile," it finds documents about "cars."

The system is built so you can switch between these instantly.

---

## 💻 System Requirements

To ensure a smooth experience with the engine's neural core and image intelligence, please refer to the following tiers:

### 🥉 Tier 1: Minimum (CPU-Only)
*   **RAM**: 16GB (Strict minimum for loading Llama 3.1 8B).
*   **CPU**: 4+ Cores (modern Intel i5/AMD Ryzen 5).
*   **Storage**: 20GB free space for models.
*   **Experience**: Acceptable chat response; Image generation will take several minutes per image.

### 🥈 Tier 2: Functional (Dedicated GPU)
*   **VRAM**: 4GB - 6GB (NVIDIA RTX 2060 / 3050 series).
*   **RAM**: 16GB.
*   **Experience**: Fast Stable Diffusion v1.5 generation; acceptable vision analysis speeds.

### 🥇 Tier 3: Recommended (High-Performance)
*   **VRAM**: 8GB - 12GB+ (NVIDIA RTX 3080 / 4070+).
*   **RAM**: 32GB.
*   **Experience**: Near-instantaneous SDXL Turbo generation; smooth multi-modal context processing.

---

## 🛠️ Environmental Setup & Prerequisites

Before proceeding, ensure the following software is installed:

### 1. The Local AI Stack (Ollama)
The engine relies on Ollama for both LLM and Vision tasks.
1.  **Install**: [Ollama.com](https://ollama.com/)
2.  **Pull Models**:
    ```bash
    ollama pull llama3.1:8b            # Primary Brain
    ollama pull mxbai-embed-large      # Similarity Senses
    ollama pull llava:latest           # Visual Intelligence (Vision)
    ```

### 2. The Python Environment
We use Python 3.10+ for core logic.
```bash
# Setup virtual environment
python -m venv venv
# Activate (Windows: venv\Scripts\activate | Unix: source venv/bin/activate)
pip install -r requirements.txt
```

### 3. GPU Acceleration (NVIDIA)
If using a GPU, ensure you have the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed to enable hardware acceleration in Torch.

---

## 🏗️ Verification & Health Check

Run the app locally:
```bash
streamlit run app.py
```
Check the **"System Health"** section in the sidebar. If "Ollama Connection" is green, you are ready. Use the **"Hardware"** indicator to verify if the system is utilizing your GPU.

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
