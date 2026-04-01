# Architecture Design & Design Philosophy
=========================================

The NLP Workspace is built as a **Hybrid-Engine Semantic Hub**. This architecture was chosen to balance the raw speed of traditional statistics with the contextual depth of modern AI.

## 🏗️ The 4-Layer Modular Structure

We follow a strict **Separation of Concerns** to ensure the project remains maintainable as it grows:

1.  **Orchestration Layer (`app.py`)**: The "Management Hub". It manages Streamlit's session state and UI rendering. It does *not* contain heavy math or file parsing logic.
2.  **Intelligence Engines (`core/`)**: The "Brain".
    - `knowledge_base.py`: Handles vectorization and search.
    - `llm_service.py`: Handles LLM communication and token analytics.
3.  **Utility & Interface (`utils/`)**: The "Toolbox".
    - `ui_components.py`: Centralizes all CSS and HTML layouts.
    - `file_processor.py`: Orchestrates multi-format data ingestion.
4.  **Persistence Layer (`data/`)**: The "Memory". Stores the neural vector cache and future datasets.

## 🧠 Design Philosophy

### 1. "Hybrid First"
By providing both **Machine Learning** (TF-IDF) and **Deep Learning** (Neural) engines, the user is never locked into a single implementation. This allows for rapid keyword-based "sanity checks" before running expensive neural queries.

### 2. Pedagogy Through Code (PTC)
The codebase is **heavily documented** at the top of every file and within every complex method. This is designed so a new developer can learn the theory of NLP simply by reading the source code.

### 3. Local-First Privacy
All processing happens Locally.
- **Ollama** ensures LLM data stays on-device.
- **Scikit-learn** ensures statistical data stays on-device.
- This architecture was chosen to satisfy high-security requirements for sensitive internal documentation.

### 4. Responsiveness Over Complexity
We use **Streaming RAG** in the UI. While the LLM calculates the full result, the user sees progress in real-time. This reduces perceived latency and improves the user experience during deep reasoning tasks.
