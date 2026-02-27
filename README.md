# Nexus Intelligence Engine (NLP Foundation)

**Project Goal:** A professional-grade Semantic Search & Retrieval workspace designed for researcher-level knowledge management. This system implements the "Vector Core" required for Retrieval-Augmented Generation (RAG) within a high-performance, wide-screen interface.

---

## 🧠 NLP Capability Mapping
The following modern NLP tasks are implemented within the core engine as foundational algorithmic blocks:

| NLP Capability | Implementation in Code | Logic Location | AI Methodology |
| :--- | :--- | :--- | :--- |
| **Question-Answering** | **Semantic Retrieval:** Finding precise "Answer Units" by calculating the cosine similarity between a natural language query and document vectors. | `knowledge_base.py:search` | **Traditional ML** |
| **Topic Modelling** | **Feature Importance:** Aggregating global TF-IDF weights to determine the dominant themes/keywords across the corpus. | `knowledge_base.py:get_top_keywords_df` | **Traditional ML** |
| **Text Classification** | **Vector Feature Extraction:** Converting raw strings into high-dimensional numerical feature vectors using TF-IDF. | `knowledge_base.py:build_index` | **Traditional ML** |
| **Text Summarization** | **Extractive Chunking:** The "Sentence-Aware Chunking" logic acts as an extractive summarizer by selecting the most relevant segments. | `knowledge_base.py:process_text` | **Traditional ML** |
| **Context Windowing** | **RAG Foundation:** Provides the specific content segments required for Large Language Models (LLMs) to generate grounded responses. | `app.py:Research Tab` | **Foundation Logic** |
| **Preprocessing & Normalization** | **NLP Pipeline:** Standardizing base vocabulary (Lemmatization) and noise reduction for semantic consistency. | `knowledge_base.py:clean_text` | **Traditional ML** |

---

## ✨ Professional Workspace Features

### 1. Multi-Tabbed Research Environment
*   **💠 Research Hub:** Unified search interface with state-aware loading monitors and persistent result cards.
*   **📊 Vector Analytics:** Deep-dive into document preprocessing metrics and TF-IDF keyword importance charts.
*   **⚙️ System Settings:** Centralized hub for knowledge ingestion and engine initialization.

### 2. High-Capacity Knowledge Ingestion
*   **Recursive Directory Mounting:** Mount any local folder path. The engine automatically scans all sub-folders for PDF and Markdown files.
*   **Manual Upload Support:** Browser-based uploads for quick, in-memory indexing of document units.

### 3. Unified Nexus Result Cards
*   **Contextual Intelligence:** Consolidates metadata, match probability, and content segments into a single interactive card.
*   **Native OS Accessibility:** 
    *   **📄 Open File:** One-click launch of the source document in its native application (Acrobat, Notepad, etc.).
    *   **📁 Open Folder:** Automaticaly opens the containing directory and highlights the specific file in Windows Explorer.

### 4. Educational Guidance
*   **NLP Tooltips:** Every UI component includes a `(?)` hover tooltip explaining the specific NLP mechanic (e.g., Cosine Similarity, Vectorization) at work.
*   **Vector Metric Captions:** Detailed explanations for "Feature Importance" and the "Preprocessing Pipeline" located within the Analytics suite.

---

## 🛠 Documentation for Learning

### 1. The Preprocessing Pipeline (`clean_text`)
Before a machine can "understand" text, it must be normalized:
*   **Lemmatization:** Reducing words like "running" or "ran" to their root "run". This ensures that a query for "run" matches documents containing any variation.
*   **Noise Reduction:** Standardizing NFKD characters and stripping URLs while preserving sentence semantics.

### 2. Semantic Chunking (`process_text`)
We use **Sentence-Aware Chunking** to break artifacts into "Knowledge Units." This preserves the context of a thought, unlike fixed-length slicing, leading to much higher search precision.

### 3. Vectorization & Mathematical Retrieval (`build_index` & `search`)
*   **TF-IDF Weights:** We calculate word importance based on global rarity vs. local frequency.
*   **Cosine Similarity:** We treat every document chunk as a vector in 1000+ dimensional space. We calculate the **Dot Product** divided by the **Magnitudes** to find the "angle" between your question and the documents.

---

## 🚀 Step-by-Step Installation

### Step 1: Create & Activate Environment
```bash
python -m venv venv
# Windows: venv\Scripts\activate | Mac: source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Engine
```bash
streamlit run app.py
```
