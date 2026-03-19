# Personal Knowledge Base AI (NLP Final Project)

**Course:** AIGC 5501 - NLP  
**Project Phase:** Final — RAG-Powered Knowledge System  

This project is an enterprise-grade **Retrieval-Augmented Generation (RAG)** engine. It replaces traditional keyword search (TF-IDF) with **Modern Neural Embeddings** (via `mxbai-embed-large`) to provide deep semantic understanding of personal notes (PDFs, Markdown, Text) and **Datasets (CSV, Excel)**.

---

## Key Features

* **Neural Semantic Search:** Utilizes **Ollama's `mxbai-embed-large`** to generate high-dimensional embeddings for every document chunk, enabling context-aware retrieval.
* **Optimized Dataset Ingestion:** Handles large CSV and Excel files by chunking them into row-wise blocks for precise data retrieval and performance.
* **Manual Vector Math:** Performs manual **Cosine Similarity** on dense neural vectors using `numpy` dot products and norms, ensuring full control over the retrieval logic.
* **LLM-Powered RAG:** Uses **Ollama's Llama 3.1 (8B)** to synthesize natural-language answers grounded in retrieved data.
* **Real-time Streaming:** Token-by-token response streaming for a seamless user experience.
* **Multimodal Ingestion:** Native support for PDFs, Markdown, CSV, and Excel spreadsheets.
* **One-Click Summarization:** Instantly generate AI summaries for any uploaded document or dataset.
* **Premium Dark UI:** A professional, responsive Streamlit interface with chat history and source attribution.

---

## System Architecture

1. **Ingestion Layer:** Reads files and parses them into text. Datasets are processed in row-blocks for efficiency.
2. **Neural Core:** Chunks are sent to the local `mxbai-embed-large` model to generate 1024-dimensional embeddings.
3. **Knowledge Index:** Chunks and their dense vectors are stored in a local memory-resident index.
4. **Semantic Retrieval:** User queries are embedded and compared against the knowledge base via manual Cosine Similarity math.
5. **RAG Pipeline:** The most relevant context is injected into Llama 3.1 to produce a grounded, human-like response.

---

## 📦 Packages & Technologies

* **`streamlit`**: The interactive GUI and streaming framework.
* **`ollama`**: Local orchestration of Llama 3.1 and Neural Embeddings.
* **`pandas` & `openpyxl`**: High-performance data processing for spreadsheets.
* **`numpy`**: Manual linear algebra and vector operations.
* **`PyPDF2`**: PDF text extraction.

---

## 🚀 Setup & Installation

### Prerequisites

1. **Python 3.10+**
2. **Ollama** ([download from ollama.com](https://ollama.com))

### Step 1: Initialize Models

```bash
# Start the server
ollama serve

# Pull the LLM and Embedding models
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

### Step 2: Environment Setup

```bash
cd path/to/project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run Application

```bash
streamlit run app.py
```

> **Note:** The "Neural Engine" indicator in the sidebar will turn green when both the server and embedding models are available.
