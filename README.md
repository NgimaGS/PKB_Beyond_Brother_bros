# 💠 NLP Engine // Hybrid Semantic Workspace
============================================

**Project Status:** 🚀 v3.5 - Refactored & Modularized

A professional-grade **Hybrid NLP Hub** designed for researcher-level knowledge management. This system implements a dual-engine vector core, enabling seamless switching between **Machine Learning** (Statistical) and **Deep Learning** (Neural) retrieval models.

---

## 📚 Documentation Suite

Explore the technical foundations and architectural philosophy of the Nexus Engine:

*   **[🏗️ Architecture & Design](./docs/architecture_design.md)**: A deep dive into the "Hybrid Hub" philosophy and modular separation of concerns.
*   **[📊 Machine Learning Guide](./docs/machine_learning.md)**: Technical breakdown of TF-IDF, Lemmatization, and Statistical Retrieval.
*   **[🧠 Deep Learning Guide](./docs/deep_learning.md)**: Explaining Neural Embeddings, RAG, and Ollama integration.
*   **[🤝 Contributor Etiquette](./docs/etiquette.md)**: Essential guidelines for coding standards, documentation, and methodology.
*   **[📝 Project Roadmap](./TODO.md)**: Current status and upcoming feature priorities.

---

## ✨ Workspace Highlights

### 1. Hybrid Intelligence
Toggle instantly between **TF-IDF Keyword Search** (High Speed) and **Neural Semantic Search** (High Accuracy). Use the AI Research Hub to chat with your local documents using RAG-grounded Llama 3.1.

### 2. Deep File Ingestion
Recursive directory mounting supports mass-indexing of PDFs, Markdown, Text, and Datasets (CSV/Excel). Every file is automatically "Sentence-Aware" chunked for maximum search precision.

### 3. Integrated Vector Analytics
Visualize the statistical importance of terms across your corpus and monitor the NLP cleaning pipeline's efficiency in real-time.

---

## 🐋 Deployment via Docker

The workspace is fully containerized for consistent deployment across any environment.

### 1. Prerequisites
- [Docker & Docker Compose](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com/) (Running on your host machine)

### 2. Launching the Engine
```bash
# 1. Clone the repository
git clone <repo-url>
cd PKB_Beyond_Brother_bros

# 2. Build and start the container
docker-compose up --build
```
*Access the interface at **http://localhost:8501***

---

## 🛠️ Local Installation (Development)

### 1. Environment Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate | Unix: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launching the Workspace
```bash
streamlit run app.py
```

---

## 🗄️ Core Structure
```text
├── app.py                 # Principal Orchestration Hub
├── core/                  # Intelligence Engines (KnowledgeBase, LLM)
├── utils/                 # Logistics (UI Components, Ingestion)
├── data/                  # Persistent Storage (Neural Cache)
└── docs/                  # Technical Guides & Methodology
```
