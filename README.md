# 💠 NLP Engine // Hybrid Semantic Workspace
============================================

**Project Status:** 🚀 v3.5 - Refactored & Modularized

A professional-grade **Hybrid NLP Hub** designed for researcher-level knowledge management. This system implements a dual-engine vector core, enabling seamless switching between **Machine Learning** (Statistical) and **Deep Learning** (Neural) retrieval models.

---

## 📚 Documentation Suite

Explore the technical foundations and architectural philosophy of the Nexus Engine:

*   **⚡ [Quick Start Guide](./docs/LAUNCH_GUIDE.md)**: step-by-step instructions for Docker, Local, and Ollama deployment.
*   **[🚀 Getting Started (Onboarding)](./docs/ONBOARDING.md)**: Hardware requirements, prerequisites, and initial configuration.
*   **[🎯 Engine Feature Matrix](./docs/FEATURES_DETAIL.md)**: Hierarchical list of all capabilities with direct links to the source code.
*   **[🏗️ Architecture & Design](./docs/architecture_design.md)**: Deep dive into the "Hybrid Hub" philosophy and query lifecycle.
*   **[🎨 Image Intelligence](./docs/IMAGE_INTELLIGENCE.md)**: Guide to local Stable Diffusion synthesis and vision-based indexing.
*   **[🤝 Contributor Etiquette](./docs/etiquette.md)**: guidelines for coding standards and "Pedagogy-First" documentation.
*   **[📝 Project Roadmap](./TODO.md)**: Current status and upcoming feature priorities.


---

## ✨ Feature Highlights

- **Hybrid Search**: Toggle instantly between **TF-IDF Keyword Matching** and **Neural Semantic Search**.
- **Deep File Ingestion**: Automated extraction of **PDF, Markdown, TXT, and CSV** with recursive directory mounting.
- **RAG-Grounded Chat**: Context-aware AI assistant utilizing **Llama 3.1** via Ollama for document-verified responses.
- **Image Intelligence**: Local **Stable Diffusion** synthesis and **Llava-based vision** for generating and indexing visual assets.
- **Vector Analytics**: Real-time visualization of term importance and neural clustering performance.
- **Progressive Indexing**: Background-safe, asynchronous indexing to ensure zero UI locking during data ingestion.

---

## ⚡ Quick Start

The Nexus Engine is designed to be up and running in minutes. Choose your preferred deployment method in the **[Launch Guide](./docs/LAUNCH_GUIDE.md)**:

1.  **Docker Compose**: The recommended pathway for a guaranteed consistent environment.
2.  **Local Installation**: Best for active development and customization.
3.  **Ollama Setup**: Essential steps for pulling the required AI and Vision models.

---

## 🗄️ Core Structure

```text
├── app.py                 # Principal Orchestration Hub: Manages UI, Session State, and Fragment logic.
├── core/                  # Intelligence Engines: The architectural core of the system.
│   ├── knowledge_base.py  # Hybrid retrieval logic (TF-IDF & FAISS) and vector store management.
│   ├── llm_service.py     # Ollama API orchestration, vision-to-text, and RAG prompt assembly.
│   ├── image_service.py   # Stable Diffusion pipelines, VRAM management, and synthesis engine.
│   └── config_manager.py  # JSON-based persistence and real-time settings coordination.
├── utils/                 # Logistics & Interface: The supporting tools and UI utilities.
│   ├── ui_components.py   # Centralized CSS, sidebar layouts, and Streamlit Fragment definitions.
│   └── file_processor.py  # Universal extractor for multi-format document ingestion and chunking.
├── data/                  # Persistence Layer: Locally stored neural cache and user data.
│   ├── index/             # Serialized vector stores for high-speed local retrieval.
│   ├── images/            # Output gallery for synthesized and semantically tagged visual assets.
│   └── settings.json      # Dynamic user preferences and system configuration.
└── docs/                  # Documentation Suite: Comprehensive technical guides and pedagogy.
```
