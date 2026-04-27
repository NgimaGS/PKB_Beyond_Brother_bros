# 🛠️ Detailed Feature Matrix

This document provides a technical breakdown of every capability within the Nexus Engine, including direct links to the implementation logic for deep-dive research.

---

## 1.0 Intelligence Core
The backbone of the engine's retrieval and generation cycle.

### 1.1 Statistical Search (Machine Learning)
- **What**: High-speed keyword matching using Scikit-learn's TF-IDF Vectorizer.
- **Why**: Provides a "sanitized" ground truth based on exact lexical matches.
- **Core Logic**: [core/knowledge_base.py](../core/knowledge_base.py)

### 1.2 Neural Semantic Search (Deep Learning)
- **What**: Context-aware retrieval using FAISS (Facebook AI Similarity Search) and Ollama embeddings.
- **Why**: Understands intent and synonyms (e.g., finding "Fruit" when searching for "Apple").
- **Core Logic**: [core/knowledge_base.py](../core/knowledge_base.py)

---

## 2.0 Multi-Modal Intelligence
Expanding retrieval beyond text into visual and contextual reasoning.

### 2.1 Local Image Synthesis
- **What**: On-device image generation using Stable Diffusion pipelines.
- **Why**: Enables visual concept generation without external API dependencies.
- **Core Logic**: [core/image_service.py](../core/image_service.py)

### 2.2 Visual Intelligence (Vision-to-Text)
- **What**: Automated semantic description of images using Llava.
- **Why**: Bridges the gap between visual assets and text-based indexing.
- **Core Logic**: [core/llm_service.py](../core/llm_service.py)

---

## 3.0 Knowledge Management
Automated ingestion and organization of technical corpora.

### 3.1 Deep File Ingestion
- **What**: Extraction and chunking logic for PDF, Markdown, TXT, and CSV formats.
- **Why**: Normalizes diverse data sources into a unified vector space.
- **Core Logic**: [utils/file_processor.py](../utils/file_processor.py)

### 3.2 Progressive Indexing
- **What**: Asynchronous, background-safe updates to the vector index.
- **Why**: Prevents UI locking during mass-ingestion of large datasets.
- **Core Logic**: [core/knowledge_base.py](../core/knowledge_base.py)

---

## 4.0 User Interaction Layer
The orchestration of complex AI workflows into a clean interface.

### 4.1 RAG-Grounded Chat
- **What**: Prompt-injection mechanism that "roots" LLM responses in retrieved context.
- **Why**: Eliminates hallucinations by forcing the model to cite specific documents.
- **Core Logic**: [core/llm_service.py](../core/llm_service.py)

### 4.2 Dynamic Configuration
- **What**: Real-time threshold adjustment and engine switching.
- **Why**: Allows users to tune performance based on their hardware and accuracy needs.
- **Core Logic**: [core/config_manager.py](../core/config_manager.py)

---
*Created by the Nexus Engine // Pedagogy-First Documentation*
