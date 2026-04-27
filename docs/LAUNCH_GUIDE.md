# 🚀 Deployment & Launch Guide

This document provides step-by-step instructions for deploying the Nexus Engine in various environments.

---

## 🛠️ Prerequisites

Before launching the engine, ensure your system has the following core dependencies installed:

1.  **Python 3.10+**: [Download here](https://www.python.org/downloads/)
2.  **Ollama**: Essential for local LLM and Vision intelligence. [Download here](https://ollama.com/)
3.  **Docker & Docker Compose**: (Optional) For containerized deployment. [Download here](https://www.docker.com/products/docker-desktop/)
4.  **CUDA Toolkit**: (Strongly Recommended) For GPU acceleration on Windows/Linux.

---

## 🐋 Option 1: Docker Deployment (Recommended for Consistency)

The workspace is fully containerized. This method ensures all dependencies (including machine learning libs) are configured correctly.

### 1. Build and Start
```bash
# Clone the repository (if not already done)
git clone <repo-url>
cd PKB_Beyond_Brother_bros

# Launch via Docker Compose
docker-compose up --build
```

### 2. Accessing the UI
Once the build is complete, access the interface at:  
👉 **http://localhost:8501**

---

## 🛠️ Option 2: Local Installation (Development Mode)

Use this method if you want to modify the source code and see changes in real-time.

### 1. Environment Setup
```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# Unix/MacOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Ollama Models
The engine requires specific models to be available on your local Ollama server.
```bash
ollama pull llama3.1:8b            # Text Intelligence
ollama pull mxbai-embed-large      # Neural Indexing
ollama pull llava:latest           # Vision & Image Description
```

### 3. Launching the Workspace
```bash
# Start the Streamlit application
streamlit run app.py
```

---

## 🩺 Verification & Health Check

After launching, check the **"System Health"** section in the sidebar within the app.
- **Ollama Connection**: Must be Green.
- **Index Status**: Will show "Active" if documents are loaded.
- **Device**: Should display "CUDA" if a GPU was detected.

---
*Back to [README.md](../README.md)*
