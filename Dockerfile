# 🐋 Docker Container Definition
# ============================
# This Dockerfile encapsulates the Python environment required to run
# the Hybrid NLP Workspace. It is optimized for size and contains
# the necessary dependencies for PDF, CSV, and Neural processing.

FROM python:3.11-slim

# Set working directory to /app
WORKDIR /app

# ⚙️ Install System Dependencies
# ----------------------------
# We need build-essential for some NLTK/Scikit-learn extensions.
# We also include 'curl' for verifying connectivity to Ollama.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 📂 Prepare the Workspace
# ----------------------
# Create a dedicated 'data' volume for the neural cache to persist.
RUN mkdir -p /app/data

# 📦 Dependency Management
# -----------------------
# Copy and install Python requirements first to leverage Docker's layer cache.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 💠 Source Ingestion
# ------------------
# Copy the modular source tree into the container.
COPY . .

# 🚢 Network Orchestration
# -----------------------
# Streamlit's default listening port.
EXPOSE 8501

# 🩺 Healthcheck
# -------------
# Verifies that the internal web server is alive.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 🚀 Launch Environment
# --------------------
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
