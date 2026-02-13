# Personal Knowledge Base AI (NLP Midterm)

**Course:** AIGC 5501 - NLP  
**Project Phase:** Midterm — The Machine Learning Foundation  

This project is a custom-built Search & Retrieval engine that acts as the foundation for a Retrieval-Augmented Generation (RAG) chatbot. It processes personal notes (PDFs and Markdown), vectorizes the text using TF-IDF, and retrieves the most relevant document chunks using **manually calculated Cosine Similarity**.

---

## Key Features (Midterm Requirements)

* **Data Preprocessing & Cleaning:** Uses Python's `re` module to strip Markdown noise (`#`, `*`, `>`) and normalizes text for clean embedding.
* **Manual Chunking Strategy:** Implements a custom sliding-window chunking algorithm (Default: 500 characters with a 50-character overlap) to preserve semantic context without diluting the vector.
* **TF-IDF Vectorization:** Utilizes `scikit-learn`'s `TfidfVectorizer` to weight words based on statistical importance rather than raw frequency.
* **The "Midterm Twist" (Manual Linear Algebra):** Bypasses high-level similarity libraries. Instead, it manually calculates the **Dot Product** and **Magnitudes (Norms)** using `numpy` to find the nearest neighbor in high-dimensional space.
* **Interactive UI:** Built with **Streamlit** to provide a seamless chat interface and dynamic file uploading.

---

## System Architecture: How It Works



This application operates in four distinct sequential phases:

**1. Document Ingestion:**
The user uploads a file (`.pdf` or `.md`) via the Streamlit interface. The system reads the binary data. If it is a PDF, it iterates through every page and extracts the raw text.

**2. Preprocessing & Chunking:**
Raw text is inherently messy. The application first strips out structural noise (like Markdown headers or excessive line breaks) using regular expressions. It then passes the cleaned text through a "sliding window" chunking algorithm. This splits the massive document into 500-character blocks, with a 50-character overlap between blocks to ensure that concepts split across two chunks do not lose their semantic context.

**3. Vectorization (TF-IDF):**
The chunks are fed into a Machine Learning vectorizer. Instead of just counting words, the system calculates the Term Frequency-Inverse Document Frequency (TF-IDF). It scales down the impact of tokens that occur very frequently in a given corpus, which are empirically less informative than features that occur in a small fraction of the training corpus. The result is a high-dimensional mathematical matrix representing the entire knowledge base.



**4. Mathematical Retrieval:**
When a user asks a question, that question is converted into a vector using the exact same vocabulary. The system then loops through every document chunk in the matrix and calculates the **Cosine Similarity** (measuring the angle between the query vector and the document vector). The chunk with the highest similarity score (closest to 1.0) is returned to the user.

---

## 📦 Packages & Technologies Used

* **`streamlit`**: The web framework used to build the interactive Graphical User Interface (GUI). It handles file uploads, session state (keeping data alive across interactions), and rendering the chat window.
* **`PyPDF2`**: A pure-Python library built as a PDF toolkit. It is used in the ingestion phase to parse the binary data of uploaded PDFs and extract human-readable text from individual pages.
* **`scikit-learn`**: A premier machine learning library in Python. This project explicitly uses its `TfidfVectorizer` to convert the collection of raw text chunks into a matrix of TF-IDF features. It handles tokenization, building the vocabulary, and automatically applying the TF-IDF weighting formulas.
* **`numpy`**: The fundamental package for scientific computing in Python. It is used to perform the manual linear algebra required for the midterm twist. Specifically, `np.dot()` calculates the dot product, and `np.linalg.norm()` calculates the vector magnitudes.
* **`re`**: Python's built-in Regular Expression engine. It is utilized in the data cleaning phase to target and substitute out specific noise patterns (like removing `#` characters and condensing multiple `\n` newlines into single spaces).
* **`os`**: Python's built-in operating system interface, used to handle secure file path routing and directory management.

---

## 📁 Project Structure

To ensure a professional separation of concerns, the logic is split into an ML backend and a web frontend:

* `knowledge_base.py` : The Machine Learning engine (Vectorization, Chunking, Math).
* `app.py` : The Streamlit User Interface and session state management.
* `requirements.txt` : Project dependencies.

---

## 🚀 Installation & Setup

This project uses a Python Virtual Environment (`venv`) to ensure all dependencies are isolated and do not conflict with your system's global Python packages.

## 🚀 Step-by-Step Instructions

### Step 1: Open your Terminal
Open your preferred terminal (Command Prompt, PowerShell, or Terminal) and navigate to your project directory where `app.py` and `requirements.txt` are located.

```bash
  cd path/to/your/project
```

### Step 2: Create the Virtual Environment
Run the following command to generate a new folder named venv inside your project directory. This folder will store the isolated Python interpreter.

```bash
  python -m venv venv
```

### Step 3: Activate the Virtual Environment
Before installing packages, you must "enter" the environment. The activation command depends on your Operating System:

```bash
  venv\Scripts\activate
```

#### macOS / Linux:
```bash
  source venv/bin/activate
```
### Step 4: Install Project Requirements
With the virtual environment active, use pip to install all dependencies listed in the requirements.txt file.

```bash
  pip install -r requirements.txt
```
### Step 4: Install Project Requirements
Once the installation is complete, launch the web interface using Streamlit.

```bash
  streamlit run app.py
```
