# Developer Etiquette & Methodology
===================================

The NLP Workspace is as much an educational resource as it is a production tool. To maintain its high standard of legibility and reliability, all contributors must follow these guidelines.

## 🏗️ The Methodology: "Modular First"

We strictly adhere to a **Separation of Concerns**.
- **Never** add HTML/CSS directly into `app.py`. Use `utils/ui_components.py`.
- **Never** perform complex math in the UI layer. Use `core/knowledge_base.py`.
- **Never** hardcode local paths. Use relative paths or configuration variables.

## ✍️ Coding & Documentation Standards

### 1. The "Rationale Block"
Every new file **MUST** begin with a multi-line string (docstring) that explains:
- The module's purpose.
- Its role in the overall architecture.
- Any technical trade-offs made (e.g., speed vs. accuracy).

### 2. Google-Style Docstrings
All classes and public methods must use the Google Docstring format:
```python
def process_data(data: list, boost: float = 1.0) -> list:
    """
    Standardizes and weights incoming data points.
    
    Args:
        data (list): Raw data strings.
        boost (float): Multiplier for weighting.
        
    Returns:
        list: Normalized vector scores.
    """
```

### 3. Inline "Narrative" Comments
Complex algorithms (like Cosine Similarity or Neural Hashing) should transition from code to "narrative" comments that explain *why* the math is being done, not just *what* the code does.

## 🧪 Unit Testing Strategy

While we prioritize rapid development, all core engines must be verifiable.
- **Location**: Use a `tests/` directory (mapped 1:1 with `core/`).
- **Framework**: `pytest`.
- **Mocking**: Use `unittest.mock` for Ollama/Embedding server calls to test logic without requiring a local GPU.

## 🛠️ The Philosophy of Coding

1.  **Fail Fast**: Catch errors at the source (e.g., check server connectivity before trying to embed).
2.  **User-Centric Feedback**: Always provide status indicators (`st.status`, `st.progress`) for long-running NLP tasks.
3.  **Local Dev Priority**: The system should always be Runnable on a standard developer machine without cloud dependencies.
