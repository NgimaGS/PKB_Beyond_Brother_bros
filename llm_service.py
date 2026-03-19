"""
LLM Service Layer — Ollama Integration for RAG and Summarization.

Encapsulates all communication with the locally-running Ollama instance.
Provides streaming RAG responses and document summarization via Llama 3.3.
"""

import ollama


class OllamaService:
    """Manages interactions with a local Ollama instance."""

    def __init__(self, model_name: str = "llama3.1:8b", embedding_model: str = "mxbai-embed-large"):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self._available = None  # cached availability status

    def embed_text(self, text: str) -> list[float]:
        """Generates a neural embedding for the given text."""
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generates neural embeddings for a list of texts using native batching."""
        try:
            # Native Ollama batch support is much faster for local models
            response = ollama.embed(model=self.embedding_model, input=texts)
            return response.get("embeddings", [])
        except Exception as e:
            print(f"Batch embedding error: {e}")
            # Fallback to single if batch fails (though error is caught)
            return []

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable and the model exists."""
        if self._available is not None:
            return self._available

        try:
            models = ollama.list()
            model_names = [m.model for m in models.models]
            # Match on base name (e.g. "llama3.3" matches "llama3.3:latest")
            self._available = any(
                self.model_name in name for name in model_names
            )
        except Exception:
            self._available = False

        return self._available

    def reset_status(self):
        """Force a fresh connectivity check on the next call."""
        self._available = None

    # ------------------------------------------------------------------
    # RAG Response
    # ------------------------------------------------------------------

    def _build_rag_messages(self, query: str, context_text: str,
                            chat_history: list[dict] | None = None) -> list[dict]:
        """Assemble the messages list for a RAG chat request."""

        system_prompt = (
            "You are a knowledgeable research assistant. Your job is to answer "
            "the user's question using ONLY the provided document context below. "
            "If the context does not contain enough information to answer, say so "
            "clearly — do NOT make things up.\n\n"
            "Rules:\n"
            "- Ground every claim in the provided context.\n"
            "- Cite the source file and page when possible.\n"
            "- Be concise but thorough.\n"
            "- Use markdown formatting for readability.\n\n"
            "--- DOCUMENT CONTEXT ---\n"
            f"{context_text}\n"
            "--- END CONTEXT ---"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Include recent chat history for multi-turn context (last 10 msgs)
        if chat_history:
            for msg in chat_history[-10:]:
                # Include standard chat and manual summaries for better context
                if msg.get("role") in ("user", "assistant") and msg.get("type") != "stats":
                    content = msg.get("plain_text", msg.get("content", ""))
                    # Clean out HTML tags if any (basic check)
                    import re
                    clean_content = re.sub(r'<[^>]+>', '', content).strip()
                    if clean_content:
                        messages.append({
                            "role": msg["role"],
                            "content": clean_content
                        })

        messages.append({"role": "user", "content": query})
        return messages

    def generate_rag_response(self, query: str, context_text: str,
                              chat_history: list[dict] | None = None):
        """
        Stream a RAG-grounded response from Ollama.

        Yields string chunks as the model generates them.
        """
        messages = self._build_rag_messages(query, context_text, chat_history)

        stream = ollama.chat(
            model=self.model_name,
            messages=messages,
            stream=True,
            options={"temperature": 0.4, "top_p": 0.9},
        )

        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token

    # ------------------------------------------------------------------
    # Document Summarization
    # ------------------------------------------------------------------

    def summarize_text(self, text: str, filename: str) -> str:
        """Return a concise summary of the given document text."""

        # Truncate very long documents to avoid exceeding context limits
        max_chars = 12_000
        truncated = text[:max_chars]
        was_truncated = len(text) > max_chars

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise document summarizer. Produce a clear, "
                    "well-structured summary of the provided document. "
                    "Use bullet points for key topics. Keep it under 300 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Summarize the following document"
                    f"{' (note: truncated due to length)' if was_truncated else ''}:\n\n"
                    f"**File:** {filename}\n\n"
                    f"{truncated}"
                ),
            },
        ]

        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            stream=False,
            options={"temperature": 0.3},
        )

        return response["message"]["content"]
