"""
LLM Service Layer — Ollama Integration & Token Analytics
========================================================

Architecture Rationale:
-----------------------
This module encapsulates all communication with the local Ollama inference server.
By isolating LLM logic here, the main application remains 'model-agnostic'. 
It manages:
1. Neural Embeddings (Dense Vector generation).
2. RAG (Retrieval-Augmented Generation) Chat Orchestration.
3. Document Summarization.
4. Token Analysis (Measuring input/output efficiency).

Design Pattern: Service Provider
The OllamaService acts as a swappable provider. If we transition to OpenAI or
Anthropic, only this file needs significant modification.
"""

import ollama
import re

class OllamaService:
    """
    Manages interactions with a local Ollama instance.
    Includes built-in connectivity checks and token usage tracking.
    """

    def __init__(self, model_name: str = "llama3.1:8b", embedding_model: str = "mxbai-embed-large"):
        """
        Initializes the service with specific models.
        
        Args:
            model_name (str): The LLM used for chat and summarization (e.g., Llama 3.1).
            embedding_model (str): The model used for generating dense vectors (e.g., mxbai).
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self._available = None  # Cached status to prevent redundant network calls
        
        # Token Analytics State
        self.last_run_stats = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

    # ------------------------------------------------------------------
    # 1. NEURAL CORE (Embeddings)
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """
        Generates a 1024-dimensional neural embedding for a single text chunk.
        Used for real-time query vectorization.
        """
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generates neural embeddings for a list of texts using native batching.
        Significantly faster than individual calls during initial document indexing.
        """
        try:
            # Use the newer .embed() API for optimized batch processing
            response = ollama.embed(model=self.embedding_model, input=texts)
            return response.get("embeddings", [])
        except Exception as e:
            print(f"Batch embedding error: {e}")
            return []

    # ------------------------------------------------------------------
    # 2. SYSTEM HEALTH (Connectivity)
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """
        Verifies that the Ollama server is reachable and required models are downloaded.
        Returns False if the server is down or models are missing.
        """
        if self._available is not None:
            return self._available
        try:
            models = ollama.list()
            # Extract models names (Ollama returns objects with a 'model' attribute)
            model_names = [m.model for m in models.models]
            
            # Check for both the LLM and the Embedding model
            has_llm = any(self.model_name in name for name in model_names)
            has_embed = any(self.embedding_model in name for name in model_names)
            
            self._available = has_llm and has_embed
            return self._available
        except Exception:
            self._available = False
            return False

    def reset_status(self):
        """Forces a fresh heart-beat check on the next availability request."""
        self._available = None

    # ------------------------------------------------------------------
    # 3. CONTEXTUAL INTELLIGENCE (RAG & Chat)
    # ------------------------------------------------------------------

    def _build_rag_messages(self, query: str, context_text: str,
                             chat_history: list[dict] | None = None) -> list[dict]:
        """
        Constructs a prompt that grounds the LLM in the provided document context.
        Implements a strict 'System' role to prevent hallucinations.
        """
        system_prompt = (
            "You are a knowledgeable research assistant. Your job is to answer "
            "the user's question using ONLY the provided document context below.\n\n"
            "--- DOCUMENT CONTEXT ---\n"
            f"{context_text}\n"
            "--- END CONTEXT ---\n\n"
            "Constraints:\n"
            "- If the answer isn't in the context, say 'I don't have enough information'.\n"
            "- Cite specific filenames and page numbers if available.\n"
            "- Keep technical explanations precise."
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Append last 5 turns of history for conversational memory
        if chat_history:
            relevant_history = [m for m in chat_history if m.get("role") in ("user", "assistant")][-5:]
            for msg in relevant_history:
                # Clean out HTML tags if any were stored in history
                clean_content = re.sub(r'<[^>]+>', '', str(msg.get("content", ""))).strip()
                if clean_content:
                    messages.append({"role": msg["role"], "content": clean_content})

        messages.append({"role": "user", "content": query})
        return messages

    def generate_rag_response(self, query: str, context_text: str,
                               chat_history: list[dict] | None = None):
        """
        Streams a RAG-grounded response from Ollama.
        Yields text chunks and captures token usage stats upon completion.
        """
        messages = self._build_rag_messages(query, context_text, chat_history)

        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options={"temperature": 0.3}
            )

            current_response = ""
            for chunk in stream:
                token_text = chunk["message"]["content"]
                current_response += token_text
                
                # Capture stats if provided in the final chunk
                if chunk.get("done"):
                    self.last_run_stats = {
                        "input_tokens": chunk.get("prompt_eval_count", 0),
                        "output_tokens": chunk.get("eval_count", 0),
                        "total_tokens": chunk.get("prompt_eval_count", 0) + chunk.get("eval_count", 0)
                    }
                
                if token_text:
                    yield token_text
        except Exception as e:
            yield f"⚠️ LLM Error: {str(e)}"

    # ------------------------------------------------------------------
    # 4. SUMMARIZATION & ANALYTICS
    # ------------------------------------------------------------------

    def summarize_text(self, text: str, filename: str) -> str:
        """
        Uses the LLM to generate a structured executive summary of a document.
        Optimized for high-density information extraction.
        """
        # Truncate to stay within context window limits (~12k chars safety)
        safe_text = text[:12000]
        
        messages = [
            {"role": "system", "content": "Summarize this document strictly into 3 bullet points: Context, Key Findings, and Conclusion."},
            {"role": "user", "content": f"Document: {filename}\nContent:\n{safe_text}"}
        ]

        try:
            response = ollama.chat(model=self.model_name, messages=messages, stream=False)
            
            # Record analytics
            self.last_run_stats = {
                "input_tokens": response.get("prompt_eval_count", 0),
                "output_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
            }
            
            return response["message"]["content"]
        except Exception as e:
            return f"Failed to generate summary: {str(e)}"

    def get_last_stats(self) -> dict:
        """Returns the token analytics from the most recent LLM operation."""
        return self.last_run_stats
