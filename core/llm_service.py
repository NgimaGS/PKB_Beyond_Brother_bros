"""
LLM Service Layer — Ollama Integration & Token Analytics
========================================================

Architecture Rationale:
-----------------------
This module encapsulates all communication with the local Ollama inference server.
By isolating LLM logic here, the main application remains 'model-agnostic'. 

Core Responsibilities:
1. **Neural Embeddings**: Dense Vector generation for semantic search.
2. **RAG Orchestration**: Injecting retrieved context into LLM prompts.
3. **Summarization**: Distilling long documents into concise briefs.
4. **Token Analytics**: Capturing inference metrics for performance monitoring.

Design Pattern: Service Provider
The OllamaService acts as a pluggable provider. If we transition to OpenAI or
Anthropic, only this file needs significant modification.
"""


from __future__ import annotations
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
        self.model_nickname = model_name # Default to model name
        self._available = None  # Cached status to prevent redundant network calls
        
        # Token Analytics State
        self.last_run_stats = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

    # 1. NEURAL CORE (Embeddings)

    def embed_text(self, text: str) -> list[float]:
        """
        Generates a neural embedding for a single text chunk.
        Used for real-time query vectorization.
        """
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    def get_embedding_dimension(self) -> int:
        """Probes the current model to determine its vector dimension."""
        try:
            # We use a very short string for the probe to minimize latency
            vec = self.embed_text("probe")
            return len(vec) if vec else 0
        except Exception:
            return 0

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generates neural embeddings for a list of texts using sub-batching.
        Processes chunks of 20 at a time to prevent Ollama timeouts on larger models.
        """
        all_embeddings = []
        batch_size = 20
        
        try:
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                # Use the newer .embed() API for optimized processing
                response = ollama.embed(model=self.embedding_model, input=chunk)
                batch_vecs = response.get("embeddings", [])
                all_embeddings.extend(batch_vecs)
            return all_embeddings
        except Exception as e:
            print(f"Batch embedding error: {e}")
            return []


    # 2. SYSTEM HEALTH (Connectivity)

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

    def get_available_models(self) -> list[str]:
        """Returns a list of all model names installed on the local Ollama instance."""
        try:
            models = ollama.list()
            return [m.model for m in models.models]
        except Exception:
            return []

    def get_running_models(self) -> list[str]:
        """Returns a list of models currently loaded in memory."""
        try:
            res = ollama.ps()
            return [m.model for m in res.models]
        except Exception:
            return []

    def get_chat_models(self) -> list[str]:
        """Returns a list of models likely to be conversational/chat-based."""
        all_models = self.get_available_models()
        # Heuristic: exclude known embedding patterns
        embed_patterns = ['embed', 'minilm', 'nomic', 'bge', 'bert']
        return [m for m in all_models if not any(p in m.lower() for p in embed_patterns)]

    def get_embedding_models(self) -> list[str]:
        """Returns a list of models specifically designed for vector embeddings."""
        all_models = self.get_available_models()
        # Heuristic: include known embedding patterns
        embed_patterns = ['embed', 'minilm', 'nomic', 'bge', 'bert']
        return [m for m in all_models if any(p in m.lower() for p in embed_patterns)]

    def set_model(self, model_name: str) -> bool:
        """Updates the active LLM after verifying its existence."""
        available = self.get_available_models()
        if any(model_name in name for name in available):
            self.model_name = model_name
            self.reset_status()
            return True
        return False

    def set_embedding_model(self, model_name: str) -> bool:
        """Updates the active embedding engine."""
        available = self.get_available_models()
        if any(model_name in name for name in available):
            self.embedding_model = model_name
            self.reset_status()
            return True
        return False

    def reset_status(self):
        """Forces a fresh heart-beat check on the next availability request."""
        self._available = None

    # 3. CONTEXTUAL INTELLIGENCE (RAG & Chat)

    def _build_rag_messages(self, query: str, context_text: str,
                             chat_history: list[dict] | None = None,
                             agent_context: str | None = None,
                             file_manifest: list[str] | None = None) -> list[dict]:
        """
        Constructs a prompt that grounds the LLM in the provided document context.
        
        Developer Insight (Prompt Engineering):
        We use a three-tier system prompt:
        1. **Identity**: Role instructions from AGENT.md.
        2. **Grounding**: The specific file manifest and retrieval context.
        3. **Constraints**: Formatting rules to prevent 'context-leaking' or hallucinations.
        """
        system_prompt = ""
        if agent_context:
            system_prompt += f"{agent_context}\n\n"
        else:
            system_prompt += "You are a knowledgeable research assistant. "

        if file_manifest:
            manifest_str = ", ".join(file_manifest)
            system_prompt += f"You have access to a Knowledge Base containing the following files: [{manifest_str}].\n\n"
            
        system_prompt += (
            "Your job is to answer the user's question using the provided document context below.\n"
            "Each document snippet is labeled with a 'Source' filename and a 'Full Path'.\n"
            "If the user asks for the location or full path of a file, ALWAYS provide the 'Full Path' correctly from the context snippets.\n"
            "If the answer isn't in the provided snippets, look at the file manifest above. If a relevant file exists but its content is missing from the snippets, tell the user you know the file exists but it didn't return a strong match for this query.\n\n"
            "--- DOCUMENT CONTEXT ---\n"
            f"{context_text}\n"
            "--- END CONTEXT ---\n\n"
            "Constraints:\n"
            "- If the answer isn't in the context and you can't infer it from the manifest, say 'I don't have enough information'.\n"
            "- Cite specific filenames and page numbers if available.\n"
            "- Provide absolute paths ONLY if specifically asked for the file location or path.\n"
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
                                chat_history: list[dict] | None = None,
                                agent_context: str | None = None,
                                file_manifest: list[str] | None = None):
        """
        Streams a RAG-grounded response from Ollama.
        
        The 'Streaming' Philosophy:
        Instead of waiting for the full response, we yield text chunks immediately.
        The last chunk (marked 'done': True) contains the mathematical metadata 
        (tokens, duration) which we capture for the UI.

        Theory Note: Repetition Penalties
        --------------------------------
        Local models (especially smaller ones) can get stuck in "feedback loops" 
        where they repeat the same sentence. 
        - repeat_penalty (1.2): Higher values make the model less likely to repeat tokens.
        - repeat_last_n (64): The "memory window" the model checks for repetitions.
        """


        messages = self._build_rag_messages(query, context_text, chat_history, agent_context, file_manifest)

        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.3,
                    "repeat_penalty": 1.2,
                    "repeat_last_n": 64
                }
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
