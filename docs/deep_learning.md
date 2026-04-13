# Deep Learning & Prompt Engineering References

This document serves as a repository of research and best practices utilized in the development of the NLP Engine's identity and interaction logic.

## 1. Structured Agent Descriptors (AGENT.md)
The concept of using a dedicated `AGENT.md` file in the root of a repository is an emerging standard for providing long-term memory and operational boundaries to AI agents.

### Key Resources:
- **The "Onboarding Manual" Pattern**: Treating the system prompt as an employee's starting guide rather than a list of constraints. [Ref: Builder.io]
- **Hierarchical Context**: How agents prioritize instructions located closer to the target file or subdirectory.

## 2. Prompt Engineering Best Practices
To ensure the LLM remains performant and "grounded," the following techniques are applied:

### Few-Shot Rooting
Instead of abstract instructions like "be polite," the engine uses concrete examples of "Thought -> Action -> Response" loops to align with the project's technical goals.

### System-Level Constraints vs. RAG Knowledge
- **Static Identity**: Hard-wired into the `system` prompt via the `AGENT.md` logic.
- **Dynamic Knowledge**: Retrieved at runtime from the `vault/` directory via semantic search.

### The "Knowledge Chain" Mechanism
To maintain performance and privacy, the system implements a tiered data flow:
1.  **Ingestion Barrier**: Files are read once, vectorized, and stored in a persistent local index. The LLM does not have direct disk access.
2.  **Retrieval Step**: Only the most relevant text segments are retrieved based on the user's query intent.
3.  **Prompt Rooting**: Retrieved segments are injected into the prompt context, "rooting" the LLM's response in ground-truth data.

## 3. Recommended Reading
- *Pre-train, Prompt, and Predict*: A modern overview of how persona-based prompting changes model behavior.
- *Chain of Thought Prompting Elicits Reasoning in Large Language Models*: The foundational paper on why "Let's think step by step" works.

---
*Created dynamically by the NLP Engine // 2026-04-10*
