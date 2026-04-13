# Machine Learning (Statistical NLP) Implementation
===================================================

The Machine Learning engine in this solution is built on **Term Frequency-Inverse Document Frequency (TF-IDF)**—a foundational technique in Information Retrieval that identifies the importance of words relative to a corpus.

## 🛠️ The Pipeline Structure

The ML retrieval process follows a strict 4-stage pipeline defined in `core/knowledge_base.py`:

1.  **Normalization**: Raw text is normalized using Unicode NFKD to standardize characters (e.g., removing accents and standardizing whitespace).
2.  **Lemmatization (NLTK)**: Utilizing the `WordNetLemmatizer`, we reduce words to their dictionary root (e.g., "calculating" and "calculation" both become "calculate"). This significantly increases match accuracy by grouping semantic variants.
3.  **Vectorization (TfidfVectorizer)**: The corpus is transformed into a **Sparse Matrix** where each row is a document segment and each column is a unique word (feature). The values indicate the statistical "importance" of each word in that segment.
4.  **Retrieval (Cosine Similarity)**: We use **Linear Algebra via NumPy** to calculate the distance between the query vector and all document vectors. 

## 📓 Learnings & Techniques

### Why TF-IDF?
- **Speed**: It is incredibly fast and requires no GPU or external server.
- **Explainability**: Unlike Neural networks, we can exactly see *which* words caused a match (available in the "Vector Analytics" tab).
- **Zero-Shot**: It works instantly on any language or technical domain without training.

### Techniques for Accuracy
- **Sentence-Aware Chunking**: We split documents by sentences rather than characters. This prevents "cutting a thought in half," ensuring each vector represents a complete semantic unit.
- **Lemmatization POS-Tagging**: By lemmatizing with `pos='v'`, we prioritize verb-root matching, which is essential for instructional/technical documentation retrieval.

### Performance Constraints
- **Keyword Blindness**: TF-IDF cannot understand synonyms. If you search for "automobile" but the document uses "car," it will not match. (See `Deep Learning` guide for the solution to this).
- **Sparse Bloat**: As the document count grows to millions, the sparse matrix can become memory-intensive. For our "Workspace" scale, it remains optimal.
