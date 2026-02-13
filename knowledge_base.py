# knowledge_base.py
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class KnowledgeBase:
    def __init__(self):
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

    def process_text(self, text):
        """Cleans text and splits it into overlapping chunks."""
        # Clean: remove markdown headers and extra newlines
        text = re.sub(r'#+\s?', '', text)
        text = re.sub(r'\n+', ' ', text).strip()

        # Chunking: sliding window approach
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            self.documents.append(text[start:end])
            start += (self.chunk_size - self.chunk_overlap)

    def build_index(self):
        """Creates the TF-IDF matrix from the document chunks."""
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents).toarray()

    def search(self, query_text, top_n=3):
        """Optimized manual Cosine Similarity search."""
        if self.tfidf_matrix is None or len(self.documents) == 0:
            return []

        # 1. Vectorize the user's query
        query_vec = self.vectorizer.transform([query_text]).toarray()[0]

        # Calculate the query's magnitude (norm) ONCE outside the loop
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []  # The query contained no recognized words

        scores = []
        # 2. Compare query against every document chunk
        for i, doc_vec in enumerate(self.tfidf_matrix):
            doc_norm = np.linalg.norm(doc_vec)
            if doc_norm == 0:
                continue

            # Manual Cosine Similarity: Dot Product / (Magnitude A * Magnitude B)
            dot_product = np.dot(query_vec, doc_vec)
            similarity = dot_product / (query_norm * doc_norm)

            scores.append((similarity, self.documents[i]))

        # 3. Sort by highest similarity score
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_n]