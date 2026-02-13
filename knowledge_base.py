import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class KnowledgeBase:
    def __init__(self, chunk_size=600):
        self.chunk_size = chunk_size
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

        # Tracking Metrics for the UI Dashboard
        self.cleaning_report = []
        self.file_chunk_counts = {}

    def clean_text(self, text):
        """
        ADVANCED DATA PREPARATION:
        We normalize text to reduce the dimensionality of the vector space.
        """
        # 1. Case Folding
        text = text.lower()
        # 2. Noise Removal (Regex for URLs and Non-semantic symbols)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Keep letters, numbers, and sentence punctuation (. ! ?)
        text = re.sub(r'[^a-z0-9\s\.!?]', ' ', text)
        # 3. Whitespace Normalization
        return re.sub(r'\s+', ' ', text).strip()

    def process_text(self, filename, raw_text):
        """Sentence-Aware Chunking (Punctuation Stop-Logic)"""
        cleaned = self.clean_text(raw_text)

        # Log cleaning metrics for the front-end report
        self.cleaning_report.append({
            "File": filename,
            "Raw Chars": len(raw_text),
            "Cleaned Chars": len(cleaned),
            "Noise Reduction": f"{((len(raw_text) - len(cleaned)) / len(raw_text) * 100):.1f}%" if len(
                raw_text) > 0 else "0%"
        })

        # Split into sentences keeping the punctuation (Positive Lookbehind)
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)

        chunks_for_file = 0
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                self.documents.append(current_chunk.strip())
                current_chunk = sentence
                chunks_for_file += 1
            else:
                current_chunk += " " + sentence

        if current_chunk:
            self.documents.append(current_chunk.strip())
            chunks_for_file += 1

        self.file_chunk_counts[filename] = chunks_for_file

    def build_index(self):
        """Builds the high-dimensional TF-IDF matrix."""
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents).toarray()

    def get_top_keywords_df(self, top_n=10):
        """Calculates global feature importance for the bar chart."""
        if self.tfidf_matrix is None: return pd.DataFrame()

        # Mean TF-IDF score across all chunks
        importance = np.mean(self.tfidf_matrix, axis=0)
        words = self.vectorizer.get_feature_names_out()
        df = pd.DataFrame({'Keyword': words, 'Importance': importance})
        return df.sort_values(by='Importance', ascending=False).head(top_n)

    def search(self, query_text, top_n=3):
        """Manual Cosine Similarity Retrieval logic."""
        if self.tfidf_matrix is None or not self.documents: return []

        query_cleaned = self.clean_text(query_text)
        query_vec = self.vectorizer.transform([query_cleaned]).toarray()[0]
        q_norm = np.linalg.norm(query_vec)

        if q_norm == 0: return []

        scores = []
        for i, doc_vec in enumerate(self.tfidf_matrix):
            d_norm = np.linalg.norm(doc_vec)
            if d_norm == 0: continue

            # Manual Math: Dot Product / (Norm A * Norm B)
            score = np.dot(query_vec, doc_vec) / (q_norm * d_norm)
            scores.append((score, self.documents[i]))

        return sorted(scores, key=lambda x: x[0], reverse=True)[:top_n]