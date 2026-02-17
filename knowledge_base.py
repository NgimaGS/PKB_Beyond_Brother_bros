import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class KnowledgeBase:
    def __init__(self, chunk_size=600):
        self.chunk_size = chunk_size
        self.documents_metadata = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

        # Store full content per file for retrieval context
        self.file_contents = {}

        # Analytics Tracking
        self.cleaning_report = []
        self.file_chunk_counts = {}

    def clean_text(self, text):
        """Advanced NLP Normalization & Noise Reduction."""
        # 1. Case Folding
        text = text.lower()
        # 2. Strip URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # 3. Strip structural symbols but keep . ! ? for sentence splitting
        text = re.sub(r'[^a-z0-9\s\.!?]', ' ', text)
        # 4. Normalize Whitespace
        return re.sub(r'\s+', ' ', text).strip()

    def process_text(self, filename, raw_text, page_num=1):
        """Processes text, logs metrics, and attaches page metadata."""
        # Store raw content for viewing/reference
        if filename not in self.file_contents:
            self.file_contents[filename] = ""
        self.file_contents[filename] += raw_text

        cleaned = self.clean_text(raw_text)

        # Log cleaning metrics
        raw_len = len(raw_text)
        clean_len = len(cleaned)

        # Avoid duplicate logs for the same file
        if not any(d['File'] == filename for d in self.cleaning_report):
            self.cleaning_report.append({
                "File": filename,
                "Original Chars": raw_len,
                "Cleaned Chars": clean_len,
                "Noise Reduction": f"{((raw_len - clean_len) / raw_len * 100):.1f}%" if raw_len > 0 else "0%"
            })

        # Sentence-Aware Chunking (Punctuation Stop-Logic)
        # Splits text while keeping the punctuation mark attached
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)

        chunks_for_file = 0
        current_chunk = ""

        for sentence in sentences:
            # If adding the next sentence exceeds chunk size, save current and start new
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                self.documents_metadata.append({
                    "text": current_chunk.strip(),
                    "file": filename,
                    "page": page_num
                })
                current_chunk = sentence
                chunks_for_file += 1
            else:
                current_chunk += " " + sentence

        # Append any remaining text
        if current_chunk:
            self.documents_metadata.append({
                "text": current_chunk.strip(),
                "file": filename,
                "page": page_num
            })
            chunks_for_file += 1

        self.file_chunk_counts[filename] = self.file_chunk_counts.get(filename, 0) + chunks_for_file

    def build_index(self):
        """Builds the TF-IDF matrix."""
        if self.documents_metadata:
            texts = [doc['text'] for doc in self.documents_metadata]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()

    def get_top_keywords_df(self, top_n=10):
        """Calculates feature importance for UI graphs."""
        if self.tfidf_matrix is None: return pd.DataFrame()

        # Calculate mean TF-IDF score for each word across all documents
        importance = np.mean(self.tfidf_matrix, axis=0)
        words = self.vectorizer.get_feature_names_out()

        df = pd.DataFrame({'Keyword': words, 'Importance': importance})
        return df.sort_values(by='Importance', ascending=False).head(top_n)

    def search(self, query_text, top_n=5):
        """Manual Cosine Similarity with metadata output."""
        if self.tfidf_matrix is None or not self.documents_metadata: return []

        query_cleaned = self.clean_text(query_text)
        query_vec = self.vectorizer.transform([query_cleaned]).toarray()[0]

        # Calculate Magnitude (Norm) of query vector
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0: return []

        results = []
        for i, doc_vec in enumerate(self.tfidf_matrix):
            d_norm = np.linalg.norm(doc_vec)
            if d_norm == 0: continue

            # Linear Algebra: Dot Product / (Norm A * Norm B)
            score = np.dot(query_vec, doc_vec) / (q_norm * d_norm)

            # Only return relevant results
            if score > 0.05:
                meta = self.documents_metadata[i].copy()
                meta['score'] = round(float(score), 4)  # Real Value correlation
                results.append(meta)

        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]