import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer 
import unicodedata

# Ensure NLTK resources are available for Lemmatization
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class KnowledgeBase:
    """
    The Semantic Core of the application. 
    This class handles document ingestion, NLP preprocessing, and vector retrieval.
    """
    def __init__(self, chunk_size=600):
        self.chunk_size = chunk_size
        self.documents_metadata = []
        
        # TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer:
        # This converts text into a numerical matrix where each value represents
        # how important a word is to a specific document vs. the whole collection.
        # [TEXT CLASSIFICATION / TOPIC MODELLING FOUNDATION]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
        # Lemmatizer: Reduces words to their dictionary root form (e.g., "running" -> "run")
        self.lemmatizer = WordNetLemmatizer()

        # Storage for full source content
        self.file_contents = {}

        # Analytics for UI reporting
        self.cleaning_report = []
        self.file_chunk_counts = {}

    def clean_text(self, text):
        """
        NLP Preprocessing Pipeline.
        Prepares raw text for mathematical vectorization.
        """
        # 1. Case Normalization: Essential for matching "Apple" and "apple"
        text = text.lower()
        
        # 2. Unicode Normalization: Standardizes characters (e.g., removing accents)
        text = unicodedata.normalize('NFKD', text)
        
        # 3. Noise Removal: Strips URLs and non-alphanumeric junk
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^a-z0-9\s\.!?]', ' ', text)
        
        # 4. Lemmatization [TEXT CLASSIFICATION PREP]:
        # We find the root-v form of every word to ensure semantic consistency.
        # We also pad punctuation so symbols like '.' don't get 'stuck' to words.
        text = re.sub(r'([.!?])', r' \1 ', text)
        words = text.split()
        text = " ".join([self.lemmatizer.lemmatize(word, pos='v') for word in words])
        
        # 5. Whitespace Normalization: Ensures single spaces for clean tokenization
        return re.sub(r'\s+', ' ', text).strip()

    def process_text(self, filename, raw_text, page_num=1, full_path=None):
        """
        Document Ingestion & Sentence-Aware Chunking.
        Breaks large files into 'Semantic Units' for better search precision.
        """
        if filename not in self.file_contents:
            self.file_contents[filename] = ""
        self.file_contents[filename] += raw_text

        cleaned = self.clean_text(raw_text)

        # Analytics Tracking: Measure how much "noise" was removed
        raw_len = len(raw_text)
        clean_len = len(cleaned)
        if not any(d['File'] == filename for d in self.cleaning_report):
            self.cleaning_report.append({
                "File": filename,
                "Original Chars": raw_len,
                "Cleaned Chars": clean_len,
                "Noise Reduction": f"{((raw_len - clean_len) / raw_len * 100):.1f}%" if raw_len > 0 else "0%"
            })

        # Sentence-Aware Chunking [TEXT SUMMARIZATION FOUNDATION]:
        # We split by sentences to avoid cutting a thought in half.
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)

        chunks_for_file = 0
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                self.documents_metadata.append({
                    "text": current_chunk.strip(),
                    "file": filename,
                    "full_path": full_path,
                    "page": page_num
                })
                current_chunk = sentence
                chunks_for_file += 1
            else:
                current_chunk += " " + sentence

        if current_chunk:
            self.documents_metadata.append({
                "text": current_chunk.strip(),
                "file": filename,
                "full_path": full_path,
                "page": page_num
            })
            chunks_for_file += 1

        self.file_chunk_counts[filename] = self.file_chunk_counts.get(filename, 0) + chunks_for_file

    def build_index(self):
        """
        Vectorization Phase.
        Converts the database of text chunks into a TF-IDF Matrix for calculation.
        """
        if self.documents_metadata:
            texts = [doc['text'] for doc in self.documents_metadata]
            # fit_transform learns the vocabulary and returns the matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()

    def get_top_keywords_df(self, top_n=10):
        """
        Feature Importance Analysis [TOPIC MODELLING].
        Identifies the words with the highest average TF-IDF weight across the corpus.
        """
        if self.tfidf_matrix is None: return pd.DataFrame()

        # Mean of TF-IDF scores across all chunks gives us global keyword importance
        importance = np.mean(self.tfidf_matrix, axis=0)
        words = self.vectorizer.get_feature_names_out()

        df = pd.DataFrame({'Keyword': words, 'Importance': importance})
        return df.sort_values(by='Importance', ascending=False).head(top_n)

    def search(self, query_text, top_n=30):
        """
        Semantic Retrieval Engine [QUESTION-ANSWERING FOUNDATION].
        Calculates manual Cosine Similarity between user query and all document chunks.
        """
        if self.tfidf_matrix is None or not self.documents_metadata: return []

        # 1. Process Query: Must use the same pipeline as the docs
        query_cleaned = self.clean_text(query_text)
        query_vec = self.vectorizer.transform([query_cleaned]).toarray()[0]

        # 2. Vector Magnitude (Norm): Needed for Cosine denominator
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0: return []

        results = []
        # 3. Linear Algebra Matchmaking: 
        # Score = (A . B) / (||A|| * ||B||)
        for i, doc_vec in enumerate(self.tfidf_matrix):
            d_norm = np.linalg.norm(doc_vec)
            if d_norm == 0: continue

            # Dot Product calculation
            score = np.dot(query_vec, doc_vec) / (q_norm * d_norm)

            if score > 0.05: # Relevance Threshold
                meta = self.documents_metadata[i].copy()
                meta['score'] = round(float(score), 4)
                results.append(meta)

        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]
