"""
KnowledgeBase Engine — The Semantic Core
========================================

Architecture Rationale:
-----------------------
The KnowledgeBase is the mathematical 'brain' of the system. It is designed to be 
'Hybrid'—supporting both classical Statistical NLP (TF-IDF) and modern 
Neural NLP (Dense Embeddings).

Structure Overview:
1.  **State Management**: Tracks document metadata, full-text storage, and cache.
2.  **The Preprocessing Pipeline**: Standardizes raw input (Noise removal, Lemmatization).
3.  **The Ingestion Engine**: Handles varied file types (PDF, MD, CSV) and 
    chunks them into searchable units.
4.  **The Vectorization Core**: Converts text into numerical matrices.
5.  **The Retrieval Logic**: Implements High-Performance Cosine Similarity 
    using Linear Algebra (NumPy).

Why this structure?
This decoupled design allows the UI (app.py) to remain thin and focused on 
presentation, while this class handles the heavy computational work.
"""

import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap
import warnings

# Suppress UMAP parallelism warnings caused by random_state seeding
warnings.filterwarnings("ignore", message="n_jobs value 1")
import nltk
from nltk.stem import WordNetLemmatizer 
import unicodedata
import json
import os
import hashlib

# --- NLTK Resource Management ---
# WordNet is used for Lemmatization (finding the root of a word).
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class KnowledgeBase:
    """
    Orchestrates the lifecycle of knowledge from raw file to searchable vector.
    """

    def __init__(self, chunk_size=600, overlap_size=100, engine_mode="Machine Learning"):
        """
        Initializes the semantic core with configurable windowing parameters.
        
        Args:
            chunk_size (int): Max characters per searchable segment.
            overlap_size (int): Context preservation between chunks.
            engine_mode (str): 'Machine Learning' (Keyword) or 'Deep Learning' (Neural).
        """
        # Configuration
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.engine_mode = engine_mode
        self.dataset_overlap = 15 
        self.ml_top_n = 5  # User-configurable limit for ML retrieval depth
        self.stop_requested = False
        self.spatial_granularity = "Segments" # 'Documents' or 'Segments'
        
        # Primary Storage
        self.documents_metadata = [] # List of dicts: {text, file, page...}
        self.documents_spatial = []  # Aggregated file-level spatial metadata
        self.documents_matrix_agg = None # Cached document centroids
        self.file_contents = {}      # Full source text storage
        
        # Analytics & UI Reporting
        self.cleaning_report = []
        self.file_chunk_counts = {}
        
        # ML Engine State (Classical Statistics)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None     # Sparse matrix after fit()
        self.lemmatizer = WordNetLemmatizer()

        # DL Engine State (Neural Embeddings)
        self.embeddings = None       # Dense NumPy array
        self.cache_file = "data/.neural_cache.json"
        self._embed_cache = self._load_disk_cache()

    # ------------------------------------------------------------------
    # PHASE 1: DISCOVERY & CACHING
    # ------------------------------------------------------------------

    def _get_text_hash(self, text):
        """Generates a fingerprint for a text chunk to prevent redundant embedding calls."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _load_disk_cache(self):
        """Loads cached neural vectors from previous sessions."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception: pass
        return {}

    def _save_disk_cache(self):
        """Persists the embedding cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._embed_cache, f)
        except Exception: pass

    # ------------------------------------------------------------------
    # PHASE 2: THE PREPROCESSING PIPELINE
    # ------------------------------------------------------------------

    def clean_text(self, text):
        """
        Transforms messy raw text into a standardized format for vectorization.
        
        Pipeline:
        1. Normalization (NFKD) to handle special unicode characters.
        2. Lowercasing for keyword consistency.
        3. Regex-based noise removal (URLs, symbols).
        4. (ML Only) Lemmatization: finding word roots (e.g., 'running' -> 'run').
        """
        if not isinstance(text, str): text = str(text)
        text = text.lower()
        
        if self.engine_mode == "Machine Learning":
            # 1. Unicode Normalization
            text = unicodedata.normalize('NFKD', text)
            # 2. Noise Removal (Regex)
            text = re.sub(r'https?://\S+|www\.\S+', '', text) # Strip Links
            text = re.sub(r'[^a-z0-9\s\.!?]', ' ', text)      # Keep only Alphanum and Punctuation
            text = re.sub(r'([.!?])', r' \1 ', text)          # Pad punctuation
            
            # 3. Token-level Lemmatization
            words = text.split()
            text = " ".join([self.lemmatizer.lemmatize(word, pos='v') for word in words])
        else: 
            # Neural Mode: Preserves more structure for LLM understanding
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'[^a-z0-9\s\.!?]', ' ', text)
        
        return re.sub(r'\s+', ' ', text).strip()

    # ------------------------------------------------------------------
    # PHASE 3: INGESTION & CHUNKING
    # ------------------------------------------------------------------

    def process_text(self, filename, raw_text, page_num=1, full_path=None):
        """
        Cleans and segments raw text into searchable chunks.
        Implementing 'Sentence-Aware' chunking for ML and 'Sliding-Window' for Neural.
        """
        if filename not in self.file_contents: self.file_contents[filename] = ""
        self.file_contents[filename] += raw_text

        cleaned = self.clean_text(raw_text)
        
        # Update Analytics
        self._record_cleaning_stats(filename, raw_text, cleaned)

        if self.engine_mode == "Machine Learning":
            self._split_sentences_aware(cleaned, filename, page_num, full_path)
        else:
            self._split_sliding_window(cleaned, filename, page_num, full_path)

    def _record_cleaning_stats(self, filename, raw, clean):
        """Internal helper to track NLP pipeline effectiveness."""
        r_len, c_len = len(raw), len(clean)
        if not any(d['File'] == filename for d in self.cleaning_report):
            self.cleaning_report.append({
                "File": filename, "Orig": r_len, "Clean": c_len,
                "Reduction": f"{((r_len - c_len) / r_len * 100):.1f}%" if r_len > 0 else "0%"
            })

    def _split_sentences_aware(self, text, filename, page, path):
        """Splits text by sentences to ensure grammatical units remain intact."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        cur_chunk = ""
        for s in sentences:
            if len(cur_chunk) + len(s) > self.chunk_size and cur_chunk:
                self.documents_metadata.append({"text": cur_chunk.strip(), "file": filename, "full_path": path, "page": page})
                cur_chunk = s
            else: cur_chunk += " " + s
        if cur_chunk:
            self.documents_metadata.append({"text": cur_chunk.strip(), "file": filename, "full_path": path, "page": page})

    def _split_sliding_window(self, text, filename, page, path):
        """Fixed-size window moving through text with configurable overlap."""
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            self.documents_metadata.append({"text": text[start:end], "file": filename, "full_path": path, "page": page})
            start += (self.chunk_size - self.overlap_size)
            if end >= len(text): break

    def process_dataset(self, filename, df):
        """Converts tabular data into block-based text for the search engine."""
        rows = len(df)
        self.file_contents[filename] = df.head(50).to_string() # Sample for summarizer
        
        # Chunk the dataset into manageable blocks
        block_size = 100
        for start in range(0, rows, block_size - self.dataset_overlap):
            end = min(start + block_size, rows)
            txt = df.iloc[start:end].to_string(index=False)
            self.documents_metadata.append({"text": txt, "file": filename, "page": f"Rows {start}-{end}"})

    # ------------------------------------------------------------------
    # PHASE 4: VECTORIZATION CORE
    # ------------------------------------------------------------------

    def build_index(self, llm_service=None):
        """
        Executes the heavy math to build the search index.
        Calculates either TF-IDF vectors or neural dense embeddings.
        Automatically triggers 3D spatial generation for the UI.
        """
        if not self.documents_metadata: return
        texts = [doc['text'] for doc in self.documents_metadata]

        if self.engine_mode == "Machine Learning":
            # Classical fit: Calculates word weights across the whole collection
            self.tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        else:
            if not llm_service: return
            self._build_neural_embeddings(texts, llm_service)
        
        # Trigger 3D Spatial Processing
        self._generate_3d_spatial_data()

    def _build_neural_embeddings(self, texts, llm):
        """Internal logic for batch embedding with disk-cache lookup."""
        embeddings = [None] * len(texts)
        to_query, to_query_meta = [], []
        
        for i, text in enumerate(texts):
            if self.stop_requested: break
            h = self._get_text_hash(text)
            if h in self._embed_cache: embeddings[i] = self._embed_cache[h]
            else:
                to_query.append(text)
                to_query_meta.append((i, h))
        
        if to_query and not self.stop_requested:
            # Batch process remaining chunks via local Ollama
            new_vecs = llm.embed_batch(to_query)
            for j, vec in enumerate(new_vecs):
                idx, h = to_query_meta[j]
                embeddings[idx] = vec
                self._embed_cache[h] = vec
            self._save_disk_cache()
            
        self.embeddings = np.array([e for e in embeddings if e is not None])

    def _generate_3d_spatial_data(self):
        """
        Orchestrates dimensional reduction and clustering based on the current granularity.
        """
        # Determine source matrix and labels
        if self.spatial_granularity == "Segments":
            matrix = self.tfidf_matrix if self.engine_mode == "Machine Learning" else self.embeddings
            source_meta = self.documents_metadata
        else:
            # Document-level aggregation (Centroids)
            matrix, source_meta = self._get_aggregated_document_matrix()
            self.documents_matrix_agg = matrix # Cache for localized drill-downs
        
        if matrix is None or len(matrix) < 2: return

        # 1. Dimensionality Reduction (UMAP 3D)
        if len(matrix) >= 2:
            n_neighbors = min(15, len(matrix) - 1)
            reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.1, random_state=42, n_jobs=1)
            coords = reducer.fit_transform(matrix)

            # 2. Automated Clustering (KMeans)
            n_clusters = min(5, len(matrix))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(matrix)
        else:
            # Fallback for single document/segment to prevent blank maps
            coords = np.array([[0.0, 0.0, 0.0]])
            clusters = np.array([0])

        # 3. Metadata Injection
        if self.spatial_granularity == "Documents": 
            self.documents_spatial = []
        
        for i, (x, y, z) in enumerate(coords):
            if i < len(source_meta):
                meta = source_meta[i]
                meta['x'] = round(float(x), 4)
                meta['y'] = round(float(y), 4)
                meta['z'] = round(float(z), 4)
                c_id = int(clusters[i])
                meta['cluster'] = c_id
                
                if self.spatial_granularity == "Documents":
                    self.documents_spatial.append(meta)
                    # PROPAGATION: Assign this cluster ID to all segments belonging to this file
                    fname = meta['file']
                    for m in self.documents_metadata:
                        if m['file'] == fname:
                            m['cluster'] = c_id
                else:
                    # Segments mode already modifies documents_metadata directly
                    pass

    def _get_aggregated_document_matrix(self):
        """Calculates centroid vectors for each unique file."""
        df = pd.DataFrame(self.documents_metadata)
        if df.empty: return None, []

        matrix = self.tfidf_matrix if self.engine_mode == "Machine Learning" else self.embeddings
        unique_files = df['file'].unique()
        
        agg_matrix = []
        agg_meta = []
        
        for fname in unique_files:
            indices = df[df['file'] == fname].index.tolist()
            file_vecs = matrix[indices]
            centroid = np.mean(file_vecs, axis=0)
            agg_matrix.append(centroid)
            agg_meta.append({
                "file": fname, 
                "text": f"Document Summary: {fname}", 
                "segments": len(indices),
                "page": "1-" + str(df[df['file'] == fname]['page'].max())
            })
            
        return np.array(agg_matrix), agg_meta

    def get_cluster_spatial_data(self, cluster_id):
        """
        Specialized localized UMAP for 'Drill-Down' exploration.
        Filters data by cluster and re-runs dimensionality reduction on the subset.
        """
        # Determine source data based on granularity
        is_doc_mode = getattr(self, 'spatial_granularity', "Segments") == "Documents"
        
        if is_doc_mode:
            source_list = self.documents_spatial
            full_matrix = self.documents_matrix_agg
        else:
            source_list = self.documents_metadata
            full_matrix = self.tfidf_matrix if self.engine_mode == "Machine Learning" else self.embeddings
            
        if full_matrix is None: return pd.DataFrame()

        # Filter by cluster mapping
        indices = [i for i, m in enumerate(source_list) if m.get('cluster') == cluster_id]
        if not indices: return pd.DataFrame()

        subset_matrix = full_matrix[indices]

        if len(subset_matrix) < 2:
            return pd.DataFrame([source_list[i] for i in indices])

        # Localized UMAP
        n_neighbors = min(15, len(subset_matrix) - 1)
        reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.1, random_state=42, n_jobs=1)
        local_coords = reducer.fit_transform(subset_matrix)

        # Build local dataframe for Plotly
        local_data = []
        for i, idx in enumerate(indices):
            meta = source_list[idx].copy()
            meta['x'], meta['y'], meta['z'] = local_coords[i]
            local_data.append(meta)
            
        return pd.DataFrame(local_data)
        
    def get_cluster_stats(self, cluster_id):
        """Calculates volume statistics for a specific galaxy."""
        is_doc_mode = getattr(self, 'spatial_granularity', "Segments") == "Documents"
        source_list = self.documents_spatial if is_doc_mode else self.documents_metadata
        
        subset = [m for m in source_list if m.get('cluster') == cluster_id]
        if not subset: return {"docs": 0, "segments": 0, "topics": []}
        
        unique_docs = len(set(m.get('file') for m in subset))
        total_segments = sum(m.get('segments', 1) for m in subset) if is_doc_mode else len(subset)
        
        return {
            "docs": unique_docs,
            "segments": total_segments,
            "topics": self.get_cluster_topics(cluster_id)
        }

    def get_cluster_topics(self, cluster_id, top_n=3):
        """Extracts dominant keywords for a cluster to provide semantic naming."""
        # Always use segments for topic modeling to get fine-grained keywords
        indices = [i for i, m in enumerate(self.documents_metadata) if m.get('cluster') == cluster_id]
        if not indices or self.tfidf_matrix is None:
            return [f"Galaxy {cluster_id}"]

        # Sum TF-IDF scores across the cluster to find top features
        cluster_tfidf = self.tfidf_matrix[indices].sum(axis=0)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Sort by weight
        sorted_indices = np.argsort(np.asarray(cluster_tfidf).flatten())[::-1]
        
        # Filter for quality (avoid common short words if possible)
        topics = []
        for idx in sorted_indices:
            word = feature_names[idx]
            if len(word) > 4: # Focus on more substantive terms
                topics.append(word.capitalize())
            if len(topics) >= top_n: break
            
        return topics if topics else [f"Cluster {cluster_id}"]

    def get_universe_stats(self):
        """Calculates global inventory and semantic map for the entire universe."""
        if not self.documents_metadata:
            return {"docs": 0, "segments": 0, "galaxy_map": {}}
            
        unique_docs = len(self.file_contents)
        total_segments = len(self.documents_metadata)
        
        # Build a map of cluster IDs to their semantic topics
        all_clusters = sorted(list(set(m.get('cluster', 0) for m in self.documents_metadata)))
        galaxy_map = {}
        for c in all_clusters:
            galaxy_map[c] = self.get_cluster_topics(c, top_n=2)
            
        return {
            "docs": unique_docs,
            "segments": total_segments,
            "galaxy_map": galaxy_map
        }

    # ------------------------------------------------------------------
    # PHASE 5: THE RETRIEVAL ENGINE
    # ------------------------------------------------------------------

    def search(self, query_text, llm_service=None, top_n=None):
        """
        Orchestrates the search request.
        
        Math: Cosine Similarity
        Score = (A . B) / (||A|| * ||B||)
        This measures the 'angle' between the query and documents in vector space.
        """
        if not self.documents_metadata: return []
        # Support for stale session objects that might lack this attribute
        default_limit = getattr(self, 'ml_top_n', 5)
        limit = top_n if top_n else default_limit

        if self.engine_mode == "Machine Learning":
            return self._search_tfidf(query_text, limit)
        else:
            return self._search_neural(query_text, llm_service, limit)

    def _search_tfidf(self, query, top_n):
        """Keyword matching via TF-IDF dot-products."""
        if self.tfidf_matrix is None: return []
        
        q_vec = self.vectorizer.transform([self.clean_text(query)]).toarray()[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0: return []
        
        results = []
        for i, doc_vec in enumerate(self.tfidf_matrix):
            d_norm = np.linalg.norm(doc_vec)
            if d_norm == 0: continue
            score = np.dot(q_vec, doc_vec) / (q_norm * d_norm)
            if score > 0.05:
                meta = self.documents_metadata[i].copy()
                meta['score'] = round(float(score), 4)
                results.append(meta)
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]

    def _search_neural(self, query, llm, top_n):
        """Contextual matching via dense vector similarity."""
        if self.embeddings is None or not llm: return []
        q_vec = np.array(llm.embed_text(query))
        if q_vec.size == 0: return []
        
        # Parallel Cosine Similarity using NumPy
        q_norm = np.linalg.norm(q_vec)
        d_norms = np.linalg.norm(self.embeddings, axis=1)
        sims = np.dot(self.embeddings, q_vec) / (q_norm * d_norms + 1e-9)

        results = []
        for i, score in enumerate(sims):
            if score > 0.35: # Slightly higher threshold for neural
                meta = self.documents_metadata[i].copy()
                meta['score'] = round(float(score), 4)
                results.append(meta)
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]

    def get_context_for_query(self, query_text, llm, top_n=5):
        """Formats the top N results as a structured text block for the LLM."""
        res = self.search(query_text, llm, top_n=top_n)
        return "\n".join([f"[Source: {r['file']} P{r['page']}]\n{r['text']}\n" for r in res])

    def get_document_text(self, filename):
        """Retrieves raw content for summarization."""
        return self.file_contents.get(filename, "")

    def get_top_keywords_df(self, top_n=10):
        """Analytics: Identifies the most statistically important terms in the index."""
        if self.engine_mode == "Machine Learning" and self.tfidf_matrix is not None:
            importance = np.mean(self.tfidf_matrix, axis=0)
            words = self.vectorizer.get_feature_names_out()
            return pd.DataFrame({'Keyword': words, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)
        return pd.DataFrame() # Analytics skipped for Neural to save performance
