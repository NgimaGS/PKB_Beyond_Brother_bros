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
presentation, while this class handles the heavy computational work. It follows
the 'Pedagogy Through Code' (PTC) philosophy, where reading the source acts
as a tutorial for building RAG systems.
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
import pickle

# --- NLTK Resource Management ---
# WordNet is used for Lemmatization (finding the root of a word).
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class KnowledgeBase:
    """
    Orchestrates the lifecycle of knowledge from raw file to searchable vector.
    """

    def __init__(self, chunk_size=600, overlap_size=100, engine_mode="Deep Learning"):
        """
        Initializes the semantic core with configurable windowing parameters.
        
        Developer Note (The "Hybrid" Philosophy):
        We maintain two completely separate state registries for ML and DL. 
        This allows the user to switch engines in the UI without losing their
        indexed data in the other mode.

        Args:
            chunk_size (int): Max characters per searchable segment.
            overlap_size (int): Context preservation between chunks.
            engine_mode (str): 'Machine Learning' (Keyword) or 'Deep Learning' (Neural).
        """
        # --- CONFIGURATION & TUNING ---
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.engine_mode = engine_mode
        self.dataset_overlap = 15 
        self.ml_top_n = 5  # Depth of keyword retrieval
        self.stop_requested = False # Flag for safe indexing termination
        self.spatial_granularity = "Segments" # Controls 3D map detail: 'Documents' or 'Segments'
        
        # --- PRIMARY DATA REGISTRY ---
        # metadata stores the 'context' (text, file name, page numbers)
        self.documents_metadata = [] 
        # spatial stores 'coordinates' (x, y, z) for the 3D visualizer
        self.documents_spatial = []  
        # matrix_agg stores file-level centroids for "Document Mode" visualization
        self.documents_matrix_agg = None 
        # file_contents stores raw full-text for secondary processing
        self.file_contents = {}      
        
        # --- OPS & REPORTING ---
        self.cleaning_report = []
        self.file_chunk_counts = {}
        self.indexing_errors = [] # JSON-serializable list of UI error cards
        self.index_embedding_model = None # Safety check to ensure model/vector alignment
        
        # --- ML ENGINE (Statistical / TF-IDF) ---
        # The vectorizer transforms text into a sparse frequency matrix.
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None     
        self.lemmatizer = WordNetLemmatizer()

        # --- DL ENGINE (Neural / Embeddings) ---
        # Dense vectors generated via Ollama.
        self.embeddings = None       
        self._embed_cache = {}       # Local JSON-backed cache to avoid re-embedding
        self._active_cache_path = None
        self.neural_threshold = 0.35 # Mathematical cutoff for 'relevance'




    # ------------------------------------------------------------------
    # PHASE 1: DISCOVERY & CACHING
    # ------------------------------------------------------------------

    def _get_text_hash(self, text):
        """Generates a fingerprint for a text chunk to prevent redundant embedding calls."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _get_cache_path(self, model_name):
        """Generates a safe filename for the model-specific neural cache."""
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name)
        return os.path.join("data", f".cache_{safe_name}.json")


    def _load_disk_cache(self, model_name):
        """Loads cached neural vectors for the specific model being used."""
        path = self._get_cache_path(model_name)
        self._active_cache_path = path
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception: pass
        return {}

    def _save_disk_cache(self):
        """Persists the current model-specific cache to disk."""
        if not self._active_cache_path: return
        try:
            with open(self._active_cache_path, 'w') as f:
                json.dump(self._embed_cache, f)
        except Exception: pass


    def clear_previous_index(self):
        """Purges old TF-IDF and Neural indices to free memory during a new build."""
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.embeddings = None
        self.documents_spatial = []
        # We preserve file_contents if we are doing a progressive update,
        # but for a clean rebuild from app.py, it will be reset.
        self.file_contents = {}
        self.documents_metadata = []

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

    def process_image_asset(self, filename, description, image_path):
        """
        Registers a generated image as a semantic unit in the knowledge base.
        This allows images to be searched and visualized in the 3D map.
        """
        # We store the Llava-generated description as the 'text' for vectorization
        self.documents_metadata.append({
            "text": f"[IMAGE ASSET: {filename}]\n{description}",
            "file": filename,
            "full_path": image_path,
            "page": "Image Asset",
            "is_image": True # Flag for specialized rendering
        })
        
        # Also store in file_contents for summarization safety
        if filename not in self.file_contents:
            self.file_contents[filename] = description

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
        
        # Track the model used for this build
        if self.engine_mode == "Deep Learning" and llm_service:
            model_name = llm_service.embedding_model
            self.index_embedding_model = model_name
            # Load model-specific vector cache to prevent dimension pollution
            self._embed_cache = self._load_disk_cache(model_name)
            
            # Determine and lock the mathematical dimension of the current model
            vec = llm_service.embed_text("dim_probe")
            self.index_embedding_dimension = len(vec) if vec is not None else 0


        else:
            self.index_embedding_model = "Classical ML (TF-IDF)"
            self.index_embedding_dimension = self.vectorizer.max_features if hasattr(self.vectorizer, 'max_features') else 0

        texts = [doc['text'] for doc in self.documents_metadata]

        # Hybrid Labeling Layer: Always build TF-IDF for semantic topic modeling
        self.tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()

        if self.engine_mode == "Deep Learning":
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
        
        if matrix is None or len(matrix) < 1: return

        # 1. Dimensionality Reduction (UMAP 3D / Fallback)
        if len(matrix) >= 3:
            # UMAP requires n_neighbors > 1.
            # With len(matrix) >= 3, min(..., len-1) will be >= 2.
            n_neighbors = max(2, min(15, len(matrix) - 1))
            reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.1, random_state=42, n_jobs=1)
            coords = reducer.fit_transform(matrix)

            # 2. Automated Clustering (KMeans)
            n_clusters = min(5, len(matrix))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(matrix)
        else:
            # Fallback for single document/segment or pairs to prevent UMAP crashes
            if len(matrix) == 1:
                coords = np.array([[0.0, 0.0, 0.0]])
                clusters = np.array([0])
            else: # len(matrix) == 2
                coords = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
                clusters = np.array([0, 1])

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
        
        Theory Note: UMAP (Uniform Manifold Approximation)
        -------------------------------------------------
        We use UMAP to project high-dimensional vectors (e.g., 768 or 1024 dims) 
        into 2D/3D space for human visualization.
        
        Key Parameters:
        - n_neighbors: Controls local vs. global structure. Low values 
          focus on small clusters; high values focus on the 'big picture'.
        - min_dist: Controls how tightly points are packed.
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
        valid_indices = [idx for idx in indices if idx < self.tfidf_matrix.shape[0]]
        if not valid_indices:
            return [f"Galaxy {cluster_id} (Syncing)"]
            
        cluster_tfidf = self.tfidf_matrix[valid_indices].sum(axis=0)
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
        Orchestrates the search request across the selected engine.
        
        The Mathematical Principle: Cosine Similarity
        We treat both the user query and the document segments as vectors in 
        high-dimensional space. We then calculate the 'angle' between them.
        *   **Angle = 0° (Similarity = 1.0)**: Perfect match.
        *   **Angle = 90° (Similarity = 0.0)**: No commonality.

        Args:
            query_text (str): The user's search query.
            llm_service: Required for 'Deep Learning' mode to vectorize the query.
            top_n (int): Number of results to return.
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
        """
        Contextual matching via dense vector similarity.
        
        Developer Note on Parallelization:
        Instead of looping through documents (O(n)), we use NumPy's vectorization
        to calculate matches across the entire index simultaneously.
        """
        if self.embeddings is None or not llm: return []
        q_vec = np.array(llm.embed_text(query))
        if q_vec.size == 0: return []
        
        # --- DIMENSION GUARDRAIL ---
        # If the user switched models (e.g., Nomic -> Gemma) without re-indexing,
        # the math will fail as the vectors have different lengths.
        if q_vec.shape[0] != self.embeddings.shape[1]:
            raise ValueError(f"Neural Dimension Mismatch: Index is {self.embeddings.shape[1]} (from {self.index_embedding_model}), but Query is {q_vec.shape[0]} (from {llm.embedding_model}). Please re-index.")

        
        # Parallel Cosine Similarity using NumPy
        # Formula: (A . B) / (||A|| * ||B||)
        q_norm = np.linalg.norm(q_vec)
        d_norms = np.linalg.norm(self.embeddings, axis=1)
        sims = np.dot(self.embeddings, q_vec) / (q_norm * d_norms + 1e-9)

        results = []
        for i, score in enumerate(sims):
            # We filter by a threshold to ensure quality in the final LLM context.
            if score > self.neural_threshold: 
                meta = self.documents_metadata[i].copy()
                meta['score'] = round(float(score), 4)
                results.append(meta)
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]


    def get_context_for_query(self, query_text, llm, top_n=5):
        """Formats the top N results as a structured text block for the LLM."""
        res = self.search(query_text, llm, top_n=top_n)
        return "\n".join([f"[Source: {r['file']} P{r['page']} | Full Path: {r.get('full_path', 'N/A')}]\n{r['text']}\n" for r in res])


    def get_document_text(self, filename):
        """Retrieves raw content for summarization."""
        return self.file_contents.get(filename, "")

    def get_file_manifest(self):
        """Returns a list of all unique filenames currently in the Knowledge Base."""
        return sorted(list(self.file_contents.keys()))

    # ------------------------------------------------------------------
    # PHASE 6: PERSISTENCE (Save/Load)
    # ------------------------------------------------------------------

    def save_to_disk(self, save_dir="data/index"):
        """Serializes the current knowledge state to disk."""
        if not self.documents_metadata: return False
        
        os.makedirs(save_dir, exist_ok=True)
        try:
            # 1. Save Text & Metadata (JSON)
            payload = {
                "metadata": self.documents_metadata,
                "file_contents": self.file_contents,
                "spatial": self.documents_spatial,
                "engine_mode": self.engine_mode,
                "index_model": getattr(self, "index_embedding_model", None),
                "index_dim": getattr(self, "index_embedding_dimension", 0),
                "granularity": self.spatial_granularity
            }
            with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                json.dump(payload, f)

            # 2. Save Matrices (Numpy)
            if self.tfidf_matrix is not None:
                np.save(os.path.join(save_dir, "tfidf_matrix.npy"), self.tfidf_matrix)
            if self.embeddings is not None:
                np.save(os.path.join(save_dir, "embeddings.npy"), self.embeddings)

            # 3. Save Vectorizer (Pickle - required for ML search)
            with open(os.path.join(save_dir, "vectorizer.pkl"), "wb") as f:
                pickle.dump(self.vectorizer, f)

            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False

    def load_from_disk(self, load_dir="data/index"):
        """Restores the knowledge state from disk."""
        if not os.path.exists(load_dir): return False
        
        try:
            # 1. Load Text & Metadata
            meta_path = os.path.join(load_dir, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    payload = json.load(f)
                    self.documents_metadata = payload.get("metadata", [])
                    self.file_contents = payload.get("file_contents", {})
                    self.documents_spatial = payload.get("spatial", [])
                    self.engine_mode = payload.get("engine_mode", "Deep Learning")
                    self.index_embedding_model = payload.get("index_model")
                    self.index_embedding_dimension = payload.get("index_dim", 0)
                    self.spatial_granularity = payload.get("granularity", "Segments")

            # 2. Load Matrices
            tfidf_path = os.path.join(load_dir, "tfidf_matrix.npy")
            if os.path.exists(tfidf_path):
                self.tfidf_matrix = np.load(tfidf_path)
            
            embed_path = os.path.join(load_dir, "embeddings.npy")
            if os.path.exists(embed_path):
                self.embeddings = np.load(embed_path)

            # 3. Load Vectorizer
            vec_path = os.path.join(load_dir, "vectorizer.pkl")
            if os.path.exists(vec_path):
                with open(vec_path, "rb") as f:
                    self.vectorizer = pickle.load(f)

            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False

    def get_top_keywords_df(self, top_n=10):
        """Analytics: Identifies the most statistically important terms in the index."""
        if self.engine_mode == "Machine Learning" and self.tfidf_matrix is not None:
            importance = np.mean(self.tfidf_matrix, axis=0)
            words = self.vectorizer.get_feature_names_out()
            return pd.DataFrame({'Keyword': words, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(top_n)
        return pd.DataFrame() # Analytics skipped for Neural to save performance
