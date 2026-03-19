import numpy as np
import re
import pandas as pd
import json
import os
import hashlib

class KnowledgeBase:
    def __init__(self, chunk_size=600, overlap_size=100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.dataset_overlap = 15 # Overlap 15 rows for datasets
        self.documents_metadata = []
        self.embeddings = None  # Dense neural vectors
        
        # Store full content per file for retrieval context
        self.file_contents = {}

        # Analytics Tracking
        self.cleaning_report = []
        self.file_chunk_counts = {}
        self.cache_file = ".neural_cache.json"
        self._embed_cache = self._load_disk_cache()

    def _get_text_hash(self, text):
        """Generate a unique ID for a text chunk to use as a cache key."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _load_disk_cache(self):
        """Load previously calculated embeddings from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
        return {}

    def _save_disk_cache(self):
        """Save the current embedding cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._embed_cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def clean_text(self, text):
        """Advanced NLP Normalization & Noise Reduction."""
        if not isinstance(text, str):
            text = str(text)
        # 1. Case Folding
        text = text.lower()
        # 2. Strip URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # 3. Strip structural symbols but keep . ! ? for sentence splitting
        text = re.sub(r'[^a-z0-9\s\.!?]', ' ', text)
        # 4. Normalize Whitespace
        return re.sub(r'\s+', ' ', text).strip()

    def process_text(self, filename, raw_text, page_num=1):
        """Processes text, logs metrics, and handles chunking (sentence-aware)."""
        # Store raw content for viewing/reference
        if filename not in self.file_contents:
            self.file_contents[filename] = ""
        self.file_contents[filename] += raw_text

        cleaned = self.clean_text(raw_text)
        
        # Log cleaning metrics
        raw_len = len(raw_text)
        clean_len = len(cleaned)

        if not any(d['File'] == filename for d in self.cleaning_report):
            self.cleaning_report.append({
                "File": filename,
                "Original Chars": raw_len,
                "Cleaned Chars": clean_len,
                "Noise Reduction": f"{((raw_len - clean_len) / raw_len * 100):.1f}%" if raw_len > 0 else "0%"
            })

        # Sentence-Aware Chunking
    def process_text(self, filename, text, page=1):
        """Implements Semantic Overlap Chunking for plain text/PDF."""
        cleaned = self.clean_text(text)
        self.file_contents[filename] = self.file_contents.get(filename, "") + "\n" + text
        
        # Sliding window chunking
        start = 0
        chunks_for_file = 0
        while start < len(cleaned):
            end = start + self.chunk_size
            chunk = cleaned[start:end]
            
            self.documents_metadata.append({
                "text": chunk,
                "file": filename,
                "page": page
            })
            chunks_for_file += 1
            
            # Slide by (chunk_size - overlap)
            # If we are near the end, stop
            if end >= len(cleaned):
                break
            start += (self.chunk_size - self.overlap_size)
            
        self.file_chunk_counts[filename] = self.file_chunk_counts.get(filename, 0) + chunks_for_file

    def process_dataset(self, filename, df):
        """Optimized row-wise ingestion with sliding window overlap."""
        rows = len(df)
        
        # Store for summarization
        self.file_contents[filename] = df.head(100).to_string()
        
        self.cleaning_report.append({
            "File": filename,
            "Entries": rows,
            "Format": "Neural Matrix (Sliding Window)",
            "Status": "Optimized Ingestion"
        })

        # Process in blocks of rows with overlap
        block_size = 100
        overlap = self.dataset_overlap
        chunks_for_file = 0
        
        start = 0
        while start < rows:
            end = min(start + block_size, rows)
            block = df.iloc[start:end]
            block_text = block.to_string(index=False)
            
            self.documents_metadata.append({
                "text": block_text,
                "file": filename,
                "page": f"Rows {start}-{end}"
            })
            chunks_for_file += 1
            
            if end >= rows:
                break
            # Slide by block_size minus the overlap
            start += (block_size - overlap)
        
        self.file_chunk_counts[filename] = chunks_for_file

    def build_index(self, llm_service):
        """Builds neural embeddings index using batch processing and caching."""
        if not self.documents_metadata:
            return

        texts = [doc["text"] for doc in self.documents_metadata]
        
        # Optimize with disk cache: only embed what we haven't seen before
        embeddings = [None] * len(texts)
        to_embed = []
        indices_to_embed = []
        hashes_to_embed = []
        
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash in self._embed_cache:
                embeddings[i] = self._embed_cache[text_hash]
            else:
                to_embed.append(text)
                indices_to_embed.append(i)
                hashes_to_embed.append(text_hash)
        
        if to_embed:
            # Embed the new ones in native batches
            batch_size = 100
            new_embeddings_generated = False
            for i in range(0, len(to_embed), batch_size):
                batch = to_embed[i:i+batch_size]
                new_vecs = llm_service.embed_batch(batch)
                
                # Assign back to main list and update cache
                for j, emb in enumerate(new_vecs):
                    if i + j < len(indices_to_embed):
                        idx = indices_to_embed[i + j]
                        h = hashes_to_embed[i + j]
                        embeddings[idx] = emb
                        self._embed_cache[h] = emb
                        new_embeddings_generated = True

            # Save updated cache to disk if we did any NEW work
            if new_embeddings_generated:
                self._save_disk_cache()

        # Filter out any failed embeddings
        valid_embeddings = [e for e in embeddings if e is not None]
        if valid_embeddings:
            self.embeddings = np.array(valid_embeddings)

    def search(self, query_text, llm_service, top_n=5):
        """Manual Cosine Similarity on dense vectors."""
        if self.embeddings is None or not self.documents_metadata:
            return []

        query_vec = np.array(llm_service.embed_text(query_text))
        if query_vec.size == 0:
            return []

        # Calculate Query Norm
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []

        # Vectorized calculation for all documents
        # Cosine Similarity = Dot(A, B) / (||A|| * ||B||)
        d_norms = np.linalg.norm(self.embeddings, axis=1)
        dot_products = np.dot(self.embeddings, query_vec)
        
        # Avoid division by zero
        similarities = np.divide(dot_products, (q_norm * d_norms), 
                                 out=np.zeros_like(dot_products), 
                                 where=(q_norm * d_norms) != 0)

        results = []
        for i, score in enumerate(similarities):
            if score > 0.3: # Neural threshold is usually higher than TF-IDF
                meta = self.documents_metadata[i].copy()
                meta['score'] = round(float(score), 4)
                results.append(meta)

        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]

    def get_context_for_query(self, query_text, llm_service, top_n=5):
        """Search and format results for LLM consumption."""
        results = self.search(query_text, llm_service, top_n=top_n)
        if not results:
            return ""

        context_parts = []
        for i, res in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {res['file']} — {res['page']}]\n"
                f"{res['text']}\n"
            )
        return "\n".join(context_parts)

    def get_document_text(self, filename):
        return self.file_contents.get(filename, "")

    def get_top_keywords_df(self, top_n=10):
        """Neural embeddings don't map directly to keywords, so we use word frequency."""
        from collections import Counter
        if not self.documents_metadata:
            return pd.DataFrame(columns=['Keyword', 'Count'])
            
        all_text = " ".join([doc["text"] for doc in self.documents_metadata])
        words = re.findall(r'\w+', all_text.lower())
        stopwords = {'the', 'and', 'to', 'of', 'in', 'is', 'it', 'for', 'on', 'with', 'as', 'this', 'that', 'by', 'at', 'an'}
        words = [w for w in words if len(w) > 3 and w not in stopwords]
        
        counts = Counter(words).most_common(top_n)
        return pd.DataFrame(counts, columns=['Keyword', 'Count'])