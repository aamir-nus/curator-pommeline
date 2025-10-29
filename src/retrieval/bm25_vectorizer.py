"""
BM25 vectorizer for sparse keyword search with persistence support.
"""

import os
import pickle
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics
from ..config import settings

logger = get_logger("bm25_vectorizer")


class BM25Vectorizer:
    """
    BM25-inspired vectorizer for keyword search.
    Uses TF-IDF as base and applies BM25-style weighting.
    """

    def __init__(self,
                 k1: float = 1.2,
                 b: float = 0.75,
                 max_features: int = 10000,
                 fixed_dimension: int = 768,
                 ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize BM25 vectorizer.

        Args:
            k1: BM25 parameter for term frequency saturation
            b: BM25 parameter for document length normalization
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Range of n-grams to consider
        """
        self.k1 = k1
        self.b = b
        self.max_features = max_features
        self.fixed_dimension = fixed_dimension
        self.ngram_range = ngram_range

        # TF-IDF vectorizer as base
        self.vectorizer = TfidfVectorizer(
            max_features=min(max_features, fixed_dimension),
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=1,
            max_df=0.95
        )

        # BM25-specific attributes
        self.doc_lengths = None
        self.avg_doc_length = None
        self.is_fitted = False
        self.actual_dimension = None

        logger.info(f"Initialized BM25Vectorizer with k1={k1}, b={b}, fixed_dim={fixed_dimension}")

    def fit(self, documents: List[str]) -> 'BM25Vectorizer':
        """
        Fit the BM25 vectorizer on a corpus of documents.

        Args:
            documents: List of document texts

        Returns:
            Self for method chaining
        """
        if not documents:
            logger.warning("Empty document list provided to fit()")
            return self

        logger.info(f"Fitting BM25Vectorizer on {len(documents)} documents")

        # Fit TF-IDF vectorizer
        tfidf_matrix = self.vectorizer.fit_transform(documents)

        # Calculate document lengths (sum of TF-IDF scores)
        self.doc_lengths = np.array(tfidf_matrix.sum(axis=1)).flatten()
        self.avg_doc_length = np.mean(self.doc_lengths)

        self.is_fitted = True
        self.actual_dimension = len(self.vectorizer.vocabulary_)
        logger.info(f"BM25Vectorizer fitted with vocabulary size: {self.actual_dimension} (fixed_dim: {self.fixed_dimension})")

        return self

    def transform(self, documents: List[str]) -> sp.csr_matrix:
        """
        Transform documents to BM25-weighted vectors.

        Args:
            documents: List of document texts

        Returns:
            Sparse matrix of BM25-weighted vectors
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")

        if not documents:
            return sp.csr_matrix((0, len(self.vectorizer.vocabulary_)))

        # Get TF-IDF vectors
        tfidf_matrix = self.vectorizer.transform(documents)

        # Apply BM25 weighting
        doc_lengths_new = np.array(tfidf_matrix.sum(axis=1)).flatten()

        # BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
        tfidf_data = tfidf_matrix.data
        tfidf_indices = tfidf_matrix.indices
        tfidf_indptr = tfidf_matrix.indptr

        # Calculate BM25 weights
        bm25_data = []
        bm25_indices = []
        bm25_indptr = [0]

        for i in range(len(documents)):
            start_idx = tfidf_indptr[i]
            end_idx = tfidf_indptr[i + 1]

            doc_length = doc_lengths_new[i] if i < len(doc_lengths_new) else 0
            length_norm = 1 - self.b + self.b * (doc_length / (self.avg_doc_length + 1e-8))

            bm25_weights = []
            for j in range(start_idx, end_idx):
                tf_score = tfidf_data[j]
                bm25_score = tf_score * (self.k1 + 1) / (tf_score + self.k1 * length_norm)
                bm25_weights.append(bm25_score)
                bm25_indices.append(tfidf_indices[j])

            bm25_data.extend(bm25_weights)
            bm25_indptr.append(len(bm25_data))

        bm25_matrix = sp.csr_matrix(
            (bm25_data, bm25_indices, bm25_indptr),
            shape=(len(documents), len(self.vectorizer.vocabulary_))
        )

        return bm25_matrix

    def fit_transform(self, documents: List[str]) -> sp.csr_matrix:
        """Fit and transform in one step."""
        return self.fit(documents).transform(documents)

    def transform_query(self, query: str) -> np.ndarray:
        """
        Transform a single query for BM25 search.

        Args:
            query: Query string

        Returns:
            Dense vector for the query
        """
        if not self.is_fitted:
            logger.warning("Vectorizer not fitted, returning zero vector")
            return np.zeros(len(self.vectorizer.vocabulary_))

        # Transform query using TF-IDF
        query_tfidf = self.vectorizer.transform([query])
        query_vector = query_tfidf.toarray().flatten()

        # Pad or truncate to fixed dimension
        if len(query_vector) < self.fixed_dimension:
            # Pad with zeros
            padded_vector = np.zeros(self.fixed_dimension)
            padded_vector[:len(query_vector)] = query_vector
            query_vector = padded_vector
        else:
            # Truncate
            query_vector = query_vector[:self.fixed_dimension]

        # Normalize the vector for dotproduct similarity (scores 0-1)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        return query_vector

    def get_feature_names(self) -> List[str]:
        """Get vocabulary feature names."""
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()

    def save(self, filepath: str) -> bool:
        """
        Save the fitted vectorizer to disk.

        Args:
            filepath: Path to save the vectorizer

        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'doc_lengths': self.doc_lengths,
                    'avg_doc_length': self.avg_doc_length,
                    'is_fitted': self.is_fitted,
                    'k1': self.k1,
                    'b': self.b,
                    'max_features': self.max_features,
                    'fixed_dimension': self.fixed_dimension,
                    'ngram_range': self.ngram_range,
                    'actual_dimension': self.actual_dimension
                }, f)

            logger.info(f"BM25Vectorizer saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save BM25Vectorizer: {e}")
            return False

    @classmethod
    def load(cls, filepath: str) -> Optional['BM25Vectorizer']:
        """
        Load a fitted vectorizer from disk.

        Args:
            filepath: Path to the saved vectorizer

        Returns:
            BM25Vectorizer instance or None if failed
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            # Create new instance
            vectorizer = cls(
                k1=data['k1'],
                b=data['b'],
                max_features=data['max_features'],
                fixed_dimension=data.get('fixed_dimension', 768),
                ngram_range=data['ngram_range']
            )

            # Restore fitted state
            vectorizer.vectorizer = data['vectorizer']
            vectorizer.doc_lengths = data['doc_lengths']
            vectorizer.avg_doc_length = data['avg_doc_length']
            vectorizer.is_fitted = data['is_fitted']
            vectorizer.actual_dimension = data.get('actual_dimension')

            logger.info(f"BM25Vectorizer loaded from {filepath}")
            return vectorizer

        except Exception as e:
            logger.error(f"Failed to load BM25Vectorizer: {e}")
            return None

    @track_latency("bm25_search")
    def search(self,
              query: str,
              document_vectors: sp.csr_matrix,
              top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search documents using BM25 similarity.

        Args:
            query: Query string
            document_vectors: Pre-computed document vectors
            top_k: Number of top results to return

        Returns:
            List of (doc_index, score) tuples
        """
        if not self.is_fitted:
            logger.warning("Vectorizer not fitted, returning empty results")
            return []

        # Transform query
        query_vector = self.transform_query(query)

        if query_vector.sum() == 0:
            logger.warning(f"Query '{query}' resulted in zero vector")
            return []

        # Calculate cosine similarity
        similarities = cosine_similarity(
            query_vector.reshape(1, -1),
            document_vectors
        ).flatten()

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0]

        logger.debug(f"BM25 search for '{query}' returned {len(results)} results")
        metrics.add_metric("bm25_search_results_count", len(results))

        return results


# Global BM25 vectorizer registry
_vectorizers: Dict[str, BM25Vectorizer] = {}


def get_bm25_vectorizer(index_id: str) -> Optional[BM25Vectorizer]:
    """
    Get or load a BM25 vectorizer for a specific index.

    Args:
        index_id: Unique identifier for the index

    Returns:
        BM25Vectorizer instance or None if not found
    """
    if index_id in _vectorizers:
        return _vectorizers[index_id]

    # Try to load from disk
    vectorizer_path = f"{settings.data_dir}/models/bm25_{index_id}.pkl"
    vectorizer = BM25Vectorizer.load(vectorizer_path)

    if vectorizer:
        _vectorizers[index_id] = vectorizer
        return vectorizer

    logger.warning(f"BM25Vectorizer not found for index: {index_id}")
    return None


def register_bm25_vectorizer(index_id: str, vectorizer: BM25Vectorizer) -> bool:
    """
    Register and save a BM25 vectorizer for an index.

    Args:
        index_id: Unique identifier for the index
        vectorizer: Fitted BM25Vectorizer instance

    Returns:
        True if successful, False otherwise
    """
    if not vectorizer.is_fitted:
        logger.error("Cannot register unfitted BM25Vectorizer")
        return False

    # Save to disk
    vectorizer_path = f"{settings.data_dir}/models/bm25_{index_id}.pkl"
    if vectorizer.save(vectorizer_path):
        _vectorizers[index_id] = vectorizer
        return True

    return False