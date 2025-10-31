import os

import numpy as np
import torch

from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer

from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics
from ..config import settings, get_embedding_config

logger = get_logger("embedder")

class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers with float16 optimization.
    Handles model loading, embedding generation, normalization, and similarity computation.

    Attributes:
        model_name (str): Name of the embedding model.
        config (dict): Configuration for the embedding model.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda', 'mps').
        normalize_embeddings (bool): Whether to normalize embeddings.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self.config = get_embedding_config()
        self.device = self.config.get("device", "cpu")
        self.normalize_embeddings = self.config.get("normalize_embeddings", True)

        self.model: Optional[SentenceTransformer] = None
        self.embedding_dimension = settings.embedding_dimension

    def _set_device(self):
        # Set device to gpu > mps > cpu

        if torch.cuda.is_available():
            self.device = "cuda"

        elif torch.backends.mps.is_available():
            self.device = "mps"
        
        else:
            self.device = "cpu"

    def load_model(self):
        """
        Load the embedding model with float16 optimization first, otherwise fallback to float32.
        
        """
        
        self._set_device()
        if self.model is not None:
            return

        logger.info(f"Loading embedding model: {self.model_name}")

        try:
            # Load model with float16 precision
            self.model = SentenceTransformer(
                self.model_name,
                token=settings.hf_api_key,
                device=self.device,
                trust_remote_code=self.config.get("trust_remote_code", True)
            )

            # # Configure for float16 if available
            # if hasattr(self.model, 'to'):
            #     self.model = self.model.to(torch.float16)

            # Test embedding to get dimension and check for validity
            test_embedding = self.model.encode("test", show_progress_bar=False)

            # Check if float16 caused NaN/Inf values and fallback to float32 if needed
            # if np.isnan(test_embedding).any() or np.isinf(test_embedding).any():
            #     logger.warning("Float16 precision caused NaN/Inf values, falling back to float32")
            #     self.model = self.model.to(torch.float32)
            #     test_embedding = self.model.encode("test", show_progress_bar=False)

            #     # Final check
            #     if np.isnan(test_embedding).any() or np.isinf(test_embedding).any():
            #         raise ValueError("Model produces invalid embeddings even with float32")

            self.embedding_dimension = len(test_embedding)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")

        except Exception as e:
            logger.warning(f"Failed to load embedding model {self.model_name}: {e}")
            logger.warning("Falling back to CPU and float32 precision.")
            
            self.model = SentenceTransformer(
                self.model_name,
                token=settings.hf_api_key
            ).to(device=self.device)

    @track_latency("embedding_generation")
    def generate_embeddings(self,
                           texts: Union[str, List[str]],
                           batch_size: int = 32,
                           show_progress: bool = False) -> np.ndarray:
        """Generate embeddings for input texts."""
        if self.model is None:
            self.load_model()

        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([])

        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings
            )

            # Check for NaN/Inf values before dtype conversion
            if np.isnan(embeddings).any() or np.isinf(embeddings).any():
                logger.warning("Model produced NaN/Inf values, embeddings may be invalid")

            # Only convert to float16 if it's safe and the model is in float16
            model_dtype = next(self.model.parameters()).dtype if hasattr(self.model, 'parameters') else torch.float32
            if model_dtype == torch.float16 and not (np.isnan(embeddings).any() or np.isinf(embeddings).any()):
                embeddings = embeddings.astype(np.float16)

            logger.debug(f"Generated embeddings for {len(texts)} texts")
            metrics.add_metric("embedding_texts_count", len(texts))

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        embedding = self.generate_embeddings(texts=text)
        return embedding[0] if len(embedding) > 0 else np.array([])

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self.model is None:
            self.load_model()
        return self.embedding_dimension

    def normalize_embeddings_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length for inner-product similarity."""
        if len(embeddings) == 0:
            return embeddings

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        if len(embedding1) == 0 or len(embedding2) == 0:
            return 0.0

        # Normalize embeddings
        emb1_norm = self.normalize_embeddings_batch(embedding1.reshape(1, -1))[0]
        emb2_norm = self.normalize_embeddings_batch(embedding2.reshape(1, -1))[0]

        # Compute dot product (cosine similarity for normalized vectors)
        return float(np.dot(emb1_norm, emb2_norm))

    def compute_similarities(self,
                           query_embedding: np.ndarray,
                           corpus_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarities between query and corpus embeddings."""
        if len(query_embedding) == 0 or len(corpus_embeddings) == 0:
            return np.array([])

        # Normalize embeddings
        query_norm = self.normalize_embeddings_batch(query_embedding.reshape(1, -1))[0]
        corpus_norm = self.normalize_embeddings_batch(corpus_embeddings)

        # Compute dot products
        similarities = np.dot(corpus_norm, query_norm)
        return similarities

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Embedding model unloaded")


# Global embedding generator instance
embedder = EmbeddingGenerator()


def get_embedder() -> EmbeddingGenerator:
    """Get the global embedding generator instance."""
    return embedder


def generate_embeddings(texts: Union[str, List[str]], **kwargs) -> np.ndarray:
    """Generate embeddings using the global embedder."""
    return embedder.generate_embeddings(texts, **kwargs)


def generate_embedding(text: str, **kwargs) -> np.ndarray:
    """Generate a single embedding using the global embedder."""
    return embedder.generate_single_embedding(text, **kwargs)