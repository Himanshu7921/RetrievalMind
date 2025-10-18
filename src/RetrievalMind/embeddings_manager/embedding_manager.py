import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import uuid

class EmbeddingManager:
    """
    Manages sentence embeddings using a SentenceTransformer model.

    This class provides a simple interface for:
      - Loading pre-trained transformer models
      - Generating embeddings for text data
      - Retrieving embedding dimensions for downstream use

    Attributes:
        model_name (str): Name of the SentenceTransformer model to load.
        model (SentenceTransformer): Loaded transformer model instance.
    """

    def __init__(self, model_name: str = "all-miniLM-L6-v2"):
        """
        Initializes the EmbeddingManager with a specified transformer model.

        Args:
            model_name (str): The name of the SentenceTransformer model to use (default: "all-miniLM-L6-v2").
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Loads the SentenceTransformer model specified during initialization.

        Raises:
            RuntimeError: If the model fails to load due to incorrect name or missing dependencies.
        """
        try:
            # Attempt to load the pre-trained model
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.model_name}': {e}")

    def generate_embeddings(self, text):
        """
        Generates embeddings for the provided text using the loaded transformer model.

        Args:
            text (str or list[str]): Input text or list of sentences to encode.

        Returns:
            np.ndarray: Numpy array of generated embeddings.

        Raises:
            ValueError: If input text is empty or model is not loaded.
            RuntimeError: If an unexpected error occurs during encoding.
        """
        try:
            # Validate input
            if not text:
                raise ValueError("Input text cannot be empty.")

            if self.model is None:
                raise ValueError("Model not loaded. Call '_load_model()' before generating embeddings.")

            # Encode text into embeddings
            embeddings = self.model.encode(text)
            return np.array(embeddings)

        except ValueError as e:
            # Raise meaningful validation errors
            raise e
        except Exception as e:
            # Catch-all for any encoding errors
            raise RuntimeError(f"Error generating embeddings: {e}")

    def get_embedding_dimensions(self):
        """
        Retrieves the dimensionality of the model's embeddings.

        Returns:
            int: Embedding dimension of the loaded model.

        Raises:
            RuntimeError: If model is not loaded.
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Please initialize or reload the model first.")

            # Return embedding size
            return self.model.get_sentence_embedding_dimension()

        except Exception as e:
            raise RuntimeError(f"Error retrieving embedding dimensions: {e}")