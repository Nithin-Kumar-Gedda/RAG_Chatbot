import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingManager:
    def __init__(self,model_name = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager

        Args: 
            model_name: HuggingFace model name for sentence embedding
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

        except Exception as e:
            print(f"Error loading model {self.model_name}:{e}")
            raise


    def generate_embeddings(self,texts: List[str]) -> np.ndarray:
        """ 
        Args:
            texts: List of text strings to embed
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("model not loaded")
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
