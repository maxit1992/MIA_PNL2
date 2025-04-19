import os

from pinecone import Pinecone
from transformers import AutoModel


class SingletonPinecone:
    """
    This class implements a singleton pattern to ensure that single instances of the Pinecone client and the embedding
    model are created and shared throughout the application.
    """
    _instance = None
    model = None
    pc = None

    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of the class is created. If an instance already exists, it returns the existing
        instance.
        """
        if not cls._instance:
            cls._instance = super(SingletonPinecone, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the Pinecone client and the embedding model if they have not already been initialized,
        using the API key from the PINECONE_API_KEY environment variable.
        """
        if self.model is None:
            self.model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-small-en', trust_remote_code=True)
        if self.pc is None:
            self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
