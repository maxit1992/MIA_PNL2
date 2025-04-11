import os

from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModel


class VectorDB:
    INDEX_NAME = "pnl2-tp1"

    def __init__(self, model_name: str = 'jinaai/jina-embeddings-v2-small-en'):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    def get_embeddings(self, text: list[str]):
        return self.model.encode(text)

    def save_text(self, text: list[str]):
        embeddings = self.get_embeddings(text)

        if self.INDEX_NAME in [index.name for index in self.pc.list_indexes()]:
            self.pc.delete_index(self.INDEX_NAME)
        self.pc.create_index(
            name=self.INDEX_NAME,
            dimension=len(embeddings[0]),
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        index = self.pc.Index(self.INDEX_NAME)
        data = [(f"id-{i}", embeddings[i], {"text": text[i]}) for i in range(len(embeddings))]
        index.upsert(vectors=data)

    def get_similar_text(self, text: str, top_k: int = 5):
        embedding = self.get_embeddings([text])
        index = self.pc.Index(self.INDEX_NAME)
        result = index.query(
            vector=embedding[0].tolist(),
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        return result
