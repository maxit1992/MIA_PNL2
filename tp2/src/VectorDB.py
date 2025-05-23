from pinecone import ServerlessSpec

from SingletonPinecone import SingletonPinecone


class VectorDB:
    """
    This class manages a vector database for storing and retrieving text.
    """

    def __init__(self, index_name: str):
        """
        Initializes the VectorDB for a specific index with the transformer model and Pinecone client.

        Args:
            index_name (str): The index for the vector database.
        """
        self.model = SingletonPinecone().model
        self.pc = SingletonPinecone().pc
        self.index_name = index_name

    def get_embeddings(self, text: list[str]):
        """
        Generates embeddings for a list of text strings.

        Args:
            text (list[str]): A list of text strings to generate embeddings for.

        Returns:
            list: A list of embeddings corresponding to the input text.
        """
        return self.model.encode(text)

    def save_text(self, text: list[str]):
        """
        Saves text data to the vector database by creating embeddings and storing them.

        Args:
            text (list[str]): A list of text strings to save in the vector database.
        """
        embeddings = self.get_embeddings(text)
        if self.index_name in [index.name for index in self.pc.list_indexes()]:
            self.pc.delete_index(self.index_name)
        self.pc.create_index(
            name=self.index_name,
            dimension=len(embeddings[0]),
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        index = self.pc.Index(self.index_name)
        data = [(f"id-{i}", embeddings[i], {"text": text[i]}) for i in range(len(embeddings))]
        index.upsert(vectors=data)

    def get_similar_text(self, text: str, top_k: int = 5):
        """
        Retrieves the most similar text entries from the vector database.

        Args:
            text (str): The input text to find similar entries for.
            top_k (int): The number of top similar entries to retrieve.

        Returns:
            dict: A dictionary containing the results of the similarity query.
        """
        embedding = self.get_embeddings([text])
        index = self.pc.Index(self.index_name)
        result = index.query(
            vector=embedding[0].tolist(),
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        return result
