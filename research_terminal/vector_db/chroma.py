from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from uuid import uuid4
import logging  
logger = logging.getLogger(__name__)
from research_terminal.scraping.processing.text import split_text
from typing import Optional

PERSIST_DIRECTORY = 'research_terminal/vector_db/db'
COLLECTION_NAME = 'Research'
MODEL_NAME = "all-MiniLM-L6-v2"


class ChromaDBClient:
    def __init__(self, persist_directory: str=PERSIST_DIRECTORY, collection_name: str=COLLECTION_NAME, model_name: str = MODEL_NAME):
        """Initializes a ChromaDB client.
        Args:
            persist_directory: The directory to persist the database in.
            collection_name: The name of the collection to use.
            model_name: The name of the model to use for embeddings."""
        logger.info("Initializing ChromaDB client...")
        self.client = chromadb.PersistentClient(persist_directory, settings=Settings(allow_reset=True, anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

        
    def add_text(self, text: str, metadata: Optional[dict] = None):
        """Adds a text to the collection.
        Args:
            text(str): The text to add."""

        texts = list(split_text(text, max_length=1024))

        metadatas = []
        for _ in texts:
            meta = metadata.copy() if metadata else {}
            meta["timestamp"] = datetime.now().isoformat(sep=" ", timespec="minutes")
            metadatas.append(meta)

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=[str(uuid4()) for _ in texts],
        )

    def query_text(self, text: str, top_k: int = 1, **kwargs):
        """Queries the collection for a text.
            Args:
                text(str): The text to query.
                top_k(int): The number of results to return.
            Returns:
                dict: The query results."""
        return self.collection.query(
            query_texts=[text],
            n_results=top_k, 
            include=["documents", "metadatas"],
            **kwargs)

    def get_text(self, ids: str, **kwargs):
        """Gets a text from the collection by its id.
            Args:
            ids(str): The id of the text to get.
            Returns:
                dict: The query results."""
        return self.collection.get(
            ids=[ids],
            **kwargs
        )

    def update_text(self, ids: str, text: str, metadatas={}, **kwargs):
        """Updates a text in the collection by its id.
            Args:
                ids(str): The id of the text to update.
                text(str): The new text.
                metadatas(dict): The new metadata."""
        if not metadatas:
            metadatas = {"timestamp": datetime.now().isoformat(sep=" ", timespec="minutes")}
        self.collection.update(
            ids=[ids],
            documents=[text],
            metadatas=[metadatas],
            **kwargs
        )

    def upsert_text(self, ids: str, text: str, metadatas={}, **kwargs):
        """Upserts a text in the collection by its id.
            Args:
                ids(str): The id of the text to upsert.
                text(str): The new text.
                metadatas(dict): The new metadata."""
        if not metadatas:
            metadatas = {"timestamp": datetime.now().isoformat(sep=" ", timespec="minutes")}
        self.collection.upsert(
            ids=[ids],
            documents=[text],
            metadatas=[metadatas], **kwargs
        )

    def delete_text(self, ids: str, **kwargs):
        """Deletes a text from the collection by its id.
            Args:
                ids(str): The id of the text to delete.
                """
        self.collection.delete(
            ids=[ids],
            **kwargs
        )

    def list_collections(self):
        """Lists all collections."""
        return self.client.list_collections()

    def create_collection(self, name: str, **kwargs):
        """Creates a new collection."""
        return self.client.get_or_create_collection(name, **kwargs)

    def delete_collection(self, name: str):
        """Deletes a collection."""
        return self.client.delete_collection(name)

    def reset(self):
        """Resets the entire database. This can't be undone!"""
        return self.client.reset()

