from __future__ import annotations

import time
import array
import numpy as np
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_postgres import PGVector

#import psycopg2
#from psycopg2.extras import Json
import argparse
from pathlib import Path
import yaml
import json

from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class PostgresVectorStore(VectorStore):
    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "PDF Collection"
    
    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        client_settings: Optional[Any] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[Any] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        verbose: Optional[bool] = False,
        connection: Optional[Any] = None,
        cursor: Optional[Any] = None,
    ) -> None:
        self.verbose = verbose
        
        # Initialize default embedding model if none provided
        if embedding_function is None:
            # Create a default embedding model
            model = SentenceTransformer('all-MiniLM-L12-v2')
            self._embedding_function = model.encode
        else:
            self._embedding_function = embedding_function

        self.collection_name = collection_name
        self.encoder = SentenceTransformer('all-MiniLM-L12-v2')
        self.override_relevance_score_fn = relevance_score_fn
        
        # Load Postgres DB credentials from config_pg.yaml
        credentials = self._load_config()
        
        host = credentials.get("PG_HOST", "localhost")
        port = int(credentials.get("PG_PORT", "5432"))
        database = credentials.get("PG_DATABASE", "langchain")
        username = credentials.get("PG_USERNAME", "langchain")
        password = credentials.get("PG_PASSWORD", "langchain")

        if not host or not password:
            raise ValueError("PostgreSQL credentials not found in config_pg.yaml. Please set PG_HOST, PG_PORT, PG_DATABASE, PG_USERNAME, and PG_PASSWORD.")

        # Connect to the database
        try:
            #prepare connection string
            conn_string = f"postgresql+psycopg://{username}:{password}@{host}:{port}/{database}"  # Uses psycopg3!

            vector_store = PGVector(
                        embeddings=self._embedding_function,
                        collection_name=collection_name,
                        connection=conn_string,
                        use_jsonb=True,
            ) 
            self.connection = vector_store

            logging.info("PostgreSQL Collection Creation successful!")
            print("PostgreSQL Collection Creation successful!")
        except Exception as e:
            print("PostgreSQL Collection Creation failed!", e)
            raise

    def _load_config(self) -> Dict[str, str]:
        """Load configuration from config_pg.yaml"""
        try:
            config_path = Path("config_pg.yaml")
            if not config_path.exists():
                print("Warning: config.yaml not found. Using empty configuration.")
                return {}
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config if config else {}
        except Exception as e:
            print(f"Warning: Error loading config: {str(e)}")
            return {}
            
    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Sanitize metadata to ensure all values are valid types for Oracle DB"""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert list to string representation
                sanitized[key] = str(value)
            elif value is None:
                # Replace None with empty string
                sanitized[key] = ""
            else:
                # Convert any other type to string
                sanitized[key] = str(value)
        return sanitized
            
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        logging.info(f"add_texts")
        """Run more texts through the embeddings and add to the vectorstore.
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        raise NotImplementedError("add_texts method must be implemented...")

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to ensure it can be serialized to JSON"""
        # Implement metadata sanitization if needed
        return metadata
        
    def add_pdf_chunks(self, chunks: List[Dict[str, Any]], document_id: str):
        """Add chunks from a PDF document to the vector store"""
        if not chunks:
            return
        
        # Prepare data for Oracle DB
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [self._sanitize_metadata(chunk["metadata"]) for chunk in chunks]
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        
        #Create a list of Document onjects with content and metadata
        documents = [
            Document(
                page_content=chunk["text"],
                metadata=self._sanitize_metadata(chunk["metadata"])
            ) for chunk in chunks
        ]

        # Add documents to the vector store
        self.add_documents(documents)
        logging.info(f"Added {len(chunks)} chunks from document {document_id} to Postgres vector store")   
        print(f"Added {len(chunks)} chunks from document {document_id} to Postgres vector store")
           
    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding_function

    def similarity_search(
        self, query: str, k: int = 3, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        results = self.connection.similarity_search(query, k=k)
        print(f"Found {len(results)} results for query '{query}':")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Content: {result.page_content}")
            print(f"Metadata: {result.metadata}")
        return results
    
    #Function to imlement similarity search with a retriever
    def similarity_search_with_retriever(self,query:str, k=3):
        retriever = self.connection.as_retriever(search_type="similarity", search_kwargs={"k": k})
        results = retriever.invoke(query)
        print(f"Found {len(results)} results for query '{query}':")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Content: {result.page_content}")
            print(f"Metadata: {result.metadata}")
        return results
    
    def as_retriever(self):
        """Return a retriever that uses this vector store for semantic search."""
        from langchain.retrievers import VectorStoreRetriever
        
        return VectorStoreRetriever(
            vectorstore=self,
            search_type="similarity",
            search_kwargs={"k": 10}
        )
    
    @classmethod
    def from_texts(
        cls: Type[PostgresVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> PostgresVectorStore:
        logging.info(f"from_texts")
        """Return VectorStore initialized from texts and embeddings."""
        raise NotImplementedError("from_texts method must be implemented...")

def main():
    parser = argparse.ArgumentParser(description="Manage Oracle DB vector store")
    parser.add_argument("--add", help="JSON file containing chunks to add")
    parser.add_argument("--query", help="Query to search for")
    
    args = parser.parse_args()
    store = PostgresVectorStore()
    
    if args.add:
        with open(args.add, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        store.add_pdf_chunks(chunks, document_id=args.add)
        print(f"✓ Added {len(chunks)} PDF chunks to Postgres vector store")

if __name__ == "__main__":
    main()     