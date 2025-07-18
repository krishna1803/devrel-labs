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

import psycopg
from psycopg.extras import Json
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

            self.connection = psycopg.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password
)
            self.cursor = self.connection.cursor()
            
            # Initialize pgvector extension if not exists
            self._initialize_pgvector()
            self._create_tables()
            logging.info("PostgreSQL Connection successful!")
            print("PostgreSQL Connection successful!")
        except Exception as e:
            print("PostgreSQL Connection failed!", e)
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
            
    def _initialize_pgvector(self):
        """Initialize pgvector extension in PostgreSQL"""
        try:
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.connection.commit()
        except Exception as e:
            print(f"Error initializing pgvector extension: {str(e)}")
            raise
            
    def _create_tables(self):
        """Create vector tables if they don't exist"""
        embedding_dim = 384  # Dimension for all-MiniLM-L12-v2
        
        # Define the table creation queries
        tables = {
            "PDFCollection": f"""
                CREATE TABLE IF NOT EXISTS PDFCollection (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({embedding_dim})
                )
            """,
            "WebCollection": f"""
                CREATE TABLE IF NOT EXISTS WebCollection (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({embedding_dim})
                )
            """,
            "RepoCollection": f"""
                CREATE TABLE IF NOT EXISTS RepoCollection (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({embedding_dim})
                )
            """,
            "GeneralCollection": f"""
                CREATE TABLE IF NOT EXISTS GeneralCollection (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({embedding_dim})
                )
            """
        }
        
        # Create each table
        for table_name, create_query in tables.items():
            try:
                self.cursor.execute(create_query)
                
                # Create index on the embedding column for faster similarity search
                index_query = f"CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx ON {table_name} USING ivfflat (embedding vector_l2_ops)"
                self.cursor.execute(index_query)
                
                self.connection.commit()
            except Exception as e:
                print(f"Error creating table {table_name}: {str(e)}")
                self.connection.rollback()
    
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
        
        # Prepare data for Postgres
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [self._sanitize_metadata(chunk["metadata"]) for chunk in chunks]
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]

        # Encode all texts in a batch
        embeddings = self.encoder.encode(texts, batch_size=32, show_progress_bar=True)

        table_name = "PDFCollection"
        # Truncate the table - keeping this behavior from original implementation
        self.cursor.execute(f"TRUNCATE TABLE {table_name}")

        # Insert embeddings into Postgres
        for docid, text, metadata, embedding in zip(ids, texts, metadatas, embeddings):
            json_metadata = json.dumps(metadata)  # Convert to JSON string
            
            self.cursor.execute(
                "INSERT INTO PDFCollection (id, text, metadata, embedding) VALUES (%s, %s, %s, %s)",
                (docid, text, json_metadata, embedding.tolist())
            )

        self.connection.commit()
    
    def add_web_chunks(self, chunks: List[Dict[str, Any]], source_id: str):
        """Add chunks from web content to the vector store"""
        if not chunks:
            return
        
        # Prepare data for Postgres
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [self._sanitize_metadata(chunk["metadata"]) for chunk in chunks]
        ids = [f"{source_id}_{i}" for i in range(len(chunks))]

        # Encode all texts in a batch
        embeddings = self.encoder.encode(texts, batch_size=32, show_progress_bar=True)

        table_name = "WebCollection"
        # No truncation for web chunks, just append new ones

        # Insert embeddings into Postgres
        for docid, text, metadata, embedding in zip(ids, texts, metadatas, embeddings):
            json_metadata = json.dumps(metadata)  # Convert to JSON string
            
            self.cursor.execute(
                "INSERT INTO WebCollection (id, text, metadata, embedding) VALUES (%s, %s, %s, %s)",
                (docid, text, json_metadata, embedding.tolist())
            )

        self.connection.commit()
    
    def add_general_knowledge(self, chunks: List[Dict[str, Any]], source_id: str):
        """Add general knowledge chunks to the vector store"""
        if not chunks:
            return
        
        # Prepare data for Postgres
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [self._sanitize_metadata(chunk["metadata"]) for chunk in chunks]
        ids = [f"{source_id}_{i}" for i in range(len(chunks))]
        
        # Encode all texts in a batch
        embeddings = self.encoder.encode(texts, batch_size=32, show_progress_bar=True)

        table_name = "GeneralCollection"
        
        # Insert embeddings into Postgres
        for docid, text, metadata, embedding in zip(ids, texts, metadatas, embeddings):
            json_metadata = json.dumps(metadata)  # Convert to JSON string
            
            self.cursor.execute(
                "INSERT INTO GeneralCollection (id, text, metadata, embedding) VALUES (%s, %s, %s, %s)",
                (docid, text, json_metadata, embedding.tolist())
            )

        self.connection.commit()
    
    def add_repo_chunks(self, chunks: List[Dict[str, Any]], document_id: str):
        """Add chunks from a repository to the vector store"""
        if not chunks:
            return
        
        # Prepare data for Postgres
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [self._sanitize_metadata(chunk["metadata"]) for chunk in chunks]
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        
        # Encode all texts in a batch
        embeddings = self.encoder.encode(texts, batch_size=32, show_progress_bar=True)

        table_name = "RepoCollection"

        # Insert embeddings into Postgres
        for docid, text, metadata, embedding in zip(ids, texts, metadatas, embeddings):
            json_metadata = json.dumps(metadata)  # Convert to JSON string
            
            self.cursor.execute(
                "INSERT INTO RepoCollection (id, text, metadata, embedding) VALUES (%s, %s, %s, %s)",
                (docid, text, json_metadata, embedding.tolist())
            )

        self.connection.commit()
    
    def get_collection_count(self, collection_name: str) -> int:
        """Get the total number of chunks in a collection
        
        Args:
            collection_name: Name of the collection (pdf_documents, web_documents, repository_documents, general_knowledge)
            
        Returns:
            Number of chunks in the collection
        """
        # Map collection names to table names
        collection_map = {
            "pdf_documents": "PDFCollection",
            "web_documents": "WebCollection",
            "repository_documents": "RepoCollection", 
            "general_knowledge": "GeneralCollection"
        }
        
        table_name = collection_map.get(collection_name)
        if not table_name:
            raise ValueError(f"Unknown collection name: {collection_name}")
        
        # Count the rows in the table
        sql = f"SELECT COUNT(*) FROM {table_name}"
        self.cursor.execute(sql)
        count = self.cursor.fetchone()[0]
        
        return count
    
    def get_latest_chunk(self, collection_name: str) -> Dict[str, Any]:
        """Get the most recently inserted chunk from a collection
        
        Args:
            collection_name: Name of the collection (pdf_documents, web_documents, repository_documents, general_knowledge)
            
        Returns:
            Dictionary containing the content and metadata of the latest chunk
        """
        # Map collection names to table names
        collection_map = {
            "pdf_documents": "PDFCollection",
            "web_documents": "WebCollection",
            "repository_documents": "RepoCollection", 
            "general_knowledge": "GeneralCollection"
        }
        
        table_name = collection_map.get(collection_name)
        if not table_name:
            raise ValueError(f"Unknown collection name: {collection_name}")
        
        # Get the most recently inserted row (using CTID as a proxy for insertion order in PostgreSQL)
        sql = f"SELECT id, text, metadata FROM {table_name} ORDER BY ctid DESC LIMIT 1"
        self.cursor.execute(sql)
        row = self.cursor.fetchone()
        
        if not row:
            raise ValueError(f"No chunks found in collection: {collection_name}")
        
        result = {
            "id": row[0],
            "content": row[1],
            "metadata": json.loads(row[2]) if isinstance(row[2], str) else row[2]
        }
        
        return result
    
    def query_pdf_collection(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Query the PDF documents collection"""
        print("🔍 [PostgreSQL] Querying PDF Collection")
        # Generate Embeddings
        embeddings = self.encoder.encode(query, batch_size=32, show_progress_bar=True)

        sql = f"""
            SELECT id, text, metadata
            FROM PDFCollection
            ORDER BY embedding <-> %s
            LIMIT %s
            """

        self.cursor.execute(sql, (embeddings.tolist(), n_results))

        # Fetch all rows
        rows = self.cursor.fetchall()
        
        # Format results
        formatted_results = []
        for row in rows:
            result = {
                "content": row[1],
                "metadata": json.loads(row[2]) if isinstance(row[2], str) else row[2]
            }
            formatted_results.append(result)
        
        print(f"🔍 [PostgreSQL] Retrieved {len(formatted_results)} chunks from PDF Collection")
        return formatted_results
    
    def query_web_collection(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Query the web documents collection"""
        print("🔍 [PostgreSQL] Querying Web Collection")
        # Generate Embeddings
        embeddings = self.encoder.encode(query, batch_size=32, show_progress_bar=True)

        sql = f"""
            SELECT id, text, metadata
            FROM WebCollection
            ORDER BY embedding <-> %s
            LIMIT %s
            """

        self.cursor.execute(sql, (embeddings.tolist(), n_results))

        # Fetch all rows
        rows = self.cursor.fetchall()

        # Format results
        formatted_results = []
        for row in rows:
            result = {
                "content": row[1],
                "metadata": json.loads(row[2]) if isinstance(row[2], str) else row[2]
            }
            formatted_results.append(result)
        
        print(f"🔍 [PostgreSQL] Retrieved {len(formatted_results)} chunks from Web Collection")
        return formatted_results
    
    def query_general_collection(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Query the general knowledge collection"""
        print("🔍 [PostgreSQL] Querying General Knowledge Collection")
        # Generate Embeddings
        embeddings = self.encoder.encode(query, batch_size=32, show_progress_bar=True)

        sql = f"""
            SELECT id, text, metadata
            FROM GeneralCollection
            ORDER BY embedding <-> %s
            LIMIT %s
            """

        self.cursor.execute(sql, (embeddings.tolist(), n_results))

        # Fetch all rows
        rows = self.cursor.fetchall()

        # Format results
        formatted_results = []
        for row in rows:
            result = {
                "content": row[1],
                "metadata": json.loads(row[2]) if isinstance(row[2], str) else row[2]
            }
            formatted_results.append(result)
        
        print(f"🔍 [PostgreSQL] Retrieved {len(formatted_results)} chunks from General Knowledge Collection")
        return formatted_results
    
    def query_repo_collection(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Query the repository documents collection"""
        print("🔍 [PostgreSQL] Querying Repository Collection")
        # Generate Embeddings
        embeddings = self.encoder.encode(query, batch_size=32, show_progress_bar=True)

        sql = f"""
            SELECT id, text, metadata
            FROM RepoCollection
            ORDER BY embedding <-> %s
            LIMIT %s
            """

        self.cursor.execute(sql, (embeddings.tolist(), n_results))

        # Fetch all rows
        rows = self.cursor.fetchall()
        
        # Format results
        formatted_results = []
        for row in rows:
            result = {
                "content": row[1],
                "metadata": json.loads(row[2]) if isinstance(row[2], str) else row[2]
            }
            formatted_results.append(result)
        
        print(f"🔍 [PostgreSQL] Retrieved {len(formatted_results)} chunks from Repository Collection")
        return formatted_results
        
    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding_function

    def similarity_search(
        self, query: str, k: int = 3, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Return docs most similar to query."""
        logging.info(f"Similarity Search")

        if self.verbose:
            logging.info(f"top_k: {k}")
            logging.info("")

        # Embed the query
        embed_query = self._embedding_function(query)

        sql = f"""
            SELECT id, text, metadata
            FROM PDFCollection
            ORDER BY embedding <-> %s
            LIMIT %s
            """

        self.cursor.execute(sql, (embed_query.tolist(), k))

        # Fetch all rows
        rows = self.cursor.fetchall()
        
        # Format results
        formatted_results = []
        for row in rows:
            result = {
                "content": row[1],
                "metadata": json.loads(row[2]) if isinstance(row[2], str) else row[2]
            }
            formatted_results.append(result)
        
        print(f"🔍 [PostgreSQL] Retrieved {len(formatted_results)} chunks from PDF Collection")
        return formatted_results
    
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