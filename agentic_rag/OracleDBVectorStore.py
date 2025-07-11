
from __future__ import annotations
# allows using postponed evaluation of type annotations
# type annotations are a way to provide hints about the types of variables and function return values
# so an object type can be used i.e. during new class creation before being defined later in the module 
# introduced in python 3.7, default behaviour starting Python 3.10

import time
import array

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

import oracledb
import argparse
import logging
from pathlib import Path
import yaml
import json

from sentence_transformers import SentenceTransformer


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

#
# OracleVectorStore
#
class OracleDBVectorStore(VectorStore): # inherits from langchain_core.vectorstores
    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "PDF Collection" # class attribute 
    # Class attributes are shared among all instances of the class. They are accessed using the class name (OracleVectorStore) rather than an instance of the class.
    # Prefixing a variable or attribute name with an underscore (_) in Python is a convention that suggests that the variable or attribute is intended for internal use within the class or module
    # It serves as a signal to other developers that the variable or attribute is not part of the public API and should be treated as implementation details. 

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
            model = SentenceTransformer('all-MiniLM-L6-v2')
            self._embedding_function = model.encode
        else:
            self._embedding_function = embedding_function

        self.collection_name = collection_name

        self.override_relevance_score_fn = relevance_score_fn
        
         # Load Oracle DB credentials from config.yaml
        credentials = self._load_config()
        
        username = credentials.get("ORACLE_DB_USERNAME", "ADMIN")
        password = credentials.get("ORACLE_DB_PASSWORD", "")
        dsn = credentials.get("ORACLE_DB_DSN", "")
        wallet_path = credentials.get("ORACLE_DB_WALLET_LOCATION")
        wallet_password = credentials.get("ORACLE_DB_WALLET_PASSWORD")
        
        if not password or not dsn:
            raise ValueError("Oracle DB credentials not found in config.yaml. Please set ORACLE_DB_USERNAME, ORACLE_DB_PASSWORD, and ORACLE_DB_DSN.")

        # Connect to the database
        try:
            if not wallet_path:
                print(f'Connecting (no wallet) to dsn {dsn} and user {username}')
                self.connection = oracledb.connect(user=username, password=password, dsn=dsn)
            else:
                print(f'Connecting (with wallet) to dsn {dsn} and user {username}')
                self.connection = oracledb.connect(user=username, password=password, dsn=dsn, 
                                           config_dir=wallet_path, wallet_location=wallet_path, wallet_password=wallet_password)
            self.cursor = self.connection.cursor()  
             # Initialize cursor
            print("Oracle DB Connection successful!")

        except Exception as e:
            print("Oracle DB Connection failed!", e)
            raise

    def _load_config(self) -> Dict[str, str]:
        """Load configuration from config.yaml"""
        try:
            config_path = Path("config.yaml")
            if not config_path.exists():
                print("Warning: config.yaml not found. Using empty configuration.")
                return {}
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config if config else {}
        except Exception as e:
            print(f"Warning: Error loading config: {str(e)}")
            return {}
    
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
        
        # Get the most recently inserted row (using ID as a proxy for insertion time)
        # This assumes IDs are assigned sequentially or have a timestamp component
        sql = f"SELECT Id, Text, MetaData FROM {table_name} ORDER BY ROWID DESC FETCH FIRST 1 ROW ONLY"
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
    
    @property # used to create a read-only property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding_function
    # The use of a property here allows the user to access the _embedding_function attribute as if it were a property of the class. 
    # For example, if obj is an instance of the class, you can access the embeddings using obj.embeddings instead of obj.embeddings().


    #
    # similarity_search
    #
    def similarity_search(
        self, query: str, k: int = 3, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Return docs most similar to query."""
        logging.info(f"Similarity Search")

        if self.verbose:
            logging.info(f"top_k: {k}")
            logging.info("")

        # 1. embed the query
        embed_query = self._embedding_function(query)
        new_vector = array.array("f", embed_query)
        sql = """
            SELECT Id, Text, MetaData, Embedding
            FROM PDFCOLLECTION
            ORDER BY VECTOR_DISTANCE(EMBEDDING, :nv, EUCLIDEAN) 
            FETCH FIRST 10 ROWS ONLY
            """

        self.cursor.execute(sql, {"nv": new_vector})

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
        
        print(f"🔍 [Oracle DB] Retrieved {len(formatted_results)} chunks from PDF Collection")
        return formatted_results
    
    # A class method is a method that is bound to the class and not the instance of the class. 
    # It takes the class itself as its first parameter (often named cls), and it can be called on the class rather than on an instance of the class.
    @classmethod
    def from_texts(
        cls: Type[OracleDBVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> OracleDBVectorStore:
        logging.info(f"from_texts")
        """Return VectorStore initialized from texts and embeddings."""
        raise NotImplementedError("from_texts method must be implemented...")