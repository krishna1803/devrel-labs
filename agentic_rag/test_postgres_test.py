from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

from sqlalchemy import text, create_engine, inspect
import pandas as pd

# For running this code, you need to have a PostgreSQL instance running with pgvector enabled.
# Use the following command to start a PostgreSQL instance with pgvector:
#sudo docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16

#embeddings = OllamaEmbeddings(model="llama3.3")
#embeddings = SentenceTransformer('all-MiniLM-L12-v2')



model_name = "sentence-transformers/all-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://postgres:RAbbithole1234##@192.168.105.32:5432/postgres"  # Uses psycopg3!
collection_name = "langchain_pg_collection"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)


def add_docs():
    docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="fresh apples are available at the market",
        metadata={"id": 3, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the market also sells fresh oranges",
        metadata={"id": 4, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the new art exhibit is fascinating",
        metadata={"id": 5, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a sculpture exhibit is also at the museum",
        metadata={"id": 6, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a new coffee shop opened on Main Street",
        metadata={"id": 7, "location": "Main Street", "topic": "food"},
    ),
    Document(
        page_content="the book club meets at the library",
        metadata={"id": 8, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="the library hosts a weekly story time for kids",
        metadata={"id": 9, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="a cooking class for beginners is offered at the community center",
        metadata={"id": 10, "location": "community center", "topic": "classes"},
    ),
    ]
    
    # Add to vector store
    vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])
    print("Documents added successfully!")

    
#Function to implement similarity search
def similarity_search(query, k=3):
    results = vector_store.similarity_search(query, k=k)
    print(f"Found {len(results)} results for query '{query}':")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {result.page_content}")
        print(f"Metadata: {result.metadata}")
    return results

#Function to imlement similarity search with a retriever
def similarity_search_with_retriever(query, k=3):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    results = retriever.invoke(query)
    print(f"Found {len(results)} results for query '{query}':")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {result.page_content}")
        print(f"Metadata: {result.metadata}")
    return results

#Function to describe the database tables and show sample rows
def describe_tables(collection_name=collection_name):
    """
    Describe the database tables related to the collection and show the first 5 rows of each table.
    
    Args:
        collection_name: The name of the collection in PGVector
    """
    # Create SQLAlchemy engine from connection string
    engine = create_engine(connection)
    
    # Get schema information about the tables
    inspector = inspect(engine)
    
    # In PGVector, the collection name is used as a prefix for tables
    # Typically, there are two tables: collection_name and collection_name_embedding
    tables = [t for t in inspector.get_table_names() if t.startswith(collection_name)]
    
    print(f"\n==== Tables related to collection '{collection_name}' ====\n")
    
    for table_name in tables:
        print(f"\n--- Table: {table_name} ---")
        
        # Get column information
        columns = inspector.get_columns(table_name)
        print("\nColumns:")
        for column in columns:
            print(f"  {column['name']}: {column['type']}")
        
        # Execute a query to get the first 5 rows
        with engine.connect() as conn:
            query = text(f"SELECT * FROM {table_name} LIMIT 5")
            result = conn.execute(query)
            rows = result.fetchall()
            
            if rows:
                # Convert to pandas DataFrame for better display
                df = pd.DataFrame(rows, columns=result.keys())
                print("\nFirst 5 rows:")
                print(df)
            else:
                print("\nNo data found in this table.")
    
    # Close the engine
    engine.dispose()

def main():
    # add_docs()
    query = "What is case between Naaman v Jaken Properties Australia Pty Limited?"
  
    
    # Describe tables and show sample data
    describe_tables()
    
    #Execute similarity search functions
    print("\nExecuting similarity search functions...\n")
    print("Using similarity_search:")
    similarity_search(query)
    similarity_search_with_retriever(query)
    
        
if __name__ == "__main__":
    main()
   
