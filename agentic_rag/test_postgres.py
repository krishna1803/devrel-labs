from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from sqlalchemy import text

# For running this code, you need to have a PostgreSQL instance running with pgvector enabled.
# Use the following command to start a PostgreSQL instance with pgvector:
#sudo docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16

embeddings = OllamaEmbeddings(model="llama3.3")

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "my_docs"

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


def main():
    add_docs()
    query = "What animals are found in the pond?"
    similarity_search(query)
    similarity_search_with_retriever(query)
    
        
        
if __name__ == "__main__":
    main()        
