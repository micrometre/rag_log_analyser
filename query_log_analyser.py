import subprocess
import ollama
import chromadb

# 1. Setup ChromaDB (Local Storage)
client = chromadb.PersistentClient(path="./log_vector_db")
collection = client.get_or_create_collection(name="system_logs")

def query_the_brain(question):
    # 1. Embed the user's question
    query_embed = ollama.embeddings(
        model='nomic-embed-text',
        prompt=f"search_query: {question}"
    )['embedding']

    # 2. Search ChromaDB for the most similar logs
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=25
    )

    print(f"\n🔎 Results for: '{question}'")
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        print(f"--- \nSummary: {metadata['summary']}\nRaw: {doc}")

# Example usage:
query_the_brain("Was there any suspicious SSH activity last week?") 