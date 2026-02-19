import ollama
import chromadb
from chromadb.utils import embedding_functions

# 1. Setup ChromaDB (Local Storage)
client = chromadb.PersistentClient(path="./log_vector_db")
collection = client.get_or_create_collection(name="system_logs")

def process_logs(log_lines):
    for i, line in enumerate(log_lines):
        # 2. Summarize the log line (or chunk) using Llama 3.2
        # This cleans the noise (timestamps/IDs) before embedding
        summary_resp = ollama.generate(
            model='llama3.2:3b', 
            prompt=f"Summarize this log line for semantic indexing: {line}"
        )
        summary = summary_resp['response']

        # 3. Generate Embedding using Nomic
        # Use 'search_document' prefix as recommended by Nomic
        embed_resp = ollama.embeddings(
            model='nomic-embed-text',
            prompt=f"search_document: {summary}"
        )
        
        # 4. Store in ChromaDB
        collection.add(
            ids=[f"log_{i}"],
            embeddings=[embed_resp['embedding']],
            documents=[line],       # Store raw log for retrieval
            metadatas=[{"summary": summary}]
        )
    print(f"Processed {len(log_lines)} log entries.")

def ask_logs(query):
    # Embed the query with 'search_query' prefix
    query_embed = ollama.embeddings(
        model='nomic-embed-text',
        prompt=f"search_query: {query}"
    )['embedding']

    # Retrieve top 3 matches
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=3
    )
    return results

# --- Example Usage ---
logs = [
    "Feb 17 12:01:04 kernel: [123.45] oom-kill: process 992 (python3) score 500",
    "Feb 17 12:05:22 systemd[1]: Started Network Time Service."
]

process_logs(logs)
print(ask_logs("Is the system running out of memory?"))