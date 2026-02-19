import subprocess
import ollama
import chromadb

# 1. Setup ChromaDB (Local Storage)
client = chromadb.PersistentClient(path="./log_vector_db")
collection = client.get_or_create_collection(name="system_logs")

def process_and_store(raw_log_line):
    """Summarizes and embeds a single log line."""
    # Step A: Summarize with Llama 3.2 (The 'Cleaner')
    summary_resp = ollama.generate(
        model='llama3.2:3b', 
        prompt=f"Summarize this log line for semantic indexing: {raw_log_line}"
    )
    summary = summary_resp['response']

    # Step B: Embed with Nomic (The 'Memory')
    embed_resp = ollama.embeddings(
        model='nomic-embed-text',
        prompt=f"search_document: {summary}"
    )
    
    # Step C: Save to Vector DB
    collection.add(
        ids=[str(hash(raw_log_line))], # Unique ID based on content
        embeddings=[embed_resp['embedding']],
        documents=[raw_log_line],
        metadatas=[{"summary": summary}]
    )
    print(f"✅ Indexed: {summary}")

# --- 2. THE INPUT ENGINE ---
# This command 'tails' the system log in real-time
cmd = ["journalctl", "-f", "-n", "5"] # -f is 'follow', -n 5 starts with last 5 lines

print("🧠 Log Brain is now listening to your system...")

try:
    # Open the process and read line-by-line
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:
            if line.strip():
                process_and_store(line.strip())
except KeyboardInterrupt:
    print("\nStopping the Brain...")