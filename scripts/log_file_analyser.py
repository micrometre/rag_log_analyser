import subprocess
import ollama
import chromadb
import argparse
import sys

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

def process_historical_file(file_path):
    print(f"📂 Reading historical file: {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # To avoid overloading the GPU, process in smaller batches
        for line in lines:
            if line.strip():
                process_and_store(line.strip())

def main():
    parser = argparse.ArgumentParser(
        description="Process historical log files and index them in ChromaDB"
    )
    parser.add_argument(
        "file_path",
        help="Path to the log file to process"
    )
    
    args = parser.parse_args()
    
    try:
        process_historical_file(args.file_path)
        print(f"\n✅ Successfully processed {args.file_path}")
    except FileNotFoundError:
        print(f"❌ Error: File '{args.file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
