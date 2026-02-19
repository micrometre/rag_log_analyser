#!/usr/bin/env python3
"""
Minimal RAG example using Ollama and ChromaDB
"""

import ollama
import chromadb

def main():
    # Initialize ChromaDB client
    client = chromadb.Client()
    
    # Create or get collection
    collection = client.get_or_create_collection(name="documents")
    
    # Sample documents to index
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence.",
        "RAG combines retrieval and generation for better AI responses.",
        "Ollama allows running large language models locally on your machine."
    ]
    
    print("Indexing documents...")
    # Add documents with embeddings
    for i, doc in enumerate(documents):
        # Get embedding from Ollama
        response = ollama.embeddings(model="nomic-embed-text", prompt=doc)
        embedding = response["embedding"]
        
        # Store in ChromaDB
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[doc]
        )
    
    print("Documents indexed!\n")
    
    # Query the RAG system
    query = "What is RAG?"
    print(f"Query: {query}\n")
    
    # Get query embedding
    query_response = ollama.embeddings(model="nomic-embed-text", prompt=query)
    query_embedding = query_response["embedding"]
    
    # Retrieve relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    
    # Build context from retrieved documents
    context = "\n".join(results["documents"][0])
    print(f"Retrieved Context:\n{context}\n")
    
    # Generate response using Ollama LLM
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    print("Generating response...\n")
    response = ollama.generate(model="phi3:mini", prompt=prompt)
    
    print(f"Answer: {response['response']}")


if __name__ == "__main__":
    main()
