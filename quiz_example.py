#!/usr/bin/env python3
"""
Quiz example using RAG with Ollama and ChromaDB
Generates quiz questions from documents and validates answers
"""

import ollama
import chromadb

def main():
    # Initialize ChromaDB client
    client = chromadb.Client()
    
    # Create or get collection
    collection = client.get_or_create_collection(name="quiz_documents")
    
    # Sample documents to index (educational content)
    documents = [
        "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to produce more accurate and contextual AI responses.",
        "Ollama allows running large language models locally on your machine, providing privacy and offline capability.",
        "ChromaDB is an open-source vector database designed for storing and querying embeddings efficiently.",
        "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes called neurons.",
        "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language."
    ]
    
    print("Indexing documents for quiz...\n")
    # Add documents with embeddings
    for i, doc in enumerate(documents):
        response = ollama.embeddings(model="nomic-embed-text", prompt=doc)
        embedding = response["embedding"]
        
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[doc]
        )
    
    print("=" * 60)
    print("QUIZ MODE: Answer questions based on the knowledge base")
    print("=" * 60)
    print()
    
    # Quiz questions
    quiz_questions = [
        {
            "question": "Who created Python and in what year?",
            "topic": "Python programming language creator"
        },
        {
            "question": "What does RAG stand for and what does it do?",
            "topic": "RAG meaning and purpose"
        },
        {
            "question": "What is machine learning?",
            "topic": "machine learning definition"
        }
    ]
    
    score = 0
    total = len(quiz_questions)
    
    for idx, quiz in enumerate(quiz_questions, 1):
        print(f"Question {idx}/{total}: {quiz['question']}")
        
        # Get relevant context for the question
        query_response = ollama.embeddings(model="nomic-embed-text", prompt=quiz['topic'])
        query_embedding = query_response["embedding"]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
        
        context = "\n".join(results["documents"][0])
        
        # Get user's answer
        user_answer = input("Your answer: ")
        print()
        
        # Validate answer using LLM with context
        validation_prompt = f"""Based on the following context, evaluate if the user's answer is correct.

Context:
{context}

Question: {quiz['question']}
User's Answer: {user_answer}

Respond with ONLY "CORRECT" or "INCORRECT" followed by a brief explanation.
If correct, award the point. If incorrect, provide the correct answer."""
        
        print("Checking answer...")
        response = ollama.generate(model="phi3:mini", prompt=validation_prompt)
        result = response['response']
        
        print(f"Result: {result}\n")
        
        # Simple scoring based on LLM response
        if "CORRECT" in result.upper().split('\n')[0]:
            score += 1
            print("✓ Point awarded!\n")
        else:
            print("✗ No point\n")
        
        print("-" * 60)
        print()
    
    # Final score
    print("=" * 60)
    print(f"QUIZ COMPLETE! Your score: {score}/{total} ({score/total*100:.0f}%)")
    print("=" * 60)
    
    # Generate personalized feedback
    feedback_prompt = f"""The user scored {score} out of {total} on a quiz about Python, machine learning, and AI concepts.
Provide brief encouraging feedback (2-3 sentences) based on their performance."""
    
    feedback_response = ollama.generate(model="llama3.2:3b", prompt=feedback_prompt)
    print(f"\nFeedback: {feedback_response['response']}")


if __name__ == "__main__":
    main()
