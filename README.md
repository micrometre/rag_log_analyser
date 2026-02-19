# RAG Quiz with Ollama and ChromaDB

An interactive quiz application powered by Retrieval-Augmented Generation (RAG) using Ollama for embeddings and language models, with ChromaDB as the vector database.

## Overview

This project demonstrates a practical RAG application that:
- Indexes educational documents using embeddings
- Generates quiz questions from a knowledge base
- Validates user answers using AI with retrieved context
- Provides personalized feedback based on performance

## Prerequisites

- Python 3.7+
- [Ollama](https://ollama.ai/) installed and running locally
- Required Ollama models pulled:
  ```bash
  ollama pull nomic-embed-text
  ollama pull phi3:mini
  ```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/micrometre/rag-quiz
   cd rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure Ollama is running with the required models.

## Usage

### Quiz Mode (Main Feature)

Run the interactive quiz:
```bash
python quiz_example.py
```

The quiz will:
1. Index educational documents about Python, ML, AI, and more
2. Ask you questions based on the knowledge base
3. Validate your answers using RAG (retrieves context + AI evaluation)
4. Track your score and provide personalized feedback

### Basic RAG Example

For a simpler RAG demonstration:
```bash
python rag_example.py
```

This script shows the core RAG workflow: indexing documents, querying, and generating answers.

## How It Works

1. **Document Indexing**: Educational content is converted to embeddings using `nomic-embed-text` and stored in ChromaDB
2. **Question Processing**: Each quiz question retrieves relevant context from the knowledge base
3. **Answer Validation**: User answers are evaluated by `phi3:mini` using the retrieved context
4. **Scoring & Feedback**: The system tracks correct answers and generates personalized feedback

## Customization

- **Change the LLM**: Replace `phi3:mini` with any Ollama model (e.g., `llama3.2`, `mistral`)
- **Add more documents**: Extend the `documents` list with your own educational content
- **Create custom quizzes**: Modify the `quiz_questions` list to create your own quizzes
- **Adjust difficulty**: Change `n_results` to retrieve more or fewer context documents

## Dependencies

- `ollama`: Python client for Ollama
- `chromadb`: Vector database for storing and querying embeddings

## License

MIT

## Contributing

Feel free to open issues or submit pull requests with improvements!
