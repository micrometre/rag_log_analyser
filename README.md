# RAG Log Analyser

A **Retrieval-Augmented Generation (RAG)** application for intelligent system log analysis using local LLMs and vector embeddings.

## Overview

This project implements a **Summary-based RAG architecture** that overcomes the challenges of analyzing highly structured and noisy log data. Instead of embedding raw logs directly, we:





1. **Summarize** log entries using Llama 3.2 (3B) for cleaner, semantic content
2. **Embed** summaries using Nomic embeddings for high-quality vector representations
3. **Store** vectors in ChromaDB for fast, persistent retrieval
4. **Query** logs intelligently by finding semantically similar entries

## Architecture

```
System Logs
    ↓
[Summarization] Llama 3.2 (3B) → Clean summaries
    ↓
[Embedding] Nomic-embed-text → Vector representations
    ↓
[Vector Store] ChromaDB → Persistent storage
    ↓
[Query] Semantic search → Relevant results
```

## Features

- ✅ Real-time log processing with `live_log_analyser.py`
- ✅ Semantic search queries with `query_log_analyser.py`
- ✅ Lightweight local vector database (ChromaDB)
- ✅ Offline operation - no external APIs required
- ✅ Efficient small model stack (3B LLM + embeddings)

## Requirements

- Python 3.8+
- Ollama (for running local models)
- Models: `llama3.2:3b` and `nomic-embed-text`

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rag_log_analyser
   ```

2. **Install Ollama:**
   Download from [ollama.ai](https://ollama.ai) and install.

3. **Pull required models:**
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Real-time Log Analysis

Monitor system logs in real-time and automatically index them:


```bash
python live_log_analyser.py
```

This script:
- Tails system logs using `journalctl`
- Summarizes each log entry
- Embeds summaries with Nomic
- Stores in ChromaDB vector database

### Query Logs

Search for relevant logs using natural language:


```bash
python query_log_analyser.py
```

Example queries:
- "Was there any suspicious SSH activity?"
- "What network errors occurred?"
- "Show me all authentication failures"

## Project Structure

```
rag_log_analyser/
├── live_log_analyser.py      # Real-time log processing
├── query_log_analyser.py     # Query interface
├── requirements.txt          # Python dependencies
├── log_vector_db/           # ChromaDB storage (auto-created)
└── scripts/                 # Example scripts
```

## How It Works

### 1. Ingestion (`live_log_analyser.py`)
- Reads system logs from `journalctl`
- Processes each log line as it arrives
- Prevents duplicate processing with content-based hashing

### 2. Processing Pipeline
```python
# For each log line:
1. Summarize:    Llama 3.2 condenses noise
2. Embed:        Nomic creates semantic vector
3. Store:        ChromaDB saves for retrieval
```

### 3. Retrieval (`query_log_analyser.py`)
- User asks a natural language question
- Question is embedded using Nomic
- ChromaDB returns semantically similar summaries
- Results include both clean summary and raw log entry

## Configuration

### Adjust Log Source
Edit `live_log_analyser.py` to change log source:
```python
cmd = ["journalctl", "-f", "-n", "5"]  # Change this line
# Or use: ["tail", "-f", "/var/log/syslog"]
```

### Tune Vector Search
In `query_log_analyser.py`, adjust the number of results:
```python
n_results=25  # Change this number
```

## Performance Notes

- **Llama 3.2 (3B)**: ~50-100ms per log entry on CPU
- **Nomic Embeddings**: ~5-10ms per entry
- **ChromaDB**: Instant (~1ms) for vector similarity search
- Total indexing time per log: ~100-150ms

## Troubleshooting




**Models not found:**
```bash
ollama list  # Check installed models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

**ChromaDB persistence issues:**
- Database stored in `./log_vector_db/`
- Safe to delete and recreate (will re-index from live logs)

**Performance slow:**
- Ensure Ollama is running: `ollama serve`
- Check system RAM (3B model needs ~4GB)
- Use GPU if available for faster inference

## Future Improvements

- [ ] Multi-model comparison (other LLMs)
- [ ] Advanced filtering and time-based queries
- [ ] Web UI for search interface
- [ ] Log pattern anomaly detection
- [ ] Batch log file ingestion

## License

MIT License