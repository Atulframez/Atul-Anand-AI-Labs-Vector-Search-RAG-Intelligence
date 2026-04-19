# VectorDB — Python Edition

Built entirely in Python.  
Implements HNSW, KD-Tree, and Brute Force from scratch without external vector libraries.  
Includes a RAG pipeline powered by local Ollama models.

---

## Quick Start

### 1 — Install dependencies

```bash
pip install fastapi uvicorn requests
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 2 — Install Ollama (for RAG / document embedding)

1. Download from **https://ollama.com**
2. Pull the required models:

```bash
ollama pull nomic-embed-text   # ~274 MB — embedding model
ollama pull llama3.2           # ~2 GB — language model
```

### 3 — Run the server

Make sure `main.py` and `index.html` are in the same folder, then:

```bash
python main.py
```

You should see:

```
=== VectorDB Engine (Python) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Open **http://localhost:8080** in your browser.

---

## What's Inside

| Module | Description |
|--------|-------------|
| `BruteForce` | O(N·d) exact search — baseline |
| `KDTree` | O(log N) exact, axis-aligned space partitioning |
| `HNSW` | O(log N) approximate, multilayer small-world graph |
| `VectorDB` | Unified interface over all 3 (16D demo vectors) |
| `DocumentDB` | HNSW-only index for real Ollama embeddings (768D) |
| `OllamaClient` | HTTP client → `/api/embeddings` + `/api/generate` |
| `chunk_text()` | Sliding-window text chunker (250 words, 30 overlap) |
| FastAPI app | Full REST API + serves `index.html` |

---

## REST API

### Demo Vectors

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/{id}` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure |
| `GET` | `/stats` | DB statistics |

### Documents & RAG

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed & store document |
| `GET` | `/doc/list` | — | List stored documents |
| `DELETE` | `/doc/delete/{id}` | — | Delete document chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status |

---

## Project Structure

```
VectorDB-Python/
├── main.py          ← Python backend (all algorithms + REST API + RAG)
├── index.html       ← Frontend (unchanged from C++ version)
├── requirements.txt ← pip dependencies
└── README.md        ← This file
```

---

## Differences from C++ Version

| Aspect | C++ | Python |
|--------|-----|--------|
| Compiler / runtime | g++ (MSYS2) | Python 3.10+ |
| HTTP server | cpp-httplib (header-only) | FastAPI + Uvicorn |
| HTTP client | cpp-httplib | requests |
| Concurrency | `std::mutex` | `threading.Lock()` |
| Dependencies | httplib.h (bundled) | fastapi, uvicorn, requests |
| Performance | ~10–100× faster | Slower, but much easier to read |
| Setup | Compile step required | Just `python main.py` |

The Python version is **significantly easier to modify and extend** — perfect for learning and experimentation.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Ollama: OFFLINE` | Run `ollama serve` in a terminal |
| Port 8080 in use | Change `port=8080` at the bottom of `main.py` |
| Slow LLM answers | Switch to `llama3.2:1b` — change `gen_model` in `OllamaClient.__init__` |
| `ModuleNotFoundError` | Run `pip install fastapi uvicorn requests` |

---

## License
Copyright (c) 2026 Atul Anand

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

**Atul Anand**  
BCA (Hons.) Research, Amity University Noida  

- 💻 AI & ML Enthusiast  
- 🚀 Building Vector Databases & RAG Systems  
- 🔗 GitHub: https://github.com/Atulframez?tab=overview&from=2026-04-01&to=2026-04-19 

