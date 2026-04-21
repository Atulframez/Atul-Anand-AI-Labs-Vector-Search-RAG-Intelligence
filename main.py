"""
VectorDB — Python Edition
Implements BruteForce, KD-Tree, and HNSW search algorithms
with a RAG pipeline powered by local Ollama models.

Run:
    pip install fastapi uvicorn requests
    python main.py

Then open http://localhost:8080
"""

import math
import random
import time
import threading
import heapq
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, List

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import requests as http_requests

# =====================================================================
#  CONSTANTS
# =====================================================================

DIMS = 16  # Demo vector dimensions

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)

def manhattan(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

def get_dist_fn(metric: str) -> Callable:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean

# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id:       int
    metadata: str
    category: str
    emb:      list[float]

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: list[VectorItem] = []

    def insert(self, item: VectorItem):
        self.items.append(item)

    def knn(self, q: list[float], k: int, dist_fn: Callable) -> list[tuple[float, int]]:
        scored = [(dist_fn(q, v.emb), v.id) for v in self.items]
        scored.sort()
        return scored[:k]

    def remove(self, id: int):
        self.items = [v for v in self.items if v.id != id]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    __slots__ = ("item", "left", "right")
    def __init__(self, item: VectorItem):
        self.item  = item
        self.left  = None
        self.right = None

class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root = None

    def insert(self, item: VectorItem):
        self.root = self._insert(self.root, item, 0)

    def _insert(self, node: Optional[KDNode], item: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(item)
        ax = depth % self.dims
        if item.emb[ax] < node.item.emb[ax]:
            node.left  = self._insert(node.left,  item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def rebuild(self, items: list[VectorItem]):
        self.root = None
        for item in items:
            self.insert(item)

    def knn(self, q: list[float], k: int, dist_fn: Callable) -> list[tuple[float, int]]:
        # max-heap (negate distance so Python's min-heap acts as max-heap)
        heap = []
        self._knn(self.root, q, k, 0, dist_fn, heap)
        result = [(-d, id_) for d, id_ in heap]
        result.sort()
        return result

    def _knn(self, node: Optional[KDNode], q: list[float], k: int,
             depth: int, dist_fn: Callable, heap: list):
        if node is None:
            return
        d = dist_fn(q, node.item.emb)
        if len(heap) < k:
            heapq.heappush(heap, (-d, node.item.id))
        elif d < -heap[0][0]:
            heapq.heapreplace(heap, (-d, node.item.id))

        ax   = depth % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, q, k, depth + 1, dist_fn, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist_fn, heap)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSW:
    class _Node:
        __slots__ = ("item", "max_lyr", "nbrs")
        def __init__(self, item: VectorItem, max_lyr: int):
            self.item    = item
            self.max_lyr = max_lyr
            self.nbrs: list[list[int]] = [[] for _ in range(max_lyr + 1)]

    def __init__(self, M: int = 16, ef_build: int = 200):
        self.M        = M
        self.M0       = 2 * M
        self.ef_build = ef_build
        self.mL       = 1.0 / math.log(M)
        self.G: dict[int, HNSW._Node] = {}
        self.top_layer = -1
        self.entry_pt  = -1
        self._rng      = random.Random(42)

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, q: list[float], ep: int, ef: int,
                      lyr: int, dist_fn: Callable) -> list[tuple[float, int]]:
        visited = {ep}
        d0      = dist_fn(q, self.G[ep].item.emb)
        # candidates: min-heap, found: max-heap (negated)
        cands = [(d0, ep)]
        found = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            worst   = -found[0][0]
            if len(found) >= ef and cd > worst:
                break
            if lyr >= len(self.G[cid].nbrs):
                continue
            for nid in self.G[cid].nbrs[lyr]:
                if nid in visited or nid not in self.G:
                    continue
                visited.add(nid)
                nd = dist_fn(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-d, id_) for d, id_ in found]
        result.sort()
        return result

    def insert(self, item: VectorItem, dist_fn: Callable):
        id_  = item.id
        lvl  = self._rand_level()
        node = self._Node(item, lvl)
        self.G[id_] = node

        if self.entry_pt == -1:
            self.entry_pt  = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W    = self._search_layer(item.emb, ep, self.ef_build, lc, dist_fn)
            maxM = self.M0 if lc == 0 else self.M
            sel  = [id__ for _, id__ in W[:maxM]]
            node.nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                nbr_node = self.G[nid]
                while len(nbr_node.nbrs) <= lc:
                    nbr_node.nbrs.append([])
                conn = nbr_node.nbrs[lc]
                conn.append(id_)
                if len(conn) > maxM:
                    ds = sorted(
                        (dist_fn(nbr_node.item.emb, self.G[c].item.emb), c)
                        for c in conn if c in self.G
                    )
                    nbr_node.nbrs[lc] = [c for _, c in ds[:maxM]]
            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt  = id_

    def knn(self, q: list[float], k: int, ef: int,
            dist_fn: Callable) -> list[tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(q, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist_fn)
        return W[:k]

    def remove(self, id: int):
        if id not in self.G:
            return
        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                if id in layer:
                    layer.remove(id)
        if self.entry_pt == id:
            self.entry_pt = next(
                (nid for nid in self.G if nid != id), -1
            )
        del self.G[id]

    def get_info(self) -> dict:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes_out = []
        edges_out = []

        for id_, nd in self.G.items():
            nodes_out.append({
                "id":       id_,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr":   nd.max_lyr,
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges_out.append({"src": id_, "dst": nid, "lyr": lc})

        return {
            "topLayer":      self.top_layer,
            "nodeCount":     len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes":         nodes_out,
            "edges":         edges_out,
        }

    def size(self) -> int:
        return len(self.G)

# =====================================================================
#  VECTOR DATABASE  (demo 16D index)
# =====================================================================

class VectorDB:
    def __init__(self, dims: int):
        self.dims   = dims
        self._store: dict[int, VectorItem] = {}
        self._bf    = BruteForce()
        self._kdt   = KDTree(dims)
        self._hnsw  = HNSW(16, 200)
        self._lock  = threading.Lock()
        self._next  = 1

    def insert(self, meta: str, cat: str, emb: list[float],
               dist_fn: Callable) -> int:
        with self._lock:
            item = VectorItem(self._next, meta, cat, emb)
            self._next += 1
            self._store[item.id] = item
            self._bf.insert(item)
            self._kdt.insert(item)
            self._hnsw.insert(item, dist_fn)
            return item.id

    def remove(self, id: int) -> bool:
        with self._lock:
            if id not in self._store:
                return False
            del self._store[id]
            self._bf.remove(id)
            self._hnsw.remove(id)
            self._kdt.rebuild(list(self._store.values()))
            return True

    def search(self, q: list[float], k: int, metric: str,
               algo: str) -> dict:
        with self._lock:
            dist_fn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "bruteforce":
                raw = self._bf.knn(q, k, dist_fn)
            elif algo == "kdtree":
                raw = self._kdt.knn(q, k, dist_fn)
            else:
                raw = self._hnsw.knn(q, k, 50, dist_fn)

            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, id_ in raw:
                if id_ in self._store:
                    v = self._store[id_]
                    hits.append({
                        "id":        v.id,
                        "metadata":  v.metadata,
                        "category":  v.category,
                        "distance":  d,
                        "embedding": v.emb,
                    })
            return {
                "results":   hits,
                "latencyUs": us,
                "algo":      algo,
                "metric":    metric,
            }

    def benchmark(self, q: list[float], k: int, metric: str) -> dict:
        with self._lock:
            dist_fn = get_dist_fn(metric)

            def timed(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)

            return {
                "bruteforceUs": timed(lambda: self._bf.knn(q, k, dist_fn)),
                "kdtreeUs":     timed(lambda: self._kdt.knn(q, k, dist_fn)),
                "hnswUs":       timed(lambda: self._hnsw.knn(q, k, 50, dist_fn)),
                "itemCount":    len(self._store),
            }

    def all(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "id":        v.id,
                    "metadata":  v.metadata,
                    "category":  v.category,
                    "embedding": v.emb,
                }
                for v in self._store.values()
            ]

    def hnsw_info(self) -> dict:
        with self._lock:
            return self._hnsw.get_info()

    def size(self) -> int:
        with self._lock:
            return len(self._store)

# =====================================================================
#  DOCUMENT DATABASE  — HNSW over real Ollama embeddings
# =====================================================================

@dataclass
class DocItem:
    id:    int
    title: str
    text:  str
    emb:   list[float]

class DocumentDB:
    def __init__(self):
        self._store: dict[int, DocItem] = {}
        self._hnsw  = HNSW(16, 200)
        self._bf    = BruteForce()
        self._lock  = threading.Lock()
        self._next  = 1
        self.dims   = 0

    def insert(self, title: str, text: str, emb: list[float]) -> int:
        with self._lock:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self._next, title, text, emb)
            self._next += 1
            self._store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self._hnsw.insert(vi, cosine)
            self._bf.insert(vi)
            return item.id

    def search(self, q: list[float], k: int,
               max_dist: float = 0.7) -> list[tuple[float, DocItem]]:
        with self._lock:
            if not self._store:
                return []
            raw = (
                self._bf.knn(q, k, cosine)
                if len(self._store) < 10
                else self._hnsw.knn(q, k, 50, cosine)
            )
            return [
                (d, self._store[id_])
                for d, id_ in raw
                if id_ in self._store and d <= max_dist
            ]

    def remove(self, id: int) -> bool:
        with self._lock:
            if id not in self._store:
                return False
            del self._store[id]
            self._hnsw.remove(id)
            self._bf.remove(id)
            return True

    def all(self) -> list[DocItem]:
        with self._lock:
            return list(self._store.values())

    def size(self) -> int:
        with self._lock:
            return len(self._store)

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250,
               overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step   = chunk_words - overlap_words
    i      = 0
    while i < len(words):
        end   = min(i + chunk_words, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
        i += step
    return chunks

# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base        = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model   = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = http_requests.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> List[float]:
        try:
            r = http_requests.post(
                f"{self.base}/api/embed",
                json={
                    "model": self.embed_model,
                    "input": text
                },
                timeout=30,
            )
            if r.status_code != 200:
                return []
            return r.json().get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = http_requests.post(
                f"{self.base}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  DEMO DATA  (16D categorical vectors)
# =====================================================================

DEMO_ITEMS = [
    # CS — dims 0–3 hot
    ("Linked List: nodes connected by pointers", "cs",
     [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
    ("Binary Search Tree: O(log n) search and insert", "cs",
     [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
    ("Dynamic Programming: memoization overlapping subproblems", "cs",
     [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
    ("Graph BFS and DFS: breadth and depth first traversal", "cs",
     [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
    ("Hash Table: O(1) lookup with collision chaining", "cs",
     [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
    # Math — dims 4–7 hot
    ("Calculus: derivatives integrals and limits", "math",
     [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
    ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
     [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
    ("Probability: distributions random variables Bayes theorem", "math",
     [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
    ("Number Theory: primes modular arithmetic RSA cryptography", "math",
     [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
    ("Combinatorics: permutations combinations generating functions", "math",
     [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
    # Food — dims 8–11 hot
    ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
    ("Sushi: vinegared rice raw fish and nori rolls", "food",
     [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
    ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
     [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
    ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
     [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
    ("Croissant: laminated pastry with buttery flaky layers", "food",
     [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
    # Sports — dims 12–15 hot
    ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
     [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
    ("Football: tackles touchdowns field goals and strategy", "sports",
     [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
    ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
    ("Chess: openings endgames tactics strategic board game", "sports",
     [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
    ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
     [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
]

def load_demo(db: VectorDB):
    dist_fn = get_dist_fn("cosine")
    for meta, cat, emb in DEMO_ITEMS:
        db.insert(meta, cat, emb, dist_fn)

# =====================================================================
#  FASTAPI APPLICATION
# =====================================================================

app    = FastAPI(title="VectorDB Python")
db     = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HELPERS ──────────────────────────────────────────────────────────

def parse_vec(s: str) -> list[float]:
    try:
        return [float(x) for x in s.split(",") if x.strip()]
    except Exception:
        return []

# ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────────

@app.get("/search")
async def search(v: str = "", k: int = 5,
                 metric: str = "cosine", algo: str = "hnsw"):
    q = parse_vec(v)
    if len(q) != DIMS:
        return JSONResponse({"error": f"need {DIMS}D vector"}, status_code=400)
    return db.search(q, k, metric, algo)


@app.post("/insert")
async def insert(request: Request):
    body = await request.json()
    meta = body.get("metadata", "")
    cat  = body.get("category", "")
    emb  = body.get("embedding", [])
    if not meta or len(emb) != DIMS:
        return JSONResponse({"error": "invalid body"}, status_code=400)
    id_ = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return {"id": id_}


@app.delete("/delete/{id}")
async def delete(id: int):
    ok = db.remove(id)
    return {"ok": ok}


@app.get("/items")
async def items():
    return db.all()


@app.get("/benchmark")
async def benchmark(v: str = "", k: int = 5, metric: str = "cosine"):
    q = parse_vec(v)
    if len(q) != DIMS:
        return JSONResponse({"error": f"need {DIMS}D vector"}, status_code=400)
    return db.benchmark(q, k, metric)


@app.get("/hnsw-info")
async def hnsw_info():
    return db.hnsw_info()


@app.get("/stats")
async def stats():
    return {
        "count":      db.size(),
        "dims":       DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics":    ["euclidean", "cosine", "manhattan"],
    }

# ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────────

@app.post("/doc/insert")
async def doc_insert(request: Request):
    body  = await request.json()
    title = body.get("title", "")
    text  = body.get("text", "")
    if not title or not text:
        return JSONResponse({"error": "need title and text"}, status_code=400)

    chunks = chunk_text(text, 250, 30)
    ids    = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return JSONResponse(
                {"error": "Ollama unavailable. Install from https://ollama.com "
                          "then run: ollama pull nomic-embed-text && ollama pull llama3.2"},
                status_code=503,
            )
        chunk_title = (
            f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
        )
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return {"ids": ids, "chunks": len(chunks), "dims": doc_db.dims}


@app.delete("/doc/delete/{id}")
async def doc_delete(id: int):
    ok = doc_db.remove(id)
    return {"ok": ok}


@app.get("/doc/list")
async def doc_list():
    docs = doc_db.all()
    result = []
    for d in docs:
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        result.append({
            "id":      d.id,
            "title":   d.title,
            "preview": preview,
            "words":   len(d.text.split()),
        })
    return result


@app.post("/doc/search")
async def doc_search(request: Request):
    body     = await request.json()
    question = body.get("question", "")
    k        = int(body.get("k", 3))
    if not question:
        return JSONResponse({"error": "need question"}, status_code=400)

    q_emb = ollama.embed(question)
    if not q_emb:
        return JSONResponse({"error": "Ollama unavailable"}, status_code=503)

    hits = doc_db.search(q_emb, k)
    return {
        "contexts": [
            {"id": item.id, "title": item.title, "distance": round(d, 4)}
            for d, item in hits
        ]
    }


@app.post("/doc/ask")
async def doc_ask(request: Request):
    body     = await request.json()
    question = body.get("question", "")
    k        = int(body.get("k", 3))
    if not question:
        return JSONResponse({"error": "need question"}, status_code=400)

    # Step 1 — embed question
    q_emb = ollama.embed(question)
    if not q_emb:
        return JSONResponse({"error": "Ollama unavailable"}, status_code=503)

    # Step 2 — retrieve top-k chunks
    hits = doc_db.search(q_emb, k)

    # Step 3 — build prompt
    ctx = "".join(
        f"[{i+1}] {item.title}:\n{item.text}\n\n"
        for i, (_, item) in enumerate(hits)
    )
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}"
        f"Question: {question}\n\nAnswer:"
    )

    # Step 4 — generate
    answer = ollama.generate(prompt)

    return {
        "answer":   answer,
        "model":    ollama.gen_model,
        "contexts": [
            {
                "id":       item.id,
                "title":    item.title,
                "text":     item.text,
                "distance": round(d, 4),
            }
            for d, item in hits
        ],
        "docCount": doc_db.size(),
    }


@app.get("/status")
async def status():
    up = ollama.is_available()
    return {
        "ollamaAvailable": up,
        "embedModel":      ollama.embed_model,
        "genModel":        ollama.gen_model,
        "docCount":        doc_db.size(),
        "docDims":         doc_db.dims,
        "demoDims":        DIMS,
        "demoCount":       db.size(),
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path("index.html")
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    load_demo(db)

    ollama_up = ollama.is_available()
    print("=== VectorDB Engine (Python) ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    uvicorn.run(app, host="0.0.0.0", port=8080)
