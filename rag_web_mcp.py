"""
Advanced MCP RAG Server — strict stack (clientless: retrieval-only with `retrieve_context`)

- Tools-first MCP server with structured JSON returns.
- DuckDuckGo HTML search parsed via BeautifulSoup (no regex fallback).
- Safe fetching: SSRF guard, size caps, timeouts, single shared httpx AsyncClient.
- HTML→text via BeautifulSoup only.
- Chunking: word-wise with overlap.
- Embeddings: SentenceTransformers ONLY (default: all-mpnet-base-v2).
- Vector store: Chroma persistent ONLY.
- NO LLM client: your LM Studio / mlx_lm model calls tools and writes answers.
- All CPU-bound work is offloaded to threads to avoid blocking the event loop.

Environment variables (optional):
- MCP_DB_PATH         (default: "./chroma_db")
- MCP_COLLECTION      (default: "documents")
- MCP_EMBED_MODEL     (default: "sentence-transformers/all-mpnet-base-v2")
- MCP_CHUNK_SIZE      (default: 180)   # approx words
- MCP_CHUNK_OVERLAP   (default: 40)
- MCP_MAX_CONCURRENCY (default: 4)
- MCP_FETCH_MAX_BYTES (default: 5_000_000)
- MCP_USER_AGENT      (default: "AdvancedMCP-RAG/1.0 (+https://example.local)")

Run:
    python this_file.py
Then connect via MCP stdio transport from your client (e.g., LM Studio or Jan).

Recommended LM Studio / mlx_lm system prompt (clientless):
> You have a tool `retrieve_context(question, top_k, max_context_chars)`.
> For any question that may require external knowledge, call the tool first.
> Then answer ONLY using the information in <context> and cite sources inline as [source N]
> where N matches the Sources list. If the answer is not present in context, say you don't know.
"""
from __future__ import annotations

import asyncio
import os
import re
import json
import socket
import ipaddress
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from html import unescape
from typing import Any, Dict, List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    raise SystemExit("The 'mcp' package is required. Install with `pip install mcp[cli]`.") from e

# ----------------------- Required dependencies -----------------------
import httpx  # async HTTP
from bs4 import BeautifulSoup  # HTML parsing
import chromadb  # persistent vector store
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# ----------------------------- Configuration -----------------------------------
DB_PATH = os.getenv("MCP_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("MCP_COLLECTION", "documents")
EMBED_MODEL_NAME = os.getenv("MCP_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
CHUNK_SIZE = int(os.getenv("MCP_CHUNK_SIZE", "180"))  # approx words
CHUNK_OVERLAP = int(os.getenv("MCP_CHUNK_OVERLAP", "40"))
MAX_CONCURRENCY = int(os.getenv("MCP_MAX_CONCURRENCY", "10"))
FETCH_MAX_BYTES = int(os.getenv("MCP_FETCH_MAX_BYTES", str(5_000_000)))
USER_AGENT = os.getenv("MCP_USER_AGENT", "Safari/1.0 (+https://example.local)")

# --------------------------- Utilities & helpers -------------------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def is_safe_url(url: str) -> bool:
    """Allow only http(s); block localhost/private/link-local/reserved/multicast targets."""
    if not url.startswith(("http://", "https://")):
        return False
    try:
        host = re.sub(r"^https?://", "", url).split("/", 1)[0]
        hostname = host.split(":")[0]
        # If IP literal, check directly; else resolve IPv4 best-effort.
        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            ip_str = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(ip_str)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    except Exception:
        return False
    return True

def ddg_unwrap(url: str) -> str:
    # DuckDuckGo often uses /l/?uddg=<encoded_url>
    try:
        m = re.search(r"[?&]uddg=([^&]+)", url)
        if m:
            from urllib.parse import unquote
            return unquote(m.group(1))
    except Exception:
        pass
    return url

def clean_html_to_text(html: str) -> str:
    """HTML→text using BeautifulSoup only (no regex fallback)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(chunk.strip() for chunk in soup.stripped_strings)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_text_into_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Approx word-based chunking with overlap."""
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    i = 0
    step = max(1, size - overlap)
    while i < len(words):
        chunk_words = words[i:i+size]
        chunks.append(" ".join(chunk_words))
        i += step
    return chunks

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

async def to_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

# ------------------------- Embedding backend (strict) --------------------------
class EmbeddingBackend:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.kind = "sentence-transformers"
        self.model_name = model_name
        try:
            self._st_model = SentenceTransformer(model_name)
        except Exception as e:
            raise SystemExit(
                f"SentenceTransformers must load model '{model_name}'. "
                f"Install with `pip install sentence-transformers torch`. Error: {e}"
            ) from e

    async def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return await to_thread(self._st_model.encode, texts, show_progress_bar=False, convert_to_numpy=False)

# --------------------------- Vector store backends -----------------------------
@dataclass
class DocMeta:
    url: Optional[str] = None
    title: Optional[str] = None
    fetched_at: Optional[str] = None
    source: Optional[str] = "web"
    checksum: Optional[str] = None           # document-level checksum (full text)
    chunk_checksum: Optional[str] = None     # chunk-level checksum
    chunk_index: Optional[int] = None
    embedding_model: Optional[str] = None

class VectorStore:
    def upsert(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]): ...
    def query(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]: ...
    def count(self) -> int: ...
    def list(self, offset: int = 0, limit: int = 100) -> List[str]: ...
    def get(self, id: str) -> Optional[str]: ...
    def delete(self, id: str) -> bool: ...
    def delete_by_url(self, url: str) -> int: ...
    def exists_by_doc_checksum(self, checksum: str) -> bool: ...

class ChromaStore(VectorStore):
    def __init__(self, path: str = DB_PATH, collection: str = COLLECTION_NAME):
        try:
            client = PersistentClient(path=path)
        except Exception as e:
            raise SystemExit(
                f"Could not initialize Chroma persistent client at {path}. "
                f"Install with `pip install chromadb` and ensure write access. Error: {e}"
            ) from e
        self.col = client.get_or_create_collection(collection)

    def upsert(self, ids, embeddings, documents, metadatas):
        self.col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, query_embedding, top_k=3):
        res = self.col.query(
            query_embeddings=[query_embedding],
            n_results=int(top_k),
            include=["documents", "metadatas", "distances", "ids"],
        )
        results: List[Dict[str, Any]] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0]
        for i in range(len(docs)):
            results.append(
                {"id": ids[i], "text": docs[i], "score": float(dists[i]), "meta": metas[i]}
            )
        return results

    def count(self) -> int:
        try:
            return int(self.col.count())
        except Exception:
            return 0

    def list(self, offset=0, limit=100) -> List[str]:
        try:
            got = self.col.get(include=["ids"], limit=limit, offset=offset)
            return list(got.get("ids", []))
        except Exception:
            return []

    def get(self, id: str) -> Optional[str]:
        try:
            got = self.col.get(ids=[id])
            docs = got.get("documents", [])
            if docs and docs[0]:
                return docs[0]
        except Exception:
            pass
        return None

    def delete(self, id: str) -> bool:
        try:
            self.col.delete(ids=[id])
            return True
        except Exception:
            return False

    def delete_by_url(self, url: str) -> int:
        try:
            got = self.col.get(where={"url": url}, include=["ids"])
            ids = list(got.get("ids", [])) if isinstance(got.get("ids", []), list) else []
            if ids:
                self.col.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def exists_by_doc_checksum(self, checksum: str) -> bool:
        try:
            got = self.col.get(where={"checksum": checksum}, include=["ids"], limit=1)
            ids = got.get("ids", [])
            return bool(ids)
        except Exception:
            return False

# --------------------------- HTTP client (shared) ------------------------------
_http_client: Optional["httpx.AsyncClient"] = None

def get_http_client() -> "httpx.AsyncClient":  # type: ignore
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"User-Agent": USER_AGENT},
        )
    return _http_client

# --------------------------- RAG orchestrator ----------------------------------
class RAGEngine:
    def __init__(self):
        # Embedding backend (required: SentenceTransformers)
        self.embedder = EmbeddingBackend(EMBED_MODEL_NAME)
        # Vector store (required: Chroma persistent)
        try:
            self.store: VectorStore = ChromaStore(DB_PATH, COLLECTION_NAME)
            self.store_kind = "chroma"
        except Exception as e:
            raise SystemExit(
                f"ChromaDB is required. Install with `pip install chromadb` and ensure write access to {DB_PATH}. "
                f"Underlying error: {e}"
            ) from e

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """DuckDuckGo HTML search parsed via BeautifulSoup (no fallback)."""
        client = get_http_client()
        params = {"q": query}
        resp = await client.get("https://duckduckgo.com/html/", params=params)
        resp.raise_for_status()
        html = resp.text
        results: List[Dict[str, str]] = []
        urls_seen = set()
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("a.result__a")[: max(1, min(limit, 10))]:
            href = a.get("href", "")
            url = ddg_unwrap(href)
            title = a.get_text(strip=True)
            if url.startswith("http") and url not in urls_seen:
                results.append({"title": title, "url": url})
                urls_seen.add(url)
        return results[: max(1, min(limit, 10))]

    async def fetch_page(self, url: str) -> Dict[str, Any]:
        """Fetch a page safely and return {ok, url, status, title, text, error}."""
        if not is_safe_url(url):
            return {"ok": False, "error": "URL not allowed"}
        client = get_http_client()
        try:
            resp = await client.get(url, timeout=httpx.Timeout(30.0, connect=10.0))  # type: ignore
            resp.raise_for_status()
        except Exception as exc:
            return {"ok": False, "error": f"Fetch failed: {exc}"}
        # Reject overly large bodies
        if resp.headers.get("content-length") and int(resp.headers.get("content-length", "0")) > FETCH_MAX_BYTES:
            return {"ok": False, "error": "Response too large"}
        content = resp.text
        if len(content.encode("utf-8", errors="ignore")) > FETCH_MAX_BYTES:
            return {"ok": False, "error": "Response too large"}
        title = None
        try:
            soup = BeautifulSoup(content, "html.parser")
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
        except Exception:
            pass
        return {
            "ok": True,
            "url": str(resp.url),
            "status": resp.status_code,
            "title": title,
            "text": clean_html_to_text(content),
        }

    async def ingest_text(self, base_id: str, text: str, meta: DocMeta) -> Dict[str, Any]:
        # Dedup on document-level checksum (skip if identical content already ingested)
        doc_checksum = meta.checksum or sha256(text)
        if self.store.exists_by_doc_checksum(doc_checksum):
            return {"ok": True, "ingested_chunks": 0, "skipped": True, "reason": "duplicate checksum"}
        chunks = split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            return {"ok": False, "error": "Empty content"}
        embs = await self.embedder.encode(chunks)
        ids = [f"{base_id}::{i}" for i in range(len(chunks))]
        metas = []
        for i, ch in enumerate(chunks):
            m = asdict(meta).copy()
            m.update({
                "chunk_index": i,
                "embedding_model": self.embedder.model_name,
                "checksum": doc_checksum,
                "chunk_checksum": sha256(ch),
            })
            metas.append(m)
        self.store.upsert(ids=ids, embeddings=embs, documents=chunks, metadatas=metas)
        return {"ok": True, "ingested_chunks": len(chunks), "skipped": False}

    async def ingest_url(self, doc_id: str, url: str, replace: bool = False) -> Dict[str, Any]:
        fetched = await self.fetch_page(url)
        if not fetched.get("ok"):
            return fetched
        text = fetched.get("text", "")
        meta = DocMeta(
            url=fetched.get("url"),
            title=fetched.get("title"),
            fetched_at=now_utc_iso(),
            source="web",
            checksum=sha256(text),
        )
        if replace:
            # Best-effort replacement: delete any previous chunks from same URL
            try:
                self.store.delete_by_url(meta.url or url)
            except Exception:
                pass
        return await self.ingest_text(doc_id, text, meta)

    async def search_and_ingest(self, query: str, limit: int = 3) -> Dict[str, Any]:
        results = await self.search(query, limit)
        sem = asyncio.Semaphore(MAX_CONCURRENCY)
        ingested = 0
        errors: List[str] = []

        async def one(idx: int, item: Dict[str, str]):
            nonlocal ingested
            async with sem:
                san = re.sub(r'\W+', '_', query).strip('_')
                did = f"{san}#{idx+1}"
                res = await self.ingest_url(did, item.get("url", ""))
                if res.get("ok"):
                    ingested += res.get("ingested_chunks", 0)
                else:
                    errors.append(res.get("error", "unknown error"))

        tasks = [one(i, r) for i, r in enumerate(results)]
        if tasks:
            await asyncio.gather(*tasks)
        return {"ok": True, "search_results": len(results), "ingested_chunks": ingested, "errors": errors}

    async def query(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        vec = (await self.embedder.encode([text]))[0]
        hits = self.store.query(vec, top_k=top_k)
        return {"ok": True, "results": hits}

    async def retrieve(self, question: str, top_k: int = 5, max_context_chars: int = 3000) -> Dict[str, Any]:
        """Build a single packed context block (sources list + passages), ready for LLM use."""
        q = await self.query(question, top_k=top_k)
        if not q.get("ok"):
            return q
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        total = 0
        for i, item in enumerate(q["results"], start=1):
            txt = item.get("text", "")
            meta = item.get("meta", {}) or {}
            if total + len(txt) > max_context_chars:
                break
            title = meta.get("title") or meta.get("url") or f"chunk {i}"
            context_parts.append(f"[source {i}] {title}\n{txt}")
            sources.append({
                "n": i,
                "id": item.get("id"),
                "url": meta.get("url"),
                "title": meta.get("title"),
                "score": item.get("score"),
            })
            total += len(txt)
        if not context_parts:
            return {"ok": False, "error": "No context available"}
        mapping_lines = []
        for s in sources:
            line_title = s.get("title") or s.get("url") or f"chunk {s['n']}"
            line_url = s.get("url") or ""
            mapping_lines.append(f"[source {s['n']}] {line_title} — {line_url}")
        sources_map = "\n".join(mapping_lines)
        context_block = (
            "Sources:\n" + sources_map + "\n\n" +
            "Context Passages:\n" + "\n\n".join(context_parts)
        )
        return {"ok": True, "context": context_block, "sources": sources, "query": question}

    def status(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "mode": "clientless-retrieval",
            "embedding_backend": "sentence-transformers",
            "embedding_model": EMBED_MODEL_NAME,
            "vector_store": getattr(self, "store_kind", "chroma"),
            "doc_chunks": self.store.count(),
            "db_path": DB_PATH,
            "collection": COLLECTION_NAME,
        }

    def list_doc_ids(self, offset: int = 0, limit: int = 100) -> List[str]:
        return self.store.list(offset=offset, limit=limit)

    def get_document(self, doc_id: str) -> Optional[str]:
        return self.store.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        return self.store.delete(doc_id)

    def delete_by_url(self, url: str) -> int:
        return self.store.delete_by_url(url)

# --------------------------------- MCP setup -----------------------------------
mcp = FastMCP("advanced_no_key_rag_server")
engine = RAGEngine()

@mcp.tool()
async def web_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """Perform a DuckDuckGo search (no API key). Returns list of {title, url}."""
    limit = max(1, min(limit, 10))
    return await engine.search(query, limit)

@mcp.tool()
async def fetch_page(url: str) -> Dict[str, Any]:
    """Fetch a page and return {ok, url, status, title, text, error}."""
    return await engine.fetch_page(url)

@mcp.tool()
async def ingest_url(doc_id: str, url: str, replace: bool = False) -> Dict[str, Any]:
    """Fetch content from URL and add chunked text to the knowledge base with metadata.

    Set replace=True to delete existing chunks for this URL before ingesting."""
    return await engine.ingest_url(doc_id, url, replace=replace)

@mcp.tool()
async def add_document(doc_id: str, content: str, url: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
    """Add raw text as chunked documents. Optional url/title metadata (stored in chunk metadata)."""
    meta = DocMeta(url=url, title=title, fetched_at=now_utc_iso(), source="manual", checksum=sha256(content))
    return await engine.ingest_text(doc_id, content, meta)

@mcp.tool()
async def search_and_ingest(query: str, limit: int = 3) -> Dict[str, Any]:
    """Search the web and ingest the top results. Returns counts and errors."""
    return await engine.search_and_ingest(query, limit)

@mcp.tool()
async def query_knowledge_base(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Vector search over the knowledge base. Returns [{id, text, score, meta}]."""
    return await engine.query(query, top_k=top_k)

def _summarize(text: str, n_sentences: int = 3, max_chars: int = 800) -> str:
    # Simple heuristic: first N sentence-like splits, clipped to max_chars
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(sents[: max(1, n_sentences)])
    return out[:max_chars]

@mcp.tool()
async def summarize_document(doc_id: str, n_sentences: int = 3) -> Dict[str, Any]:
    """Summarize the specified document chunk by id."""
    doc = engine.get_document(doc_id)
    if not doc:
        return {"ok": False, "error": "Document not found"}
    return {"ok": True, "summary": _summarize(doc, n_sentences=n_sentences)}

@mcp.tool()
async def server_status() -> Dict[str, Any]:
    """Return status about backends, counts, and configuration."""
    return engine.status()

@mcp.tool()
async def list_documents(offset: int = 0, limit: int = 100) -> Dict[str, Any]:
    """List document chunk IDs with pagination."""
    ids = engine.list_doc_ids(offset=offset, limit=limit)
    return {"ok": True, "ids": ids, "offset": offset, "limit": limit}

@mcp.tool()
async def delete_document(doc_id: str) -> Dict[str, Any]:
    """Delete a single document chunk by id."""
    ok = engine.delete_document(doc_id)
    return {"ok": ok}

@mcp.tool()
async def delete_by_url(url: str) -> Dict[str, Any]:
    """Delete all document chunks with metadata.url == url."""
    count = engine.delete_by_url(url)
    return {"ok": True, "deleted": count}

@mcp.tool()
async def retrieve_context(question: str, top_k: int = 5, max_context_chars: int = 3000) -> Dict[str, Any]:
    """Return grounded context + source map for the question (no model call).

    Response:
    - ok: bool
    - context: str   # single packed block: 'Sources' list + 'Context Passages'
    - sources: [{n, id, url, title, score}]
    - query: str
    - error?: str
    """
    return await engine.retrieve(question, top_k=top_k, max_context_chars=max_context_chars)

if __name__ == "__main__":
    # stdio transport for desktop hosts like LM Studio or Jan
    mcp.run(transport="stdio")
