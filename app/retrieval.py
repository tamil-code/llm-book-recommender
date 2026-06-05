"""Semantic book retrieval backed by a persisted Chroma vector store.

Resources (the embedding model, the Chroma index, and the book catalog) are
loaded lazily once and reused across searches. ``search`` returns ordered book
records enriched with metadata from ``cleaned_books.csv``.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = PROJECT_ROOT / "chroma_db"
CATALOG_CSV = PROJECT_ROOT / "cleaned_books.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "books"


@functools.lru_cache(maxsize=1)
def _load_resources() -> tuple[Chroma, pd.DataFrame]:
    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"Vector index not found at {PERSIST_DIR}. Build it first with "
            "`python scripts/build_index.py`."
        )

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(PERSIST_DIR),
    )

    catalog = pd.read_csv(CATALOG_CSV)
    catalog["isbn13"] = catalog["isbn13"].astype(str)
    catalog = catalog.set_index("isbn13", drop=False)
    return vector_store, catalog


def warm_up() -> None:
    """Eagerly load model, index, and catalog (useful at app startup)."""
    _load_resources()


def search(query: str, top_k: int = 8) -> list[dict]:
    """Return up to ``top_k`` books most semantically similar to ``query``.

    Results preserve Chroma's similarity ranking. Each item includes the book
    metadata plus a ``score`` (lower distance = more similar).
    """
    query = (query or "").strip()
    if not query:
        return []

    vector_store, catalog = _load_resources()
    matches = vector_store.similarity_search_with_score(query, k=top_k)

    results: list[dict] = []
    for document, score in matches:
        isbn13 = str(document.metadata.get("isbn13", "")).strip()
        if isbn13 not in catalog.index:
            continue
        row = catalog.loc[isbn13]
        results.append(
            {
                "isbn13": isbn13,
                "title": _clean(row.get("title")),
                "authors": _clean(row.get("authors")),
                "categories": _clean(row.get("categories")),
                "thumbnail": _clean(row.get("thumbnail")),
                "description": _clean(row.get("description")),
                "published_year": _year(row.get("published_year")),
                "average_rating": row.get("average_rating"),
                "num_pages": _int(row.get("num_pages")),
                "score": round(float(score), 4),
            }
        )
    return results


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _year(value):
    year = _int(value)
    return year if year else None
