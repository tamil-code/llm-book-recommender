"""Build a persistent Chroma vector index from the tagged book descriptions.

Run this once locally before deploying. It reads ``tagged_description.txt``
(one ``<isbn13> <description>`` line per book), embeds each description with
``sentence-transformers/all-MiniLM-L6-v2``, and writes a persisted Chroma store
to ``chroma_db/`` so the app starts without re-embedding ~5K books.

Usage:
    python scripts/build_index.py
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAGGED_DESCRIPTIONS = PROJECT_ROOT / "tagged_description.txt"
PERSIST_DIR = PROJECT_ROOT / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "books"

# isbn13 is the first token of each line; some lines are wrapped in a leading
# double quote because the column was written out via pandas.to_csv.
ISBN_PATTERN = re.compile(r'^"?(\d{13})\s+(.*)$', re.DOTALL)


def load_documents() -> list[Document]:
    documents: list[Document] = []
    skipped = 0
    with TAGGED_DESCRIPTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            match = ISBN_PATTERN.match(line)
            if not match:
                skipped += 1
                continue
            isbn13, description = match.group(1), match.group(2).strip().strip('"')
            documents.append(
                Document(page_content=description, metadata={"isbn13": isbn13})
            )
    if skipped:
        print(f"Skipped {skipped} lines that did not start with a 13-digit ISBN.")
    return documents


def main() -> None:
    if not TAGGED_DESCRIPTIONS.exists():
        raise FileNotFoundError(
            f"Could not find {TAGGED_DESCRIPTIONS}. Run the notebook's cleaning "
            "steps first to generate tagged_description.txt."
        )

    documents = load_documents()
    print(f"Loaded {len(documents)} book descriptions.")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Embedding and persisting to {PERSIST_DIR} ...")
    Chroma.from_documents(
        documents,
        embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )
    print("Done. Vector index is ready.")


if __name__ == "__main__":
    main()
