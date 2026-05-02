"""
Corpus Indexer — loads all markdown files from data/ and indexes them into
ChromaDB with company and category metadata for filtered retrieval.

Usage:
    from indexer import build_or_load_index
    vectorstore = build_or_load_index()
"""

import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    CORPUS_DIR,
    INDEX_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COMPANY_DIR_MAP,
)


def _detect_company_from_path(file_path: Path) -> str:
    """Determine company from the top-level directory under data/."""
    try:
        relative = file_path.relative_to(CORPUS_DIR)
        top_dir = relative.parts[0].lower()
        return COMPANY_DIR_MAP.get(top_dir, "Unknown")
    except (ValueError, IndexError):
        return "Unknown"


def _detect_category_from_path(file_path: Path) -> str:
    """Extract the second-level directory name as the category."""
    try:
        relative = file_path.relative_to(CORPUS_DIR)
        # parts[0] = company, parts[1] = category (if it's a dir, not a file)
        if len(relative.parts) > 2:
            return relative.parts[1]
        return "general"
    except (ValueError, IndexError):
        return "general"


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared embedding function."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_documents() -> list[dict]:
    """Read all .md files from the corpus directory."""
    documents = []
    for md_file in sorted(CORPUS_DIR.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            if not content.strip():
                continue

            company = _detect_company_from_path(md_file)
            category = _detect_category_from_path(md_file)
            source = str(md_file.relative_to(CORPUS_DIR))

            # Extract a title from the first markdown heading if available
            title = ""
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("#"):
                    title = stripped.lstrip("#").strip()
                    break

            documents.append(
                {
                    "content": content,
                    "metadata": {
                        "company": company,
                        "category": category,
                        "source": source,
                        "title": title or md_file.stem,
                    },
                }
            )
        except Exception as exc:
            print(f"  [WARN] Skipping {md_file}: {exc}", file=sys.stderr)

    return documents


def _chunk_documents(documents: list[dict]) -> tuple[list[str], list[dict]]:
    """Split documents into overlapping chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
        keep_separator=True,
    )

    all_texts: list[str] = []
    all_metas: list[dict] = []

    for doc in documents:
        chunks = splitter.split_text(doc["content"])
        for chunk in chunks:
            all_texts.append(chunk)
            all_metas.append(doc["metadata"].copy())

    return all_texts, all_metas


def build_or_load_index(force_rebuild: bool = False) -> Chroma:
    """
    Return a ChromaDB vectorstore backed by the support corpus.

    If a persisted index already exists at INDEX_DIR it is loaded directly.
    Pass ``force_rebuild=True`` to re-index from scratch.
    """
    embeddings = _get_embeddings()

    if INDEX_DIR.exists() and not force_rebuild:
        # Check if the collection actually has data
        try:
            store = Chroma(
                persist_directory=str(INDEX_DIR),
                embedding_function=embeddings,
                collection_name="support_corpus",
            )
            count = store._collection.count()
            if count > 0:
                print(f"[OK] Loaded existing index ({count:,} chunks) from {INDEX_DIR}")
                return store
        except Exception:
            pass  # Fall through to rebuild

    # ── Build from scratch ───────────────────────────────────────────────
    print("Building vector index from corpus ...")
    documents = _load_documents()
    print(f"  Loaded {len(documents)} documents")

    texts, metas = _chunk_documents(documents)
    print(f"  Created {len(texts):,} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Ensure clean directory
    if INDEX_DIR.exists():
        import shutil
        shutil.rmtree(INDEX_DIR)

    store = Chroma.from_texts(
        texts=texts,
        metadatas=metas,
        embedding=embeddings,
        persist_directory=str(INDEX_DIR),
        collection_name="support_corpus",
    )
    print(f"  [OK] Index built and persisted to {INDEX_DIR}")
    return store


# ── CLI helper ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    force = "--force" in sys.argv
    build_or_load_index(force_rebuild=force)
