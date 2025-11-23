"""Utility script to ingest documents (txt/md/pdf) into Chroma collections."""

import argparse
from typing import Optional

from chunk import split_text_into_sentences
from document_loader import load_document_text
from embedding import embed_chunk
from DB import save_embeddings, get_db


def ingest_text(
    text: str,
    language: str = "Chinese",
    chunk_size: int = 300,
    persistent: bool = True,
    collection: Optional[str] = None,
):
    chunks = split_text_into_sentences(text, max_chunk_size=chunk_size, language=language)
    if not chunks:
        raise ValueError("No textual content extracted from document.")

    embeddings = [embed_chunk(chunk, language) for chunk in chunks]

    collection_name = collection or ("default" if persistent else "session_temp")
    db = get_db(persistent=persistent, name=collection_name)
    save_embeddings(db, chunks, embeddings)
    return collection_name, len(chunks)


def build_index(
    path: str,
    language: str = "Chinese",
    chunk_size: int = 300,
    persistent: bool = True,
    collection: Optional[str] = None,
):
    text = load_document_text(path)
    return ingest_text(
        text=text,
        language=language,
        chunk_size=chunk_size,
        persistent=persistent,
        collection=collection,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build embeddings index from document")
    parser.add_argument("path", help="Path to document (txt/md/pdf)")
    parser.add_argument("--language", default="Chinese", help="Language used for chunking/embeddings")
    parser.add_argument("--chunk-size", type=int, default=300, help="Maximum characters per chunk")
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Store embeddings in persistent Chroma collection (default: temporary)",
    )
    parser.add_argument(
        "--collection",
        help="Collection name (defaults to 'default' for persistent or 'session_temp' for temporary)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    coll, count = build_index(
        path=args.path,
        language=args.language,
        chunk_size=args.chunk_size,
        persistent=args.persistent,
        collection=args.collection,
    )
    print(f"Indexed {count} chunks into collection '{coll}' (persistent={args.persistent}).")
