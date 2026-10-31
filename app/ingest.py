"""Ingests policy PDFs into Postgres: parse, chunk, embed, store.

    python -m app.ingest                  # everything in data/policies
    python -m app.ingest path/to/file.pdf
    python -m app.ingest --reset          # drop and rebuild (after changing the embedding model)
"""
import argparse
import hashlib
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from app.chunking import chunk_document
from app.config import get_settings
from app.db import init_schema
from app.models import get_embedder
from app.pdf_parser import parse_pdf

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"

UPSERT_DOCUMENT = """
INSERT INTO documents (code, title, source_path, sha256)
VALUES (%(code)s, %(title)s, %(source_path)s, %(sha256)s)
ON CONFLICT (code) DO UPDATE
    SET title = EXCLUDED.title, source_path = EXCLUDED.source_path,
        sha256 = EXCLUDED.sha256, ingested_at = now()
RETURNING id
"""

INSERT_CHUNK = """
INSERT INTO chunks (document_id, section_no, section_title, chunk_index, page, heading, content, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


def pdf_paths(targets: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        paths += sorted(target.glob("*.pdf")) if target.is_dir() else [target]
    return paths


def ingest(targets: list[Path], reset: bool = False) -> None:
    settings = get_settings()
    embedder = get_embedder()
    with psycopg.connect(settings.database_url) as conn:
        init_schema(conn, settings.embedding_dim, reset=reset)
        register_vector(conn)
        for path in pdf_paths(targets):
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            doc = parse_pdf(path)
            existing = conn.execute("SELECT sha256 FROM documents WHERE code = %s", (doc.code,)).fetchone()
            if existing and existing[0] == sha256:
                print(f"{doc.code}: unchanged, skipped")
                continue

            chunks = chunk_document(doc, settings.chunk_chars, settings.chunk_overlap_chars)
            vectors = embedder.embed_passages([c.embed_text for c in chunks])
            with conn.transaction():
                doc_id = conn.execute(UPSERT_DOCUMENT, {
                    "code": doc.code, "title": doc.title, "source_path": str(path), "sha256": sha256,
                }).fetchone()[0]
                conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                with conn.cursor() as cur:
                    cur.executemany(INSERT_CHUNK, [
                        (doc_id, c.section_no, c.section_title, c.chunk_index, c.page, c.heading, c.content, v)
                        for c, v in zip(chunks, vectors)
                    ])
            print(f"{doc.code}: {len(doc.sections)} sections, {len(chunks)} chunks  ({path.name})")
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(f"{total} chunks in the index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_DIR])
    parser.add_argument("--reset", action="store_true", help="drop all documents and chunks first")
    args = parser.parse_args()
    ingest(args.paths, reset=args.reset)


if __name__ == "__main__":
    main()
