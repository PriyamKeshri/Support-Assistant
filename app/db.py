from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

SCHEMA = Path(__file__).with_name("schema.sql")


def init_schema(conn: psycopg.Connection, dim: int, reset: bool = False) -> None:
    """Creates the extension, tables and indexes. `reset` drops existing data first,
    which is needed after changing the embedding model or its dimension."""
    if reset:
        conn.execute("DROP TABLE IF EXISTS chunks, documents")
    conn.execute(SCHEMA.read_text().replace("{dim}", str(int(dim))))
    conn.commit()


def create_pool(url: str) -> ConnectionPool:
    # register_vector looks up the vector type, so the extension must already exist.
    return ConnectionPool(url, min_size=1, max_size=5, configure=register_vector, open=True)
