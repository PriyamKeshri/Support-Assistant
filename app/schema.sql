-- {dim} is replaced with the embedding dimension when the schema is created.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          bigserial PRIMARY KEY,
    code        text NOT NULL UNIQUE,          -- policy ID, e.g. IT-SEC-001
    title       text NOT NULL,
    source_path text NOT NULL,
    sha256      text NOT NULL,                 -- skip re-ingesting unchanged files
    ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id            bigserial PRIMARY KEY,
    document_id   bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    section_no    int NOT NULL,
    section_title text NOT NULL,
    chunk_index   int NOT NULL,
    page          int NOT NULL,
    heading       text NOT NULL,               -- "Document title > N. Section title"
    content       text NOT NULL,
    embedding     vector({dim}) NOT NULL,
    -- Keyword side of hybrid search: headings weigh more than body text.
    tsv           tsvector GENERATED ALWAYS AS (
                      setweight(to_tsvector('english', heading), 'A') ||
                      setweight(to_tsvector('english', content), 'B')
                  ) STORED,
    UNIQUE (document_id, section_no, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
