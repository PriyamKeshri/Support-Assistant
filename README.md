# Agentic Support Assistant

An IT help desk assistant for a company's policy documents. Phase 1 is the retrieval core: it reads policy PDFs, indexes them in **PostgreSQL + pgvector**, finds the right passage with **hybrid search (keyword + semantic) and cross-encoder reranking**, answers with citations, and measures all of it against a labelled question set.

Later phases add the agent itself: tools for looking up and changing tickets, served over MCP, with human approval before risky actions (see [Roadmap](#roadmap)).

> The company, Kestrel Analytics, and its nine IT policies are fictional, written for this project.

## Results

Retrieval over 78 labelled questions (59 chunks from 9 policies), on an Apple M4 laptop. From `python -m evals.retrieval`; per-question detail in [`evals/results/retrieval.json`](evals/results/retrieval.json).

| Mode | Hit@1 | Recall@5 | MRR@10 | nDCG@10 | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| keyword | 0.56 | 0.88 | 0.71 | 0.77 | 0.4 ms | 0.8 ms |
| semantic | 0.80 | 0.96 | 0.88 | 0.90 | 16.3 ms | 18.6 ms |
| hybrid (RRF) | 0.72 | 0.94 | 0.83 | 0.86 | 18.2 ms | 21.2 ms |
| **hybrid + rerank** | **0.89** | **0.98** | **0.93** | **0.94** | 94.9 ms | 112.8 ms |

Recall@5 by question type:

| Mode | exact term (10) | lookup (23) | needs 2 sections (4) | paraphrase (41) |
|---|---|---|---|---|
| keyword | 1.00 | 1.00 | 0.38 | 0.83 |
| semantic | 1.00 | 1.00 | 0.62 | 0.96 |
| hybrid (RRF) | 1.00 | 1.00 | 0.38 | 0.94 |
| hybrid + rerank | 1.00 | 1.00 | 0.75 | 0.99 |

What the numbers say:

- **Reranking is worth its cost here.** It puts the right section first for 89% of questions, against 80% for semantic search alone, for about 80 ms more per query.
- **Equal-weight hybrid search did worse than semantic search alone.** Keyword search ranks chunks by shared words, and words like "data", "laptop" and "call" appear across many policies. "What laptop do data scientists get?" sends keyword search to the Data Classification policy, and fusion lets that pull a wrong section to the top. The reranker recovers most of these cases. Weighting the semantic list higher is the obvious next experiment; it needs a separate tuning split so the weights aren't fitted to this test set.
- **Exact terms didn't separate the retrievers.** With only 59 chunks, the embedding model finds error codes like `VPN-809` too. Keyword search should matter more as the corpus grows and similar documents multiply.
- **Questions that need two sections are the weak spot** (only 4 questions, so a small sample). Even with reranking, a quarter of the sections they need are missing from the top 5.

### Answer quality

All 84 questions, answered from the top 5 reranked chunks by `llama3.1:8b` (4-bit, via Ollama on the same laptop) and graded by the same model. From `python -m evals.answers`; every answer and verdict is in [`evals/results/answers_llama3.1_8b.json`](evals/results/answers_llama3.1_8b.json).

| Metric | Result |
|---|---|
| Correct, as graded by the judge | 0.82 (64 of 78 answerable) |
| Grounded: every claim supported by the excerpts | 0.92 (72 of 78 answers) |
| Hallucination rate | 0.08 |
| Answerable questions refused | 0 of 78 |
| Unanswerable questions declined | 6 of 6 |
| End-to-end latency, p50 / p95 | 7.9 s / 13.0 s (generation is 7.6 s of the p50) |
| Tokens per question | 659 in, 42 out ($0, local model) |

**The judge is the weakest part.** I read the 14 answers it marked wrong or ungrounded: 5 are real mistakes, 7 are judge errors (correct answers marked wrong, or "unsupported" claims that the policy does contain) and 2 are borderline. So true correctness is closer to 0.9, and the judge's scores are a floor. Four of the five real mistakes are reasoning slips with the right section retrieved: for example, the model rated a whole team being blocked as P3 instead of P2, and inferred that accounts are ready "2 business days" before a start date. The fifth is the two-section weak spot from the retrieval results: asked about customer phone numbers on a laptop, it got the storage rules but not the section that says customer personal data is Restricted, so it guessed Confidential. Next steps: a stronger judge than the model being graded, and checking the judge's agreement against hand labels before trusting its numbers.

## How it works

```mermaid
flowchart LR
    PDF[Policy PDFs] --> P[Parse: pypdf, strip headers and footers, numbered sections]
    P --> C[Chunk per section, with overlap]
    C --> E[Embed: bge-small-en-v1.5]
    E --> DB[(Postgres<br/>tsvector + GIN<br/>pgvector + HNSW)]
    Q[Question] --> K[Keyword: ts_rank_cd]
    Q --> S[Semantic: cosine distance]
    DB --- K & S
    K & S --> F[Reciprocal rank fusion, in SQL]
    F --> R[Rerank top 30: cross-encoder]
    R --> L[LLM answer with citations]
```

**Ingestion** (`app/ingest.py`). Each PDF is parsed with pypdf. Running headers and footers are removed (a line that ends in a page number, or appears on every page), bullet glyphs that come out of PDFs as odd characters are normalised, and the text is split into numbered sections. A heading only counts if it is the next number in sequence, so a numbered list inside a section isn't mistaken for a new section. Sections become chunks of at most 1,200 characters, split on sentence boundaries with a 200-character overlap. Each chunk is embedded together with its heading ("Password Policy > 4. Account Lockout"), so it keeps its context. Re-running ingestion skips files whose SHA-256 hasn't changed.

**Storage** (`app/schema.sql`). One `chunks` table holds both indexes: a generated `tsvector` column (heading weighted above body text) with a GIN index for keyword search, and a `vector(384)` column with an HNSW index for semantic search.

**Retrieval** (`app/search.py`). Four modes over the same index, so they can be compared like for like:

| Mode | How |
|---|---|
| `keyword` | Postgres full-text search. Questions are turned into OR queries, because `plainto_tsquery` ANDs every word and misses most natural questions. |
| `semantic` | Cosine distance between the question's embedding and each chunk's. |
| `hybrid` | Both lists fused with reciprocal rank fusion, `1 / (60 + rank)`, in a single SQL query. RRF uses only ranks, so it never has to compare a `ts_rank_cd` score with a cosine similarity. |
| `rerank` | The top 30 hybrid results are re-scored by a cross-encoder (`ms-marco-MiniLM-L-6-v2`), which reads the question and passage together. |

**Answers** (`app/answer.py`). The top 5 reranked chunks go to any OpenAI-compatible model (OpenAI, Azure OpenAI, Groq or a local Ollama model) with instructions to cite every statement as `[n]`, and to reply with a fixed "couldn't find this" sentence when the excerpts don't answer the question.

## Evaluation

`evals/questions.jsonl` holds 84 questions about the policies: 78 answerable, each labelled with the section(s) that answer it (e.g. `IT-OPS-003#4`) and a reference answer, plus 6 that the policies can't answer. Each question has a type, so the results show where each retriever is strong:

| Type | Example |
|---|---|
| `lookup` | "What is the minimum password length?" |
| `paraphrase` | "Someone stole my laptop from my car" (the policy says "lost or stolen devices") |
| `exact_term` | "What does error VPN-809 mean?" |
| `multi_section` | Needs two sections, e.g. a data classification plus a sharing rule |
| `unanswerable` | "How many days of parental leave do I get?" |

Labels are at section level, not chunk level, so they survive changes to the chunking settings.

- `python -m evals.retrieval` runs every mode over the answerable questions and reports Hit@1, Recall@5, MRR@10, nDCG@10 and p50/p95 latency, overall and by question type. It needs no API key. Per-question rankings are saved in `evals/results/retrieval.json` for error analysis.
- `python -m evals.answers` runs the full pipeline and has an LLM judge grade each answer for groundedness (is every claim supported by the retrieved excerpts?) and correctness (does it match the reference?). It also reports false refusals, abstention on unanswerable questions, latency, tokens and cost per question. Each graded question is saved as it finishes, so an interrupted run resumes where it stopped (`--fresh` starts over). A judge model stronger than the one being graded gives more trustworthy scores; set it with `JUDGE_MODEL`.

## Quick start

Needs Docker and Python 3.11+.

```bash
make setup      # virtualenv + dependencies
make db         # Postgres 17 + pgvector on localhost:5433
make ingest     # parse, chunk, embed and store the 9 policy PDFs
make test       # unit + integration tests
make eval       # retrieval comparison table
make api        # http://localhost:8000/docs
```

To get answers as well as search results, copy `.env.example` to `.env` and set `LLM_BASE_URL` and `LLM_MODEL`. With [Ollama](https://ollama.com) running locally:

```bash
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.1:8b
```

Everything in Docker instead (inside the container, `localhost` is the container itself, so to use Ollama on your machine also set `DOCKER_LLM_BASE_URL=http://host.docker.internal:11434/v1`):

```bash
docker compose up -d --build
docker compose exec api python -m app.ingest
docker compose exec api python -m evals.retrieval
```

### API

```bash
curl -s localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "my VPN says VPN-412", "mode": "rerank", "k": 3}'

curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "I lost my laptop at the airport. What should I do?"}'
```

`/search` takes `mode` = `keyword`, `semantic`, `hybrid` or `rerank` (the default). `/ask` returns the answer, the cited sections with page numbers, retrieval and generation time, token counts and cost.

## Project layout

```
app/
  pdf_parser.py   PDF text -> cleaned lines -> numbered sections
  chunking.py     sections -> overlapping chunks with headings
  schema.sql      documents + chunks, GIN and HNSW indexes
  ingest.py       CLI: parse, chunk, embed, upsert
  search.py       keyword / semantic / hybrid (RRF in SQL) / rerank
  answer.py       grounded answers with [n] citations
  main.py         FastAPI: /health, /search, /ask
evals/
  questions.jsonl labelled question set
  metrics.py      Hit@k, Recall@k, MRR, nDCG, percentiles
  retrieval.py    compares the retrieval modes
  answers.py      end-to-end answers graded by an LLM judge
  results/        saved runs
data/
  policies_src/   policy sources in Markdown
  policies/       the PDFs that get ingested
scripts/build_policy_pdfs.py
tests/
```

## Roadmap

1. **Retrieval core (this phase):** Postgres + pgvector hybrid search, reranking, citations, evals.
2. **Agent:** a LangGraph agent with ticket tools (look up, create, update) served from an MCP server, a human approval step before changing tickets, and tracing with Langfuse.
3. **Harder inputs:** OCR for scanned PDFs, masking personal data, a prompt-injection test using a poisoned policy document, and a single-agent vs supervisor comparison on the eval set.
4. **Deployment:** Azure OpenAI with an open-source fallback model, evals in GitHub Actions on every pull request, Kubernetes manifests, and Azure Container Apps.
5. **Ticket triage model:** a small PyTorch classifier that routes tickets, tracked in MLflow and compared with the LLM on accuracy, latency and cost.
