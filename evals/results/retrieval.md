Retrieval on 78 labelled questions · 2026-10-26 · Darwin arm64
Embeddings: `BAAI/bge-small-en-v1.5` · reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

| Mode | Hit@1 | Recall@5 | MRR@10 | nDCG@10 | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| keyword | 0.56 | 0.88 | 0.71 | 0.77 | 0.4 ms | 0.8 ms |
| semantic | 0.80 | 0.96 | 0.88 | 0.90 | 16.3 ms | 18.6 ms |
| hybrid | 0.72 | 0.94 | 0.83 | 0.86 | 18.2 ms | 21.2 ms |
| rerank | 0.89 | 0.98 | 0.93 | 0.94 | 94.9 ms | 112.8 ms |

Recall@5 by question type:

| Mode | exact_term (10) | lookup (23) | multi_section (4) | paraphrase (41) |
|---|---|---|---|---|
| keyword | 1.00 | 1.00 | 0.38 | 0.83 |
| semantic | 1.00 | 1.00 | 0.62 | 0.96 |
| hybrid | 1.00 | 1.00 | 0.38 | 0.94 |
| rerank | 1.00 | 1.00 | 0.75 | 0.99 |
