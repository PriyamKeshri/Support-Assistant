PY ?= .venv/bin/python

.PHONY: setup db pdfs ingest api test eval eval-answers

setup:            ## create the virtualenv and install everything
	python3 -m venv .venv && $(PY) -m pip install -e ".[dev]"

db:               ## start Postgres + pgvector
	docker compose up -d db

pdfs:             ## rebuild data/policies/*.pdf from the Markdown sources
	$(PY) scripts/build_policy_pdfs.py

ingest:           ## parse, chunk, embed and store the policy PDFs
	$(PY) -m app.ingest --reset

api:              ## run the API at http://localhost:8000/docs
	$(PY) -m uvicorn app.main:app --reload

test:
	$(PY) -m pytest

eval:             ## compare the four retrieval modes
	$(PY) -m evals.retrieval

eval-answers:     ## answer quality with an LLM judge (needs LLM_* in .env)
	$(PY) -m evals.answers
