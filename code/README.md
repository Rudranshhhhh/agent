# Support Triage Agent

A terminal-based AI agent that triages support tickets across **HackerRank**, **Claude**, and **Visa** using RAG (Retrieval-Augmented Generation) over the provided support corpus.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   main.py    │────▶│   agent.py   │────▶│  retriever.py│
│  (CLI entry) │     │  (Triage LLM)│     │  (RAG search)│
└──────────────┘     └──────────────┘     └──────────────┘
                            │                     │
                     ┌──────┴──────┐        ┌─────┴──────┐
                     │  prompts.py │        │  indexer.py │
                     │  (Templates)│        │ (ChromaDB)  │
                     └─────────────┘        └────────────┘
```

### Flow per ticket

1. **Company Detection** — if `company` is `None`, the LLM infers it from ticket content.
2. **RAG Retrieval** — queries ChromaDB (filtered by company) for the top-k relevant corpus chunks.
3. **LLM Triage** — Groq LLM classifies the ticket and generates a grounded response or escalation.

### Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point — reads CSV, runs pipeline, writes output |
| `agent.py` | Triage agent — company detection, LLM calls, JSON parsing |
| `retriever.py` | RAG wrapper around ChromaDB with company filtering |
| `indexer.py` | Corpus ingestion, chunking, and ChromaDB indexing |
| `prompts.py` | All prompt templates (company detection, triage) |
| `config.py` | Configuration constants (paths, model, RAG params) |

## Setup

### 1. Install dependencies

```bash
cd code
pip install -r requirements.txt
```

### 2. Configure environment

```bash
# From the project root
cp .env.example .env
# Edit .env and add your Groq API key
```

### 3. Run the agent

The primary command to process the complete evaluation dataset (`support_tickets.csv`) is:

```bash
# Process the actual support tickets
python main.py
```

Additional run options:
```bash
# Test with sample tickets first
python main.py --sample

# Force rebuild the vector index
python main.py --reindex
```

The first run builds the vector index (~774 markdown files → ChromaDB). Subsequent runs load the cached index.

## Output

Results are written to `support_tickets/output.csv` with columns:

| Column | Description |
|--------|-------------|
| `issue` | Original ticket text |
| `subject` | Original subject |
| `company` | Original or detected company |
| `response` | Corpus-grounded answer or escalation message |
| `product_area` | Support category (e.g., screen, billing, privacy) |
| `status` | `replied` or `escalated` |
| `request_type` | `product_issue`, `feature_request`, `bug`, or `invalid` |
| `justification` | Explanation of the triage decision |

## Design Decisions

- **Robust Retry & Backoff**: Implemented a 5-retry strategy with a 15-second exponential backoff in `agent.py` to seamlessly handle API rate limits (TPM/TPD) and technical failures.
- **Groq + LLaMA 3.1 8B**: Fast inference with strong reasoning for classification and response generation, ensuring high throughput.
- **ChromaDB + HuggingFace embeddings**: Fully local vector search — no embedding API costs, deterministic retrieval.
- **Company-filtered retrieval**: When the company is known, searches are scoped to that company's corpus for higher relevance.
- **Structured JSON output**: The LLM returns a strict JSON schema; the agent validates and falls back to escalation on parse errors.
- **Escalation-first safety**: When in doubt (parse failure, insufficient context, sensitive topics), the agent escalates rather than guessing.

## Cost Optimization

- **Local ChromaDB Index**: The 774-document corpus is chunked and indexed entirely locally using HuggingFace sentence-transformers. This means **zero embedding API costs** and completely deterministic offline retrieval.
- **Efficient Open-Weight Models**: Leveraging Groq's high-speed inference for LLaMA models minimizes token processing time and costs compared to proprietary APIs.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Your Groq API key |
| `GROQ_MODEL` | No | Override the default model (default: `llama-3.1-8b-instant`) |
