# Architecture Diagram — Support Triage Agent

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SUPPORT TRIAGE AGENT                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌──────────────────────────┐
│  INPUT CSV   │────────▶│   main.py    │────────▶│       output.csv         │
│              │         │ (orchestrator)│         │ (status, product_area,   │
│ - issue      │         │              │         │  response, justification,│
│ - subject    │         │ ThreadPool   │         │  request_type)           │
│ - company    │         │ (3 workers)  │         │                          │
└──────────────┘         └──────┬───────┘         └──────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      agent.py         │
                    │   (TriageAgent)       │
                    └───────────┬───────────┘
                                │
                 ┌──────────────┼──────────────────┐
                 ▼              ▼                   ▼
        ┌────────────┐  ┌─────────────┐   ┌──────────────┐
        │  Stage 1   │  │   Stage 2   │   │   Stage 3    │
        │ Pre-checks │  │  Retrieval  │   │  LLM Triage  │
        └────────────┘  └─────────────┘   └──────────────┘
```

---

## Detailed Pipeline (per ticket)

```
                          ┌─────────────────────────┐
                          │     INPUT TICKET         │
                          │  issue + subject + co.   │
                          └────────────┬────────────┘
                                       │
                                       ▼
                    ┌───────────────────────────────────┐
                    │  1. NORMALIZE COMPANY             │
                    │  - "nan"/"null"/"" → "None"       │
                    └──────────────────┬────────────────┘
                                       │
                                       ▼
                    ┌───────────────────────────────────┐
                    │  2. VAGUE TICKET CHECK            │
                    │  - Filter filler words            │
                    │  - If ≤2 meaningful words AND     │
                    │    company was originally unknown  │
                    │  → ESCALATE immediately           │
                    └──────────────────┬────────────────┘
                                       │ (not vague)
                                       ▼
                    ┌───────────────────────────────────┐
                    │  3. COMPANY DETECTION (LLM)       │
                    │  - If company == "None"           │
                    │  - Sends issue+subject to LLM     │
                    │  - Returns: HackerRank/Claude/    │
                    │    Visa/None                      │
                    └──────────────────┬────────────────┘
                                       │
                                       ▼
                    ┌───────────────────────────────────┐
                    │  4. FORCE-ESCALATE PATTERNS       │
                    │  (Deterministic regex gate)       │
                    │                                   │
                    │  A. Access restoration demands    │
                    │  B. Refund/money-back demands     │
                    │  C. Action against third party    │
                    │  D. Manual score/test review      │
                    │  E. Infosec/compliance forms      │
                    │  F. Certificate/account updates   │
                    │  G. Subscription pause/cancel     │
                    │  H. Security vulnerability/bounty │
                    │  I. Employee removal from account │
                    │  J. Complete service outage       │
                    │                                   │
                    │  If matched → ESCALATE with       │
                    │  predefined product_area & type   │
                    └──────────────────┬────────────────┘
                                       │ (no pattern match)
                                       ▼
                    ┌───────────────────────────────────┐
                    │  5. RAG RETRIEVAL                 │
                    │  (retriever.py + ChromaDB)        │
                    │                                   │
                    │  - Query = subject + issue        │
                    │  - Filter by company (metadata)   │
                    │  - Top-K=6 similar chunks         │
                    │  - Returns formatted context      │
                    │                                   │
                    │  If NO results → ESCALATE         │
                    │  ("Missing Corpus")               │
                    └──────────────────┬────────────────┘
                                       │ (has context)
                                       ▼
                    ┌───────────────────────────────────┐
                    │  6. LLM TRIAGE (Groq/LLaMA)      │
                    │                                   │
                    │  Prompt includes:                 │
                    │  - Retrieved corpus context       │
                    │  - Escalation rules               │
                    │  - Reply rules                    │
                    │  - Output JSON schema             │
                    │                                   │
                    │  Returns JSON:                    │
                    │  {status, product_area, response, │
                    │   justification, request_type}    │
                    └──────────────────┬────────────────┘
                                       │
                                       ▼
                    ┌───────────────────────────────────┐
                    │  7. RESPONSE PARSING              │
                    │  - Strip markdown fences          │
                    │  - Extract JSON via regex         │
                    │  - Normalize fields               │
                    │  - On failure: retry with strict  │
                    │    prompt, else regex extraction   │
                    └──────────────────┬────────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │     OUTPUT ROW           │
                          │  Written to output.csv   │
                          └─────────────────────────┘
```

---

## Indexing Pipeline (ChromaDB)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INDEXER (indexer.py)                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│     data/ corpus     │
│                      │
│  data/hackerrank/    │──┐
│  data/claude/        │  │
│  data/visa/          │  │
│                      │  │
│  (*.md files)        │  │
└──────────────────────┘  │
                          ▼
              ┌───────────────────────────┐
              │  1. LOAD DOCUMENTS        │
              │  - Recursively scan *.md  │
              │  - Extract metadata:      │
              │    • company (from path)  │
              │    • category (subdir)    │
              │    • source (rel path)    │
              │    • title (first # line) │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │  2. CHUNK DOCUMENTS       │
              │  - RecursiveCharText      │
              │    Splitter               │
              │  - chunk_size=1500 chars  │
              │  - overlap=300 chars      │
              │  - Splits on headings,    │
              │    paragraphs, sentences  │
              │  - Preserves metadata     │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │  3. EMBED & STORE         │
              │  - Model: all-MiniLM-L6-v2│
              │    (sentence-transformers)│
              │  - Normalized embeddings  │
              │  - Stored in ChromaDB     │
              │    (data/index/)          │
              │  - Collection:            │
              │    "support_corpus"       │
              └───────────────────────────┘
```

---

## ChromaDB Schema

```
Collection: "support_corpus"
Persist Dir: data/index/

Each document chunk stored with:
┌──────────────────────────────────────────────┐
│  Vector: 384-dim (all-MiniLM-L6-v2)         │
│  Content: chunk text (up to 1500 chars)      │
│  Metadata:                                   │
│    ├── company: "HackerRank"|"Claude"|"Visa" │
│    ├── category: e.g. "screen", "billing"    │
│    ├── source: "claude/claude-code/1234.md"  │
│    └── title: "How to reset your password"   │
└──────────────────────────────────────────────┘
```

---

## Retrieval at Query Time

```
  Query: "How do I remove an interviewer?"
  Company filter: "HackerRank"
         │
         ▼
  ┌──────────────────────────────────┐
  │  ChromaDB Similarity Search      │
  │                                  │
  │  1. Embed query with same model  │
  │     (all-MiniLM-L6-v2)          │
  │  2. Filter: company="HackerRank" │
  │  3. Cosine similarity ranking    │
  │  4. Return top-6 chunks          │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │  Format as context string:       │
  │                                  │
  │  [Doc 1] HackerRank / interviews │
  │  — Manage Interview Templates    │
  │  <chunk content>                 │
  │                                  │
  │  ---                             │
  │                                  │
  │  [Doc 2] HackerRank / general    │
  │  — October 2025 Release Notes    │
  │  <chunk content>                 │
  │  ...                             │
  └──────────────────────────────────┘
```

---

## Data Corpus Structure

```
data/
├── hackerrank/          ← HackerRank support articles
│   ├── index.md
│   ├── screen/          (test management, proctoring)
│   ├── interviews/      (CodePair, templates, lobby)
│   ├── general-help/    (release notes, FAQs)
│   ├── integrations/    (Zapier, ATS, SSO)
│   ├── library/         (question library)
│   ├── settings/        (account, billing)
│   └── ...
├── claude/              ← Claude/Anthropic support articles
│   ├── index.md
│   ├── claude/          (product features, limits)
│   ├── claude-api-and-console/
│   ├── claude-code/
│   ├── claude-desktop/
│   ├── safeguards/      (model safety, bug bounty)
│   ├── privacy-and-legal/
│   └── ...
├── visa/                ← Visa consumer support articles
│   ├── index.md
│   ├── support.md
│   └── support/         (fraud, travel, disputes)
└── index/               ← ChromaDB persisted store
    ├── chroma.sqlite3
    └── <collection_uuid>/
```

---

## Technology Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Language        | Python 3.11                         |
| LLM             | Groq API (LLaMA 3.1 8B Instant)    |
| Embeddings      | all-MiniLM-L6-v2 (HuggingFace)     |
| Vector DB       | ChromaDB (local, persisted)         |
| Framework       | LangChain (Chroma + Groq + HF)     |
| Parallelism     | concurrent.futures ThreadPool (3)   |
| Config          | .env (secrets) + config.py (paths)  |

---

## Key Design Decisions

1. **Deterministic pre-LLM escalation** — Regex patterns catch high-risk tickets before any LLM call, ensuring safety-critical decisions are never delegated to a probabilistic model.

2. **Company-filtered retrieval** — ChromaDB metadata filtering ensures retrieval only pulls docs from the relevant company's corpus, preventing cross-domain confusion.

3. **Temperature=0** — LLM calls use temperature=0 for maximum determinism.

4. **Retry with stricter prompt** — If JSON parsing fails, the agent retries with an explicit "return only JSON" instruction before falling back.

5. **Vague ticket detection** — Extremely short tickets with unknown company are escalated pre-retrieval to avoid hallucinated responses from random corpus matches.
