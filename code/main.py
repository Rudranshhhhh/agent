"""
Support Triage Agent — Main Entry Point
========================================
Reads support tickets from CSV, runs each through the triage pipeline
(company detection → RAG retrieval → LLM classification & response),
and writes the results to output.csv.

Usage:
    python main.py                  # Process support_tickets.csv
    python main.py --sample         # Process sample_support_tickets.csv (for testing)
    python main.py --reindex        # Force rebuild the vector index
"""

import argparse
import concurrent.futures
import sys
import time
from threading import Lock

import pandas as pd
from dotenv import load_dotenv

from config import INPUT_CSV, SAMPLE_CSV, OUTPUT_CSV, LLM_TEMPERATURE
from indexer import build_or_load_index
from retriever import CorpusRetriever
from agent import TriageAgent
from llm_provider import get_llm_provider


def main():
    # ── Parse arguments ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Run the support triage agent on a CSV of tickets."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample_support_tickets.csv instead of the full set.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force-rebuild the vector index from the corpus.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["groq", "gemini", "nvidia"],
        default=None,
        help="LLM provider to use (overrides LLM_PROVIDER env var).",
    )
    args = parser.parse_args()

    # ── Load environment ─────────────────────────────────────────────────
    load_dotenv()
    llm = get_llm_provider(provider=args.provider, temperature=LLM_TEMPERATURE)

    # ── Build / load index ───────────────────────────────────────────────
    print("=" * 60)
    print("  Support Triage Agent")
    print(f"  LLM: {llm.name}")
    print("=" * 60)
    vectorstore = build_or_load_index(force_rebuild=args.reindex)

    # ── Initialise components ────────────────────────────────────────────
    retriever = CorpusRetriever(vectorstore)
    agent = TriageAgent(retriever, llm_provider=llm)

    # ── Load input CSV ───────────────────────────────────────────────────
    input_path = SAMPLE_CSV if args.sample else INPUT_CSV
    print(f"\nReading tickets from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Found {len(df)} tickets to process.\n")

    # ── Process each ticket ──────────────────────────────────────────────
    results = [None] * len(df)
    start_time = time.time()
    csv_lock = Lock()

    def process_ticket_task(idx, row):
        issue = row.get("Issue", row.get("issue", ""))
        subject = row.get("Subject", row.get("subject", ""))
        company = row.get("Company", row.get("company", "None"))

        ticket_num = idx + 1
        print(f"[{ticket_num}/{len(df)}] Processing: {str(subject)[:60] or '(no subject)'}...")

        try:
            result = agent.process_ticket(issue, subject, company)
        except Exception as exc:
            print(f"  [ERR] Error on ticket {ticket_num}: {exc}", file=sys.stderr)
            result = {
                "status": "escalated",
                "product_area": "general",
                "response": (
                    "This issue has been escalated to a human support agent "
                    "due to a processing error."
                ),
                "justification": f"Automated processing encountered an error: {exc}",
                "request_type": "product_issue",
            }

        def _clean(val: str) -> str:
            return str(val).replace("\n", " ").replace("\r", "").strip()

        # Combine input + output with clean formatting
        cleaned_result = {
            "issue": _clean(issue),
            "subject": _clean(subject),
            "company": _clean(company),
            "response": _clean(result["response"]),
            "product_area": _clean(result["product_area"]),
            "status": _clean(result["status"]),
            "request_type": _clean(result["request_type"]),
            "justification": _clean(result["justification"]),
        }

        status_icon = ">>" if result["status"] == "escalated" else "OK"
        print(
            f"  [{ticket_num}] {status_icon} {result['status']} | {result['request_type']} | "
            f"{result['product_area']}"
        )

        with csv_lock:
            results[idx] = cleaned_result
            # ── Write output CSV incrementally ───────────────────────────────────
            valid_results = [r for r in results if r is not None]
            output_df = pd.DataFrame(valid_results)
            output_df.to_csv(OUTPUT_CSV, index=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(process_ticket_task, idx, row)
            for idx, row in df.iterrows()
        ]
        concurrent.futures.wait(futures)

    valid_results = [r for r in results if r is not None]
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Done! Processed {len(valid_results)} tickets in {elapsed:.1f}s")
    print(f"  Output written to: {OUTPUT_CSV}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
