"""
Triage Agent — the "brain" of the support agent.
"""

import json
import re
import time
import sys

from langchain_groq import ChatGroq

from config import GROQ_MODEL, GROQ_TEMPERATURE
from prompts import COMPANY_DETECTION_PROMPT, TRIAGE_PROMPT
from retriever import CorpusRetriever


class TriageAgent:
    VALID_COMPANIES = {"HackerRank", "Claude", "Visa", "None"}
    VALID_STATUSES = {"replied", "escalated"}
    VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}

    def __init__(self, retriever: CorpusRetriever, groq_api_key: str):
        self.retriever = retriever
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=GROQ_TEMPERATURE,
            api_key=groq_api_key,
            max_retries=3,
        )

    # ── Deterministic Pre-LLM Escalation Gate ────────────────────────────

    # Patterns that MUST escalate regardless of what the corpus says.
    # These are cases where the user is demanding a human action or has
    # explicitly stated they cannot self-serve.

    # (pattern, product_area, request_type, justification)
    _FORCE_ESCALATE_PATTERNS: list[tuple] = [
        # A. Access restoration where user says they can't do it themselves
        (
            r"(restore|reinstate|give back|re-?add).{0,60}(access|seat|permission)"
            r"|"
            r"(access|seat|removed).{0,60}(restore|reinstate|re-?add)",
            "account_management", "product_issue",
            "User is demanding access restoration on a specific account — requires human agent action.",
        ),
        # B. Explicit refund / money-back demands
        (
            r"\b(refund|give.{0,10}money back|money back|give.{0,10}refund|want.{0,10}refund|need.{0,10}refund)\b",
            "billing", "product_issue",
            "User is demanding a financial refund — requires human agent action.",
        ),
        # C. Action against a third party (ban, force, sue, report)
        (
            r"\b(ban|block|remove|force|sue|report).{0,30}(seller|merchant|company|vendor|them)\b",
            "general", "product_issue",
            "User is requesting action against a third party — requires human agent action.",
        ),
        # D. Manual score/test review
        (
            r"(review|re-?grade|re-?evaluate|increase|change).{0,50}(score|grade|result|answer)",
            "screen", "product_issue",
            "User is requesting manual review or modification of a test result — requires human agent action.",
        ),
        # E. Infosec / compliance / forms that support must fill for the user
        (
            r"(fill.{0,20}form|filling.{0,20}form|complete.{0,20}form|infosec.{0,40}(process|form)|security.{0,30}(questionnaire|form|assessment))",
            "security", "product_issue",
            "User is requesting support to fill in compliance/security forms on their behalf — requires human agent action.",
        ),
        # F. Certificate / account detail update on a specific account
        (
            r"(update|change|fix|correct).{0,30}(name|detail|certificate|cert).{0,30}(my|our|the)\b"
            r"|"
            r"(my|the).{0,30}(name|certificate).{0,30}(update|change|fix|incorrect|wrong)",
            "certifications", "product_issue",
            "User is requesting a manual update to a specific certificate or account detail — requires human agent action.",
        ),
        # G. Subscription / account pause or cancellation
        (
            r"\b(pause|cancel|suspend|stop).{0,30}(subscription|plan|account|service|hiring)\b",
            "subscriptions", "product_issue",
            "User is requesting manual modification to their subscription — requires human agent action.",
        ),
        # H. Security vulnerability / bug bounty reports
        (
            r"(security vulnerability|bug bounty|found.{0,30}vulnerability|major.{0,20}(vulnerability|exploit|security))",
            "safeguards", "bug",
            "User is reporting a security vulnerability — requires human agent to investigate and address.",
        ),
        # I. Remove employee / user from account (admin action)
        (
            r"(employee|staff|team member).{0,30}(left|leaving|departed|gone)"
            r"|"
            r"(remove|delete).{0,30}(employee|staff|team member|them).{0,30}(account|platform|system)",
            "account_management", "product_issue",
            "User is requesting removal of an employee from their account — requires human agent action.",
        ),
        # J. Complete product/service outage (all requests failing, everything down)
        (
            r"(stopped working completely|all requests?.{0,20}failing|completely.{0,20}(down|broken|stopped))"
            r"|"
            r"(everything.{0,20}(down|failing|broken)|none of.{0,20}(pages|requests).{0,20}(work|accessible))",
            "general_support", "bug",
            "User is reporting a complete service outage — requires human agent to investigate.",
        ),
    ]

    def _check_force_escalate(self, issue: str, subject: str) -> dict | None:
        """
        Check if this ticket must be escalated before calling the LLM.
        Returns an escalation dict if matched, else None.
        """
        combined = f"{subject} {issue}"
        for pattern, product_area, request_type, justification in self._FORCE_ESCALATE_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                return {
                    "status": "escalated",
                    "product_area": product_area,
                    "response": (
                        "This issue has been escalated to a human support agent "
                        "who can assist with this issue. They will follow up with you shortly."
                    ),
                    "justification": justification,
                    "request_type": request_type,
                }
        return None

    # ── Main Pipeline ─────────────────────────────────────────────────────

    def process_ticket(self, issue: str, subject: str, company: str) -> dict:
        issue = str(issue).strip() if issue else ""
        subject = str(subject).strip() if subject else ""
        company = str(company).strip() if company else "None"

        # Normalize pandas NaN values to "None"
        if company.lower() in ("nan", "", "none", "null"):
            company = "None"

        original_company_unknown = (company == "None")

        # ── Vague ticket detection (before LLM calls) ────────────────────
        # If the issue is extremely short/vague AND company is unknown from CSV,
        # escalate rather than risk hallucinating an irrelevant answer.
        combined_text = f"{subject} {issue}".strip()
        meaningful_words = [w for w in combined_text.split() if w.lower() not in
                           ("help", "please", "needed", "me", "i", "it's", "its",
                            "the", "a", "an", "not", "working", "work")]
        if len(meaningful_words) <= 2 and original_company_unknown:
            return {
                "status": "escalated",
                "product_area": "general_support",
                "response": (
                    "This issue has been escalated to a human support agent "
                    "who can assist with this issue. They will follow up with you shortly."
                ),
                "justification": "Ticket is too vague to determine intent or product area — escalated for safety.",
                "request_type": "product_issue",
            }

        # Company detection
        if company not in self.VALID_COMPANIES or company == "None":
            company = self._detect_company(issue, subject)

        # ── Deterministic escalation gate (before any LLM call) ───────────
        forced = self._check_force_escalate(issue, subject)
        if forced:
            return forced

        # Retrieval
        query = f"{subject} {issue}".strip()
        raw_results = self.retriever.retrieve(query, company=company)
        context = self.retriever.format_context(raw_results)

        # 🔥 HARD FAIL-SAFE (correct)
        if not raw_results or "No relevant documentation found" in context:
            return {
                "status": "escalated",
                "product_area": "general",
                "response": "This issue has been escalated to a human support agent who can assist with this issue. They will follow up with you shortly.",
                "justification": "Escalated due to Missing Corpus: No relevant documentation found.",
                "request_type": "product_issue"
            }

        # LLM triage
        result = self._triage(issue, subject, company, context)
        return result

    # ── LLM Calls ─────────────────────────────────────────────────────────

    def _detect_company(self, issue: str, subject: str) -> str:
        prompt = COMPANY_DETECTION_PROMPT.format(issue=issue, subject=subject)
        response = self._call_llm(prompt)

        company = response.strip().strip('"').strip("'")

        for valid in self.VALID_COMPANIES:
            if valid.lower() in company.lower():
                return valid
        return "None"

    def _triage(self, issue: str, subject: str, company: str, context: str) -> dict:
        prompt = TRIAGE_PROMPT.format(
            context=context,
            company=company,
            subject=subject,
            issue=issue,
        )

        raw_response = self._call_llm(prompt)
        result = self._parse_triage_response(raw_response)

        # If parsing hit the fallback, retry once with a stricter prompt
        if result.get("justification", "").startswith("Automated triage could not parse"):
            retry_prompt = (
                prompt + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                "Return ONLY a raw JSON object with keys: status, product_area, response, justification, request_type. "
                "No markdown, no explanation, no text outside the JSON."
            )
            try:
                raw_response = self._call_llm(retry_prompt)
                result = self._parse_triage_response(raw_response)
            except Exception:
                pass  # Keep the fallback result

        return result

    def _call_llm(self, prompt: str, retries: int = 5) -> str:
        for attempt in range(retries):
            try:
                result = self.llm.invoke(prompt)
                content = result.content if result and result.content else ""
                if not content.strip():
                    raise ValueError("Empty LLM response")
                return content
            except Exception as exc:
                if attempt < retries - 1:
                    wait = 15 * (2 ** attempt)
                    print(f"[WARN] LLM failed (attempt {attempt+1}/{retries}): {exc}, retrying in {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                else:
                    raise

    # ── 🔥 FIXED PARSER ───────────────────────────────────────────────────

    def _parse_triage_response(self, raw: str) -> dict:
        cleaned = raw.strip()

        # Remove markdown fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        # Remove any leading/trailing non-JSON text
        cleaned = re.sub(r"^[^{]*", "", cleaned)
        cleaned = re.sub(r"[^}]*$", "", cleaned)

        # Try parsing directly
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            # Extract JSON block — greedy match to capture full object
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    print("[PARSE ERROR] Could not parse extracted JSON")
                    return self._fallback_result(raw)
            else:
                print("[PARSE ERROR] No JSON found")
                return self._fallback_result(raw)

        # 🔧 Normalize fields
        result["status"] = result.get("status", "escalated").lower().strip()
        if result["status"] not in self.VALID_STATUSES:
            result["status"] = "escalated"

        result["request_type"] = result.get("request_type", "product_issue").lower().strip()
        if result["request_type"] not in self.VALID_REQUEST_TYPES:
            result["request_type"] = "product_issue"

        result["product_area"] = (
            result.get("product_area", "general").lower().strip().replace(" ", "_")
        )
        result.setdefault("response", "Escalated to human agent.")
        result.setdefault("justification", "Fallback used.")

        return {
            "status": result["status"],
            "product_area": result["product_area"],
            "response": result["response"],
            "justification": result["justification"],
            "request_type": result["request_type"],
        }

    # ── Fallback ──────────────────────────────────────────────────────────

    def _fallback_result(self, raw_response: str) -> dict:
        """Try to salvage a result from a malformed LLM response before giving up."""
        # Attempt to extract partial fields from the raw text
        status = "escalated"
        product_area = "general"
        request_type = "product_issue"
        response = "This issue has been escalated to a human support agent who can assist with this issue. They will follow up with you shortly."
        justification = "Automated triage could not parse LLM output. Escalated for safety."

        # Try extracting fields via regex from the raw response
        status_match = re.search(r'"status"\s*:\s*"(replied|escalated)"', raw_response, re.IGNORECASE)
        if status_match:
            status = status_match.group(1).lower()

        area_match = re.search(r'"product_area"\s*:\s*"([^"]+)"', raw_response, re.IGNORECASE)
        if area_match:
            product_area = area_match.group(1).lower().strip().replace(" ", "_")

        type_match = re.search(r'"request_type"\s*:\s*"([^"]+)"', raw_response, re.IGNORECASE)
        if type_match:
            val = type_match.group(1).lower().strip()
            if val in self.VALID_REQUEST_TYPES:
                request_type = val

        resp_match = re.search(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_response)
        if resp_match:
            response = resp_match.group(1).replace('\\n', ' ').replace('\\"', '"')

        just_match = re.search(r'"justification"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_response)
        if just_match:
            justification = just_match.group(1).replace('\\n', ' ').replace('\\"', '"')

        return {
            "status": status,
            "product_area": product_area,
            "response": response,
            "justification": justification,
            "request_type": request_type,
        }