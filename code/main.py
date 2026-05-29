from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from llm import LLMAdapter
from models import CSV_HEADERS, TicketInput
from orchestrator import run_structured_decision
from retrieval import CorpusIndex


def repo_root() -> Path:
    """Resolve the absolute path to the repository root."""
    return Path(__file__).resolve().parents[1]


def _parse_issue(raw_issue: str) -> list[dict[str, str]]:
    """Defensively parse raw CSV issue strings into structured message arrays."""
    try:
        parsed = json.loads(raw_issue)
    except json.JSONDecodeError:
        return [{"role": "user", "content": raw_issue}]

    if not isinstance(parsed, list):
        return [{"role": "user", "content": str(parsed)}]

    messages: list[dict[str, str]] = []
    for item in parsed:
        if isinstance(item, dict):
            messages.append(
                {
                    "role": str(item.get("role", "user"))[:30],
                    "content": str(item.get("content", "")),
                }
            )
        else:
            messages.append({"role": "user", "content": str(item)})
    return messages


def run() -> int:
    """Execute the end-to-end batch processing pipeline."""
    root = repo_root()
    input_path = root / "support_tickets" / "support_tickets.csv"
    output_path = root / "support_tickets" / "output.csv"

    if not input_path.exists():
        print(f"CRITICAL ERROR: Input file not found at {input_path.as_posix()}", file=sys.stderr)
        return 1

    print("Initializing AI Support Agent Pipeline...")

    index = CorpusIndex(root)
    llm = LLMAdapter()
    results: list[dict[str, str]] = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=2):
            raw_issue = row.get("issue") or row.get("Issue") or ""
            raw_subject = row.get("subject") or row.get("Subject") or ""
            raw_company = row.get("company") or row.get("Company") or "None"

            try:
                ticket = TicketInput(
                    issue=_parse_issue(raw_issue),
                    subject=raw_subject,
                    company=raw_company,
                )
                output = run_structured_decision(ticket, index, llm)
                results.append(output.to_csv_row())
            except Exception as exc:
                print(f"WARNING: Failed to process row {row_idx}: {type(exc).__name__}", file=sys.stderr)
                results.append(
                    {
                        "issue": raw_issue,
                        "subject": raw_subject,
                        "company": raw_company,
                        "response": "I need to route this ticket to a human support specialist because it could not be processed safely.",
                        "product_area": "general",
                        "status": "escalated",
                        "request_type": "invalid",
                        "justification": "Escalated because the ticket could not be parsed or processed safely by the automated pipeline.",
                        "confidence_score": "0.000",
                        "source_documents": "",
                        "risk_level": "high",
                        "pii_detected": "false",
                        "language": "en",
                        "actions_taken": "[]",
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(results)

    print(f"Successfully processed {len(results)} tickets. Output saved to: {output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
