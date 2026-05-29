#!/usr/bin/env python3
"""Validate output.csv structure for the MLE Hiring Challenge."""

import csv
import json
import os
import sys


EXPECTED_HEADERS = [
    "issue", "subject", "company", "response", "product_area",
    "status", "request_type", "justification", "confidence_score",
    "source_documents", "risk_level", "pii_detected", "language",
    "actions_taken",
]

VALID_STATUS = {"replied", "escalated"}
VALID_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}
VALID_RISK_LEVEL = {"low", "medium", "high", "critical"}
VALID_PII_DETECTED = {"true", "false"}


def validate() -> bool:
    output_path = os.path.join(os.path.dirname(__file__), "..", "support_tickets", "output.csv")
    input_path = os.path.join(os.path.dirname(__file__), "..", "support_tickets", "support_tickets.csv")

    if not os.path.exists(output_path):
        print("FAIL: output.csv not found at", output_path)
        return False
    if not os.path.exists(input_path):
        print("FAIL: support_tickets.csv not found at", input_path)
        return False

    with open(input_path, "r", encoding="utf-8") as handle:
        input_reader = csv.reader(handle)
        next(input_reader)
        input_count = sum(1 for _ in input_reader)

    errors: list[str] = []
    warnings: list[str] = []

    with open(output_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        actual_headers = reader.fieldnames
        if actual_headers is None:
            print("FAIL: output.csv is empty or has no headers")
            return False

        missing_headers = set(EXPECTED_HEADERS) - set(actual_headers)
        extra_headers = set(actual_headers) - set(EXPECTED_HEADERS)
        if missing_headers:
            errors.append(f"Missing columns: {', '.join(sorted(missing_headers))}")
        if extra_headers:
            warnings.append(f"Extra columns (will be ignored): {', '.join(sorted(extra_headers))}")

        rows = list(reader)
        output_count = len(rows)
        if output_count != input_count:
            errors.append(f"Row count mismatch: expected {input_count}, got {output_count}")

        for index, row in enumerate(rows, start=1):
            status = row.get("status", "").strip().lower()
            if status not in VALID_STATUS:
                errors.append(f"Row {index}: invalid status '{status}'")

            request_type = row.get("request_type", "").strip().lower()
            if request_type not in VALID_REQUEST_TYPE:
                errors.append(f"Row {index}: invalid request_type '{request_type}'")

            if not row.get("response", "").strip():
                warnings.append(f"Row {index}: empty response")

            confidence = row.get("confidence_score", "").strip()
            if confidence:
                try:
                    confidence_value = float(confidence)
                    if not 0.0 <= confidence_value <= 1.0:
                        errors.append(f"Row {index}: confidence_score {confidence_value} out of range [0.0, 1.0]")
                except ValueError:
                    errors.append(f"Row {index}: confidence_score '{confidence}' is not a valid float")
            else:
                warnings.append(f"Row {index}: empty confidence_score")

            risk = row.get("risk_level", "").strip().lower()
            if risk and risk not in VALID_RISK_LEVEL:
                errors.append(f"Row {index}: invalid risk_level '{risk}'")
            elif not risk:
                warnings.append(f"Row {index}: empty risk_level")

            pii = row.get("pii_detected", "").strip().lower()
            if pii and pii not in VALID_PII_DETECTED:
                errors.append(f"Row {index}: invalid pii_detected '{pii}'")
            elif not pii:
                warnings.append(f"Row {index}: empty pii_detected")

            language = row.get("language", "").strip().lower()
            if not language:
                warnings.append(f"Row {index}: empty language")
            elif len(language) > 5:
                warnings.append(f"Row {index}: language '{language}' seems too long for ISO 639-1")

            actions = row.get("actions_taken", "").strip()
            if not actions:
                warnings.append(f"Row {index}: actions_taken is empty (expected '[]' if no actions)")
            else:
                try:
                    parsed = json.loads(actions)
                    if not isinstance(parsed, list):
                        errors.append(f"Row {index}: actions_taken must be a JSON array")
                except json.JSONDecodeError as exc:
                    errors.append(f"Row {index}: actions_taken is not valid JSON ({exc})")

    print("=" * 60)
    print("MLE Hiring Challenge - Output Validation Report")
    print("=" * 60)
    print(f"\nInput tickets:  {input_count}")
    print(f"Output rows:    {output_count}")
    print(f"Columns found:  {len(actual_headers)}/{len(EXPECTED_HEADERS)}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for error in errors[:20]:
            print(f"   - {error}")
        if len(errors) > 20:
            print(f"   ... and {len(errors) - 20} more errors")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for warning in warnings[:10]:
            print(f"   - {warning}")
        if len(warnings) > 10:
            print(f"   ... and {len(warnings) - 10} more warnings")

    if not errors:
        print("\nPASS: Output format is valid.")
        print("   Note: This validates structure only, NOT correctness.")
        print("   Your submission will also be evaluated on a hidden test set.")
    else:
        print(f"\nFAIL: {len(errors)} errors found. Fix them before submitting.")

    print("=" * 60)
    return len(errors) == 0


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
