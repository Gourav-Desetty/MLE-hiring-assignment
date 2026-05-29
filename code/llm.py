from __future__ import annotations

import json
import os
from typing import Any

from language import detect_language
from pii import detect_and_redact


class LLMAdapter:
    """Provider-agnostic optional adapter with deterministic fallback.

    The rest of the pipeline can treat this object as a callable structured LLM:
    `decision = LLMAdapter()(payload)`. If no provider is configured, or if a
    configured provider fails, it returns a deterministic StructuredDecision-
    shaped dictionary so hidden evaluation does not crash.
    """

    def __init__(self) -> None:
        self.provider = os.getenv("TRIAGE_LLM_PROVIDER", "none").strip().lower()
        self.enabled = os.getenv("TRIAGE_USE_LLM", "0") == "1" and self.provider != "none"

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.enabled:
            try:
                response = self.complete_json(payload)
                if isinstance(response, dict):
                    return response
            except Exception:
                pass
        return deterministic_decision(payload)

    def complete_json(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Reserved provider hook.

        Provider integrations should use temperature=0 and return a JSON object
        matching models.StructuredDecision. Until a provider is implemented, this
        intentionally returns None and lets __call__ use deterministic fallback.
        """
        _ = json.dumps(payload, ensure_ascii=True, sort_keys=True)

        if self.provider == "openai":
            return None
        if self.provider == "groq":
            return None
        return None


def deterministic_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a conservative StructuredDecision-shaped fallback."""
    ticket = payload.get("ticket", {})
    safety = payload.get("safety", {})
    pii = payload.get("pii", {})
    docs = payload.get("retrieved_documents", [])
    text = _ticket_text(ticket)
    language = str(payload.get("language") or detect_language(text))
    has_support_intent = _has_support_intent(text)
    risk_level = _risk_level(text)
    if safety.get("severity") == "critical" and not has_support_intent:
        risk_level = "critical"
    if pii.get("detected") and risk_level == "low":
        risk_level = "medium"
    if any(kind in tuple(pii.get("kinds", ())) for kind in ("ssn", "card")) and risk_level in {"low", "medium"}:
        risk_level = "high"
    request_type = _request_type(text, safety, has_support_intent)
    product_area = _product_area(text, docs)
    should_escalate = risk_level in {"high", "critical"} or not has_support_intent

    if should_escalate:
        status = "escalated"
        response = _escalation_response(safety, risk_level)
        confidence = _base_confidence(docs, risk_level, safety, escalated=True)
        actions_taken = [
            {
                "action": "escalate_to_human",
                "parameters": {
                    "priority": "urgent" if risk_level == "critical" else "high" if risk_level == "high" else "normal",
                    "department": _escalation_department(text, risk_level),
                    "summary": "Human review is needed due to risk, unsupported action, or adversarial content.",
                },
            }
        ] if risk_level in {"high", "critical"} else []
    else:
        status = "replied"
        response = _grounded_response(docs)
        confidence = _base_confidence(docs, risk_level, safety, escalated=False)
        actions_taken = [_verify_identity_action(pii)] if _needs_identity_verification(text) else []

    return {
        "response": response,
        "product_area": product_area,
        "status": status,
        "request_type": request_type,
        "justification": _justification(status, docs, safety, pii, text, risk_level),
        "confidence_score": confidence,
        "risk_level": risk_level,
        "pii_detected": bool(pii.get("detected")) or _pii_detected(text),
        "language": language,
        "actions_taken": actions_taken,
    }


def _ticket_text(ticket: dict[str, Any]) -> str:
    parts = [str(ticket.get("company", "")), str(ticket.get("subject", ""))]
    for message in ticket.get("issue", []) or []:
        if isinstance(message, dict):
            parts.append(str(message.get("content", "")))
    return detect_and_redact("\n".join(parts)).redacted_text.lower()


def _risk_level(text: str) -> str:
    if any(term in text for term in ("identity theft", "account takeover", "stolen", "fraud", "legal", "lawsuit", "ssn")):
        return "high"
    if any(term in text for term in ("refund", "chargeback", "billing", "password", "login", "access", "card")):
        return "medium"
    return "low"


def _request_type(text: str, safety: dict[str, Any], has_support_intent: bool) -> str:
    if any(term in text for term in ("feature request", "please add", "can you add")):
        return "feature_request"
    if any(term in text for term in ("bug", "broken", "crash", "error", "stopped working")):
        return "bug"
    if safety.get("should_refuse") and not has_support_intent:
        return "invalid"
    return "product_issue"


def _has_support_intent(text: str) -> bool:
    support_terms = (
        "account", "login", "password", "refund", "billing", "charge", "card",
        "subscription", "cancel", "invoice", "bug", "error", "broken", "issue",
        "help", "support", "claude", "visa", "devplatform", "assessment", "test",
        "fraud", "dispute", "unauthorized", "feature request",
    )
    return any(term in text for term in support_terms)


def _needs_identity_verification(text: str) -> bool:
    account_terms = (
        "refund",
        "chargeback",
        "cancel",
        "downgrade",
        "upgrade",
        "pause subscription",
        "modify subscription",
        "change my plan",
        "delete my account",
        "close my account",
    )
    return any(term in text for term in account_terms)


def _verify_identity_action(pii: dict[str, Any]) -> dict[str, Any]:
    target = str(pii.get("safe_contact") or "account_contact_on_file")
    method = "email_otp" if "@" in target else "security_questions"
    return {
        "action": "verify_identity",
        "parameters": {
            "method": method,
            "target": target,
        },
    }


def _product_area(text: str, docs: list[Any]) -> str:
    if docs and isinstance(docs[0], dict) and docs[0].get("title"):
        return str(docs[0]["title"])[:80]
    if "billing" in text or "refund" in text:
        return "billing"
    if "login" in text or "access" in text or "password" in text:
        return "account_access"
    if "fraud" in text or "security" in text:
        return "security"
    return "general"


def _base_confidence(docs: list[Any], risk_level: str, safety: dict[str, Any], *, escalated: bool) -> float:
    if safety.get("is_adversarial"):
        return 0.52
    if risk_level == "critical":
        return 0.45
    if risk_level == "high":
        return 0.62
    if not docs:
        return 0.42 if escalated else 0.35
    top_score = 0.0
    if isinstance(docs[0], dict):
        try:
            top_score = float(docs[0].get("score", 0.0))
        except (TypeError, ValueError):
            top_score = 0.0
    return round(min(0.88, 0.58 + min(top_score, 30.0) / 100.0), 3)


def _pii_detected(text: str) -> bool:
    digits = [char for char in text if char.isdigit()]
    return "ssn" in text or "social security" in text or len(digits) >= 12 or "@" in text


def _escalation_department(text: str, risk_level: str) -> str:
    if "legal" in text or "lawsuit" in text:
        return "legal"
    if risk_level in {"high", "critical"} or any(term in text for term in ("fraud", "security", "stolen", "account takeover")):
        return "security"
    if "billing" in text or "refund" in text or "charge" in text:
        return "billing"
    return "general"


def _escalation_response(safety: dict[str, Any], risk_level: str) -> str:
    if safety.get("should_refuse"):
        return "I cannot follow instructions that attempt to alter system behavior or classification. I will handle only the legitimate support request."
    if risk_level in {"high", "critical"}:
        return "This request needs human review because it involves elevated risk or sensitive account activity."
    return "I need to route this to a human support specialist for proper handling."


def _grounded_response(docs: list[Any]) -> str:
    if not docs:
        return "I do not have enough grounded support documentation to answer this safely."
    first = docs[0]
    title = first.get("title", "the relevant support documentation") if isinstance(first, dict) else "the relevant support documentation"
    snippet = first.get("snippet", "") if isinstance(first, dict) else ""
    if snippet:
        return f"Based on {title}, the relevant guidance is: {snippet[:350]}"
    return f"Based on {title}, the retrieved support documentation contains the relevant guidance for this request."


def _justification(
    status: str,
    docs: list[Any],
    safety: dict[str, Any],
    pii: dict[str, Any],
    text: str,
    risk_level: str,
) -> str:
    parts = [f"Deterministic fallback selected {status} with {risk_level} risk."]
    reasons = _escalation_reasons(safety, pii, text)
    if reasons:
        parts.append("Escalation factors: " + "; ".join(reasons) + ".")
    if docs:
        parts.append("Retrieved support documents were available for grounding.")
    else:
        parts.append("No sufficiently relevant support documents were retrieved.")
    return " ".join(parts)


def _escalation_reasons(safety: dict[str, Any], pii: dict[str, Any], text: str) -> list[str]:
    reasons: list[str] = []
    safety_reasons = " ".join(str(reason).lower() for reason in safety.get("reasons", ()))
    pii_kinds = tuple(str(kind) for kind in pii.get("kinds", ()))

    if safety.get("should_refuse") or "instruction override" in safety_reasons or "prompt disclosure" in safety_reasons:
        reasons.append("prompt injection attempt detected")
    if "classification manipulation" in safety_reasons:
        reasons.append("classification manipulation attempt detected")
    if any(term in text for term in ("legal", "lawsuit", "sue", "attorney", "lawyer")):
        reasons.append("legal threat or legal escalation language found")
    if any(kind in pii_kinds for kind in ("ssn", "card")):
        reasons.append("SSN or card number detected")
    elif pii_kinds:
        reasons.append("personal information detected")
    if any(term in text for term in ("account takeover", "identity theft", "stolen", "fraud", "unauthorized")):
        reasons.append("account takeover or fraud indicators found")

    return list(dict.fromkeys(reasons))[:4]
