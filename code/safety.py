from __future__ import annotations

import base64
import codecs
import html
import re
import unicodedata
from urllib.parse import unquote

from models import SafetyFinding, TicketInput


ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=])")


# Refusal means the user is asking the agent to do an unsafe meta-action, such
# as leaking instructions or obeying a jailbreak. The legitimate support issue
# can still be answered if one exists after ignoring the adversarial text.
REFUSAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction override",
        re.compile(
            r"\b(ignore|disregard|forget|bypass|override|violate|skip)\b.{0,120}"
            r"\b(previous|prior|above|system|developer|instructions?|rules?|policy|safety|guardrails?)\b",
            re.I | re.S,
        ),
    ),
    (
        "system role injection",
        re.compile(
            r"(\[system\b|\bsystem\s*(override|message|update|instruction)\b|"
            r"<\s*/?\s*system\s*>|\bdeveloper message\b|\bdeveloper\s*(update|instruction)\b|"
            r"\byou are now\b|\bact as\b.{0,50}\b(system|developer|admin|root)\b|maintenance mode)",
            re.I | re.S,
        ),
    ),
    (
        "prompt disclosure",
        re.compile(
            r"\b(show|print|reveal|dump|leak|exfiltrate|share|give|copy|display)\b.{0,160}"
            r"\b(prompt|system instructions?|developer instructions?|hidden rules?|policy|guardrails?|"
            r"chain[- ]?of[- ]?thought|internal reasoning|retrieval algorithm|knowledge base|corpus|documents?)\b",
            re.I | re.S,
        ),
    ),
    (
        "data exfiltration",
        re.compile(
            r"\b(upload|post|send|forward|copy|paste)\b.{0,140}"
            r"\b(corpus|documents?|tickets?|customer data|customer information|internal data|all data)\b"
            r".{0,140}\b(https?://|webhook|email|server|endpoint)\b",
            re.I | re.S,
        ),
    ),
    (
        "jailbreak persona",
        re.compile(
            r"\b(DAN|do anything now|jailbreak|unrestricted mode|no safety protocols?|"
            r"developer mode|god mode|evil assistant)\b",
            re.I,
        ),
    ),
    (
        "classification manipulation",
        re.compile(
            r"("
            r"\b(classify|mark|label|set|output|return|write)\b.{0,100}"
            r"\b(status|request_type|risk_level|confidence_score|actions_taken|this|ticket|case)\b.{0,100}"
            r"\b(replied|escalated|invalid|low|1\.?0?|empty|\[\])\b"
            r"|"
            r"\b(status|request_type|risk_level|confidence_score|actions_taken)\s*[:=]\s*"
            r"(replied|escalated|invalid|low|1\.?0?|\[\])"
            r")",
            re.I | re.S,
        ),
    ),
    (
        "evaluation manipulation",
        re.compile(
            r"\b(hidden test|evaluation|evaluator|grader|scoring|rubric)\b.{0,120}"
            r"\b(pass|maximize|game|cheat|ignore|do not reveal)\b",
            re.I | re.S,
        ),
    ),
    (
        "multilingual instruction override",
        re.compile(
            r"\b(ignorieren|ignorez|ignora|ignorer|ignorar|\u5ffd\u7565|\u7121\u8996|\ubb34\uc2dc)\b.{0,100}"
            r"\b(systemanweisungen|instructions?|instrucciones|\u6307\u4ee4|\u6307\u793a|\uaddc\uce59|\u30eb\u30fc\u30eb)\b",
            re.I | re.S,
        ),
    ),
)


# Suspicious means it should increase risk and be mentioned in justification,
# but it is not automatically a reason to refuse all support handling.
SUSPICIOUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "authority social engineering",
        re.compile(
            r"\b(i am|as a|this is)\b.{0,80}"
            r"\b(employee|internal|trust\s*&?\s*safety|senior qa|auditor|monitoring system)\b"
            r".{0,140}\b(bypass|override|refund|credit|delete|unlock|restore|reveal|escalate|approve)\b",
            re.I | re.S,
        ),
    ),
    (
        "admin authority pressure",
        re.compile(
            r"\b(i am|as an?|this is)\b.{0,60}\b(admin|owner)\b.{0,120}"
            r"\b(bypass|override|without verification|ignore policy|restore access|refund|delete)\b",
            re.I | re.S,
        ),
    ),
    (
        "refund social engineering",
        re.compile(
            r"\b(previous agent|agent id|colleague|promised|screenshot)\b.{0,140}"
            r"\b(refund|credit|entitled|commitment|approved)\b",
            re.I | re.S,
        ),
    ),
    (
        "spreadsheet formula payload",
        re.compile(r"^\s*[=+\-@].{0,40}(cmd|powershell|calc|hyperlink|importxml|webservice)", re.I | re.M),
    ),
    (
        "fake vulnerability alert",
        re.compile(r"\b(CVE-\d{4}-\d+|automated message from monitoring|urgent security override)\b", re.I),
    ),
    (
        "link-only payload",
        re.compile(r"^\s*(https?://\S+\s*){1,4}$", re.I),
    ),
)


COMPACT_REFUSAL_PHRASES = (
    "ignoreallpreviousinstructions",
    "revealthesystemprompt",
    "showthesystemprompt",
    "printthesystemprompt",
    "dumpallcorpusdocuments",
    "classifythisasreplied",
    "setstatustoreplied",
    "donotescalate",
)


def detect_prompt_injection(text: str) -> SafetyFinding:
    """Detect prompt injection and social-engineering attempts in raw text."""
    return _audit_segments((("text", text),))


def detect_ticket_safety(ticket: TicketInput) -> SafetyFinding:
    """Audit subject, company, and every conversation turn independently."""
    segments: list[tuple[str, str]] = [
        ("company", ticket.company),
        ("subject", ticket.subject),
    ]
    for index, message in enumerate(ticket.issue):
        role = str(message.get("role", "unknown")).strip() or "unknown"
        content = str(message.get("content", ""))
        segments.append((f"turn[{index}].{role}", content))
    return audit_conversation(segments)


def audit_conversation(segments: list[tuple[str, str]] | tuple[tuple[str, str], ...]) -> SafetyFinding:
    """Sequentially audit labeled conversation segments.

    This catches mid-conversation control attempts without relying on the final
    user message or on one large concatenated string. Labels should identify the
    source, e.g. "subject" or "turn[2].user".
    """
    return _audit_segments(tuple(segments))


def _audit_segments(segments: tuple[tuple[str, str], ...]) -> SafetyFinding:
    reasons: list[str] = []
    adversarial_segments: list[str] = []
    refusal_found = False
    suspicious_found = False

    for label, text in segments:
        segment_reasons: list[str] = []
        segment_refusal = False
        segment_suspicious = False
        for surface_name, surface_text in _inspection_surfaces(text):
            surface_reasons, surface_refusal, surface_suspicious = _scan_surface(surface_text)
            segment_refusal = segment_refusal or surface_refusal
            segment_suspicious = segment_suspicious or surface_suspicious
            for reason in surface_reasons:
                surface_reason = reason if surface_name == "raw" else f"{surface_name}: {reason}"
                segment_reasons.append(surface_reason)
                reasons.append(f"{label}: {surface_reason}")

        if segment_reasons:
            adversarial_segments.append(label)
        refusal_found = refusal_found or segment_refusal
        suspicious_found = suspicious_found or segment_suspicious

    unique_reasons = tuple(dict.fromkeys(reasons))
    unique_segments = tuple(dict.fromkeys(adversarial_segments))
    severity = _severity(unique_reasons, refusal_found, suspicious_found)
    return SafetyFinding(
        is_adversarial=bool(unique_reasons),
        reasons=unique_reasons,
        severity=severity,
        should_refuse=refusal_found,
        should_ignore_instructions=bool(unique_reasons),
        audited_turns=len(segments),
        adversarial_turns=unique_segments,
    )


def _scan_surface(text: str) -> tuple[list[str], bool, bool]:
    reasons: list[str] = []
    refusal_found = False
    suspicious_found = False

    normalized = _normalize_text(text)
    for label, pattern in REFUSAL_PATTERNS:
        if pattern.search(normalized):
            reasons.append(label)
            refusal_found = True

    for label, pattern in SUSPICIOUS_PATTERNS:
        if pattern.search(normalized):
            reasons.append(label)
            suspicious_found = True

    compact = re.sub(r"[^a-z0-9]+", "", normalized.lower())
    if any(phrase in compact for phrase in COMPACT_REFUSAL_PHRASES):
        reasons.append("obfuscated instruction override")
        refusal_found = True

    return reasons, refusal_found, suspicious_found


def _inspection_surfaces(text: str) -> list[tuple[str, str]]:
    raw = _normalize_text(text)
    surfaces = [("raw", raw)]
    for label, decoded in _decoded_candidates(raw):
        normalized = _normalize_text(decoded)
        if normalized and normalized != raw:
            surfaces.append((label, normalized))
    return surfaces


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = ZERO_WIDTH_RE.sub("", normalized)
    normalized = html.unescape(normalized)
    normalized = unquote(normalized)
    normalized = normalized.replace("\r", "\n")
    return normalized


def _decoded_candidates(text: str) -> list[tuple[str, str]]:
    decoded: list[tuple[str, str]] = []

    for token in BASE64_RE.findall(text):
        padded = token + ("=" * (-len(token) % 4))
        try:
            raw = base64.b64decode(padded, validate=True)
            value = raw.decode("utf-8", errors="ignore")
        except Exception:
            continue
        if _looks_textual(value):
            decoded.append(("base64", value))

    try:
        rot13 = codecs.decode(text, "rot_13")
    except Exception:
        rot13 = ""
    if rot13 and rot13 != text and _contains_safety_keyword(rot13):
        decoded.append(("rot13", rot13))

    return decoded


def _looks_textual(value: str) -> bool:
    if len(value.strip()) < 8:
        return False
    printable = sum(1 for char in value if char.isprintable() or char.isspace())
    return printable / max(len(value), 1) > 0.85


def _contains_safety_keyword(value: str) -> bool:
    lowered = value.lower()
    return any(
        keyword in lowered
        for keyword in (
            "ignore",
            "system",
            "developer",
            "prompt",
            "instructions",
            "reveal",
            "status",
            "replied",
            "escalate",
        )
    )


def _severity(reasons: tuple[str, ...], refusal_found: bool, suspicious_found: bool) -> str:
    if not reasons:
        return "low"
    if refusal_found:
        return "critical"
    if suspicious_found:
        return "high"
    return "medium"
