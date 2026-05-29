# Support Triage Agent Architecture

## System Overview And Design Goals

This repository implements a terminal-based support triage agent for the MLE Hiring Challenge. The agent processes support tickets across DevPlatform, Claude, Visa, and ambiguous cross-domain cases, then writes a deterministic `support_tickets/output.csv` matching the required evaluation schema.

The core design goals are:

- **Grounded support handling:** use only the local `data/` corpus for factual support responses and source attribution.
- **Adversarial robustness:** detect prompt injection, classification manipulation, social engineering, and data-exfiltration attempts before decision making.
- **PII safety:** detect and redact sensitive information before model-facing payloads and before final response serialization.
- **Determinism:** avoid randomness and network dependencies by default; use stable lexical retrieval and deterministic fallback decision logic.
- **Evaluator compatibility:** preserve required columns, valid enum values, valid JSON tool calls, row count alignment, and reproducible CLI execution.
- **Graceful degradation:** avoid crashing on malformed tickets, missing tool specs, malformed conversations, or provider failures.

## Pipeline Flow

```text
support_tickets.csv
    |
    v
Input Parsing
    |
    v
Safety & PII Scanning
    |
    v
Retrieval (BM25)
    |
    v
Orchestration (Decision Logic)
    |
    v
Output Serialization
    |
    v
support_tickets/output.csv
```

### 1. Input Parsing

`main.py` reads `support_tickets/support_tickets.csv`, parses each `issue` field as a JSON conversation array, normalizes malformed rows into a conservative user-message shape, and constructs a strict `TicketInput` model. It preloads the retrieval index and LLM adapter once for the full batch run.

If a row cannot be processed safely, `main.py` writes a structurally valid escalated fallback row rather than crashing the batch job. The fallback does not expose raw exception details in the user-facing output.

### 2. Safety & PII Scanning

`safety.py` audits the subject, company field, and every conversation turn independently. It detects:

- instruction overrides
- system/developer role injection
- prompt or corpus disclosure attempts
- data exfiltration requests
- jailbreak personas
- classification manipulation
- evaluation manipulation
- multilingual instruction override patterns
- social engineering and suspicious authority claims
- obfuscated payloads through normalization, URL decoding, HTML unescaping, base64, ROT13, and zero-width character removal

`pii.py` performs deterministic regex-based PII detection and redaction for:

- email addresses
- phone numbers
- SSNs
- payment cards, only when Luhn-valid
- addresses
- labeled account/customer/user identifiers

The PII layer avoids treating long order, tracking, reference, transaction, ticket, or case numbers as payment cards unless they pass Luhn. It also avoids reclassifying those operational IDs as phone numbers when they appear in operational-ID contexts.

### 3. Retrieval (BM25)

`retrieval.py` builds an in-memory lexical index over all Markdown files in `data/`. It uses a BM25-style scoring formula with:

- tokenization over file path, title, and document text
- lightweight stopword filtering
- a small stem map for common support variants such as `cancellation -> cancel`
- company/domain-aware score adjustments when the ticket itself supports the company hint
- small boosts for risk-sensitive areas such as billing, fraud, account, security, privacy, refund, and dispute
- deterministic sorting by score, matched terms, document recency metadata, and path

When a ticket contains malicious instructions plus a real support request, `orchestrator.py` builds a retrieval-only sanitized query by removing adversarial clauses such as "ignore previous instructions" or "reveal the system prompt." This allows hybrid handling: the malicious instruction is ignored, but the safe support request can still retrieve relevant documentation.

### 4. Orchestration (Decision Logic)

`orchestrator.py` coordinates the full decision path:

1. Detect safety findings.
2. Detect and redact PII.
3. Retrieve support documents using a sanitized query.
4. Build a structured payload with redacted ticket content, safety findings, PII metadata, language hint, and retrieved documents.
5. Call the provider-agnostic LLM adapter.
6. Validate the returned decision with `StructuredDecision`.
7. Apply deterministic post-processing:
   - final response redaction
   - forced `pii_detected` from regex findings
   - risk elevation for SSNs/cards and other PII
   - confidence reduction for high-risk, adversarial, unsupported, or contradictory cases
   - source-document validation and citation filtering
   - destructive-action guardrails
   - concise safety/PII escalation factors in justification
   - deterministic language output

The orchestrator treats `should_refuse` as "ignore malicious instructions," not "drop the whole ticket." Pure prompt-disclosure or meta-abuse tickets become invalid/escalated, while mixed malicious-plus-support tickets continue through retrieval and support handling.

### 5. Output Serialization

`models.py` defines strict Pydantic models for all input, intermediate, and output structures. `TicketOutput.to_csv_row()` serializes every required output column:

- `issue`
- `subject`
- `company`
- `response`
- `product_area`
- `status`
- `request_type`
- `justification`
- `confidence_score`
- `source_documents`
- `risk_level`
- `pii_detected`
- `language`
- `actions_taken`

`validate_output.py` checks output headers, row count, enum values, confidence ranges, PII flags, language formatting, and JSON validity for `actions_taken`.

## File Guide

- `main.py`: CLI entry point. Reads the input CSV, initializes the retrieval index and LLM adapter, processes every row, and writes `output.csv`.
- `models.py`: Strict Pydantic schema layer for tickets, retrieved documents, safety findings, structured decisions, PII findings, and CSV serialization.
- `orchestrator.py`: Main control plane for safety, PII, retrieval, structured decision parsing, action guarding, confidence calibration, citation handling, and final output construction.
- `retrieval.py`: Deterministic BM25-style corpus index over local Markdown documents with lightweight stemming and source path validation.
- `safety.py`: Rule-based adversarial and social-engineering detector with normalization and decoding surfaces.
- `pii.py`: Regex-based PII detector and redactor, including Luhn validation for payment cards and context-aware phone/card false-positive controls.
- `tools.py`: Tool schema loader and validator for `actions_taken`; falls back to embedded schemas if `internal_tools.json` is missing or malformed.
- `llm.py`: Provider-agnostic LLM adapter with deterministic fallback decision logic. Provider hooks are present but disabled unless explicitly configured.
- `language.py`: Lightweight deterministic language detector for English, French, Spanish, German, and Chinese.
- `validate_output.py`: Structural validator for `support_tickets/output.csv`.

## Security Design

The security model is layered rather than delegated to a model prompt.

Adversarial handling:

- Safety scanning happens before retrieval or decision making.
- Prompt injection and classification manipulation are recorded in structured safety metadata.
- Malicious clauses are removed from retrieval queries so the retriever does not over-focus on prompt-disclosure text.
- The final justification records concise factors such as prompt injection, classification manipulation, legal language, SSN/card detection, or account takeover indicators.
- Pure meta-abuse is escalated/invalid; mixed support tickets continue with malicious instructions ignored.

PII protection:

- PII is detected before payload construction.
- LLM-facing ticket content is redacted.
- Final responses are redacted again before serialization.
- The `pii_detected` flag is forced from deterministic regex findings, not left solely to the decision layer.
- SSNs and Luhn-valid cards elevate risk.

Tool safety:

- `tools.py` validates action schemas, required parameters, primitive types, and enum-like strings.
- `orchestrator.py` blocks destructive actions (`issue_refund`, `modify_subscription`) unless identity was already verified in the conversation.
- If a model emits `verify_identity` and a destructive action in the same turn, the destructive action is still withheld. Verification must happen first.
- Risky billing/account intents generate `verify_identity` when no action is otherwise produced.

Citation safety:

- Output citations are clamped to existing repository paths.
- Invalid, unsupported, or pure meta-abuse tickets keep `source_documents` empty.
- Escalated support tickets can preserve citations when retrieved documents grounded the decision.

## Retrieval Strategy

The retrieval layer deliberately uses deterministic lexical retrieval rather than embeddings. This keeps the system easy to reproduce, fast to run, and independent of external services.

The index includes path, title, and body text, because path names often carry high-value domain signals such as `billing`, `fraud`, `security`, or `subscription`. BM25 scoring rewards exact and repeated term overlap while normalizing document length. A small stem map improves recall for common support variants without adding a dependency or making the matching opaque.

The retrieval strategy favors conservative evidence:

- only local Markdown files under `data/` are indexed
- source paths must physically exist before being emitted
- company hints are boosted only when the ticket text itself supports the company
- high-risk domain paths get a modest boost
- results are sorted deterministically

Known limitation: lexical retrieval can still retrieve semantically adjacent but imperfect documents, especially for compound or vague tickets. The system partially mitigates this with confidence calibration and source eligibility rules, but embeddings or reranking would improve precision.

## Decision Pipeline

The default decision path is deterministic. `LLMAdapter` has provider hooks, but without explicit environment configuration it returns a structured fallback decision. This is intentional for hidden evaluation robustness: the agent should run without API keys or network access.

The fallback decision uses:

- support-intent heuristics
- retrieved document availability
- safety severity
- PII categories
- risk keywords
- action intent keywords
- source availability

It chooses between `replied` and `escalated`, identifies request type, sets risk level, chooses product area, emits safe actions, and creates concise justifications. The orchestrator then validates and post-processes that decision before output.

Confidence is deterministic and bounded. It is reduced for:

- high and critical risk
- adversarial tickets
- safety refusal findings
- missing source documents for replies
- company/content contradictions

This is not true statistical calibration, but it avoids extreme overconfidence in known hard cases.

## Failure Handling

The system is designed to preserve output structure under failure:

- malformed `issue` values are converted into user-message arrays when possible
- row-level exceptions become valid escalated fallback rows
- missing or malformed `internal_tools.json` falls back to embedded tool schemas
- invalid model/tool actions are converted to safe escalation actions where possible
- source citations are clamped to valid repository paths
- provider failures fall back to deterministic decision logic
- console output is ASCII-safe for Windows terminals

The batch job is expected to produce the same number of output rows as input rows even when individual tickets are malformed.

## Trade Offs

- **Determinism over fluency:** responses are short and mostly snippet-based. This reduces hallucination risk but can sound less natural than a fully generative LLM response.
- **Rule-based safety over broad model judgment:** explicit regex detectors are explainable and fast, but hidden adversarial patterns can still bypass them.
- **BM25 over embeddings:** lexical retrieval is reproducible and dependency-light, but less robust to paraphrase and semantic drift.
- **Conservative action execution:** destructive actions are delayed behind identity verification. This may under-execute some valid requests but reduces severe tool misuse risk.
- **Simple language detection:** heuristic language detection handles common cases for `en`, `fr`, `es`, `de`, and `zh`, but it is not a full multilingual classifier.
- **Fallback tool schemas:** embedded schemas protect runtime stability if the JSON spec is missing, but they must be kept aligned with `data/api_specs/internal_tools.json` if that file changes.

## Future Improvements

- **Structured LLM provider integration:** implement OpenAI, Anthropic, Groq, or another provider behind `LLMAdapter.complete_json()` with temperature `0`, strict JSON schema validation, timeout handling, and deterministic fallback on provider failure.
- **Hybrid retrieval (BM25 + embeddings):** combine lexical BM25 with local or API-based embeddings, then rerank by domain, recency, and source agreement.
- **Confidence calibration using evaluation data:** use visible validation outcomes or a held-out labeled set to fit confidence values instead of relying on fixed heuristic caps.
- **More advanced multilingual support:** replace marker-based detection with a compact deterministic language identifier and add multilingual safety and retrieval normalization beyond the current high-value phrases.

## Self-Assessment

Estimated performance by evaluation dimension:

- Adversarial Robustness: 7/10. The safety layer covers direct, obfuscated, multilingual, and classification-manipulation attacks, but regex-based detection may miss novel hidden patterns.
- Escalation Precision: 6/10. Sensitive PII, fraud, account takeover, legal language, and unsupported cases are handled conservatively; some answerable account-access cases may still be over- or under-escalated.
- Response Quality: 5/10. Responses are grounded and deterministic, but snippet-based answers can be incomplete or less polished for compound tickets.
- Source Attribution: 7/10. Paths are validated and citations are preserved for grounded escalations, but lexical retrieval can still cite adjacent documents.
- Tool Calling & Action Execution: 7/10. Tool schemas and identity prerequisites are enforced; action selection remains heuristic.
- PII Detection & Handling: 7/10. Email, phone, SSN, address, account IDs, and Luhn-valid cards are covered with final redaction, but names and some international PII formats are limited.
- Architecture & Code Quality: 8/10. Modules are separated by concern, deterministic, and runnable; deeper tests and provider integration would improve confidence.
- Confidence Calibration: 5/10. Confidence is bounded and risk-aware but not statistically calibrated.
- Determinism & Reproducibility: 8/10. No random sampling is used and default execution is local/deterministic.

Three hardest visible-ticket categories:

- **Mixed adversarial plus legitimate support requests:** handled by ignoring malicious clauses for retrieval while preserving safety factors in justification.
- **High-risk fraud/account takeover tickets:** handled through escalation, security-oriented actions, PII redaction, and grounded citations where available.
- **Account or billing action requests:** handled through `verify_identity` before destructive tools, even if a model attempts same-turn verification plus action execution.

Predicted hidden adversarial categories:

- prompt injections written in less common languages
- benign-looking requests containing embedded data-exfiltration instructions
- tickets that mix valid support needs with classification manipulation
- operational IDs that resemble PII or payment cards
- misleading company labels that conflict with issue content

Known failure mode:

The largest remaining weakness is response quality for compound, nuanced, or semantically paraphrased tickets. The deterministic fallback avoids hallucination, but it can retrieve imperfect sources and produce terse snippet-based responses. A structured LLM provider with strict grounding and a stronger retrieval reranker would improve this.
