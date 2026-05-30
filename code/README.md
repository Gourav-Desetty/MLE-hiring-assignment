# Support Triage Agent

Terminal-based support ticket triage agent for the MLE Hiring Challenge. The agent reads `support_tickets/support_tickets.csv`, uses the local `data/` support corpus for retrieval-grounded decisions, and writes predictions to `support_tickets/output.csv`.

## Requirements

- Python 3.11+
- Dependencies from the repository root `requirements.txt`
- No API key is required for the default deterministic run

Install dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

## Run The Agent

From the repository root:

```powershell
python code\main.py
```

Output is written to:

```text
support_tickets/output.csv
```

The default and recommended final-submission mode is deterministic:

```text
TRIAGE_USE_LLM=0
```

This avoids provider latency, rate limits, and nondeterministic API behavior.

## Validate Output Format

After running the agent:

```powershell
python code\validate_output.py
```

The validator checks row count and required output columns. It does not judge answer correctness.

## Optional LLM Provider Mode

The pipeline can optionally call a structured LLM provider for decision generation, while still falling back to deterministic rules if the provider fails or returns malformed JSON.

Create a `.env` file in the repository root. Do not commit `.env`.

OpenAI example:

```text
TRIAGE_USE_LLM=1
TRIAGE_LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=12
TRIAGE_LLM_MAX_FAILURES=3
```

Groq example:

```text
TRIAGE_USE_LLM=1
TRIAGE_LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TIMEOUT_SECONDS=8
TRIAGE_LLM_MAX_FAILURES=3
```

For final reproducibility, switch back to:

```text
TRIAGE_USE_LLM=0
```

## Main Files

- `main.py`: batch entry point, CSV parsing, pipeline execution, and CSV writing.
- `orchestrator.py`: safety scan, PII scan, retrieval, LLM/fallback decision, calibration, citation handling, and tool guardrails.
- `retrieval.py`: deterministic BM25-style local corpus retrieval.
- `safety.py`: adversarial and social-engineering detection.
- `pii.py`: regex-based PII detection and redaction with Luhn validation for payment cards.
- `llm.py`: optional OpenAI/Groq structured LLM adapter with deterministic fallback.
- `models.py`: strict Pydantic input, decision, and output models.
- `tools.py`: internal tool schema loading and action validation.
- `language.py`: lightweight deterministic language detection.
- `validate_output.py`: structural output validator.
- `ARCHITECTURE.md`: technical design document and self-assessment.

## Notes For Reviewers

The agent is designed to be deterministic by default, robust to malformed tickets, conservative around high-risk account or billing actions, and grounded in the local support corpus. It preserves required output columns and validates internal tool calls before serialization.
