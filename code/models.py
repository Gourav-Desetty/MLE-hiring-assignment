from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tools import actions_to_json, validate_actions_taken

# Global target schema fields tracking evaluation order
CSV_HEADERS = [
    "issue", "subject", "company", "response", "product_area", "status",
    "request_type", "justification", "confidence_score", "source_documents",
    "risk_level", "pii_detected", "language", "actions_taken"
]

STRICT_CONFIG = ConfigDict(
    extra="forbid",
    str_strip_whitespace=True,
    validate_assignment=True
)

class TicketInput(BaseModel):
    """Clean representation of incoming CSV ticket data."""
    model_config = STRICT_CONFIG
    
    issue: list[dict[str, str]]
    subject: str = ""
    company: str = "None"

    @field_validator("issue", mode="before")
    @classmethod
    def parse_issue_json(cls, value: any) -> any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("issue column must contain a valid JSON conversation array") from exc
        return value

    def issue_as_json(self) -> str:
        return json.dumps(self.issue, ensure_ascii=False, separators=(",", ":"))


class TicketOutput(BaseModel):
    """Strict serialization layer to guarantee zero output formatting penalties."""
    model_config = STRICT_CONFIG

    issue: str = ""
    subject: str = ""
    company: str = "None"
    response: str = Field(min_length=1)
    product_area: str = Field(min_length=1)
    status: str = "escalated"
    request_type: str = "invalid"
    justification: str = Field(min_length=1)
    confidence_score: float = Field(ge=0.0, le=1.0)
    source_documents: str = ""
    risk_level: str = "low"
    pii_detected: bool = False
    language: str = "en"
    actions_taken: list[dict] = Field(default_factory=list)

    @field_validator("actions_taken")
    @classmethod
    def check_actions_taken(cls, v: list[dict]) -> list[dict]:
        return validate_actions_taken(v)

    @field_validator("status")
    @classmethod
    def check_status(cls, v: str) -> str:
        return v if v in {"replied", "escalated"} else "escalated"

    @field_validator("request_type")
    @classmethod
    def check_request(cls, v: str) -> str:
        return v if v in {"product_issue", "feature_request", "bug", "invalid"} else "invalid"

    @field_validator("risk_level")
    @classmethod
    def check_risk(cls, v: str) -> str:
        return v if v in {"low", "medium", "high", "critical"} else "medium"

    def clamp_citations(self, root_path: Path) -> None:
        """Validates physical repository path constraints to eliminate citation penalties."""
        if not self.source_documents:
            return
        valid_paths = []
        for path in self.source_documents.split("|"):
            normalized = path.replace("\\", "/").strip()
            if normalized and (root_path / normalized).exists() and normalized not in valid_paths:
                valid_paths.append(normalized)
        self.source_documents = "|".join(valid_paths[:4])

    def to_csv_row(self) -> dict[str, str]:
        """Maps data properties strictly matching target evaluator headers layout."""
        return {
            "issue": self.issue,
            "subject": self.subject,
            "company": self.company,
            "response": self.response,
            "product_area": self.product_area,
            "status": self.status,
            "request_type": self.request_type,
            "justification": self.justification,
            "confidence_score": f"{self.confidence_score:.3f}".rstrip("0").rstrip("."),
            "source_documents": self.source_documents,
            "risk_level": self.risk_level,
            "pii_detected": str(self.pii_detected).lower(),
            "language": self.language[:2].lower(),
            "actions_taken": actions_to_json(self.actions_taken)
        }
