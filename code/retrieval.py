from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from models import RetrievedDoc


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,}", re.I)
ISO_DATE_RE = re.compile(r'(?:last_updated_iso|last_modified):\s*"([^"]+)"', re.I)
EXACT_DATE_RE = re.compile(r'last_updated_exact:\s*"([^"]+)"', re.I)
STOPWORDS = {
    "the", "and", "for", "you", "your", "with", "that", "this", "from", "are",
    "was", "were", "have", "has", "can", "how", "what", "why", "when", "please",
    "help", "need", "issue", "support", "about", "into", "after", "before",
}


class CorpusIndex:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.docs: list[dict] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_len = 1.0
        self._load()

    def _load(self) -> None:
        data_root = self.root / "data"
        lengths: list[int] = []
        for path in sorted(data_root.rglob("*.md")):
            rel = path.relative_to(self.root).as_posix()
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            title = _title_for(path, text)
            tokens = _tokenize(f"{rel} {title} {text}")
            counts = Counter(tokens)
            for token in counts:
                self.doc_freq[token] += 1
            lengths.append(len(tokens) or 1)
            self.docs.append(
                {
                    "path": rel,
                    "title": title,
                    "text": text,
                    "tokens": counts,
                    "length": len(tokens) or 1,
                    "updated_at": _updated_timestamp(text),
                }
            )
        if lengths:
            self.avg_len = sum(lengths) / len(lengths)

    def search(
        self,
        query: str,
        company_hint: str = "",
        limit: int = 4,
        min_score: float = 2.0,
        min_term_matches: int = 2,
    ) -> list[RetrievedDoc]:
        terms = _tokenize(query)
        if not terms:
            return []
        query_counts = Counter(terms)
        scored: list[tuple[float, int, dict]] = []
        n_docs = max(1, len(self.docs))
        for doc in self.docs:
            score = 0.0
            matched_terms = 0
            for term, q_count in query_counts.items():
                tf = doc["tokens"].get(term, 0)
                if not tf:
                    continue
                matched_terms += 1
                idf = math.log(1 + (n_docs - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                denom = tf + 1.2 * (1 - 0.75 + 0.75 * doc["length"] / self.avg_len)
                score += idf * (tf * 2.2 / denom) * min(q_count, 3)

            lower_path = doc["path"].lower()
            hint = company_hint.lower()
            hint_supported = hint and hint != "none" and _company_terms_present(query, hint)
            if hint_supported:
                if _path_matches_company(lower_path, hint):
                    score *= 1.3
                else:
                    score *= 0.25
            if any(area in lower_path for area in ("billing", "refund", "dispute", "fraud", "account", "security", "privacy")):
                score *= 1.05
            if score >= min_score and matched_terms >= min_term_matches:
                scored.append((score, matched_terms, doc))

        scored.sort(key=lambda item: (-item[0], -item[1], -item[2]["updated_at"], item[2]["path"]))
        results: list[RetrievedDoc] = []
        seen_paths: set[str] = set()
        for score, _, doc in scored:
            if doc["path"] in seen_paths:
                continue
            seen_paths.add(doc["path"])
            results.append(
                RetrievedDoc(
                    path=doc["path"],
                    title=doc["title"],
                    snippet=_snippet(doc["text"], terms),
                    score=round(float(score), 4),
                )
            )
            if len(results) >= limit:
                break
        return results

    def source_documents(self, results: list[RetrievedDoc], limit: int = 4) -> str:
        valid_paths: list[str] = []
        for doc in results:
            normalized = doc.path.replace("\\", "/").strip()
            if (
                normalized
                and normalized not in valid_paths
                and normalized.startswith("data/")
                and (self.root / normalized).is_file()
            ):
                valid_paths.append(normalized)
            if len(valid_paths) >= limit:
                break
        return "|".join(valid_paths)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS and len(t) > 1]


def _title_for(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:120]
    return path.stem.replace("-", " ").replace("_", " ").title()


def _snippet(text: str, terms: list[str]) -> str:
    paragraphs = [p.strip().replace("\n", " ") for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return ""
    term_set = set(terms)
    best = max(paragraphs[:80], key=lambda p: sum(1 for t in term_set if t in p.lower()))
    return best[:700]

def _updated_timestamp(text: str) -> int:
    iso_match = ISO_DATE_RE.search(text)
    if iso_match:
        value = iso_match.group(1).replace("Z", "+00:00")
        try:
            return int(datetime.fromisoformat(value).timestamp())
        except ValueError:
            pass

    exact_match = EXACT_DATE_RE.search(text)
    if exact_match:
        value = exact_match.group(1)
        for fmt in ("%b %d, %Y, %I:%M %p", "%B %d, %Y, %I:%M %p"):
            try:
                return int(datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).timestamp())
            except ValueError:
                continue
    return 0


def _company_terms_present(query: str, hint: str) -> bool:
    lowered = query.lower()
    aliases = {
        "claude": ("claude", "anthropic"),
        "devplatform": ("devplatform", "hackerrank"),
        "visa": ("visa",),
    }
    return any(term in lowered for term in aliases.get(hint, (hint,)))


def _path_matches_company(path: str, hint: str) -> bool:
    path_aliases = {
        "claude": ("data/claude/",),
        "devplatform": ("data/devplatform/",),
        "visa": ("data/visa/",),
    }
    return any(path.startswith(prefix) for prefix in path_aliases.get(hint, (f"data/{hint}/",)))
