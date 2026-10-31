"""Single-source [S1] citation checks, relevance fallback, and conflict awareness."""

import math
import re


NO_EVIDENCE = "I could not find enough information in the indexed documents to answer that."
CONFLICT_WARNING = "The retrieved documents may contain conflicting information. Review the cited sources."
DEFAULT_MIN_RELEVANCE = 0.30
KEYWORD_MIN_COVERAGE = 0.5
CITABLE_LABEL = "S1"
MARKER = re.compile(r"\[\s*[sS]\s*(\d+)\s*\]")
VALID_MARKER = re.compile(r"\[(S[1-4])\]")
STOP_WORDS = set("the and are was were for from with that this these those their they them have has been will would should about into only also than then there here such which what when where who how each both does did is be as of to in on at by a an it its or but per can may must not never required optional allowed permitted prohibited forbidden document documents policy policies source sources states says records record first second differ different disagree conflict whereas versus".split())
UNIT_WORDS = set("day days week weeks month months year years hour hours minute minutes percent user users request requests".split())
NUMBER = re.compile(r"(?<![\w\d])\d+(?:[./:-]\d+)*%?|(?<=\bv)\d+(?:\.\d+)+", re.I)
NEGATIVE = re.compile(r"\b(?:not|never|cannot|can't|prohibited|forbidden|disallowed)\b", re.I)
# Header lines such as "Version: 2.0" or "Effective date: ..." describe a document, not a policy claim.
METADATA_LINE = re.compile(r"^\s*(?:\*\*)?(?:document\s+id|doc\s+id|department|version|effective\s+date|status|"
                           r"document\s+role|role|owner|classification)\s*(?::|\|)", re.I)


def finite_score(value) -> float | None:
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return max(-1.0, min(1.0, float(value)))


def finite_number(value) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def is_relevant(item: dict, minimum=DEFAULT_MIN_RELEVANCE) -> bool:
    """Semantic similarity passes the threshold, or the query's keywords match the passage."""
    score = finite_score(item.get("similarity_score"))
    if score is None:
        return False
    coverage = finite_number(item.get("keyword_coverage")) or 0.0
    return (score >= minimum or coverage >= 1.0
            or (coverage >= KEYWORD_MIN_COVERAGE and score >= minimum / 2))


def has_relevant_evidence(evidence: list[dict], minimum=DEFAULT_MIN_RELEVANCE) -> bool:
    """Only the top-ranked passage can become [S1], so only it can satisfy the gate."""
    return bool(evidence) and is_relevant(evidence[0], minimum)


def _words(text: str) -> set[str]:
    return {word[:-1] if word.endswith("s") and len(word) > 4 else word
            for word in re.findall(r"[a-zA-Z]{3,}", text.lower()) if word not in STOP_WORDS}


def _statements(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+(?!\[)|(?<=\])\s+(?=[A-Z])", text) if part.strip()]


def _claims(text: str) -> list[str]:
    return [statement for statement in _statements(text) if not METADATA_LINE.match(statement)]


def _features(statement: str) -> dict[str, set]:
    features = {}
    for value, unit in re.findall(r"\b(\d+(?:\.\d+)?)\s*(days?|weeks?|months?|years?|hours?|minutes?|percent|%|gb|mb|users?|requests?)(?![A-Za-z])", statement, re.I):
        features.setdefault("quantity:" + unit.lower().rstrip("s"), set()).add(value)
    dates = re.findall(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", statement, re.I)
    if dates:
        features["date"] = {value.lower() for value in dates}
    elif re.search(r"\b(?:effective|expires|deadline|year|starting)\b", statement, re.I):
        years = re.findall(r"\b(?:19|20)\d{2}\b", statement)
        if years:
            features["date"] = set(years)
    versions = re.findall(r"\b(?:version\s*|v)(\d+(?:\.\d+)*)\b", statement, re.I)
    if versions:
        features["version"] = set(versions)
    if re.search(r"\b(?:may|can|cannot|allowed|permitted|prohibited|forbidden|must|optional|required)\b", statement, re.I):
        features["permission"] = {"deny" if NEGATIVE.search(statement) else "allow"}
        if re.search(r"\b(?:must|required|optional)\b", statement, re.I):
            features["requirement"] = {"optional" if re.search(r"\boptional\b|not required", statement, re.I) else "required"}
    # Plain quantities/deadlines without units also need a small, shared-topic signal.
    if re.search(r"\b(?:quantity|maximum|minimum|limit|deadline|count)\b", statement, re.I):
        features["number"] = set(NUMBER.findall(statement))
    return features


def conflict_pairs(evidence: list[dict], minimum=DEFAULT_MIN_RELEVANCE) -> list[tuple[str, str]]:
    """Compare the primary passage with other relevant passages from different documents."""
    pairs = []
    if not evidence:
        return pairs
    first = evidence[0]
    for second in evidence[1:]:
        if first["document_id"] == second["document_id"] or not is_relevant(second, minimum):
            continue
        found = False
        for left in _claims(first["chunk_text"]):
            for right in _claims(second["chunk_text"]):
                left_words, right_words = _words(left) - UNIT_WORDS, _words(right) - UNIT_WORDS
                shared = left_words & right_words
                # One incidental shared word (e.g. "approval") is not a shared topic.
                if not shared or (len(shared) < 2 and shared != min(left_words, right_words, key=len)):
                    continue
                left_features, right_features = _features(left), _features(right)
                if any(left_features[key] and right_features[key] and left_features[key].isdisjoint(right_features[key])
                       for key in left_features.keys() & right_features.keys()):
                    found = True
                    break
            if found:
                break
        if found:
            pairs.append((first["source_label"], second["source_label"]))
    return pairs


def _supported(claim: str, passages: list[dict]) -> bool:
    """Reject novel numbers and obvious polarity reversals; this is not full entailment."""
    words = _words(claim)
    numbers = set(NUMBER.findall(claim))
    text = "\n".join(item["chunk_text"] for item in passages)
    if not numbers.issubset(set(NUMBER.findall(text))):
        return False
    if words and len(words & _words(text)) / len(words) < 0.5:
        return False
    for quantifier in ("all", "every", "always", "only", "unlimited"):
        if re.search(rf"\b{quantifier}\b", claim, re.I) and not re.search(rf"\b{quantifier}\b", text, re.I):
            return False
    candidates = _statements(text)
    if candidates:
        best = max(candidates, key=lambda sentence: len(words & _words(sentence)))
        if words & _words(best) and bool(NEGATIVE.search(claim)) != bool(NEGATIVE.search(best)):
            return False
    return bool(words or numbers)


def validate_citations(answer: str, evidence: list[dict]) -> tuple[str, list[str]]:
    """Accept only answers whose every statement is supported by, and cites, the top passage [S1]."""
    if not isinstance(answer, str) or not answer.strip():
        return NO_EVIDENCE, []
    if answer.strip() == NO_EVIDENCE:
        return NO_EVIDENCE, []
    by_label = {item["source_label"]: item for item in evidence}
    if CITABLE_LABEL not in by_label:
        return NO_EVIDENCE, []
    # Do not allow generated links, HTML, or code to masquerade as local source markers.
    if re.search(r"https?://|\]\s*\(|<[^>]+>|```", answer, re.I):
        return NO_EVIDENCE, []
    def normalize(match):
        label = "S" + match.group(1)
        return f"[{label}]" if label in by_label else ""
    cleaned = MARKER.sub(normalize, answer).strip()
    for sentence in _statements(cleaned):
        sentence = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", sentence).strip()
        tail = re.search(r"((?:\[S[1-4]\]\s*)+)[.!?]?\s*$", sentence)
        if tail is None:
            return NO_EVIDENCE, []
        labels = list(dict.fromkeys(VALID_MARKER.findall(sentence)))
        claim = VALID_MARKER.sub("", sentence).strip(" .!?*")
        # Lower-ranked passages are context only; citing them means [S1] alone was not enough.
        if labels != [CITABLE_LABEL] or re.search(r"\[[^]]*\]", claim):
            return NO_EVIDENCE, []
        if not _supported(claim, [by_label[CITABLE_LABEL]]):
            return NO_EVIDENCE, []
    return (cleaned, [CITABLE_LABEL]) if VALID_MARKER.search(cleaned) else (NO_EVIDENCE, [])


def _excerpt(text: str, answer: str) -> str:
    words = _words(answer)
    statements = _statements(text)
    sentence = max(statements, key=lambda value: len(words & _words(value))) if statements else text
    if len(sentence) <= 300:
        return sentence
    hits = [match.start() for number in NUMBER.findall(answer)
            if (match := re.search(r"(?<!\d)" + re.escape(number) + r"(?!\d)", sentence))]
    if not hits:
        hits = [match.start() for word in words
                if (match := re.search(r"\b" + re.escape(word) + r"\w*", sentence, re.I))]
    start = min(max(0, min(hits) - 100), len(sentence) - 300) if hits else 0
    end = start + 300
    return ("…" if start else "") + sentence[start:end] + ("…" if end < len(sentence) else "")


def review_answer(answer: str, evidence: list[dict], minimum=DEFAULT_MIN_RELEVANCE, *, secret="") -> dict:
    # Labels follow retrieval order, so [S1] is always the rank-1 passage.
    evidence = [dict(item, source_label=f"S{position + 1}", rank=position + 1) for position, item in enumerate(evidence[:4])]
    if secret:
        if isinstance(answer, str):
            answer = answer.replace(secret, "[redacted]")
        for item in evidence:
            for field in ("filename", "section_title", "chunk_text"):
                if isinstance(item.get(field), str):
                    item[field] = item[field].replace(secret, "[redacted]")
    if not has_relevant_evidence(evidence, minimum):
        cleaned, used = NO_EVIDENCE, []
    else:
        cleaned, used = validate_citations(answer, evidence)
    sources = [{key: item.get(key) for key in (
        "source_label", "filename", "section_title", "page_number", "chunk_id", "document_id", "rank",
    )} | {"excerpt": _excerpt(item["chunk_text"], cleaned)} for item in evidence[:1] if used]
    # Informational only: other documents may disagree with the cited primary source.
    return {"answer": cleaned, "evidence": evidence, "sources": sources,
            "conflict_warning": bool(used) and bool(conflict_pairs(evidence, minimum))}
