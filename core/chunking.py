"""Heading-aware character windows; no model or vector-store dependencies."""

import re


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
PAGE_MARKER = re.compile(r"^\s*(?:<!--\s*|\[|---\s*)?Page\s+(\d+)(?:\s*of\s+\d+)?\s*(?:-->|\]|---)?\s*$", re.IGNORECASE)


def _sections(text: str):
    lines = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\x00PAGE_BREAK\n").splitlines(keepends=True)
    page = None
    title = None
    buffer = []
    fenced = False
    position = 0
    while position < len(lines):
        line = lines[position]
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
        marker = PAGE_MARKER.fullmatch(stripped) if not fenced else None
        if not fenced and (marker or stripped == "\x00PAGE_BREAK"):
            if buffer:
                yield "".join(buffer), page or (1 if stripped == "\x00PAGE_BREAK" else None), title
            buffer = []
            page = int(marker.group(1)) if marker else (page or 1) + 1
            position += 1
            continue
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line) if not fenced else None
        setext = (not fenced and bool(stripped) and position + 1 < len(lines)
                  and re.fullmatch(r"\s*(?:={3,}|-{3,})\s*", lines[position + 1]) is not None)
        if heading or setext:
            if buffer:
                yield "".join(buffer), page, title
            buffer = []
            title = heading.group(1).strip() if heading else stripped
            buffer.append(line)
            if setext:
                position += 1
                buffer.append(lines[position])
        else:
            buffer.append(line)
        position += 1
    if buffer:
        yield "".join(buffer), page, title


def split_document(text: str, document_name: str = "", *, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[dict]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than chunk size.")
    chunks = []
    for section, page, title in _sections(text):
        section = section.strip()
        start = 0
        while start < len(section):
            end = min(start + chunk_size, len(section))
            if end < len(section):
                minimum = start + max(overlap + 1, chunk_size // 2)
                for separator in ("\n\n", "\n", " "):
                    boundary = section.rfind(separator, minimum, end)
                    if boundary >= minimum:
                        end = boundary + len(separator)
                        break
            body = section[start:end]
            if body.strip():
                chunks.append({
                    "chunk_index": len(chunks), "chunk_text": body, "chunk_length": len(body),
                    "page_number": page, "section_title": title, "document_name": document_name,
                })
            if end >= len(section):
                break
            start = max(start + 1, end - overlap)
    return chunks
