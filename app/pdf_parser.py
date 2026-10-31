"""Turns a policy PDF into a document code, a title and numbered sections."""
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

POLICY_ID = re.compile(r"Policy ID:\s*([A-Z]{2,}-[A-Z]{2,}-\d{3})")
HEADING = re.compile(r"^(\d{1,2})\.\s+([A-Z][^.]{2,80})$")
PAGE_NUMBER = re.compile(r"\bPage \d+( of \d+)?\s*$")
# Bullet glyphs often come out of PDFs as odd characters: \x7f,  (Symbol font), and so on.
BULLET = re.compile(r"^(?:[•▪●◦‣\x7f*]|- )\s*")


@dataclass
class Section:
    number: int
    title: str
    page: int
    text: str


@dataclass
class ParsedDocument:
    code: str
    title: str
    sections: list[Section]


def _shape(line: str) -> str:
    return re.sub(r"\d+", "#", line)


def clean_lines(pages: list[str]) -> list[tuple[int, str]]:
    """Returns (page number, line) pairs without headers, footers and page numbers.

    A line counts as page furniture if it ends in a page number, or if the same line
    (ignoring digits) appears on every page of a multi-page document."""
    page_lines = [[line.strip() for line in text.splitlines() if line.strip()] for text in pages]
    repeated: set[str] = set()
    if len(page_lines) > 1:
        counts = Counter(shape for lines in page_lines for shape in {_shape(line) for line in lines})
        repeated = {shape for shape, n in counts.items() if n == len(page_lines)}
    return [
        (page_no, line)
        for page_no, lines in enumerate(page_lines, start=1)
        for line in lines
        if not PAGE_NUMBER.search(line) and _shape(line) not in repeated
    ]


def _join(lines: list[str]) -> str:
    """Re-joins wrapped lines into paragraphs, keeping each bullet on its own line."""
    out: list[str] = []
    for line in lines:
        if BULLET.match(line):
            out.append("- " + BULLET.sub("", line, count=1))
        elif out:
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


def parse_lines(lines: list[tuple[int, str]]) -> ParsedDocument:
    """Splits cleaned lines into numbered sections.

    Headings must count up 1, 2, 3..., so a numbered list inside a section
    is not mistaken for a new heading."""
    if not lines:
        raise ValueError("no text found (scanned PDFs need OCR, which comes in phase 3)")
    title = lines[0][1]
    code = next((m.group(1) for _, line in lines if (m := POLICY_ID.search(line))), None)
    if code is None:
        raise ValueError("no 'Policy ID:' line found")

    sections: list[tuple[int, str, int, list[str]]] = []
    for page, line in lines[1:]:
        match = HEADING.match(line)
        if match and int(match.group(1)) == len(sections) + 1:
            sections.append((int(match.group(1)), match.group(2).strip(), page, []))
        elif sections:
            sections[-1][3].append(line)
    if not sections:
        raise ValueError("no numbered sections found")
    return ParsedDocument(
        code=code,
        title=title,
        sections=[Section(number, title, page, _join(body)) for number, title, page, body in sections],
    )


def parse_pdf(path: Path) -> ParsedDocument:
    pages = [page.extract_text() or "" for page in PdfReader(path).pages]
    return parse_lines(clean_lines(pages))
