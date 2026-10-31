"""Splits sections into chunks that fit the embedding model, without crossing section boundaries."""
import re
from dataclasses import dataclass

from app.pdf_parser import ParsedDocument

SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


@dataclass
class Chunk:
    section_no: int
    section_title: str
    chunk_index: int
    page: int
    heading: str
    content: str

    @property
    def embed_text(self) -> str:
        # The heading gives a chunk its context: "lockout" means little without "Password Policy".
        return f"{self.heading}\n{self.content}"


def _join(units: list[str]) -> str:
    return "".join(("\n" if u.startswith("- ") else " ") + u for u in units).strip()


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Packs sentences and bullets into pieces of at most max_chars (unless one sentence is
    longer). Each new piece starts with the last sentences of the previous one, up to
    overlap_chars, so a fact split across a boundary is still retrievable."""
    if len(text) <= max_chars:
        return [text]
    units = [u for line in text.split("\n") for u in SENTENCE_END.split(line) if u.strip()]
    pieces: list[list[str]] = [[]]
    for unit in units:
        current = pieces[-1]
        if current and len(_join(current + [unit])) > max_chars:
            carry: list[str] = []
            for previous in reversed(current):
                if len(_join([previous] + carry)) > overlap_chars:
                    break
                carry.insert(0, previous)
            pieces.append(carry + [unit])
        else:
            current.append(unit)
    return [_join(piece) for piece in pieces]


def chunk_document(doc: ParsedDocument, max_chars: int, overlap_chars: int) -> list[Chunk]:
    chunks = []
    for section in doc.sections:
        heading = f"{doc.title} > {section.number}. {section.title}"
        for index, piece in enumerate(split_text(section.text, max_chars, overlap_chars)):
            chunks.append(Chunk(section.number, section.title, index, section.page, heading, piece))
    return chunks
