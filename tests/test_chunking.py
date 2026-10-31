from app.chunking import chunk_document, split_text
from app.pdf_parser import ParsedDocument, Section


def test_short_text_is_one_chunk():
    assert split_text("One sentence. Two sentences.", max_chars=100, overlap_chars=20) == [
        "One sentence. Two sentences."]


def test_long_text_splits_on_sentences_with_overlap():
    sentences = [f"Sentence number {i} is here." for i in range(10)]  # 27-28 chars each
    pieces = split_text(" ".join(sentences), max_chars=100, overlap_chars=30)
    assert len(pieces) > 1
    assert all(len(p) <= 100 for p in pieces)
    for previous, current in zip(pieces, pieces[1:]):
        # each piece starts with the last sentence of the one before it
        assert current.startswith(previous.split(". ")[-1].rstrip("."))
    assert "Sentence number 9 is here." in pieces[-1]


def test_bullets_stay_on_their_own_lines():
    text = "Intro sentence here.\n- First bullet item text.\n- Second bullet item text.\n- Third bullet item."
    pieces = split_text(text, max_chars=60, overlap_chars=0)
    assert all(line.startswith(("- ", "Intro")) for p in pieces for line in p.split("\n"))


def test_chunks_carry_document_and_section_context():
    doc = ParsedDocument("IT-SEC-001", "Password Policy", [Section(4, "Account Lockout", 2, "Locked for 30 minutes.")])
    [chunk] = chunk_document(doc, max_chars=1200, overlap_chars=200)
    assert chunk.heading == "Password Policy > 4. Account Lockout"
    assert chunk.page == 2
    assert chunk.embed_text == "Password Policy > 4. Account Lockout\nLocked for 30 minutes."
