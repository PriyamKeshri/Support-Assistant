from pathlib import Path

import pytest

from app.pdf_parser import clean_lines, parse_lines, parse_pdf

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"


def test_footer_and_repeated_header_are_removed():
    pages = [
        "Kestrel Analytics | Internal | Page 1\nTitle\nbody one",
        "Kestrel Analytics | Internal | Page 2\nConfidential draft\nbody two",
        "Kestrel Analytics | Internal | Page 3\nConfidential draft\nbody three",
    ]
    lines = clean_lines(pages)
    # "Confidential draft" is on 2 of 3 pages only, so it is kept as content.
    assert [line for _, line in lines] == ["Title", "body one", "Confidential draft", "body two",
                                          "Confidential draft", "body three"]
    assert [page for page, _ in lines] == [1, 1, 2, 2, 3, 3]


def test_lines_on_every_page_are_dropped():
    pages = ["Draft v2\nalpha", "Draft v2\nbeta"]
    assert [line for _, line in clean_lines(pages)] == ["alpha", "beta"]


def test_sections_wrapped_lines_and_bullets():
    lines = [(1, "Remote Access Policy"), (1, "Policy ID: IT-OPS-003 | Version 4.0"),
             (1, "1. Scope"), (1, "Applies to every"), (1, "connection."),
             (1, "2. Errors"), (1, "\x7f VPN-809: blocked"), (2, "port."), (2, "• VPN-412: expired")]
    doc = parse_lines(lines)
    assert doc.code == "IT-OPS-003"
    assert doc.title == "Remote Access Policy"
    assert [(s.number, s.title, s.page) for s in doc.sections] == [(1, "Scope", 1), (2, "Errors", 1)]
    assert doc.sections[0].text == "Applies to every connection."
    assert doc.sections[1].text == "- VPN-809: blocked port.\n- VPN-412: expired"


def test_numbered_list_inside_a_section_is_not_a_heading():
    lines = [(1, "Policy"), (1, "Policy ID: IT-SEC-001"), (1, "1. Scope"), (1, "text"), (1, "2. Steps"),
             (1, "1. Disconnect from Wi-Fi"), (1, "2. Call the hotline"), (1, "3. Next Section"), (1, "text")]
    doc = parse_lines(lines)
    assert [s.title for s in doc.sections] == ["Scope", "Steps", "Next Section"]
    assert doc.sections[1].text == "1. Disconnect from Wi-Fi 2. Call the hotline"


def test_missing_policy_id_is_an_error():
    with pytest.raises(ValueError, match="Policy ID"):
        parse_lines([(1, "Title"), (1, "1. Scope"), (1, "text")])


@pytest.mark.skipif(not POLICIES.exists(), reason="sample PDFs not built")
def test_every_sample_pdf_parses_into_sections():
    for pdf in sorted(POLICIES.glob("*.pdf")):
        doc = parse_pdf(pdf)
        assert doc.code in pdf.name.upper()
        assert len(doc.sections) >= 6, pdf.name
        assert all(s.text for s in doc.sections), pdf.name
        assert not any("Page " in s.text for s in doc.sections), "footer leaked into " + pdf.name
